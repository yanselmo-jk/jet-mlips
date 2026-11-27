/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Gijin Kim, Hyungmin An (SNU)

   GPU layout: the pair sums run over 32 x 32 tiles of atom pairs, one warp
   per tile (I, J) with J >= I, so every unordered pair is visited once
   (Newton's third law). Lane l owns atom j = 32 J + l (its data and partial
   sums stay in registers) and at step s the whole warp works on atom
   i = 32 I + s, so the 32 pair vectors of a step are as coherent as in a
   plain warp-per-atom loop (consecutive atoms are close in space, which
   keeps the lattice-image loops converged); the i-side terms of a step are
   warp-reduced and kept by lane s. Both sides are flushed with one fp64
   atomic per atom and tile. Energy and virial are warp-reduced and added
   into 32 accumulator slots. C6(CN_i, CN_j) is evaluated inside the force
   kernel from per-atom reference vectors.

   The atoms are packed in cell order (then by type), so the 32 atoms of a
   block are neighbours in space and a tile is skipped as a whole when the
   two blocks' fractional-coordinate bounds are farther apart than the
   cut-off (blocks_near): the tile scan then only touches the blocks inside
   the cut-off instead of all N^2/2 of them. A small box (fewer than 3 cells
   of half the cut-off per periodic direction) has no prefilter (every block
   pair has images inside the cut-off) but keeps the cell order, which keeps
   the image loops of a warp coherent.

   On more than one MPI rank (see launch_multi_async) every rank holds all N
   atoms: one MPI_Allgatherv of the local (tag, x, type) per step, packed
   exactly like the one-rank path (cell order, block prefilter, periodic
   images), and the same kernels run on that packing. Only the i side is
   restricted to the atoms this rank owns (a per-atom flag, and a compacted
   i-block list), so a pair is computed once in the whole job, with the full
   (not halved) energy, virial, forces and dE/dCN of both sides. The i-role
   block of an off-diagonal tile (I, J) is the lower one when I + J is even
   and the higher one when it is odd, so the cross-rank pairs split evenly
   over the ranks instead of loading the rank of the lower blocks. The CN is
   split the same way (each rank sums the pairs of its own atoms, both
   sides) and the partial CN arrays are summed over the ranks; dE/dCN is
   summed over the ranks before the dcn kernel, then the forces are summed
   over the ranks and each rank adds its own atoms' into f[]. No ghost atoms
   are involved, so newton pair may be on or off. Under pair_style
   hybrid/overlay this first half (gather, pack, CN, allreduce, weights and
   forces launch) runs from PairJetD3::prelaunch(), which pair_e3gnn calls after
   it has launched its GPU work, so it overlaps that work; compute() only
   collects (see finish_multi).
------------------------------------------------------------------------- */

#include "pair_jet_d3.h"

#include <chrono>
#include <cstdlib>
#include <cstring>

#include "force.h"
#include "update.h"

using namespace LAMMPS_NS;

#define CHECK_CUDA(call) do {                                            \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      error->one(FLERR, "pair_style d3: CUDA error {}: {}",                \
                 cudaGetErrorName(status_), cudaGetErrorString(status_));  \
    }                                                                    \
} while (0)

static constexpr int D3_MAXC = 5;
static constexpr int D3_BLOCK = 128;   // 4 warps = 4 pair tiles per block
// Float partial sums are flushed into the double accumulators every D3_CHUNK
// atoms j per lane (about 60 image terms); fp64 adds run at 1/64 rate on
// consumer GPUs and the flushes were a bottleneck at 8.
static constexpr int D3_CHUNK = 16;
static constexpr unsigned D3_FULL = 0xffffffffu;
static constexpr int D3_SLOTS = 32;    // energy / virial accumulator slots of 8 doubles (atomic contention)

/* ----------------------------------------------------------------------
   Constructor (Required)
------------------------------------------------------------------------- */

PairJetD3::PairJetD3(LAMMPS* lmp) : Pair(lmp) {
    single_enable = 0;      // potential is not pair-wise additive.
    restartinfo = 0;        // Many-body potentials are usually not
                            // written to binary restart files.
    one_coeff = 1;          // Many-body potnetials typically read all
                            // parameters from a file, so only one
                            // pair_coeff statement is needed.
    manybody_flag = 1;
    no_virial_fdotr_compute = 1;
    tab = D3Tables{};
    // SEVENN_TIMING=1: per-step host / kernel breakdown of compute(), printed at the end of each run
    const char *timing_flag = std::getenv("SEVENN_TIMING");
    timing = timing_flag && std::strcmp(timing_flag, "0") != 0;
}

/* ----------------------------------------------------------------------
   Destructor (Required)
------------------------------------------------------------------------- */

PairJetD3::~PairJetD3() {
    if (allocated) {
        memory->destroy(setflag);
        memory->destroy(cutsq);
        cudaFree((void*)tab.r2r4);
        cudaFree((void*)tab.rcov);
        cudaFree((void*)tab.r0ab);
        cudaFree((void*)tab.mxc);
        cudaFree((void*)tab.c6ref);
        cudaFree((void*)tab.cnref);
    }
    if (tev[0]) { for (cudaEvent_t e : tev) cudaEventDestroy(e); }
    if (stream) { cudaStreamDestroy(stream); }
    if (n_save >= 0) {
        cudaFree(d_in); cudaFreeHost(h_in);
        cudaFree(d_cn); cudaFree(d_w); cudaFree(d_vu);
        cudaFree(d_out); cudaFreeHost(h_out);
        if (multi) { cudaFree(d_iloc); cudaFree(d_iblocks); cudaFreeHost(h_cn); }
    }
}

/* ----------------------------------------------------------------------
   Allocate the per-type arrays (Required); per-atom arrays are sized in compute()
------------------------------------------------------------------------- */

void PairJetD3::allocate() {
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        error->all(FLERR, "pair_style d3: no CUDA device found");
    }
    allocated = 1;
    // own non-blocking stream: no implicit sync with the legacy stream 0 (torch streams of pair_e3gnn)
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // Pair sums in plain fp64 (default); JET_D3_FP64=0: float-float sums, much faster where fp64 adds run at 1/32 of
    // the fp32 rate or less (consumer GPUs). The user chooses: no automatic choice by architecture.
    const char *e = std::getenv("JET_D3_FP64");
    fp64_sums = !(e && std::strcmp(e, "0") == 0);

    int np1 = atom->ntypes + 1;
    memory->create(setflag, np1, np1, "pair:setflag");
    memory->create(cutsq, np1, np1, "pair:cutsq");
    for (int i = 1; i < np1; i++) {
        for (int j = 1; j < np1; j++) {
            setflag[i][j] = 0;
        }
    }

    tab.np1 = np1;
    CHECK_CUDA(cudaMalloc((void**)&tab.r2r4, np1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&tab.rcov, np1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&tab.r0ab, np1 * np1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&tab.mxc, np1 * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&tab.c6ref, np1 * np1 * MAXC * MAXC * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&tab.cnref, np1 * MAXC * sizeof(float)));
}

/* ----------------------------------------------------------------------
   Settings : read from pair_style (Required) -> pair_style d3 vdw_sq cn_sq damp_name func_name
------------------------------------------------------------------------- */

void PairJetD3::settings(int narg, char **arg) {
    if (narg != 4 && narg != 8) {
        error->all(FLERR,
                "Pair_style d3 needs Four arguments, or 8 with functional `custom s6 s8 a1 a2`:\n"
                "\t rthr: cutoff radius for dispersion interaction (a.u.^2)\n"
                "\t cnthr: cutoff raius for coordination number (a.u.^2)\n"
                "\t damping: name of the damping function (e.g., damp_zero, damp_bj)\n"
                "\t functional: name of the functional (e.g., pbe, b3-lyp)\n"
                );
    }
    rthr   = utils::numeric(FLERR, arg[0], false, lmp);
    cnthr = utils::numeric(FLERR, arg[1], false, lmp);

    std::map<std::string, int> commandMap = {
        {"damp_zero", 0}, {"damp_bj", 1}, {"damp_zerom", 2}, {"damp_bjm", 3},
    };

    if (commandMap.find(arg[2]) == commandMap.end()) {
        error->all(FLERR, "Unknown damping function");
    }

    damping = commandMap[arg[2]];
    functional = arg[3];
    custom_params = (narg == 8);
    if (custom_params) {
        if (functional != "custom") error->all(FLERR, "pair_style d3: 8 arguments need functional `custom`");
        custom_s6 = utils::numeric(FLERR, arg[4], false, lmp);
        custom_s8 = utils::numeric(FLERR, arg[5], false, lmp);
        custom_a1 = utils::numeric(FLERR, arg[6], false, lmp);
        custom_a2 = utils::numeric(FLERR, arg[7], false, lmp);
    }
    if (damping >= 2) {
        error->all(FLERR, "pair_style d3: damp_zerom and damp_bjm are not implemented");
    }

    setfuncpar();
}


/* ----------------------------------------------------------------------
   finds atomic number (used in PairJetD3::coeff)
------------------------------------------------------------------------- */

int PairJetD3::find_atomic_number(std::string& key) {
    std::transform(key.begin(), key.end(), key.begin(), ::tolower);
    if (key.length() == 1) { key += " "; }
    key.resize(2);

    std::vector<std::string> element_table = {
        "h ","he",
        "li","be","b ","c ","n ","o ","f ","ne",
        "na","mg","al","si","p ","s ","cl","ar",
        "k ","ca","sc","ti","v ","cr","mn","fe","co","ni","cu",
        "zn","ga","ge","as","se","br","kr",
        "rb","sr","y ","zr","nb","mo","tc","ru","rh","pd","ag",
        "cd","in","sn","sb","te","i ","xe",
        "cs","ba","la","ce","pr","nd","pm","sm","eu","gd","tb","dy",
        "ho","er","tm","yb","lu","hf","ta","w ","re","os","ir","pt",
        "au","hg","tl","pb","bi","po","at","rn",
        "fr","ra","ac","th","pa","u ","np","pu"
    };

    for (size_t i = 0; i < element_table.size(); ++i) {
        if (element_table[i] == key) {
            int atomic_number = i + 1;
            return atomic_number;
        }
    }

    // if not the case
    return -1;
}

/* ----------------------------------------------------------------------
   Check whether an integer value in an integer array (used in PairJetD3::coeff)
------------------------------------------------------------------------- */

int PairJetD3::is_int_in_array(int arr[], int size, int value) {
    for (int i = 0; i < size; i++) {
        if (arr[i] == value) { return i; } // returns the index
    }
    return -1;
}

/* ----------------------------------------------------------------------
   Read r0ab values from the table (used in PairJetD3::coeff)
------------------------------------------------------------------------- */

void PairJetD3::read_r0ab(int* atomic_numbers, int ntypes, std::vector<float>& r0ab) {
    const double r0ab_table[94][94] = R0AB_TABLE;
    const int np1 = ntypes + 1;

    for (int i = 1; i <= ntypes; i++) {
        for (int j = 1; j <= ntypes; j++) {
            r0ab[i * np1 + j] = r0ab_table[atomic_numbers[i-1]-1][atomic_numbers[j-1]-1] / AU_TO_ANG;
        }
    }
}


/* ----------------------------------------------------------------------
   Get atom pair indices and grid indices (used in PairJetD3::read_c6ab)
------------------------------------------------------------------------- */

void PairJetD3::get_limit_in_pars_array(int& idx_atom_1, int& idx_atom_2, int& idx_i, int& idx_j) {
    const int shift = 100;

    idx_i = (idx_atom_1 - 1) / shift + 1;
    idx_j = (idx_atom_2 - 1) / shift + 1;

    idx_atom_1 = (idx_atom_1 - 1) % shift + 1;
    idx_atom_2 = (idx_atom_2 - 1) % shift + 1;

    // the code above replaces the code below
    //idx_i = 1;
    //idx_j = 1;
    //int shift = 100;
    //while (idx_atom_1 > shift) { idx_atom_1 -= shift; idx_i++; }
    //while (idx_atom_2 > shift) { idx_atom_2 -= shift; idx_j++; }
}

/* ----------------------------------------------------------------------
   Read c6ab values from the table (used in PairJetD3::coeff)

   c6ref[ti][tj][a][b] is the reference C6 between reference geometry a of
   type ti and geometry b of type tj; cnref[ti][a] is the coordination
   number of that geometry (it depends only on the element, not on the
   partner). The table has every (a, b) combination for a < mxc[ti],
   b < mxc[tj].
------------------------------------------------------------------------- */

void PairJetD3::read_c6ab(int* atomic_numbers, int ntypes,
                       std::vector<float>& c6ref, std::vector<float>& cnref, std::vector<int>& mxc) {
    const int np1 = ntypes + 1;
    for (int i = 1; i <= ntypes; i++) { mxc[i] = 0; }
    int grid_i = 0, grid_j = 0;

    const double c6ab_table[32385][5] = C6AB_TABLE;

    for (int i = 0; i < 32385; i++) {
        const double ref_c6 = c6ab_table[i][0];
        int atom_number_1 = static_cast<int>(c6ab_table[i][1]);
        int atom_number_2 = static_cast<int>(c6ab_table[i][2]);
        get_limit_in_pars_array(atom_number_1, atom_number_2, grid_i, grid_j);
        if (is_int_in_array(atomic_numbers, ntypes, atom_number_1) < 0) { continue; }
        if (is_int_in_array(atomic_numbers, ntypes, atom_number_2) < 0) { continue; }
        const double ref_cn1 = c6ab_table[i][3];
        const double ref_cn2 = c6ab_table[i][4];
        const int a = grid_i - 1, b = grid_j - 1;

        // several LAMMPS types may map to the same element: fill every (t1, t2) pair
        for (int t1 = 1; t1 <= ntypes; t1++) {
            if (atomic_numbers[t1 - 1] != atom_number_1) { continue; }
            mxc[t1] = std::max(mxc[t1], grid_i);
            cnref[t1 * MAXC + a] = ref_cn1;
            for (int t2 = 1; t2 <= ntypes; t2++) {
                if (atomic_numbers[t2 - 1] != atom_number_2) { continue; }
                mxc[t2] = std::max(mxc[t2], grid_j);
                cnref[t2 * MAXC + b] = ref_cn2;
                c6ref[((t1 * np1 + t2) * MAXC + a) * MAXC + b] = ref_c6;
                c6ref[((t2 * np1 + t1) * MAXC + b) * MAXC + a] = ref_c6;
            }
        }
    }
}


/* ----------------------------------------------------------------------
   Set functional parameters (used in PairJetD3::coeff)
------------------------------------------------------------------------- */

void PairJetD3::setfuncpar_zero() {
    if (custom_params) error->all(FLERR, "pair_style d3: custom parameters are for damp_bj only");
    s6 = 1.0;
    alp = 14.0;
    rs18 = 1.0;

    // default def2-QZVP (almost basis set limit)
    std::unordered_map<std::string, int> commandMap = {
    { "slater-dirac-exchange", 1}, { "b-lyp", 2 },    { "b-p", 3 },       { "b97-d", 4 },      { "revpbe", 5 },
    { "pbe", 6 },                  { "pbesol", 7 },   { "rpw86-pbe", 8 }, { "rpbe", 9 },       { "tpss", 10 },
    { "b3-lyp", 11 },              { "pbe0", 12 },    { "hse06", 13 },    { "revpbe38", 14 },  { "pw6b95", 15 },
    { "tpss0", 16 },               { "b2-plyp", 17 }, { "pwpb95", 18 },   { "b2gp-plyp", 19 }, { "ptpss", 20 },
    { "hf", 21 },                  { "mpwlyp", 22 },  { "bpbe", 23 },     { "bh-lyp", 24 },    { "tpssh", 25 },
    { "pwb6k", 26 },               { "b1b95", 27 },   { "bop", 28 },      { "o-lyp", 29 },     { "o-pbe", 30 },
    { "ssb", 31 },                 { "revssb", 32 },  { "otpss", 33 },    { "b3pw91", 34 },    { "revpbe0", 35 },
    { "pbe38", 36 },               { "mpw1b95", 37 }, { "mpwb1k", 38 },   { "bmk", 39 },       { "cam-b3lyp", 40 },
    { "lc-wpbe", 41 },             { "m05", 42 },     { "m052x", 43 },    { "m06l", 44 },      { "m06", 45 },
    { "m062x", 46 },               { "m06hf", 47 },   { "hcth120", 48 }
    };

    int commandCode = commandMap[functional];
    switch (commandCode) {
    case 1: rs6 = 0.999; s18 = -1.957; rs18 = 0.697; break;
    case 2: rs6 = 1.094; s18 = 1.682; break;
    case 3: rs6 = 1.139; s18 = 1.683; break;
    case 4: rs6 = 0.892; s18 = 0.909; break;
    case 5: rs6 = 0.923; s18 = 1.010; break;
    case 6: rs6 = 1.217; s18 = 0.722; break;
    case 7: rs6 = 1.345; s18 = 0.612; break;
    case 8: rs6 = 1.224; s18 = 0.901; break;
    case 9: rs6 = 0.872; s18 = 0.514; break;
    case 10: rs6 = 1.166; s18 = 1.105; break;
    case 11: rs6 = 1.261; s18 = 1.703; break;
    case 12: rs6 = 1.287; s18 = 0.928; break;
    case 13: rs6 = 1.129; s18 = 0.109; break;
    case 14: rs6 = 1.021; s18 = 0.862; break;
    case 15: rs6 = 1.532; s18 = 0.862; break;
    case 16: rs6 = 1.252; s18 = 1.242; break;
    case 17: rs6 = 1.427; s18 = 1.022; s6 = 0.64; break;
    case 18: rs6 = 1.557; s18 = 0.705; s6 = 0.82; break;
    case 19: rs6 = 1.586; s18 = 0.760; s6 = 0.56; break;
    case 20: rs6 = 1.541; s18 = 0.879; s6 = 0.75; break;
    case 21: rs6 = 1.158; s18 = 1.746; break;
    case 22: rs6 = 1.239; s18 = 1.098; break;
    case 23: rs6 = 1.087; s18 = 2.033; break;
    case 24: rs6 = 1.370; s18 = 1.442; break;
    case 25: rs6 = 1.223; s18 = 1.219; break;
    case 26: rs6 = 1.660; s18 = 0.550; break;
    case 27: rs6 = 1.613; s18 = 1.868; break;
    case 28: rs6 = 0.929; s18 = 1.975; break;
    case 29: rs6 = 0.806; s18 = 1.764; break;
    case 30: rs6 = 0.837; s18 = 2.055; break;
    case 31: rs6 = 1.215; s18 = 0.663; break;
    case 32: rs6 = 1.221; s18 = 0.560; break;
    case 33: rs6 = 1.128; s18 = 1.494; break;
    case 34: rs6 = 1.176; s18 = 1.775; break;
    case 35: rs6 = 0.949; s18 = 0.792; break;
    case 36: rs6 = 1.333; s18 = 0.998; break;
    case 37: rs6 = 1.605; s18 = 1.118; break;
    case 38: rs6 = 1.671; s18 = 1.061; break;
    case 39: rs6 = 1.931; s18 = 2.168; break;
    case 40: rs6 = 1.378; s18 = 1.217; break;
    case 41: rs6 = 1.355; s18 = 1.279; break;
    case 42: rs6 = 1.373; s18 = 0.595; break;
    case 43: rs6 = 1.417; s18 = 0.000; break;
    case 44: rs6 = 1.581; s18 = 0.000; break;
    case 45: rs6 = 1.325; s18 = 0.000; break;
    case 46: rs6 = 1.619; s18 = 0.000; break;
    case 47: rs6 = 1.446; s18 = 0.000; break;
    /* DFTB3(zeta = 4.0), old deprecated parameters; case ("dftb3"); rs6 = 1.235; s18 = 0.673; */
    case 48: rs6 = 1.221; s18 = 1.206; break;
    default:
        error->all(FLERR, "Functional name unknown");
        break;
    }
}

void PairJetD3::setfuncpar_bj() {
    s6 = 1.0;
    alp = 14.0;
    if (custom_params) { s6 = custom_s6; s18 = custom_s8; rs6 = custom_a1; rs18 = custom_a2; return; }

    std::unordered_map<std::string, int> commandMap = {
        {"b-p", 1}, {"b-lyp", 2}, {"revpbe", 3}, {"rpbe", 4}, {"b97-d", 5}, {"pbe", 6},
        {"rpw86-pbe", 7}, {"b3-lyp", 8}, {"tpss", 9}, {"hf", 10}, {"tpss0", 11}, {"pbe0", 12},
        {"hse06", 13}, {"revpbe38", 14}, {"pw6b95", 15}, {"b2-plyp", 16}, {"dsd-blyp", 17},
        {"dsd-blyp-fc", 18}, {"bop", 19}, {"mpwlyp", 20}, {"o-lyp", 21}, {"pbesol", 22}, {"bpbe", 23},
        {"opbe", 24}, {"ssb", 25}, {"revssb", 26}, {"otpss", 27}, {"b3pw91", 28}, {"bh-lyp", 29},
        {"revpbe0", 30}, {"tpssh", 31}, {"mpw1b95", 32}, {"pwb6k", 33}, {"b1b95", 34}, {"bmk", 35},
        {"cam-b3lyp", 36}, {"lc-wpbe", 37}, {"b2gp-plyp", 38}, {"ptpss", 39}, {"pwpb95", 40},
        {"hf/mixed", 41}, {"hf/sv", 42}, {"hf/minis", 43}, {"b3-lyp/6-31gd", 44}, {"hcth120", 45},
        {"pw1pw", 46}, {"pwgga", 47}, {"hsesol", 48}, {"hf3c", 49}, {"hf3cv", 50}, {"pbeh3c", 51},
        {"pbeh-3c", 52}, {"wb97m", 53}, {"r2scan", 54}
    };

    int commandCode = commandMap[functional];
    switch (commandCode) {
        case 1: rs6 = 0.3946; s18 = 3.2822; rs18 = 4.8516; break;
        case 2: rs6 = 0.4298; s18 = 2.6996; rs18 = 4.2359; break;
        case 3: rs6 = 0.5238; s18 = 2.3550; rs18 = 3.5016; break;
        case 4: rs6 = 0.1820; s18 = 0.8318; rs18 = 4.0094; break;
        case 5: rs6 = 0.5545; s18 = 2.2609; rs18 = 3.2297; break;
        case 6: rs6 = 0.4289; s18 = 0.7875; rs18 = 4.4407; break;
        case 7: rs6 = 0.4613; s18 = 1.3845; rs18 = 4.5062; break;
        case 8: rs6 = 0.3981; s18 = 1.9889; rs18 = 4.4211; break;
        case 9: rs6 = 0.4535; s18 = 1.9435; rs18 = 4.4752; break;
        case 10: rs6 = 0.3385; s18 = 0.9171; rs18 = 2.8830; break;
        case 11: rs6 = 0.3768; s18 = 1.2576; rs18 = 4.5865; break;
        case 12: rs6 = 0.4145; s18 = 1.2177; rs18 = 4.8593; break;
        case 13: rs6 = 0.383; s18 = 2.310; rs18 = 5.685; break;
        case 14: rs6 = 0.4309; s18 = 1.4760; rs18 = 3.9446; break;
        case 15: rs6 = 0.2076; s18 = 0.7257; rs18 = 6.3750; break;
        case 16: rs6 = 0.3065; s18 = 0.9147; rs18 = 5.0570; break; s6 = 0.64;
        case 17: rs6 = 0.0000; s18 = 0.2130; rs18 = 6.0519; s6 = 0.50; break;
        case 18: rs6 = 0.0009; s18 = 0.2112; rs18 = 5.9807; s6 = 0.50; break;
        case 19: rs6 = 0.4870; s18 = 3.2950; rs18 = 3.5043; break;
        case 20: rs6 = 0.4831; s18 = 2.0077; rs18 = 4.5323; break;
        case 21: rs6 = 0.5299; s18 = 2.6205; rs18 = 2.8065; break;
        case 22: rs6 = 0.4466; s18 = 2.9491; rs18 = 6.1742; break;
        case 23: rs6 = 0.4567; s18 = 4.0728; rs18 = 4.3908; break;
        case 24: rs6 = 0.5512; s18 = 3.3816; rs18 = 2.9444; break;
        case 25: rs6 = -0.0952; s18 = -0.1744; rs18 = 5.2170; break;
        case 26: rs6 = 0.4720; s18 = 0.4389; rs18 = 4.0986; break;
        case 27: rs6 = 0.4634; s18 = 2.7495; rs18 = 4.3153; break;
        case 28: rs6 = 0.4312; s18 = 2.8524; rs18 = 4.4693; break;
        case 29: rs6 = 0.2793; s18 = 1.0354; rs18 = 4.9615; break;
        case 30: rs6 = 0.4679; s18 = 1.7588; rs18 = 3.7619; break;
        case 31: rs6 = 0.4529; s18 = 2.2382; rs18 = 4.6550; break;
        case 32: rs6 = 0.1955; s18 = 1.0508; rs18 = 6.4177; break;
        case 33: rs6 = 0.1805; s18 = 0.9383; rs18 = 7.7627; break;
        case 34: rs6 = 0.2092; s18 = 1.4507; rs18 = 5.5545; break;
        case 35: rs6 = 0.1940; s18 = 2.0860; rs18 = 5.9197; break;
        case 36: rs6 = 0.3708; s18 = 2.0674; rs18 = 5.4743; break;
        case 37: rs6 = 0.3919; s18 = 1.8541; rs18 = 5.0897; break;
        case 38: rs6 = 0.0000; s18 = 0.2597; rs18 = 6.3332; s6 = 0.560; break;
        case 39: rs6 = 0.0000; s18 = 0.2804; rs18 = 6.5745; s6 = 0.750; break;
        case 40: rs6 = 0.0000; s18 = 0.2904; rs18 = 7.3141; s6 = 0.820; break;
        // special HF / DFT with eBSSE correction;
        case 41: rs6 = 0.5607; s18 = 3.9027; rs18 = 4.5622; break;
        case 42: rs6 = 0.4249; s18 = 2.1849; rs18 = 4.2783; break;
        case 43: rs6 = 0.1702; s18 = 0.9841; rs18 = 3.8506; break;
        case 44: rs6 = 0.5014; s18 = 4.0672; rs18 = 4.8409; break;
        case 45: rs6 = 0.3563; s18 = 1.0821; rs18 = 4.3359; break;
        /*     DFTB3 old, deprecated parameters : ;
            *     case ("dftb3"); rs6 = 0.7461; s18 = 3.209; rs18 = 4.1906;
            *     special SCC - DFTB parametrization;
            *     full third order DFTB, self consistent charges, hydrogen pair damping with; exponent 4.2;
        */
        case 46: rs6 = 0.3807; s18 = 2.3363; rs18 = 5.8844; break;
        case 47: rs6 = 0.2211; s18 = 2.6910; rs18 = 6.7278; break;
        case 48: rs6 = 0.4650; s18 = 2.9215; rs18 = 6.2003; break;
        // special HF - D3 - gCP - SRB / MINIX parametrization;
        case 49: rs6 = 0.4171; s18 = 0.8777; rs18 = 2.9149; break;
        // special HF - D3 - gCP - SRB2 / ECP - 2G parametrization;
        case 50: rs6 = 0.3063; s18 = 0.5022; rs18 = 3.9856; break;
        // special PBEh - D3 - gCP / def2 - mSVP parametrization;
        case 51: rs6 = 0.4860; s18 = 0.0000; rs18 = 4.5000; break;
        case 52: rs6 = 0.4860; s18 = 0.0000; rs18 = 4.5000; break;
        case 53: rs6 = 0.5660; s18 = 0.3908; rs18 = 3.1280; break;  // J. Chem. Theory Comput. 14, 5725 (2018)
        case 54: rs6 = 0.4948; s18 = 0.7898; rs18 = 5.7308; break;  // S. Ehlert et al.,  J. Chem. Phys. 154, 061101 (2021)
        default:
            error->all(FLERR, "Functional name unknown");
            break;
    }
}

void PairJetD3::setfuncpar_zerom() {
    s6 = 1.0;
    alp = 14.0;

    std::unordered_map<std::string, int> commandMap = {
        {"b2-plyp", 1}, {"b3-lyp", 2}, {"b97-d", 3}, {"b-lyp", 4},
        {"b-p", 5}, {"pbe", 6}, {"pbe0", 7}, {"lc-wpbe", 8}
    };

    int commandCode = commandMap[functional];
    switch (commandCode) {
        case 1: rs6 = 1.313134; s18 = 0.717543; rs18 = 0.016035; s6 = 0.640000; break;
        case 2: rs6 = 1.338153; s18 = 1.532981; rs18 = 0.013988; break;
        case 3: rs6 = 1.151808; s18 = 1.020078; rs18 = 0.035964; break;
        case 4: rs6 = 1.279637; s18 = 1.841686; rs18 = 0.014370; break;
        case 5: rs6 = 1.233460; s18 = 1.945174; rs18 = 0.000000; break;
        case 6: rs6 = 2.340218; s18 = 0.000000; rs18 = 0.129434; break;
        case 7: rs6 = 2.077949; s18 = 0.000081; rs18 = 0.116755; break;
        case 8: rs6 = 1.366361; s18 = 1.280619; rs18 = 0.003160; break;
        default:
            error->all(FLERR, "Functional name unknown");
            break;
    }
}

void PairJetD3::setfuncpar_bjm() {
    s6 = 1.0;
    alp = 14.0;

    std::unordered_map<std::string, int> commandMap = {
        {"b2-plyp", 1}, {"b3-lyp", 2}, {"b97-d", 3}, {"b-lyp", 4},
        {"b-p", 5}, {"pbe", 6}, {"pbe0", 7}, {"lc-wpbe", 8}
    };

    int commandCode = commandMap[functional];
    switch (commandCode) {
        case 1: rs6 = 0.486434; s18 = 0.672820; rs18 = 3.656466; s6 = 0.640000; break;
        case 2: rs6 = 0.278672; s18 = 1.466677; rs18 = 4.606311; break;
        case 3: rs6 = 0.240184; s18 = 1.206988; rs18 = 3.864426; break;
        case 4: rs6 = 0.448486; s18 = 1.875007; rs18 = 3.610679; break;
        case 5: rs6 = 0.821850; s18 = 3.140281; rs18 = 2.728151; break;
        case 6: rs6 = 0.012092; s18 = 0.358940; rs18 = 5.938951; break;
        case 7: rs6 = 0.007912; s18 = 0.528823; rs18 = 6.162326; break;
        case 8: rs6 = 0.563761; s18 = 0.906564; rs18 = 3.593680; break;
        default:
            error->all(FLERR, "Functional name unknown");
            break;
    }
}

void PairJetD3::setfuncpar() {
    void (PairJetD3::*setfuncpar_damp[4])() = {
        &PairJetD3::setfuncpar_zero,
        &PairJetD3::setfuncpar_bj,
        &PairJetD3::setfuncpar_zerom,
        &PairJetD3::setfuncpar_bjm
    };
    (this->*setfuncpar_damp[damping])();

    rs8 = rs18;
    alp6 = alp;
    alp8 = alp + 2.0;
    // rs10 = rs18
    // alp10 = alp + 4.0;

    a1 = rs6;
    a2 = rs8;
    s8 = s18;
    // s6 is already defined
}

/* ----------------------------------------------------------------------
   Coeff : read from pair_coeff (Required) -> pair_coeff * * element1 element2 ...
------------------------------------------------------------------------- */

void PairJetD3::coeff(int narg, char **arg) {
    if (!allocated) allocate();

    int ntypes = atom->ntypes;
    if (narg != ntypes + 2) { error->all(FLERR, "Pair_coeff needs: * * element1 element2 ..."); }

    std::string element;
    std::vector<int> atomic_numbers(ntypes);
    for (int i = 0; i < ntypes; i++) {
        element = arg[i+2];
        atomic_numbers[i] = find_atomic_number(element);
        if (atomic_numbers[i] < 0) { error->all(FLERR, "pair_style d3: unknown element {}", arg[i+2]); }
    }

    for (int i = 1; i <= ntypes; i++) {
        for (int j = 1; j <= ntypes; j++) {
            setflag[i][j] = 1;
        }
    }

    /*
    scale r4/r2 values of the atoms by sqrt(Z)
    sqrt is also globally close to optimum
    together with the factor 1/2 this yield reasonable
    c8 for he, ne and ar. for larger Z, C8 becomes too large
    which effectively mimics higher R^n terms neglected due
    to stability reasons

    r2r4 =sqrt(0.5*r2r4(i)*dfloat(i)**0.5 ) with i=elementnumber
    the large number of digits is just to keep the results consistent
    with older versions. They should not imply any higher accuracy than
    the old values
    */
    double r2r4_ref[94] = {
         2.00734898,  1.56637132,  5.01986934,  3.85379032,  3.64446594,
         3.10492822,  2.71175247,  2.59361680,  2.38825250,  2.21522516,
         6.58585536,  5.46295967,  5.65216669,  4.88284902,  4.29727576,
         4.04108902,  3.72932356,  3.44677275,  7.97762753,  7.07623947,
         6.60844053,  6.28791364,  6.07728703,  5.54643096,  5.80491167,
         5.58415602,  5.41374528,  5.28497229,  5.22592821,  5.09817141,
         6.12149689,  5.54083734,  5.06696878,  4.87005108,  4.59089647,
         4.31176304,  9.55461698,  8.67396077,  7.97210197,  7.43439917,
         6.58711862,  6.19536215,  6.01517290,  5.81623410,  5.65710424,
         5.52640661,  5.44263305,  5.58285373,  7.02081898,  6.46815523,
         5.98089120,  5.81686657,  5.53321815,  5.25477007, 11.02204549,
        10.15679528,  9.35167836,  9.06926079,  8.97241155,  8.90092807,
         8.85984840,  8.81736827,  8.79317710,  7.89969626,  8.80588454,
         8.42439218,  8.54289262,  8.47583370,  8.45090888,  8.47339339,
         7.83525634,  8.20702843,  7.70559063,  7.32755997,  7.03887381,
         6.68978720,  6.05450052,  5.88752022,  5.70661499,  5.78450695,
         7.79780729,  7.26443867,  6.78151984,  6.67883169,  6.39024318,
         6.09527958, 11.79156076, 11.10997644,  9.51377795,  8.67197068,
         8.77140725,  8.65402716,  8.53923501,  8.85024712
    }; // atomic <r^2>/<r^4> values

    /*
    covalent radii (taken from Pyykko and Atsumi, Chem. Eur. J. 15, 2009, 188-197)
    values for metals decreased by 10 %
    !      data rcov/
    !     .  0.32, 0.46, 1.20, 0.94, 0.77, 0.75, 0.71, 0.63, 0.64, 0.67
    !     ., 1.40, 1.25, 1.13, 1.04, 1.10, 1.02, 0.99, 0.96, 1.76, 1.54
    !     ., 1.33, 1.22, 1.21, 1.10, 1.07, 1.04, 1.00, 0.99, 1.01, 1.09
    !     ., 1.12, 1.09, 1.15, 1.10, 1.14, 1.17, 1.89, 1.67, 1.47, 1.39
    !     ., 1.32, 1.24, 1.15, 1.13, 1.13, 1.08, 1.15, 1.23, 1.28, 1.26
    !     ., 1.26, 1.23, 1.32, 1.31, 2.09, 1.76, 1.62, 1.47, 1.58, 1.57
    !     ., 1.56, 1.55, 1.51, 1.52, 1.51, 1.50, 1.49, 1.49, 1.48, 1.53
    !     ., 1.46, 1.37, 1.31, 1.23, 1.18, 1.16, 1.11, 1.12, 1.13, 1.32
    !     ., 1.30, 1.30, 1.36, 1.31, 1.38, 1.42, 2.01, 1.81, 1.67, 1.58
    !     ., 1.52, 1.53, 1.54, 1.55 /

    these new data are scaled with k2=4./3.  and converted a_0 via
    autoang=0.52917726d0
    */

    double rcov_ref[94] = {
        0.80628308, 1.15903197, 3.02356173, 2.36845659, 1.94011865,
        1.88972601, 1.78894056, 1.58736983, 1.61256616, 1.68815527,
        3.52748848, 3.14954334, 2.84718717, 2.62041997, 2.77159820,
        2.57002732, 2.49443835, 2.41884923, 4.43455700, 3.88023730,
        3.35111422, 3.07395437, 3.04875805, 2.77159820, 2.69600923,
        2.62041997, 2.51963467, 2.49443835, 2.54483100, 2.74640188,
        2.82199085, 2.74640188, 2.89757982, 2.77159820, 2.87238349,
        2.94797246, 4.76210950, 4.20778980, 3.70386304, 3.50229216,
        3.32591790, 3.12434702, 2.89757982, 2.84718717, 2.84718717,
        2.72120556, 2.89757982, 3.09915070, 3.22513231, 3.17473967,
        3.17473967, 3.09915070, 3.32591790, 3.30072128, 5.26603625,
        4.43455700, 4.08180818, 3.70386304, 3.98102289, 3.95582657,
        3.93062995, 3.90543362, 3.80464833, 3.82984466, 3.80464833,
        3.77945201, 3.75425569, 3.75425569, 3.72905937, 3.85504098,
        3.67866672, 3.45189952, 3.30072128, 3.09915070, 2.97316878,
        2.92277614, 2.79679452, 2.82199085, 2.84718717, 3.32591790,
        3.27552496, 3.27552496, 3.42670319, 3.30072128, 3.47709584,
        3.57788113, 5.06446567, 4.56053862, 4.20778980, 3.98102289,
        3.82984466, 3.85504098, 3.88023730, 3.90543362
    }; // covalent radii

    const int np1 = ntypes + 1;
    std::vector<float> r2r4(np1, 0.0f), rcov(np1, 0.0f), r0ab(np1 * np1, 0.0f);
    std::vector<float> c6ref(np1 * np1 * MAXC * MAXC, -1.0f), cnref(np1 * MAXC, 0.0f);
    std::vector<int> mxc(np1, 0);
    for (int i = 0; i < ntypes; i++) {
        r2r4[i+1] = r2r4_ref[atomic_numbers[i]-1];
        rcov[i+1] = rcov_ref[atomic_numbers[i]-1];
    }
    read_r0ab(atomic_numbers.data(), ntypes, r0ab);
    read_c6ab(atomic_numbers.data(), ntypes, c6ref, cnref, mxc);
    for (int i = 1; i <= ntypes; i++) {
        if (mxc[i] == 0) { error->all(FLERR, "pair_style d3: no C6 reference data for element {}", arg[i+1]); }
        // the factorised C6 (k_d3_weights) assumes every (a, b) reference pair of (ti, tj) is tabulated
        for (int j = 1; j <= ntypes; j++) for (int a = 0; a < mxc[i]; a++) for (int b = 0; b < mxc[j]; b++) {
            if (c6ref[((i * np1 + j) * MAXC + a) * MAXC + b] <= 0.0f) {
                error->all(FLERR, "pair_style d3: incomplete C6 reference grid for {} {}", arg[i+1], arg[j+1]);
            }
        }
    }

    CHECK_CUDA(cudaMemcpy((void*)tab.r2r4, r2r4.data(), r2r4.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy((void*)tab.rcov, rcov.data(), rcov.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy((void*)tab.r0ab, r0ab.data(), r0ab.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy((void*)tab.mxc, mxc.data(), mxc.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy((void*)tab.c6ref, c6ref.data(), c6ref.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy((void*)tab.cnref, cnref.data(), cnref.size() * sizeof(float), cudaMemcpyHostToDevice));
}

/* ----------------------------------------------------------------------
   Device helpers
------------------------------------------------------------------------- */

// Integer-valued floats: rint / floor / ceil without the float <-> int
// conversion unit, which runs at a fraction of the FP32 rate and was the
// bottleneck of the pair loops (|v| < 2^22; __fadd_rn keeps the two adds).
__device__ __forceinline__ float rint_f(float v) { return __fadd_rn(__fadd_rn(v, 12582912.0f), -12582912.0f); }
__device__ __forceinline__ float floor_f(float v) { const float r = rint_f(v); return (r > v) ? r - 1.0f : r; }
__device__ __forceinline__ float ceil_f(float v) { const float r = rint_f(v); return (r < v) ? r + 1.0f : r; }

// Images tau_k of a pair with fractional separation s can only lie inside the
// cutoff when |s_k + tau_k| * h_k <= rc for every direction k (h_k is the
// spacing of the lattice planes normal to k, rch = rc / h_k). tau stays a
// float (exact small integers): no int conversions in the image loops.
__device__ __forceinline__ void image_bounds(float s, float rch, int per, float &lo, float &hi) {
    if (!per) { lo = 0.0f; hi = 0.0f; return; }
    lo = ceil_f(-s - rch);
    hi = floor_f(-s + rch);
}

// Calls body(rij, r2) for every candidate image of the pair separation d
// (cartesian) / s (fractional); the tau = 0 image is skipped for self pairs.
template <class F>
__device__ __forceinline__ void for_images(const D3Cell &c, const float *rch, const float d[3], const float s[3],
                                           bool self, F &&body) {
    float lo[3], hi[3];
    #pragma unroll
    for (int k = 0; k < 3; k++) { image_bounds(s[k], rch[k], c.per[k], lo[k], hi[k]); }
    for (float t0 = lo[0]; t0 <= hi[0]; t0 += 1.0f) {
        float b[3];
        #pragma unroll
        for (int k = 0; k < 3; k++) { b[k] = fmaf(t0, c.a[0][k], d[k]); }
        for (float t1 = lo[1]; t1 <= hi[1]; t1 += 1.0f) {
            float e[3];
            #pragma unroll
            for (int k = 0; k < 3; k++) { e[k] = fmaf(t1, c.a[1][k], b[k]); }
            for (float t2 = lo[2]; t2 <= hi[2]; t2 += 1.0f) {
                if (self && t0 == 0.0f && t1 == 0.0f && t2 == 0.0f) { continue; }
                float r[3];
                #pragma unroll
                for (int k = 0; k < 3; k++) { r[k] = fmaf(t2, c.a[2][k], e[k]); }
                body(r, r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
            }
        }
    }
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) { v += __shfl_xor_sync(D3_FULL, v, off); }
    return v;
}

// Float-float accumulation (hi + lo, Knuth TwoSum with rounded intrinsics so nothing is contracted
// or reassociated): the per-atom force sums add pair terms over ~6 orders of magnitude (near pairs
// ~1e-2 eV/A, pairs near a 50 A cut-off ~1e-9), and plain fp32 sums dropped the small terms
// differently on the i side (lane reduction) and the j side (running sum over steps), which left
// a systematic net force (whole-system drift, 2026-11-20).
__device__ __forceinline__ void two_sum(float &hi, float &lo, float b) {
    const float s = __fadd_rn(hi, b);
    const float bb = __fsub_rn(s, hi);
    lo = __fadd_rn(lo, __fadd_rn(__fsub_rn(hi, __fsub_rn(s, bb)), __fsub_rn(b, bb)));
    hi = s;
}

__device__ __forceinline__ void warp_sum2(float &hi, float &lo) {
    for (int off = 16; off > 0; off >>= 1) {
        const float oh = __shfl_xor_sync(D3_FULL, hi, off), ol = __shfl_xor_sync(D3_FULL, lo, off);
        two_sum(hi, lo, oh);
        lo = __fadd_rn(lo, ol);
    }
}

/* ----------------------------------------------------------------------
   Pair tiles (see the header comment)
------------------------------------------------------------------------- */

// Tile (I, J) of this warp; false when the warp has none (I > J or past the last tile).
// One rank (iblocks == nullptr): the tile is (I = ib, J = blockIdx.x) with J >= I, as before.
// Several ranks: iblocks is the compacted list of the i-role blocks (blocks that hold local
// atoms) and blockIdx.x is a partner column, so the grid is (partner, 4 i-blocks per block).
// Column p of i-role block K is the p-th partner block below K with K + J odd, then the p-th
// above K with K + J even, then the diagonal (K, K) last: an unordered pair of blocks is the
// i-role of exactly one of its two owners (the pair's lower block when the sum is even, its
// higher block when odd), so every pair is still computed exactly once in the whole job but
// the cross-rank pairs split evenly over the ranks.
__device__ __forceinline__ bool tile_of_warp(int ntiles, int niblocks, const int *iblocks, int &I, int &J) {
    const int ib = blockIdx.y * (D3_BLOCK / 32) + (threadIdx.x >> 5);
    if (ib >= niblocks) { return false; }
    if (!iblocks) {
        J = blockIdx.x;
        I = ib;
        return I <= J;
    }
    I = iblocks[ib];
    const int lo = (I + 1) / 2;                  // partner blocks J < I with I + J odd
    const int hi = (ntiles - 1 - I) / 2;         // partner blocks J > I with I + J even
    const int p = blockIdx.x;
    if (p > lo + hi) { return false; }
    if (p < lo) { J = 2 * p + ((I & 1) ? 0 : 1); }
    else if (p < lo + hi) { J = I + 2 + 2 * (p - lo); }
    else { J = I; }
    return true;
}

// Prefilter of tile (I, J), b[k] = the 8 floats of block k (see load_atom_info):
// fractional centre 0..2, cartesian radius 3, fractional half extents 4..6.
// False when no pair of atoms of the two 32-atom blocks can have an image
// inside the cut-off: the blocks are spheres (orthogonal cells) or boxes
// (triclinic) around their centres, so a centre distance beyond rc + both
// radii, or a per-direction distance beyond rch + both half extents (the same
// per-direction bound for_images uses), leaves every image outside. Both tests
// are conservative: a pair inside the cut-off always passes, so only tiles the
// pair loop would drop are skipped.
__device__ __forceinline__ bool blocks_near(const float *bi, const float *bj, const D3Cell &c,
                                            float rc, const float *rch) {
    float d[3];
    #pragma unroll
    for (int k = 0; k < 3; k++) {
        d[k] = bj[k] - bi[k];
        if (c.per[k]) { d[k] -= rint_f(d[k]); }   // nearest periodic image
    }
    if (c.ortho) {
        const float lim = rc + bi[3] + bj[3];
        const float d2 = fmaf(d[0] * c.h[0], d[0] * c.h[0],
                              fmaf(d[1] * c.h[1], d[1] * c.h[1], d[2] * c.h[2] * d[2] * c.h[2]));
        return d2 <= lim * lim;
    }
    #pragma unroll
    for (int k = 0; k < 3; k++) {
        if (fabsf(d[k]) - bi[4 + k] - bj[4 + k] > rch[k]) { return false; }
    }
    return true;
}

// Runs body(i, self, jside, ci, cj, ev) for the 32 steps of tile (I, J): at step s the
// warp works on atom i = 32 I + s and lane l on its own atom j = 32 J + l. body adds the pair's
// terms to the i side ci[NI] (warp-reduced afterwards), the lane's j accumulators cj[NI] (only
// when jside) and the symmetric terms ev[NEV]. self: j == i (skip tau = 0). On the diagonal
// tile too every unordered pair is visited once (j > i, both sides), so a pair force is added
// as +F / -F of the same fp32 value: visiting both ordered pairs and keeping only the i side
// computed C6_ij and C6_ji through different fp32 factorisations, whose difference left a
// systematic net force (whole-system drift, 2026-11-20). The self pair (j == i, its periodic
// images, i side only) keeps its ev halved (both tau and -tau are visited). body sums one pair
// (all its images) in fp32 into ci / cj, which are fresh every step; the sums over pairs (lane
// reduction for i, steps for j) are float-float (two_sum), so +F and -F of a pair reach the two
// atoms with the same rounding. The partial sums are flushed every D3_CHUNK steps into Ai (atom
// 32 I + lane), Aj (atom 32 J + lane) and Ev (warp total of component k on lane k, one fp64 add per flush).
// F64 (default; JET_D3_FP64=0 selects float-float, see PairJetD3::allocate): the sums over pairs are plain fp64 instead,
// the i side reduced over the lanes in fp64 and both sides added straight into Ai / Aj (exact sums of
// the same fp32 pair values, so at least as accurate as float-float, with a fraction of the adds).
// ilocal (several ranks): a step whose i atom this rank does not own is skipped, so a pair is
// computed once in the whole job, by the rank of its lower-block atom; the j side of an
// off-diagonal tile still counts (the owner of j sums it after the ranks are allreduced).
template <int NI, int NEV, int U, bool F64, class Body>
__device__ __forceinline__ void tile_pairs(int n, int I, int J, const unsigned char *ilocal,
                                           double (&Ai)[NI], double (&Aj)[NI], double &Ev, Body &&body) {
    const int lane = threadIdx.x & 31;
    const bool diag = (I == J);
    const int j = 32 * J + lane;
    float fi[NI], fj[NI], fil[NI], fjl[NI], fe[NEV > 0 ? NEV : 1], fs[NEV > 0 ? NEV : 1];   // fil / fjl: low parts; fs: self-pair ev (halved)
    #pragma unroll
    for (int k = 0; k < NI; k++) { fi[k] = fj[k] = fil[k] = fjl[k] = 0.0f; Ai[k] = Aj[k] = 0.0; }
    #pragma unroll
    for (int k = 0; k < NEV; k++) { fe[k] = fs[k] = 0.0f; }
    Ev = 0.0;
    // U steps at a time: their i-side reductions overlap (the per-step dependency
    // chain, not the issue rate, limits the cheap kernels)
    for (int s = 0; s < 32; s += U) {
        float ci[U][NI], cj[U][NI];
        #pragma unroll
        for (int u = 0; u < U; u++) {
            #pragma unroll
            for (int k = 0; k < NI; k++) { ci[u][k] = cj[u][k] = 0.0f; }
            const int i = 32 * I + s + u;
            const bool self = diag && s + u == lane;
            if (i < n && j < n && (!ilocal || ilocal[i]) && (!diag || lane >= s + u)) { body(i, self, !self, ci[u], cj[u], self ? fs : fe); }
            #pragma unroll
            for (int k = 0; k < NI; k++) {
                if (F64) { Aj[k] += static_cast<double>(cj[u][k]); } else { two_sum(fj[k], fjl[k], cj[u][k]); }
            }
        }
        #pragma unroll
        for (int k = 0; k < NI; k++) {
            #pragma unroll
            for (int u = 0; u < U; u++) {
                if (F64) {
                    double t = static_cast<double>(ci[u][k]);
                    for (int off = 16; off > 0; off >>= 1) { t += __shfl_xor_sync(D3_FULL, t, off); }
                    if (lane == s + u) { Ai[k] += t; }
                } else {
                    float t = ci[u][k], tl = 0.0f;
                    warp_sum2(t, tl);
                    if (lane == s + u) { two_sum(fi[k], fil[k], t); fil[k] = __fadd_rn(fil[k], tl); }
                }
            }
        }
        if (((s + U) & (D3_CHUNK - 1)) == 0) {
            #pragma unroll
            for (int k = 0; k < NI && !F64; k++) {
                Ai[k] += static_cast<double>(fi[k]) + static_cast<double>(fil[k]);
                Aj[k] += static_cast<double>(fj[k]) + static_cast<double>(fjl[k]);
                fi[k] = fj[k] = fil[k] = fjl[k] = 0.0f;
            }
            float mine = 0.0f;
            #pragma unroll
            for (int k = 0; k < NEV; k++) {
                const float t = warp_sum(fmaf(0.5f, fs[k], fe[k])); mine = (lane == k) ? t : mine; fe[k] = fs[k] = 0.0f;
            }
            Ev += mine;
        }
    }
}

// Per-tile flush of the warp's energy / virial sums (component k on lane k) into one of the slots
__device__ __forceinline__ void add_ev(double *acc, int I, int J, int first, int cnt, double Ev) {
    const int lane = threadIdx.x & 31;
    if (lane < cnt) { atomicAdd(acc + ((I * 7 + J) & (D3_SLOTS - 1)) * 8 + first + lane, Ev); }
}

/* ----------------------------------------------------------------------
   Coordination number: cn[i] = sum over j and images of the damped count
   (every rank of a multi-rank job computes the pairs of its own atoms, both
   sides, see tile_of_warp; the partial arrays are summed by the caller)
------------------------------------------------------------------------- */

template <bool F64>
__global__ void k_d3_cn(
    int n, int niblocks, const int *__restrict__ iblocks, const unsigned char *__restrict__ ilocal,
    D3Cell cell, D3Tables tab, float cnthr, float K1,
    const float *__restrict__ x, const float *__restrict__ frac, const int *__restrict__ type,
    const float *__restrict__ bounds, double *__restrict__ cn
) {
    int I, J;
    const int ntiles = (n + 31) / 32;
    if (!tile_of_warp(ntiles, niblocks, iblocks, I, J)) { return; }
    const float rcn = sqrtf(cnthr);
    if (bounds && !blocks_near(bounds + 8 * I, bounds + 8 * J, cell, rcn, cell.rch_cn)) { return; }
    const int lane = threadIdx.x & 31;
    const int j = 32 * J + lane, jj = min(j, n - 1);
    const float xj[3] = { x[3*jj], x[3*jj+1], x[3*jj+2] };
    const float sj[3] = { frac[3*jj], frac[3*jj+1], frac[3*jj+2] };
    const float rcov_j = tab.rcov[type[jj]];

    double Ai[1], Aj[1], Ev;
    tile_pairs<1, 0, 4, F64>(n, I, J, ilocal, Ai, Aj, Ev, [&](int i, bool self, bool jside, float *ci, float *cj, float *) {
        const float rcov_sum = tab.rcov[type[i]] + rcov_j;
        const float d[3] = { xj[0] - x[3*i], xj[1] - x[3*i+1], xj[2] - x[3*i+2] };
        const float s[3] = { sj[0] - frac[3*i], sj[1] - frac[3*i+1], sj[2] - frac[3*i+2] };
        float t = 0.0f;
        for_images(cell, cell.rch_cn, d, s, self, [&](const float *, float r2) {
            if (r2 <= cnthr) {
                const float r_rc = rsqrtf(r2);
                t += 1.0f / (1.0f + expf(-K1 * ((rcov_sum * r_rc) - 1.0f)));
            }
        });
        ci[0] += t;
        if (jside) { cj[0] += t; }
    });
    // the j side is flushed on the diagonal tile too (it holds the pairs j > i, see tile_pairs)
    const int i = 32 * I + lane;
    if (i < n && (!ilocal || ilocal[i])) { atomicAdd(&cn[i], Ai[0]); }
    if (j < n) { atomicAdd(&cn[j], Aj[0]); }
}

/* ----------------------------------------------------------------------
   Per-atom C6 reference weights, factorised over the pair

   C6(CN_i, CN_j) = sum_ab L_ab C6ref_ab / sum_ab L_ab with
   L_ab = exp(K3 [(CN_i - cnref_a)^2 + (CN_j - cnref_b)^2]) = w_i[a] w_j[b],
   so with normalised weights wn = w / sum(w):
     C6_ij        = sum_b wn_j[b] v_i^{tj}[b],  v_i^t[b] = sum_a wn_i[a] C6ref[ti][t][a][b]
     dC6_ij/dCN_i = sum_b wn_j[b] u_i^{tj}[b],  u_i^t[b] = 2 K3 sum_a wn_i[a] (d_ia - <d_i>) C6ref[ti][t][a][b]
   with d_ia = CN_i - cnref_a and <d_i> = sum_a wn_i[a] d_ia. The weights are
   shifted by the nearest reference so the largest exponent is 0. The force
   kernel uses the transposed form with the partner's vectors:
   C6_ij = wn_i . v_j^{ti}, dC6_ij/dCN_i = wu_i . v_j^{ti}, dC6_ij/dCN_j = wn_i . u_j^{ti}
   with wu_i[a] = 2 K3 wn_i[a] (d_ia - <d_i>), so a lane keeps its atom's
   v_j, u_j in registers while the warp's i atoms have one type.
   Layout: w[n][16] (wn in 0..4, wu in 8..12), vu[n][np1][16] (v in 0..4, u in 8..12).
------------------------------------------------------------------------- */

static constexpr int D3_WSTRIDE = 16;
static constexpr int D3_VUSTRIDE = 16;

__global__ void k_d3_weights(
    int n, D3Tables tab, float K3,
    const int *__restrict__ type, const double *__restrict__ cn,
    float *__restrict__ w, float *__restrict__ vu
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) { return; }
    const int t = type[i];
    const int mx = tab.mxc[t];
    const double cni = cn[i];
    double d[D3_MAXC], wd[D3_MAXC], m = 1e300, W = 0.0, dbar = 0.0;
    for (int a = 0; a < D3_MAXC; a++) {
        d[a] = (a < mx) ? cni - static_cast<double>(tab.cnref[t * D3_MAXC + a]) : 0.0;
        if (a < mx) { m = fmin(m, d[a] * d[a]); }
    }
    for (int a = 0; a < D3_MAXC; a++) { wd[a] = (a < mx) ? exp(K3 * (d[a] * d[a] - m)) : 0.0; W += wd[a]; }
    for (int a = 0; a < D3_MAXC; a++) { wd[a] /= W; dbar += wd[a] * d[a]; }
    float wn[D3_MAXC], wu[D3_MAXC];
    for (int a = 0; a < D3_MAXC; a++) {
        wn[a] = static_cast<float>(wd[a]);
        wu[a] = static_cast<float>(2.0 * K3 * wd[a] * (d[a] - dbar));
    }
    for (int a = 0; a < 8; a++) {
        w[i * D3_WSTRIDE + a] = (a < D3_MAXC) ? wn[a] : 0.0f;
        w[i * D3_WSTRIDE + 8 + a] = (a < D3_MAXC) ? wu[a] : 0.0f;
    }
    for (int tj = 1; tj < tab.np1; tj++) {
        const float *c = tab.c6ref + (t * tab.np1 + tj) * D3_MAXC * D3_MAXC;
        const int mxj = tab.mxc[tj];
        float v[D3_MAXC] = {}, u[D3_MAXC] = {};
        for (int a = 0; a < mx; a++) {
            for (int b = 0; b < mxj; b++) {
                v[b] = fmaf(wn[a], c[a * D3_MAXC + b], v[b]);
                u[b] = fmaf(wu[a], c[a * D3_MAXC + b], u[b]);
            }
        }
        float *o = vu + (i * tab.np1 + tj) * D3_VUSTRIDE;
        for (int b = 0; b < 8; b++) { o[b] = (b < D3_MAXC) ? v[b] : 0.0f; o[8 + b] = (b < D3_MAXC) ? u[b] : 0.0f; }
    }
}

/* ----------------------------------------------------------------------
   Two-body dispersion: forces, dE/dCN, energy and virial

   DAMP: 0 zero damping, 1 Becke-Johnson. The j side of a pair needs the
   force -F and dC6_ij/dCN_j = wn_i . u_j^{ti}. EV = false leaves out the
   energy and virial sums, for the steps on which LAMMPS asks for neither
   (most MD steps without a barostat): forces and dE/dCN are the same.
------------------------------------------------------------------------- */

// 8 blocks per SM: the extra warps hide the per-image dependency chains better
// than the registers they cost (64 instead of ~100, a little spilling)
template <int DAMP, bool F64, bool EV>
__global__ void __launch_bounds__(D3_BLOCK, 8) k_d3_forces(
    int n, int niblocks, const int *__restrict__ iblocks, const unsigned char *__restrict__ ilocal,
    D3Cell cell, D3Tables tab, float rthr,
    float s6, float s8, float a1, float a2, float alp6, float alp8,
    const float *__restrict__ x, const float *__restrict__ frac, const int *__restrict__ type,
    const float *__restrict__ w, const float *__restrict__ vu,
    const float *__restrict__ bounds,
    double *__restrict__ dc6i, double *__restrict__ f, double *__restrict__ acc
) {
    int I, J;
    const int ntiles = (n + 31) / 32;
    if (!tile_of_warp(ntiles, niblocks, iblocks, I, J)) { return; }
    const float rvdw = sqrtf(rthr);
    if (bounds && !blocks_near(bounds + 8 * I, bounds + 8 * J, cell, rvdw, cell.rch_vdw)) { return; }
    const int lane = threadIdx.x & 31;
    const int j = 32 * J + lane, jj = min(j, n - 1);
    const float xj[3] = { x[3*jj], x[3*jj+1], x[3*jj+2] };
    const float sj[3] = { frac[3*jj], frac[3*jj+1], frac[3*jj+2] };
    const int tj = type[jj];
    const float r2r4_j = tab.r2r4[tj];
    // C6 vectors of atom j and the damping constants of the pair for the current
    // i type (atoms are grouped by cell then type: few reloads per tile)
    int tcur = -1;
    float vj[D3_MAXC], uj[D3_MAXC];
    float R0_6 = 0.0f, R0_8 = 0.0f, s8r42x3 = 0.0f;   // Becke-Johnson
    float unit_a1 = 0.0f, unit_a2 = 0.0f, s8r42 = 0.0f; // zero damping

    // ci / cj: f (3), dE/dCN; ev: energy, virial xx yy zz xy xz yz
    double Ai[4], Aj[4], Ev;
    tile_pairs<4, EV ? 7 : 0, 1, F64>(n, I, J, ilocal, Ai, Aj, Ev, [&](int i, bool self, bool jside, float *ci, float *cj, float *ev) {
        const int ti = type[i];
        const float d[3] = { xj[0] - x[3*i], xj[1] - x[3*i+1], xj[2] - x[3*i+2] };
        const float s[3] = { sj[0] - frac[3*i], sj[1] - frac[3*i+1], sj[2] - frac[3*i+2] };
        if (ti != tcur) {
            const float *vj_t = vu + (jj * tab.np1 + ti) * D3_VUSTRIDE;
            #pragma unroll
            for (int b = 0; b < D3_MAXC; b++) { vj[b] = vj_t[b]; uj[b] = vj_t[8 + b]; }
            if (DAMP == 1) {
                const float r42x3 = tab.r2r4[ti] * r2r4_j * 3.0f;
                const float R0 = fmaf(a1, sqrtf(r42x3), a2);
                const float R0_2 = R0 * R0;
                R0_6 = R0_2 * R0_2 * R0_2;
                R0_8 = R0_6 * R0_2;
                s8r42x3 = s8 * r42x3;
            } else {
                const float r0 = tab.r0ab[ti * tab.np1 + tj];
                unit_a1 = a1 * r0;
                unit_a2 = a2 * r0;
                s8r42 = s8 * tab.r2r4[ti] * r2r4_j;
            }
            tcur = ti;
        }

        // C6, dC6/dCN_i and dC6/dCN_j from the factorised reference weights
        const float *wi = w + i * D3_WSTRIDE;
        float c6 = 0.0f, dc6 = 0.0f, dc6j = 0.0f;
        #pragma unroll
        for (int b = 0; b < D3_MAXC; b++) {
            c6 = fmaf(wi[b], vj[b], c6);
            dc6 = fmaf(wi[8 + b], vj[b], dc6);
            if (jside) { dc6j = fmaf(wi[b], uj[b], dc6j); }
        }

        // C6 is the same for every image of a pair, so the image loop sums the C6-free terms (fp32, one
        // pair): G = sum g r with the pair force F = C6 g r, S = sum of the damped 1/r^6 and 1/r^8 terms
        // (energy and dE/dCN) and P = sum g r r (virial). C6 and dC6/dCN scale them once per pair below.
        float G[3] = { 0.0f, 0.0f, 0.0f }, S = 0.0f, P[6] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        for_images(cell, cell.rch_vdw, d, s, self, [&](const float *rij, float r2) {
            if (r2 > rthr) { return; }
            float g, dc6_rest;
            if (DAMP == 1) {
                // even powers of r only (no rsqrt) and one reciprocal (of the product of the two damping denominators)
                const float r4 = r2 * r2;
                const float r6 = r4 * r2;
                const float d6 = r6 + R0_6;
                const float d8 = fmaf(r4, r4, R0_8);
                const float d68_rc = 1.0f / (d6 * d8);
                const float t6_rc = d8 * d68_rc;
                const float t8_rc = d6 * d68_rc;
                g = -fmaf(8.0f * s8r42x3 * r6, t8_rc * t8_rc, 6.0f * s6 * r4 * (t6_rc * t6_rc));
                dc6_rest = fmaf(s8r42x3, t8_rc, s6 * t6_rc);
            } else {
                const float r_rc = rsqrtf(r2);
                float unit_rc_a1 = unit_a1 * r_rc;
                float t6 = unit_rc_a1 * unit_rc_a1; // ^2
                t6 *= unit_rc_a1; // ^3
                t6 *= t6; // ^6
                t6 *= unit_rc_a1; // ^7
                t6 *= t6; // ^14
                const float damp6 = 1.0f / fmaf(t6, 6.0f, 1.0f);
                float unit_rc_a2 = unit_a2 * r_rc;
                float t8 = unit_rc_a2 * unit_rc_a2; // ^2
                t8 *= t8; // ^4
                t8 *= t8; // ^8
                t8 *= t8; // ^16
                const float damp8 = 1.0f / fmaf(t8, 6.0f, 1.0f);
                const float r2_rc = r_rc * r_rc;
                const float r6_rc = r2_rc * r2_rc * r2_rc;
                const float r8_rc = r6_rc * r2_rc;
                g = 6.0f * r8_rc * fmaf(r2_rc, s8r42 * damp8 * fmaf(3.0f * alp8 * t8, damp8, -4.0f), s6 * damp6 * fmaf(alp6 * t6, damp6, -1.0f));
                dc6_rest = r6_rc * fmaf(3.0f * r2_rc, s8r42 * damp8, s6 * damp6);
            }
            // explicit fma: the forces must not depend on EV through the compiler's choice of contraction
            G[0] = fmaf(g, rij[0], G[0]); G[1] = fmaf(g, rij[1], G[1]); G[2] = fmaf(g, rij[2], G[2]);
            S += dc6_rest;
            if (EV) {
                const float gr[3] = { g * rij[0], g * rij[1], g * rij[2] };
                P[0] = fmaf(gr[0], rij[0], P[0]);
                P[1] = fmaf(gr[1], rij[1], P[1]);
                P[2] = fmaf(gr[2], rij[2], P[2]);
                P[3] = fmaf(gr[0], rij[1], P[3]);
                P[4] = fmaf(gr[0], rij[2], P[4]);
                P[5] = fmaf(gr[1], rij[2], P[5]);
            }
        });
        if (!self) {
            #pragma unroll
            for (int k = 0; k < 3; k++) {
                const float fk = c6 * G[k];   // +F on j and -F on i: the same fp32 value
                ci[k] -= fk;
                if (jside) { cj[k] += fk; }
            }
        }
        if (EV) {
            #pragma unroll
            for (int k = 0; k < 6; k++) { ev[1 + k] += c6 * P[k]; }
            ev[0] -= S * c6;
        }
        ci[3] += S * dc6;
        if (jside) { cj[3] += S * dc6j; }
    });
    const int i = 32 * I + lane;
    if (i < n && (!ilocal || ilocal[i])) {
        atomicAdd(&f[3*i], Ai[0]); atomicAdd(&f[3*i+1], Ai[1]); atomicAdd(&f[3*i+2], Ai[2]);
        atomicAdd(&dc6i[i], Ai[3]);
    }
    if (j < n) {
        atomicAdd(&f[3*j], Aj[0]); atomicAdd(&f[3*j+1], Aj[1]); atomicAdd(&f[3*j+2], Aj[2]);
        atomicAdd(&dc6i[j], Aj[3]);
    }
    if (EV) { add_ev(acc, I, J, 0, 7, Ev); }
}

/* ----------------------------------------------------------------------
   Force and virial from dE/dCN through the coordination number
   (EV = false: no virial sums, see k_d3_forces)
------------------------------------------------------------------------- */

template <bool F64, bool EV>
__global__ void k_d3_forces_dcn(
    int n, int niblocks, const int *__restrict__ iblocks, const unsigned char *__restrict__ ilocal,
    D3Cell cell, D3Tables tab, float cnthr, float K1,
    const float *__restrict__ x, const float *__restrict__ frac, const int *__restrict__ type,
    const float *__restrict__ bounds,
    const double *__restrict__ dc6i, double *__restrict__ f, double *__restrict__ acc
) {
    int I, J;
    const int ntiles = (n + 31) / 32;
    if (!tile_of_warp(ntiles, niblocks, iblocks, I, J)) { return; }
    const float rcn = sqrtf(cnthr);
    if (bounds && !blocks_near(bounds + 8 * I, bounds + 8 * J, cell, rcn, cell.rch_cn)) { return; }
    const int lane = threadIdx.x & 31;
    const int j = 32 * J + lane, jj = min(j, n - 1);
    const float xj[3] = { x[3*jj], x[3*jj+1], x[3*jj+2] };
    const float sj[3] = { frac[3*jj], frac[3*jj+1], frac[3*jj+2] };
    const float rcov_j = tab.rcov[type[jj]];
    const double dc6i_j = dc6i[jj];

    double Ai[3], Aj[3], Ev;
    tile_pairs<3, EV ? 6 : 0, 1, F64>(n, I, J, ilocal, Ai, Aj, Ev, [&](int i, bool self, bool jside, float *ci, float *cj, float *ev) {
        const float rcov_sum = tab.rcov[type[i]] + rcov_j;
        const float dc6i_sum = static_cast<float>(dc6i[i] + dc6i_j);
        const float d[3] = { xj[0] - x[3*i], xj[1] - x[3*i+1], xj[2] - x[3*i+2] };
        const float s[3] = { sj[0] - frac[3*i], sj[1] - frac[3*i+1], sj[2] - frac[3*i+2] };
        for_images(cell, cell.rch_cn, d, s, self, [&](const float *rij, float r2) {
            if (r2 >= cnthr) { return; }
            const float r_rc = rsqrtf(r2);
            const float expterm = expf(-K1 * (rcov_sum * r_rc - 1.0f));
            const float unit_rc = 1.0f / (r2 * (expterm + 1.0f) * (expterm + 1.0f));
            const float dcnn = -K1 * rcov_sum * expterm * unit_rc;
            const float x1 = dcnn * dc6i_sum;
            // rounded products (no contraction into the sums below): the forces must not depend on EV
            const float vec[3] = { __fmul_rn(x1 * rij[0], r_rc), __fmul_rn(x1 * rij[1], r_rc), __fmul_rn(x1 * rij[2], r_rc) };
            if (!self) {
                ci[0] -= vec[0]; ci[1] -= vec[1]; ci[2] -= vec[2];
                if (jside) { cj[0] += vec[0]; cj[1] += vec[1]; cj[2] += vec[2]; }
            }
            if (EV) {
                ev[0] += vec[0] * rij[0];
                ev[1] += vec[1] * rij[1];
                ev[2] += vec[2] * rij[2];
                ev[3] += vec[0] * rij[1];
                ev[4] += vec[0] * rij[2];
                ev[5] += vec[1] * rij[2];
            }
        });
    });
    const int i = 32 * I + lane;
    if (i < n && (!ilocal || ilocal[i])) { atomicAdd(&f[3*i], Ai[0]); atomicAdd(&f[3*i+1], Ai[1]); atomicAdd(&f[3*i+2], Ai[2]); }
    if (j < n) { atomicAdd(&f[3*j], Aj[0]); atomicAdd(&f[3*j+1], Aj[1]); atomicAdd(&f[3*j+2], Aj[2]); }
    if (EV) { add_ev(acc, I, J, 1, 6, Ev); }
}

// jet/d3/kk: add the packed forces to the Kokkos device force view, f[dest[k]] += conv * fd[k]
// (dest < 0: another rank owns the atom; every destination appears once, so no atomics)
__global__ void k_d3_add_forces(int n, const int *__restrict__ dest, const double *__restrict__ fd, double conv,
                                double *__restrict__ f) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) { return; }
    const int i = dest[k];
    if (i < 0) { return; }
    for (int m = 0; m < 3; m++) { f[3 * i + m] += fd[3 * k + m] * conv; }
}

// The kernel instances of the run-time choices: [fp64 pair sums][energy and virial sums], k_d3_forces [damping] first
static const decltype(&k_d3_cn<false>) cn_kernels[2] = { k_d3_cn<false>, k_d3_cn<true> };
static const decltype(&k_d3_forces<0, false, false>) forces_kernels[2][2][2] = {
    { { k_d3_forces<0, false, false>, k_d3_forces<0, false, true> }, { k_d3_forces<0, true, false>, k_d3_forces<0, true, true> } },
    { { k_d3_forces<1, false, false>, k_d3_forces<1, false, true> }, { k_d3_forces<1, true, false>, k_d3_forces<1, true, true> } } };
static const decltype(&k_d3_forces_dcn<false, false>) dcn_kernels[2][2] = {
    { k_d3_forces_dcn<false, false>, k_d3_forces_dcn<false, true> }, { k_d3_forces_dcn<true, false>, k_d3_forces_dcn<true, true> } };

/* ----------------------------------------------------------------------
   Per-atom buffers (host pinned + device), resized when the atom count changes
   (n = the local atoms on one rank, all N atoms on several)
------------------------------------------------------------------------- */

void PairJetD3::reallocate_arrays(int n) {
    if (n_save >= 0) {
        cudaFree(d_in); cudaFreeHost(h_in);
        cudaFree(d_cn); cudaFree(d_w); cudaFree(d_vu);
        cudaFree(d_out); cudaFreeHost(h_out);
        if (multi) { cudaFree(d_iloc); cudaFree(d_iblocks); cudaFreeHost(h_cn); }
    }
    n_save = n;
    const int nblocks = (n + 31) / 32;
    const size_t in = 8 * n + 8 * nblocks;   // x, frac, type, per-block bounds, dest
    CHECK_CUDA(cudaMalloc(&d_in, in * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_in, in * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_cn, 2 * n * sizeof(double)));
    d_dc6i = d_cn + n;
    CHECK_CUDA(cudaMalloc(&d_w, n * D3_WSTRIDE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_vu, static_cast<size_t>(n) * tab.np1 * D3_VUSTRIDE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, (3 * n + 8 * D3_SLOTS) * sizeof(double)));
    CHECK_CUDA(cudaMallocHost(&h_out, (3 * n + 8 * D3_SLOTS) * sizeof(double)));
    h_iloc.resize(n);
    if (multi) {
        CHECK_CUDA(cudaMalloc(&d_iloc, n * sizeof(unsigned char)));
        CHECK_CUDA(cudaMalloc(&d_iblocks, nblocks * sizeof(int)));
        CHECK_CUDA(cudaMallocHost(&h_cn, n * sizeof(double)));
        gx.resize(3 * n); rx.resize(3 * n);
        gtype.resize(n); rtype.resize(n); rslot.resize(n);
        locidx.assign(n, -1);
    }
}

/* ----------------------------------------------------------------------
   Lattice vectors (bohr) and the per-direction image bounds
------------------------------------------------------------------------- */

void PairJetD3::set_cell(D3Cell &cell, double lat[3][3]) {
    lat[0][0] = domain->xprd / AU_TO_ANG; lat[0][1] = 0.0;                    lat[0][2] = 0.0;
    lat[1][0] = domain->xy / AU_TO_ANG;   lat[1][1] = domain->yprd / AU_TO_ANG; lat[1][2] = 0.0;
    lat[2][0] = domain->xz / AU_TO_ANG;   lat[2][1] = domain->yz / AU_TO_ANG;   lat[2][2] = domain->zprd / AU_TO_ANG;

    double cp[3][3];
    MathExtra::cross3(lat[1], lat[2], cp[0]);
    MathExtra::cross3(lat[2], lat[0], cp[1]);
    MathExtra::cross3(lat[0], lat[1], cp[2]);
    const double vol = std::abs(MathExtra::dot3(lat[0], cp[0]));
    const int per[3] = { domain->xperiodic, domain->yperiodic, domain->zperiodic };
    for (int k = 0; k < 3; k++) {
        const double h = vol / MathExtra::len3(cp[k]);   // spacing of the lattice planes normal to k
        cell.h[k] = h;
        cell.rch_vdw[k] = std::sqrt(rthr) / h + 1e-5;
        cell.rch_cn[k] = std::sqrt(cnthr) / h + 1e-5;
        cell.per[k] = per[k];
        for (int m = 0; m < 3; m++) { cell.a[k][m] = lat[k][m]; }
    }
    cell.ortho = !domain->triclinic;
}

/* ----------------------------------------------------------------------
   Wrap atoms into the cell; fill the pinned input buffer (x, fractional x,
   type) and the per-block bounds of the fractional coordinates.

   The packed order is by cell, then by type (order[k] = source atom of
   packed atom k): a 32-atom block is then compact in space, so the pair
   tiles between far-apart blocks are skipped by blocks_near, while the i
   atoms of a pair tile keep one type so the force kernel reloads its
   per-type C6 vectors a few times per tile instead of every step. The grid
   only sets that order (an ~8-atom-per-cell cap keeps its bucket array
   small); the bounds are exact for any order, so the grid never changes a
   result. A small box (!use_cells: no prefilter, no bounds) is packed in
   the same order: the lanes of a warp, whose atoms are then neighbours,
   agree on which periodic images lie inside the cut-off, so fewer of them
   idle in the image loops.

   On several ranks the source atoms are the gathered tag-ordered ones
   (gx / gtype, index tag - 1), so the packing is the one-rank packing;
   h_iloc marks the atoms this rank owns and h_iblocks lists the packed
   blocks that contain at least one of them.
------------------------------------------------------------------------- */

void PairJetD3::load_atom_info(const double lat[3][3], const D3Cell &cell) {
    const int nloc = atom->nlocal;
    const int n = multi ? static_cast<int>(atom->natoms) : nloc;
    double cp[3][3];
    MathExtra::cross3(lat[1], lat[2], cp[0]);
    MathExtra::cross3(lat[2], lat[0], cp[1]);
    MathExtra::cross3(lat[0], lat[1], cp[2]);
    const double vol = MathExtra::dot3(lat[0], cp[0]);
    // cp: rows of the inverse lattice per Angstrom, so the loop over the atoms has no division
    for (int k = 0; k < 3; k++) { for (int m = 0; m < 3; m++) { cp[k][m] /= vol * AU_TO_ANG; } }

    int *type = atom->type;
    double **x = atom->x;
    const double *lo = domain->boxlo;

    // ~7 A cells: a 32-atom block then spans about one cell, so its sphere
    // is as small as the density allows and the prefilter rejects most
    // far-away tiles (one cell when the box is smaller than two of them)
    int bins[3];
    for (int k = 0; k < 3; k++) { bins[k] = std::max(1, static_cast<int>(3.0 / cell.rch_cn[k])); }
    long ncells = static_cast<long>(bins[0]) * bins[1] * bins[2];
    if (ncells > n / 8) {
        const double f = std::cbrt(n / 8.0 / ncells);
        ncells = 1;
        for (int k = 0; k < 3; k++) { bins[k] = std::max(1, static_cast<int>(bins[k] * f)); ncells *= bins[k]; }
    }
    const int nt = atom->ntypes + 1;                 // type 1..ntypes, one bucket per type
    h_s.resize(3 * n);
    h_key.resize(n);
    h_first.assign(ncells * nt + 1, 0);              // counting sort by (cell, type)
    for (int i = 0; i < n; i++) {
        const double *xi = multi ? gx.data() + 3 * i : x[i];
        const int ti = multi ? gtype[i] : type[i];
        const double p[3] = { xi[0] - lo[0], xi[1] - lo[1], xi[2] - lo[2] };
        double *s = &h_s[3 * i];
        int cellkey = 0;
        for (int k = 0; k < 3; k++) {
            double sk = MathExtra::dot3(cp[k], p);
            if (domain->periodicity[k]) { sk -= std::floor(sk); }
            s[k] = sk;
            int b = static_cast<int>(sk * bins[k]);   // truncated: the clamp covers sk < 0 (non-periodic)
            b = std::min(std::max(b, 0), bins[k] - 1);
            cellkey = cellkey * bins[k] + b;
        }
        h_key[i] = cellkey * nt + ti;
        h_first[h_key[i] + 1]++;
    }
    for (long k = 1; k <= ncells * nt; k++) { h_first[k] += h_first[k - 1]; }
    order.resize(n);
    for (int i = 0; i < n; i++) { order[h_first[h_key[i]]++] = i; }

    float *xh = h_in, *fh = h_in + 3 * n;
    int *th = reinterpret_cast<int*>(h_in + 6 * n);
    float *bh = h_in + 7 * n;
    int *dest = reinterpret_cast<int*>(bh + 8 * ((n + 31) / 32));   // LAMMPS index of packed atom k2, -1 if not local
    for (int k2 = 0; k2 < n; k2++) {
        const int i = order[k2];
        dest[k2] = multi ? locidx[i] : i;
        const double *s = &h_s[3 * i];
        for (int k = 0; k < 3; k++) { fh[3 * k2 + k] = s[k]; }
        for (int m = 0; m < 3; m++) {
            xh[3 * k2 + m] = s[0] * lat[0][m] + s[1] * lat[1][m] + s[2] * lat[2][m];
        }
        th[k2] = multi ? gtype[i] : type[i];
    }

    if (multi) {
        niblocks = 0;
        h_iblocks.resize((n + 31) / 32);
        for (int k2 = 0; k2 < n; k2++) {
            const bool loc = locidx[order[k2]] >= 0;
            h_iloc[k2] = loc;
            if (loc && (niblocks == 0 || h_iblocks[niblocks - 1] != k2 / 32)) { h_iblocks[niblocks++] = k2 / 32; }
        }
    }

    // Per-block bounds for the kernel prefilter, as the floats the kernels read:
    // fractional centre (from the block's first atom, so a block across the
    // periodic boundary stays compact), cartesian radius (orthogonal cells) and
    // per-direction half extents (triclinic cells). The 1e-3 / 1e-5 margins
    // keep float rounding from shrinking a bound.
    if (!use_cells) { return; }
    const int nblocks = (n + 31) / 32;
    for (int b = 0; b < nblocks; b++) {
        const int k0 = 32 * b, k1 = std::min(k0 + 32, n);
        const float *ref = fh + 3 * k0;
        double ctr[3] = { 0.0, 0.0, 0.0 };
        for (int k2 = k0; k2 < k1; k2++) {
            for (int k = 0; k < 3; k++) {
                double d = fh[3 * k2 + k] - ref[k];
                if (domain->periodicity[k]) { d -= std::nearbyint(d); }
                ctr[k] += d;
            }
        }
        float c[3];
        for (int k = 0; k < 3; k++) { c[k] = static_cast<float>(ref[k] + ctr[k] / (k1 - k0)); }
        float r2 = 0.0f, ext[3] = { 0.0f, 0.0f, 0.0f };
        for (int k2 = k0; k2 < k1; k2++) {
            float d[3], d2 = 0.0f;
            for (int k = 0; k < 3; k++) {
                d[k] = fh[3 * k2 + k] - c[k];
                if (domain->periodicity[k]) { d[k] -= static_cast<float>(std::nearbyint(d[k])); }
                ext[k] = std::max(ext[k], std::fabs(d[k]));
                d2 = fmaf(d[k] * cell.h[k], d[k] * cell.h[k], d2);
            }
            r2 = std::max(r2, d2);
        }
        bh[8 * b] = c[0]; bh[8 * b + 1] = c[1]; bh[8 * b + 2] = c[2];
        bh[8 * b + 3] = std::sqrt(r2) + 1e-3f;
        for (int k = 0; k < 3; k++) { bh[8 * b + 4 + k] = ext[k] + 1e-5f; }
    }
}

/* ----------------------------------------------------------------------
   Add energy, forces and virial to LAMMPS
------------------------------------------------------------------------- */

void PairJetD3::update(int eflag, int vflag) {
    const int n = n_save;                   // packed atoms: the local ones, or all N on several ranks
    const double *fd = h_out;
    const size_t dest_off = 7 * n + 8 * ((n + 31) / 32);   // see load_atom_info
    double ev[7] = {};
    for (int s = 0; s < D3_SLOTS; s++) {
        for (int k = 0; k < 7; k++) { ev[k] += h_out[3 * n + 8 * s + k]; }
    }

    // this rank's share of eng_vdwl / virial: every pair is computed once in the whole
    // job (see launch_multi_async), so the LAMMPS rank sums give the one-rank values
    if (eflag) { eng_vdwl += ev[0] * AU_TO_EV; }

    const double conv = AU_TO_EV / AU_TO_ANG;
    if (f_dev) {   // jet/d3/kk: add on the device; on several ranks the rank-summed forces go back first
        if (multi) { CHECK_CUDA(cudaMemcpyAsync(d_out, h_out, 3 * n * sizeof(double), cudaMemcpyHostToDevice, stream)); }
        k_d3_add_forces<<<(n + 255) / 256, 256, 0, stream>>>(n, reinterpret_cast<const int*>(d_in + dest_off), d_out, conv, f_dev);
        CHECK_CUDA(cudaStreamSynchronize(stream));
    } else {
        const int *dest = reinterpret_cast<const int*>(h_in + dest_off);
        double **f = atom->f;
        for (int k = 0; k < n; k++) {
            const int i = dest[k];
            if (i < 0) { continue; }            // another rank owns it
            f[i][0] += fd[3 * k] * conv;
            f[i][1] += fd[3 * k + 1] * conv;
            f[i][2] += fd[3 * k + 2] * conv;
        }
    }

    if (vflag) {
        for (int k = 0; k < 6; k++) { virial[k] += ev[1 + k] * AU_TO_EV; }
    }
}

/* ----------------------------------------------------------------------
   Several ranks: allgather the local (tag, x, type) so every rank holds all
   N atoms in tag order (source index tag - 1), then mark the atoms this rank
   owns: locidx[tag - 1] = local LAMMPS index, -1 otherwise.
------------------------------------------------------------------------- */

void PairJetD3::gather_atoms() {
    const int n = static_cast<int>(atom->natoms);
    const int nloc = atom->nlocal;
    const int np = comm->nprocs;
    std::vector<int> counts(np), displs(np + 1);
    MPI_Allgather(&nloc, 1, MPI_INT, counts.data(), 1, MPI_INT, world);
    displs[0] = 0;
    for (int r = 1; r <= np; r++) { displs[r] = displs[r - 1] + counts[r - 1]; }

    // local staging: positions (atom->x is an array of rows), tag - 1 and type
    std::vector<double> xs(3 * nloc);
    std::vector<int> islot(nloc), itype(nloc);
    std::fill(locidx.begin(), locidx.end(), -1);
    bool badtag = false;
    for (int i = 0; i < nloc; i++) {
        xs[3 * i] = atom->x[i][0]; xs[3 * i + 1] = atom->x[i][1]; xs[3 * i + 2] = atom->x[i][2];
        islot[i] = static_cast<int>(atom->tag[i]) - 1;
        itype[i] = atom->type[i];
        badtag = badtag || islot[i] < 0 || islot[i] >= n;   // a removed atom leaves a tag hole
    }
    if (badtag) { error->all(FLERR, "pair_style d3 on more than one rank needs consecutive atom tags"); }
    for (int i = 0; i < nloc; i++) { locidx[islot[i]] = i; }
    std::vector<int> xcounts(np), xd(np + 1), td(np + 1);
    for (int r = 0; r < np; r++) { xcounts[r] = 3 * counts[r]; }
    for (int r = 0; r <= np; r++) { xd[r] = 3 * displs[r]; td[r] = displs[r]; }
    MPI_Allgatherv(xs.data(), 3 * nloc, MPI_DOUBLE, rx.data(), xcounts.data(), xd.data(), MPI_DOUBLE, world);
    MPI_Allgatherv(itype.data(), nloc, MPI_INT, rtype.data(), counts.data(), td.data(), MPI_INT, world);
    MPI_Allgatherv(islot.data(), nloc, MPI_INT, rslot.data(), counts.data(), td.data(), MPI_INT, world);

    for (int k = 0; k < n; k++) {
        const int i = rslot[k];
        gx[3 * i] = rx[3 * k]; gx[3 * i + 1] = rx[3 * k + 1]; gx[3 * i + 2] = rx[3 * k + 2];
        gtype[i] = rtype[k];
    }
}

/* ----------------------------------------------------------------------
   Launch the whole GPU evaluation for the current positions (no wait).
   One rank: compute() does this itself unless pair_e3gnn already called it
   through extract("d3_prelaunch") at the start of its own compute(), so the
   D3 kernels overlap its host-side edge build.
   Several ranks: pair_e3gnn calls this after it has launched its GPU work
   (see launch_multi_async), so the D3 host part and kernels overlap that
   work; without an e3gnn prelaunch (d3 alone) compute() calls it itself.
   Each rank reaches this at the same point of the step, so the collectives
   below stay matched.
   ev: also sum the energy and the virial (see k_d3_forces). compute() knows
   it from eflag / vflag; a prelaunch comes before those, so it reads the
   steps the integrator marked in Update for the global energy / virial, and
   compute() launches again if it is asked for sums that were left out.
------------------------------------------------------------------------- */

void PairJetD3::prelaunch(Pair *pair) {
    auto *d3 = static_cast<PairJetD3 *>(pair);
    const Update *up = d3->lmp->update;
    const bool ev = up->eflag_global == up->ntimestep || up->vflag_global == up->ntimestep;
    if (d3->multi) { d3->launch_multi_async(ev); } else { d3->launch(ev); }
}

void PairJetD3::launch(bool ev) {
    const int n = atom->nlocal;
    if (static_cast<bigint>(n) != atom->natoms) { error->one(FLERR, "pair_style d3: all atoms must be local (single rank, no lost atoms)"); }
    if (n == 0) { return; }
    if (n != n_save) { reallocate_arrays(n); }

    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    if (timing && !tev[0]) { for (cudaEvent_t &e : tev) cudaEventCreate(&e); }
    auto mark = [&](int k) { if (timing) cudaEventRecord(tev[k], stream); };

    D3Cell cell;
    double lat[3][3];
    set_cell(cell, lat);
    // Block prefilter (see load_atom_info): worth it only when a periodic
    // direction holds more than ~3 cells of half the vdW cut-off. Below that
    // a cut-off sphere spans the box (one or two images per direction) and
    // every block pair is computed.
    use_cells = true;
    for (int k = 0; k < 3; k++) {
        if (cell.per[k] && cell.rch_vdw[k] > 2.0 / 3.0) { use_cells = false; }
    }
    load_atom_info(lat, cell);
    const auto t1 = clk::now();

    const int nblocks = (n + 31) / 32;
    const float *x = d_in, *frac = d_in + 3 * n;
    const int *type = reinterpret_cast<const int*>(d_in + 6 * n);
    const float *bounds = use_cells ? d_in + 7 * n : nullptr;
    double *f = d_out, *acc = d_out + 3 * n;

    mark(0);
    CHECK_CUDA(cudaMemcpyAsync(d_in, h_in, (8 * n + 8 * nblocks) * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemsetAsync(d_out, 0, (3 * n + 8 * D3_SLOTS) * sizeof(double), stream));
    CHECK_CUDA(cudaMemsetAsync(d_cn, 0, 2 * n * sizeof(double), stream));   // cn and dc6i
    mark(1);

    const int ntiles = (n + 31) / 32;
    const dim3 tiles(ntiles, (ntiles + D3_BLOCK / 32 - 1) / (D3_BLOCK / 32));
    cn_kernels[fp64_sums]<<<tiles, D3_BLOCK, 0, stream>>>(n, ntiles, nullptr, nullptr, cell, tab, cnthr, K1, x, frac, type, bounds, d_cn);
    mark(2);
    k_d3_weights<<<(n + D3_BLOCK - 1) / D3_BLOCK, D3_BLOCK, 0, stream>>>(n, tab, K3, type, d_cn, d_w, d_vu);
    mark(3);
    forces_kernels[damping][fp64_sums][ev]<<<tiles, D3_BLOCK, 0, stream>>>(n, ntiles, nullptr, nullptr, cell, tab, rthr, s6, s8, a1, a2,
                                                                           alp6, alp8, x, frac, type, d_w, d_vu, bounds, d_dc6i, f, acc);
    mark(4);
    dcn_kernels[fp64_sums][ev]<<<tiles, D3_BLOCK, 0, stream>>>(n, ntiles, nullptr, nullptr, cell, tab, cnthr, K1, x, frac, type, bounds,
                                                               d_dc6i, f, acc);
    mark(5);

    const size_t off = f_dev ? 3 * n : 0;   // jet/d3/kk adds the forces on the device: only the energy / virial slots come back
    CHECK_CUDA(cudaMemcpyAsync(h_out + off, d_out + off, (3 * n + 8 * D3_SLOTS - off) * sizeof(double), cudaMemcpyDeviceToHost, stream));
    mark(6);
    pending = true;
    ev_launched = ev;
    if (timing) {
        const auto ms = [](clk::duration d) { return std::chrono::duration<double, std::milli>(d).count(); };
        tsum[0] += ms(t1 - t0);
        tsum[1] += ms(clk::now() - t1);
    }
}

/* ----------------------------------------------------------------------
   Several ranks, first half: every rank holds all N atoms (one gather per
   step) and runs the one-rank kernels on the one-rank packing (cell order,
   block prefilter, periodic images). The i side of the pair sums is this
   rank's local atoms only (a per-atom flag gates the steps, and the launched
   i-role blocks are the ones that hold a local atom), so an unordered pair is
   computed exactly once in the whole job, with the full energy, virial,
   forces and dE/dCN of both sides. Local-local pairs of a diagonal tile are
   the one-rank case (i side only, energy halved), so their two ordered copies
   land on their two owners. The rank sums of eng_vdwl and virial are then the
   one-rank values.

   The CN is split the same way: each rank sums the damped counts of its own
   pairs (both sides, see tile_of_warp), and the partial arrays are summed
   over the ranks, because the weights and the force kernels of both sides
   need the whole array. dE/dCN is summed over the ranks before the dcn
   kernel, and the forces are summed over the ranks after it; each rank adds
   its own atoms' forces into f[] and nothing goes to a ghost, so newton pair
   may be on or off.

   pair_e3gnn calls this through extract("d3_prelaunch") after it has launched
   its GPU work, so the gather, pack, h2d, CN, CN allreduce and the launch of
   the weights and forces kernels overlap that work; finish_multi() then does
   the rest from compute(). d3 alone on several ranks runs both halves from
   compute(). All ranks reach this at the same point of the step.
------------------------------------------------------------------------- */

void PairJetD3::launch_multi_async(bool ev) {
    const int n = static_cast<int>(atom->natoms);
    pending = false;
    if (n == 0) { return; }
    if (n != n_save) { reallocate_arrays(n); }

    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    if (timing && !tev[0]) { for (cudaEvent_t &e : tev) cudaEventCreate(&e); }
    auto mark = [&](int k) { if (timing) cudaEventRecord(tev[k], stream); };
    const auto ms = [](clk::duration d) { return std::chrono::duration<double, std::milli>(d).count(); };

    gather_atoms();
    D3Cell cell;
    double lat[3][3];
    set_cell(cell, lat);
    use_cells = true;
    for (int k = 0; k < 3; k++) {
        if (cell.per[k] && cell.rch_vdw[k] > 2.0 / 3.0) { use_cells = false; }
    }
    load_atom_info(lat, cell);
    const auto t1 = clk::now();

    const int nblocks = (n + 31) / 32, ntiles = nblocks;
    const float *x = d_in, *frac = d_in + 3 * n;
    const int *type = reinterpret_cast<const int*>(d_in + 6 * n);
    const float *bounds = use_cells ? d_in + 7 * n : nullptr;
    double *f = d_out, *acc = d_out + 3 * n;
    // one warp per (i-role block of this rank, partner block) tile, see tile_of_warp; a rank
    // whose subdomain holds no atom launches nothing (niblocks == 0)
    const dim3 tiles(ntiles / 2 + 1, std::max(1, (niblocks + D3_BLOCK / 32 - 1) / (D3_BLOCK / 32)));

    mark(0);
    CHECK_CUDA(cudaMemcpyAsync(d_in, h_in, (8 * n + 8 * nblocks) * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_iloc, h_iloc.data(), n, cudaMemcpyHostToDevice, stream));
    if (niblocks) {
        CHECK_CUDA(cudaMemcpyAsync(d_iblocks, h_iblocks.data(), niblocks * sizeof(int), cudaMemcpyHostToDevice, stream));
    }
    CHECK_CUDA(cudaMemsetAsync(d_out, 0, (3 * n + 8 * D3_SLOTS) * sizeof(double), stream));
    CHECK_CUDA(cudaMemsetAsync(d_cn, 0, 2 * n * sizeof(double), stream));   // cn and dc6i
    mark(1);

    // CN of this rank's pairs, both sides; the weights and force kernels need the whole array
    cn_kernels[fp64_sums]<<<tiles, D3_BLOCK, 0, stream>>>(n, niblocks, d_iblocks, d_iloc, cell, tab, cnthr, K1, x, frac, type, bounds, d_cn);
    mark(2);
    const auto t1b = clk::now();
    CHECK_CUDA(cudaStreamSynchronize(stream));
    const auto t1c = clk::now();
    // async + explicit sync: a blocking D2H would also serialize with the other CUDA streams
    CHECK_CUDA(cudaMemcpyAsync(h_cn, d_cn, n * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    MPI_Allreduce(MPI_IN_PLACE, h_cn, n, MPI_DOUBLE, MPI_SUM, world);
    CHECK_CUDA(cudaMemcpyAsync(d_cn, h_cn, n * sizeof(double), cudaMemcpyHostToDevice, stream));
    const auto t1d = clk::now();

    k_d3_weights<<<(n + D3_BLOCK - 1) / D3_BLOCK, D3_BLOCK, 0, stream>>>(n, tab, K3, type, d_cn, d_w, d_vu);
    mark(3);
    forces_kernels[damping][fp64_sums][ev]<<<tiles, D3_BLOCK, 0, stream>>>(n, niblocks, d_iblocks, d_iloc, cell, tab, rthr, s6, s8, a1, a2,
                                                                           alp6, alp8, x, frac, type, d_w, d_vu, bounds, d_dc6i, f, acc);
    mark(4);
    const auto t1e = clk::now();
    pending = true;
    ev_launched = ev;

    if (timing) {
        tsum[0] += ms(t1 - t0);
        tsum[1] += ms(t1b - t1) + ms(t1e - t1d);
        tsum[2] += ms(t1c - t1b);       // the CN kernel of this rank
        tsum[10] += ms(t1d - t1c);      // its allreduce
    }
}

/* ----------------------------------------------------------------------
   Several ranks, second half: wait for the forces, sum dE/dCN over the ranks
   and run the dcn kernel, then sum the forces and add this rank's atoms.
------------------------------------------------------------------------- */

void PairJetD3::finish_multi(int eflag, int vflag) {
    const int n = n_save;
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    auto mark = [&](int k) { if (timing) cudaEventRecord(tev[k], stream); };
    const auto ms = [](clk::duration d) { return std::chrono::duration<double, std::milli>(d).count(); };

    D3Cell cell;
    double lat[3][3];
    set_cell(cell, lat);
    const int ntiles = (n + 31) / 32;
    const float *x = d_in, *frac = d_in + 3 * n;
    const int *type = reinterpret_cast<const int*>(d_in + 6 * n);
    const float *bounds = use_cells ? d_in + 7 * n : nullptr;
    double *f = d_out, *acc = d_out + 3 * n;
    const dim3 tiles(ntiles / 2 + 1, std::max(1, (niblocks + D3_BLOCK / 32 - 1) / (D3_BLOCK / 32)));

    CHECK_CUDA(cudaStreamSynchronize(stream));
    const auto t1 = clk::now();
    mark(5);
    // dE/dCN of this rank's pairs, summed over the ranks: the dcn kernel needs all of it
    CHECK_CUDA(cudaMemcpyAsync(h_cn, d_dc6i, n * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    MPI_Allreduce(MPI_IN_PLACE, h_cn, n, MPI_DOUBLE, MPI_SUM, world);
    CHECK_CUDA(cudaMemcpyAsync(d_dc6i, h_cn, n * sizeof(double), cudaMemcpyHostToDevice, stream));
    const auto t2 = clk::now();
    mark(6);

    dcn_kernels[fp64_sums][ev_launched]<<<tiles, D3_BLOCK, 0, stream>>>(n, niblocks, d_iblocks, d_iloc, cell, tab, cnthr, K1, x, frac,
                                                                        type, bounds, d_dc6i, f, acc);
    mark(7);
    CHECK_CUDA(cudaMemcpyAsync(h_out, d_out, (3 * n + 8 * D3_SLOTS) * sizeof(double), cudaMemcpyDeviceToHost, stream));
    mark(8);
    const auto t3 = clk::now();
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaGetLastError());
    const auto t4 = clk::now();

    // per-atom forces of the whole job, then only this rank's atoms go into f[]
    MPI_Allreduce(MPI_IN_PLACE, h_out, 3 * n, MPI_DOUBLE, MPI_SUM, world);
    const auto t5 = clk::now();
    pending = false;
    update(eflag, vflag);
    const auto t6 = clk::now();

    if (timing) {
        tsum[1] += ms(t3 - t2);
        tsum[2] += ms(t1 - t0) + ms(t4 - t3);
        tsum[3] += ms(t6 - t5);
        tsum[10] += ms(t2 - t1) + ms(t5 - t4);
        const int ev[6] = { 0, 1, 2, 3, 6, 7 };   // h2d, cn, weights, forces, dcn, d2h (4-5 and 5-6 are the MPI gaps)
        for (int k = 0; k < 6; k++) {
            float g = 0;
            cudaEventElapsedTime(&g, tev[ev[k]], tev[ev[k] + 1]);
            tsum[4 + k] += g;
        }
        tsteps++;
    }
}

/* ----------------------------------------------------------------------
   Compute : energy, force, and stress (Required)
------------------------------------------------------------------------- */

void PairJetD3::compute(int eflag, int vflag) {
    ev_init(eflag, vflag);
    if (eflag_atom || vflag_atom) { error->all(FLERR, "pair_style d3 does not compute per-atom energy or virial"); }
    const bool ev = eflag || vflag;
    if (pending && ev && !ev_launched) {   // prelaunched without the sums that are wanted after all: once more with them
        CHECK_CUDA(cudaStreamSynchronize(stream));
        pending = false;
    }
    if (multi) {
        // pair_e3gnn already ran the first half through prelaunch(); d3 alone runs it here
        if (!pending) { launch_multi_async(ev); }
        if (pending) { finish_multi(eflag, vflag); }
        return;
    }
    if (!pending) { launch(ev); }
    if (atom->nlocal == 0) { return; }

    using clk = std::chrono::steady_clock;
    const auto t2 = clk::now();
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaGetLastError());
    pending = false;
    const auto t3 = clk::now();

    update(eflag, vflag);
    if (timing) {
        const auto ms = [](clk::duration d) { return std::chrono::duration<double, std::milli>(d).count(); };
        tsum[2] += ms(t3 - t2);
        tsum[3] += ms(clk::now() - t3);
        for (int k = 0; k < 6; k++) {
            float g = 0;
            cudaEventElapsedTime(&g, tev[k], tev[k + 1]);
            tsum[4 + k] += g;
        }
        tsteps++;
    }
}

/* ----------------------------------------------------------------------
   extract: "d3_prelaunch" hands the launch entry point to pair_e3gnn,
   "sevenn_pair_first" lets a hybrid sub-style find out which of the two
   comes first in the hybrid list (PairHybrid::extract returns the first match)
------------------------------------------------------------------------- */

void *PairJetD3::extract(const char *str, int &dim) {
    dim = 0;
    if (std::strcmp(str, "d3_prelaunch") == 0) { return reinterpret_cast<void *>(&PairJetD3::prelaunch); }
    if (std::strcmp(str, "sevenn_pair_first") == 0) { return this; }
    return nullptr;
}

void PairJetD3::finish() {
    if (!timing || tsteps == 0) return;
    double m[11];
    for (int k = 0; k < 11; k++) m[k] = tsum[k] / tsteps;
    // several ranks: min / max over the ranks per column, so the split of the work is visible
    // (only rank 0's logfile line survives; a line per rank would be lost on the other ranks)
    if (multi) {
        double lo[11], hi[11];
        MPI_Reduce(m, lo, 11, MPI_DOUBLE, MPI_MIN, 0, world);
        MPI_Reduce(m, hi, 11, MPI_DOUBLE, MPI_MAX, 0, world);
        if (comm->me == 0) {
            utils::logmesg(lmp, "PairJetD3 timing, min/max over {} ranks, ms/step over {} steps: "
                           "host pack {:.3f}/{:.3f} launch {:.3f}/{:.3f} wait {:.3f}/{:.3f} update {:.3f}/{:.3f} | "
                           "gpu h2d {:.3f}/{:.3f} cn {:.3f}/{:.3f} weights {:.3f}/{:.3f} forces {:.3f}/{:.3f} "
                           "dcn {:.3f}/{:.3f} d2h {:.3f}/{:.3f} | mpi {:.3f}/{:.3f}\n",
                           comm->nprocs, tsteps,
                           lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], lo[3], hi[3], lo[4], hi[4], lo[5], hi[5],
                           lo[6], hi[6], lo[7], hi[7], lo[8], hi[8], lo[9], hi[9], lo[10], hi[10]);
        }
    } else {
        utils::logmesg(lmp, "PairJetD3 timing, ms/step over {} steps: host pack {:.3f} launch {:.3f} wait {:.3f} update {:.3f} | "
                       "gpu h2d {:.3f} cn {:.3f} weights {:.3f} forces {:.3f} dcn {:.3f} d2h {:.3f} | mpi {:.3f}\n",
                       tsteps, m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10]);
    }
    for (double &t : tsum) t = 0;
    tsteps = 0;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairJetD3::init_one(int i, int j) {
    if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");
    return 0.0;
}

/* ----------------------------------------------------------------------
   init specific to this pair style: no neighbor list, compute() sums over all
   atoms; on several ranks every rank keeps a copy of the whole system (one
   allgather per step), so no ghost cut-off is requested and newton pair does
   not matter
------------------------------------------------------------------------- */

void PairJetD3::init_style() {
    multi = comm->nprocs > 1;
    if (!multi) return;
    if (!atom->tag_consecutive()) {
        error->all(FLERR, "pair_style d3 on more than one rank requires consecutive atom tags");
    }
}

/* ----------------------------------------------------------------------
   restart files: nothing is written (restartinfo = 0)
------------------------------------------------------------------------- */

void PairJetD3::write_restart(FILE *fp) {}
void PairJetD3::read_restart(FILE *fp) {}
void PairJetD3::write_restart_settings(FILE *fp) {}
void PairJetD3::read_restart_settings(FILE *fp) {}
