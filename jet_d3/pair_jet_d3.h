/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
PairStyle(jet/d3, PairJetD3)

#else

#ifndef LMP_PAIR_JET_D3
#define LMP_PAIR_JET_D3

#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <cuda_runtime.h>

#include "pair.h"
#include "utils.h"
#include "atom.h"
#include "domain.h"
#include "error.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "math_extra.h"
#include "memory.h"

#include "pair_jet_d3_pars.h"

namespace LAMMPS_NS {

    // Cell description passed to the kernels by value (all lengths in bohr).
    struct D3Cell {
        float a[3][3];      // lattice vectors, a[k] is the k-th vector
        float rch_vdw[3];   // vdW cutoff / spacing of the lattice planes normal to direction k
        float rch_cn[3];    // same for the coordination-number cutoff
        float h[3];         // spacing of the lattice planes normal to direction k
        int per[3];         // periodic along direction k
        bool ortho;         // a is diagonal: fraction * h is the cartesian distance
    };

    // Element tables on the device, indexed by LAMMPS type (1-based, stride ntypes+1).
    struct D3Tables {
        int np1;
        const float *r2r4;
        const float *rcov;
        const float *r0ab;   // [np1][np1]
        const int *mxc;      // number of C6 reference geometries per type
        const float *c6ref;  // [np1][np1][MAXC][MAXC]
        const float *cnref;  // [np1][MAXC] reference coordination numbers
    };

    class PairJetD3 : public Pair {
    public:
        PairJetD3(class LAMMPS*);
        ~PairJetD3() override;

        void compute(int, int) override;
        void finish() override;
        void *extract(const char *, int &) override;
        static void prelaunch(Pair *);        // launch() through extract("d3_prelaunch")
        void settings(int, char**) override;
        void coeff(int, char**) override;
        double init_one(int i, int j) override;
        void init_style() override;

        void write_restart(FILE*) override;
        void read_restart(FILE*) override;
        void write_restart_settings(FILE*) override;
        void read_restart_settings(FILE*) override;

    protected:
        virtual void allocate();

        int find_atomic_number(std::string&);
        int is_int_in_array(int*, int, int);
        void read_r0ab(int*, int, std::vector<float>&);
        void get_limit_in_pars_array(int&, int&, int&, int&);
        void read_c6ab(int*, int, std::vector<float>&, std::vector<float>&, std::vector<int>&);

        void setfuncpar_zero();
        void setfuncpar_bj();
        void setfuncpar_zerom();
        void setfuncpar_bjm();
        void setfuncpar();

        void reallocate_arrays(int);
        void gather_atoms();
        void set_cell(D3Cell&, double lat[3][3]);
        void load_atom_info(const double lat[3][3], const D3Cell &cell);
        void launch();
        void launch_multi_async();
        void finish_multi(int, int);
        void update(int, int);
        bool pending = false;              // launch() / launch_multi_async() done for the current positions, compute() must still collect

        static constexpr int MAXC = 5;                  // maximum C6 references per element
        static constexpr double AU_TO_ANG = 0.52917726; // atomic unit --> angstrom
        static constexpr double AU_TO_EV = 27.21138505; // atomic unit --> eV
        static constexpr float K1 = 16.0;               // global ad hoc parameters
        static constexpr float K3 = -4.0;

        int damping;
        std::string functional;
        bool custom_params = false;        // pair_style d3 ... damp_bj custom s6 s8 a1 a2
        double custom_s6 = 1.0, custom_s8 = 0.0, custom_a1 = 0.0, custom_a2 = 0.0;
        float rthr;                        // R^2 cutoff for the dispersion sum (bohr^2)
        float cnthr;                       // R^2 cutoff for the coordination number (bohr^2)
        float s6, s8, s18, rs6, rs8, rs18, alp, alp6, alp8, a1, a2;

        D3Tables tab;                      // device element tables
        cudaStream_t stream = nullptr;     // all D3 GPU work runs and is waited for on this stream
        bool multi = false;                // comm->nprocs > 1: all N atoms replicated, images (see launch_multi_async)

        // per-atom device buffers (natoms: the local atoms on one rank, all atoms on several)
        int n_save = -1;
        float *d_in = nullptr;             // x[3n] frac[3n] type[n] bounds[8 nblocks]
        float *h_in = nullptr;             // pinned mirror of d_in
        std::vector<int> order;            // packed atom k is source atom order[k] (cell, then type)
        std::vector<int> h_key;            // per-atom sort key of the current step
        std::vector<int> h_first;          // counting-sort buckets
        std::vector<double> h_s;           // fractional coordinates of the current step
        std::vector<unsigned char> h_iloc; // multi: per packed atom, 1 if this rank owns it
        std::vector<int> h_iblocks;        // multi: packed blocks that hold at least one local atom
        int niblocks = 0;                  // multi: h_iblocks.size()
        bool use_cells = false;            // prefilter per 32-atom block (see launch())
        double *d_cn = nullptr;
        double *d_dc6i = nullptr;          // = d_cn + n (one allocation)
        unsigned char *d_iloc = nullptr;   // multi: packed ownership flag
        int *d_iblocks = nullptr;          // multi: i-block list
        float *d_w = nullptr;              // [n][16] normalised C6 reference weights (0..4) and dC6/dCN weights (8..12)
        float *d_vu = nullptr;             // [n][np1][16] per partner type: C6 vector v (0..4), dC6/dCN vector u (8..12)
        double *d_out = nullptr;           // f[3n], then D3_SLOTS x 8: energy and 6 virial components
        double *h_out = nullptr;           // pinned mirror of d_out
        // multi-rank: the all-N arrays exchanged with MPI, and the gather (tag-ordered) inputs
        double *h_cn = nullptr;            // pinned, 1 double per atom (dE/dCN staging)
        std::vector<double> gx;            // gathered positions, tag order: gx[3 (tag - 1) + k]
        std::vector<double> rx;            // gather receive buffer (rank order)
        std::vector<int> gtype;            // gathered types, tag order
        std::vector<int> rtype;            // gather receive buffer (rank order)
        std::vector<int> rslot;            // gather receive buffer: tag - 1
        std::vector<int> locidx;           // multi: tag - 1 -> local LAMMPS index, -1 if not local

        // SEVENN_TIMING=1 (see compute() and finish())
        bool timing = false;
        cudaEvent_t tev[9] = {};
        double tsum[11] = {};
        long tsteps = 0;
    };
}

#endif // LMP_PAIR_JET_D3
#endif // PAIR_CLASS
