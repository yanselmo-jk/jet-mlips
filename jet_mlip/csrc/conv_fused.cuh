// conv_fused: one fused kernel per SevenNet convolution layer (inference path).
//   out[dst] = (1/denom) * sum_{edges} TP(x[src], SH(vec), MLP(Bessel(|vec|) * XPLOR(|vec|)))
// Forward + first-order backward wrt x and edge_vec only (no parameter grads, no
// double backward: the backward op is a plain CUDA kernel without autograd).
//
// Structure (see docs/kernel_design.md):
//   block = 8 warps, 1 block/SM (dynamic smem <= 99 KB), grid = #SMs, warps claim dst nodes from a
//   per-launch work counter (one per path group; atomicAdd by lane 0).
//   warp = one dst node, lane = channel u. Channel *units* (path, 32-channel chunk; config.h) are the
//   unit of all per-channel work: mul 64/128 paths span 2/4 units, a mul-8/16 path idles lanes >= mul.
//   edges of a node stream in tiles of EPL edges (EPL = 4 fp32, 2 fp64):
//     stage   lanes < EPL: r, XPLOR cutoff, sh[SH_DIM] (lmax <= 3), src; lanes (e,i): Bessel B[e][i] (one sin per lane)
//     MLP     lane = k computes rows k and k+32 of H1[e][k], H2[e][k] for all EPL edges; the
//             staging is transposed (h1T[j][EPL], h2T[k][EPL]) so every w1s/w2s element is loaded
//             once per tile and one 16 B broadcast load feeds all EPL edges.
//     W tile  lane = u: W[e][i] = sum_k H2[e][k] * W3T[col(i)+u][k] (register tile, 16 B W3T loads)
//     TP      x[src] gathered into registers, fully unrolled constexpr fiber list, register acc.
//   W3T[col][W3S] (col = wcol+u, k contiguous, stride 64 + 16B/sizeof(T)) so lane u loads 16 B
//   vectors conflict-free. The W3 tile does not fit for the 11-path layer, so units are split
//   into NG balanced groups (compile-time, from the smem left after the fixed part); the kernel
//   loops groups as the outer loop (re-streaming every node's edges, recomputing the MLP) and
//   each group writes its own out columns (no atomics in the forward).
//
// Shared memory layout (bytes, T = float|double):
//   [w1s NB*(NH+1)] [w2s NH*(NH+1)]  (row stride NH+1: conflict-free row and column access) [W3T UPG*32*W3S] [Stage x 8 warps] ([bwd: grad_x row x 8 unless it aliases the dead MLP staging; Groups<>])
//
// Backward: recompute everything per node/group, then per edge (lane = u)
//   gW[p] += coef*gacc*x*sh ; gx += coef*gacc*sh*W (per-warp smem row -> contiguous atomicAdd)
//   gsh += coef*gacc*x*W (transpose-reduce over u) ; gW -> smem -> grad_H2 (lane = k) -> silu' ->
//   grad_H1 -> grad_B (lane = (e,i)) -> grad_r ; grad_vec = grad_r * v/r + J_sh^T gsh (lane e).
// Split backward (a.gh2 != nullptr, the default; exact MLP only): bwd_kernel stops at grad_H2 and adds it
// (RED) into gh2[E][NH]; bwd_mlp_kernel then runs the MLP backward once per edge (grad_H1, grad_B,
// grad_r, radial part of grad_vec) instead of once per path group. Costs E*NH*sizeof(T) transient bytes.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <torch/autograd.h>

#include <cmath>
#include <cstdlib>
#include <utility>

#include "configs_gen.h"

#pragma once
namespace jet_mlip {
namespace cf {

constexpr int NB = 8;       // Bessel basis size
constexpr int NH = 64;      // radial MLP hidden width
constexpr int WARPS = 8;
constexpr int THREADS = WARPS * 32;
#ifndef JET_SMEM_MAX          // set by setup.py from the build GPU's opt-in limit (sm_89: 101376)
#define JET_SMEM_MAX 101376
#endif
constexpr int SMEM_MAX = JET_SMEM_MAX;   // bigger limit (sm_80 163 KB, sm_90 227 KB) = fewer path groups
template <class T> constexpr int W3S = NH + 16 / (int)sizeof(T);    // W3T row stride (16B loads conflict-free)

// ---- compile-time unrolling over constexpr tables ---------------------------------------
template <class F, int... I>
__device__ __forceinline__ void static_for_impl(F&& f, std::integer_sequence<int, I...>) {
  (f(std::integral_constant<int, I>{}), ...);
}
template <int N, class F>
__device__ __forceinline__ void static_for(F&& f) {
  static_for_impl(f, std::make_integer_sequence<int, N>{});
}

// ---- config metadata (all constexpr; unit / x-block helpers in config.h) --------------------
constexpr int cpow2(int n) { int p = 1; while (p < n) p *= 2; return p; }
constexpr int clog2(int p) { int l = 0; while ((1 << l) < p) ++l; return l; }
// x register slots of unit group [U0,U1): one (block b, chunk ch) slot per pair read by a unit
template <class Cfg> constexpr bool xs_used(int U0, int U1, int b, int ch) {
  for (int i = U0; i < U1; ++i) if (xb_of<Cfg>(u_path<Cfg>(i)) == b && u_ch<Cfg>(i) == ch) return true;
  return false;
}
template <class Cfg> constexpr int xr_off(int U0, int U1, int b, int ch) {
  int o = 0;
  for (int c = 0; c < nxb<Cfg>(); ++c)
    for (int cc = 0; cc < nchmax<Cfg>(); ++cc) {
      if (c == b && cc == ch) return o;
      if (xs_used<Cfg>(U0, U1, c, cc)) o += 2 * xb_l<Cfg>(c) + 1;
    }
  return o;
}
// = xr_off(U0, U1, nxb, 0) in one pass over the units (the xr_off form exceeds the front end's constexpr step limit
// for 7net-omni's 128-channel lmax-3 shapes)
template <class Cfg> constexpr int nxr(int U0, int U1) {
  constexpr int NB = nxb<Cfg>(), NC = nchmax<Cfg>();
  bool used[NB][NC] = {};
  for (int i = U0; i < U1; ++i) used[xb_of<Cfg>(u_path<Cfg>(i))][u_ch<Cfg>(i)] = true;
  int o = 0;
  for (int c = 0; c < NB; ++c)
    for (int cc = 0; cc < NC; ++cc) o += used[c][cc] ? 2 * xb_l<Cfg>(c) + 1 : 0;
  return o;
}
// largest x slot count of any window of w consecutive units (bwd: grad_x row staging of a group)
template <class Cfg> constexpr int max_nxr(int w) {
  int m = 0;
  for (int u0 = 0; u0 < nunits<Cfg>(); ++u0) {
    const int n = nxr<Cfg>(u0, cmin(nunits<Cfg>(), u0 + w));
    m = n > m ? n : m;
  }
  return m;
}
// accumulator slot of (unit i, m3) for the group starting at U0
template <class Cfg> constexpr int acc_off(int U0, int i) {
  int o = 0;
  for (int j = U0; j < i; ++j) o += 2 * Cfg::paths[u_path<Cfg>(j)].l3 + 1;
  return o;
}
// (fiber, chunk) pairs whose unit lies in group [U0,U1), in (fiber, chunk) order: a group's fiber loops unroll over
// these only, unit by unit. Unrolling all NFIBER x chunks behind an `if constexpr` unit filter instantiated every fiber in every
// group, so front-end memory grew with NG x fibers (16-ch lmax 3, 611 fibers: > 6 GB for one kernel).
template <class Cfg> constexpr bool in_group(int U0, int U1, int k, int ch) {
  const int i = unit_of<Cfg>(Cfg::fibers[k].p, ch);
  return i >= U0 && i < U1;
}
template <class Cfg> constexpr int n_group_fibers(int U0, int U1) {
  int n = 0;
  for (int k = 0; k < Cfg::NFIBER; ++k)
    for (int ch = 0; ch < nch_of<Cfg>(Cfg::fibers[k].p); ++ch) n += in_group<Cfg>(U0, U1, k, ch);
  return n;
}
template <int N> struct FiberList { int f[N > 0 ? N : 1], ch[N > 0 ? N : 1]; };
template <class Cfg, int N> constexpr FiberList<N> group_fibers(int U0, int U1) {
  FiberList<N> l{};
  int n = 0;
  for (int i = U0; i < U1; ++i) {   // unit-major: each unit is one run (W / dW applied once per run)
    const int p = u_path<Cfg>(i), ch = u_ch<Cfg>(i);
    for (int k = 0; k < Cfg::NFIBER; ++k)
      if (Cfg::fibers[k].p == p) { l.f[n] = k; l.ch[n] = ch; ++n; }
  }
  return l;
}
template <class Cfg, int U0, int U1> struct GroupFibers {
  static constexpr int N = n_group_fibers<Cfg>(U0, U1);
  static constexpr FiberList<N> list = group_fibers<Cfg, N>(U0, U1);
  static constexpr int unit(int k) { return unit_of<Cfg>(Cfg::fibers[list.f[k]].p, list.ch[k]); }
  // last fiber of a run of one unit: per-unit partial sums are folded there (W / dW applied once per run)
  static constexpr bool run_end(int k) { return k + 1 == N || unit(k + 1) != unit(k); }
};
// W3T tile columns of group [U0,U1) (live columns; the tile is padded to (U1-U0)*32 with zeros)
template <class Cfg> constexpr int g_col0(int U0) { return u_col<Cfg>(U0); }
template <class Cfg> constexpr int g_ncol(int U0, int U1) { return u_col<Cfg>(U1 - 1) + u_ncol<Cfg>(U1 - 1) - u_col<Cfg>(U0); }
// lanes of out chunk [c0, c0+32) not covered by a live path (bit set -> write zero)
template <class Cfg> constexpr unsigned dead_mask(int c0) {
  unsigned m = 0;
  for (int l = 0; l < 32; ++l) {
    const int c = c0 + l;
    if (c >= Cfg::OUT_DIM) continue;
    bool live = false;
    for (int p = 0; p < Cfg::NPATH; ++p) {
      const auto& pp = Cfg::paths[p];
      if (c >= pp.off3 && c < pp.off3 + pp.mul * (2 * pp.l3 + 1)) live = true;
    }
    if (!live) m |= 1u << l;
  }
  return m;
}

// per-warp staging for one edge tile (transposed: one 16 B broadcast load feeds all EPL edges).
// MLP == false (radial table mode): the Bessel / hidden staging is not allocated.
template <class T, int EPL, bool MLP> struct MlpStage {
  T h2T[NH][EPL];    // MLP hidden 2 (bwd: overwritten by grad of pre-activation z2)
  T h1T[NH][EPL];    // MLP hidden 1 (bwd: grad z1)
  T BT[NB][EPL];     // Bessel * cutoff
};
template <class T, int EPL> struct MlpStage<T, EPL, false> {};
template <class T, int EPL, int SH_DIM, bool MLP = true> struct alignas(16) Stage : MlpStage<T, EPL, MLP> {
  T sh[EPL][SH_DIM];
  T r[EPL], cut[EPL];
  int src[EPL], ti[EPL];
};

// Units per group of the exact forward (smem-derived); table mode reuses it as a register budget.
template <class T, class Cfg, int EPL> constexpr int fwd_maxu() {
  constexpr int FIXED = (int)sizeof(T) * (NB * (NH + 1) + NH * (NH + 1))
                        + WARPS * (int)sizeof(Stage<T, EPL, Cfg::SH_DIM, true>);
  constexpr int PER_UNIT = (int)sizeof(T) * 32 * W3S<T>;
  return cmin(nunits<Cfg>(), (SMEM_MAX - FIXED) / PER_UNIT);
}

// Register budget per lane. launch_bounds(THREADS, 1): one block per SM, so 64K / THREADS registers per
// thread, capped by the ISA at 255 on every architecture (Kepler .. Blackwell); 8 kept as margin for ptxas.
constexpr int REG_BUDGET = cmin(65536 / THREADS, 255) - 8;
// Per-lane register estimate of a path group (calibrated on ptxas -v; lmax 3 hits the ceiling otherwise):
//   fixed: loop state, r/cut/src (~48) + x of EPL edges for every x block ((2 l1 + 1) each) + sh (bwd: + grad sh)
//   per unit: accumulators (2 l3 + 1) (bwd: + their grads) + W of EPL edges (bwd: + grad W)
template <class Cfg> constexpr int max_l3() { int m = 0; for (int p = 0; p < Cfg::NPATH; ++p) m = Cfg::paths[p].l3 > m ? Cfg::paths[p].l3 : m; return m; }
template <class Cfg, int EPL> constexpr int x_regs() { int s = 0; for (int b = 0; b < nxb<Cfg>(); ++b) s += EPL * (2 * xb_l<Cfg>(b) + 1); return s; }
template <class Cfg, int EPL, bool BWD> constexpr int fixed_regs() { return 48 + x_regs<Cfg, EPL>() + (BWD ? 2 : 1) * Cfg::SH_DIM; }
template <class Cfg, int EPL, bool BWD> constexpr int unit_regs() { return (BWD ? 3 : 1) * (2 * max_l3<Cfg>() + 1) + (BWD ? 2 : 1) * EPL; }   // bwd: acc, grad acc, grad W staging
// Registers of a group of w units with x counted per (block, chunk) slot the group actually reads (max over windows).
template <class Cfg, int EPL, bool BWD> constexpr int group_regs(int w) {
  return 48 + EPL * max_nxr<Cfg>(w) + (BWD ? 2 : 1) * Cfg::SH_DIM + w * unit_regs<Cfg, EPL, BWD>();
}
// Units per group within the register budget. Floor: the all-blocks count (x of every block, read or not), whose
// groups are ptxas-validated (sm_90: 7net-0 L1 bwd 5x6 at 255 regs / 0 spill although the slot count says 260).
// The slot count then grows the group where the units read few of the blocks (7net-omni: 7 x blocks, a group of
// 3 reads 2-3; the all-blocks count left 1 unit per table-bwd group, 94 groups for L2).
// Growth only for the exact-MLP backward: on sm_90 the grown fwd groups spilled (exact fwd 212 B) and ran 22 % slower
// than the floor split; the grown exact bwd is 12-14 % faster. The radial-table kernels have their own count (tab_maxu).
template <class Cfg, int EPL, bool BWD, bool GROW = false> constexpr int maxu_regs() {
  const int left = REG_BUDGET - fixed_regs<Cfg, EPL, BWD>();
  int w = left >= unit_regs<Cfg, EPL, BWD>() ? left / unit_regs<Cfg, EPL, BWD>() : 1;
  while (GROW && w < nunits<Cfg>() && group_regs<Cfg, EPL, BWD>(w + 1) <= REG_BUDGET) ++w;
  return w;
}
// Radial-table backward: live registers of group [U0,U1) -- loop state (48), x of EPL edges and the grad_x slots
// (nxr each), sh, grad_sh (pow2 for the transpose reduce) and the staged edge's grad_sh, gacc (2 l3 + 1 per unit), W and
// dW of EPL edges. Units per group: the fewest equal groups whose every group stays <= TBWD_REGS. TBWD_REGS is past
// the 255 ceiling on purpose: H200 (264k edges), the fewer groups win over the spills up to about this count --
// EquFlash L1 bwd 1 group (model 273) 1.43 ms vs 6 groups 3.06 / 3 groups 2.10, EquFlashV2 L1 4 groups (279) 5.43 ms
// vs 6 (249) 6.24 / 3 (324) 6.10.
constexpr int TBWD_REGS = 280;
template <class Cfg, int EPL> constexpr int tbwd_regs(int U0, int U1) {
  return 48 + (EPL + 1) * nxr<Cfg>(U0, U1) + 2 * Cfg::SH_DIM + cpow2(Cfg::SH_DIM) + acc_off<Cfg>(U0, U1) + 2 * EPL * (U1 - U0);
}
// Radial-table forward: loop state, x of EPL edges, sh, acc (2 l3 + 1 per unit), W of EPL edges, the current unit
// run's partial sums (y). Same limit: H200, EquFlash L1 fwd 1 group (model 209) 0.59 ms vs 3 groups 0.84,
// EquFlashV2 L1 3 groups (237) 1.71 ms vs 4 groups at EPL 2 (231) 2.24; no spills up to 289.
template <class Cfg, int EPL> constexpr int tfwd_regs(int U0, int U1) {
  return 48 + EPL * nxr<Cfg>(U0, U1) + Cfg::SH_DIM + acc_off<Cfg>(U0, U1) + EPL * (U1 - U0) + 2 * max_l3<Cfg>() + 1;
}
template <class Cfg, int EPL, bool BWD> constexpr int tab_maxu() {
  constexpr int NU = nunits<Cfg>();
  for (int ng = 1; ng < NU; ++ng) {
    const int upg = (NU + ng - 1) / ng;
    bool ok = true;
    for (int u0 = 0; ok && u0 < NU; u0 += upg)
      ok = (BWD ? tbwd_regs<Cfg, EPL>(u0, cmin(NU, u0 + upg)) : tfwd_regs<Cfg, EPL>(u0, cmin(NU, u0 + upg))) <= TBWD_REGS;
    if (ok) return upg;
  }
  return 1;
}

// Unit groups: as many 32-column units per W3T tile as fit in the smem left after the fixed part
// AND in the register budget (the second binds for lmax 3 and for wide table-mode backwards).
// TABLE (radial table mode): no w1s/w2s, no MLP staging, no W3T tile and no gW staging, so the
// groups get bigger (nano L1: fwd 6+5 -> 11, bwd 4+4+3 -> 6+5).
// MLP backward: the gW -> grad_H2 staging (two units at a time) and the grad_x row (the x slots of one
// group) alias the dead h1T / h2T, so the bwd group is as big as the fwd one (nano L1 bwd 4+4+3 -> 6+5,
// 7net-0 L1 10 -> 5 groups on sm_89). The group never exceeds the register budget: on sm_90 (227 KB) smem
// groups split into register sub-groups (v0.3.1 candidate) ran the 7net-0 L1 bwd 13 % slower than v0.3.0.
template <class T, class Cfg, int EPL, bool BWD, bool TABLE = false> struct Groups {
  static constexpr int NU = nunits<Cfg>();
  static constexpr int MAXU_REG = TABLE ? tab_maxu<Cfg, EPL, BWD>() : maxu_regs<Cfg, EPL, BWD, BWD && !TABLE>();
  // fp32: grad_x rows staged in smem for contiguous atomics; fp64 (gradcheck only) adds straight from registers
  static constexpr bool STAGE_GX = sizeof(T) == 4;
  static constexpr bool MLP_BWD = BWD && !TABLE;
  // grad_x row per warp: the x slots of one group (table mode: the full x row)
  static constexpr int SGX_ELEMS = !(BWD && STAGE_GX) ? 0 : TABLE ? Cfg::IN_DIM : 32 * max_nxr<Cfg>(MAXU_REG);
  // MLP bwd grad_x row aliases 1 = h1T, 2 = h2T+h1T, 0 = its own row after the stages
  static constexpr int SGX_ALIAS = !(MLP_BWD && STAGE_GX) ? 0 : SGX_ELEMS <= NH * EPL ? 1 : SGX_ELEMS <= 2 * NH * EPL ? 2 : 0;
  static constexpr int FIXED = (TABLE ? 0 : (int)sizeof(T) * (NB * (NH + 1) + NH * (NH + 1)))
                               + WARPS * (int)sizeof(Stage<T, EPL, Cfg::SH_DIM, !TABLE>)
                               + (BWD && STAGE_GX && SGX_ALIAS == 0 ? WARPS * (int)sizeof(T) * SGX_ELEMS : 0);
  static constexpr int PER_UNIT = TABLE ? 0 : (int)sizeof(T) * 32 * W3S<T>;
  // TABLE has smem to spare; the binding constraint is the per-unit register tile, so reuse the
  // forward MLP group size (validated at 0 spills for every config) in both directions.
  static constexpr int MAXU_SMEM = TABLE ? fwd_maxu<T, Cfg, EPL>() : cmin(NU, (SMEM_MAX - FIXED) / PER_UNIT);
  static constexpr int MAXU_RAW = cmin(MAXU_SMEM, MAXU_REG);
  static constexpr bool FITS = MAXU_RAW >= 1;   // checked by launch / EPL_OF
  static constexpr int MAXU = FITS ? MAXU_RAW : 1;
  static constexpr int NG = (NU + MAXU - 1) / MAXU;
  static constexpr int UPG = (NU + NG - 1) / NG;
  static constexpr int W3T_ELEMS = TABLE ? 0 : UPG * 32 * W3S<T>;
  static constexpr int SMEM = FIXED + UPG * PER_UNIT;
};

// edges per tile: the largest of 4/2/1 whose fwd and bwd tiles fit in smem and whose backward keeps at least 2 units
// per group in the register budget (lmax 3, wide x: EPL 4 -> 2). fp64 (gradcheck only) stops at 2: where the
// v0.3.1 bwd smem cut let it fit 4, the lmax-3 fp64 kernels doubled their group count and compile memory.
// Per direction: the 2-unit test uses the grown bwd count (maxu_regs GROW) for the bwd only; the fwd keeps the
// v0.4.1 EPL (omni L2/L3 fwd at the bwd's EPL 4 spilled and ran 22 % / 10 % slower than at EPL 2).
template <class T, class Cfg, int EPL, bool BWD> constexpr bool epl_fits() {
  return Groups<T, Cfg, EPL, false>::FITS && Groups<T, Cfg, EPL, true>::FITS
         && (EPL == 1 || maxu_regs<Cfg, EPL, true, BWD>() >= 2);
}
template <class T, class Cfg, bool BWD> constexpr int EPL_OF =
    sizeof(T) == 4 && epl_fits<T, Cfg, 4, BWD>() ? 4 : epl_fits<T, Cfg, 2, BWD>() ? 2 : 1;

template <class T> struct Args {
  const T* x; const T* vec; const int* src; const int* rowptr; const int* perm;
  const T* w1; const T* w2; const T* w3; const T* coeffs;
  T* out;                       // fwd
  const T* gout; T* gx; T* gvec; // bwd
  T* gh2;                       // bwd, split mode: grad_H2 per edge (E, NH), zeroed; nullptr -> recompute mode
  T* gshb;                      // table bwd: grad_sh and grad_r per edge (E, SH_DIM + 1), zeroed, summed over the path
                                // groups; tbwd_vec_kernel turns it into grad_vec
  const T* table;               // radial table (ntab, 4*nw) [k][c][v0, m0, d, m1] or (ntables, ntab, 4*nw) with
                                // tabidx; nullptr -> exact MLP path
  const int* tabidx;            // per-edge table index into the first dim of a 3-D table, or nullptr (= 0)
  int* ctr;                     // per-launch work counters, one per path group (dynamic node scheduling)
  T denom, r_on, r_cut, cst;
  T two_rc, rc2, ron2, inv_d3;  // per-launch constants: 2/r_cut, r_cut^2, r_on^2, 1/(rc2-ron2)^3 (XPLOR)
  T h, inv_h, inv_h_lo;         // radial table grid step r_cut/(ntab-1); its inverse as hi + lo (see w_tile_table)
  int n_out, nw, ntab;          // nw = NW = w3 row stride
  int n_edge;                   // E (bwd_mlp_kernel)
};

// ---- vector smem access: N consecutive T at a 16B-aligned address ----------------------
template <class T, int N> __device__ __forceinline__ void ldv(const T* p, T* o) {
  if constexpr (N * sizeof(T) == 16) {
    if constexpr (sizeof(T) == 4) { float4 v = *reinterpret_cast<const float4*>(p); o[0] = v.x; o[1] = v.y; o[2] = v.z; o[3] = v.w; }
    else { double2 v = *reinterpret_cast<const double2*>(p); o[0] = v.x; o[1] = v.y; }
  } else if constexpr (N * sizeof(T) == 32) {
    ldv<T, N / 2>(p, o); ldv<T, N / 2>(p + N / 2, o + N / 2);
  } else {
#pragma unroll
    for (int i = 0; i < N; ++i) o[i] = p[i];
  }
}
template <class T, int N> __device__ __forceinline__ void stv(T* p, const T* v) {
  if constexpr (N * sizeof(T) == 16) {
    if constexpr (sizeof(T) == 4) *reinterpret_cast<float4*>(p) = make_float4(v[0], v[1], v[2], v[3]);
    else *reinterpret_cast<double2*>(p) = make_double2(v[0], v[1]);
  } else if constexpr (N * sizeof(T) == 32) {
    stv<T, N / 2>(p, v); stv<T, N / 2>(p + N / 2, v + N / 2);
  } else {
#pragma unroll
    for (int i = 0; i < N; ++i) p[i] = v[i];
  }
}
// 1/x correctly rounded without the IEEE-division slow-path scaffolding (fp32: MUFU.RCP + fixup)
__device__ __forceinline__ float rcp(float x) { return __frcp_rn(x); }
__device__ __forceinline__ double rcp(double x) { return 1.0 / x; }
template <class T> __device__ __forceinline__ T silu(T z) { return z * rcp(T(1) + exp(-z)); }
template <class T> __device__ __forceinline__ T dsilu(T z) {
  T s = rcp(T(1) + exp(-z));
  return s * (T(1) + z * (T(1) - s));
}
// Sum v[NP] (NP power of 2 <= 32) over the 32 lanes with a transpose-reduce (NP shuffles instead of
// 5*NP). Afterwards lane l holds the total of value index (l >> (5 - log2 NP)) & (NP - 1).
// Recursive template so every v[] index is a compile-time constant (keeps v in registers).
template <class T, int NP, int S> __device__ __forceinline__ T warp_reduce_t(T* v) {
  if constexpr (S == 0) {
    return v[0];
  } else if constexpr (NP > 1) {
    const bool upper = (threadIdx.x & S) != 0;
#pragma unroll
    for (int i = 0; i < NP / 2; ++i) {
      const T send = upper ? v[i] : v[i + NP / 2];
      const T keep = upper ? v[i + NP / 2] : v[i];
      v[i] = keep + __shfl_xor_sync(0xffffffffu, send, S);
    }
    return warp_reduce_t<T, NP / 2, S / 2>(v);
  } else {
    v[0] += __shfl_xor_sync(0xffffffffu, v[0], S);
    return warp_reduce_t<T, 1, S / 2>(v);
  }
}

// ---- edge staging: r, cutoff, spherical harmonics (lanes < EPL); Bessel (lanes (e,i)) -----
template <class T, int SH_DIM>
__device__ __forceinline__ void sh_eval(T nx, T ny, T nz, T* sh) {
  const T s3 = T(1.7320508075688772), s15 = T(3.872983346207417), s5 = T(2.23606797749979);
  sh[0] = T(1);
  if constexpr (SH_DIM > 1) { sh[1] = s3 * nx; sh[2] = s3 * ny; sh[3] = s3 * nz; }
  if constexpr (SH_DIM > 4) {
    sh[4] = s15 * nx * nz;
    sh[5] = s15 * nx * ny;
    sh[6] = s5 * (ny * ny - T(0.5) * (nx * nx + nz * nz));
    sh[7] = s15 * ny * nz;
    sh[8] = T(0.5) * s15 * (nz * nz - nx * nx);
  }
  if constexpr (SH_DIM > 9) {   // l = 3, e3nn component normalization (same formulas as edge_embed.cu)
    const T a = T(1.0801234497346435), s7 = T(2.6457513110645907), b = T(1.6201851746019651);   // sqrt(42)/6, sqrt(7), sqrt(168)/8
    const T y2 = ny * ny, x2z2 = nx * nx + nz * nz;
    sh[9] = a * (sh[4] * nz + sh[8] * nx);
    sh[10] = s7 * sh[4] * ny;
    sh[11] = b * (T(4) * y2 - x2z2) * nx;
    sh[12] = T(0.5) * s7 * ny * (T(2) * y2 - T(3) * x2z2);
    sh[13] = b * nz * (T(4) * y2 - x2z2);
    sh[14] = s7 * sh[8] * ny;
    sh[15] = a * (sh[8] * nz - sh[4] * nx);
  }
  if constexpr (SH_DIM > 16) {   // l = 4: e3nn component normalization as degree-4 monomials of n, fitted in fp64 to
    // e3nn o3.spherical_harmonics(4, n, normalize=True, normalization="component") (residual 7e-15)
      sh[16] = T(8.8741196746494264) * nx * nz * nz * nz + T(-8.8741196746494246) * nx * nx * nx * nz;
      sh[17] = T(18.824850597016706) * nx * ny * nz * nz + T(-6.2749501990055689) * nx * nx * nx * ny;
      sh[18] = T(-3.3541019662496874) * nx * nz * nz * nz + T(20.124611797498112) * nx * ny * ny * nz + T(-3.3541019662496869) * nx * nx * nx * nz;
      sh[19] = T(-7.115124735378858) * nx * ny * nz * nz + T(9.4868329805051435) * nx * ny * ny * ny + T(-7.1151247353788598) * nx * nx * nx * ny;
      sh[20] = T(1.1250000000000011) * nz * nz * nz * nz + T(-9.0000000000000071) * ny * ny * nz * nz + T(2.9999999999999964) * ny * ny * ny * ny + T(2.2499999999999925) * nx * nx * nz * nz + T(-9.0000000000000018) * nx * nx * ny * ny + T(1.1250000000000011) * nx * nx * nx * nx;
      sh[21] = T(-7.1151247353788465) * ny * nz * nz * nz + T(9.4868329805051363) * ny * ny * ny * nz + T(-7.1151247353788589) * nx * nx * ny * nz;
      sh[22] = T(-1.6770509831248428) * nz * nz * nz * nz + T(10.062305898749058) * ny * ny * nz * nz + T(-10.062305898749058) * nx * nx * ny * ny + T(1.677050983124841) * nx * nx * nx * nx;
      sh[23] = T(6.2749501990055663) * ny * nz * nz * nz + T(-18.824850597016706) * nx * nx * ny * nz;
      sh[24] = T(2.2185299186623557) * nz * nz * nz * nz + T(-13.311179511974125) * nx * nx * nz * nz + T(2.2185299186623522) * nx * nx * nx * nx;
  }
  static_assert(SH_DIM == 1 || SH_DIM == 4 || SH_DIM == 9 || SH_DIM == 16 || SH_DIM == 25, "conv_fused: spherical harmonics up to lmax 4");
}
template <class T> __device__ __forceinline__ T xplor(T r, const Args<T>& a) {
  if (r < a.r_on) return T(1);
  const T r2 = r * r, c = a.rc2 - r2;
  return c * c * (a.rc2 + T(2) * r2 - T(3) * a.ron2) * a.inv_d3;
}
template <class T> __device__ __forceinline__ T dxplor(T r, const Args<T>& a) {
  if (r < a.r_on) return T(0);
  const T r2 = r * r, c = a.rc2 - r2, b = a.rc2 + T(2) * r2 - T(3) * a.ron2;
  return (T(-4) * r * c * b + c * c * T(4) * r) * a.inv_d3;
}

// Edge loads for one tile, held in registers by lanes < EPL so the next tile's perm/src/vec
// round trips overlap the current tile's compute (software pipelining).
template <class T> struct EdgePre { int eidx, s, ti; T vx, vy, vz; };
template <class T, int EPL>
__device__ __forceinline__ EdgePre<T> prefetch_edges(const Args<T>& a, int eb, int e1) {
  const int lane = threadIdx.x & 31;
  EdgePre<T> p{-1, 0, 0, T(0), T(0), T(0)};
  if (lane < EPL && eb + lane < e1) {
    p.eidx = a.perm ? a.perm[eb + lane] : eb + lane;
    p.s = a.src[p.eidx];
    p.ti = a.tabidx ? a.tabidx[p.eidx] : 0;
    p.vx = a.vec[3 * p.eidx]; p.vy = a.vec[3 * p.eidx + 1]; p.vz = a.vec[3 * p.eidx + 2];
  }
  return p;
}
// Stage the prefetched tile: r, cutoff, sh, src (lanes < EPL), then Bessel B[e][i] (lane = (e,i)).
// cp_hi + cp_lo = c_i / pi of this lane's Bessel slot i = lane % NB (double-float, computed once per
// kernel) so the phase fma(cp_hi, r, cp_lo * r) is rounded once, like c_i * r was; sinpi has no
// large-argument slow path, unlike sin.
template <class T, class Cfg, int EPL, bool MLP = true>
__device__ __forceinline__ void stage_edges(const Args<T>& a, Stage<T, EPL, Cfg::SH_DIM, MLP>* st, const EdgePre<T>& p,
                                            T cp_hi, T cp_lo) {
  const int lane = threadIdx.x & 31;
  if (lane < EPL) {
    if (p.eidx >= 0) {
      st->src[lane] = p.s; st->ti[lane] = p.ti;
      const T r = sqrt(p.vx * p.vx + p.vy * p.vy + p.vz * p.vz);
      st->r[lane] = r; st->cut[lane] = xplor(r, a);
      const T inv = rcp(max(r, T(1e-12)));
      sh_eval<T, Cfg::SH_DIM>(p.vx * inv, p.vy * inv, p.vz * inv, st->sh[lane]);
    } else {   // padding edge: zero basis and sh -> zero contribution
      st->src[lane] = 0; st->ti[lane] = 0; st->r[lane] = T(1); st->cut[lane] = T(0);
#pragma unroll
      for (int i = 0; i < Cfg::SH_DIM; ++i) st->sh[lane][i] = T(0);
    }
  }
  __syncwarp();
  if constexpr (MLP) {
    static_assert(EPL * NB <= 32, "conv_fused: one Bessel slot (e, i) per lane");
    const int e = lane / NB;
    if (e < EPL) {
      const T r = st->r[e];
      st->BT[lane % NB][e] = a.two_rc * sinpi(fma(cp_hi, r, cp_lo * r)) * rcp(r) * st->cut[e];
    }
    __syncwarp();
  }
}

// radial MLP for the tile: lane k computes rows k and k+32. Pre-activations returned in z1/z2.
template <class T, int EPL, int SH_DIM>
__device__ __forceinline__ void mlp_tile(Stage<T, EPL, SH_DIM, true>* st, const T* w1s, const T* w2s,
                                         T cst, T (*z1)[2], T (*z2)[2]) {
  const int lane = threadIdx.x & 31;
  T a[EPL][2];
#pragma unroll
  for (int e = 0; e < EPL; ++e) { a[e][0] = T(0); a[e][1] = T(0); }
#pragma unroll
  for (int i = 0; i < NB; ++i) {
    T b[EPL];
    ldv<T, EPL>(st->BT[i], b);
    const T w0 = w1s[i * (NH + 1) + lane], w1 = w1s[i * (NH + 1) + lane + 32];
#pragma unroll
    for (int e = 0; e < EPL; ++e) { a[e][0] += b[e] * w0; a[e][1] += b[e] * w1; }
  }
  T h[2][EPL];
#pragma unroll
  for (int e = 0; e < EPL; ++e) {
    z1[e][0] = a[e][0]; z1[e][1] = a[e][1];
    h[0][e] = cst * silu(a[e][0]); h[1][e] = cst * silu(a[e][1]);
    a[e][0] = T(0); a[e][1] = T(0);
  }
  stv<T, EPL>(st->h1T[lane], h[0]); stv<T, EPL>(st->h1T[lane + 32], h[1]);
  __syncwarp();
#pragma unroll 8
  for (int j = 0; j < NH; ++j) {
    T hh[EPL];
    ldv<T, EPL>(st->h1T[j], hh);
    const T w0 = w2s[j * (NH + 1) + lane], w1 = w2s[j * (NH + 1) + lane + 32];
#pragma unroll
    for (int e = 0; e < EPL; ++e) { a[e][0] += hh[e] * w0; a[e][1] += hh[e] * w1; }
  }
#pragma unroll
  for (int e = 0; e < EPL; ++e) {
    z2[e][0] = a[e][0]; z2[e][1] = a[e][1];
    h[0][e] = cst * silu(a[e][0]); h[1][e] = cst * silu(a[e][1]);
  }
  stv<T, EPL>(st->h2T[lane], h[0]); stv<T, EPL>(st->h2T[lane + 32], h[1]);
  __syncwarp();
}

// per-edge TP weights for this lane's channel of every unit in the group:
// W[e][j] = sum_k H2[e][k] * W3T[col(U0+j) - col(U0) + lane][k]
template <class T, class Cfg, int EPL, int U0, int U1>
__device__ __forceinline__ void w_tile(const Stage<T, EPL, Cfg::SH_DIM, true>* st, const T* w3t, T (*W)[U1 - U0]) {
  constexpr int NUG = U1 - U0;
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int e = 0; e < EPL; ++e)
#pragma unroll
    for (int j = 0; j < NUG; ++j) W[e][j] = T(0);
#pragma unroll 4
  for (int k = 0; k < NH; k += 4) {
    T h[4][EPL];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) ldv<T, EPL>(st->h2T[k + kk], h[kk]);
    static_for<NUG>([&](auto jj) {
      constexpr int j = decltype(jj)::value;
      T w[4];
      ldv<T, 4>(w3t + (u_col<Cfg>(U0 + j) - u_col<Cfg>(U0) + lane) * W3S<T> + k, w);
#pragma unroll
      for (int e = 0; e < EPL; ++e) {
        W[e][j] += h[0][e] * w[0]; W[e][j] += h[1][e] * w[1];
        W[e][j] += h[2][e] * w[2]; W[e][j] += h[3][e] * w[3];
      }
    });
  }
}

// Per-edge TP weights from the radial table instead of the MLP: cubic Hermite on [r_k, r_k+1],
// r_k = k*h, h = r_cut/(ntab-1); table[k][c] = {v0, m0, d, m1} = {W(r_k), dW/dr(r_k), (W(r_k+1)-W(r_k))/h,
// dW/dr(r_k+1)} of column c, so one (edge, column) is one 16 B load (v0.3.0 read 4 scalars from 2 rows).
// The node difference is tabulated instead of being formed as v1 - v0 in fp32: that cancellation,
// amplified by the 1/h of the Hermite derivative, put max|dF| at 1e-3 eV/A for ntab = 8192.
//   W  = v0 + h*[(3t^2-2t^3) d + (t^3-2t^2+t) m0 + (t^3-t^2) m1]
//   W' = (6t-6t^2) d + (3t^2-4t+1) m0 + (3t^2-2t) m1
// r >= r_cut clamps to t = 1, i.e. the last node, where XPLOR makes W and dW/dr exactly zero.
template <class T, class Cfg, int EPL, int U0, int U1, bool GRAD>
__device__ __forceinline__ void w_tile_table(const Args<T>& a, const Stage<T, EPL, Cfg::SH_DIM, false>* st,
                                             T (*W)[U1 - U0], T (*dW)[U1 - U0]) {
  constexpr int NUG = U1 - U0;
  const int lane = threadIdx.x & 31;
  const T h = a.h, inv_h = a.inv_h, inv_h_lo = a.inv_h_lo;
#pragma unroll
  for (int e = 0; e < EPL; ++e) {
    // hi + lo: a rounded 1/h alone scales every edge's grid coordinate by the same 1 + d (|d| <= 6e-8), a systematic
    // W(r (1 + d)) that biased the energy (NequIP-OAM, 2026-11-04); the pair leaves only per-edge rounding
    const T xg = fma(st->r[e], inv_h, st->r[e] * inv_h_lo);
    int k = (int)xg;
    k = k < 0 ? 0 : (k > a.ntab - 2 ? a.ntab - 2 : k);
    T t = xg - T(k);
    t = t < T(0) ? T(0) : (t > T(1) ? T(1) : t);
    const T t2 = t * t, t3 = t2 * t;
    const T b1 = (t3 - T(2) * t2 + t) * h, b2 = (T(3) * t2 - T(2) * t3) * h, b3 = (t3 - t2) * h;
    T c1 = T(0), c2 = T(0), c3 = T(0);
    if constexpr (GRAD) {
      c1 = T(3) * t2 - T(4) * t + T(1); c2 = T(6) * t - T(6) * t2; c3 = T(3) * t2 - T(2) * t;
    }
    const T* row = a.table + ((size_t)st->ti[e] * a.ntab + k) * (4 * a.nw);
    const T* mine = row + 4 * lane;   // full units: one base per edge + immediate column offsets
    static_for<NUG>([&](auto jj) {
      constexpr int j = decltype(jj)::value, C0 = u_col<Cfg>(U0 + j), NC = u_ncol<Cfg>(U0 + j);
      const T* p = NC == 32 ? mine + 4 * C0 : row + 4 * cmin(C0 + lane, a.nw - 1);   // idle lanes of a mul<32 unit
      T q[4];   // {v0, m0, d, m1}: one LDG.128 (fp32 only; the table path is never instantiated for fp64)
      if constexpr (sizeof(T) == 4) {
        const float4 v = __ldg(reinterpret_cast<const float4*>(p));
        q[0] = v.x; q[1] = v.y; q[2] = v.z; q[3] = v.w;
      } else {
        ldv<T, 4>(p, q);
      }
      const T v0 = q[0], m0 = q[1], d = q[2], m1 = q[3];
      W[e][j] = v0 + (b1 * m0 + b2 * d + b3 * m1);
      if constexpr (GRAD) dW[e][j] = c1 * m0 + c2 * d + c3 * m1;
    });
  }
}

// gather this lane's channel of x[src] for the (block, chunk) slots used by group [U0,U1)
template <class T, class Cfg, int U0, int U1>
__device__ __forceinline__ void gather_x(const T* x, int s, T* xr) {
  const int lane = threadIdx.x & 31;
  static_for<nxb<Cfg>()>([&](auto bb) {
    constexpr int b = decltype(bb)::value;
    static_for<nchmax<Cfg>()>([&](auto cc) {
      constexpr int ch = decltype(cc)::value;
      if constexpr (xs_used<Cfg>(U0, U1, b, ch)) {
        constexpr int L = xb_l<Cfg>(b), NC = xb_ncol<Cfg>(b, ch), o = xr_off<Cfg>(U0, U1, b, ch);
        const T* row = x + (size_t)s * Cfg::IN_DIM + xb_off<Cfg>(b) + (ch * 32 + lane) * (2 * L + 1);
        const bool live = NC == 32 || lane < NC;
#pragma unroll
        for (int m = 0; m < 2 * L + 1; ++m) xr[o + m] = live ? row[m] : T(0);
      }
    });
  });
}

// ---- forward ------------------------------------------------------------------------------
template <class T, class Cfg, int EPL, int U0, int U1, bool TABLE>
__device__ __forceinline__ void fwd_node(const Args<T>& a, int node, const T* w1s, const T* w2s,
                                         const T* w3t, Stage<T, EPL, Cfg::SH_DIM, !TABLE>* st, T cp_hi, T cp_lo) {
  constexpr int NUG = U1 - U0, NACC = acc_off<Cfg>(U0, U1), NXR = nxr<Cfg>(U0, U1);
  const int lane = threadIdx.x & 31;
  const int e0 = a.rowptr[node], e1 = a.rowptr[node + 1];
  T acc[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) acc[i] = T(0);

  EdgePre<T> pre = prefetch_edges<T, EPL>(a, e0, e1);
  for (int eb = e0; eb < e1; eb += EPL) {
    const int ne = min(EPL, e1 - eb);
    stage_edges<T, Cfg, EPL, !TABLE>(a, st, pre, cp_hi, cp_lo);
    pre = prefetch_edges<T, EPL>(a, eb + EPL, e1);   // next tile's loads in flight
    T xr[EPL][NXR];                                   // x[src] gathers in flight during the MLP
#pragma unroll
    for (int e = 0; e < EPL; ++e) {
      if (e >= ne) break;
      gather_x<T, Cfg, U0, U1>(a.x, st->src[e], xr[e]);
    }
    T W[EPL][NUG];
    if constexpr (TABLE) {
      w_tile_table<T, Cfg, EPL, U0, U1, false>(a, st, W, W);
    } else {
      T z1[EPL][2], z2[EPL][2];
      mlp_tile<T, EPL, Cfg::SH_DIM>(st, w1s, w2s, a.cst, z1, z2);
      w_tile<T, Cfg, EPL, U0, U1>(st, w3t, W);
    }
#pragma unroll
    for (int e = 0; e < EPL; ++e) {
      if (e >= ne) break;
      T sh[Cfg::SH_DIM];
#pragma unroll
      for (int m = 0; m < Cfg::SH_DIM; ++m) sh[m] = st->sh[e][m];
      using GF = GroupFibers<Cfg, U0, U1>;
      T y[NACC];   // W-free partial sums of the current unit run (zeros are RZ operands, not registers)
#pragma unroll
      for (int k = 0; k < NACC; ++k) y[k] = T(0);
      static_for<GF::N>([&](auto kk) {
        constexpr int k = decltype(kk)::value, ch = GF::list.ch[k];
        constexpr auto fb = Cfg::fibers[GF::list.f[k]];
        constexpr int i = unit_of<Cfg>(fb.p, ch), xi = xr_off<Cfg>(U0, U1, xb_of<Cfg>(fb.p), ch) + fb.m1;
        constexpr int ai = acc_off<Cfg>(U0, i) + fb.m3, si = Cfg::paths[fb.p].off2 + fb.m2;
        y[ai] += T(fb.coef) * xr[e][xi] * sh[si];
        if constexpr (GF::run_end(k)) {
          constexpr int a0 = acc_off<Cfg>(U0, i), D3 = 2 * Cfg::paths[fb.p].l3 + 1;
#pragma unroll
          for (int m = 0; m < D3; ++m) { acc[a0 + m] += W[e][i - U0] * y[a0 + m]; y[a0 + m] = T(0); }
        }
      });
    }
    __syncwarp();   // stage reused by the next tile
  }
  // write this group's columns (each column exactly once, no atomics)
  T* orow = a.out + (size_t)node * Cfg::OUT_DIM;
  static_for<NUG>([&](auto jj) {
    constexpr int i = U0 + decltype(jj)::value, NC = u_ncol<Cfg>(i);
    constexpr auto P = Cfg::paths[u_path<Cfg>(i)];
    const int u = u_ch<Cfg>(i) * 32 + lane;
    if (NC == 32 || lane < NC) {
#pragma unroll
      for (int m = 0; m < 2 * P.l3 + 1; ++m)
        orow[P.off3 + u * (2 * P.l3 + 1) + m] = acc[acc_off<Cfg>(U0, i) + m] / a.denom;
    }
  });
  if constexpr (U0 == 0) {   // dead (pruned) output blocks are zero
    static_for<(Cfg::OUT_DIM + 31) / 32>([&](auto cc) {
      constexpr int c0 = 32 * decltype(cc)::value;
      constexpr unsigned dead = dead_mask<Cfg>(c0);
      if constexpr (dead != 0) {
        if ((dead >> lane) & 1u) orow[c0 + lane] = T(0);
      }
    });
  }
}

template <class T>
__device__ __forceinline__ void load_weights(const Args<T>& a, T* w1s, T* w2s) {
  for (int i = threadIdx.x; i < NB * NH; i += THREADS) w1s[(i / NH) * (NH + 1) + i % NH] = a.w1[i] / sqrt(T(NB));
  for (int i = threadIdx.x; i < NH * NH; i += THREADS)
    w2s[(i / NH) * (NH + 1) + i % NH] = a.w2[i] / sqrt(T(NH));
}
// W3T tile of group [U0,U1): live columns [C0, C0+NC) of w3, zero padding to (U1-U0)*32 columns
template <class T, class Cfg, int U0, int U1>
__device__ __forceinline__ void load_w3t(const Args<T>& a, T* w3t) {
  constexpr int C0 = g_col0<Cfg>(U0), NC = g_ncol<Cfg>(U0, U1), NCP = (U1 - U0) * 32;
  for (int i = threadIdx.x; i < NCP * NH; i += THREADS) {
    const int k = i / NCP, cl = i % NCP;
    w3t[cl * W3S<T> + k] = cl < NC ? a.w3[k * a.nw + C0 + cl] / sqrt(T(NH)) : T(0);
  }
}

// c_i / pi of this lane's Bessel slot (i = lane % NB) as a double-float hi + lo: the coefficients are
// trainable, so once per kernel; the phase fma(hi, r, lo * r) is then rounded once, like c_i * r was.
template <class T> __device__ __forceinline__ void bessel_cp(const Args<T>& a, T& hi, T& lo) {
  const double cp = (double)a.coeffs[(threadIdx.x & 31) % NB] * 0.318309886183790671537767526745;
  hi = T(cp); lo = T(cp - (double)hi);
}
// Dynamic node scheduling: lane 0 claims the next node of this path group's counter (static striding
// left the slowest warps alone at the end: ~5 % of the SM time idle on sm_89). >= n_out: done.
__device__ __forceinline__ int next_node(int* ctr) {
  int n = 0;
  if ((threadIdx.x & 31) == 0) n = atomicAdd(ctr, 1);
  return __shfl_sync(0xffffffffu, n, 0);
}

template <class T, class Cfg, int EPL, bool TABLE>
__global__ void __launch_bounds__(THREADS, 1) fwd_kernel(Args<T> a) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  using G = Groups<T, Cfg, EPL, false, TABLE>;
  T* w1s = reinterpret_cast<T*>(smem_raw);
  T* w2s = w1s + (TABLE ? 0 : NB * (NH + 1));
  T* w3t = w2s + (TABLE ? 0 : NH * (NH + 1));
  auto* st = reinterpret_cast<Stage<T, EPL, Cfg::SH_DIM, !TABLE>*>(w3t + G::W3T_ELEMS) + (threadIdx.x >> 5);
  T cp_hi = T(0), cp_lo = T(0);
  if constexpr (!TABLE) bessel_cp(a, cp_hi, cp_lo);
  if constexpr (!TABLE) load_weights<T>(a, w1s, w2s);
  static_for<G::NG>([&](auto gg) {
    constexpr int U0 = decltype(gg)::value * G::UPG, U1 = cmin(G::NU, U0 + G::UPG);
    if constexpr (!TABLE) {
      __syncthreads();
      load_w3t<T, Cfg, U0, U1>(a, w3t);
      __syncthreads();
    }
    for (int node; (node = next_node(a.ctr + decltype(gg)::value)) < a.n_out;)
      fwd_node<T, Cfg, EPL, U0, U1, TABLE>(a, node, w1s, w2s, w3t, st, cp_hi, cp_lo);
  });
}

// ---- backward -----------------------------------------------------------------------------
// upstream grad of units [G0,G1) of this node, / denom as torch's div backward does
template <class T, class Cfg, int G0, int G1>
__device__ __forceinline__ void load_gacc(const Args<T>& a, int node, T* gacc) {
  const int lane = threadIdx.x & 31;
  const T* grow = a.gout + (size_t)node * Cfg::OUT_DIM;
  static_for<G1 - G0>([&](auto jj) {
    constexpr int i = G0 + decltype(jj)::value, NC = u_ncol<Cfg>(i);
    constexpr auto P = Cfg::paths[u_path<Cfg>(i)];
    const int u = u_ch<Cfg>(i) * 32 + lane;
    const bool live = NC == 32 || lane < NC;
#pragma unroll
    for (int m = 0; m < 2 * P.l3 + 1; ++m)
      gacc[acc_off<Cfg>(G0, i) + m] = live ? grow[P.off3 + u * (2 * P.l3 + 1) + m] / a.denom : T(0);
  });
}
// One staged edge tile (ne edges) of group [U0,U1): gather x, W tile, fiber loop (gW, grad_x, grad_sh),
// grad_x atomics, grad_sh reduce into gsh_mine, then (MLP mode) gW -> grad_H2 into g2.
template <class T, class Cfg, int EPL, int U0, int U1, bool TABLE>
__device__ __forceinline__ void bwd_tile(const Args<T>& a, int ne, const T* w1s, const T* w2s, const T* w3t,
                                         Stage<T, EPL, Cfg::SH_DIM, !TABLE>* st, T* sgx, const T* gacc,
                                         T (*z1)[2], T (*z2)[2], T (*g2)[2], T* gsh_mine, T* gr, int eidx) {
  using G = Groups<T, Cfg, EPL, true, TABLE>;
  constexpr int NUG = U1 - U0, SHD = Cfg::SH_DIM, NXR = nxr<Cfg>(U0, U1);
  constexpr int NRED = cpow2(SHD), RSH = 5 - clog2(NRED);   // transpose-reduce layout of grad_sh
  constexpr int NRG = cpow2(EPL);
  const int lane = threadIdx.x & 31;
  T xr[EPL][NXR];   // x[src] gathers in flight during the MLP
#pragma unroll
  for (int e = 0; e < EPL; ++e) {
    if (e >= ne) break;
    gather_x<T, Cfg, U0, U1>(a.x, st->src[e], xr[e]);
  }
  T W[EPL][NUG], dW[EPL][TABLE ? NUG : 1];   // dW/dr: table mode only
  if constexpr (TABLE) {
    w_tile_table<T, Cfg, EPL, U0, U1, true>(a, st, W, dW);
  } else {
    mlp_tile<T, EPL, SHD>(st, w1s, w2s, a.cst, z1, z2);
    w_tile<T, Cfg, EPL, U0, U1>(st, w3t, W);
    __syncwarp();   // the grad_x row may alias h2T
#pragma unroll
    for (int e = 0; e < EPL; ++e) { g2[e][0] = T(0); g2[e][1] = T(0); }
  }
#pragma unroll
  for (int m = 0; m < SHD; ++m) gsh_mine[m] = T(0);
#pragma unroll
  for (int i = 0; i < (TABLE ? NRG : 1); ++i) gr[i] = T(0);
  T gW[TABLE ? 1 : NUG][EPL];   // grad of the per-edge weights of this lane's columns (MLP mode)
#pragma unroll
  for (int j = 0; j < (TABLE ? 1 : NUG); ++j)
#pragma unroll
    for (int e = 0; e < EPL; ++e) gW[j][e] = T(0);
#pragma unroll
  for (int e = 0; e < EPL; ++e) {
    if (e >= ne) break;
    T sh[SHD], gx[NXR], gsh[NRED];
#pragma unroll
    for (int m = 0; m < SHD; ++m) sh[m] = st->sh[e][m];
#pragma unroll
    for (int m = 0; m < NRED; ++m) gsh[m] = T(0);
#pragma unroll
    for (int i = 0; i < NXR; ++i) gx[i] = T(0);
    using GF = GroupFibers<Cfg, U0, U1>;
    constexpr int NY = TABLE ? acc_off<Cfg>(U0, U1) : 1;
    T y[NY];   // table mode: W-free message of the current unit run, gr += dW * <gacc, y> at its end
#pragma unroll
    for (int k = 0; k < NY; ++k) y[k] = T(0);
    static_for<GF::N>([&](auto kk) {
      constexpr int k = decltype(kk)::value, ch = GF::list.ch[k];
      constexpr auto fb = Cfg::fibers[GF::list.f[k]];
      constexpr int i = unit_of<Cfg>(fb.p, ch), xi = xr_off<Cfg>(U0, U1, xb_of<Cfg>(fb.p), ch) + fb.m1;
      constexpr int ai = acc_off<Cfg>(U0, i) + fb.m3, si = Cfg::paths[fb.p].off2 + fb.m2;
      if constexpr (TABLE) {
        const T cx = T(fb.coef) * xr[e][xi], g = gacc[ai] * W[e][i - U0];   // g: CSE'd per (unit, m3)
        y[ai] += cx * sh[si];
        gsh[si] += cx * g;
        gx[xi] += T(fb.coef) * sh[si] * g;
        if constexpr (GF::run_end(k)) {
          constexpr int a0 = acc_off<Cfg>(U0, i), D3 = 2 * Cfg::paths[fb.p].l3 + 1;
          T s = T(0);
#pragma unroll
          for (int m = 0; m < D3; ++m) { s += gacc[a0 + m] * y[a0 + m]; y[a0 + m] = T(0); }
          gr[e] += dW[e][i - U0] * s;
        }
      } else {
        const T t = T(fb.coef) * gacc[ai];
        gW[i - U0][e] += t * xr[e][xi] * sh[si];
        gx[xi] += t * sh[si] * W[e][i - U0];
        gsh[si] += t * xr[e][xi] * W[e][i - U0];
      }
    });
    // grad_x[src]: the group's x slots assembled in smem (slot map), then contiguous atomics
    // (skipped when not needed, e.g. layer 0)
    if (a.gx == nullptr) {
    } else if constexpr (!G::STAGE_GX) {
      T* grow_x = a.gx + (size_t)st->src[e] * Cfg::IN_DIM;
      static_for<nxb<Cfg>()>([&](auto bb) {
        constexpr int b = decltype(bb)::value;
        static_for<nchmax<Cfg>()>([&](auto cc) {
          constexpr int ch = decltype(cc)::value;
          if constexpr (xs_used<Cfg>(U0, U1, b, ch)) {
            constexpr int L = xb_l<Cfg>(b), NC = xb_ncol<Cfg>(b, ch);
            if (NC == 32 || lane < NC) {
#pragma unroll
              for (int m = 0; m < 2 * L + 1; ++m)
                atomicAdd(grow_x + xb_off<Cfg>(b) + (ch * 32 + lane) * (2 * L + 1) + m, gx[xr_off<Cfg>(U0, U1, b, ch) + m]);
            }
          }
        });
      });
    } else {
      static_for<nxb<Cfg>()>([&](auto bb) {
        constexpr int b = decltype(bb)::value;
        static_for<nchmax<Cfg>()>([&](auto cc) {
          constexpr int ch = decltype(cc)::value;
          if constexpr (xs_used<Cfg>(U0, U1, b, ch)) {
            constexpr int L = xb_l<Cfg>(b), NC = xb_ncol<Cfg>(b, ch), o = xr_off<Cfg>(U0, U1, b, ch);
            constexpr int S0 = TABLE ? xb_off<Cfg>(b) + ch * 32 * (2 * L + 1) : 32 * o;   // row offset of the slot
            if (NC == 32 || lane < NC) {
#pragma unroll
              for (int m = 0; m < 2 * L + 1; ++m) sgx[S0 + lane * (2 * L + 1) + m] = gx[o + m];
            }
          }
        });
      });
      __syncwarp();
      T* grow_x = a.gx + (size_t)st->src[e] * Cfg::IN_DIM;
      static_for<nxb<Cfg>()>([&](auto bb) {
        constexpr int b = decltype(bb)::value;
        static_for<nchmax<Cfg>()>([&](auto cc) {
          constexpr int ch = decltype(cc)::value;
          if constexpr (xs_used<Cfg>(U0, U1, b, ch)) {
            constexpr int L = xb_l<Cfg>(b), NC = xb_ncol<Cfg>(b, ch), C0 = xb_off<Cfg>(b) + ch * 32 * (2 * L + 1);
            constexpr int S0 = TABLE ? C0 : 32 * xr_off<Cfg>(U0, U1, b, ch);
#pragma unroll
            for (int q = 0; q < 2 * L + 1; ++q) {
              const int idx = q * 32 + lane;
              if (NC == 32 || idx < NC * (2 * L + 1)) atomicAdd(grow_x + C0 + idx, sgx[S0 + idx]);
            }
          }
        });
      });
      __syncwarp();
    }
    // grad_sh: sum over channels u (lane m << RSH holds component m). Table mode: one RED per component into the
    // edge's buffer row (the l = 4 Jacobian ran on one lane per edge and path group: 12 % of the EquFlashV2 bwd);
    // MLP mode: delivered to lane e.
    const T red = warp_reduce_t<T, NRED, 16>(gsh);
    if constexpr (TABLE) {
      const int ei = __shfl_sync(0xffffffffu, eidx, e);
      if ((lane & ((1 << RSH) - 1)) == 0 && (lane >> RSH) < SHD) atomicAdd(a.gshb + (size_t)ei * (SHD + 1) + (lane >> RSH), red);
    } else {
#pragma unroll
      for (int m = 0; m < SHD; ++m) {
        const T s = __shfl_sync(0xffffffffu, red, m << RSH);
        if (lane == e) gsh_mine[m] += s;
      }
    }
  }
  if constexpr (!TABLE) {
    // grad_H2 (lane = k, k+32): sum over the group's columns of gW[col][e] * W3T[col][k];
    // gW of two units at a time staged in the dead h1T
    T* sgw = &st->h1T[0][0];
    static_for<(NUG + 1) / 2>([&](auto pp) {
      constexpr int J0 = U0 + 2 * decltype(pp)::value, J1 = cmin(U1, J0 + 2), NCOL = g_ncol<Cfg>(J0, J1);
      static_assert(NCOL <= NH, "conv_fused: two units of gW per h1T");
      static_for<J1 - J0>([&](auto jj) {
        constexpr int j = J0 + decltype(jj)::value, NC = u_ncol<Cfg>(j);
        if (NC == 32 || lane < NC) stv<T, EPL>(sgw + (u_col<Cfg>(j) - u_col<Cfg>(J0) + lane) * EPL, gW[j - U0]);
      });
      __syncwarp();
      const T* w3c = w3t + (u_col<Cfg>(J0) - u_col<Cfg>(U0)) * W3S<T> + lane;
#pragma unroll 8
      for (int col = 0; col < NCOL; ++col) {
        const T w0 = w3c[col * W3S<T>], w1 = w3c[col * W3S<T> + 32];
        T g[EPL];
        ldv<T, EPL>(sgw + col * EPL, g);
#pragma unroll
        for (int e = 0; e < EPL; ++e) { g2[e][0] += g[e] * w0; g2[e][1] += g[e] * w1; }
      }
      __syncwarp();
    });
  }
}

// MLP backward of one staged tile from grad_H2 (g2, lane = k, k+32; overwritten): dsilu -> grad z2 (h2T) -> grad_H1
// (w2s) -> dsilu -> grad z1 (h1T) -> grad_B (lane = (e, i)) -> grad_r. Returns grad_r of lane e's edge (lanes < EPL).
template <class T, class Cfg, int EPL>
__device__ __forceinline__ T mlp_bwd_tile(const Args<T>& a, Stage<T, EPL, Cfg::SH_DIM, true>* st, const T* w1s,
                                          const T* w2s, const T (*z1)[2], const T (*z2)[2], T (*g2)[2], T cp_hi, T cp_lo) {
  const int lane = threadIdx.x & 31;
  T gr_mine = T(0);
  {
    T v[2][EPL];
#pragma unroll
    for (int e = 0; e < EPL; ++e) {
      v[0][e] = g2[e][0] * a.cst * dsilu(z2[e][0]);
      v[1][e] = g2[e][1] * a.cst * dsilu(z2[e][1]);
    }
    stv<T, EPL>(st->h2T[lane], v[0]); stv<T, EPL>(st->h2T[lane + 32], v[1]);   // now grad z2
  }
  __syncwarp();
  // grad_H1 (lane = j, j+32)
#pragma unroll
  for (int e = 0; e < EPL; ++e) { g2[e][0] = T(0); g2[e][1] = T(0); }
#pragma unroll 8
  for (int k = 0; k < NH; ++k) {
    T g[EPL];
    ldv<T, EPL>(st->h2T[k], g);
    const T w0 = w2s[lane * (NH + 1) + k], w1 = w2s[(lane + 32) * (NH + 1) + k];
#pragma unroll
    for (int e = 0; e < EPL; ++e) { g2[e][0] += g[e] * w0; g2[e][1] += g[e] * w1; }
  }
  {
    T v[2][EPL];
#pragma unroll
    for (int e = 0; e < EPL; ++e) {
      v[0][e] = g2[e][0] * a.cst * dsilu(z1[e][0]);
      v[1][e] = g2[e][1] * a.cst * dsilu(z1[e][1]);
    }
    stv<T, EPL>(st->h1T[lane], v[0]); stv<T, EPL>(st->h1T[lane + 32], v[1]);   // now grad z1
  }
  __syncwarp();
  // grad_B and grad_r: lane = (e, i) slot (one per lane, see stage_edges)
  {
    const int e = lane / NB, i = lane % NB;
    T gr = T(0);
    if (e < EPL) {
      T gB = T(0);
#pragma unroll 8
      for (int j = 0; j < NH; ++j) gB += st->h1T[j][e] * w1s[i * (NH + 1) + j];   // stride NH+1: 8 rows hit 8 banks
      const T r = st->r[e], cut = st->cut[e], ci = a.coeffs[i];
      T sn, cs;
      sincospi(fma(cp_hi, r, cp_lo * r), &sn, &cs);
      const T ir = rcp(r);
      const T Bi = a.two_rc * sn * ir;
      const T dBi = a.two_rc * (ci * cs * ir - sn * ir * ir);
      gr = gB * (dBi * cut + Bi * dxplor(r, a));
    }
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) gr += __shfl_xor_sync(0xffffffffu, gr, o);   // sum over i of edge e
    const T v = __shfl_sync(0xffffffffu, gr, (lane % EPL) * NB);
    if (lane < EPL) gr_mine = v;
  }
  return gr_mine;
}

// grad_vec of one edge from grad_sh (g, SH_DIM) and grad_r: (d sh / d n)^T g projected by dn/dv = (I - n n^T) / r,
// plus grad_r * v / r
template <class T, int SHD>
__device__ __forceinline__ void vec_grad(T vx, T vy, T vz, T r, const T* gsh_mine, T gr_mine, T* o) {
  const T inv = rcp(max(r, T(1e-12))), ir = rcp(r);
  const T nx = vx * inv, ny = vy * inv, nz = vz * inv;
  const T s3 = T(1.7320508075688772), s15 = T(3.872983346207417), s5 = T(2.23606797749979);
  T gx = 0, gy = 0, gz = 0;   // (d sh / d n)^T gsh
  if constexpr (SHD > 1) { gx += s3 * gsh_mine[1]; gy += s3 * gsh_mine[2]; gz += s3 * gsh_mine[3]; }
  if constexpr (SHD > 4) {
    gx += s15 * nz * gsh_mine[4] + s15 * ny * gsh_mine[5] - s5 * nx * gsh_mine[6] - s15 * nx * gsh_mine[8];
    gy += s15 * nx * gsh_mine[5] + T(2) * s5 * ny * gsh_mine[6] + s15 * nz * gsh_mine[7];
    gz += s15 * nx * gsh_mine[4] - s5 * nz * gsh_mine[6] + s15 * ny * gsh_mine[7] + s15 * nz * gsh_mine[8];
  }
  if constexpr (SHD > 9) {   // l = 3; sh20 = s15 x z, sh24 = s15/2 (z^2 - x^2), so d(sh20 z + sh24 x)/dx = 3 sh24 etc.
    const T a = T(1.0801234497346435), s7 = T(2.6457513110645907), b = T(1.6201851746019651);
    const T x2 = nx * nx, y2 = ny * ny, z2 = nz * nz, sh20 = s15 * nx * nz, sh24 = T(0.5) * s15 * (z2 - x2);
    const T* g = gsh_mine;
    gx += T(3) * a * sh24 * g[9] + s7 * s15 * ny * nz * g[10] + b * (T(4) * y2 - T(3) * x2 - z2) * g[11]
          - T(3) * s7 * nx * ny * g[12] - T(2) * b * nx * nz * g[13] - s7 * s15 * nx * ny * g[14] - T(3) * a * sh20 * g[15];
    gy += s7 * sh20 * g[10] + T(8) * b * nx * ny * g[11] + T(1.5) * s7 * (T(2) * y2 - x2 - z2) * g[12]
          + T(8) * b * ny * nz * g[13] + s7 * sh24 * g[14];
    gz += T(3) * a * sh20 * g[9] + s7 * s15 * nx * ny * g[10] - T(2) * b * nx * nz * g[11] - T(3) * s7 * ny * nz * g[12]
          + b * (T(4) * y2 - x2 - T(3) * z2) * g[13] + s7 * s15 * ny * nz * g[14] + T(3) * a * sh24 * g[15];
  }
  if constexpr (SHD > 16) {   // l = 4: d/dn of the sh_eval monomials (the projection below drops the radial part)
    const T* g = gsh_mine;
    gx += T(8.8741196746494264) * nz * nz * nz * g[16] + T(-26.622359023948274) * nx * nx * nz * g[16] + T(18.824850597016706) * ny * nz * nz * g[17] + T(-18.824850597016706) * nx * nx * ny * g[17] + T(-3.3541019662496874) * nz * nz * nz * g[18] + T(20.124611797498112) * ny * ny * nz * g[18] + T(-10.062305898749061) * nx * nx * nz * g[18] + T(-7.115124735378858) * ny * nz * nz * g[19] + T(9.4868329805051435) * ny * ny * ny * g[19] + T(-21.345374206136579) * nx * nx * ny * g[19] + T(4.4999999999999849) * nx * nz * nz * g[20] + T(-18.000000000000004) * nx * ny * ny * g[20] + T(4.5000000000000044) * nx * nx * nx * g[20] + T(-14.230249470757718) * nx * ny * nz * g[21] + T(-20.124611797498115) * nx * ny * ny * g[22] + T(6.7082039324993641) * nx * nx * nx * g[22] + T(-37.649701194033412) * nx * ny * nz * g[23] + T(-26.622359023948249) * nx * nz * nz * g[24] + T(8.8741196746494087) * nx * nx * nx * g[24];
    gy += T(18.824850597016706) * nx * nz * nz * g[17] + T(-6.2749501990055689) * nx * nx * nx * g[17] + T(40.249223594996224) * nx * ny * nz * g[18] + T(-7.115124735378858) * nx * nz * nz * g[19] + T(28.460498941515432) * nx * ny * ny * g[19] + T(-7.1151247353788598) * nx * nx * nx * g[19] + T(-18.000000000000014) * ny * nz * nz * g[20] + T(11.999999999999986) * ny * ny * ny * g[20] + T(-18.000000000000004) * nx * nx * ny * g[20] + T(-7.1151247353788465) * nz * nz * nz * g[21] + T(28.460498941515411) * ny * ny * nz * g[21] + T(-7.1151247353788589) * nx * nx * nz * g[21] + T(20.124611797498115) * ny * nz * nz * g[22] + T(-20.124611797498115) * nx * nx * ny * g[22] + T(6.2749501990055663) * nz * nz * nz * g[23] + T(-18.824850597016706) * nx * nx * nz * g[23];
    gz += T(26.622359023948277) * nx * nz * nz * g[16] + T(-8.8741196746494246) * nx * nx * nx * g[16] + T(37.649701194033412) * nx * ny * nz * g[17] + T(-10.062305898749063) * nx * nz * nz * g[18] + T(20.124611797498112) * nx * ny * ny * g[18] + T(-3.3541019662496869) * nx * nx * nx * g[18] + T(-14.230249470757716) * nx * ny * nz * g[19] + T(4.5000000000000044) * nz * nz * nz * g[20] + T(-18.000000000000014) * ny * ny * nz * g[20] + T(4.4999999999999849) * nx * nx * nz * g[20] + T(-21.34537420613654) * ny * nz * nz * g[21] + T(9.4868329805051363) * ny * ny * ny * g[21] + T(-7.1151247353788589) * nx * nx * ny * g[21] + T(-6.7082039324993712) * nz * nz * nz * g[22] + T(20.124611797498115) * ny * ny * nz * g[22] + T(18.824850597016699) * ny * nz * nz * g[23] + T(-18.824850597016706) * nx * nx * ny * g[23] + T(8.8741196746494229) * nz * nz * nz * g[24] + T(-26.622359023948249) * nx * nx * nz * g[24];
  }
  const T dot = nx * gx + ny * gy + nz * gz;
  o[0] = (gx - nx * dot) * inv + gr_mine * vx * ir;
  o[1] = (gy - ny * dot) * inv + gr_mine * vy * ir;
  o[2] = (gz - nz * dot) * inv + gr_mine * vz * ir;
}

// Group [U0,U1) of one dst node. Per edge tile: stage_edges, then bwd_tile (MLP forward, W tile, fiber loop,
// grad_x atomics, grad_sh reduce, gW -> grad_H2), then dsilu, grad_H1, grad_B, grad_r, grad_vec
// (split mode: grad_H2 -> a.gh2 instead, grad_vec gets the angular part only here).
template <class T, class Cfg, int EPL, int U0, int U1, bool TABLE>
__device__ __forceinline__ void bwd_node(const Args<T>& a, int node, const T* w1s, const T* w2s,
                                         const T* w3t, Stage<T, EPL, Cfg::SH_DIM, !TABLE>* st, T* sgx,
                                         T cp_hi, T cp_lo) {
  constexpr int SHD = Cfg::SH_DIM, NRG = cpow2(EPL), RSHG = 5 - clog2(NRG);   // transpose-reduce layout of grad_r
  const int lane = threadIdx.x & 31;
  const int e0 = a.rowptr[node], e1 = a.rowptr[node + 1];
  if (e0 == e1) return;
  T gacc[acc_off<Cfg>(U0, U1)];   // upstream grads of this group's columns, once per node
  load_gacc<T, Cfg, U0, U1>(a, node, gacc);

  EdgePre<T> pre = prefetch_edges<T, EPL>(a, e0, e1);
  for (int eb = e0; eb < e1; eb += EPL) {
    const int ne = min(EPL, e1 - eb);
    stage_edges<T, Cfg, EPL, !TABLE>(a, st, pre, cp_hi, cp_lo);
    const int eidx = pre.eidx;
    const T pvx = pre.vx, pvy = pre.vy, pvz = pre.vz;
    pre = prefetch_edges<T, EPL>(a, eb + EPL, e1);   // next tile's loads in flight
    T z1[EPL][2], z2[EPL][2];     // MLP mode only (pre-activations for the MLP backward)
    T g2[EPL][2];                 // grad_H2 (lane = k, k+32)
    T gsh_mine[SHD];              // lane e keeps grad_sh of its staged edge
    T gr[TABLE ? NRG : 1];        // table mode: grad_r partial of this lane (sum over its columns)
    bwd_tile<T, Cfg, EPL, U0, U1, TABLE>(a, ne, w1s, w2s, w3t, st, sgx, gacc, z1, z2, g2, gsh_mine, gr, eidx);
    T gr_mine = T(0);
    if constexpr (TABLE) {   // grad_r = sum over this group's W columns of grad_W * dW/dr
      const T red = warp_reduce_t<T, NRG, 16>(gr);
      gr_mine = __shfl_sync(0xffffffffu, red, (lane & (NRG - 1)) << RSHG);
    } else if (a.gh2 != nullptr) {   // split: grad_H2 of this group -> per-edge buffer; the MLP backward runs in bwd_mlp_kernel
#pragma unroll
      for (int e = 0; e < EPL; ++e) {
        if (e >= ne) break;
        T* g = a.gh2 + (size_t)__shfl_sync(0xffffffffu, eidx, e) * NH + lane;
        atomicAdd(g, g2[e][0]); atomicAdd(g + 32, g2[e][1]);
      }
    } else {
      gr_mine = mlp_bwd_tile<T, Cfg, EPL>(a, st, w1s, w2s, z1, z2, g2, cp_hi, cp_lo);
    }
    // grad_vec for the staged edge (lane e < ne); table mode: grad_r into the buffer row, tbwd_vec_kernel does the rest
    if (lane < ne) {
      if constexpr (TABLE) {
        atomicAdd(a.gshb + (size_t)eidx * (SHD + 1) + SHD, gr_mine);
      } else {
        T o[3];
        vec_grad<T, SHD>(pvx, pvy, pvz, st->r[lane], gsh_mine, gr_mine, o);
        // no-return atomicAdd (RED): groups / warps may add in any order
        atomicAdd(a.gvec + 3 * eidx, o[0]); atomicAdd(a.gvec + 3 * eidx + 1, o[1]); atomicAdd(a.gvec + 3 * eidx + 2, o[2]);
      }
    }
    __syncwarp();
  }
}


template <class T, class Cfg, int EPL, bool TABLE>
__global__ void __launch_bounds__(THREADS, 1) bwd_kernel(Args<T> a) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  using G = Groups<T, Cfg, EPL, true, TABLE>;
  using St = Stage<T, EPL, Cfg::SH_DIM, !TABLE>;
  const int warp = threadIdx.x >> 5;
  T* w1s = reinterpret_cast<T*>(smem_raw);
  T* w2s = w1s + (TABLE ? 0 : NB * (NH + 1));
  T* w3t = w2s + (TABLE ? 0 : NH * (NH + 1));
  auto* st = reinterpret_cast<St*>(w3t + G::W3T_ELEMS) + warp;
  // grad_x row of this warp: the dead MLP staging (h1T or h2T+h1T) or its own row after the stages
  T* sgx = reinterpret_cast<T*>(reinterpret_cast<St*>(w3t + G::W3T_ELEMS) + WARPS) + warp * G::SGX_ELEMS;
  if constexpr (G::SGX_ALIAS == 1) sgx = &st->h1T[0][0];
  else if constexpr (G::SGX_ALIAS == 2) sgx = &st->h2T[0][0];
  T cp_hi = T(0), cp_lo = T(0);
  if constexpr (!TABLE) bessel_cp(a, cp_hi, cp_lo);
  if constexpr (!TABLE) load_weights<T>(a, w1s, w2s);
  static_for<G::NG>([&](auto gg) {
    constexpr int U0 = decltype(gg)::value * G::UPG, U1 = cmin(G::NU, U0 + G::UPG);
    if constexpr (!TABLE) {
      __syncthreads();
      load_w3t<T, Cfg, U0, U1>(a, w3t);
      __syncthreads();
    }
    for (int node; (node = next_node(a.ctr + decltype(gg)::value)) < a.n_out;)
      bwd_node<T, Cfg, EPL, U0, U1, TABLE>(a, node, w1s, w2s, w3t, st, sgx, cp_hi, cp_lo);
  });
}

// Split backward, kernel 2: one warp per tile of EPL edges in storage order (grid-stride). Bessel, H1, H2
// recomputed once per edge, then mlp_bwd_tile on the grad_H2 summed over the path groups by bwd_kernel;
// grad_vec += grad_r * v / r (each edge written by exactly one lane: no atomics). No W3T tile, so the
// block needs only the MLP weights + staging and several blocks share an SM.
template <class T, class Cfg, int EPL>
constexpr int mlp_bwd_smem() { return (int)sizeof(T) * (NB * (NH + 1) + NH * (NH + 1)) + WARPS * (int)sizeof(Stage<T, EPL, Cfg::SH_DIM, true>); }
template <class T, class Cfg, int EPL>
__global__ void __launch_bounds__(THREADS) bwd_mlp_kernel(Args<T> a) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  using St = Stage<T, EPL, Cfg::SH_DIM, true>;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  T* w1s = reinterpret_cast<T*>(smem_raw);
  T* w2s = w1s + NB * (NH + 1);
  auto* st = reinterpret_cast<St*>(w2s + NH * (NH + 1)) + warp;
  T cp_hi, cp_lo;
  bessel_cp(a, cp_hi, cp_lo);
  load_weights<T>(a, w1s, w2s);
  __syncthreads();
  // the CSR's edges only, as bwd_kernel: rows of vec past rowptr[n_out] are not edges (LAMMPS GPU edges: the
  // capacity tail, vec 0 -> r 0 -> NaN grad_vec, which the force reduction then summed)
  const int n_edge = a.rowptr[a.n_out];
  for (int eb = (blockIdx.x * WARPS + warp) * EPL; eb < n_edge; eb += gridDim.x * WARPS * EPL) {
    const int ne = min(EPL, n_edge - eb);
    const EdgePre<T> pre = prefetch_edges<T, EPL>(a, eb, n_edge);
    stage_edges<T, Cfg, EPL, true>(a, st, pre, cp_hi, cp_lo);
    T z1[EPL][2], z2[EPL][2], g2[EPL][2];
    mlp_tile<T, EPL, Cfg::SH_DIM>(st, w1s, w2s, a.cst, z1, z2);
#pragma unroll
    for (int e = 0; e < EPL; ++e) {
      const T* g = a.gh2 + (size_t)__shfl_sync(0xffffffffu, pre.eidx, e) * NH + lane;
      g2[e][0] = e < ne ? g[0] : T(0); g2[e][1] = e < ne ? g[32] : T(0);
    }
    const T gr = mlp_bwd_tile<T, Cfg, EPL>(a, st, w1s, w2s, z1, z2, g2, cp_hi, cp_lo);
    if (lane < ne) {
      const T s = gr * rcp(st->r[lane]);
      T* gv = a.gvec + 3 * pre.eidx;
      gv[0] += pre.vx * s; gv[1] += pre.vy * s; gv[2] += pre.vz * s;
    }
    __syncwarp();   // stage reused by the next tile
  }
}


// Table backward epilogue: grad_vec of every CSR edge from its summed grad_sh / grad_r row (one thread per edge).
template <class T, int SHD>
__global__ void tbwd_vec_kernel(Args<T> a) {
  const int n_edge = a.rowptr[a.n_out];
  for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n_edge; k += gridDim.x * blockDim.x) {
    const int e = a.perm ? a.perm[k] : k;
    const T vx = a.vec[3 * e], vy = a.vec[3 * e + 1], vz = a.vec[3 * e + 2];
    const T* g = a.gshb + (size_t)e * (SHD + 1);
    T o[3];
    vec_grad<T, SHD>(vx, vy, vz, sqrt(vx * vx + vy * vy + vz * vz), g, g[SHD], o);
    a.gvec[3 * e] = o[0]; a.gvec[3 * e + 1] = o[1]; a.gvec[3 * e + 2] = o[2];   // table mode: the only writer
  }
}

// ---- host side ----------------------------------------------------------------------------
template <class T>
Args<T> make_args(const at::Tensor& x, const at::Tensor& vec, const at::Tensor& src, const at::Tensor& rowptr,
                  const at::Tensor& perm, const at::Tensor& w1, const at::Tensor& w2, const at::Tensor& w3,
                  const at::Tensor& coeffs, double denom, double r_on, double r_cut, double cst,
                  const std::optional<at::Tensor>& table, const std::optional<at::Tensor>& tabidx) {
  Args<T> a{};
  a.x = x.data_ptr<T>(); a.vec = vec.data_ptr<T>(); a.src = src.data_ptr<int>();
  a.rowptr = rowptr.data_ptr<int>(); a.perm = perm.data_ptr<int>();
  a.w1 = w1.data_ptr<T>(); a.w2 = w2.data_ptr<T>(); a.w3 = w3.data_ptr<T>(); a.coeffs = coeffs.data_ptr<T>();
  a.denom = T(denom); a.r_on = T(r_on); a.r_cut = T(r_cut); a.cst = T(cst);
  a.two_rc = T(2 / r_cut); a.rc2 = T(r_cut * r_cut); a.ron2 = T(r_on * r_on);
  a.inv_d3 = T(1 / std::pow(r_cut * r_cut - r_on * r_on, 3));
  a.n_out = (int)rowptr.numel() - 1; a.nw = (int)w3.size(1); a.n_edge = (int)vec.size(0);
  if (table.has_value()) {
    a.table = table->data_ptr<T>(); a.ntab = (int)table->size(table->dim() - 2);
    const double inv_h = (a.ntab - 1) / r_cut;
    a.h = T(r_cut / (a.ntab - 1)); a.inv_h = T(inv_h); a.inv_h_lo = T(inv_h - double(a.inv_h));
  }
  if (tabidx.has_value()) a.tabidx = tabidx->data_ptr<int>();
  return a;
}

// One block per SM as soon as there is a node for each (nodes are handed out per warp): with fewer than 8 nodes per SM
// the warps of ceil(n_out / 8) full blocks shared half of the SMs (512 atoms: 64 blocks on 132 SMs).
inline int grid_for(int n_out) {
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  return std::max(1, std::min(sms, n_out));
}

template <class T, class Cfg, int EPL, bool BWD, bool TABLE>
void launch_k(const Args<T>& a) {
  auto kern = BWD ? bwd_kernel<T, Cfg, EPL, TABLE> : fwd_kernel<T, Cfg, EPL, TABLE>;
  constexpr int smem = Groups<T, Cfg, EPL, BWD, TABLE>::SMEM;
  static_assert(Groups<T, Cfg, EPL, BWD, TABLE>::FITS && smem <= SMEM_MAX,
                "conv_fused: config not supported (no path group fits in shared memory)");
  const auto stream = at::cuda::getCurrentCUDAStream();
  static bool attr_set[C10_COMPILE_TIME_MAX_GPUS] = {};   // the opt-in is per device, once per instantiation
  if (!attr_set[stream.device_index()]) {
    const int optin = at::cuda::getCurrentDeviceProperties()->sharedMemPerBlockOptin;
    TORCH_CHECK(smem <= optin, "conv_fused: built for ", SMEM_MAX, " B of shared memory per block but this GPU allows ",
                optin, " B; rebuild jet on this GPU (or with JET_SMEM_MAX=", optin, ")");
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    if (const char* cv = std::getenv("JET_CARVEOUT"))   // 4.6b experiment: preferred smem carveout (%), table kernels
      if (TABLE) C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributePreferredSharedMemoryCarveout, std::atoi(cv)));
    attr_set[stream.device_index()] = true;
  }
  // one work counter per path group (a fill node inside a CUDA graph); freed on this stream after the kernel
  const at::Tensor ctr = at::zeros({Groups<T, Cfg, EPL, BWD, TABLE>::NG}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));
  Args<T> ad = a;
  ad.ctr = ctr.data_ptr<int>();
  kern<<<grid_for(a.n_out), THREADS, smem, stream>>>(ad);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Radial table mode is fp32 only (deploy path); fp64 (gradcheck) always takes the exact MLP. Edges per tile (1/2/4/8)
// from the group counts of tab_maxu (register model) at each tile size.
template <class Cfg, bool BWD> constexpr int epl_tab() {
  using G4 = Groups<float, Cfg, 4, BWD, true>; using G8 = Groups<float, Cfg, 8, BWD, true>;
  using G2 = Groups<float, Cfg, 2, BWD, true>; using G1 = Groups<float, Cfg, 1, BWD, true>;
  constexpr int n8 = G8::FITS ? G8::NG : 1 << 20, n4 = G4::FITS ? G4::NG : 1 << 20, n2 = G2::FITS ? G2::NG : 1 << 20;
  // fewest groups (each group re-stages every edge). Ties: the backward and any layer whose EPL-4 tile already fits
  // one group keep EPL 4 (else the larger tile); a multi-group forward takes the smaller tile. H200, fwd: EquFlashV2 L1
  // 2.0 -> 1.7 ms, Nequix L1 1.3 -> 1.0, MACE L1 1.1 -> 0.9 at the smaller tile, while the 4-5 path layers went
  // 0.1 -> 0.2-0.3 ms there.
  if constexpr (BWD || n4 == 1)
    return n8 < n4 && n8 <= n2 && n8 <= G1::NG ? 8 : n4 <= n2 && n4 <= G1::NG ? 4 : n2 <= G1::NG ? 2 : 1;
  else
    return G1::NG <= n2 && G1::NG <= n4 ? 1 : n2 <= n4 ? 2 : 4;
}
template <class T, class Cfg, int EPL>
void launch_mlp_bwd(const Args<T>& a) {
  auto kern = bwd_mlp_kernel<T, Cfg, EPL>;
  constexpr int smem = mlp_bwd_smem<T, Cfg, EPL>();
  const auto stream = at::cuda::getCurrentCUDAStream();
  static bool attr_set[C10_COMPILE_TIME_MAX_GPUS] = {};
  if (!attr_set[stream.device_index()]) {   // fp64 exceeds the 48 KB default
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    attr_set[stream.device_index()] = true;
  }
  const int tiles = (a.n_edge + WARPS * EPL - 1) / (WARPS * EPL);
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  kern<<<std::max(1, std::min(tiles, 8 * sms)), THREADS, smem, stream>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <class T, class Cfg, int EPL, bool BWD>
void launch(const Args<T>& a) {
  if constexpr (sizeof(T) == 4) {
    if (a.table != nullptr) {
      launch_k<T, Cfg, epl_tab<Cfg, BWD>(), BWD, true>(a);
      if constexpr (BWD) {
        const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
        tbwd_vec_kernel<T, Cfg::SH_DIM><<<std::max(1, std::min((a.n_edge + 255) / 256, 8 * sms)), 256, 0,
                                          at::cuda::getCurrentCUDAStream()>>>(a);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
      return;
    }
  }
  launch_k<T, Cfg, EPL, BWD, false>(a);
  if constexpr (BWD) {
    if (a.gh2 != nullptr) launch_mlp_bwd<T, Cfg, EPL>(a);
  }
}

}  // namespace cf
}  // namespace jet_mlip
