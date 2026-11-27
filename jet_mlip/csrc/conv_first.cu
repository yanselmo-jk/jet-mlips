// conv_first: conv_fused for a scalar-only input of at most 32 channels (x = mul x 0e: the type embedding a first
// layer reads), radial table mode, fp32. Every path is 0e x Y_l -> l and wigner_3j(0, l, l) = coef * delta(m2, m3), so
//   out[dst, p, u, m] = coef_p / denom * sum_edges x[src, u] * W_p,u(r) * Y_p,m
// with no mixing over m. Warp = dst node, lane = channel u, as conv_fused.cuh, but a lane keeps x * W (one value per
// path and edge) and one accumulator row: few enough registers for KB blocks per SM where the generic kernel is
// built for one, and these kernels are latency bound (the blocks per SM are most of the gain: the generic kernel
// forced to 3 blocks per SM runs the lmax-2 shape at 0.11 / 0.21 ms). Same CSR / perm / table / grad buffers as the
// generic kernel; conv_fused.cu routes here (JET_FIRST_LAYER=0: generic kernel).
// H200, electrolyte 10,040 atoms, 6 A (803k edges), ms generic -> here, fwd / bwd: 32x0e lmax 1 0.21 -> 0.13 / 0.40 ->
// 0.21, lmax 2 (5 A, 452k) 0.15 -> 0.11 / 0.31 -> 0.19, lmax 3 0.30 -> 0.24 / 0.59 -> 0.38, lmax 4 0.37 -> 0.28 /
// 0.77 -> 0.59. Wider inputs stay generic: as (node, 32-channel chunk) items this kernel only ties there (128x0e
// lmax 3: 0.86 -> 0.84 / 1.23 -> 1.26; 96x0e 0.65 -> 0.63 / 1.00 -> 0.97), the forward following the table read
// (16 B per edge and column; 0.52 ms with a 0.1 MiB table) and the backward paying one grad_sh reduction per chunk;
// a backward with the chunk loop inside the edge loop (upstream grads in shared memory) gave 1.23 -> 1.20, 128x0e
// lmax 2 0.89 -> 0.85, 96x0e 0.99 -> 0.92: not kept.
#include "conv_fused.cuh"

namespace jet_mlip {
namespace cf {

constexpr int EPL1 = 8;   // edges per tile: 8 beat 4 and 2 for every shape and KB measured
// blocks per SM = register budget 65536 / (KB * THREADS) per lane; H200, fwd / bwd ms at KB 2, 3, 4:
// lmax 1 0.16 0.14 0.13 / 0.27 0.23 0.22; lmax 2 0.12 0.11 0.11 / 0.21 0.20 0.19; lmax 3 0.23 0.24 0.24 / 0.41 0.39 0.40;
// lmax 4 0.28 0.34 0.39 / 0.59 0.63 0.68
template <class Cfg> constexpr int KB1 = Cfg::SH_DIM <= 4 ? 4 : Cfg::SH_DIM <= 16 ? 3 : 2;

template <class Cfg> constexpr double path_coef(int p) {
  for (int k = 0; k < Cfg::NFIBER; ++k) if (Cfg::fibers[k].p == p) return Cfg::fibers[k].coef;
  return 0;
}
// one 0e x block of <= 32 channels read by every path, l2 == l3, diagonal fibers with one coefficient per path, no
// pruned output block
template <class Cfg> constexpr bool first_cfg() {
  if (nxb<Cfg>() != 1 || xb_l<Cfg>(0) != 0 || Cfg::IN_DIM != Cfg::paths[0].mul || Cfg::IN_DIM > 32) return false;
  int live = 0;
  for (int p = 0; p < Cfg::NPATH; ++p) {
    if (Cfg::paths[p].l2 != Cfg::paths[p].l3) return false;
    live += Cfg::paths[p].mul * (2 * Cfg::paths[p].l3 + 1);
  }
  for (int k = 0; k < Cfg::NFIBER; ++k) {
    const auto f = Cfg::fibers[k];
    const double d = f.coef - path_coef<Cfg>(f.p);   // e3nn: one coefficient per path up to the last bit
    if (f.m1 != 0 || f.m2 != f.m3 || d > 1e-12 || d < -1e-12) return false;
  }
  return live == Cfg::OUT_DIM && Cfg::NFIBER == oslot<Cfg>(Cfg::NPATH);
}

// cubic Hermite position of one edge in the radial table (w_tile_table's formulas)
struct TabPos { const float* row; float b1, b2, b3, c1, c2, c3; };
__device__ __forceinline__ TabPos tab_pos(const Args<float>& a, float r, int ti) {
  const float xg = fmaf(r, a.inv_h, r * a.inv_h_lo);
  int k = (int)xg;
  k = k < 0 ? 0 : (k > a.ntab - 2 ? a.ntab - 2 : k);
  float t = xg - float(k);
  t = t < 0.f ? 0.f : (t > 1.f ? 1.f : t);
  const float t2 = t * t, t3 = t2 * t;
  return {a.table + ((size_t)ti * a.ntab + k) * (4 * a.nw),
          (t3 - 2.f * t2 + t) * a.h, (3.f * t2 - 2.f * t3) * a.h, (t3 - t2) * a.h,
          3.f * t2 - 4.f * t + 1.f, 6.f * t - 6.f * t2, 3.f * t2 - 2.f * t};
}
// W (and dW/dr) of table column col: one 16 B load {v0, m0, d, m1}
__device__ __forceinline__ float tab_w(const TabPos& tp, int col, float* dW) {
  const float4 q = __ldg(reinterpret_cast<const float4*>(tp.row + 4 * col));
  if (dW != nullptr) *dW = tp.c1 * q.y + tp.c2 * q.z + tp.c3 * q.w;
  return q.x + (tp.b1 * q.y + tp.b2 * q.z + tp.b3 * q.w);
}

template <class Cfg>
__global__ void __launch_bounds__(THREADS, KB1<Cfg>) first_fwd_kernel(Args<float> a) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  constexpr int EPL = EPL1, NACC = oslot<Cfg>(Cfg::NPATH);
  auto* st = reinterpret_cast<Stage<float, EPL, Cfg::SH_DIM, false>*>(smem_raw) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const bool live = lane < Cfg::IN_DIM;   // mul 8 / 16: idle lanes
  for (int node; (node = next_node(a.ctr)) < a.n_out;) {
    const int e0 = a.rowptr[node], e1 = a.rowptr[node + 1];
    float acc[NACC];
#pragma unroll
    for (int i = 0; i < NACC; ++i) acc[i] = 0.f;
    EdgePre<float> pre = prefetch_edges<float, EPL>(a, e0, e1);
    for (int eb = e0; eb < e1; eb += EPL) {
      const int ne = min(EPL, e1 - eb);
      stage_edges<float, Cfg, EPL, false>(a, st, pre, 0.f, 0.f);
      pre = prefetch_edges<float, EPL>(a, eb + EPL, e1);   // next tile's loads in flight
#pragma unroll
      for (int e = 0; e < EPL; ++e) {
        if (e >= ne) break;
        const float x = live ? a.x[(size_t)st->src[e] * Cfg::IN_DIM + lane] : 0.f;
        const TabPos tp = tab_pos(a, st->r[e], st->ti[e]);
        static_for<Cfg::NPATH>([&](auto pp) {
          constexpr auto P = Cfg::paths[decltype(pp)::value];
          constexpr int o = oslot<Cfg>(decltype(pp)::value);
          const float xw = x * tab_w(tp, min(P.wcol + lane, a.nw - 1), nullptr);
#pragma unroll
          for (int m = 0; m < 2 * P.l3 + 1; ++m) acc[o + m] += xw * st->sh[e][P.off2 + m];
        });
      }
      __syncwarp();   // stage reused by the next tile
    }
    if (live) {
      float* orow = a.out + (size_t)node * Cfg::OUT_DIM;
      static_for<Cfg::NPATH>([&](auto pp) {
        constexpr auto P = Cfg::paths[decltype(pp)::value];
        constexpr int o = oslot<Cfg>(decltype(pp)::value), D = 2 * P.l3 + 1;
        const float s = float(path_coef<Cfg>(decltype(pp)::value)) / a.denom;
#pragma unroll
        for (int m = 0; m < D; ++m) orow[P.off3 + lane * D + m] = acc[o + m] * s;
      });
    }
  }
}

// grad_x straight from the lanes (one value per channel: contiguous atomics); grad_sh / grad_r into the edge's row of
// a.gshb (this warp is its only writer), which tbwd_vec_kernel turns into grad_vec as for the generic table backward
template <class Cfg>
__global__ void __launch_bounds__(THREADS, KB1<Cfg>) first_bwd_kernel(Args<float> a) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  constexpr int EPL = EPL1, NACC = oslot<Cfg>(Cfg::NPATH), SHD = Cfg::SH_DIM;
  constexpr int NRED = cpow2(SHD), RSH = 5 - clog2(NRED), NRG = cpow2(EPL), RSHG = 5 - clog2(NRG);
  auto* st = reinterpret_cast<Stage<float, EPL, SHD, false>*>(smem_raw) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const bool live = lane < Cfg::IN_DIM;
  for (int node; (node = next_node(a.ctr)) < a.n_out;) {
    const int e0 = a.rowptr[node], e1 = a.rowptr[node + 1];
    if (e0 == e1) continue;
    float g[NACC];   // upstream grad of this lane's channel, coef / denom folded in
    const float* grow = a.gout + (size_t)node * Cfg::OUT_DIM;
    static_for<Cfg::NPATH>([&](auto pp) {
      constexpr auto P = Cfg::paths[decltype(pp)::value];
      constexpr int o = oslot<Cfg>(decltype(pp)::value), D = 2 * P.l3 + 1;
      const float s = float(path_coef<Cfg>(decltype(pp)::value)) / a.denom;
#pragma unroll
      for (int m = 0; m < D; ++m) g[o + m] = live ? grow[P.off3 + lane * D + m] * s : 0.f;
    });
    EdgePre<float> pre = prefetch_edges<float, EPL>(a, e0, e1);
    for (int eb = e0; eb < e1; eb += EPL) {
      const int ne = min(EPL, e1 - eb);
      stage_edges<float, Cfg, EPL, false>(a, st, pre, 0.f, 0.f);
      const int eidx = pre.eidx;
      pre = prefetch_edges<float, EPL>(a, eb + EPL, e1);
      float gr[NRG];
#pragma unroll
      for (int i = 0; i < NRG; ++i) gr[i] = 0.f;
#pragma unroll
      for (int e = 0; e < EPL; ++e) {
        if (e >= ne) break;
        const int s = st->src[e];
        const float x = live ? a.x[(size_t)s * Cfg::IN_DIM + lane] : 0.f;
        const TabPos tp = tab_pos(a, st->r[e], st->ti[e]);
        float gsh[NRED], gxv = 0.f, gre = 0.f;
#pragma unroll
        for (int m = 0; m < NRED; ++m) gsh[m] = 0.f;
        static_for<Cfg::NPATH>([&](auto pp) {
          constexpr auto P = Cfg::paths[decltype(pp)::value];
          constexpr int o = oslot<Cfg>(decltype(pp)::value);
          float dW;
          const float W = tab_w(tp, min(P.wcol + lane, a.nw - 1), &dW), xw = x * W;
          float q = 0.f;   // <grad out, Y> of this path
#pragma unroll
          for (int m = 0; m < 2 * P.l3 + 1; ++m) {
            q += g[o + m] * st->sh[e][P.off2 + m];
            gsh[P.off2 + m] += xw * g[o + m];
          }
          gxv += W * q;
          gre += dW * q;
        });
        gr[e] = x * gre;
        if (a.gx != nullptr && live) atomicAdd(a.gx + (size_t)s * Cfg::IN_DIM + lane, gxv);
        // grad_sh summed over the channels: lane m << RSH holds component m
        const float red = warp_reduce_t<float, NRED, 16>(gsh);
        const int ei = __shfl_sync(0xffffffffu, eidx, e);
        if ((lane & ((1 << RSH) - 1)) == 0 && (lane >> RSH) < SHD) a.gshb[(size_t)ei * (SHD + 1) + (lane >> RSH)] = red;
      }
      const float red = warp_reduce_t<float, NRG, 16>(gr);   // grad_r of edge e to lane e
      const float gr_mine = __shfl_sync(0xffffffffu, red, (lane & (NRG - 1)) << RSHG);
      if (lane < ne) a.gshb[(size_t)eidx * (SHD + 1) + SHD] = gr_mine;
      __syncwarp();
    }
  }
}

template <class Cfg> bool launch_first(const Args<float>& a, bool bwd) {
  if constexpr (first_cfg<Cfg>()) {
    const auto stream = at::cuda::getCurrentCUDAStream();
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    // Small systems stay generic: up to about its WARPS * sms warps every node has a warp of its own, and the generic
    // kernel's single stream is the faster one (H200, Si 5 A, lmax 2, fwd / bwd ms generic vs here: 512 and 1000
    // atoms 0.025 / 0.041 vs 0.029 / 0.053, 1728 atoms 0.034 / 0.059 vs 0.032 / 0.056, 2744 atoms 0.043 / 0.075 vs
    // 0.033 / 0.057; 7net-nano Si 512 in LAMMPS ran 4-5 % slower through this kernel)
    if (!many_nodes(a.n_out)) return false;
    const at::Tensor ctr = at::zeros({1}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));   // as launch_k
    Args<float> ad = a;
    ad.ctr = ctr.data_ptr<int>();
    constexpr int smem = WARPS * (int)sizeof(Stage<float, EPL1, Cfg::SH_DIM, false>);
    auto kern = bwd ? first_bwd_kernel<Cfg> : first_fwd_kernel<Cfg>;
    kern<<<std::max(1, std::min(KB1<Cfg> * sms, a.n_out)), THREADS, smem, stream>>>(ad);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (bwd) {
      tbwd_vec_kernel<float, Cfg::SH_DIM><<<std::max(1, std::min((a.n_edge + 255) / 256, 8 * sms)), 256, 0, stream>>>(a);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return true;
  } else {
    return false;
  }
}

bool first_layer(int64_t cfg_id, const Args<float>& a, bool bwd) {
  if (a.table == nullptr) return false;
  const char* e = std::getenv("JET_FIRST_LAYER");   // read per launch: a CUDA graph keeps what its capture saw
  if (e != nullptr && e[0] == '0') return false;
  switch (cfg_id) {
#define JET_FIRST(ID, CFG) case ID: return launch_first<CFG>(a, bwd);
    JET_FOREACH_CONFIG(JET_FIRST)
#undef JET_FIRST
    default: return false;
  }
}

}  // namespace cf
}  // namespace jet_mlip
