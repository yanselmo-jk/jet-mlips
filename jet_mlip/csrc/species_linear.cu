// species_linear: SevenNet's 'nequip' self-connection, e3nn FullyConnectedTensorProduct(x, one_hot(species)) with
//   irreps_in2 = S x 0e, which is one linear map per (species, irrep path):
//     out[n, oo + w*d + m] = sum_u W[s(n), wo + u*V + w] * x[n, io + u*d + m]     (e3nn mul_ir layout)
//   W (S, P) fp32 carries e3nn's path normalization (sevenn jet_helper.JetSelfConnectionIntro builds it), paths =
//   flat [io, oo, U, V, d] per path, each input and output irrep in at most one path; columns no path writes are 0.
//   Nodes are bucketed by species (counting sort), then one fp32 SIMT GEMM tile (64 rows x 128 or 32 x 256 (node, m)
//   columns, whole nodes of one species) per block. grad x is the same kernel with W transposed; no grad for W
//   (inference).
//   species None: one shared (1, P) weight, an e3nn o3.Linear (sevenn IrrepsLinear); bias (dim_out) is added to
//   the columns the paths write (sevenn's modal one-hot input of a linear folded into a bias).
// CUDA-graph capturable: no sync, allocations depend on N and S only. species must lie in [0, S).
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include <algorithm>
#include <cstdlib>
#include <tuple>

namespace jet_mlip {
namespace {

constexpr int kThreads = 256, kMaxPaths = 16, kSmallBlocksPerSM = 2;
constexpr int kSmem = 256 * (32 + 4) + 256;   // 4-byte words: Cs and the node ids of the 32 x 256 tile, the larger one

// one GEMM per path: out[:, oo + r*d + m] = sum_k W[wo + r*rs + k*ks] * in[:, io + k*d + m]. Its blocks are
// [b0, b0 + n4): 64-row tiles, nx4 node tiles per row tile; then one row of 32-row tiles for a last R % 64 <= 32 rows.
struct Path { int io, oo, K, R, d, wo, rs, ks, b0, nx4, n4; };
struct Paths { int np; Path p[kMaxPaths]; };
struct Args {
  const float* in; const float* W; const int* start; const int* perm; const float* bias; float* out;
  int ldi, ldo, P, S, n;
};

__global__ void count_kernel(const int64_t* __restrict__ sp, int n, int S, int* cnt) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && sp[i] >= 0 && sp[i] < S) atomicAdd(cnt + sp[i], 1);
}

// cnt[0, S) counts -> start[0, S] exclusive prefix (in place, cnt has S+1 entries), cursor[0, S) = start
__global__ void scan_kernel(int* cnt, int* cursor, int S) {
  int acc = 0;
  for (int s = 0; s < S; s++) {
    const int c = cnt[s];
    cnt[s] = cursor[s] = acc;
    acc += c;
  }
  cnt[S] = acc;
}

// order within a species is arbitrary (atomics); every node's result is computed the same way wherever it lands
__global__ void place_kernel(const int64_t* __restrict__ sp, int n, int S, int* cursor, int* perm) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && sp[i] >= 0 && sp[i] < S) perm[atomicAdd(cursor + sp[i], 1)] = i;
}

// The same counting sort in one block over a shared-memory histogram, for up to kBucketN nodes: one launch instead of
// four (zero fill, count, scan, place), which is what a small system pays for at every self-connection (H200, with a
// 1 x 1 GEMM: 44 -> 29 us at 512 nodes, 57 -> 50 at 16384; slower than the four from about 30000 nodes on).
constexpr int kBucketN = 1 << 14, kBucketS = 4096;
__global__ void bucket_kernel(const int64_t* __restrict__ sp, int n, int S, int* start, int* perm) {
  __shared__ int cur[kBucketS];
  const int tid = threadIdx.x;
  for (int s = tid; s < S; s += blockDim.x) cur[s] = 0;
  __syncthreads();
  for (int i = tid; i < n; i += blockDim.x)
    if (sp[i] >= 0 && sp[i] < S) atomicAdd(cur + sp[i], 1);
  __syncthreads();
  if (tid == 0) {
    int acc = 0;
    for (int s = 0; s < S; s++) {
      const int c = cur[s];
      start[s] = cur[s] = acc;
      acc += c;
    }
    start[S] = acc;
  }
  __syncthreads();
  for (int i = tid; i < n; i += blockDim.x)
    if (sp[i] >= 0 && sp[i] < S) perm[atomicAdd(cur + sp[i], 1)] = i;
}

// e / D for 0 <= e < 2^22, inv = 1.f / D: (e + 0.5) / D is at least 0.5 / D away from an integer and the fp32
// error is below e * 2^-23 / D
__device__ __forceinline__ int divf(int e, float inv) { return (int)(((float)e + 0.5f) * inv); }

// One tile: rows r0 + [0, BM), node tile x (BN / d nodes of one species, all their d columns), fp32 SIMT GEMM,
// BM x BN x BK = 64 x 128 x 16 (TR 4) or 32 x 256 x 8 (TR 2). Thread (tx, ty) owns rows TR tx + [0, TR) and columns
// TC ty + [0, TC) in registers (TR TC = 32).
// Global memory is read and written in its own order, 32 consecutive floats per warp: a node's (k, m) block of the
// input and its (r, m) block of the output are contiguous whatever d is. The result goes register tile -> Cs -> out
// (storing the register tile directly hit 32 sectors per warp instruction: 2.9x the time of the transposed GEMMs).
// Two (A, B) buffers: the next k-step is loaded while this one is multiplied, one __syncthreads per step.
// PIPE = the kernel of small problems (gemm_kernel<true>), where a block's latency is what counts.
template <int TR, bool PIPE>
__device__ __forceinline__ void tile(const Args& a, const Path& p, int r0, int x, float* sm, int* seg_s) {
  constexpr int TC = 32 / TR, BM = 16 * TR, BN = 16 * TC, BK = 4 * TR, SA = BM + 4, SB = BN + 4, SC = BM + 4;   // +4: bank spread
  constexpr int NBUF = BK * (SA + SB), EPA = BM * BK / kThreads, EPT = BN * BK / kThreads, TAB = 2 * NBUF, CEND = BN * SC;
  constexpr int NODE = TAB + EPT * kThreads > CEND ? TAB + EPT * kThreads : CEND;   // node ids: live through the epilogue
  static_assert(NODE + BN <= kSmem && EPT == 8, "species_linear: shared memory layout");
  const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4, lane = tid & 31, wp = tid >> 5, d = p.d, nq = BN / d;
  int t = x, s = 0, seg = 0, nn = a.n;   // node tile -> (species s, tile t within it)
  if (a.perm) {
    if (tid == 0) {
      for (; s < a.S; s++) {
        const int nt = (a.start[s + 1] - a.start[s] + nq - 1) / nq;
        if (t < nt) break;
        t -= nt;
      }
      seg_s[0] = s;
      seg_s[1] = t;
    }
    __syncthreads();
    s = seg_s[0];
    t = seg_s[1];
    if (s == a.S) return;   // block-uniform
    seg = a.start[s];
    nn = a.start[s + 1] - seg;
  }
  const int q0 = t * nq;
  if (q0 >= nn) return;
  const int nqv = min(nq, nn - q0);   // nodes of this tile
  if (a.perm)
    for (int q = tid; q < nqv; q += kThreads) sm[NODE + q] = __int_as_float(a.perm[seg + q0 + q]);
  auto node_of = [&](int q) { return a.perm ? __float_as_int(sm[NODE + q]) : q0 + q; };
  // B (input) loads: element e = (node q, k, m), m fastest (the input's own order), thread tid takes e = tid + 256 i:
  // packed B slot << 18 | q << 10 | offset in the node's k-step block; dead elements: the pad slot, offset 1023.
  // Kept in a shared-memory table, read as float so that the compiler cannot hoist the 8 entries of a thread into
  // registers (the kernel has to fit 64, 4 blocks per SM); PIPE has the registers and no table to wait for.
  const float inv_d = 1.f / d, inv_kd = 1.f / (BK * d);
  auto entry = [&](int i) {
    const int e = tid + kThreads * i, q = divf(e, inv_kd), w = e - q * (BK * d), kk = divf(w, inv_d);
    return e < nqv * BK * d ? (kk * (SB - d) + q * d + w) << 18 | q << 10 | w : (BK * SB - 1) << 18 | 1023;
  };
  if constexpr (!PIPE) {
#pragma unroll
    for (int i = 0; i < EPT; i++) sm[TAB + tid + kThreads * i] = __int_as_float(entry(i));
  }
  auto elem = [&](int i) { return PIPE ? entry(i) : __float_as_int(sm[TAB + tid + kThreads * i]); };
  if (!PIPE || a.perm) __syncthreads();
  // A (weight) loads, unit-stride index fastest: thread element i is (k, row) = (ak + i dk, ar + i dr)
  const bool a_kfast = p.ks == 1;
  const int ak = a_kfast ? tid % BK : tid / BM, ar = a_kfast ? tid / BK : tid % BM;
  const int dk = a_kfast ? 0 : kThreads / BM, dr = a_kfast ? kThreads / BK : 0;
  const float* Ws = a.W + (int64_t)s * a.P + p.wo + (r0 + ar) * p.rs + ak * p.ks;
  // A step's 12 (TR 4) / 9 global loads of a thread are all issued before its first store: in flight together.
  // PIPE: the stores wait until after the multiply, so the load latency hides behind the FMAs.
  float va[EPA], v[EPT];
  auto issue = [&](int k0) {
    const int kmax = p.K - k0;   // live k of this step
    const int wmax = min(kmax, BK) * d;
#pragma unroll
    for (int i = 0; i < EPA; i++)
      va[i] = (ak + i * dk < kmax && r0 + ar + i * dr < p.R) ? Ws[i * (dr * p.rs + dk * p.ks) + k0 * p.ks] : 0.f;
    const float* src = a.in + p.io + k0 * d;
#pragma unroll
    for (int i = 0; i < EPT; i++) {
      const int e = elem(i), w = e & 1023;
      v[i] = w < wmax ? src[(int64_t)node_of((e >> 10) & 255) * a.ldi + w] : 0.f;
    }
  };
  auto commit = [&](int buf) {
    float* As = sm + buf;
#pragma unroll
    for (int i = 0; i < EPA; i++) As[(ak + i * dk) * SA + ar + i * dr] = va[i];
#pragma unroll
    for (int i = 0; i < EPT; i++) As[BK * SA + (elem(i) >> 18)] = v[i];
  };
  float acc[TR][TC] = {};
  int cur = 0;   // offset of the buffer being multiplied
  issue(0);
  commit(cur);
  __syncthreads();
  for (int k0 = 0; k0 < p.K; k0 += BK) {
    const bool more = k0 + BK < p.K;
    if (more) {
      issue(k0 + BK);
      if constexpr (!PIPE) commit(NBUF - cur);
    }
    const float* As = sm + cur + TR * tx;
    const float* Bs = sm + cur + BK * SA + TC * ty;
#pragma unroll
    for (int kk = 0; kk < BK; kk++) {
      float av[TR], bv[TC];
      if constexpr (TR == 4) {
        const float4 f = *reinterpret_cast<const float4*>(As + kk * SA);
        av[0] = f.x; av[1] = f.y; av[2] = f.z; av[3] = f.w;
      } else {
        const float2 f = *reinterpret_cast<const float2*>(As + kk * SA);
        av[0] = f.x; av[1] = f.y;
      }
#pragma unroll
      for (int j = 0; j < TC; j += 4) {
        const float4 f = *reinterpret_cast<const float4*>(Bs + kk * SB + j);
        bv[j] = f.x; bv[j + 1] = f.y; bv[j + 2] = f.z; bv[j + 3] = f.w;
      }
#pragma unroll
      for (int i = 0; i < TR; i++)
#pragma unroll
        for (int j = 0; j < TC; j++) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
    }
    if constexpr (PIPE) {
      if (more) commit(NBUF - cur);
    }
    __syncthreads();
    cur = NBUF - cur;
  }
  const int rows = min(BM, p.R - r0), col = p.oo + r0 * d;
  if (PIPE && d == 1) {   // few blocks: the fewest instructions per thread, straight from the registers (d > 1: Cs wins)
#pragma unroll
    for (int j = 0; j < TC; j++) {
      const int c = TC * ty + j, q = divf(c, inv_d);
      if (q >= nqv) break;
      const int cm = col + c - q * d;
      float* o = a.out + (int64_t)node_of(q) * a.ldo + cm;
#pragma unroll
      for (int i = 0; i < TR; i++) {
        const int r = TR * tx + i;
        if (r < rows) o[r * d] = a.bias ? acc[i][j] + a.bias[cm + r * d] : acc[i][j];
      }
    }
    return;
  }
  // epilogue: register tile -> Cs[column][row] (buffers and table are dead), then out in its own order
  float* Cs = sm;
#pragma unroll
  for (int j = 0; j < TC; j++) {
    float* c = Cs + (TC * ty + j) * SC + TR * tx;
    if constexpr (TR == 4) *reinterpret_cast<float4*>(c) = make_float4(acc[0][j], acc[1][j], acc[2][j], acc[3][j]);
    else *reinterpret_cast<float2*>(c) = make_float2(acc[0][j], acc[1][j]);
  }
  __syncthreads();
  for (int q = wp; q < nqv; q += 8) {   // warp = one node, its rows * d floats (r, m) 32 at a time
    float* dst = a.out + (int64_t)node_of(q) * a.ldo + col;
    const float* csq = Cs + q * d * SC;
    for (int w = lane; w < rows * d; w += 32) {
      const int r = divf(w, inv_d);
      const float f = csq[(w - r * d) * SC + r];
      dst[w] = a.bias ? f + a.bias[col + w] : f;
    }
  }
}

// SMALL: 64-row tiles only, PIPE (few blocks: the kernel takes as long as one block). Else 4 blocks per SM.
template <bool SMALL>
__global__ void __launch_bounds__(kThreads, SMALL ? 2 : 4) gemm_kernel(Args a, Paths ps) {
  __shared__ __align__(16) float sm[kSmem];
  __shared__ int seg_s[2];
  int pi = 0;
  while (pi + 1 < ps.np && (int)blockIdx.x >= ps.p[pi + 1].b0) pi++;
  const Path p = ps.p[pi];
  const int l = blockIdx.x - p.b0;
  if constexpr (SMALL) {
    tile<4, true>(a, p, (l / p.nx4) * 64, l % p.nx4, sm, seg_s);
  } else {
    if (l < p.n4) tile<4, false>(a, p, (l / p.nx4) * 64, l % p.nx4, sm, seg_s);
    else tile<2, false>(a, p, (p.n4 / p.nx4) * 64, l - p.n4, sm, seg_s);
  }
}

// paths flat [io, oo, U, V, d]*; transpose = grad x (in = grad out, out = grad x)
at::Tensor run(const at::Tensor& in, const at::Tensor& perm, const at::Tensor& start, const at::Tensor& W,
               const std::vector<int64_t>& paths, int64_t dim_out, bool transpose, const at::Tensor& bias = {}) {
  const int np = (int)paths.size() / 5, S = (int)W.size(0);
  const int64_t n = in.size(0);
  Paths ps{};
  ps.np = np;
  int64_t wo = 0, covered = 0;
  // node tiles: every species segment rounds up, at most min(S, n) segments are not empty
  const int64_t extra = perm.defined() ? std::min<int64_t>(S, n) : 0;
  for (int i = 0; i < np; i++) {
    const int64_t* q = &paths[5 * i];
    const int io = q[0], oo = q[1], U = q[2], V = q[3], d = q[4];
    TORCH_CHECK(d >= 1 && d <= 32, "species_linear: path ", i, " has irrep dimension ", d);
    Path& p = ps.p[i];
    p = transpose ? Path{oo, io, V, U, d, (int)wo, V, 1} : Path{io, oo, U, V, d, (int)wo, 1, V};
    wo += (int64_t)U * V;
    covered += (int64_t)p.R * d;
    TORCH_CHECK(p.io + (int64_t)p.K * d <= in.size(1) && p.oo + (int64_t)p.R * d <= dim_out,
                "species_linear: path ", i, " out of range");
    p.nx4 = (int)((n + 128 / d - 1) / (128 / d) + extra);
  }
  TORCH_CHECK(wo == W.size(1), "species_linear: weight has ", W.size(1), " columns, paths need ", wo);
  auto out = covered == dim_out ? at::empty({n, dim_out}, in.options()) : at::zeros({n, dim_out}, in.options());
  if (n == 0 || np == 0) return out;
  // block layout. small: 64-row tiles only; else a last R % 64 <= 32 rows take 32-row tiles
  auto layout = [&](bool small) {
    int64_t nblk = 0;
    for (int i = 0; i < np; i++) {
      Path& p = ps.p[i];
      const int64_t rem = p.R % 64, nq2 = 256 / p.d;
      p.b0 = (int)nblk;
      p.n4 = (p.R / 64 + (small ? rem > 0 : rem > 32)) * p.nx4;
      nblk += p.n4 + (!small && rem > 0 && rem <= 32 ? (n + nq2 - 1) / nq2 + extra : 0);
      TORCH_CHECK(nblk < INT32_MAX, "species_linear: too many tiles");
    }
    return nblk;
  };
  // few blocks: every block runs at once and the kernel takes as long as one block (its k-steps)
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  int64_t nblk = layout(true);
  bool small = nblk <= kSmallBlocksPerSM * sms;
  if (const char* e = std::getenv("JET_GEMM_SMALL")) small = e[0] == '1';   // tests: force either kernel
  if (!small) nblk = layout(false);
  const Args a{in.data_ptr<float>(), W.data_ptr<float>(), perm.defined() ? start.data_ptr<int>() : nullptr,
               perm.defined() ? perm.data_ptr<int>() : nullptr, bias.defined() ? bias.data_ptr<float>() : nullptr,
               out.data_ptr<float>(), (int)in.size(1), (int)dim_out, (int)W.size(1), S, (int)n};
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (small) gemm_kernel<true><<<(unsigned)nblk, kThreads, 0, stream>>>(a, ps);
  else gemm_kernel<false><<<(unsigned)nblk, kThreads, 0, stream>>>(a, ps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// -> (perm int32 (N): node ids grouped by species, start int32 (S+1): species segment offsets); undefined without species
std::tuple<at::Tensor, at::Tensor> bucket(const std::optional<at::Tensor>& sp, int64_t S, int64_t n) {
  if (!sp.has_value()) return {at::Tensor(), at::Tensor()};
  const auto species = sp->to(at::kLong).contiguous();
  const auto opt = species.options().dtype(at::kInt);
  auto perm = at::empty({n}, opt);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (n > 0 && n <= kBucketN && S <= kBucketS) {
    auto start = at::empty({S + 1}, opt);
    bucket_kernel<<<1, 1024, 0, stream>>>(species.data_ptr<int64_t>(), (int)n, (int)S, start.data_ptr<int>(),
                                          perm.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {perm, start};
  }
  auto buf = at::zeros({2 * S + 1}, opt);
  if (n > 0) {
    const int blocks = (int)((n + 255) / 256);
    int* cnt = buf.data_ptr<int>();
    count_kernel<<<blocks, 256, 0, stream>>>(species.data_ptr<int64_t>(), (int)n, (int)S, cnt);
    scan_kernel<<<1, 1, 0, stream>>>(cnt, cnt + S + 1, (int)S);
    place_kernel<<<blocks, 256, 0, stream>>>(species.data_ptr<int64_t>(), (int)n, (int)S, cnt + S + 1,
                                              perm.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {perm, buf.slice(0, 0, S + 1)};
}

at::Tensor checked_bias(const at::Tensor& x, const std::optional<at::Tensor>& sp, const at::Tensor& W,
                        const std::vector<int64_t>& paths, int64_t dim_out, const std::optional<at::Tensor>& bias) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat && x.dim() == 2, "species_linear: x (N, D) float32 CUDA");
  TORCH_CHECK(!sp.has_value() || (sp->is_cuda() && sp->dim() == 1 && sp->size(0) == x.size(0)), "species: (N,) CUDA");
  TORCH_CHECK(sp.has_value() || W.size(0) == 1, "species_linear: no species needs a (1, P) weight");
  TORCH_CHECK(W.is_cuda() && W.scalar_type() == at::kFloat && W.dim() == 2 && W.is_contiguous(),
              "weight: (S, P) float32 contiguous CUDA");
  TORCH_CHECK(paths.size() % 5 == 0 && paths.size() / 5 <= kMaxPaths, "paths: [io, oo, U, V, d] per path, <= 16");
  if (!bias.has_value()) return {};
  TORCH_CHECK(bias->is_cuda() && bias->scalar_type() == at::kFloat && bias->numel() == dim_out, "bias: (dim_out,) float32");
  return bias->contiguous();
}

at::Tensor species_linear_cuda(const at::Tensor& x, const std::optional<at::Tensor>& species, const at::Tensor& W,
                               std::vector<int64_t> paths, int64_t dim_out, const std::optional<at::Tensor>& bias) {
  const auto b = checked_bias(x, species, W, paths, dim_out, bias);
  auto [perm, start] = bucket(species, W.size(0), x.size(0));
  return run(x.contiguous(), perm, start, W, paths, dim_out, false, b);
}

struct SpeciesLinearFn : public torch::autograd::Function<SpeciesLinearFn> {
  static at::Tensor forward(torch::autograd::AutogradContext* ctx, const at::Tensor& x,
                            const std::optional<at::Tensor>& species, const at::Tensor& W, std::vector<int64_t> paths,
                            int64_t dim_out, const std::optional<at::Tensor>& bias) {
    const auto b = checked_bias(x, species, W, paths, dim_out, bias);
    at::AutoDispatchBelowADInplaceOrView guard;
    auto [perm, start] = bucket(species, W.size(0), x.size(0));
    ctx->save_for_backward({perm, start, W});
    ctx->saved_data["paths"] = paths;
    ctx->saved_data["din"] = x.size(1);
    ctx->saved_data["wgrad"] = W.requires_grad() || (b.defined() && b.requires_grad());
    return run(x.contiguous(), perm, start, W, paths, dim_out, false, b);
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::variable_list gouts) {
    TORCH_CHECK(!ctx->saved_data["wgrad"].toBool(),   // (needs_input_grad indices shift with a None species)
                "species_linear: no gradient for the weight / bias (inference only; train mode keeps e3nn)");
    const auto s = ctx->get_saved_variables();
    at::Tensor gx;
    if (gouts[0].defined() && ctx->needs_input_grad(0)) {
      at::NoGradGuard no_grad;   // no double backward
      gx = run(gouts[0].contiguous(), s[0], s[1], s[2], ctx->saved_data["paths"].toIntVector(),
               ctx->saved_data["din"].toInt(), true);
    }
    return {gx, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

at::Tensor species_linear_autograd(const at::Tensor& x, const std::optional<at::Tensor>& species, const at::Tensor& W,
                                   std::vector<int64_t> paths, int64_t dim_out, const std::optional<at::Tensor>& bias) {
  return SpeciesLinearFn::apply(x, species, W, paths, dim_out, bias);
}

}  // namespace
}  // namespace jet_mlip

TORCH_LIBRARY_FRAGMENT(jet_mlip, m) {
  m.def("species_linear(Tensor x, Tensor? species, Tensor weight, int[] paths, int dim_out, Tensor? bias=None) -> Tensor");
}
TORCH_LIBRARY_IMPL(jet_mlip, CUDA, m) {
  m.impl("species_linear", &jet_mlip::species_linear_cuda);
}
TORCH_LIBRARY_IMPL(jet_mlip, Autograd, m) {
  m.impl("species_linear", &jet_mlip::species_linear_autograd);
}
