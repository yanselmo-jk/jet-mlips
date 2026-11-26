// gate: e3nn nn.Gate (sevenn EquivariantGate) as one elementwise kernel per direction, driven by per-column tables
//   (ops.gate_tables builds them from the module, e3nn's normalize2mom constants and the Extract order folded in).
//   forward, out column c, fwd_i[c] = (src, gcol, act, -), fwd_f[c] = (cst, coef):
//     gcol < 0: out = cst * f(x[src])                     (scalar)
//     else:     out = coef * x[src] * cst * f(x[gcol])    (gated irrep component times its activated gate)
//   backward, input column j, bwd_i[j] = (role, o, aux, act, d), bwd_f[j] = (cst, coef):
//     role 0: unused (0); 1 scalar: g[o] cst f'(x[j]); 2 gated: g[o] coef cst f(x[aux]);
//     3 gate: cst f'(x[j]) coef sum_{m<d} g[o+m] x[aux+m]
//   act 0 identity, 1 silu, 2 tanh, 3 sigmoid. First-order autograd in x only. CUDA-graph capturable.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include <algorithm>

namespace jet_mlip {
namespace {

__device__ __forceinline__ float act_f(int a, float z) {
  switch (a) {
    case 1: return z / (1.f + expf(-z));
    case 2: return tanhf(z);
    case 3: return 1.f / (1.f + expf(-z));
    default: return z;
  }
}

__device__ __forceinline__ float act_df(int a, float z) {
  switch (a) {
    case 1: { const float s = 1.f / (1.f + expf(-z)); return s * (1.f + z * (1.f - s)); }
    case 2: { const float t = tanhf(z); return 1.f - t * t; }
    case 3: { const float s = 1.f / (1.f + expf(-z)); return s * (1.f - s); }
    default: return 1.f;
  }
}

// A block takes n / gridDim.x consecutive rows, a thread the columns threadIdx.x + k 256 of them, the column's table
// entry in registers: a warp reads and writes 32 consecutive floats of a row and no index is divided (one thread per
// element with i / dout and i % dout in 64 bits: 0.207 -> 0.157 ms, 7net-omni's gate, 32768 rows, H200). The backward
// stays one thread per element: the same loop nest was 11 % slower there (0.339 -> 0.378 ms).
__global__ void gate_fwd(const float* __restrict__ x, int din, const int4* __restrict__ ti, const float2* __restrict__ tf,
                         float* __restrict__ out, int dout, int n) {
  const int per = (n + gridDim.x - 1) / gridDim.x, r0 = blockIdx.x * per, r1 = min(n, r0 + per);
  for (int c = threadIdx.x; c < dout; c += blockDim.x) {
    const int4 t = ti[c];
    const float2 f = tf[c];
    for (int r = r0; r < r1; r++) {
      const float* xr = x + (int64_t)r * din;
      out[(int64_t)r * dout + c] = t.y < 0 ? f.x * act_f(t.z, xr[t.x]) : f.y * xr[t.x] * f.x * act_f(t.z, xr[t.y]);
    }
  }
}

__global__ void gate_bwd(const float* __restrict__ g, int dout, const float* __restrict__ x, int din,
                         const int* __restrict__ ti, const float2* __restrict__ tf, float* __restrict__ gx, int64_t total) {
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < total; i += (int64_t)gridDim.x * blockDim.x) {
    const int64_t n = i / din;
    const int j = (int)(i - n * din);
    const int* t = ti + 5 * j;
    const float2 f = tf[j];
    const float* xr = x + n * din;
    const float* gr = g + n * dout;
    float v = 0.f;
    switch (t[0]) {
      case 1: v = gr[t[1]] * f.x * act_df(t[3], xr[j]); break;
      case 2: v = gr[t[1]] * f.y * f.x * act_f(t[3], xr[t[2]]); break;
      case 3: {
        float s = 0.f;
        for (int m = 0; m < t[4]; m++) s += gr[t[1] + m] * xr[t[2] + m];
        v = f.x * act_df(t[3], xr[j]) * f.y * s;
        break;
      }
      default: break;
    }
    gx[i] = v;
  }
}

int blocks(int64_t total) { return (int)std::min<int64_t>((total + 255) / 256, 65536); }

int grid_of(int64_t n) {   // forward: about 8 blocks per SM
  return (int)std::min<int64_t>(n, 8 * at::cuda::getCurrentDeviceProperties()->multiProcessorCount);
}

at::Tensor gate_cuda(const at::Tensor& x_, const at::Tensor& fi, const at::Tensor& ff, const at::Tensor& bi,
                     const at::Tensor& bf) {
  TORCH_CHECK(x_.is_cuda() && x_.scalar_type() == at::kFloat && x_.dim() == 2, "gate: x (N, D) float32 CUDA");
  TORCH_CHECK(fi.scalar_type() == at::kInt && fi.dim() == 2 && fi.size(1) == 4 && fi.is_contiguous() &&
              ff.scalar_type() == at::kFloat && ff.sizes() == at::IntArrayRef({fi.size(0), 2}) && ff.is_contiguous(),
              "gate: fwd tables (Dout, 4) int32, (Dout, 2) float32");
  TORCH_CHECK(bi.size(0) == x_.size(1), "gate: bwd tables have one row per input column");
  const auto x = x_.contiguous();
  const int64_t n = x.size(0), dout = fi.size(0), total = n * dout;
  auto out = at::empty({n, dout}, x.options());
  if (total == 0) return out;
  gate_fwd<<<grid_of(n), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), (int)x.size(1), reinterpret_cast<const int4*>(fi.data_ptr<int>()),
      reinterpret_cast<const float2*>(ff.data_ptr<float>()), out.data_ptr<float>(), (int)dout, (int)n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

struct GateFn : public torch::autograd::Function<GateFn> {
  static at::Tensor forward(torch::autograd::AutogradContext* ctx, const at::Tensor& x, const at::Tensor& fi,
                            const at::Tensor& ff, const at::Tensor& bi, const at::Tensor& bf) {
    at::AutoDispatchBelowADInplaceOrView guard;
    auto out = gate_cuda(x, fi, ff, bi, bf);
    ctx->save_for_backward({x.contiguous(), bi, bf});
    return out;
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::variable_list gouts) {
    const auto s = ctx->get_saved_variables();
    const auto &x = s[0], &bi = s[1], &bf = s[2];
    at::Tensor gx;
    if (gouts[0].defined()) {
      at::NoGradGuard no_grad;   // no double backward
      const auto g = gouts[0].contiguous();
      gx = at::empty_like(x);
      const int64_t total = x.numel();
      if (total > 0) {
        gate_bwd<<<blocks(total), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            g.data_ptr<float>(), (int)g.size(1), x.data_ptr<float>(), (int)x.size(1), bi.data_ptr<int>(),
            reinterpret_cast<const float2*>(bf.data_ptr<float>()), gx.data_ptr<float>(), total);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    }
    return {gx, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

at::Tensor gate_autograd(const at::Tensor& x, const at::Tensor& fi, const at::Tensor& ff, const at::Tensor& bi,
                         const at::Tensor& bf) {
  TORCH_CHECK(bi.scalar_type() == at::kInt && bi.dim() == 2 && bi.size(1) == 5 && bi.is_contiguous() &&
              bf.scalar_type() == at::kFloat && bf.sizes() == at::IntArrayRef({bi.size(0), 2}) && bf.is_contiguous(),
              "gate: bwd tables (Din, 5) int32, (Din, 2) float32");
  return GateFn::apply(x, fi, ff, bi, bf);
}

}  // namespace
}  // namespace jet_mlip

TORCH_LIBRARY_FRAGMENT(jet_mlip, m) {
  m.def("gate(Tensor x, Tensor fwd_i, Tensor fwd_f, Tensor bwd_i, Tensor bwd_f) -> Tensor");
}
TORCH_LIBRARY_IMPL(jet_mlip, CUDA, m) {
  m.impl("gate", &jet_mlip::gate_cuda);
}
TORCH_LIBRARY_IMPL(jet_mlip, Autograd, m) {
  m.impl("gate", &jet_mlip::gate_autograd);
}
