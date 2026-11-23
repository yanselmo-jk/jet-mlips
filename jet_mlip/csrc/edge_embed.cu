// Fused per-edge elementwise chains of the SevenNet training path (any-order autograd).
//
// edge_embed: sevenn EdgeEmbedding in one kernel (thread per edge):
//   B[e, i]  = (2/rc) sin(c_i r)/r * xplor(r)         r = |vec|, c = bessel coeffs (trainable)
//   sh[e, :] = e3nn spherical_harmonics(lmax <= 3, normalize=True, normalization='component')
// Autograd to any order: the backward evaluates the same math as plain ATen ops
// (edge_embed_ref) and differentiates it with torch::autograd::grad, so the first order is
// ~30 ATen kernels and every higher order is autograd through that graph. Only the forward
// is a hand-written kernel (measured: the first-order ATen backward is cheap enough, see
// docs/kernel_design.md).
//
// silu_norm(z, cst) = cst * silu(z): e3nn normalize2mom(silu) inside the radial MLP, on the
// (E, 64) hidden activations. Three kernels: forward, backward (g * d1), backward-of-backward
// (two outputs); third order is refused (never needed: force loss = double backward).
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <torch/torch.h>

namespace jet_mlip {
namespace {

template <class T> __device__ __forceinline__ T xplor(T r, T r_on, T r_cut) {
  if (r < r_on) return T(1);
  const T rc2 = r_cut * r_cut, ron2 = r_on * r_on, r2 = r * r;
  const T d = rc2 - ron2, a = rc2 - r2;
  return a * a * (rc2 + T(2) * r2 - T(3) * ron2) / (d * d * d);
}

template <class T>
__global__ void edge_embed_kernel(const T* __restrict__ vec, const T* __restrict__ coeffs, int nb, T r_on,
                                  T r_cut, int lmax, T* __restrict__ B, T* __restrict__ sh, int64_t E) {
  const int64_t e = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (e >= E) return;
  const T vx = vec[3 * e], vy = vec[3 * e + 1], vz = vec[3 * e + 2];
  const T r = sqrt(vx * vx + vy * vy + vz * vz);
  const T cut = xplor(r, r_on, r_cut) * (T(2) / r_cut) / r;
  T* b = B + e * nb;
  for (int i = 0; i < nb; ++i) b[i] = sin(coeffs[i] * r) * cut;

  const T inv = T(1) / max(r, T(1e-12));  // torch.nn.functional.normalize
  const T x = vx * inv, y = vy * inv, z = vz * inv;
  const int sd = (lmax + 1) * (lmax + 1);
  T* s = sh + e * sd;
  s[0] = T(1);
  if (lmax < 1) return;
  const T s3 = T(1.7320508075688772);
  s[1] = s3 * x; s[2] = s3 * y; s[3] = s3 * z;
  if (lmax < 2) return;
  const T s15 = T(3.872983346207417), s5 = T(2.23606797749979);
  const T y2 = y * y, x2z2 = x * x + z * z;
  const T sh20 = s15 * x * z, sh24 = T(0.5) * s15 * (z * z - x * x);
  s[4] = sh20; s[5] = s15 * x * y; s[6] = s5 * (y2 - T(0.5) * x2z2); s[7] = s15 * y * z; s[8] = sh24;
  if (lmax < 3) return;
  const T s42_6 = T(1.0801234497346435), s7 = T(2.6457513110645907), s168_8 = T(1.6201851746019651);
  s[9] = s42_6 * (sh20 * z + sh24 * x);
  s[10] = s7 * sh20 * y;
  s[11] = s168_8 * (T(4) * y2 - x2z2) * x;
  s[12] = T(0.5) * s7 * y * (T(2) * y2 - T(3) * x2z2);
  s[13] = s168_8 * z * (T(4) * y2 - x2z2);
  s[14] = s7 * sh24 * y;
  s[15] = s42_6 * (sh24 * z - sh20 * x);
}

std::tuple<at::Tensor, at::Tensor> edge_embed_cuda(const at::Tensor& vec, const at::Tensor& coeffs, double r_on,
                                                   double r_cut, int64_t lmax) {
  TORCH_CHECK(vec.dim() == 2 && vec.size(1) == 3, "edge_embed: edge_vec must be (E, 3)");
  TORCH_CHECK(coeffs.dim() == 1 && coeffs.scalar_type() == vec.scalar_type(), "edge_embed: coeffs (NB,) same dtype");
  TORCH_CHECK(lmax >= 0 && lmax <= 3, "edge_embed: lmax must be in 0..3");
  const c10::cuda::CUDAGuard guard(vec.device());
  auto v = vec.contiguous(), c = coeffs.contiguous();
  const int64_t E = v.size(0), nb = c.numel(), sd = (lmax + 1) * (lmax + 1);
  auto B = at::empty({E, nb}, v.options());
  auto sh = at::empty({E, sd}, v.options());
  if (E == 0) return {B, sh};
  const int threads = 128;
  const dim3 grid((E + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(v.scalar_type(), "edge_embed", [&] {
    edge_embed_kernel<scalar_t><<<grid, threads, 0, stream>>>(
        v.data_ptr<scalar_t>(), c.data_ptr<scalar_t>(), (int)nb, (scalar_t)r_on, (scalar_t)r_cut, (int)lmax,
        B.data_ptr<scalar_t>(), sh.data_ptr<scalar_t>(), E);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {B, sh};
}

// ---- same math in ATen ops (differentiable to any order); used by the backward ----------
std::tuple<at::Tensor, at::Tensor> edge_embed_ref(const at::Tensor& vec, const at::Tensor& coeffs, double r_on,
                                                  double r_cut, int64_t lmax) {
  auto r = vec.norm(2, -1);
  auto ur = r.unsqueeze(-1);
  auto r2 = r * r;
  const double rc2 = r_cut * r_cut, ron2 = r_on * r_on, d = rc2 - ron2;
  auto cut = at::where(r < r_on, at::ones_like(r), (rc2 - r2).pow(2) * (rc2 + 2 * r2 - 3 * ron2) / (d * d * d));
  auto B = (2.0 / r_cut) * at::sin(coeffs * ur) / ur * cut.unsqueeze(-1);

  auto n = vec / at::clamp_min(ur, 1e-12);
  auto x = n.select(-1, 0), y = n.select(-1, 1), z = n.select(-1, 2);
  std::vector<at::Tensor> s{at::ones_like(x)};
  if (lmax >= 1) {
    const double s3 = std::sqrt(3.0);
    s.insert(s.end(), {s3 * x, s3 * y, s3 * z});
  }
  at::Tensor y2, x2z2, sh20, sh24;
  if (lmax >= 2) {
    const double s15 = std::sqrt(15.0), s5 = std::sqrt(5.0);
    y2 = y * y; x2z2 = x * x + z * z;
    sh20 = s15 * x * z; sh24 = 0.5 * s15 * (z * z - x * x);
    s.insert(s.end(), {sh20, s15 * x * y, s5 * (y2 - 0.5 * x2z2), s15 * y * z, sh24});
  }
  if (lmax >= 3) {
    const double s42_6 = std::sqrt(42.0) / 6, s7 = std::sqrt(7.0), s168_8 = std::sqrt(168.0) / 8;
    s.insert(s.end(), {s42_6 * (sh20 * z + sh24 * x), s7 * sh20 * y, s168_8 * (4 * y2 - x2z2) * x,
                       0.5 * s7 * y * (2 * y2 - 3 * x2z2), s168_8 * z * (4 * y2 - x2z2), s7 * sh24 * y,
                       s42_6 * (sh24 * z - sh20 * x)});
  }
  return {B, at::stack(s, -1)};
}

using torch::autograd::AutogradContext;
using torch::autograd::tensor_list;

struct EdgeEmbedFn : torch::autograd::Function<EdgeEmbedFn> {
  static tensor_list forward(AutogradContext* ctx, const at::Tensor& vec, const at::Tensor& coeffs, double r_on,
                             double r_cut, int64_t lmax) {
    ctx->save_for_backward({vec, coeffs});
    ctx->saved_data["r_on"] = r_on;
    ctx->saved_data["r_cut"] = r_cut;
    ctx->saved_data["lmax"] = lmax;
    auto out = edge_embed_cuda(vec, coeffs, r_on, r_cut, lmax);
    return {std::get<0>(out), std::get<1>(out)};
  }
  static tensor_list backward(AutogradContext* ctx, tensor_list go) {
    auto s = ctx->get_saved_variables();
    const bool create_graph = torch::autograd::GradMode::is_enabled();  // outer create_graph=True
    at::AutoGradMode enable(true);
    // the saved inputs themselves (not detached copies): with create_graph the returned grads
    // stay connected to the outer graph, which is what makes the higher orders work
    const auto &vec = s[0], &coeffs = s[1];
    auto out = edge_embed_ref(vec, coeffs, ctx->saved_data["r_on"].toDouble(),
                              ctx->saved_data["r_cut"].toDouble(), ctx->saved_data["lmax"].toInt());
    tensor_list outs, gos, ins;
    if (go[0].defined()) { outs.push_back(std::get<0>(out)); gos.push_back(go[0]); }
    if (go[1].defined()) { outs.push_back(std::get<1>(out)); gos.push_back(go[1]); }
    const bool nv = ctx->needs_input_grad(0), nc = ctx->needs_input_grad(1);
    if (nv) ins.push_back(vec);
    if (nc) ins.push_back(coeffs);
    tensor_list g = ins.empty() || outs.empty() ? tensor_list(ins.size())
                                                : torch::autograd::grad(outs, ins, gos, {}, create_graph, true);
    at::Tensor gv, gc;
    size_t k = 0;
    if (nv) gv = g[k++];
    if (nc) gc = g[k++];
    return {gv, gc, {}, {}, {}};
  }
};

std::tuple<at::Tensor, at::Tensor> edge_embed_ag(const at::Tensor& vec, const at::Tensor& coeffs, double r_on,
                                                 double r_cut, int64_t lmax) {
  auto o = EdgeEmbedFn::apply(vec, coeffs, r_on, r_cut, lmax);
  return {o[0], o[1]};
}

// ---- silu_norm: cst * silu(z) on the (E, 64) hidden activations, fwd / bwd / bwd-bwd kernels ----
template <class T> __device__ __forceinline__ T sigm(T z) { return T(1) / (T(1) + exp(-z)); }

template <class T>
__global__ void silu_norm_kernel(const T* __restrict__ z, T cst, T* __restrict__ y, int64_t n) {
  const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i < n) y[i] = cst * z[i] * sigm(z[i]);
}
// d1 = cst * silu'(z) = cst * s (1 + z (1 - s))
template <class T>
__global__ void silu_norm_bwd_kernel(const T* __restrict__ g, const T* __restrict__ z, T cst, T* __restrict__ gz,
                                     int64_t n) {
  const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= n) return;
  const T s = sigm(z[i]);
  gz[i] = g[i] * cst * s * (T(1) + z[i] * (T(1) - s));
}
// out = g * d1(z):  d out/d g = gg * d1 ;  d out/d z = gg * g * d2,  d2 = cst * s (1-s) (2 + z (1 - 2 s))
template <class T>
__global__ void silu_norm_bwd_bwd_kernel(const T* __restrict__ gg, const T* __restrict__ g, const T* __restrict__ z,
                                         T cst, T* __restrict__ ggrad, T* __restrict__ gz, int64_t n) {
  const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= n) return;
  const T s = sigm(z[i]), zi = z[i];
  ggrad[i] = gg[i] * cst * s * (T(1) + zi * (T(1) - s));
  gz[i] = gg[i] * g[i] * cst * s * (T(1) - s) * (T(2) + zi * (T(1) - T(2) * s));
}

constexpr int SN_THREADS = 256;
inline dim3 sn_grid(int64_t n) { return dim3((n + SN_THREADS - 1) / SN_THREADS); }

at::Tensor silu_norm_cuda(const at::Tensor& z, double cst) {
  const c10::cuda::CUDAGuard guard(z.device());
  auto zc = z.contiguous();
  auto y = at::empty_like(zc);
  const int64_t n = zc.numel();
  if (n == 0) return y;
  AT_DISPATCH_FLOATING_TYPES(zc.scalar_type(), "silu_norm", [&] {
    silu_norm_kernel<scalar_t><<<sn_grid(n), SN_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        zc.data_ptr<scalar_t>(), (scalar_t)cst, y.data_ptr<scalar_t>(), n);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor silu_norm_bwd_cuda(const at::Tensor& g, const at::Tensor& z, double cst) {
  const c10::cuda::CUDAGuard guard(z.device());
  auto zc = z.contiguous(), gc = g.contiguous();
  TORCH_CHECK(gc.sizes() == zc.sizes() && gc.scalar_type() == zc.scalar_type(), "silu_norm_bwd: g must match z");
  auto gz = at::empty_like(zc);
  const int64_t n = zc.numel();
  if (n == 0) return gz;
  AT_DISPATCH_FLOATING_TYPES(zc.scalar_type(), "silu_norm_bwd", [&] {
    silu_norm_bwd_kernel<scalar_t><<<sn_grid(n), SN_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        gc.data_ptr<scalar_t>(), zc.data_ptr<scalar_t>(), (scalar_t)cst, gz.data_ptr<scalar_t>(), n);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return gz;
}

std::tuple<at::Tensor, at::Tensor> silu_norm_bwd_bwd_cuda(const at::Tensor& gg, const at::Tensor& g,
                                                          const at::Tensor& z, double cst) {
  const c10::cuda::CUDAGuard guard(z.device());
  auto zc = z.contiguous(), gc = g.contiguous(), ggc = gg.contiguous();
  TORCH_CHECK(gc.sizes() == zc.sizes() && ggc.sizes() == zc.sizes(), "silu_norm_bwd_bwd: shapes must match");
  auto ggrad = at::empty_like(zc), gz = at::empty_like(zc);
  const int64_t n = zc.numel();
  if (n == 0) return {ggrad, gz};
  AT_DISPATCH_FLOATING_TYPES(zc.scalar_type(), "silu_norm_bwd_bwd", [&] {
    silu_norm_bwd_bwd_kernel<scalar_t><<<sn_grid(n), SN_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        ggc.data_ptr<scalar_t>(), gc.data_ptr<scalar_t>(), zc.data_ptr<scalar_t>(), (scalar_t)cst,
        ggrad.data_ptr<scalar_t>(), gz.data_ptr<scalar_t>(), n);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {ggrad, gz};
}

at::Tensor op_silu_norm_bwd(const at::Tensor& g, const at::Tensor& z, double cst) {  // dispatcher: autograd-recorded
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("jet_mlip::silu_norm_bwd", "")
                       .typed<at::Tensor(const at::Tensor&, const at::Tensor&, double)>();
  return op.call(g, z, cst);
}

struct SiluNormFn : torch::autograd::Function<SiluNormFn> {
  static at::Tensor forward(AutogradContext* ctx, const at::Tensor& z, double cst) {
    ctx->save_for_backward({z});
    ctx->saved_data["cst"] = cst;
    return silu_norm_cuda(z, cst);
  }
  static tensor_list backward(AutogradContext* ctx, tensor_list go) {
    auto z = ctx->get_saved_variables()[0];
    return {op_silu_norm_bwd(go[0], z, ctx->saved_data["cst"].toDouble()), {}};
  }
};

struct SiluNormBwdFn : torch::autograd::Function<SiluNormBwdFn> {
  static at::Tensor forward(AutogradContext* ctx, const at::Tensor& g, const at::Tensor& z, double cst) {
    ctx->save_for_backward({g, z});
    ctx->saved_data["cst"] = cst;
    return silu_norm_bwd_cuda(g, z, cst);
  }
  static tensor_list backward(AutogradContext* ctx, tensor_list go) {
    TORCH_CHECK(!torch::autograd::GradMode::is_enabled(), "jet_mlip::silu_norm: third-order autograd not supported");
    auto s = ctx->get_saved_variables();
    auto r = silu_norm_bwd_bwd_cuda(go[0], s[0], s[1], ctx->saved_data["cst"].toDouble());
    return {std::get<0>(r), std::get<1>(r), {}};
  }
};

at::Tensor silu_norm_ag(const at::Tensor& z, double cst) { return SiluNormFn::apply(z, cst); }
at::Tensor silu_norm_bwd_ag(const at::Tensor& g, const at::Tensor& z, double cst) {
  return SiluNormBwdFn::apply(g, z, cst);
}

}  // namespace
}  // namespace jet_mlip

TORCH_LIBRARY_FRAGMENT(jet_mlip, m) {
  m.def("edge_embed(Tensor edge_vec, Tensor coeffs, float r_on, float r_cut, int lmax) -> (Tensor, Tensor)");
  m.def("silu_norm(Tensor z, float cst) -> Tensor");
  m.def("silu_norm_bwd(Tensor g, Tensor z, float cst) -> Tensor");
}
TORCH_LIBRARY_IMPL(jet_mlip, CUDA, m) {
  m.impl("edge_embed", &jet_mlip::edge_embed_cuda);
  m.impl("silu_norm", &jet_mlip::silu_norm_cuda);
  m.impl("silu_norm_bwd", &jet_mlip::silu_norm_bwd_cuda);
}
TORCH_LIBRARY_IMPL(jet_mlip, Autograd, m) {
  m.impl("edge_embed", &jet_mlip::edge_embed_ag);
  m.impl("silu_norm", &jet_mlip::silu_norm_ag);
  m.impl("silu_norm_bwd", &jet_mlip::silu_norm_bwd_ag);
}
