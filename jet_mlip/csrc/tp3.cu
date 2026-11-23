// tp3: trilinear tensor product + scatter, differentiable to any order.
//   out[dst[e], c] += coef * x[src[e], a] * sh[e, b] * W[e, w]   (see config.h for a,b,c,w)
// Lane = channel (chunks of 32 channels looped for mul > 32; per-path mul). X / W / SH
// slots: warp per edge. OUT slot: warp per dst node, looping the node's edges through the
// dst CSR (build_csr) once per chunk, so the row is accumulated in registers (one chunk's
// worth, same footprint as the edge kernel) and written once: sevenn's edge lists are
// dst-grouped, which serialized the per-edge atomics.
// One kernel template does the forward and all first-order partials: the four
// operands x (rows gathered by src), sh (per edge), W (per edge), g (rows gathered
// by dst) enter the same 4-way contraction and SLOT selects which one is produced.
// Every op's autograd backward is written with the other three *dispatcher* ops,
// so higher-order autograd needs no extra kernels.
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <torch/torch.h>

#include <type_traits>

#include "tp3.cuh"
#include "configs_gen.h"

namespace jet_mlip {
std::tuple<at::Tensor, at::Tensor> build_csr(const at::Tensor& dst, int64_t n_nodes);  // csr.cu
namespace {

using namespace tp3k;
}  // namespace
namespace tp3k {
// instantiated in csrc/inst/tp3_<id>*.cu (configs.py), one config per TU
#define JET_EXTERN1(T, CFG, S) \
  extern template void launch_cfg<T, CFG, S>(const T*, const T*, const T*, const T*, const int*, const int*, const int*, \
                                             const int*, T*, int, cudaStream_t);
#define JET_EXTERN(ID, CFG) \
  JET_EXTERN1(float, CFG, 0) JET_EXTERN1(float, CFG, 1) JET_EXTERN1(float, CFG, 2) JET_EXTERN1(float, CFG, 3) \
  JET_EXTERN1(double, CFG, 0) JET_EXTERN1(double, CFG, 1) JET_EXTERN1(double, CFG, 2) JET_EXTERN1(double, CFG, 3)
JET_FOREACH_CONFIG(JET_EXTERN)
#undef JET_EXTERN
#undef JET_EXTERN1
}  // namespace tp3k
namespace {

struct Dims {
  int in_dim, sh_dim, out_dim, nw;
};
Dims dims(int64_t cfg_id) {
  switch (cfg_id) {
#define JET_DIMS(i, C) \
  case i: return {C::IN_DIM, C::SH_DIM, C::OUT_DIM, C::NW};
    JET_FOREACH_CONFIG(JET_DIMS)
#undef JET_DIMS
    default: TORCH_CHECK(false, "jet: bad config id ", cfg_id);
  }
}

const void* ptr_or_null(const at::Tensor& t) { return t.defined() ? t.data_ptr() : nullptr; }

// Launch with operand SLOT produced into `out`; the SLOT input tensor is undefined.
// n = warps: edges (X / W / SH) or dst nodes (OUT, with the dst CSR rowptr/perm).
template <typename T, int SLOT>
void launch(int64_t cfg_id, const T* x, const T* sh, const T* w, const T* g, const int* src, const int* dst,
            const int* rowptr, const int* perm, T* out, int n, cudaStream_t stream) {
  switch (cfg_id) {
#define JET_LAUNCH(i, C) \
  case i: launch_cfg<T, C, SLOT>(x, sh, w, g, src, dst, rowptr, perm, out, n, stream); break;
    JET_FOREACH_CONFIG(JET_LAUNCH)
#undef JET_LAUNCH
    default: TORCH_CHECK(false, "jet: bad config id ", cfg_id);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int SLOT>
void run(const at::Tensor& x, const at::Tensor& sh, const at::Tensor& w, const at::Tensor& g,
         const at::Tensor& src, const at::Tensor& dst, at::Tensor& out, int64_t cfg_id) {
  const int64_t E = src.numel();
  if (E == 0) return;
  const c10::cuda::CUDAGuard guard(out.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  at::Tensor rowptr, perm;
  if constexpr (SLOT == OUT) std::tie(rowptr, perm) = build_csr(dst, out.size(0));
  AT_DISPATCH_FLOATING_TYPES(out.scalar_type(), "tp3", [&] {
    launch<scalar_t, SLOT>(cfg_id, (const scalar_t*)ptr_or_null(x), (const scalar_t*)ptr_or_null(sh),
                           (const scalar_t*)ptr_or_null(w), (const scalar_t*)ptr_or_null(g), src.data_ptr<int>(),
                           dst.data_ptr<int>(), (const int*)ptr_or_null(rowptr), (const int*)ptr_or_null(perm),
                           out.data_ptr<scalar_t>(), SLOT == OUT ? (int)out.size(0) : (int)E, stream);
  });
}

// Shape/dtype checks. Operands are (name, tensor, rows expected (-1 = any), cols).
void check(const char* op, std::initializer_list<std::tuple<const char*, const at::Tensor*, int64_t, int64_t>> ts,
           const at::Tensor& src, const at::Tensor& dst) {
  const at::Tensor* first = nullptr;
  for (auto& [name, t, rows, cols] : ts) {
    TORCH_CHECK(t->is_cuda() && t->is_contiguous(), op, ": ", name, " must be a contiguous CUDA tensor");
    TORCH_CHECK(t->dim() == 2 && t->size(1) == cols && (rows < 0 || t->size(0) == rows), op, ": ", name,
                " has shape ", t->sizes(), ", expected (", rows < 0 ? -1 : rows, ", ", cols, ")");
    if (!first) first = t;
    TORCH_CHECK(t->scalar_type() == first->scalar_type(), op, ": dtype mismatch on ", name);
  }
  for (auto* idx : {&src, &dst})
    TORCH_CHECK(idx->is_cuda() && idx->is_contiguous() && idx->scalar_type() == at::kInt && idx->dim() == 1 &&
                    idx->numel() == src.numel(),
                op, ": src/dst must be contiguous int32 CUDA 1-D tensors of equal length");
}

// ---- CUDA impls ------------------------------------------------------------------
at::Tensor tp3_cuda(const at::Tensor& x, const at::Tensor& sh, const at::Tensor& W, const at::Tensor& src,
                    const at::Tensor& dst, int64_t n_out, int64_t cfg_id) {
  auto D = dims(cfg_id);
  const int64_t E = src.numel();
  check("tp3", {{"x", &x, -1, D.in_dim}, {"sh", &sh, E, D.sh_dim}, {"W", &W, E, D.nw}}, src, dst);
  auto out = at::zeros({n_out, D.out_dim}, x.options());
  run<OUT>(x, sh, W, {}, src, dst, out, cfg_id);
  return out;
}

at::Tensor tp3_x_cuda(const at::Tensor& g, const at::Tensor& sh, const at::Tensor& W, const at::Tensor& src,
                      const at::Tensor& dst, int64_t n_x, int64_t cfg_id) {
  auto D = dims(cfg_id);
  const int64_t E = src.numel();
  check("tp3_x", {{"g", &g, -1, D.out_dim}, {"sh", &sh, E, D.sh_dim}, {"W", &W, E, D.nw}}, src, dst);
  auto out = at::zeros({n_x, D.in_dim}, g.options());
  run<X>({}, sh, W, g, src, dst, out, cfg_id);
  return out;
}

at::Tensor tp3_w_cuda(const at::Tensor& g, const at::Tensor& x, const at::Tensor& sh, const at::Tensor& src,
                      const at::Tensor& dst, int64_t cfg_id) {
  auto D = dims(cfg_id);
  const int64_t E = src.numel();
  check("tp3_w", {{"g", &g, -1, D.out_dim}, {"x", &x, -1, D.in_dim}, {"sh", &sh, E, D.sh_dim}}, src, dst);
  auto out = at::empty({E, D.nw}, g.options());
  run<W>(x, sh, {}, g, src, dst, out, cfg_id);
  return out;
}

at::Tensor tp3_sh_cuda(const at::Tensor& g, const at::Tensor& x, const at::Tensor& W, const at::Tensor& src,
                       const at::Tensor& dst, int64_t cfg_id) {
  auto D = dims(cfg_id);
  const int64_t E = src.numel();
  check("tp3_sh", {{"g", &g, -1, D.out_dim}, {"x", &x, -1, D.in_dim}, {"W", &W, E, D.nw}}, src, dst);
  auto out = at::empty({E, D.sh_dim}, g.options());
  run<SH>(x, {}, W, g, src, dst, out, cfg_id);
  return out;
}

// ---- dispatcher calls (go through Autograd, so backward-of-backward is recorded) --
using T7 = at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                      const at::Tensor&, int64_t, int64_t);
using T6 = at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                      const at::Tensor&, int64_t);
#define JET_OP(name, SIG) \
  c10::Dispatcher::singleton().findSchemaOrThrow("jet_mlip::" #name, "").typed<SIG>()
at::Tensor op_tp3(const at::Tensor& x, const at::Tensor& sh, const at::Tensor& W, const at::Tensor& src,
                  const at::Tensor& dst, int64_t n_out, int64_t cfg) {
  static auto op = JET_OP(tp3, T7);
  return op.call(x, sh, W, src, dst, n_out, cfg);
}
at::Tensor op_tp3_x(const at::Tensor& g, const at::Tensor& sh, const at::Tensor& W, const at::Tensor& src,
                    const at::Tensor& dst, int64_t n_x, int64_t cfg) {
  static auto op = JET_OP(tp3_x, T7);
  return op.call(g, sh, W, src, dst, n_x, cfg);
}
at::Tensor op_tp3_w(const at::Tensor& g, const at::Tensor& x, const at::Tensor& sh, const at::Tensor& src,
                    const at::Tensor& dst, int64_t cfg) {
  static auto op = JET_OP(tp3_w, T6);
  return op.call(g, x, sh, src, dst, cfg);
}
at::Tensor op_tp3_sh(const at::Tensor& g, const at::Tensor& x, const at::Tensor& W, const at::Tensor& src,
                     const at::Tensor& dst, int64_t cfg) {
  static auto op = JET_OP(tp3_sh, T6);
  return op.call(g, x, W, src, dst, cfg);
}
#undef JET_OP

// ---- autograd: each backward = the same contraction with the incoming grad in the
// produced operand's slot. Operand roles: x-slot rows gathered by src, g-slot rows
// gathered by dst, sh/W-slot per edge.
using torch::autograd::AutogradContext;
using torch::autograd::tensor_list;

struct Tp3Fn : torch::autograd::Function<Tp3Fn> {
  static at::Tensor forward(AutogradContext* ctx, const at::Tensor& x, const at::Tensor& sh, const at::Tensor& W,
                            const at::Tensor& src, const at::Tensor& dst, int64_t n_out, int64_t cfg) {
    ctx->save_for_backward({x, sh, W, src, dst});
    ctx->saved_data["cfg"] = cfg;
    return tp3_cuda(x, sh, W, src, dst, n_out, cfg);
  }
  static tensor_list backward(AutogradContext* ctx, tensor_list go) {
    auto s = ctx->get_saved_variables();
    const auto &x = s[0], &sh = s[1], &W = s[2], &src = s[3], &dst = s[4];
    auto g = go[0].contiguous();
    int64_t cfg = ctx->saved_data["cfg"].toInt();
    at::Tensor gx, gsh, gW;
    if (ctx->needs_input_grad(0)) gx = op_tp3_x(g, sh, W, src, dst, x.size(0), cfg);
    if (ctx->needs_input_grad(1)) gsh = op_tp3_sh(g, x, W, src, dst, cfg);
    if (ctx->needs_input_grad(2)) gW = op_tp3_w(g, x, sh, src, dst, cfg);
    return {gx, gsh, gW, {}, {}, {}, {}};
  }
};

struct Tp3XFn : torch::autograd::Function<Tp3XFn> {
  static at::Tensor forward(AutogradContext* ctx, const at::Tensor& g, const at::Tensor& sh, const at::Tensor& W,
                            const at::Tensor& src, const at::Tensor& dst, int64_t n_x, int64_t cfg) {
    ctx->save_for_backward({g, sh, W, src, dst});
    ctx->saved_data["cfg"] = cfg;
    return tp3_x_cuda(g, sh, W, src, dst, n_x, cfg);
  }
  static tensor_list backward(AutogradContext* ctx, tensor_list go) {
    auto s = ctx->get_saved_variables();
    const auto &g = s[0], &sh = s[1], &W = s[2], &src = s[3], &dst = s[4];
    auto ggx = go[0].contiguous();  // (n_x, IN_DIM): takes the x slot
    int64_t cfg = ctx->saved_data["cfg"].toInt();
    at::Tensor gg, gsh, gW;
    if (ctx->needs_input_grad(0)) gg = op_tp3(ggx, sh, W, src, dst, g.size(0), cfg);
    if (ctx->needs_input_grad(1)) gsh = op_tp3_sh(g, ggx, W, src, dst, cfg);
    if (ctx->needs_input_grad(2)) gW = op_tp3_w(g, ggx, sh, src, dst, cfg);
    return {gg, gsh, gW, {}, {}, {}, {}};
  }
};

struct Tp3WFn : torch::autograd::Function<Tp3WFn> {
  static at::Tensor forward(AutogradContext* ctx, const at::Tensor& g, const at::Tensor& x, const at::Tensor& sh,
                            const at::Tensor& src, const at::Tensor& dst, int64_t cfg) {
    ctx->save_for_backward({g, x, sh, src, dst});
    ctx->saved_data["cfg"] = cfg;
    return tp3_w_cuda(g, x, sh, src, dst, cfg);
  }
  static tensor_list backward(AutogradContext* ctx, tensor_list go) {
    auto s = ctx->get_saved_variables();
    const auto &g = s[0], &x = s[1], &sh = s[2], &src = s[3], &dst = s[4];
    auto ggw = go[0].contiguous();  // (E, NW): takes the W slot
    int64_t cfg = ctx->saved_data["cfg"].toInt();
    at::Tensor gg, gx, gsh;
    if (ctx->needs_input_grad(0)) gg = op_tp3(x, sh, ggw, src, dst, g.size(0), cfg);
    if (ctx->needs_input_grad(1)) gx = op_tp3_x(g, sh, ggw, src, dst, x.size(0), cfg);
    if (ctx->needs_input_grad(2)) gsh = op_tp3_sh(g, x, ggw, src, dst, cfg);
    return {gg, gx, gsh, {}, {}, {}};
  }
};

struct Tp3ShFn : torch::autograd::Function<Tp3ShFn> {
  static at::Tensor forward(AutogradContext* ctx, const at::Tensor& g, const at::Tensor& x, const at::Tensor& W,
                            const at::Tensor& src, const at::Tensor& dst, int64_t cfg) {
    ctx->save_for_backward({g, x, W, src, dst});
    ctx->saved_data["cfg"] = cfg;
    return tp3_sh_cuda(g, x, W, src, dst, cfg);
  }
  static tensor_list backward(AutogradContext* ctx, tensor_list go) {
    auto s = ctx->get_saved_variables();
    const auto &g = s[0], &x = s[1], &W = s[2], &src = s[3], &dst = s[4];
    auto ggs = go[0].contiguous();  // (E, SH_DIM): takes the sh slot
    int64_t cfg = ctx->saved_data["cfg"].toInt();
    at::Tensor gg, gx, gW;
    if (ctx->needs_input_grad(0)) gg = op_tp3(x, ggs, W, src, dst, g.size(0), cfg);
    if (ctx->needs_input_grad(1)) gx = op_tp3_x(g, ggs, W, src, dst, x.size(0), cfg);
    if (ctx->needs_input_grad(2)) gW = op_tp3_w(g, x, ggs, src, dst, cfg);
    return {gg, gx, gW, {}, {}, {}};
  }
};

at::Tensor tp3_ag(const at::Tensor& x, const at::Tensor& sh, const at::Tensor& W, const at::Tensor& src,
                  const at::Tensor& dst, int64_t n_out, int64_t cfg) {
  return Tp3Fn::apply(x, sh, W, src, dst, n_out, cfg);
}
at::Tensor tp3_x_ag(const at::Tensor& g, const at::Tensor& sh, const at::Tensor& W, const at::Tensor& src,
                    const at::Tensor& dst, int64_t n_x, int64_t cfg) {
  return Tp3XFn::apply(g, sh, W, src, dst, n_x, cfg);
}
at::Tensor tp3_w_ag(const at::Tensor& g, const at::Tensor& x, const at::Tensor& sh, const at::Tensor& src,
                    const at::Tensor& dst, int64_t cfg) {
  return Tp3WFn::apply(g, x, sh, src, dst, cfg);
}
at::Tensor tp3_sh_ag(const at::Tensor& g, const at::Tensor& x, const at::Tensor& W, const at::Tensor& src,
                     const at::Tensor& dst, int64_t cfg) {
  return Tp3ShFn::apply(g, x, W, src, dst, cfg);
}

}  // namespace
}  // namespace jet_mlip

TORCH_LIBRARY_FRAGMENT(jet_mlip, m) {
  m.def("tp3(Tensor x, Tensor sh, Tensor W, Tensor src, Tensor dst, int n_out, int cfg_id) -> Tensor");
  m.def("tp3_x(Tensor g, Tensor sh, Tensor W, Tensor src, Tensor dst, int n_x, int cfg_id) -> Tensor");
  m.def("tp3_w(Tensor g, Tensor x, Tensor sh, Tensor src, Tensor dst, int cfg_id) -> Tensor");
  m.def("tp3_sh(Tensor g, Tensor x, Tensor W, Tensor src, Tensor dst, int cfg_id) -> Tensor");
}
TORCH_LIBRARY_IMPL(jet_mlip, CUDA, m) {
  m.impl("tp3", &jet_mlip::tp3_cuda);
  m.impl("tp3_x", &jet_mlip::tp3_x_cuda);
  m.impl("tp3_w", &jet_mlip::tp3_w_cuda);
  m.impl("tp3_sh", &jet_mlip::tp3_sh_cuda);
}
TORCH_LIBRARY_IMPL(jet_mlip, Autograd, m) {
  m.impl("tp3", &jet_mlip::tp3_ag);
  m.impl("tp3_x", &jet_mlip::tp3_x_ag);
  m.impl("tp3_w", &jet_mlip::tp3_w_ag);
  m.impl("tp3_sh", &jet_mlip::tp3_sh_ag);
}
