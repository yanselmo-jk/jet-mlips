// The per-step host work of jet_mlip.mliap (GraphMLIAP) as two ops, for small systems where a step is host-bound:
//   mliap_load: LAMMPS' pair_i / pair_j / rij / elems (raw device pointers) into the raw buffers, device memcpys
//   mliap_prep: the pair-list buffers of the model (ghost fold, padding, mask, node types) in one kernel instead of a
//               dozen elementwise ones; CUDA-graph capturable
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <algorithm>

namespace jet_mlip {
namespace {

// Edge e < npairs is (pair_i, owner[pair_j]) with mask 1; a padding edge sits on local row e % nlocal with mask 0.
// Node types: the local elements, then the remote rows' (several ranks) or a valid one (padding rows).
struct PrepArgs {
  const int* pi; const int* pj; const int64_t* owner; const int64_t* npairs; const int64_t* elems; const int64_t* rtypes;
  void* ei; int* mask; int64_t* types;
  int ecap, ncap, nlocal, tcap, ei64;
};
__global__ void mliap_prep_kernel(PrepArgs a) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e < a.ecap) {
    const bool real = e < a.npairs[0];
    const int pad = e % a.nlocal;
    const int64_t i = real ? a.pi[e] : pad, j = real ? a.owner[min(max(a.pj[e], 0), a.tcap - 1)] : pad;
    if (a.ei64) {
      static_cast<int64_t*>(a.ei)[e] = i;
      static_cast<int64_t*>(a.ei)[a.ecap + e] = j;
    } else {
      static_cast<int*>(a.ei)[e] = (int)i;
      static_cast<int*>(a.ei)[a.ecap + e] = (int)j;
    }
    a.mask[e] = real;
  }
  if (e < a.ncap) a.types[e] = e < a.nlocal ? a.elems[e] : a.rtypes ? a.rtypes[e] : a.elems[0];
}

bool i32(const at::Tensor& t) { return t.is_cuda() && t.scalar_type() == at::kInt && t.is_contiguous(); }
bool i64(const at::Tensor& t) { return t.is_cuda() && t.scalar_type() == at::kLong && t.is_contiguous(); }

// writes ei (2, ecap) int32 / int64, mask (ecap) int32, types (ncap) int64; rtypes (ncap) int64 or None
void mliap_prep(const at::Tensor& pi, const at::Tensor& pj, const at::Tensor& owner, const at::Tensor& npairs,
                const at::Tensor& elems, const std::optional<at::Tensor>& rtypes, at::Tensor ei, at::Tensor mask,
                at::Tensor types) {
  const int64_t ecap = mask.numel(), ncap = types.numel();
  TORCH_CHECK(i32(pi) && i32(pj) && i32(mask) && i64(owner) && i64(npairs) && i64(elems) && i64(types) &&
              (i32(ei) || i64(ei)) && (!rtypes.has_value() || i64(*rtypes)), "mliap_prep: dtypes");
  TORCH_CHECK(pi.numel() == ecap && pj.numel() == ecap && ei.dim() == 2 && ei.size(0) == 2 && ei.size(1) == ecap &&
              elems.numel() >= 1 && elems.numel() <= ncap && (!rtypes.has_value() || rtypes->numel() == ncap) &&
              owner.numel() >= 1 && npairs.numel() == 1, "mliap_prep: sizes");
  const c10::cuda::CUDAGuard guard(mask.device());   // the kernels below go to the current device's stream
  const PrepArgs a{pi.data_ptr<int>(), pj.data_ptr<int>(), owner.data_ptr<int64_t>(), npairs.data_ptr<int64_t>(),
                   elems.data_ptr<int64_t>(), rtypes.has_value() ? rtypes->data_ptr<int64_t>() : nullptr, ei.data_ptr(),
                   mask.data_ptr<int>(), types.data_ptr<int64_t>(), (int)ecap, (int)ncap, (int)elems.numel(),
                   (int)owner.numel(), ei.scalar_type() == at::kLong};
  const int64_t nthr = std::max(ecap, ncap);
  if (nthr == 0) return;
  mliap_prep_kernel<<<(unsigned)((nthr + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// pi / pj (npairs int32) and rij (npairs x 3 fp64) are device pointers of the current device; the pairs are copied
// for a new list only. elems (int32 device pointer, 0: none): the first nlocal into r_el, every step
void mliap_load(int64_t pi, int64_t pj, int64_t rij, int64_t npairs, bool new_list, at::Tensor r_pi, at::Tensor r_pj,
                at::Tensor r_rij, at::Tensor r_n, int64_t elems, int64_t nlocal, const std::optional<at::Tensor>& r_el) {
  TORCH_CHECK(i32(r_pi) && i32(r_pj) && r_rij.is_cuda() && r_rij.scalar_type() == at::kDouble && r_rij.is_contiguous() &&
              npairs >= 0 && npairs <= r_pi.numel() && npairs <= r_pj.numel() && 3 * npairs <= r_rij.numel(),
              "mliap_load: buffers");
  TORCH_CHECK(!r_el.has_value() || (elems != 0 && i32(*r_el) && nlocal >= 0 && nlocal <= r_el->numel()), "mliap_load: elems");
  const c10::cuda::CUDAGuard guard(r_rij.device());   // the kernels below go to the current device's stream
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto copy = [&](void* dst, int64_t src, size_t bytes) {
    if (bytes)
      C10_CUDA_CHECK(cudaMemcpyAsync(dst, reinterpret_cast<const void*>(src), bytes, cudaMemcpyDeviceToDevice, stream));
  };
  if (new_list) {
    copy(r_pi.data_ptr(), pi, npairs * sizeof(int));
    copy(r_pj.data_ptr(), pj, npairs * sizeof(int));
    r_n.fill_(npairs);
  }
  copy(r_rij.data_ptr(), rij, npairs * 3 * sizeof(double));
  if (r_el.has_value()) copy(r_el->data_ptr(), elems, nlocal * sizeof(int));
}

}  // namespace
}  // namespace jet_mlip

TORCH_LIBRARY_FRAGMENT(jet_mlip, m) {
  m.def("mliap_prep(Tensor pi, Tensor pj, Tensor owner, Tensor npairs, Tensor elems, Tensor? rtypes, Tensor(a!) ei, "
        "Tensor(b!) mask, Tensor(c!) types) -> ()");
  m.def("mliap_load(int pi, int pj, int rij, int npairs, bool new_list, Tensor(a!) r_pi, Tensor(b!) r_pj, "
        "Tensor(c!) r_rij, Tensor(d!) r_n, int elems=0, int nlocal=0, Tensor? r_el=None) -> ()");
}
TORCH_LIBRARY_IMPL(jet_mlip, CompositeExplicitAutograd, m) {
  m.impl("mliap_prep", &jet_mlip::mliap_prep);
  m.impl("mliap_load", &jet_mlip::mliap_load);
}
