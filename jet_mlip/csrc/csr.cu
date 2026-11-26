// build_csr: dst-grouped edge order for the node-centric kernels.
//   rowptr (int32, n_nodes+1), perm (int32, E): edge e' = perm[k] for k in
//   [rowptr[i], rowptr[i+1]) has dst == i. Stable sort so grouped inputs keep
//   their order (perm == arange when dst is already sorted).
// No host sync anywhere (CUDA-graph capturable): counts via index_add_ (at::bincount
// syncs on max()) and always sort (the sortedness check synced on .item() and cost
// more than the int32 stable sort it was meant to skip).
// build_csr_masked (CUDA): the same for a padded list, mask 1 = real edge; rowptr counts the real edges, the padding
//   (and any dst outside [0, n_nodes)) comes last. A LAMMPS pair list arrives grouped by i with the padding at its end,
//   so no sort: the count kernel also checks the order on the GPU and the place kernel then writes perm = arange.
//   Any other order still gives a valid CSR, by atomics: the edge order inside a row is then arbitrary.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

namespace jet_mlip {

std::tuple<at::Tensor, at::Tensor> build_csr(const at::Tensor& dst, int64_t n_nodes) {
  TORCH_CHECK(dst.dim() == 1, "dst must be 1-D");
  auto d = dst.to(at::kInt);
  auto counts = at::zeros({n_nodes}, d.options()).index_add_(0, d, at::ones_like(d));
  auto rowptr = at::zeros({n_nodes + 1}, d.options());
  rowptr.slice(0, 1, n_nodes + 1).copy_(at::cumsum(counts, 0));
  auto perm = std::get<1>(at::sort(d, /*stable=*/true, /*dim=*/0, /*descending=*/false));
  return {rowptr, perm.to(at::kInt)};
}

namespace {

__device__ __forceinline__ int csr_key(const int* dst, const int* mask, int e, int n) {
  const int d = dst[e];
  return mask[e] && d >= 0 && d < n ? d : n;
}

// buf: [0] = 0, [1 + k] = edges with key k < n, [1 + n] = 0, [n + 2] = 1 if the keys are not ascending.
// One atomic per run of equal keys (per row for a grouped list), cut every 64 edges: the first edge of a piece counts it.
__global__ void csr_count_kernel(const int* __restrict__ dst, const int* __restrict__ mask, int E, int n, int* buf) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= E) return;
  const int k = csr_key(dst, mask, e, n), kp = e > 0 ? csr_key(dst, mask, e - 1, n) : -1;
  if (kp > k) buf[n + 2] = 1;
  if (k == n || (kp == k && (e & 63))) return;
  int len = 1;
  while (e + len < E && ((e + len) & 63) && csr_key(dst, mask, e + len, n) == k) ++len;
  atomicAdd(buf + 1 + k, len);
}

// keys not ascending: buf[1 + k] counts down (the padding's counts up from 0), every edge takes one slot of its row
__global__ void csr_place_kernel(const int* __restrict__ dst, const int* __restrict__ mask, int E, int n, int* buf,
                                 const int* __restrict__ rowptr, int* perm) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= E) return;
  if (!buf[n + 2]) {
    perm[e] = e;
    return;
  }
  const int k = csr_key(dst, mask, e, n);
  perm[rowptr[k] + (k == n ? atomicAdd(buf + 1 + n, 1) : atomicSub(buf + 1 + k, 1) - 1)] = e;
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> build_csr_masked(const at::Tensor& dst, const at::Tensor& mask, int64_t n_nodes) {
  TORCH_CHECK(dst.dim() == 1 && dst.is_cuda() && mask.is_cuda() && mask.numel() == dst.numel(),
              "build_csr_masked: dst, mask (E,) CUDA");
  const auto d = dst.to(at::kInt).contiguous(), m = mask.to(at::kInt).contiguous();
  const int E = (int)d.numel(), n = (int)n_nodes;
  auto buf = at::zeros({n_nodes + 3}, d.options());
  auto perm = at::empty({E}, d.options());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int blocks = (E + 255) / 256;
  if (E > 0) {
    csr_count_kernel<<<blocks, 256, 0, stream>>>(d.data_ptr<int>(), m.data_ptr<int>(), E, n, buf.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  auto rowptr = at::cumsum(buf.slice(0, 0, n_nodes + 1), 0, at::kInt);   // n + 1 entries; [n] = real edges
  if (E > 0) {
    csr_place_kernel<<<blocks, 256, 0, stream>>>(d.data_ptr<int>(), m.data_ptr<int>(), E, n, buf.data_ptr<int>(),
                                                 rowptr.data_ptr<int>(), perm.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {rowptr, perm};
}

}  // namespace jet_mlip

TORCH_LIBRARY_FRAGMENT(jet_mlip, m) {
  m.def("build_csr(Tensor dst, int n_nodes) -> (Tensor, Tensor)");
  m.def("build_csr_masked(Tensor dst, Tensor mask, int n_nodes) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(jet_mlip, CompositeExplicitAutograd, m) {
  m.impl("build_csr", &jet_mlip::build_csr);
  m.impl("build_csr_masked", &jet_mlip::build_csr_masked);
}
