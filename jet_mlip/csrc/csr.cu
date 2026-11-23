// build_csr: dst-grouped edge order for the node-centric kernels.
//   rowptr (int32, n_nodes+1), perm (int32, E): edge e' = perm[k] for k in
//   [rowptr[i], rowptr[i+1]) has dst == i. Stable sort so grouped inputs keep
//   their order (perm == arange when dst is already sorted).
// No host sync anywhere (CUDA-graph capturable): counts via index_add_ (at::bincount
// syncs on max()) and always sort (the sortedness check synced on .item() and cost
// more than the int32 stable sort it was meant to skip).
#include <ATen/ATen.h>
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

}  // namespace jet_mlip

TORCH_LIBRARY_FRAGMENT(jet_mlip, m) {
  m.def("build_csr(Tensor dst, int n_nodes) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(jet_mlip, CompositeExplicitAutograd, m) {
  m.impl("build_csr", &jet_mlip::build_csr);
}
