// edge_build: pair_e3gnn's edge list built on the GPU, bitwise the host's.
//   Graph node ii (LAMMPS atom i = center[ii]) visits its candidates cand_j[cand_ptr[ii]:cand_ptr[ii+1]] in
//   order; j with owner[j] >= 0 (graph index of its owner, -1 = not local) and |x[j]-x[i]|^2 < cutoff_sq is one
//   edge: ctr = ii (edge_index[0]), nbr = owner[j] (edge_index[1]), vec = float(x[j]-x[i]). Edges are grouped by
//   ii in candidate order, rowptr is the CSR over ii, total[0] = edge count. Entries e >= cap = nbr.size(0) are
//   not written; the caller re-runs with a larger cap. The host build has no FMA (-march=nocona), so the double
//   math uses the _rn intrinsics in its order: ((d0*d0 + d1*d1) + d2*d2).
// CUDA-graph capturable: no host sync, no data-dependent allocation, in-place outputs.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

namespace jet_mlip {
namespace {

// one warp per node. COUNT: rowptr[ii+1] = #edges of ii. FILL: write edges at rowptr[ii] + rank.
template <bool FILL>
__global__ void edge_build_kernel(int nlocal, const double* __restrict__ x, const int* __restrict__ cand_ptr,
                                  const int* __restrict__ cand_j, const int* __restrict__ center,
                                  const int* __restrict__ owner, double cutoff_sq, int* rowptr,
                                  int* __restrict__ nbr, int* __restrict__ ctr, float* __restrict__ vec, int cap,
                                  int* __restrict__ total) {
  const int lane = threadIdx.x & 31;
  const int ii = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (ii >= nlocal) return;   // warp-uniform
  const int i = center[ii];
  const double xi0 = x[3 * i], xi1 = x[3 * i + 1], xi2 = x[3 * i + 2];
  const int end = cand_ptr[ii + 1];
  int e = FILL ? rowptr[ii] : 0;
  for (int k = cand_ptr[ii]; k < end; k += 32) {
    int o = -1;
    double d0 = 0, d1 = 0, d2 = 0;
    bool ok = false;
    if (k + lane < end) {
      const int j = cand_j[k + lane];
      o = owner[j];
      d0 = __dsub_rn(x[3 * j], xi0);
      d1 = __dsub_rn(x[3 * j + 1], xi1);
      d2 = __dsub_rn(x[3 * j + 2], xi2);
      const double r = __dadd_rn(__dadd_rn(__dmul_rn(d0, d0), __dmul_rn(d1, d1)), __dmul_rn(d2, d2));
      ok = o >= 0 && r < cutoff_sq;
    }
    const unsigned b = __ballot_sync(0xffffffffu, ok);
    if (FILL && ok) {
      const int idx = e + __popc(b & ((1u << lane) - 1));
      if (idx < cap) {
        nbr[idx] = o;
        ctr[idx] = ii;
        vec[3 * idx] = __double2float_rn(d0);
        vec[3 * idx + 1] = __double2float_rn(d1);
        vec[3 * idx + 2] = __double2float_rn(d2);
      }
    }
    e += __popc(b);
  }
  if (lane != 0) return;
  if (!FILL) rowptr[ii + 1] = e;
  if (ii == 0) {
    if (FILL) total[0] = rowptr[nlocal]; else rowptr[0] = 0;
  }
}

}  // namespace

void edge_build(const at::Tensor& x, const at::Tensor& cand_ptr, const at::Tensor& cand_j, const at::Tensor& center,
                const at::Tensor& owner, double cutoff_sq, at::Tensor& rowptr, at::Tensor& nbr, at::Tensor& ctr,
                at::Tensor& vec, at::Tensor& total) {
  const int64_t nlocal = center.numel(), cap = nbr.size(0);
  TORCH_CHECK(x.scalar_type() == at::kDouble && x.dim() == 2 && x.size(1) == 3 && x.is_contiguous(), "x: (nall, 3) float64 contiguous");
  const at::Tensor* ints[] = {&cand_ptr, &cand_j, &center, &owner, &rowptr, &nbr, &ctr, &total};
  for (const at::Tensor* t : ints)
    TORCH_CHECK(t->scalar_type() == at::kInt && t->is_contiguous() && t->is_cuda(), "int32 contiguous CUDA tensors expected");
  TORCH_CHECK(cand_ptr.numel() == nlocal + 1 && rowptr.numel() == nlocal + 1, "cand_ptr / rowptr: nlocal + 1");
  TORCH_CHECK(owner.numel() == x.size(0), "owner: one entry per x row");
  TORCH_CHECK(ctr.size(0) == cap && vec.size(0) == cap && vec.size(1) == 3 && vec.scalar_type() == at::kFloat && vec.is_contiguous(), "nbr / ctr / vec: cap rows");
  TORCH_CHECK(total.numel() == 1, "total: one int32");
  if (nlocal == 0) {
    rowptr.zero_();
    total.zero_();
    return;
  }
  const dim3 block(256), grid((nlocal * 32 + 255) / 256);
  const c10::cuda::CUDAGuard guard(nbr.device());   // the kernels below go to the current device's stream
  auto stream = at::cuda::getCurrentCUDAStream();
  edge_build_kernel<false><<<grid, block, 0, stream>>>(
      nlocal, x.data_ptr<double>(), cand_ptr.data_ptr<int>(), cand_j.data_ptr<int>(), center.data_ptr<int>(),
      owner.data_ptr<int>(), cutoff_sq, rowptr.data_ptr<int>(), nullptr, nullptr, nullptr, 0, nullptr);
  auto counts = rowptr.narrow(0, 1, nlocal);
  at::cumsum_out(counts, counts, 0);   // in place; cub scan, capturable
  edge_build_kernel<true><<<grid, block, 0, stream>>>(
      nlocal, x.data_ptr<double>(), cand_ptr.data_ptr<int>(), cand_j.data_ptr<int>(), center.data_ptr<int>(),
      owner.data_ptr<int>(), cutoff_sq, rowptr.data_ptr<int>(), nbr.data_ptr<int>(), ctr.data_ptr<int>(),
      vec.data_ptr<float>(), (int)cap, total.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace jet_mlip

TORCH_LIBRARY_FRAGMENT(jet_mlip, m) {
  m.def("edge_build(Tensor x, Tensor cand_ptr, Tensor cand_j, Tensor center, Tensor owner, float cutoff_sq, "
        "Tensor(a!) rowptr, Tensor(b!) nbr, Tensor(c!) ctr, Tensor(d!) vec, Tensor(e!) total) -> ()");
}
TORCH_LIBRARY_IMPL(jet_mlip, CUDA, m) {
  m.impl("edge_build", &jet_mlip::edge_build);
}
