// force_reduce: pair_e3gnn's force and virial from dE/dr in one kernel, written into the step's D2H buffer.
//   Edges are grouped by center (edge_build's CSR: rows [rowptr[ii], rowptr[ii+1]) have ctr = ii, nbr[e] = j).
//   F_ii += sum_e dEdr[e] (warp sum, no atomics), F_j -= dEdr[e] (double atomics), virial = -sum_e voigt(vec[e], dEdr[e])
//   with voigt = (r0 d0, r1 d1, r2 d2, r0 d1, r1 d2, r2 d0), the old torch order. Sums in double. out (3n + 8) float64:
//   [0, 3n) force rows, [3n+1, 3n+7) virial; out[3n] (energy) and out[3n+7] (edge count) are left for the caller.
//   Replaces zeros + 2 index_add + 4 mul/cat + sum + 2 cat + to(double) (nano-6.0, 11k atoms, H200: ~0.4 ms of 4.4).
// CUDA-graph capturable: no sync, no allocation.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

namespace jet_mlip {
namespace {

constexpr int kBlock = 256;

__global__ void force_reduce_kernel(int nrows, const int* __restrict__ rowptr, const int* __restrict__ nbr,
                                    const float* __restrict__ dEdr, const float* __restrict__ vec, double* out,
                                    int64_t nforce) {
  __shared__ double vs[kBlock / 32][6];
  const int lane = threadIdx.x & 31, wib = threadIdx.x >> 5;
  const int ii = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  double f[3] = {0, 0, 0}, v[6] = {0, 0, 0, 0, 0, 0};
  if (ii < nrows) {   // warp-uniform
    const int end = rowptr[ii + 1];
    for (int base = rowptr[ii]; base < end; base += 32) {
      const int e = base + lane;
      if (e < end) {
        const double d0 = dEdr[3 * e], d1 = dEdr[3 * e + 1], d2 = dEdr[3 * e + 2];
        const double r0 = vec[3 * e], r1 = vec[3 * e + 1], r2 = vec[3 * e + 2];
        const int j = nbr[e];
        atomicAdd(out + 3 * j, -d0);
        atomicAdd(out + 3 * j + 1, -d1);
        atomicAdd(out + 3 * j + 2, -d2);
        f[0] += d0; f[1] += d1; f[2] += d2;
        v[0] += r0 * d0; v[1] += r1 * d1; v[2] += r2 * d2; v[3] += r0 * d1; v[4] += r1 * d2; v[5] += r2 * d0;
      }
    }
  }
  for (int o = 16; o > 0; o >>= 1) {
    for (int k = 0; k < 3; k++) f[k] += __shfl_xor_sync(0xffffffffu, f[k], o);
    for (int k = 0; k < 6; k++) v[k] += __shfl_xor_sync(0xffffffffu, v[k], o);
  }
  if (lane == 0) {
    if (ii < nrows)
      for (int k = 0; k < 3; k++) atomicAdd(out + 3 * ii + k, f[k]);
    for (int k = 0; k < 6; k++) vs[wib][k] = v[k];
  }
  __syncthreads();
  if (threadIdx.x < 6) {
    double s = 0;
    for (int w = 0; w < kBlock / 32; w++) s += vs[w][threadIdx.x];
    atomicAdd(out + 3 * nforce + 1 + threadIdx.x, -s);
  }
}

}  // namespace

void force_reduce(const at::Tensor& dEdr, const at::Tensor& vec, const at::Tensor& rowptr, const at::Tensor& nbr,
                  at::Tensor& out) {
  TORCH_CHECK(dEdr.is_cuda() && dEdr.scalar_type() == at::kFloat && dEdr.dim() == 2 && dEdr.size(1) == 3 && dEdr.is_contiguous(),
              "dEdr: (cap, 3) float32 contiguous CUDA");
  TORCH_CHECK(vec.sizes() == dEdr.sizes() && vec.scalar_type() == at::kFloat && vec.is_contiguous(), "vec: like dEdr");
  TORCH_CHECK(rowptr.scalar_type() == at::kInt && nbr.scalar_type() == at::kInt && rowptr.is_contiguous() && nbr.is_contiguous(),
              "rowptr / nbr: int32 contiguous");
  TORCH_CHECK(nbr.numel() >= dEdr.size(0) || rowptr.numel() == 1, "nbr: one entry per dEdr row");
  TORCH_CHECK(out.scalar_type() == at::kDouble && out.is_contiguous() && out.numel() >= 8 && (out.numel() - 8) % 3 == 0,
              "out: (3n + 8) float64");
  const int64_t nforce = (out.numel() - 8) / 3, nrows = rowptr.numel() - 1;
  TORCH_CHECK(nrows <= nforce, "more CSR rows than force rows");
  out.zero_();
  if (nrows == 0) return;
  const dim3 grid((nrows * 32 + kBlock - 1) / kBlock);
  force_reduce_kernel<<<grid, kBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
      (int)nrows, rowptr.data_ptr<int>(), nbr.data_ptr<int>(), dEdr.data_ptr<float>(), vec.data_ptr<float>(),
      out.data_ptr<double>(), nforce);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace jet_mlip

TORCH_LIBRARY_FRAGMENT(jet_mlip, m) {
  m.def("force_reduce(Tensor dEdr, Tensor vec, Tensor rowptr, Tensor nbr, Tensor(a!) out) -> ()");
}
TORCH_LIBRARY_IMPL(jet_mlip, CUDA, m) {
  m.impl("force_reduce", &jet_mlip::force_reduce);
}
