// cand_build: pair_e3gnn's neighbor (candidate) list built on the GPU, replacing LAMMPS' CPU full list on one rank.
//   For every center ii < ncenters (row ii of x = LAMMPS local atom ii) it lists the rows j != ii of x (local and
//   ghost atoms) with |x[j]-x[ii]|^2 < cut_sq, cut = cutoff + skin as LAMMPS' list. Cell list: bins of edge >= cut
//   over the bounding box of x, rows stably sorted by bin; a center scans its 27 neighbor bins in fixed order, so
//   the candidate order is deterministic (it is not LAMMPS' order; edge_build keeps whatever order it gets).
//   Returns cand_ptr (ncenters+1) int32 and cand_j int32, the inputs of edge_build. Called once per neighbor
//   rebuild, outside the CUDA graph: it allocates and syncs (bin extent, candidate count).
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cmath>
#include <tuple>

namespace jet_mlip {
namespace {

// one warp per center. COUNT: ptr[ii+1] = #candidates of ii. FILL: write them at ptr[ii] + rank.
template <bool FILL>
__global__ void cand_kernel(int ncenters, const double* __restrict__ x, const int* __restrict__ bin_of,
                            const int* __restrict__ bin_start, const int* __restrict__ order, int nx, int ny, int nz,
                            double cut_sq, int* ptr, int* __restrict__ cand) {
  const int lane = threadIdx.x & 31;
  const int ii = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (ii >= ncenters) return;   // warp-uniform
  const double xi0 = x[3 * ii], xi1 = x[3 * ii + 1], xi2 = x[3 * ii + 2];
  const int b = bin_of[ii], bx = b % nx, by = (b / nx) % ny, bz = b / (nx * ny);
  int e = FILL ? ptr[ii] : 0;
  for (int dz = -1; dz <= 1; dz++) {
    const int z = bz + dz;
    if (z < 0 || z >= nz) continue;
    for (int dy = -1; dy <= 1; dy++) {
      const int y = by + dy;
      if (y < 0 || y >= ny) continue;
      for (int dx = -1; dx <= 1; dx++) {
        const int xx = bx + dx;
        if (xx < 0 || xx >= nx) continue;
        const int c = (z * ny + y) * nx + xx, end = bin_start[c + 1];
        for (int k = bin_start[c]; k < end; k += 32) {
          int j = -1;
          bool ok = false;
          if (k + lane < end) {
            j = order[k + lane];
            const double d0 = x[3 * j] - xi0, d1 = x[3 * j + 1] - xi1, d2 = x[3 * j + 2] - xi2;
            ok = j != ii && d0 * d0 + d1 * d1 + d2 * d2 < cut_sq;
          }
          const unsigned m = __ballot_sync(0xffffffffu, ok);
          if (FILL && ok) cand[e + __popc(m & ((1u << lane) - 1))] = j;
          e += __popc(m);
        }
      }
    }
  }
  if (!FILL && lane == 0) ptr[ii + 1] = e;
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> cand_build(const at::Tensor& x, int64_t ncenters, double cut_sq) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kDouble && x.dim() == 2 && x.size(1) == 3 && x.is_contiguous(),
              "x: (nall, 3) float64 contiguous CUDA");
  const c10::cuda::CUDAGuard guard(x.device());   // the kernels below go to the current device's stream
  TORCH_CHECK(ncenters >= 0 && ncenters <= x.size(0) && cut_sq > 0, "0 <= ncenters <= nall, cut_sq > 0");
  const auto i32 = x.options().dtype(at::kInt);
  at::Tensor ptr = at::zeros({ncenters + 1}, i32);
  if (ncenters == 0) return {ptr, at::empty({0}, i32)};
  const int64_t nall = x.size(0);
  // bins: edge >= cut, at most ~2 per atom (a sparse box, e.g. a molecule in vacuum, gets larger bins)
  const at::Tensor lo = std::get<0>(x.min(0)), ext = (std::get<0>(x.max(0)) - lo).cpu();
  const double* E = ext.data_ptr<double>();
  double w = std::sqrt(cut_sq);
  auto dims = [&](double w, int64_t* n) {
    for (int k = 0; k < 3; k++) n[k] = (int64_t)std::floor(E[k] / w) + 1;
    return n[0] * n[1] * n[2];
  };
  int64_t n[3];
  for (int64_t nb = dims(w, n); nb > 2 * nall + 64; nb = dims(w, n)) w *= std::cbrt((double)nb / (2 * nall + 64)) * 1.01;
  const at::Tensor cell = ((x - lo) / w).floor().to(at::kLong);
  const at::Tensor ix = cell.select(1, 0).clamp(0, n[0] - 1), iy = cell.select(1, 1).clamp(0, n[1] - 1),
                   iz = cell.select(1, 2).clamp(0, n[2] - 1);
  const at::Tensor bin = (iz * n[1] + iy) * n[0] + ix;
  const at::Tensor order = std::get<1>(bin.sort(/*stable=*/true, 0, false)).to(at::kInt);
  at::Tensor bin_start = at::zeros({n[0] * n[1] * n[2] + 1}, x.options().dtype(at::kLong));
  bin_start.narrow(0, 1, n[0] * n[1] * n[2]).copy_(at::bincount(bin, {}, n[0] * n[1] * n[2]).cumsum(0));
  const at::Tensor bin_of = bin.to(at::kInt), start32 = bin_start.to(at::kInt);

  const dim3 block(256), grid((ncenters * 32 + 255) / 256);
  auto stream = at::cuda::getCurrentCUDAStream();
  cand_kernel<false><<<grid, block, 0, stream>>>((int)ncenters, x.data_ptr<double>(), bin_of.data_ptr<int>(),
                                                 start32.data_ptr<int>(), order.data_ptr<int>(), (int)n[0],
                                                 (int)n[1], (int)n[2], cut_sq, ptr.data_ptr<int>(), nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto counts = ptr.narrow(0, 1, ncenters);
  at::cumsum_out(counts, counts, 0);
  at::Tensor cand = at::empty({ptr[ncenters].item<int>()}, i32);
  cand_kernel<true><<<grid, block, 0, stream>>>((int)ncenters, x.data_ptr<double>(), bin_of.data_ptr<int>(),
                                                start32.data_ptr<int>(), order.data_ptr<int>(), (int)n[0], (int)n[1],
                                                (int)n[2], cut_sq, ptr.data_ptr<int>(), cand.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {ptr, cand};
}

}  // namespace jet_mlip

TORCH_LIBRARY_FRAGMENT(jet_mlip, m) {
  m.def("cand_build(Tensor x, int ncenters, float cut_sq) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(jet_mlip, CUDA, m) {
  m.impl("cand_build", &jet_mlip::cand_build);
}
