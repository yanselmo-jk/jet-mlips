/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   pair_style jet/d3/kk: PairJetD3 (pair_jet_d3.cu) inside a Kokkos run.

   The plain style is a host style for LAMMPS: Kokkos copies x, v and f to
   the host before its compute() and every per-atom array back to the device
   after it (10 blocking copies per step on 10,040 atoms + ghosts, 0.6 ms on
   an H200). This style declares what it really touches: it reads x (and
   type / tag) and adds to f in the device view (k_d3_add_forces), so those
   copies are gone. The cell-order packing and, on several ranks, the MPI
   gather still run on the host, so the local positions come down by one
   copy per step; types and tags only after they changed.
------------------------------------------------------------------------- */

#include "pair_jet_d3_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "kokkos_type.h"

using namespace LAMMPS_NS;

#ifndef KOKKOS_ENABLE_CUDA
#error "pair_style jet/d3/kk needs KOKKOS with CUDA (its kernels write into the device force view)"
#endif

PairJetD3Kokkos::PairJetD3Kokkos(LAMMPS *lmp) : PairJetD3(lmp)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = Device;
  datamask_read = X_MASK | F_MASK | TYPE_MASK | TAG_MASK;
  datamask_modify = F_MASK;
}

void PairJetD3Kokkos::init_style()
{
  atomKK->sync(Host, TAG_MASK);    // PairJetD3::init_style checks the host tags on several ranks
  PairJetD3::init_style();
}

void PairJetD3Kokkos::compute(int eflag, int vflag)
{
  atomKK->sync(Host, TYPE_MASK | TAG_MASK);    // a copy only after an exchange / sort changed them on the device
  // local rows of x, device view -> host view (the base class packs from atom->x); no sync flag changes,
  // the ghost rows and every other array stay where they are
  if (atomKK->k_x.need_sync_host()) {
    const std::pair<int, int> rows(0, atom->nlocal);
    Kokkos::deep_copy(Kokkos::subview(atomKK->k_x.view_host(), rows, Kokkos::ALL),
                      Kokkos::subview(atomKK->k_x.view_device(), rows, Kokkos::ALL));
  }
  // the forces of the styles before this one must be in the view before k_d3_add_forces (own stream) adds to it
  Kokkos::fence();
  auto d_f = atomKK->k_f.view_device();
  static_assert(std::is_same<decltype(d_f)::value_type, double>::value &&
                std::is_same<decltype(d_f)::array_layout, Kokkos::LayoutRight>::value,
                "jet/d3/kk adds into the device force view as double f[n][3] (the default KOKKOS build)");
  f_dev = d_f.data();
  PairJetD3::compute(eflag, vflag);
}
