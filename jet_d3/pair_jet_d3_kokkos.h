/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(jet/d3/kk,PairJetD3Kokkos);
PairStyle(jet/d3/kk/device,PairJetD3Kokkos);
// clang-format on
#else

#ifndef LMP_PAIR_JET_D3_KOKKOS_H
#define LMP_PAIR_JET_D3_KOKKOS_H

#include "pair_jet_d3.h"

namespace LAMMPS_NS {

// jet/d3 on Kokkos device data (KOKKOS with CUDA): the same kernels, the forces are added in the device force view,
// so a Kokkos run (alone or under hybrid/overlay with mliap/kk) keeps its per-atom arrays on the device.
class PairJetD3Kokkos : public PairJetD3 {
 public:
  PairJetD3Kokkos(class LAMMPS *);
  void init_style() override;
  void compute(int, int) override;
};

}    // namespace LAMMPS_NS

#endif
#endif
