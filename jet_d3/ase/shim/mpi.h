/* One-rank MPI stand-in for pair_jet_d3 outside LAMMPS (GPL-2.0, with jet_d3/). PairJetD3 calls these only on several
   ranks (comm->nprocs > 1), which the ASE extension never has; they still have one-rank semantics. */
#pragma once

#include <cstring>

typedef int MPI_Comm;
typedef int MPI_Datatype;
typedef int MPI_Op;
#define MPI_INT 1
#define MPI_DOUBLE 2
#define MPI_SUM 1
#define MPI_MIN 2
#define MPI_MAX 3
#define MPI_IN_PLACE (reinterpret_cast<void *>(1))

inline int shim_mpi_size(MPI_Datatype t) { return t == MPI_DOUBLE ? 8 : 4; }
inline int MPI_Allgather(const void *s, int n, MPI_Datatype t, void *r, int, MPI_Datatype, MPI_Comm) {
    std::memcpy(r, s, static_cast<size_t>(n) * shim_mpi_size(t));
    return 0;
}
inline int MPI_Allgatherv(const void *s, int n, MPI_Datatype t, void *r, const int *, const int *d, MPI_Datatype, MPI_Comm) {
    std::memcpy(static_cast<char *>(r) + static_cast<size_t>(d[0]) * shim_mpi_size(t), s, static_cast<size_t>(n) * shim_mpi_size(t));
    return 0;
}
inline int MPI_Allreduce(const void *s, void *r, int n, MPI_Datatype t, MPI_Op, MPI_Comm) {
    if (s != MPI_IN_PLACE) std::memcpy(r, s, static_cast<size_t>(n) * shim_mpi_size(t));
    return 0;
}
inline int MPI_Reduce(const void *s, void *r, int n, MPI_Datatype t, MPI_Op, int, MPI_Comm) {
    if (s != MPI_IN_PLACE) std::memcpy(r, s, static_cast<size_t>(n) * shim_mpi_size(t));
    return 0;
}
