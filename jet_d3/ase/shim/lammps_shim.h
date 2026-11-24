/* Minimal stand-in for the parts of LAMMPS that pair_jet_d3.{h,cu} use, so that the unchanged pair style compiles into
   the ASE extension (jet_d3_ase). One rank, no neighbor lists: PairJetD3 sums over all atoms itself.
   GPL-2.0 (with jet_d3/, derived from LAMMPS). Only what pair_jet_d3 touches is here. */
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mpi.h"

#define FLERR __FILE__, __LINE__

namespace LAMMPS_NS {

typedef int64_t bigint;
typedef int tagint;

// "{}" / "{:spec}" placeholders replaced in order by the streamed arguments (the format spec is ignored)
inline void shim_format_into(std::ostringstream &os, const char *f) { os << f; }
template <class T, class... R>
void shim_format_into(std::ostringstream &os, const char *f, const T &v, const R &...rest) {
    for (; *f; ++f) {
        if (*f == '{') {
            const char *e = std::strchr(f, '}');
            if (e) { os << v; shim_format_into(os, e + 1, rest...); return; }
        }
        os << *f;
    }
}
template <class... A>
std::string shim_format(const std::string &fmt, const A &...a) {
    std::ostringstream os;
    shim_format_into(os, fmt.c_str(), a...);
    return os.str();
}

class Memory {
public:
    template <class T>
    T **create(T **&a, int n1, int n2, const char *) {
        T *data = static_cast<T *>(std::calloc(static_cast<size_t>(n1) * n2, sizeof(T)));
        a = static_cast<T **>(std::malloc(sizeof(T *) * n1));
        for (int i = 0; i < n1; i++) a[i] = data + static_cast<size_t>(i) * n2;
        return a;
    }
    template <class T>
    void destroy(T **&a) {
        if (a) { std::free(a[0]); std::free(a); }
        a = nullptr;
    }
};

class Error {
public:
    template <class... A>
    [[noreturn]] void all(const char *file, int line, const std::string &fmt, const A &...a) {
        throw std::runtime_error(shim_format(fmt, a...) + " (" + file + ":" + std::to_string(line) + ")");
    }
    template <class... A>
    [[noreturn]] void one(const char *file, int line, const std::string &fmt, const A &...a) { all(file, line, fmt, a...); }
};

class Atom {
public:
    double **x = nullptr, **f = nullptr;
    int *type = nullptr;
    tagint *tag = nullptr;
    int nlocal = 0, ntypes = 0;
    bigint natoms = 0;
    int tag_consecutive() { return 1; }
};

class Domain {
public:
    double xprd = 0, yprd = 0, zprd = 0, xy = 0, xz = 0, yz = 0;
    double boxlo[3] = {0, 0, 0};
    int triclinic = 0;
    int xperiodic = 1, yperiodic = 1, zperiodic = 1;
    int periodicity[3] = {1, 1, 1};
};

class Comm {
public:
    int me = 0, nprocs = 1;
};

class Force {};
class Neighbor {};
class NeighList {};

class LAMMPS {
public:
    Memory *memory = nullptr;
    Error *error = nullptr;
    Atom *atom = nullptr;
    Domain *domain = nullptr;
    Comm *comm = nullptr;
    Force *force = nullptr;
    Neighbor *neighbor = nullptr;
    MPI_Comm world = 0;
};

class Pointers {
public:
    explicit Pointers(LAMMPS *p)
        : lmp(p), memory(p->memory), error(p->error), atom(p->atom), domain(p->domain), comm(p->comm), force(p->force),
          neighbor(p->neighbor), world(p->world) {}
    virtual ~Pointers() = default;

protected:
    LAMMPS *lmp;
    Memory *&memory;
    Error *&error;
    Atom *&atom;
    Domain *&domain;
    Comm *&comm;
    Force *&force;
    Neighbor *&neighbor;
    MPI_Comm &world;
};

class Pair : protected Pointers {
public:
    explicit Pair(LAMMPS *p) : Pointers(p) {}
    ~Pair() override = default;

    double eng_vdwl = 0.0, eng_coul = 0.0;
    double virial[6] = {0, 0, 0, 0, 0, 0};
    int allocated = 0, single_enable = 1, restartinfo = 1, one_coeff = 0, manybody_flag = 0;
    int no_virial_fdotr_compute = 0;
    int eflag_atom = 0, vflag_atom = 0, eflag_global = 0, vflag_global = 0;
    int **setflag = nullptr;
    double **cutsq = nullptr;

    virtual void compute(int, int) = 0;
    virtual void finish() {}
    virtual void *extract(const char *, int &) { return nullptr; }
    virtual void settings(int, char **) = 0;
    virtual void coeff(int, char **) = 0;
    virtual double init_one(int, int) { return 0.0; }
    virtual void init_style() {}
    virtual void write_restart(FILE *) {}
    virtual void read_restart(FILE *) {}
    virtual void write_restart_settings(FILE *) {}
    virtual void read_restart_settings(FILE *) {}

protected:
    void ev_init(int eflag, int vflag) {
        eflag_global = eflag; vflag_global = vflag;
        eflag_atom = vflag_atom = 0;
        eng_vdwl = eng_coul = 0.0;
        for (double &v : virial) v = 0.0;
    }
};

namespace utils {
// a finite number filling the whole string (as LAMMPS' utils::numeric), else std::invalid_argument
inline double numeric(const char *, int, const std::string &s, bool, LAMMPS *) {
    size_t used = 0;
    const double v = std::stod(s, &used);
    if (used != s.size() || !std::isfinite(v)) throw std::invalid_argument("expected a finite number, got '" + s + "'");
    return v;
}
template <class... A>
void logmesg(LAMMPS *, const std::string &fmt, const A &...a) { std::fputs(shim_format(fmt, a...).c_str(), stderr); }
}  // namespace utils

namespace MathExtra {
inline void cross3(const double *a, const double *b, double *c) {
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
}
inline double dot3(const double *a, const double *b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline double len3(const double *a) { return std::sqrt(dot3(a, a)); }
}  // namespace MathExtra

}  // namespace LAMMPS_NS
