/* jet_d3_ase._C: the LAMMPS pair style jet/d3 (jet_d3/pair_jet_d3.cu, compiled unchanged against the stand-in in shim/)
   driven from Python for one rank. GPL-2.0, like jet_d3/.

   D3(settings, elements): settings = the pair_style arguments after "jet/d3" (e.g. 9000 1600 damp_bj pbe),
   elements = the element of each LAMMPS type, in type order (pair_coeff * * <elements>).
   compute(x, type, box, periodic) with x (N, 3) in Angstrom in the LAMMPS frame (restricted triclinic),
   type (N,) 1-based, box = (xprd, yprd, zprd, xy, xz, yz, xlo, ylo, zlo), periodic = 3 flags
   -> (energy eV, forces (N, 3) eV/A, virial (xx, yy, zz, xy, xz, yz) eV), as LAMMPS' eng_vdwl / f / virial. */
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

#include "pair_jet_d3.h"

namespace py = pybind11;
using namespace LAMMPS_NS;

class D3 {
public:
    D3(const std::vector<std::string> &settings, const std::vector<std::string> &elements) {
        lmp.memory = &memory; lmp.error = &error; lmp.atom = &atom; lmp.domain = &domain; lmp.comm = &comm;
        lmp.force = &force; lmp.neighbor = &neighbor;
        atom.ntypes = static_cast<int>(elements.size());
        pair = std::make_unique<PairJetD3>(&lmp);
        std::vector<std::string> s = settings;
        call(s, [&](int n, char **a) { pair->settings(n, a); });
        std::vector<std::string> c = {"*", "*"};
        c.insert(c.end(), elements.begin(), elements.end());
        call(c, [&](int n, char **a) { pair->coeff(n, a); });
        pair->init_style();
    }

    py::tuple compute(py::array_t<double, py::array::c_style | py::array::forcecast> x,
                      py::array_t<int, py::array::c_style | py::array::forcecast> type, std::vector<double> box,
                      std::vector<int> periodic) {
        if (x.ndim() != 2 || x.shape(1) != 3) throw std::invalid_argument("x must be (N, 3)");
        const int n = static_cast<int>(x.shape(0));
        if (type.ndim() != 1 || type.shape(0) != n) throw std::invalid_argument("type must be (N,)");
        if (box.size() != 9 || periodic.size() != 3) throw std::invalid_argument("box needs 9 values, periodic 3");
        for (double b : box) if (!std::isfinite(b)) throw std::invalid_argument("box values must be finite");
        if (!(box[0] > 0 && box[1] > 0 && box[2] > 0)) throw std::invalid_argument("box lengths must be positive");
        const int *t = type.data();
        for (int i = 0; i < n; i++) {
            if (t[i] < 1 || t[i] > atom.ntypes) throw std::invalid_argument("type out of range 1..ntypes");
        }
        domain.xprd = box[0]; domain.yprd = box[1]; domain.zprd = box[2];
        domain.xy = box[3]; domain.xz = box[4]; domain.yz = box[5];
        for (int k = 0; k < 3; k++) { domain.boxlo[k] = box[6 + k]; domain.periodicity[k] = periodic[k] ? 1 : 0; }
        domain.xperiodic = domain.periodicity[0]; domain.yperiodic = domain.periodicity[1]; domain.zperiodic = domain.periodicity[2];
        domain.triclinic = (box[3] != 0.0 || box[4] != 0.0 || box[5] != 0.0) ? 1 : 0;

        xs.assign(x.data(), x.data() + 3 * static_cast<size_t>(n));
        fs.assign(3 * static_cast<size_t>(n), 0.0);
        types.assign(t, t + n);
        tags.resize(n);
        xrows.resize(n); frows.resize(n);
        for (int i = 0; i < n; i++) { xrows[i] = &xs[3 * i]; frows[i] = &fs[3 * i]; tags[i] = i + 1; }
        atom.x = xrows.data(); atom.f = frows.data(); atom.type = types.data(); atom.tag = tags.data();
        atom.nlocal = n; atom.natoms = n;

        pair->compute(1, 1);

        py::array_t<double> f({n, 3});
        std::memcpy(f.mutable_data(), fs.data(), fs.size() * sizeof(double));
        py::array_t<double> v(6);
        std::memcpy(v.mutable_data(), pair->virial, 6 * sizeof(double));
        return py::make_tuple(pair->eng_vdwl, f, v);
    }

private:
    template <class F>
    static void call(std::vector<std::string> &args, F &&f) {
        std::vector<char *> a;
        for (std::string &s : args) a.push_back(s.data());
        f(static_cast<int>(a.size()), a.data());
    }

    Memory memory; Error error; Atom atom; Domain domain; Comm comm; Force force; Neighbor neighbor;
    LAMMPS lmp;
    std::unique_ptr<PairJetD3> pair;
    std::vector<double> xs, fs;
    std::vector<double *> xrows, frows;
    std::vector<int> types;
    std::vector<tagint> tags;
};

PYBIND11_MODULE(_C, m) {
    m.doc() = "jet/d3 (LAMMPS pair style, GPU D3 dispersion) for one process; GPL-2.0";
    py::class_<D3>(m, "D3")
        .def(py::init<const std::vector<std::string> &, const std::vector<std::string> &>(), py::arg("settings"),
             py::arg("elements"))
        .def("compute", &D3::compute, py::arg("x"), py::arg("type"), py::arg("box"), py::arg("periodic"));
}
