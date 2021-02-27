/*
 * pyace: atomic cluster expansion and its python bindings
 *
 * Copyright 2021  (c) Yury Lysogorskiy, Sarath Menon,
 * Anton Bochkarev, Ralf Drautz
 *
 * Ruhr-University Bochum, Bochum, Germany
 *
 * See the LICENSE file.
 * This FILENAME is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <vector>
#include <string>
#include <cmath>

#include "ace_types.h"
#include "ace_arraynd.h"
#include "ace_atoms.h"
#include "ace_utils.h"

namespace py = pybind11;
using namespace std;

/*
We could add a pure python version of this class to use it alongside
ase and other tools
*/

string ACEAtomicEnvironment__repr__(ACEAtomicEnvironment& ae){
    stringstream s;
    s << "ACEAtomicEnvironment(n_atoms_real="<<ae.n_atoms_real << ", "\
    << "n_atoms_extended=" << ae.n_atoms_extended << ""\
    << ")";
    return s.str();
}

pybind11::tuple ACEAtomicEnvironment__getstate__(const ACEAtomicEnvironment &ae) {
    return py::make_tuple(ae.get_x(), ae.get_species_types(), ae.get_neighbour_list(), ae.get_origins());
}

ACEAtomicEnvironment ACEAtomicEnvironment__setstate__(py::tuple t) {
    if (t.size() != 4)
        throw std::runtime_error("Invalid state of ACEAtomicEnvironment-tuple");
    ACEAtomicEnvironment ae;
    auto new_x = t[0].cast<vector<vector<DOUBLE_TYPE>>>();
    auto new_species_types = t[1].cast<vector<SPECIES_TYPE>>();
    auto new_neighbour_list = t[2].cast<vector<vector<int>>>();
    auto new_origins = t[3].cast<vector<int>>();

    ae.set_x(new_x);
    ae.set_species_types(new_species_types);
    ae.set_neighbour_list(new_neighbour_list);
    ae.set_origins(new_origins);

    return ae;
}

PYBIND11_MODULE(catomicenvironment, m) {
    py::options options;
    options.disable_function_signatures();

py::class_<ACEAtomicEnvironment>(m,"ACEAtomicEnvironment", R"mydelimiter(

    Atomic environment class

    Attributes
    ----------
    n_atoms_real
    n_atoms_extended
    x
    species_type
    neighbour_list
    )mydelimiter")
        .def(py::init())
        .def(py::init<int>())
        .def_readwrite("n_atoms_real", &ACEAtomicEnvironment::n_atoms_real)
        .def_readwrite("n_atoms_extended", &ACEAtomicEnvironment::n_atoms_extended)
        .def_property("x", &ACEAtomicEnvironment::get_x, &ACEAtomicEnvironment::set_x)
        .def_property("species_type", &ACEAtomicEnvironment::get_species_types,
                      &ACEAtomicEnvironment::set_species_types)
        .def_property("neighbour_list", &ACEAtomicEnvironment::get_neighbour_list,
                      &ACEAtomicEnvironment::set_neighbour_list)
        .def_property("origins", &ACEAtomicEnvironment::get_origins, &ACEAtomicEnvironment::set_origins)
        .def("__repr__", &ACEAtomicEnvironment__repr__)
        .def("load_full", &ACEAtomicEnvironment::load_full)
        .def("save_full", &ACEAtomicEnvironment::save_full)
    .def(py::pickle( &ACEAtomicEnvironment__getstate__,&ACEAtomicEnvironment__setstate__))
    ;


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
