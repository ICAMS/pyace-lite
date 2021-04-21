#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <vector>
#include <string>

#include "ace_atoms.h"
#include "ace_evaluator.h"
#include "ace_calculator.h"
//#include "multiarray_converter.h"

namespace py = pybind11;
using namespace std;

vector<vector<DOUBLE_TYPE>> ACECalculator_get_forces(ACECalculator& calc) {
    //return convert_array2d_to_vec_double(calc.forces);
    return calc.forces.to_vector();
}
//bindings
PYBIND11_MODULE(calculator, m) {
    py::options options;
    options.disable_function_signatures();

//NOTE: Although the derived pyace calculator is called PyACECalculator,
//it is exposed to the python space as ACECalculator
py::class_<ACECalculator>(m, "ACECalculator", R"mydelimiter(

    )mydelimiter")
        .def(py::init<>())
        .def(py::init<ACEEvaluator &>(), py::arg("ACEEvaluator"))
        .def("set_evaluator", &ACECalculator::set_evaluator)
        .def("compute", &ACECalculator::compute, py::arg("atomic_environment"), py::arg("verbose") = false)
        .def_property_readonly("forces", &ACECalculator_get_forces)
        .def_readonly("energy", &ACECalculator::energy)
        .def_property_readonly("energies", [](const ACECalculator &calc) { return calc.energies.to_vector(); })
        .def_property_readonly("virial", [](const ACECalculator &calc) { return calc.virial.to_vector(); })
    .def_readonly("basis_projections_rank1", &ACECalculator::basis_peratom_projections_rank1)
    .def_readonly("basis_projections", &ACECalculator::basis_peratom_projections)
    ;


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
