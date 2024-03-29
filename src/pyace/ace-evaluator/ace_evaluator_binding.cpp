#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <vector>
#include <string>

#include "ace_arraynd.h"
#include "ace_timing.h"
#include "ace_evaluator.h"
#include "ace_b_evaluator.h"

namespace py = pybind11;
using namespace std;

/*
ACEEvaluator has a derived ACECTildeEvaluator, both are bound to make pybind11
aware of the relationship.

NOte: There are many virtual overrides - one must be careful if these need
to be exposed. Then trampoline classes are need to properly do this. For now,
since the overridden methods are not used, it is fine.
*/
PYBIND11_MODULE(evaluator, m) {
    py::options options;
    options.disable_function_signatures();

//base class bindings
py::class_<ACEEvaluator> (m,"ACEEvaluator", R"mydelimiter(

    )mydelimiter")
    ;



//derived class
    py::class_<ACEBEvaluator, ACEEvaluator>(m,"ACEBEvaluator", R"mydelimiter(

    )mydelimiter")
            .def(py::init <>())
            .def(py::init <ACEBBasisSet&>(), py::arg("bBasisSet"))
            .def(py::init <BBasisConfiguration&>(), py::arg("bBasisConfiguration"))
            .def("set_basis", &ACEBEvaluator::set_basis, R"mydelimiter(

    Set a basis to the evaluator

    Parameters
    ----------
    basis : ACECTildeBasisSet object

    Returns
    -------
    None

    )mydelimiter")
            .def_property("element_type_mapping",
                          [](const ACEBEvaluator &e) { return e.element_type_mapping.to_vector(); },
                          [](ACEBEvaluator &e, vector<int> v) { e.element_type_mapping = v; })
            ;


//derived class
py::class_<ACECTildeEvaluator, ACEEvaluator>(m,"ACECTildeEvaluator", R"mydelimiter(

    )mydelimiter")
    .def(py::init <>())
    .def(py::init <ACECTildeBasisSet&>(), py::arg("cTildeBasisSet"))
    .def("set_basis", &ACECTildeEvaluator::set_basis, R"mydelimiter(

    Set a basis to the evaluator

    Parameters
    ----------
    basis : ACECTildeBasisSet object

    Returns
    -------
    None

    )mydelimiter")
        .def_property("element_type_mapping",
                      [](const ACECTildeEvaluator &e) { return e.element_type_mapping.to_vector(); },
                      [](ACECTildeEvaluator &e, vector<int> v) { e.element_type_mapping = v; })
    ;


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}

