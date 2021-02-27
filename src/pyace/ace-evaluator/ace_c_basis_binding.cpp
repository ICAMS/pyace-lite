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
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>


#include "ace_types.h"

#include "ace_bbasis_spec_helper.h"
#include "ace_radial_helper.h"
#include "ace_c_basisfunction_helper.h"
#include "ace_c_basis_helper.h"


namespace py = pybind11;
using namespace std;

#include <pybind11/stl_bind.h>

PYBIND11_MAKE_OPAQUE(std::map<string, string>)


//TODO: make general
template<typename T>
py::array_t<DOUBLE_TYPE> to_numpy(const Array2D<T> &arr) {
    // Create a Python object that will free the allocated
    // memory when destroyed:
    py::capsule free_when_done(arr.get_data(), [](void *f) {});

    return py::array_t<T>(
            arr.get_shape(), // shape
            arr.get_memory_strides(), // C-style contiguous strides for double
            arr.get_data(), // the data pointer
            free_when_done
    ); // numpy array references this parent
}

template<typename T>
py::array_t<DOUBLE_TYPE> to_numpy(const Array3D<T> &arr) {
    // Create a Python object that will free the allocated
    // memory when destroyed:
    if (arr.get_data() == nullptr)
        throw invalid_argument("Array is empty, couldn't transform into numpy");

    py::capsule free_when_done(arr.get_data(), [](void *f) {});

    return py::array_t<T>(
            arr.get_shape(), // shape
            arr.get_memory_strides(), // C-style contiguous strides for double
            arr.get_data(), // the data pointer
            free_when_done
    ); // numpy array references this parent
}

//CTildeBasisSet can be directly bound
PYBIND11_MODULE(basis, m) {
    py::options options;
    options.disable_function_signatures();

    py::bind_map<std::map<string, string>>(m, "MapStringString");

//    Fexp
    m.def("Fexp", [](DOUBLE_TYPE x, DOUBLE_TYPE m) {
        double F, DF;
        Fexp(x, m, F, DF);
        return make_tuple(F, DF);
    });

    //    Fexp
    m.def("FexpShiftedScaled", [](DOUBLE_TYPE x, DOUBLE_TYPE m) {
        double F, DF;
        FexpShiftedScaled(x, m, F, DF);
        return make_tuple(F, DF);
    });

    py::class_<ACEAbstractBasisFunction>(m, "ACEAbstractBasisFunction");

    py::class_<ACEBBasisFunction, ACEAbstractBasisFunction>(m, "ACEBBasisFunction")
            .def(py::init<>())
            .def(py::init<BBasisFunctionSpecification &, bool, bool>(), py::arg("bBasisSpecification"),
                 py::arg("is_half_basis") = true, py::arg("compress") = true)
            .def("print", [](ACEBBasisFunction &func) {
                py::scoped_ostream_redirect stream(
                        std::cout,                               // std::ostream&
                        py::module::import("sys").attr("stdout") // Python output
                );
                func.print();
            })
            .def_property_readonly("ms_combs", &get_ms_combs)
            .def_readonly("num_ms_combs", &ACEBBasisFunction::num_ms_combs)
            .def_property_readonly("gen_cgs", &get_gen_cgs)
            .def_property_readonly("coeffs", &get_coeff)
            .def_property_readonly("mus", &get_mus)
            .def_property_readonly("ns", &get_ns)
            .def_property_readonly("ls", &get_ls)
            .def_property_readonly("LS", &get_LS)
            .def_property_readonly("rank", [](const ACEBBasisFunction &f) { return (int) f.rank; })
            .def_property_readonly("rankL", [](const ACEBBasisFunction &f) { return (int) f.rankL; })
            .def_readonly("ndensity", &ACEBBasisFunction::ndensity)
            .def_readonly("mu0", &ACEBBasisFunction::mu0)
            .def_readonly("is_half_ms_basis", &ACEBBasisFunction::is_half_ms_basis)
            .def_readonly("is_proxy", &ACEBBasisFunction::is_proxy)
            .def_readonly("sort_order", &ACEBBasisFunction::sort_order);

    py::class_<AbstractRadialBasis>(m, "AbstractRadialBasis")
            .def_readonly("nelements", &AbstractRadialBasis::nelements)
            .def_readonly("lmax", &AbstractRadialBasis::lmax)
            .def_readonly("nradial", &AbstractRadialBasis::nradial)
            .def_readonly("nradbase", &AbstractRadialBasis::nradbase)
            .def_readonly("deltaSplineBins", &AbstractRadialBasis::deltaSplineBins)
            .def_readonly("radbasename", &AbstractRadialBasis::radbasename)
            .def_readonly("cutoff", &AbstractRadialBasis::cutoff)
            .def_property("cut",
                          [](const AbstractRadialBasis &r) { return r.cut.to_vector(); },
                          [](AbstractRadialBasis &r, vector<vector<DOUBLE_TYPE>> val) { r.cut = val; }
            )
            .def_property("dcut",
                          [](const AbstractRadialBasis &r) { return r.dcut.to_vector(); },
                          [](AbstractRadialBasis &r, vector<vector<DOUBLE_TYPE>> val) { r.dcut = val; }
            )
            .def_property_readonly("gr", [](const AbstractRadialBasis &r) { return r.gr.to_vector(); })
            .def_property_readonly("dgr", [](const AbstractRadialBasis &r) { return r.dgr.to_vector(); })
            .def_property_readonly("fr", [](const AbstractRadialBasis &r) { return r.fr.to_vector(); })
            .def_property_readonly("dfr", [](const AbstractRadialBasis &r) { return r.dfr.to_vector(); })
            .def_readonly("cr", &AbstractRadialBasis::cr)
            .def_readonly("dcr", &AbstractRadialBasis::dcr)
            .def("setuplookupRadspline", &AbstractRadialBasis::setuplookupRadspline)
            .def("evaluate", &AbstractRadialBasis::evaluate, py::arg("r"),
                 py::arg("nradbase_c"),
                 py::arg("nradial_c"),
                 py::arg("mu_i"),
                 py::arg("mu_j"),
                 py::arg("calc_second_derivatives") = false
            );

    py::class_<ACERadialFunctions, AbstractRadialBasis>(m, "ACERadialFunctions", R"mydelimiter(
        ACERadialFunctions
    )mydelimiter")
            .def(py::init<>())
            .def_property_readonly("cheb", [](const ACERadialFunctions &r) { return r.cheb.to_vector(); })
            .def_property_readonly("dcheb", [](const ACERadialFunctions &r) { return r.dcheb.to_vector(); })
            .def_property_readonly("cheb2", [](const ACERadialFunctions &r) { return r.cheb2.to_vector(); })

            .def_property("crad",
                          [](const ACERadialFunctions &r) { return r.crad.to_vector(); },
                          [](ACERadialFunctions &r,
                             vector<vector<vector<vector<vector<DOUBLE_TYPE>>>>> val) { r.crad = val; }
            )
            .def_property("lamb",
                          [](const ACERadialFunctions &r) { return r.lambda.to_vector(); },
                          [](ACERadialFunctions &r, vector<vector<DOUBLE_TYPE>> val) { r.lambda = val; }
            )
            .def_property("prehc",
                          [](const ACERadialFunctions &r) { return r.prehc.to_vector(); },
                          [](ACERadialFunctions &r, vector<vector<DOUBLE_TYPE>> val) { r.prehc = val; }
            )
            .def_property("lambhc",
                          [](const ACERadialFunctions &r) { return r.lambdahc.to_vector(); },
                          [](ACERadialFunctions &r, vector<vector<DOUBLE_TYPE>> val) { r.lambdahc = val; }
            )

            .def("calcCheb", &ACERadialFunctions::calcCheb)
            .def("radbase", &ACERadialFunctions::radbase)
            .def("radcore", &ACERadialFunctions::radcore)
            .def("radfunc", &ACERadialFunctions::radfunc)
            .def("evaluate_range", &ACERadialFunctions::evaluate_range)
            .def_property_readonly("gr_vec", [](const ACERadialFunctions &r) { return to_numpy(r.gr_vec); })
            .def_property_readonly("dgr_vec", [](const ACERadialFunctions &r) { return to_numpy(r.dgr_vec); })
            .def_property_readonly("d2gr_vec", [](const ACERadialFunctions &r) { return to_numpy(r.d2gr_vec); })

            .def_property_readonly("fr_vec", [](const ACERadialFunctions &r) { return to_numpy(r.fr_vec); })
            .def_property_readonly("dfr_vec", [](const ACERadialFunctions &r) { return to_numpy(r.dfr_vec); })
            .def_property_readonly("d2fr_vec", [](const ACERadialFunctions &r) { return to_numpy(r.d2fr_vec); })

            .def(py::pickle(&ACERadialFunctions_getstate, &ACERadialFunctions_setstate));


    py::class_<ACECTildeBasisFunction>(m, "ACECTildeBasisFunction", R"mydelimiter(
        ACECTildeBasisFunction
    )mydelimiter")
            .def(py::init<>())
            .def_property_readonly("ms_combs", [](const ACECTildeBasisFunction &f) { return get_ms_combs(f); })
            .def_property_readonly("ctildes", [](const ACECTildeBasisFunction &f) { return get_ctildes(f); })
            .def_property_readonly("mus", [](const ACECTildeBasisFunction &f) { return get_mus(f); })
            .def_property_readonly("ns", [](const ACECTildeBasisFunction &f) { return get_ns(f); })
            .def_property_readonly("ls", [](const ACECTildeBasisFunction &f) { return get_ls(f); })
            .def_readonly("num_ms_combs", &ACECTildeBasisFunction::num_ms_combs)
            .def_property_readonly("rank", [](const ACECTildeBasisFunction &f) { return (int) f.rank; })
            .def_readonly("ndensity", &ACECTildeBasisFunction::ndensity)
            .def_readonly("mu0", &ACECTildeBasisFunction::mu0)
            .def_readonly("is_half_ms_basis", &ACECTildeBasisFunction::is_half_ms_basis)
            .def_readonly("is_proxy", &ACECTildeBasisFunction::is_proxy)
            .def(py::pickle(&ACECTildeBasisFunction_getstate, &ACECTildeBasisFunction_setstate))
            .def("print", [](ACECTildeBasisFunction &func) {
                py::scoped_ostream_redirect stream(
                        std::cout,                               // std::ostream&
                        py::module::import("sys").attr("stdout") // Python output
                );
                func.print();
            })
            ;


    py::class_<ACEBBasisSet>(m, "ACEBBasisSet")
            .def(py::init<>())
            .def(py::init<BBasisConfiguration &>(), py::arg("bBasisConfiguration"))
            .def(py::init<string>(), py::arg("yaml_file_name"))
            .def_readonly("nelements", &ACEBBasisSet::nelements)
            .def_property_readonly("elements_name", [](const ACEBBasisSet &b) {
                vector<string> res(b.nelements);
                for (int i = 0; i < b.nelements; i++)
                    res[i] = b.elements_name[i];
                return res;
            })
            .def_readonly("ndensitymax", &ACEBBasisSet::ndensitymax)
            .def_readonly("nradbase", &ACEBBasisSet::nradbase)
            .def_readonly("lmax", &ACEBBasisSet::lmax)
            .def_readonly("nradmax", &ACEBBasisSet::nradmax)
            .def_readonly("cutoffmax", &ACEBBasisSet::cutoffmax)
            .def_property_readonly("basis_rank1", &ACEBBasisSet_get_basis_rank1)
            .def_property_readonly("basis", &ACEBBasisSet_get_basis)
            .def_property("all_coeffs", &ACEBBasisSet_get_all_coeffs, &ACEBBasisSet_set_all_coeffs)
            .def_property("crad_coeffs", &ACEBBasisSet_get_crad_coeffs, &ACEBBasisSet_set_crad_coeffs)
            .def_property("basis_coeffs", &ACEBBasisSet_get_basis_coeffs, &ACEBBasisSet_set_basis_coeffs)
            .def_readonly("radial_functions", &ACEBBasisSet::radial_functions)
            .def("save", &ACEBBasisSet::save)
            .def("load", &ACEBBasisSet::load)
            .def("to_ACECTildeBasisSet", &ACEBBasisSet::to_ACECTildeBasisSet) //, py::return_value_policy::copy
            .def("initialize_basis", &ACEBBasisSet::initialize_basis)
            .def("to_BBasisConfiguration", &ACEBBasisSet::to_BBasisConfiguration)
            .def_readwrite("metadata", &ACEBBasisSet::metadata)
            .def(py::pickle(&ACEBBasisSet_getstate, &ACEBBasisSet_setstate));


    py::class_<ACECTildeBasisSet>(m, "ACECTildeBasisSet", R"mydelimiter(

    ACECTildeBasisSet class which reads in the potential file.

    Attributes
    ----------
    nelements
    ndensitymax
    nradbase
    lmax
    nradmax
    cutoffmax

    Notes
    -----
    This class contains multi-level inheritance, the base classes are
    not exposed. Values of parameters read in from the potential file
    are read-only.

    )mydelimiter")

            .def(py::init<>())
            .def(py::init<string>(), py::arg("ace_file_name"))
            .def("load", &ACECTildeBasisSet::load, R"mydelimiter(

        load a potential file.

        Parameters
        ----------
        filename : string
            name of the input potential file

        Returns
        -------
        None

        )mydelimiter")

            .def_readonly("nelements", &ACECTildeBasisSet::nelements, R"mydelimiter(
        *int*
        Number of elements
        )mydelimiter")
            .def_property_readonly("elements_name", [](const ACECTildeBasisSet &b) {
                vector<string> res(b.nelements);
                for (int i = 0; i < b.nelements; i++)
                    res[i] = b.elements_name[i];
                return res;
            })
            .def_readonly("ndensitymax", &ACECTildeBasisSet::ndensitymax)
            .def_readonly("nradbase", &ACECTildeBasisSet::nradbase)
            .def_readonly("lmax", &ACECTildeBasisSet::lmax)
            .def_readonly("nradmax", &ACECTildeBasisSet::nradmax)
            .def_readonly("cutoffmax", &ACECTildeBasisSet::cutoffmax)
            .def_property_readonly("basis_rank1", &ACECTildeBasisSet_get_basis_rank1)
            .def_property_readonly("basis", &ACECTildeBasisSet_get_basis)
            .def_readonly("radial_functions", &ACECTildeBasisSet::radial_functions)
            .def("save", &ACECTildeBasisSet::save)
            .def(py::pickle(&ACECTildeBasisSet_getstate, &ACECTildeBasisSet_setstate))
            .def_property("all_coeffs", &ACECTildeBasisSet_get_all_coeffs, &ACECTildeBasisSet_set_all_coeffs);


    py::class_<BBasisFunctionSpecification>(m, "BBasisFunctionSpecification", R"mydelimiter(
        B-basis function specification class. Example:

        BBasisFunctionSpecification(elements=["Al","Al"],  ns=[1],  ls=[0],  coeffs=[1., 2.])
        BBasisFunctionSpecification(elements=["Al","Al","Al"],  ns=[1,1,1,1],  ls=[1,1,1,1], LS=[0],  coeffs=[1., 2.])

    )mydelimiter")
            .def(py::init<>())
            .def(py::init<vector<string> &, ACEBBasisFunction &>())
            .def(py::init<vector<string>, vector<NS_TYPE>, vector<LS_TYPE>, vector<LS_TYPE>, vector<DOUBLE_TYPE> >(),
                 py::arg("elements"),
                 py::arg("ns"),
                 py::arg("ls"),
                 py::arg("LS"),
                 py::arg("coeffs"))
            .def(py::init<vector<string>, vector<NS_TYPE>, vector<LS_TYPE>, vector<DOUBLE_TYPE> >(),
                 py::arg("elements"),
                 py::arg("ns"),
                 py::arg("ls"),
                 py::arg("coeffs"))
            .def(py::init<vector<string>, vector<NS_TYPE>, vector<DOUBLE_TYPE> >(),
                 py::arg("elements"),
                 py::arg("ns"),
                 py::arg("coeffs"))
            .def_readwrite("elements", &BBasisFunctionSpecification::elements)
            .def_readwrite("ns", &BBasisFunctionSpecification::ns)
            .def_readwrite("ls", &BBasisFunctionSpecification::ls)
            .def_readwrite("LS", &BBasisFunctionSpecification::LS)
            .def_readwrite("coeffs", &BBasisFunctionSpecification::coeffs)
            .def("__repr__", &BBasisFunctionSpecification::to_string)
            .def("__eq__", [](BBasisFunctionSpecification &f1, BBasisFunctionSpecification &f2) {
                return f1 == f2;
            })
//            .def("__lt__", [](BBasisFunctionSpecification &f1, BBasisFunctionSpecification &f2) {
//                return f1.less_specification_than(f2);
//            })
//            .def("__gt__", [](BBasisFunctionSpecification &f1, BBasisFunctionSpecification &f2) {
//                return f1.great_specification_than(f2);
//            })
            .def(py::pickle(&BBasisFunctionSpecification_getstate, &BBasisFunctionSpecification_setstate));

    py::class_<BBasisFunctionsSpecificationBlock>(m, "BBasisFunctionsSpecificationBlock", R"mydelimiter(
        BBasisFunctionsSpecificationBlock
    )mydelimiter")
            .def(py::init<>())

            .def_readwrite("block_name", &BBasisFunctionsSpecificationBlock::block_name)
            .def_readwrite("rankmax", &BBasisFunctionsSpecificationBlock::rankmax)
            .def_readwrite("number_of_species", &BBasisFunctionsSpecificationBlock::number_of_species)
            .def_readwrite("elements_vec", &BBasisFunctionsSpecificationBlock::elements_vec)
            .def_readwrite("mu0", &BBasisFunctionsSpecificationBlock::mu0)
            .def_readwrite("lmaxi", &BBasisFunctionsSpecificationBlock::lmaxi)
            .def_readwrite("nradmaxi", &BBasisFunctionsSpecificationBlock::nradmaxi)
            .def_readwrite("ndensityi", &BBasisFunctionsSpecificationBlock::ndensityi)
            .def_readwrite("npoti", &BBasisFunctionsSpecificationBlock::npoti)
            .def_readwrite("fs_parameters", &BBasisFunctionsSpecificationBlock::fs_parameters)
            .def_readwrite("core_rep_parameters", &BBasisFunctionsSpecificationBlock::core_rep_parameters)

            .def_readwrite("rho_cut", &BBasisFunctionsSpecificationBlock::rho_cut)
            .def_readwrite("drho_cut", &BBasisFunctionsSpecificationBlock::drho_cut)

            .def_readwrite("rcutij", &BBasisFunctionsSpecificationBlock::rcutij)
            .def_readwrite("dcutij", &BBasisFunctionsSpecificationBlock::dcutij)

            .def_readwrite("NameOfCutoffFunctionij", &BBasisFunctionsSpecificationBlock::NameOfCutoffFunctionij)
            .def_readwrite("nradbaseij", &BBasisFunctionsSpecificationBlock::nradbaseij)
            .def_readwrite("radbase", &BBasisFunctionsSpecificationBlock::radbase)
            .def_readwrite("radparameters", &BBasisFunctionsSpecificationBlock::radparameters)
            .def_readwrite("radcoefficients", &BBasisFunctionsSpecificationBlock::radcoefficients)
            .def("get_all_coeffs", &BBasisFunctionsSpecificationBlock::get_all_coeffs)

            .def_property("funcspecs",
                          [](BBasisFunctionsSpecificationBlock &block) { return block.funcspecs; },
                          [](BBasisFunctionsSpecificationBlock &block, vector<BBasisFunctionSpecification> &new_spec) {
                              block.funcspecs = new_spec;
                              block.update_params();
                          }
            )
            .def("__repr__", &BBasisFunctionsSpecificationBlock_repr_)
            .def(py::pickle(&BBasisFunctionsSpecificationBlock_getstate, &BBasisFunctionsSpecificationBlock_setstate));


    py::class_<BBasisConfiguration>(m, "BBasisConfiguration", R"mydelimiter(
        BBasisConfiguration
    )mydelimiter")
            .def(py::init<>())
            .def(py::init<string>())
            .def_readwrite("funcspecs_blocks", &BBasisConfiguration::funcspecs_blocks)
            .def_readwrite("deltaSplineBins", &BBasisConfiguration::deltaSplineBins)
            .def("save", &BBasisConfiguration::save)
            .def("load", &BBasisConfiguration::load)
            .def("get_all_coeffs", &BBasisConfiguration::get_all_coeffs)
            .def("set_all_coeffs", &BBasisConfiguration::set_all_coeffs)
            .def_readwrite("metadata", &BBasisConfiguration::metadata)
            .def("copy", [](BBasisConfiguration &conf) { return BBasisConfiguration(conf); })
            .def("__repr__", &BBasisConfiguration_repr)
            .def_property_readonly("total_number_of_functions", [](const BBasisConfiguration &conf) {
                int total_num = 0;
                for (const auto &block: conf.funcspecs_blocks) {
                    total_num += block.funcspecs.size();
                }
                return total_num;
            })
            .def(py::pickle(&BBasisConfiguration_getstate, &BBasisConfiguration_setstate));

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
