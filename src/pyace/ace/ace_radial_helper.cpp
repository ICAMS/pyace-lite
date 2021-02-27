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

//
// Created by Lysogorskiy Yury on 11.05.2020.
//
#include "ace_radial_helper.h"

#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include "ace_radial.h"
#include "ace_types.h"



namespace py = pybind11;
using namespace std;


py::tuple ACERadialFunctions_getstate(const AbstractRadialBasis *radial_functions) {
    return py::make_tuple(
            radial_functions->nradbase,              //0
            radial_functions->lmax,                  //1
            radial_functions->nradial,               //2
            radial_functions->deltaSplineBins,                  //3
            radial_functions->nelements,             //4
            radial_functions->cutoff,                //5
            radial_functions->prehc.to_vector(),     //6
            radial_functions->lambdahc.to_vector(),  //7
            radial_functions->lambda.to_vector(),    //8
            radial_functions->cut.to_vector(),       //9
            radial_functions->dcut.to_vector(),      //10
            radial_functions->crad.to_vector(),       //11
            radial_functions->radbasename            //12
    );
}

ACERadialFunctions *ACERadialFunctions_setstate(const py::tuple &t) {
    if (t.size() != 13)
        throw std::runtime_error("Invalid state of ACECTildeBasisSet-tuple");

    NS_TYPE nradbase = t[0].cast<NS_TYPE>();        //0
    LS_TYPE lmax = t[1].cast<LS_TYPE>();                 //1
    NS_TYPE nradial = t[2].cast<NS_TYPE>();               //2
    DOUBLE_TYPE deltaSplineBins = t[3].cast<DOUBLE_TYPE>();                  //3
    SPECIES_TYPE nelements = t[4].cast<SPECIES_TYPE>();             //4
    DOUBLE_TYPE cutoff = t[5].cast<DOUBLE_TYPE>();                //5

    auto prehc = t[6].cast<vector<vector<DOUBLE_TYPE>>>();     //6
    auto lambdahc = t[7].cast<vector<vector<DOUBLE_TYPE>>>();  //7
    auto lambda = t[8].cast<vector<vector<DOUBLE_TYPE>>>();    //8
    auto cut = t[9].cast<vector<vector<DOUBLE_TYPE>>>();//9
    auto dcut = t[10].cast<vector<vector<DOUBLE_TYPE>>>();//10
    auto crad = t[11].cast<vector<vector<vector<vector<vector<DOUBLE_TYPE>>>>>>();       //11
    auto radbasename = t[12].cast<string>();            // 12
    ACERadialFunctions *radial_functions = new ACERadialFunctions(nradbase, lmax, nradial, deltaSplineBins, nelements,
                                                                  cutoff,
                                                                  radbasename);

    radial_functions->prehc = prehc;
    radial_functions->lambdahc = lambdahc;
    radial_functions->lambda = lambda;
    radial_functions->cut = cut;
    radial_functions->dcut = dcut;
    radial_functions->crad = crad;
    radial_functions->setuplookupRadspline();
    return radial_functions;
}