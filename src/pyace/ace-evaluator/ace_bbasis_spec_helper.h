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

#ifndef PYACE_ACE_BBASIS_FUNC_SPEC_HELPER_H
#define PYACE_ACE_BBASIS_FUNC_SPEC_HELPER_H
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ace_b_basis.h"
#include "ace_utils.h"

namespace py = pybind11;
using namespace std;

string BBasisFunctionsSpecificationBlock_repr_(const BBasisFunctionsSpecificationBlock &block);

string BBasisConfiguration_repr(BBasisConfiguration &config);

py::tuple BBasisFunctionSpecification_getstate(const BBasisFunctionSpecification &spec);

BBasisFunctionSpecification BBasisFunctionSpecification_setstate(const py::tuple &t);

py::tuple BBasisFunctionsSpecificationBlock_getstate(const BBasisFunctionsSpecificationBlock &block);

BBasisFunctionsSpecificationBlock BBasisFunctionsSpecificationBlock_setstate(const py::tuple &tuple);

py::tuple BBasisConfiguration_getstate(const BBasisConfiguration &config);

BBasisConfiguration BBasisConfiguration_setstate(const py::tuple &tuple);


// ACEBBasisSet pickling
py::tuple ACEBBasisSet_getstate(const ACEBBasisSet &bbasisSet);

ACEBBasisSet ACEBBasisSet_setstate(const py::tuple &tuple);

#endif //PYACE_ACE_BBASIS_FUNC_SPEC_HELPER_H
