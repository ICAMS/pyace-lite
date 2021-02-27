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

#ifndef PYACE_ACE_C_BASIS_HELPER_H
#define PYACE_ACE_C_BASIS_HELPER_H

#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ace_c_basisfunction.h"
#include "ace_c_basis.h"
#include "ace_b_basis.h"

namespace py = pybind11;
using namespace std;

vector<DOUBLE_TYPE> ACEBBasisSet_get_crad_coeffs(const ACEBBasisSet &basis);
vector<DOUBLE_TYPE> ACEBBasisSet_get_basis_coeffs(const ACEBBasisSet &basis);
vector<DOUBLE_TYPE> ACEBBasisSet_get_all_coeffs(const ACEBBasisSet& basis);

void ACEBBasisSet_set_crad_coeffs(ACEBBasisSet& basis, const vector<DOUBLE_TYPE>& crad_flatten_coeffs);
void ACEBBasisSet_set_basis_coeffs(ACEBBasisSet& basis, const vector<DOUBLE_TYPE>& basis_coeffs_vector);
void ACEBBasisSet_set_all_coeffs(ACEBBasisSet& basis, const vector<DOUBLE_TYPE>& coeffs);

vector<vector<ACEBBasisFunction>> ACEBBasisSet_get_basis_rank1(const ACEBBasisSet& basis);
vector<vector<ACEBBasisFunction>> ACEBBasisSet_get_basis(const ACEBBasisSet& basis);


vector<DOUBLE_TYPE> ACECTildeBasisSet_get_all_coeffs(const ACECTildeBasisSet &basis);
void ACECTildeBasisSet_set_all_coeffs(ACECTildeBasisSet &basis, const vector<DOUBLE_TYPE> &coeffs);
vector<vector<ACECTildeBasisFunction>> ACECTildeBasisSet_get_basis_rank1(const ACECTildeBasisSet& basis);
vector<vector<ACECTildeBasisFunction>> ACECTildeBasisSet_get_basis(const ACECTildeBasisSet& basis);
py::tuple ACECTildeBasisSet_getstate(const ACECTildeBasisSet &cbasisSet);
ACECTildeBasisSet ACECTildeBasisSet_setstate(const py::tuple &t);


#endif //PYACE_ACE_C_BASIS_HELPER_H
