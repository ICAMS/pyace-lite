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

#ifndef PYACE_ACE_C_BASISFUNCTION_HELPER_H
#define PYACE_ACE_C_BASISFUNCTION_HELPER_H
#include <pybind11/pybind11.h>

#include "ace_c_basisfunction.h"
#include "ace_b_basisfunction.h"

namespace py = pybind11;
using namespace std;

vector<SPECIES_TYPE> get_mus(const ACEAbstractBasisFunction &func );
vector<NS_TYPE> get_ns(const ACEAbstractBasisFunction &func );
vector<LS_TYPE> get_ls(const ACEAbstractBasisFunction &func );
vector<vector<MS_TYPE>> get_ms_combs(const ACEAbstractBasisFunction &func );

vector<DOUBLE_TYPE> get_gen_cgs(const ACEBBasisFunction &func );
vector<DOUBLE_TYPE> get_coeff(const ACEBBasisFunction &func );
vector<LS_TYPE> get_LS(const ACEBBasisFunction &func );

vector<vector<DOUBLE_TYPE>> get_ctildes(const ACECTildeBasisFunction &func );

py::tuple ACECTildeBasisFunction_getstate(const ACECTildeBasisFunction &func);
ACECTildeBasisFunction ACECTildeBasisFunction_setstate(const py::tuple &tuple);



#endif //PYACE_ACE_C_BASISFUNCTION_HELPER_H
