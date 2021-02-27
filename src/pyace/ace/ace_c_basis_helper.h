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
