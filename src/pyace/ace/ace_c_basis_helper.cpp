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
#include "ace_c_basis_helper.h"
#include "ace_b_basisfunction.h"
#include "ace_radial_helper.h"


namespace py = pybind11;
using namespace std;

vector<DOUBLE_TYPE> ACEBBasisSet_get_crad_coeffs(const ACEBBasisSet &basis) {
    auto coeffs = basis.radial_functions->crad.to_flatten_vector();
    return coeffs;
}

vector<DOUBLE_TYPE> ACEBBasisSet_get_basis_coeffs(const ACEBBasisSet &basis) {
    vector<DOUBLE_TYPE> coeffs;
    for (SPECIES_TYPE mu = 0; mu < basis.nelements; mu++) {
        for (SHORT_INT_TYPE func_ind = 0; func_ind < basis.total_basis_size_rank1[mu]; func_ind++) {
            for (DENSITY_TYPE p = 0; p < basis.basis_rank1[mu][func_ind].ndensity; p++)
                coeffs.emplace_back(basis.basis_rank1[mu][func_ind].coeff[p]);
        }

        for (SHORT_INT_TYPE func_ind = 0; func_ind < basis.total_basis_size[mu]; func_ind++) {
            for (DENSITY_TYPE p = 0; p < basis.basis[mu][func_ind].ndensity; p++)
                coeffs.emplace_back(basis.basis[mu][func_ind].coeff[p]);
        }
    }

    return coeffs;
}



vector<DOUBLE_TYPE> ACEBBasisSet_get_all_coeffs(const ACEBBasisSet &basis) {
    auto cradCoeffs = ACEBBasisSet_get_crad_coeffs(basis);
    auto basisCoeffs = ACEBBasisSet_get_basis_coeffs(basis);

    vector<DOUBLE_TYPE> coeffs;
    coeffs.reserve( cradCoeffs.size() + basisCoeffs.size());
    coeffs.insert( coeffs.end(), cradCoeffs.begin(), cradCoeffs.end() );
    coeffs.insert( coeffs.end(), basisCoeffs.begin(), basisCoeffs.end() );

    return coeffs;
}

void ACEBBasisSet_set_crad_coeffs(ACEBBasisSet &basis, const vector<DOUBLE_TYPE> &crad_flatten_coeffs) {
    basis.radial_functions->crad = crad_flatten_coeffs;
    basis.radial_functions->setuplookupRadspline();
}

void ACEBBasisSet_set_basis_coeffs(ACEBBasisSet &basis, const vector<DOUBLE_TYPE> &basis_coeffs_vector) {
    size_t coeffs_ind = 0;
    size_t sequential_func_ind = 0;
    for (SPECIES_TYPE mu = 0; mu < basis.nelements; mu++) {
        for (SHORT_INT_TYPE func_ind = 0; func_ind < basis.total_basis_size_rank1[mu]; func_ind++, sequential_func_ind++) {
            for (DENSITY_TYPE p = 0; p < basis.basis_rank1[mu][func_ind].ndensity; p++, coeffs_ind++) {
                basis.basis_rank1[mu][func_ind].coeff[p] = basis_coeffs_vector[coeffs_ind];
                //update also mu0_bbasis_vector for consistency
                basis.mu0_bbasis_vector[mu][sequential_func_ind].coeff[p] = basis_coeffs_vector[coeffs_ind];
            }
        }

        for (SHORT_INT_TYPE func_ind = 0; func_ind < basis.total_basis_size[mu]; func_ind++, sequential_func_ind++) {
            for (DENSITY_TYPE p = 0; p < basis.basis[mu][func_ind].ndensity; p++, coeffs_ind++) {
                basis.basis[mu][func_ind].coeff[p] = basis_coeffs_vector[coeffs_ind];
                //update also mu0_bbasis_vector for consistency
                basis.mu0_bbasis_vector[mu][sequential_func_ind].coeff[p] = basis_coeffs_vector[coeffs_ind];
            }
        }
    }
}

void ACEBBasisSet_set_all_coeffs(ACEBBasisSet &basis, const vector<DOUBLE_TYPE> &coeffs) {
    size_t crad_size = basis.radial_functions->crad.get_size();

    vector<DOUBLE_TYPE> crad_flatten_vector(coeffs.begin(), coeffs.begin() + crad_size);
    vector<DOUBLE_TYPE> basis_coeffs_vector(coeffs.begin() + crad_size, coeffs.end());

    ACEBBasisSet_set_crad_coeffs(basis, crad_flatten_vector);

    ACEBBasisSet_set_basis_coeffs(basis, basis_coeffs_vector);
}


vector<DOUBLE_TYPE> ACECTildeBasisSet_get_all_coeffs(const ACECTildeBasisSet &basis) {
    auto coeffs = basis.radial_functions->crad.to_flatten_vector();

    for (SPECIES_TYPE mu = 0; mu < basis.nelements; mu++) {
        for (SHORT_INT_TYPE func_ind = 0; func_ind < basis.total_basis_size_rank1[mu]; func_ind++) {
            auto ndens = basis.basis_rank1[mu][func_ind].ndensity;
            for (SHORT_INT_TYPE ms_ind = 0; ms_ind < basis.basis_rank1[mu][func_ind].num_ms_combs; ms_ind++) {
                for (DENSITY_TYPE p = 0; p < ndens; p++)
                    coeffs.emplace_back(basis.basis_rank1[mu][func_ind].ctildes[ms_ind * ndens + p]);
            }
        }

        for (SHORT_INT_TYPE func_ind = 0; func_ind < basis.total_basis_size[mu]; func_ind++) {
            auto ndens = basis.basis[mu][func_ind].ndensity;
            for (SHORT_INT_TYPE ms_ind = 0; ms_ind < basis.basis[mu][func_ind].num_ms_combs; ms_ind++) {
                for (DENSITY_TYPE p = 0; p < ndens; p++)
                    coeffs.emplace_back(basis.basis[mu][func_ind].ctildes[ms_ind * ndens + p]);
            }
        }
    }

    return coeffs;
}

void ACECTildeBasisSet_set_all_coeffs(ACECTildeBasisSet &basis, const vector<DOUBLE_TYPE> &coeffs) {
    size_t crad_size = basis.radial_functions->crad.get_size();
    vector<DOUBLE_TYPE> crad_flatten_vector(coeffs.begin(), coeffs.begin() + crad_size);
    vector<DOUBLE_TYPE> basis_coeffs_vector(coeffs.begin() + crad_size, coeffs.end());

    basis.radial_functions->crad = crad_flatten_vector;
    basis.radial_functions->setuplookupRadspline();

    size_t coeffs_ind = 0;
    for (SPECIES_TYPE mu = 0; mu < basis.nelements; mu++) {
        for (SHORT_INT_TYPE func_ind = 0; func_ind < basis.total_basis_size_rank1[mu]; func_ind++) {
            auto ndens = basis.basis_rank1[mu][func_ind].ndensity;
            for (SHORT_INT_TYPE ms_ind = 0; ms_ind < basis.basis_rank1[mu][func_ind].num_ms_combs; ms_ind++) {
                for (DENSITY_TYPE p = 0; p < ndens; p++, coeffs_ind++) {
                    basis.basis_rank1[mu][func_ind].ctildes[ms_ind * ndens + p] = basis_coeffs_vector[coeffs_ind];
                }
            }
        }

        for (SHORT_INT_TYPE func_ind = 0; func_ind < basis.total_basis_size[mu]; func_ind++) {
            auto ndens = basis.basis[mu][func_ind].ndensity;
            for (SHORT_INT_TYPE ms_ind = 0; ms_ind < basis.basis[mu][func_ind].num_ms_combs; ms_ind++) {
                for (DENSITY_TYPE p = 0; p < ndens; p++, coeffs_ind++) {
                    basis.basis[mu][func_ind].ctildes[ms_ind * ndens + p] = basis_coeffs_vector[coeffs_ind];
                }
            }
        }
    }
}


vector<vector<ACEBBasisFunction>> ACEBBasisSet_get_basis_rank1(const ACEBBasisSet &basis) {
    vector<vector<ACEBBasisFunction>> res;
    res.resize(basis.nelements);
    for (SPECIES_TYPE mu = 0; mu < basis.nelements; mu++) {
        SHORT_INT_TYPE size = basis.total_basis_size_rank1[mu];
        res[mu].resize(size);
        for (SHORT_INT_TYPE func_ind = 0; func_ind < size; func_ind++) {
            //turn off the proxying for proper copying when export to Python
            bool old_proxy = basis.basis_rank1[mu][func_ind].is_proxy;
            basis.basis_rank1[mu][func_ind].is_proxy = false;
            res[mu][func_ind] = basis.basis_rank1[mu][func_ind];
            basis.basis_rank1[mu][func_ind].is_proxy = old_proxy;
        }
    }
    return res;
}

vector<vector<ACEBBasisFunction>> ACEBBasisSet_get_basis(const ACEBBasisSet &basis) {
    vector<vector<ACEBBasisFunction>> res;
    res.resize(basis.nelements);
    for (SPECIES_TYPE mu = 0; mu < basis.nelements; mu++) {
        SHORT_INT_TYPE size = basis.total_basis_size[mu];
        res[mu].resize(size);
        for (SHORT_INT_TYPE t = 0; t < size; t++) {
            //turn off the proxying for proper copying when export to Python
            bool old_proxy = basis.basis[mu][t].is_proxy;
            basis.basis[mu][t].is_proxy = false;
            res[mu][t] = basis.basis[mu][t];
            basis.basis[mu][t].is_proxy = old_proxy;
        }
    }
    return res;
}


vector<vector<ACECTildeBasisFunction>> ACECTildeBasisSet_get_basis_rank1(const ACECTildeBasisSet &basis) {
    vector<vector<ACECTildeBasisFunction>> res;

    res.resize(basis.nelements);
    for (SPECIES_TYPE mu = 0; mu < basis.nelements; mu++) {
        SHORT_INT_TYPE size = basis.total_basis_size_rank1[mu];
        res[mu].resize(size);
        for (SHORT_INT_TYPE t = 0; t < size; t++) {
            //turn off the proxying for proper copying when export to Python
            bool old_proxy = basis.basis_rank1[mu][t].is_proxy;
            basis.basis_rank1[mu][t].is_proxy = false;
            res[mu][t] = basis.basis_rank1[mu][t];
            basis.basis_rank1[mu][t].is_proxy = old_proxy;
        }
    }
    return res;
}

vector<vector<ACECTildeBasisFunction>> ACECTildeBasisSet_get_basis(const ACECTildeBasisSet &basis) {
    vector<vector<ACECTildeBasisFunction>> res;
    res.resize(basis.nelements);
    for (SPECIES_TYPE mu = 0; mu < basis.nelements; mu++) {
        SHORT_INT_TYPE size = basis.total_basis_size[mu];
        res[mu].resize(size);
        for (SHORT_INT_TYPE t = 0; t < size; t++) {
            //turn off the proxying for proper copying when export to Python
            bool old_proxy = basis.basis[mu][t].is_proxy;
            basis.basis[mu][t].is_proxy = false;
            res[mu][t] = basis.basis[mu][t];
            basis.basis[mu][t].is_proxy = old_proxy;
        }
    }
    return res;
}

py::tuple ACECTildeBasisSet_getstate(const ACECTildeBasisSet &cbasisSet) {
    vector<string> elements_name(cbasisSet.nelements);
    for (SPECIES_TYPE mu = 0; mu < cbasisSet.nelements; ++mu)
        elements_name[mu] = cbasisSet.elements_name[mu];

    auto tuple = py::make_tuple(
            cbasisSet.lmax,  //0
            cbasisSet.nradbase, //1
            cbasisSet.nradmax, //2
            cbasisSet.nelements, //3
            cbasisSet.rankmax, //4
            cbasisSet.ndensitymax, //5
            cbasisSet.cutoffmax, //6
            cbasisSet.deltaSplineBins, //7
            cbasisSet.FS_parameters, //8
            ACERadialFunctions_getstate(cbasisSet.radial_functions), //9
            cbasisSet.rho_core_cutoffs.to_vector(), //10
            cbasisSet.drho_core_cutoffs.to_vector(), //11
            elements_name, //12
            ACECTildeBasisSet_get_basis_rank1(cbasisSet), //13
            ACECTildeBasisSet_get_basis(cbasisSet),//14
            cbasisSet.E0vals.to_vector() //15
    );
    return tuple;
}

ACECTildeBasisSet ACECTildeBasisSet_setstate(const py::tuple &t) {
    if (t.size() != 16)
        throw std::runtime_error("Invalid state of ACECTildeBasisSet-tuple");

    ACECTildeBasisSet new_cbasis;
    new_cbasis.lmax = t[0].cast<LS_TYPE>();  //0
    new_cbasis.nradbase = t[1].cast<NS_TYPE>(); //1
    new_cbasis.nradmax = t[2].cast<NS_TYPE>(); //2
    new_cbasis.nelements = t[3].cast<SPECIES_TYPE>(); //3
    new_cbasis.rankmax = t[4].cast<RANK_TYPE>(); //4
    new_cbasis.ndensitymax = t[5].cast<DENSITY_TYPE>(); //5
    new_cbasis.cutoffmax = t[6].cast<DOUBLE_TYPE>(); //6
    new_cbasis.deltaSplineBins = t[7].cast<DOUBLE_TYPE>(); //7
    new_cbasis.FS_parameters = t[8].cast<vector<DOUBLE_TYPE>>(); //8
    new_cbasis.spherical_harmonics.init(new_cbasis.lmax);
    new_cbasis.radial_functions = ACERadialFunctions_setstate(t[9].cast<py::tuple>()); //9

    new_cbasis.rho_core_cutoffs = t[10].cast<vector<DOUBLE_TYPE>>(); //10
    new_cbasis.drho_core_cutoffs = t[11].cast<vector<DOUBLE_TYPE>>(); //11
    auto elements_name = t[12].cast<vector<string>>(); //12
    auto basis_rank1 = t[13].cast<vector<vector<ACECTildeBasisFunction>>>(); //13
    auto basis = t[14].cast<vector<vector<ACECTildeBasisFunction>>>(); //14
    new_cbasis.E0vals = t[15].cast<vector<DOUBLE_TYPE>>();//15

    new_cbasis.elements_name = new string[elements_name.size()];
    for (int i = 0; i < elements_name.size(); i++) {
        new_cbasis.elements_name[i] = elements_name[i];
    }

    new_cbasis.total_basis_size_rank1 = new SHORT_INT_TYPE[new_cbasis.nelements];
    new_cbasis.basis_rank1 = new ACECTildeBasisFunction *[new_cbasis.nelements];
    for (SPECIES_TYPE mu = 0; mu < new_cbasis.nelements; ++mu) {
        SHORT_INT_TYPE size = basis_rank1[mu].size();
        new_cbasis.total_basis_size_rank1[mu] = size;
        new_cbasis.basis_rank1[mu] = new ACECTildeBasisFunction[size];
    }
    for (SPECIES_TYPE mu = 0; mu < new_cbasis.nelements; mu++)
        for (SHORT_INT_TYPE func_ind = 0; func_ind < new_cbasis.total_basis_size_rank1[mu]; ++func_ind) {
            new_cbasis.basis_rank1[mu][func_ind] = basis_rank1[mu][func_ind];
        }

    new_cbasis.total_basis_size = new SHORT_INT_TYPE[new_cbasis.nelements];
    new_cbasis.basis = new ACECTildeBasisFunction *[new_cbasis.nelements];
    for (SPECIES_TYPE mu = 0; mu < new_cbasis.nelements; ++mu) {
        SHORT_INT_TYPE size = basis[mu].size();
        new_cbasis.total_basis_size[mu] = size;
        new_cbasis.basis[mu] = new ACECTildeBasisFunction[size];
    }

    for (SPECIES_TYPE mu = 0; mu < new_cbasis.nelements; mu++)
        for (SHORT_INT_TYPE func_ind = 0; func_ind < new_cbasis.total_basis_size[mu]; ++func_ind) {
            new_cbasis.basis[mu][func_ind] = basis[mu][func_ind];
        }

    new_cbasis.pack_flatten_basis();
    return new_cbasis;
}