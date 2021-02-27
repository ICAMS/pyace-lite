/*
 * Atomic cluster expansion
 *
 * Copyright 2021  (c) Yury Lysogorskiy, Sarath Menon,
 * Matteo Rinaldi, Thomas Hammerschmidt, Matous Mrovec,
 * Ralf Drautz
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
// Created by Yury Lysogorskiy on 16.03.2020.
//

#ifndef ACE_B_BASIS_H
#define ACE_B_BASIS_H

#include <list>
#include <map>
#include <tuple>
#include <vector>

#include "yaml-cpp/yaml.h"

#include "ace_b_basisfunction.h"
#include "ace_flatten_basis.h"
#include "ace_clebsch_gordan.h"
#include "ace_c_basis.h"

using namespace std;

typedef vector<NS_TYPE> Vector_ns;
typedef vector<LS_TYPE> Vector_ls;
typedef vector<MS_TYPE> Vector_ms;
typedef vector<SPECIES_TYPE> Vector_Xs;
typedef tuple<SPECIES_TYPE, Vector_ns, Vector_ls, Vector_Xs> Basis_index_key;
typedef list<ACEBBasisFunction *> Basis_function_ptr_list;
typedef map<Basis_index_key, Basis_function_ptr_list> Basis_functions_map;
typedef vector<vector<ACEBBasisFunction>> B_full_basis_vector2d;

/**
Split a given string by tabs or space

@param mainkey, string - input string

@returns splitted, string - the string after splitting
*/
vector<string> split_key(string mainkey) ;


/**
Now a class called SpeciesBasisFunctionsBlock - This one stores all the necessary information to
 initialize the ACEBBasisFunction
*/
class BBasisFunctionsSpecificationBlock {
public:
    std::string block_name; //"Al Al"
    RANK_TYPE rankmax = 0;
    SPECIES_TYPE number_of_species = 0;
    std::vector<std::string> elements_vec;
    std::string mu0;
    LS_TYPE lmaxi = 0;
    //max value of n in Eq.(27), R_nl
    NS_TYPE nradmaxi = 0;
    DENSITY_TYPE ndensityi = 0;

    //Finnis-Sinclair
    std::string npoti;

    //paremeter of Finnis-Sinclair function
    std::vector<DOUBLE_TYPE> fs_parameters;

    //hard-core repulsion parameters
    //ex:  core-repulsion: [3.0, 3.0]
    std::vector<DOUBLE_TYPE> core_rep_parameters{0., 0.};

    // energy cutoff repulsion parameter
    // for f(rho_core) cutoff function
    DOUBLE_TYPE rho_cut = 100000.0;
    DOUBLE_TYPE drho_cut = 250.0;

    DOUBLE_TYPE rcutij = 0;
    DOUBLE_TYPE dcutij = 0;
    std::string NameOfCutoffFunctionij;
    NS_TYPE nradbaseij = 0;
    std::string radbase = "ChebExpCos"; ///< * type of radial basis function \f$ g_k(r) \f$ (default: "ChebExpCos")
    std::vector<DOUBLE_TYPE> radparameters;
    vector<vector<vector<DOUBLE_TYPE>>> radcoefficients;///< crad: order: [n=0..nradmax-1][l=0..lmax][k=0..nradbase-1]

    std::vector<BBasisFunctionSpecification> funcspecs; // 0 dim = rank, 1 dim - basis func ind

    YAML::Node to_YAML() const;

    int get_number_of_coeffs() const;

    vector<DOUBLE_TYPE> get_all_coeffs() const;

    void set_all_coeffs(const vector<DOUBLE_TYPE> &new_coeffs);

    void update_params();

};

struct BBasisConfiguration {
    DOUBLE_TYPE deltaSplineBins;
    vector<BBasisFunctionsSpecificationBlock> funcspecs_blocks;

    map<string, string> metadata;

    BBasisConfiguration() = default;

    explicit BBasisConfiguration(const string &filename) {
        this->load(filename);
    }

    void load(const string &yaml_file_name);

    void save(const string &yaml_file_name);

    vector<DOUBLE_TYPE> get_all_coeffs() const;

    void set_all_coeffs(const vector<DOUBLE_TYPE> &new_all_coeffs);
};


class ACEBBasisSet : public ACEFlattenBasisSet {
public:

    //[mu][func_ind]
    ACEBBasisFunction **basis_rank1 = nullptr;
    ACEBBasisFunction **basis = nullptr;


    //contiguous array of generalized Clebsh-Gordan coefficients (for packing ACEBBasisFunction)
    size_t total_num_of_ms_comb_rank1 = 0; //size for full_gencg_rank1
    size_t total_num_of_ms_comb = 0; //size for full_gencg
    //size == total num of ms combinations
    DOUBLE_TYPE *full_gencg_rank1 = nullptr; //size = total_num_of_ms_comb_rank1
    DOUBLE_TYPE *full_gencg = nullptr; //size =total_num_of_ms_comb

    size_t total_LS_size = 0;
    LS_TYPE *full_LS = nullptr;

    //contiguous array of coefficients (for packing ACEBBasisFunction)
    // size==number of basis function * ndensity == coeff_array_total_size
    DOUBLE_TYPE *full_coeff_rank1 = nullptr; // size = coeff_array_total_size_rank1
    DOUBLE_TYPE *full_coeff = nullptr; // size = coeff_array_total_size

    map<string, string> metadata;

    ACEBBasisSet() = default;

    ACEBBasisSet(string yaml_file_name);

    ACEBBasisSet(BBasisConfiguration &bBasisConfiguration);

    // copy constructor, operator= and destructor (see. Rule of Three)
    ACEBBasisSet(const ACEBBasisSet &other);

    ACEBBasisSet &operator=(const ACEBBasisSet &other);

    ~ACEBBasisSet();

    //[mu][func_ind]
    B_full_basis_vector2d mu0_bbasis_vector;

    void save(const string &filename) override;

    void load(const string filename) override;

    void _clean_basis_arrays();

    void compute_array_sizes(ACEBBasisFunction** basis_rank1, ACEBBasisFunction** basis);
    void pack_flatten_basis() override;

    ACECTildeBasisSet to_ACECTildeBasisSet() const;

    BBasisConfiguration to_BBasisConfiguration() const;

    // other routines
    void compress_basis_functions();
    void flatten_basis() override;

    // routines for copying and cleaning dynamic memory of the class (see. Rule of Three)
    void _clean() override;

    void _copy_dynamic_memory(const ACEBBasisSet &src);

    void _copy_scalar_memory(const ACEBBasisSet &src);

    void initialize_basis(BBasisConfiguration &basisSetup);

    void _clean_contiguous_arrays();
};

void order_and_compress_b_basis_function(ACEBBasisFunction &func);
void convert_B_to_Ctilde_basis_functions(const vector<ACEBBasisFunction> &b_basis_vector,
                                         vector<ACECTildeBasisFunction> &ctilde_basis_vector);

#endif //ACE_ACE_B_BASIS_H
