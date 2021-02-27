//
// Created by Yury Lysogorskiy on 16.03.2020.
//
#include "ace_b_basis.h"

#include <algorithm>
#include <sstream>


#include "ace_yaml_input.h"
#include "ace_couplings.h"
#include "ace_c_basis.h"
#include "ace_utils.h"

void group_basis_functions_by_index(const vector<ACEBBasisFunction> &basis,
                                    Basis_functions_map &basis_functions_map) {
    for (const auto &cur_basfunc : basis) {
        auto *current_basis_function = const_cast<ACEBBasisFunction *>(&cur_basfunc);
        SPECIES_TYPE X0 = current_basis_function->mu0;
        RANK_TYPE r = cur_basfunc.rank;
        Vector_ns vector_ns(current_basis_function->ns, current_basis_function->ns + r - 1 + 1);
        Vector_ls vector_ls(current_basis_function->ls, current_basis_function->ls + r - 1 + 1);
        Vector_Xs vector_Xs(current_basis_function->mus, current_basis_function->mus + r - 1 + 1);
        Basis_index_key key(X0, vector_ns, vector_ls, vector_Xs);
        auto search = basis_functions_map.find(key);
        if (search == basis_functions_map.end()) { // not in dict
            basis_functions_map[key] = Basis_function_ptr_list();
        }
        basis_functions_map[key].push_back(current_basis_function);
    }
}


void summation_over_LS(Basis_functions_map &basis_functions_map,
                       vector<ACECTildeBasisFunction> &ctilde_basis) {

#ifdef DEBUG_C_TILDE
    cout << "rankmax=" << (int) r << "\t";
    cout << "number of basis functions (len dict)= " << basis_functions_map.size() << endl;
#endif
    // loop over dictionary of grouped basis functions
    ctilde_basis.resize(basis_functions_map.size());
    int new_b_tilde_index = 0;
    for (auto it = basis_functions_map.begin(); basis_functions_map.end() != it; ++it, ++new_b_tilde_index) {

        ACECTildeBasisFunction &new_func = ctilde_basis[new_b_tilde_index];
        const SPECIES_TYPE &X0 = get<0>(it->first);
        const Vector_ns &ns = get<1>(it->first);
        const Vector_ls &ls = get<2>(it->first);
        const Vector_Xs &XS = get<3>(it->first);

        Basis_function_ptr_list b_basis_list = it->second;
        ACEBBasisFunction *first_basis_function = b_basis_list.front();

        //RANK_TYPE cur_rank = first_basis_function->rank;
        RANK_TYPE cur_rank = ns.size();
        new_func.rank = cur_rank;
        new_func.ndensity = first_basis_function->ndensity;

        new_func.mu0 = X0;
        delete[] new_func.mus;
        new_func.mus = new SPECIES_TYPE[cur_rank];

        for (RANK_TYPE i = 0; i < cur_rank; ++i) new_func.mus[i] = XS[i];

        delete[] new_func.ns;
        new_func.ns = new NS_TYPE[cur_rank];
        for (RANK_TYPE i = 0; i < cur_rank; ++i) new_func.ns[i] = ns[i];

        delete[] new_func.ls;
        new_func.ls = new LS_TYPE[cur_rank];
        for (RANK_TYPE i = 0; i < cur_rank; ++i) new_func.ls[i] = ls[i];


        //TODO: join the ms combinations, not only take the first one
        map<Vector_ms, vector<DOUBLE_TYPE>> ms_combinations_coefficients_map;
        for (const auto &bas_func: b_basis_list) {
            for (int ms_ind = 0; ms_ind < bas_func->num_ms_combs; ++ms_ind) {
                Vector_ms ms_vec(cur_rank);
                for (RANK_TYPE rr = 0; rr < cur_rank; ++rr)
                    ms_vec[rr] = bas_func->ms_combs[ms_ind * cur_rank + rr];
                if (ms_combinations_coefficients_map.find(ms_vec) == ms_combinations_coefficients_map.end())
                    ms_combinations_coefficients_map[ms_vec].resize(new_func.ndensity);
                //sum-up vector
                for (DENSITY_TYPE p = 0; p < new_func.ndensity; ++p)
                    ms_combinations_coefficients_map[ms_vec][p] += bas_func->coeff[p] * bas_func->gen_cgs[ms_ind];
            }
        }

        new_func.num_ms_combs = ms_combinations_coefficients_map.size();

        delete[] new_func.ms_combs;
        new_func.ms_combs = new MS_TYPE[cur_rank * new_func.num_ms_combs];
        delete[] new_func.ctildes;
        new_func.ctildes = new DOUBLE_TYPE[new_func.num_ms_combs * new_func.ndensity];

        int ms_ind = 0;
        for (const auto &ms_coeff_pair: ms_combinations_coefficients_map) {
            Vector_ms ms_vec = ms_coeff_pair.first;
            vector<DOUBLE_TYPE> coeff = ms_coeff_pair.second;

            //copy ms combination
            for (RANK_TYPE rr = 0; rr < cur_rank; rr++)
                new_func.ms_combs[ms_ind * cur_rank + rr] = ms_vec[rr];
            //copy corresponding c_tilde coefficient
            for (DENSITY_TYPE p = 0; p < new_func.ndensity; ++p)
                new_func.ctildes[ms_ind * new_func.ndensity + p] = coeff[p];

            SHORT_INT_TYPE sign = 0;
            for (RANK_TYPE t = 0; t < cur_rank; ++t)
                if (ms_vec[t] < 0) {
                    sign = -1;
                    break;
                } else if (ms_vec[t] > 0) {
                    sign = +1;
                    break;
                }


            ms_ind++;
        }

        new_func.is_half_ms_basis = first_basis_function->is_half_ms_basis;

#ifdef DEBUG_C_TILDE
        cout << "new_func=" << endl;
        print_C_tilde_B_basis_function(ctilde_basis[r - 1][new_b_tilde_index]);
#endif
    }
}

template<typename T>
T max_of(vector<T> vec) {
    return *max_element(vec.begin(), vec.end());
}

LS_TYPE get_lmax(vector<LS_TYPE> ls, vector<LS_TYPE> LS) {
    LS_TYPE func_lmax = max_of(ls);
    LS_TYPE func_Lmax = 0;
    if (LS.size() > 0)
        func_Lmax = max_of(LS);
    return (func_lmax > func_Lmax ? func_lmax : func_Lmax);
}


//constructor from BBasisConfiguration
ACEBBasisSet::ACEBBasisSet(BBasisConfiguration &bBasisConfiguration) {
    initialize_basis(bBasisConfiguration);
}

//constructor by loading from YAML file
ACEBBasisSet::ACEBBasisSet(string yaml_file_name) {
    ACEBBasisSet::load(yaml_file_name);
}

//copy constructor
ACEBBasisSet::ACEBBasisSet(const ACEBBasisSet &other) {
    ACEBBasisSet::_copy_scalar_memory(other);
    ACEBBasisSet::_copy_dynamic_memory(other);
    ACEBBasisSet::pack_flatten_basis();
}

//operator=
ACEBBasisSet &ACEBBasisSet::operator=(const ACEBBasisSet &other) {
    if (this != &other) {
        ACEBBasisSet::_clean();
        ACEBBasisSet::_copy_scalar_memory(other);
        ACEBBasisSet::_copy_dynamic_memory(other);
        ACEBBasisSet::pack_flatten_basis();
    }
    return *this;
}

ACEBBasisSet::~ACEBBasisSet() {
    ACEBBasisSet::_clean();
}


//pack into 1D array with all basis functions
void ACEBBasisSet::flatten_basis() {
    _clean_basis_arrays();

    if (total_basis_size_rank1 != nullptr) delete[] total_basis_size_rank1;
    if (total_basis_size != nullptr) delete[] total_basis_size;

    total_basis_size_rank1 = new SHORT_INT_TYPE[nelements];
    total_basis_size = new SHORT_INT_TYPE[nelements];


    basis_rank1 = new ACEBBasisFunction *[nelements];
    basis = new ACEBBasisFunction *[nelements];

    size_t tot_size_rank1 = 0;
    size_t tot_size = 0;

    for (SPECIES_TYPE mu = 0; mu < this->nelements; ++mu) {
        tot_size = 0;
        tot_size_rank1 = 0;

        for (auto &func: this->mu0_bbasis_vector[mu]) {
            if (func.rank == 1) tot_size_rank1 += 1;
            else tot_size += 1;
        }


        total_basis_size_rank1[mu] = tot_size_rank1;
        basis_rank1[mu] = new ACEBBasisFunction[tot_size_rank1];

        total_basis_size[mu] = tot_size;
        basis[mu] = new ACEBBasisFunction[tot_size];
    }


    for (SPECIES_TYPE mu = 0; mu < this->nelements; ++mu) {
        size_t ind_rank1 = 0;
        size_t ind = 0;

        for (auto &func: this->mu0_bbasis_vector[mu]) {
            if (func.rank == 1) { //r=0, rank=1
                basis_rank1[mu][ind_rank1] = func;
                ind_rank1 += 1;
            } else { //r>0, rank>1
                basis[mu][ind] = func;
                ind += 1;
            }
        }

    }
}

void ACEBBasisSet::_clean() {
    // call parent method
    ACEFlattenBasisSet::_clean();
    _clean_contiguous_arrays();
    _clean_basis_arrays();
}

void ACEBBasisSet::_clean_contiguous_arrays() {
    if (full_gencg_rank1 != nullptr) delete[] full_gencg_rank1;
    full_gencg_rank1 = nullptr;

    if (full_gencg != nullptr) delete[] full_gencg;
    full_gencg = nullptr;

    if (full_coeff_rank1 != nullptr) delete[] full_coeff_rank1;
    full_coeff_rank1 = nullptr;

    if (full_coeff != nullptr) delete[] full_coeff;
    full_coeff = nullptr;

    if (full_LS != nullptr) delete[] full_LS;
    full_LS = nullptr;
}

void ACEBBasisSet::_clean_basis_arrays() {
    if (basis_rank1 != nullptr)
        for (SPECIES_TYPE mu = 0; mu < nelements; ++mu) {
            delete[] basis_rank1[mu];
            basis_rank1[mu] = nullptr;
        }

    if (basis != nullptr)
        for (SPECIES_TYPE mu = 0; mu < nelements; ++mu) {
            delete[] basis[mu];
            basis[mu] = nullptr;
        }
    delete[] basis;
    basis = nullptr;

    delete[] basis_rank1;
    basis_rank1 = nullptr;
}

void ACEBBasisSet::_copy_scalar_memory(const ACEBBasisSet &src) {
    ACEFlattenBasisSet::_copy_scalar_memory(src);
    mu0_bbasis_vector = src.mu0_bbasis_vector;

    total_num_of_ms_comb_rank1 = src.total_num_of_ms_comb_rank1;
    total_num_of_ms_comb = src.total_num_of_ms_comb;

    total_LS_size = src.total_LS_size;
}

void ACEBBasisSet::_copy_dynamic_memory(const ACEBBasisSet &src) {//allocate new memory
    ACEFlattenBasisSet::_copy_dynamic_memory(src);

    if (src.basis_rank1 == nullptr)
        throw runtime_error("Could not copy ACEBBasisSet::basis_rank1 - array not initialized");
    if (src.basis == nullptr)
        throw runtime_error("Could not copy ACEBBasisSet::basis - array not initialized");

    basis_rank1 = new ACEBBasisFunction *[nelements];
    basis = new ACEBBasisFunction *[nelements];

    //copy basis arrays
    for (SPECIES_TYPE mu = 0; mu < nelements; ++mu) {
        basis_rank1[mu] = new ACEBBasisFunction[total_basis_size_rank1[mu]];

        for (size_t i = 0; i < total_basis_size_rank1[mu]; i++) {
            this->basis_rank1[mu][i] = src.basis_rank1[mu][i];
        }

        basis[mu] = new ACEBBasisFunction[total_basis_size[mu]];
        for (size_t i = 0; i < total_basis_size[mu]; i++) {
            basis[mu][i] = src.basis[mu][i];
        }
    }

    //DON"T COPY CONTIGUOUS ARRAY, REBUILD THEM
}

void ACEBBasisSet::pack_flatten_basis() {
    compute_array_sizes(basis_rank1, basis);

    //2. allocate contiguous arrays
    full_ns_rank1 = new NS_TYPE[rank_array_total_size_rank1];
    full_ls_rank1 = new NS_TYPE[rank_array_total_size_rank1];
    full_mus_rank1 = new SPECIES_TYPE[rank_array_total_size_rank1];
    full_ms_rank1 = new MS_TYPE[rank_array_total_size_rank1];

    full_gencg_rank1 = new DOUBLE_TYPE[total_num_of_ms_comb_rank1];
    full_coeff_rank1 = new DOUBLE_TYPE[coeff_array_total_size_rank1];


    full_ns = new NS_TYPE[rank_array_total_size];
    full_ls = new LS_TYPE[rank_array_total_size];
    full_LS = new LS_TYPE[total_LS_size];

    full_mus = new SPECIES_TYPE[rank_array_total_size];
    full_ms = new MS_TYPE[ms_array_total_size];

    full_gencg = new DOUBLE_TYPE[total_num_of_ms_comb];
    full_coeff = new DOUBLE_TYPE[coeff_array_total_size];

    //3. copy the values from private C_tilde_B_basis_function arrays to new contigous space
    //4. clean private memory
    //5. reassign private array pointers

    //r = 0, rank = 1
    size_t rank_array_ind_rank1 = 0;
    size_t coeff_array_ind_rank1 = 0;
    size_t ms_array_ind_rank1 = 0;
    size_t gen_cg_ind_rank1 = 0;

    for (SPECIES_TYPE mu = 0; mu < nelements; ++mu) {
        for (int func_ind_r1 = 0; func_ind_r1 < total_basis_size_rank1[mu]; ++func_ind_r1) {
            auto &func = basis_rank1[mu][func_ind_r1];
            //copy values ns from c_tilde_basis_function private memory to contigous memory part
            full_ns_rank1[rank_array_ind_rank1] = func.ns[0];

            //copy values ls from c_tilde_basis_function private memory to contigous memory part
            full_ls_rank1[rank_array_ind_rank1] = func.ls[0];

            //copy values mus from c_tilde_basis_function private memory to contigous memory part
            full_mus_rank1[rank_array_ind_rank1] = func.mus[0];

            //copy values full_coeff_rank1 from c_tilde_basis_function private memory to contigous memory part
            memcpy(&full_coeff_rank1[coeff_array_ind_rank1], func.coeff,
                   func.ndensity * sizeof(DOUBLE_TYPE));

            memcpy(&full_gencg_rank1[gen_cg_ind_rank1], func.gen_cgs,
                   func.num_ms_combs * sizeof(DOUBLE_TYPE));

            //copy values mus from c_tilde_basis_function private memory to contigous memory part
            memcpy(&full_ms_rank1[ms_array_ind_rank1], func.ms_combs,
                   func.num_ms_combs *
                   func.rank * sizeof(MS_TYPE));

            //release memory of each ACECTildeBasisFunction if it is not proxy
            func._clean();

            func.ns = &full_ns_rank1[rank_array_ind_rank1];
            func.ls = &full_ls_rank1[rank_array_ind_rank1];
            func.mus = &full_mus_rank1[rank_array_ind_rank1];
            func.coeff = &full_coeff_rank1[coeff_array_ind_rank1];
            func.gen_cgs = &full_gencg_rank1[gen_cg_ind_rank1];
            func.ms_combs = &full_ms_rank1[ms_array_ind_rank1];
            func.is_proxy = true;

            rank_array_ind_rank1 += func.rank;
            ms_array_ind_rank1 += func.rank *
                                  func.num_ms_combs;

            coeff_array_ind_rank1 += func.ndensity;
            gen_cg_ind_rank1 += func.num_ms_combs;
        }
    }


    //rank>1, r>0
    size_t rank_array_ind = 0;
    size_t coeff_array_ind = 0;
    size_t ms_array_ind = 0;
    size_t gen_cg_ind = 0;
    size_t LS_array_ind = 0;

    for (SPECIES_TYPE mu = 0; mu < nelements; ++mu) {
        for (int func_ind = 0; func_ind < total_basis_size[mu]; ++func_ind) {
            ACEBBasisFunction &func = basis[mu][func_ind];

            //copy values ns from c_tilde_basis_function private memory to contigous memory part
            memcpy(&full_ns[rank_array_ind], func.ns,
                   func.rank * sizeof(NS_TYPE));
            //copy values ls from c_tilde_basis_function private memory to contigous memory part
            memcpy(&full_ls[rank_array_ind], func.ls,
                   func.rank * sizeof(LS_TYPE));
            //copy values mus from c_tilde_basis_function private memory to contigous memory part
            memcpy(&full_mus[rank_array_ind], func.mus,
                   func.rank * sizeof(SPECIES_TYPE));

            //copy values mus from c_tilde_basis_function private memory to contigous memory part
            memcpy(&full_LS[LS_array_ind], func.LS,
                   func.rankL * sizeof(LS_TYPE));


            //copy values full_coeff_rank1 from c_tilde_basis_function private memory to contigous memory part
            memcpy(&full_coeff[coeff_array_ind], func.coeff,
                   func.ndensity * sizeof(DOUBLE_TYPE));

            memcpy(&full_gencg[gen_cg_ind], func.gen_cgs,
                   func.num_ms_combs * sizeof(DOUBLE_TYPE));

            //copy values mus from c_tilde_basis_function private memory to contigous memory part
            memcpy(&full_ms[ms_array_ind], func.ms_combs,
                   func.num_ms_combs *
                   func.rank * sizeof(MS_TYPE));

            //release memory of each ACECTildeBasisFunction if it is not proxy
            func._clean();

            func.ns = &full_ns[rank_array_ind];
            func.ls = &full_ls[rank_array_ind];
            func.mus = &full_mus[rank_array_ind];
            func.LS = &full_LS[LS_array_ind];

            func.coeff = &full_coeff[coeff_array_ind];
            func.gen_cgs = &full_gencg[gen_cg_ind];

            func.ms_combs = &full_ms[ms_array_ind];
            func.is_proxy = true;

            rank_array_ind += func.rank;
            ms_array_ind += func.rank *
                            func.num_ms_combs;
            coeff_array_ind += func.ndensity;
            gen_cg_ind += func.num_ms_combs;
            LS_array_ind += func.rankL;

        }
    }

}


vector<string> split_key(string mainkey) {

    vector<string> splitted;
    istringstream stream(mainkey);

    for (string mainkey; stream >> mainkey;)
        splitted.emplace_back(mainkey);

    return splitted;
}

void ACEBBasisSet::load(string filename) {
    BBasisConfiguration basisSetup;
    basisSetup.load(filename);
    initialize_basis(basisSetup);
}

void order_and_compress_b_basis_function(ACEBBasisFunction &func) {
    vector<tuple<SPECIES_TYPE, NS_TYPE, LS_TYPE, MS_TYPE, int> > v;

    vector<SPECIES_TYPE> new_XS(func.rank);
    vector<NS_TYPE> new_NS(func.rank);
    vector<LS_TYPE> new_LS(func.rank);
    vector<int> sort_order(func.rank);

    map<vector<MS_TYPE>, DOUBLE_TYPE> ms_map;
    int new_ms_ind = 0;
    vector<MS_TYPE> new_ms;
    DOUBLE_TYPE new_gen_cg;

    for (SHORT_INT_TYPE ms_comb_ind = 0; ms_comb_ind < func.num_ms_combs; ms_comb_ind++) {

        v.clear();
        for (RANK_TYPE r = 0; r < func.rank; r++) {
            v.emplace_back(
                    make_tuple(func.mus[r], func.ns[r], func.ls[r], func.ms_combs[ms_comb_ind * func.rank + r], r));
        }

        sort(v.begin(), v.end());

        //check if (tup(0..2) always the same
        if (ms_comb_ind == 0) {
            for (RANK_TYPE r = 0; r < func.rank; r++) {
                new_XS[r] = get<0>(v[r]);
                new_NS[r] = get<1>(v[r]);
                new_LS[r] = get<2>(v[r]);
                sort_order[r] = get<4>(v[r]);
            }
        }

        for (RANK_TYPE r = 0; r < func.rank; r++) {
            if (new_XS[r] != get<0>(v[r]) ||
                new_NS[r] != get<1>(v[r]) ||
                new_LS[r] != get<2>(v[r])) {
                stringstream s;
                s << "INCONSISTENT SORTED BLOCK!\n";
                s << "->>sorted XS-ns-ls-ms combinations: {\n";
                char buf[1024];
                for (const auto &tup: v) {
                    sprintf(buf, "(%d, %d, %d, %d)\n", get<0>(tup), get<1>(tup), get<2>(tup), get<3>(tup));
                    s << buf;
                }
                s << "}";
                throw logic_error(s.str());
            }
        }

        vector<MS_TYPE> new_ms(func.rank);
        for (RANK_TYPE r = 0; r < func.rank; r++)
            new_ms[r] = get<3>(v[r]);

        auto search = ms_map.find(new_ms);
        if (search == ms_map.end()) { // not in dict
            ms_map[new_ms] = 0;
        }

        ms_map[new_ms] += func.gen_cgs[ms_comb_ind];
    }
    //  drop-out the k,v pairs from ms_map when value is zero
    for (auto it = ms_map.begin(); it != ms_map.end();) {
        auto key_ms = it->first;
        auto val_gen_cg = it->second;
        if (abs(val_gen_cg) < 1e-15) {
            ms_map.erase(it++);
        } else {
            ++it;
        }
    }


    int gain = func.num_ms_combs - ms_map.size();

    if (gain > 0) {
        for (RANK_TYPE r = 0; r < func.rank; r++) {
            func.mus[r] = new_XS[r];
            func.ns[r] = new_NS[r];
            func.ls[r] = new_LS[r];
        }
        func.sort_order = sort_order;
        SHORT_INT_TYPE new_num_of_ms_combinations = ms_map.size();


        delete[] func.gen_cgs;
        delete[] func.ms_combs;

        func.gen_cgs = new DOUBLE_TYPE[new_num_of_ms_combinations];
        func.ms_combs = new MS_TYPE[new_num_of_ms_combinations * func.rank];


        for (auto it = ms_map.begin(); it != ms_map.end(); ++it, ++new_ms_ind) {
            new_ms = it->first;
            new_gen_cg = it->second;

            for (RANK_TYPE r = 0; r < func.rank; r++)
                func.ms_combs[new_ms_ind * func.rank + r] = new_ms[r];


            func.gen_cgs[new_ms_ind] = new_gen_cg;
        }

        func.num_ms_combs = new_num_of_ms_combinations;

    }

}

// compress each basis function by considering A*A*..*A symmetry wrt. permutations
void ACEBBasisSet::compress_basis_functions() {
    SHORT_INT_TYPE tot_ms_combs = 0, num_ms_combs = 0;
    SHORT_INT_TYPE tot_new_ms_combs = 0, new_ms_combs = 0;
    for (SPECIES_TYPE elei = 0; elei < this->nelements; ++elei) {

        num_ms_combs = 0;
        new_ms_combs = 0;
        vector<ACEBBasisFunction> &sub_basis = this->mu0_bbasis_vector[elei];

        for (ACEBBasisFunction &func: sub_basis) {
            tot_ms_combs += func.num_ms_combs;
            num_ms_combs += func.num_ms_combs;

            order_and_compress_b_basis_function(func);
            tot_new_ms_combs += func.num_ms_combs;
            new_ms_combs += func.num_ms_combs;
        }
        if (new_ms_combs < num_ms_combs) {
            printf("element: %d - basis compression from %d to %d by %d ms-combinations (%.2f%%) \n",
                   (int) elei, num_ms_combs, new_ms_combs,
                   num_ms_combs - new_ms_combs, 1. * (num_ms_combs - new_ms_combs) / num_ms_combs * 100.);
        }

    }

    if (new_ms_combs < num_ms_combs) {
        printf("Total basis compression from %d to %d by %d ms-combinations\n",
               tot_ms_combs, tot_new_ms_combs,
               tot_ms_combs - tot_new_ms_combs);
    }
}

//TODO: save to YAML
void ACEBBasisSet::save(const string &filename) {
    BBasisConfiguration config = this->to_BBasisConfiguration();
    config.save(filename);
}

BBasisConfiguration ACEBBasisSet::to_BBasisConfiguration() const {
    BBasisConfiguration config;

    config.metadata = this->metadata;

    config.deltaSplineBins = radial_functions->deltaSplineBins;
    vector<BBasisFunctionsSpecificationBlock> blocks(this->nelements); //TODO - change to nblocks!!! for multispecies
    vector<string> elements_mapping(nelements);

    for (SPECIES_TYPE s = 0; s < nelements; s++)
        elements_mapping[s] = elements_name[s];

    for (SPECIES_TYPE mu = 0; mu < nelements; mu++) {
        BBasisFunctionsSpecificationBlock &block = blocks[mu];
        block.block_name = elements_name[mu];
        block.rankmax = rankmax;
        block.number_of_species = 2;//TODO: fix, hardcoded
        block.elements_vec = {elements_name[mu], elements_name[mu]};
        block.mu0 = elements_name[mu];
        block.lmaxi = lmax;
        block.nradmaxi = nradmax;
        block.ndensityi = ndensitymax;
        block.npoti = npoti;
        block.fs_parameters = FS_parameters;
        block.core_rep_parameters[0] = radial_functions->prehc(mu, mu);
        block.core_rep_parameters[1] = radial_functions->lambdahc(mu, mu);

        block.rho_cut = rho_core_cutoffs(mu);
        block.drho_cut = drho_core_cutoffs(0);

        block.rcutij = radial_functions->cut(mu, mu);
        block.dcutij = radial_functions->dcut(mu, mu);
        block.NameOfCutoffFunctionij = "cos";
        block.nradbaseij = radial_functions->nradbase;
        block.radbase = radial_functions->radbasename;

        block.radparameters = {radial_functions->lambda(mu, mu)};

        block.radcoefficients.resize(radial_functions->nradial);
        for (NS_TYPE n = 0; n < nradmax; n++) {
            block.radcoefficients.at(n).resize(block.lmaxi + 1);
            for (LS_TYPE l = 0; l <= lmax; l++) {
                block.radcoefficients.at(n).at(l).resize(block.nradbaseij);
                for (NS_TYPE k = 0; k < nradbase; k++) {
                    block.radcoefficients.at(n).at(l).at(k) = radial_functions->crad(mu, mu, n, l, k);
                }
            }
        }
        vector<BBasisFunctionSpecification> funcspecs;

        auto basis_r1 = basis_rank1[mu];
        auto n_basis_r1 = total_basis_size_rank1[mu];
        for (int func_ind = 0; func_ind < n_basis_r1; func_ind++) {
            BBasisFunctionSpecification spec(elements_mapping, basis_r1[func_ind]);
            funcspecs.emplace_back(spec);
        }

        auto basis_high_r = basis[mu];
        auto n_basis_high_r = total_basis_size[mu];
        for (int func_ind = 0; func_ind < n_basis_high_r; func_ind++) {
            BBasisFunctionSpecification spec(elements_mapping, basis_high_r[func_ind]);
            funcspecs.emplace_back(spec);
        }
        block.funcspecs = funcspecs;
    }

    config.funcspecs_blocks = blocks;

    return config;
}


void ACEBBasisSet::compute_array_sizes(ACEBBasisFunction **basis_rank1, ACEBBasisFunction **basis) {
    //compute arrays sizes
    rank_array_total_size_rank1 = 0;
    //ms_array_total_size_rank1 = rank_array_total_size_rank1;
    coeff_array_total_size_rank1 = 0;

    total_num_of_ms_comb_rank1 = 0;

    for (SPECIES_TYPE mu = 0; mu < nelements; ++mu) {
        if (total_basis_size_rank1[mu] > 0) {
            rank_array_total_size_rank1 += total_basis_size_rank1[mu];
            //only one ms-comb per rank-1 basis func
            total_num_of_ms_comb_rank1 += total_basis_size_rank1[mu]; // compute size for full_gencg_rank1
            ACEAbstractBasisFunction &func = basis_rank1[mu][0];
            coeff_array_total_size_rank1 += total_basis_size_rank1[mu] * func.ndensity;// *size of full_coeff_rank1
        }
    }

    //rank>1
    rank_array_total_size = 0;
    coeff_array_total_size = 0;

    ms_array_total_size = 0;
    max_dB_array_size = 0;

    total_num_of_ms_comb = 0;

    max_B_array_size = 0;

    total_LS_size = 0;

    size_t cur_ms_size = 0;
    size_t cur_ms_rank_size = 0;

    for (SPECIES_TYPE mu = 0; mu < nelements; ++mu) {
        cur_ms_size = 0;
        cur_ms_rank_size = 0;
        if (total_basis_size[mu] == 0) continue;
        ACEAbstractBasisFunction &func = basis[mu][0];
        coeff_array_total_size += total_basis_size[mu] * func.ndensity; // size of full_coeff
        for (int func_ind = 0; func_ind < total_basis_size[mu]; ++func_ind) {
            auto &func = basis[mu][func_ind];
            rank_array_total_size += func.rank;
            ms_array_total_size += func.rank * func.num_ms_combs;
            total_num_of_ms_comb += func.num_ms_combs; // compute size for full_gencg
            cur_ms_size += func.num_ms_combs;
            cur_ms_rank_size += func.rank * func.num_ms_combs;
            total_LS_size += func.rankL;
        }

        if (cur_ms_size > max_B_array_size)
            max_B_array_size = cur_ms_size;

        if (cur_ms_rank_size > max_dB_array_size)
            max_dB_array_size = cur_ms_rank_size;
    }
}


ACECTildeBasisSet ACEBBasisSet::to_ACECTildeBasisSet() const {
    C_tilde_full_basis_vector2d mu0_ctilde_basis_vector(nelements);
    SHORT_INT_TYPE num_ctilde_max = 0;
    for (SPECIES_TYPE mu0 = 0; mu0 < this->nelements; mu0++) {
        auto const &b_basis_vector = this->mu0_bbasis_vector.at(mu0);
        auto &ctilde_basis_vectors = mu0_ctilde_basis_vector[mu0];
        convert_B_to_Ctilde_basis_functions(b_basis_vector, ctilde_basis_vectors);
        if (num_ctilde_max < ctilde_basis_vectors.size())
            num_ctilde_max = ctilde_basis_vectors.size();
    }

    ACECTildeBasisSet dest;
    //imitate the copy constructor of ACECTildeBasisSet:

    // ACECTildeBasisSet::_copy_scalar_memory(const ACECTildeBasisSet
    dest.ACEFlattenBasisSet::_copy_scalar_memory(*this);
    dest.num_ctilde_max = num_ctilde_max;


    // ACECTildeBasisSet::_copy_dynamic_memory(const ACECTildeBasisSet &src)

    // call copy_dynamic memory of ACEAbstractBasisSet but not for ACEFlattenbasisSet
    dest.ACEFlattenBasisSet::_copy_dynamic_memory(*this);

    //could not copied, should be recomputed !!!
//    this->basis = new ACECTildeBasisFunction *[src.nelements];
//    this->basis_rank1 = new ACECTildeBasisFunction *[src.nelements];
//    this->full_c_tildes_rank1 = new DOUBLE_TYPE[src.coeff_array_total_size_rank1];
//    this->full_c_tildes = new DOUBLE_TYPE[src.coeff_array_total_size];

    //pack into 1D array with all basis functions
    dest.flatten_basis(mu0_ctilde_basis_vector);
    dest.pack_flatten_basis();

    return dest;
};

void ACEBBasisSet::initialize_basis(BBasisConfiguration &basisSetup) {

    ACEClebschGordan clebsch_gordan;

    map<string, SPECIES_TYPE> elements_to_index_map;
    string radbasename = "";

    for (auto &func_spec_block: basisSetup.funcspecs_blocks) {
        func_spec_block.update_params();

        if (func_spec_block.rankmax > this->rankmax)
            this->rankmax = func_spec_block.rankmax;
        if (func_spec_block.ndensityi > this->ndensitymax)
            this->ndensitymax = func_spec_block.ndensityi;
        if (func_spec_block.lmaxi > this->lmax)
            this->lmax = func_spec_block.lmaxi;
        if (func_spec_block.nradbaseij > this->nradbase)
            this->nradbase = func_spec_block.nradbaseij;

        if (func_spec_block.nradmaxi > this->nradmax)
            this->nradmax = func_spec_block.nradmaxi;

        if (func_spec_block.rcutij > this->cutoffmax)
            this->cutoffmax = func_spec_block.rcutij;

        if (radbasename.size() == 0)
            radbasename = func_spec_block.radbase;
        else if (radbasename != func_spec_block.radbase) {
            throw invalid_argument(
                    "Radial basis functiona name: `" + func_spec_block.radbase + "` differs from previous value: " +
                    radbasename);
        }

        //update elements_to_index_map
        for (const auto &el: func_spec_block.elements_vec)
            if (!is_key_in_map(el, elements_to_index_map)) {
                int current_map_size = elements_to_index_map.size();
                elements_to_index_map[el] = static_cast<SPECIES_TYPE>(current_map_size);
            }
    }


    deltaSplineBins = basisSetup.deltaSplineBins;
    nelements = elements_to_index_map.size();

    clebsch_gordan.init(2 * lmax);
    spherical_harmonics.init(lmax);
    if (!radial_functions)
        radial_functions = new ACERadialFunctions();

    radial_functions->init(nradbase, lmax, nradmax,
                           basisSetup.deltaSplineBins,
                           nelements,
                           cutoffmax, radbasename);
    rho_core_cutoffs.init(nelements, "rho_core_cutoffs");
    rho_core_cutoffs.fill(0);
    drho_core_cutoffs.init(nelements, "drho_core_cutoffs");
    drho_core_cutoffs.fill(0);

    E0vals.init(nelements, "E0 values");
    E0vals.fill(0.0);
    //setting up the basis functions, from file or like that
    num_ms_combinations_max = 0;

    // loop over functions specifications blocks
    // fill-in rank, max rnk, ndensity max
    // construct elements_to_index_map,
    // from PAIR (A-A or A-B) species blocks:
    //  - set radial_functions.lambda, cut, dcut, crad and prehc and lambdahc,
    //  - rho_core_cutoffs, drho_core_cutoffs and FS_parameters
    for (auto &func_spec_block: basisSetup.funcspecs_blocks) {
        //below, only pair_species blocks (i.e. A-A or A-B) are considered
        if (func_spec_block.number_of_species != 2) continue;

        SPECIES_TYPE ele_i = elements_to_index_map[func_spec_block.elements_vec[0]];
        SPECIES_TYPE ele_j = elements_to_index_map[func_spec_block.elements_vec[1]];

        radial_functions->lambda(ele_i, ele_j) = func_spec_block.radparameters[0];
        radial_functions->cut(ele_i, ele_j) = func_spec_block.rcutij;
        radial_functions->dcut(ele_i, ele_j) = func_spec_block.dcutij;
        for (NS_TYPE n = 0; n < func_spec_block.nradmaxi; n++)
            for (LS_TYPE l = 0; l <= func_spec_block.lmaxi; l++)
                for (NS_TYPE k = 0; k < func_spec_block.nradbaseij; k++) {
                    radial_functions->crad(ele_i, ele_j, n, l, k) = func_spec_block.radcoefficients.at(n).at(l).at(k);
                }

        //set hard-core repulsion core-repulsion parameters:
        radial_functions->prehc(ele_i, ele_j) = func_spec_block.core_rep_parameters.at(0); //prehc
        radial_functions->lambdahc(ele_i, ele_j) = func_spec_block.core_rep_parameters.at(1); //lambdahc

        //set energy-based cutoff parameters
        rho_core_cutoffs(ele_i) = func_spec_block.rho_cut;
        drho_core_cutoffs(ele_i) = func_spec_block.drho_cut;

        //TODO: make multispecies
        FS_parameters = func_spec_block.fs_parameters;
        npoti = func_spec_block.npoti;
    } // end loop over pairs_species_blocks


    //invert "elements_to_index_map" to index->element array "elements_name"
    elements_name = new string[nelements];
    for (auto const &elem_ind : elements_to_index_map) {
        elements_name[elem_ind.second] = elem_ind.first;
    }

    radial_functions->setuplookupRadspline();

    //0 dim - X_0: ele_0; central element type(0..nelements-1)
    //1 dim - vector<C_tilde_B_basis_function> for different [X_, n_, l_]
//    mu0_ctilde_basis_vector.resize(nelements);
    mu0_bbasis_vector.resize(nelements);

    // loop over all B-basis functions specification blocks,
    // construction of actual ACEBBasisFunction
    for (auto species_block: basisSetup.funcspecs_blocks) { // n
        SPECIES_TYPE mu0 = elements_to_index_map[species_block.mu0];
        NS_TYPE *nr;
        LS_TYPE *ls;
        LS_TYPE *LS;
        DOUBLE_TYPE *cs;


        if (species_block.funcspecs.empty()) continue;

        //[basis_ind]
        vector<ACEBBasisFunction> &basis = mu0_bbasis_vector[mu0];

        for (auto &curr_bFuncSpec: species_block.funcspecs) {
            //auto &new_basis_func = basis[rank - 1][basis_ind];
            ACEBBasisFunction new_basis_func;
            //BBasisFunctionSpecification &curr_bFuncSpec = species_block.bfuncspec_vector[rank - 1][basis_ind];
            RANK_TYPE rank = curr_bFuncSpec.rank;
            nr = &curr_bFuncSpec.ns[0];
            ls = &curr_bFuncSpec.ls[0];
            cs = &curr_bFuncSpec.coeffs[0]; // len = ndensity
            if (rank > 2)
                LS = &curr_bFuncSpec.LS[0];
            else
                LS = nullptr;

            try {
                generate_basis_function_n_body(rank, nr, ls, LS, new_basis_func,
                                               clebsch_gordan, true);
            } catch (const invalid_argument &exc) {
                stringstream s;
                s << curr_bFuncSpec.to_string() << " could not be constructed: " << endl << exc.what();
                throw invalid_argument(s.str());
                // exit(EXIT_FAILURE);
            }

            new_basis_func.mu0 = elements_to_index_map[curr_bFuncSpec.elements.at(0)];
            //TODO: move new mus here
            for (RANK_TYPE r = 1; r <= rank; r++)
                new_basis_func.mus[r - 1] = elements_to_index_map[curr_bFuncSpec.elements.at(r)];

            new_basis_func.ndensity = species_block.ndensityi;
            new_basis_func.coeff = new DOUBLE_TYPE[new_basis_func.ndensity];
            for (DENSITY_TYPE p = 0; p < species_block.ndensityi; ++p)
                new_basis_func.coeff[p] = cs[p];

            if (num_ms_combinations_max < new_basis_func.num_ms_combs)
                num_ms_combinations_max = new_basis_func.num_ms_combs;

            basis.emplace_back(new_basis_func);

        } //end loop over rank
    }

    metadata = basisSetup.metadata;

    compress_basis_functions();
    flatten_basis();
    pack_flatten_basis();
}

void convert_B_to_Ctilde_basis_functions(const vector<ACEBBasisFunction> &b_basis_vector,
                                         vector<ACECTildeBasisFunction> &ctilde_basis_vector) {
    Basis_functions_map basis_functions_map;
    group_basis_functions_by_index(b_basis_vector, basis_functions_map);
#ifdef DEBUG_C_TILDE
    for(int r=0; r<rankmax; ++r) {
        auto basis_functions_map = basis_functions_map[r];
        if(basis_functions_map.empty()) continue;
        cout<<"rankmax="<<(int)r<<"\t";
        cout<<"number of b_basis_vector functions = "<<basis_functions_map.size()<<endl;

        for (auto & it : basis_functions_map) {
            const Vector_ns &num_ms_combs = get<1>(it.first);
            const Vector_ls &ls = get<2>(it.first);
            Basis_function_ptr_list bas_list = it.second;

            cout << "size=" << bas_list.size() << endl;

            for (auto &it : bas_list) {
                print_B_basis_function(*it);
            }
            cout << endl;

        }

    }
#endif

    summation_over_LS(basis_functions_map, ctilde_basis_vector);
}


void BBasisConfiguration::save(const string &yaml_file_name) {
//    throw runtime_error("BBasisConfiguration::save is not implemented");
    YAML::Node out_yaml;

    YAML::Node global_yaml;
    global_yaml["DeltaSplineBins"] = deltaSplineBins;


    vector<YAML::Node> species;
    for (auto &block: funcspecs_blocks) {
        YAML::Node block_yaml = block.to_YAML();
        species.emplace_back(block_yaml);
    }

    if (metadata.size() > 0)
        out_yaml["metadata"] = metadata;

    out_yaml["global"] = global_yaml;
    out_yaml["species"] = species;

    YAML::Emitter yaml_emitter;
//    yaml_emitter << YAML::Flow << out_yaml;
    yaml_emitter << out_yaml;
    std::ofstream fout(yaml_file_name);
    fout << yaml_emitter.c_str() << endl;
}

void BBasisConfiguration::load(const string &yaml_file_name) {
    Input input;
    input.parse_input(yaml_file_name);

    this->metadata = input.global.metadata;

    this->deltaSplineBins = input.global.DeltaSplineBins;
    this->funcspecs_blocks = input.bbasis_func_spec_blocks_vector;
}

vector<DOUBLE_TYPE> BBasisConfiguration::get_all_coeffs() const {
    vector<DOUBLE_TYPE> res;
    for (auto &block: this->funcspecs_blocks) {
        auto coeffs = block.get_all_coeffs();
        res.insert(end(res), begin(coeffs), end(coeffs));
    }
    return res;
}

void BBasisConfiguration::set_all_coeffs(const vector<DOUBLE_TYPE> &new_all_coeffs) {
    size_t ind = 0;
    for (auto &block: this->funcspecs_blocks) {
        size_t expected_num_of_coeffs = block.get_number_of_coeffs();
        vector<DOUBLE_TYPE> block_new_coeffs = vector<DOUBLE_TYPE>(new_all_coeffs.begin() + ind,
                                                                   new_all_coeffs.begin() + ind +
                                                                   expected_num_of_coeffs);
        block.set_all_coeffs(block_new_coeffs);
        ind += expected_num_of_coeffs;
    }

}


YAML::Node BBasisFunctionsSpecificationBlock::to_YAML() const {
    YAML::Node block_node;
    block_node["speciesblock"] = this->block_name; //join(this->elements_vec, " ");
    block_node["speciesblock"].SetStyle(YAML::EmitterStyle::Flow);
    block_node["nradmaxi"] = nradmaxi;
    block_node["lmaxi"] = lmaxi;
    block_node["ndensityi"] = ndensityi;
    block_node["npoti"] = npoti;
    block_node["parameters"] = fs_parameters;
    block_node["parameters"].SetStyle(YAML::EmitterStyle::Flow);

    block_node["rcutij"] = rcutij;
    block_node["dcutij"] = dcutij;
    block_node["NameOfCutoffFunctionij"] = NameOfCutoffFunctionij;


    block_node["nradbaseij"] = nradbaseij;

    block_node["radbase"] = radbase;
    block_node["radparameters"] = radparameters;
    block_node["radparameters"].SetStyle(YAML::EmitterStyle::Flow);

    block_node["radcoefficients"] = radcoefficients;
    block_node["radcoefficients"].SetStyle(YAML::EmitterStyle::Flow);

    block_node["core-repulsion"] = core_rep_parameters;
    block_node["core-repulsion"].SetStyle(YAML::EmitterStyle::Flow);

    block_node["rho_core_cut"] = rho_cut;
    block_node["drho_core_cut"] = drho_cut;

    vector<YAML::Node> nbody;

    for (auto &funcspec: funcspecs) {
        nbody.emplace_back(funcspec.to_YAML());
    }

    block_node["nbody"] = nbody;

    return block_node;
}

vector<DOUBLE_TYPE> BBasisFunctionsSpecificationBlock::get_all_coeffs() const {
    vector<DOUBLE_TYPE> res;
    for (NS_TYPE n = 0; n < this->nradmaxi; n++)
        for (LS_TYPE l = 0; l <= this->lmaxi; l++)
            for (NS_TYPE k = 0; k < this->nradbaseij; k++) {
                res.emplace_back(this->radcoefficients.at(n).at(l).at(k));
            }

    for (auto &f: this->funcspecs) {
        for (auto c: f.coeffs)
            res.emplace_back(c);
    }

    return res;
}

void BBasisFunctionsSpecificationBlock::set_all_coeffs(const vector<DOUBLE_TYPE> &new_coeffs) {
    size_t total_size = this->get_number_of_coeffs();
    if (total_size != new_coeffs.size())
        throw invalid_argument("Number of new coefficients " + to_string(new_coeffs.size()) +
                               " differs from expected number of coefficients: " + to_string(total_size));
    size_t ind = 0;

    for (NS_TYPE n = 0; n < this->nradmaxi; n++)
        for (LS_TYPE l = 0; l <= this->lmaxi; l++)
            for (NS_TYPE k = 0; k < this->nradbaseij; k++, ind++) {
                this->radcoefficients.at(n).at(l).at(k) = new_coeffs[ind];
            }

//    for (int i = 0; i < this->radcoefficients.size(); i++, ind++)
//        this->radcoefficients[i] = new_coeffs[ind];

    for (auto &spec: this->funcspecs) {
        for (DENSITY_TYPE p = 0; p < spec.coeffs.size(); p++, ind++)
            spec.coeffs[p] = new_coeffs[ind];
    }
}

int BBasisFunctionsSpecificationBlock::get_number_of_coeffs() const {
    size_t num = this->nradmaxi * (this->lmaxi + 1) * this->nradbaseij; //this->radcoefficients.size();
    for (auto &func: funcspecs) {
        num += func.coeffs.size();
    }
    return num;
}

void BBasisFunctionsSpecificationBlock::update_params() {
    int block_rankmax = -1;
    int func_n_density = -1;

    for (auto &funcSpec: funcspecs) {
        funcSpec.validate();
        //ndensity: initilaize first valid value of func_n_density
        if (func_n_density == -1) func_n_density = funcSpec.coeffs.size();

        if (func_n_density != funcSpec.coeffs.size()) {
            stringstream s;
            s << funcSpec.to_string() << ":" << endl
              << "Number of function 'coeffs'(" << funcSpec.coeffs.size()
              << ") is inconsistent with the expected density(" << func_n_density << ")";
            throw invalid_argument(s.str());
        }

        if (funcSpec.rank > block_rankmax) block_rankmax = funcSpec.rank;
    }

    this->elements_vec = split_key(this->block_name);
    if (this->elements_vec.size() == 1)
        this->elements_vec.emplace_back(this->elements_vec[0]);
    this->rankmax = block_rankmax;
    this->ndensityi = func_n_density;
    this->number_of_species = this->elements_vec.size();
    this->mu0 = this->elements_vec[0];
}

