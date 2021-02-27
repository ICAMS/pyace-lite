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

#ifndef ACE_COUPLINGS_H
#define ACE_COUPLINGS_H

#include <list>
#include <map>
#include <vector>
#include <string>
#include <sstream>

#include "ace_types.h"
#include "ace_b_basisfunction.h"
#include "ace_clebsch_gordan.h"
#include "ace_c_basisfunction.h"


struct ms_cg_pair {
    vector<MS_TYPE> ms;
    DOUBLE_TYPE c = 0;
    SHORT_INT_TYPE sign;
};

vector<SHORT_INT_TYPE> generate_coupling_tree(RANK_TYPE r);

int get_ms_sign(const RANK_TYPE rank, const MS_TYPE *ms);

int generate_basis_function_n_body(RANK_TYPE rank, const NS_TYPE *ns_rad, const LS_TYPE *ls, const LS_TYPE *LS,
                                   ACEBBasisFunction &b_basis_function, const ACEClebschGordan &cs,
                                   bool half_basis = false) noexcept (false);

int generate_ms_cg_list(const RANK_TYPE rank, const LS_TYPE *ls, const LS_TYPE *LS, const bool half_basis,
                        const ACEClebschGordan &cs, list<ms_cg_pair> &ms_cs_pairs_list)  noexcept(false);

bool validate_ls_LS(vector<LS_TYPE> ls, vector<LS_TYPE> LS);
void expand_ls_LS(RANK_TYPE rank, vector<LS_TYPE> &ls, vector<LS_TYPE> &LS);

class ACECouplingTree {

    void initialize_coupling_tree();

public:
    RANK_TYPE rank = 1;
    SHORT_INT_TYPE tree_map_size = 0;
    vector<SHORT_INT_TYPE> tree_indices_array;

    explicit ACECouplingTree(RANK_TYPE rank = 1);
};

class ACECouplingTreesCache {
public:
    RANK_TYPE rank_max = 0;
    vector<ACECouplingTree> coupling_trees_vector;

    explicit ACECouplingTreesCache(RANK_TYPE rank_max);

};


#endif
