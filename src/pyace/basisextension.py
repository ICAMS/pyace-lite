# /*
# * Atomic cluster expansion
# *
# * Copyright 2021  (c) Yury Lysogorskiy, Anton Bochkarev,
# * Sarath Menon, Ralf Drautz
# *
# * Ruhr-University Bochum, Bochum, Germany
# *
# * See the LICENSE file.
# * This FILENAME is free software: you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation, either version 3 of the License, or
# * (at your option) any later version.
#
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
#     * You should have received a copy of the GNU General Public License
# * along with this program.  If not, see <http://www.gnu.org/licenses/>.
# */

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from typing import List, Dict, Union

from pyace.basis import ACEBBasisSet, BBasisConfiguration, BBasisFunctionsSpecificationBlock, \
    BBasisFunctionSpecification
from pyace.const import *

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def is_basisfunc_equivalent(func1: BBasisFunctionSpecification, func2: BBasisFunctionSpecification) -> bool:
    return (func1.ns == func2.ns) and (func1.ls == func2.ls) and (func1.LS == func2.LS)


def is_f1_eq_f2(f1: BBasisFunctionSpecification, f2: BBasisFunctionSpecification) -> bool:
    return tuple(f1.ns) == tuple(f2.ns) and tuple(f1.ls) == tuple(f2.ls) and tuple(f1.LS) == tuple(f2.LS)


def is_f1_greater_than_f2(f1: BBasisFunctionSpecification, f2: BBasisFunctionSpecification) -> bool:
    return f2 < f1


def is_f1_less_f2(f1, f2):
    return (tuple(f1.ns), tuple(f1.ls), tuple(f1.LS)) < (tuple(f2.ns), tuple(f2.ls), tuple(f2.LS))


def get_actual_ladder_step(ladder_step_param: Union[int, float, List],
                           current_number_of_funcs: int,
                           final_number_of_funcs: int) -> int:
    ladder_discrete_step: int = 0
    ladder_frac: float = 0.0
    val_exc = ValueError(
        "Invalid ladder step parameter: {}. Should be integer >= 1 or  0<float<1 or list of both [int, float]".format(
            ladder_step_param))
    if isinstance(ladder_step_param, int) and ladder_step_param >= 1:
        ladder_discrete_step = int(ladder_step_param)
    elif isinstance(ladder_step_param, float) and 1. > ladder_step_param > 0:
        ladder_frac = float(ladder_step_param)
    elif isinstance(ladder_step_param, (list, tuple)):
        if len(ladder_step_param) > 2:
            raise val_exc
        for p in ladder_step_param:
            if p >= 1:
                ladder_discrete_step = int(p)
            elif 0 < p < 1:
                ladder_frac = float(p)
            else:
                raise val_exc
    else:
        raise val_exc

    ladder_frac_step = int(round(ladder_frac * current_number_of_funcs))
    ladder_step = max(ladder_discrete_step, ladder_frac_step, 1)
    log.info(
        'Possible ladder steps: discrete - {}, fraction - {}. Selected maximum - {}'.format(ladder_discrete_step,
                                                                                            ladder_frac_step,
                                                                                            ladder_step))

    if current_number_of_funcs + ladder_step > final_number_of_funcs:
        ladder_step = final_number_of_funcs - current_number_of_funcs
        log.info("Ladder step is too large and adjusted to {}".format(ladder_step))

    return ladder_step


class BasisFuncsList:
    __slots__ = ['nradmax', 'lmax', 'rank', 'funcs']

    def __init__(self, basisConfig: BBasisConfiguration):
        self.funcs = defaultdict(list)  # per-rank list
        self.nradmax = defaultdict(tuple)  # per-rank value
        self.lmax = defaultdict(tuple)  # per-rank value
        for block in basisConfig.funcspecs_blocks:

            for func in block.funcspecs:
                rank = len(func.ns)
                self.funcs[rank].append(func)

                self.nradmax[rank] = max(tuple(func.ns), self.nradmax[rank])

                self.lmax[rank] = max(tuple(func.ls), self.lmax[rank])

        for k, v in self.funcs.items():
            self.funcs[k] = sort_funcspecs_list(v, 'body_order')

    def find_existing(self, func: BBasisFunctionSpecification) -> bool:
        rank = len(func.ns)
        for other_func in self.funcs[rank]:
            if is_basisfunc_equivalent(func, other_func):
                return other_func
        return None

    def at_ns_area_border(self, func: BBasisFunctionSpecification) -> bool:
        rank = len(func.ns)
        nradmax = tuple(func.ns)
        return nradmax == self.nradmax[rank]

    def is_max_func(self, func: BBasisFunctionSpecification) -> bool:
        rank = len(func.ns)
        for existing_func in self.funcs[rank]:
            if is_f1_less_f2(func, existing_func) or is_f1_eq_f2(func, existing_func):
                return False
        return True

    def escribed_area_contains(self, func: BBasisFunctionSpecification) -> bool:
        rank = len(func.ns)
        nradmax = tuple(func.ns)
        lmax = tuple(func.ls)
        res = (nradmax <= self.nradmax[rank]) and (lmax <= self.lmax[rank])
        return res


def prepare_bbasisfuncspecifications(potential_config: Dict) -> List[BBasisFunctionSpecification]:
    try:
        if potential_config[POTENTIAL_BASISDF_KW].endswith("gzip"):
            compression = "gzip"
        else:
            compression = "infer"
        funcs_df = pd.read_pickle(potential_config[POTENTIAL_BASISDF_KW], compression=compression)
    except Exception as e:
        log.error(
            "Couldn't load  basis specifications from  {}, error: {}".format(
                potential_config[POTENTIAL_BASISDF_KW],
                str(e)))
        raise e

    funcs_df = funcs_df.loc[funcs_df['rank'] <= potential_config[POTENTIAL_RANKMAX_KW]]
    mask = []

    for k in range(potential_config[POTENTIAL_RANKMAX_KW]):
        df = funcs_df.loc[funcs_df['rank'] == k + 1]
        ns_val = np.array([np.array(v.ns) for v in df['spec'].values])
        if k == 0:
            mask_ns = [(v <= potential_config[POTENTIAL_NRADMAX_KW][k]).all() for v in ns_val]
            mask.append(mask_ns)
        else:
            ls_val = np.array([np.array(v.ls) for v in df['spec'].values])
            mask_ns = [(v <= potential_config[POTENTIAL_NRADMAX_KW][k]).all() for v in ns_val]
            mask_ls = [(v <= potential_config[POTENTIAL_LMAX_KW][k]).all() for v in ls_val]
            mask.append(np.logical_and(mask_ls, mask_ns))
    mask = np.hstack(mask)
    selected_df = funcs_df.loc[mask]

    # zero the coefficients
    spec_list = selected_df['spec'].tolist()
    element = potential_config['element']
    for spec in spec_list:
        spec.coeffs = np.ones(potential_config.get(POTENTIAL_NDENSITY_KW, 1)) * 0.0
        spec.elements = [element] * (len(spec.ns) + 1)
    return spec_list


def construct_bbasisconfiguration(potential_config: Dict) -> BBasisConfiguration:
    assert len(potential_config[POTENTIAL_NRADMAX_KW]) == potential_config[POTENTIAL_RANKMAX_KW], \
        "Length of the {} do not match {}".format(POTENTIAL_NRADMAX_KW, POTENTIAL_RANKMAX_KW)
    assert len(potential_config[POTENTIAL_LMAX_KW]) == potential_config[POTENTIAL_RANKMAX_KW], \
        "Length of the {} do not match {}".format(POTENTIAL_LMAX_KW, POTENTIAL_RANKMAX_KW)

    block = BBasisFunctionsSpecificationBlock()
    block.block_name = potential_config[POTENTIAL_ELEMENT_KW]  # "Al"
    block.elements_vec = [potential_config[POTENTIAL_ELEMENT_KW]]  # ["Al"]
    if potential_config[POTENTIAL_RANKMAX_KW] == 1:
        block.nradmaxi = 0
    else:
        block.nradmaxi = np.max(potential_config[POTENTIAL_NRADMAX_KW][1:])
    block.lmaxi = np.max(potential_config[POTENTIAL_LMAX_KW])
    block.npoti = potential_config[POTENTIAL_NPOT_KW]  # "FinnisSinclair"
    block.fs_parameters = potential_config[POTENTIAL_FS_PARAMETERS_KW]  # [1, 1, 1, 0.5]
    block.rcutij = potential_config[POTENTIAL_RCUT_KW]
    block.dcutij = potential_config[POTENTIAL_DCUT_KW]  # 0.01
    block.NameOfCutoffFunctionij = potential_config[POTENTIAL_CUTOFF_FUNCTION_KW]  # "cos"
    block.nradbaseij = potential_config[POTENTIAL_NRADMAX_KW][0]

    block.radbase = potential_config[POTENTIAL_RADBASE_KW]  # "ChebExpCos"
    block.radparameters = potential_config[POTENTIAL_RADPARAMETERS_KW]  # ex  lmbda

    if POTENTIAL_RADCOEFFICIENTS_KW not in potential_config:
        # crad shape [nelements][nelements][nradial][lmax + 1][nradbase] -> Kronecker delta (nrad=nbase)
        lmaxp1 = block.lmaxi + 1
        nmax = block.nradmaxi
        kmax = block.nradbaseij
        crad = np.zeros((nmax, lmaxp1, kmax))
        for n in range(nmax):
            crad[n, :, n] = 1.0
    else:
        crad = potential_config[POTENTIAL_RADCOEFFICIENTS_KW]

    block.radcoefficients = crad
    block.funcspecs = prepare_bbasisfuncspecifications(potential_config)

    if "core-repulsion" in potential_config:
        block.core_rep_parameters = potential_config["core-repulsion"]

    if "rho_core_cut" in potential_config:
        block.rho_cut = potential_config["rho_core_cut"]

    if "drho_core_cut" in potential_config:
        block.drho_cut = potential_config["drho_core_cut"]

    basis_configuration = BBasisConfiguration()
    basis_configuration.deltaSplineBins = potential_config[POTENTIAL_DELTASPLINEBINS_KW]  # 0.001
    basis_configuration.funcspecs_blocks = [block]
    if POTENTIAL_METADATA_KW in potential_config:
        for k, v in potential_config[POTENTIAL_METADATA_KW].items():
            basis_configuration.metadata[k] = v
    return basis_configuration


def sort_funcspecs_list(lst: List[BBasisFunctionSpecification], ltype: str) -> List[BBasisFunctionSpecification]:
    if ltype == 'power_order':
        return list(sorted(lst, key=lambda func: len(func.ns)+sum(func.ns)+sum(func.ls)))
    elif ltype == 'body_order':
        return list(sorted(lst, key=lambda func: (tuple(func.elements), tuple(func.ns), tuple(func.ls), tuple(func.LS))))
    else:
        raise ValueError('Specified Ladder type ({}) is not implemented'.format(ltype))


def extend_basis(initial_basis: BBasisConfiguration, final_basis: BBasisConfiguration,
                 ladder_type: str, func_step: int = None) -> BBasisConfiguration:
    if initial_basis.total_number_of_functions == final_basis.total_number_of_functions:
        return initial_basis.copy()
    # grow basis by func_step
    initial_basis_funcs_list = BasisFuncsList(initial_basis)

    final_basis_funcs = []
    for block in final_basis.funcspecs_blocks:
        final_basis_funcs += block.funcspecs

    final_basis_funcs = sort_funcspecs_list(final_basis_funcs, ladder_type)

    new_func_list = []
    existing_func_list = []

    skipped_functions = 0

    for new_func in final_basis_funcs:
        if initial_basis_funcs_list.escribed_area_contains(new_func):
            existing_func = initial_basis_funcs_list.find_existing(new_func)
            if existing_func is not None:
                existing_func_list.append(existing_func)  # copy with existing coefficients
            elif initial_basis_funcs_list.at_ns_area_border(new_func):
                ## ASSUME CORNER CASE NOT A HOLE!!!
                if initial_basis_funcs_list.is_max_func(new_func):
                    new_func_list.append(new_func)
                elif ladder_type != 'body_order':
                    new_func_list.append(new_func)
                else:
                    skipped_functions += 1  # skip, as non corner case
            # For other types of ladder growth the possibility of having hole is not considered
            elif ladder_type != 'body_order':
                new_func_list.append(new_func)
            else:
                skipped_functions += 1  # skip, because it is hole
        else:  # add new, green zone
            new_func_list.append(new_func)

    log.info("Skipped functions number: {}".format(skipped_functions))

    # new_func_list = sort_funcspecs_list(new_func_list, 'std_ranking')

    if func_step is not None and len(new_func_list) > func_step:
        new_func_list = new_func_list[:func_step]

    new_func_list = sort_funcspecs_list(new_func_list, 'body_order')

    new_basis_config = initial_basis.copy()
    # TODO: currentlu  only single func spec block is assumed
    new_basis_config.funcspecs_blocks[0].funcspecs = sort_funcspecs_list(existing_func_list + new_func_list, 'body_order')
    # update nradmax, lmax, nradabse
    new_nradmax = 0
    new_nradbase = 0
    new_lmax = 0
    new_rankmax = 0
    for func in new_basis_config.funcspecs_blocks[0].funcspecs:
        rank = len(func.ns)
        new_rankmax = max(rank, new_rankmax)
        if rank == 1:
            new_nradbase = max(max(func.ns), new_nradbase)
        else:
            new_nradmax = max(max(func.ns), new_nradmax)
        new_lmax = max(max(func.ls), new_lmax)
    new_basis_config.funcspecs_blocks[0].lmaxi = new_lmax
    new_basis_config.funcspecs_blocks[0].nradmaxi = new_nradmax
    new_basis_config.funcspecs_blocks[0].nradbaseij = new_nradbase
    # update crad
    old_crad = np.array(new_basis_config.funcspecs_blocks[0].radcoefficients)
    new_crad = np.zeros((new_nradmax, new_lmax + 1, new_nradbase))
    for n in range(min(new_nradmax, new_nradbase)):
        new_crad[n, :, n] = 1.
    # print("old_crad.shape = ", old_crad.shape)
    # print("new_crad.shape = ", new_crad.shape)
    if old_crad.shape != (0,):
        common_shape = [min(s1, s2) for s1, s2 in zip(old_crad.shape, new_crad.shape)]
        new_crad[:common_shape[0], :common_shape[1], :common_shape[2]] = old_crad[:common_shape[0], :common_shape[1],
                                                                         :common_shape[2]]
    new_basis_config.funcspecs_blocks[0].radcoefficients = new_crad

    # core-repulsion translating from final_basis
    new_basis_config.funcspecs_blocks[0].core_rep_parameters = final_basis.funcspecs_blocks[0].core_rep_parameters
    new_basis_config.funcspecs_blocks[0].rho_cut = final_basis.funcspecs_blocks[0].rho_cut
    new_basis_config.funcspecs_blocks[0].drho_cut = final_basis.funcspecs_blocks[0].drho_cut

    return new_basis_config
