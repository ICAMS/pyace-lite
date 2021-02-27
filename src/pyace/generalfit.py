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


import gc
import logging
from datetime import datetime
from functools import partial
from typing import Union, Dict, List, Callable

import numpy as np
import pandas as pd
from pyace.basis import ACEBBasisSet, BBasisConfiguration
from pyace.basisextension import construct_bbasisconfiguration, extend_basis, get_actual_ladder_step
from pyace.const import *
from pyace.fitadapter import FitBackendAdapter
from pyace.preparedata import get_fitting_dataset
from pyace.pyacefit import LossFunctionSpecification, required_structures_dataframe_columns, FWEIGHTS_COL, EWEIGHTS_COL, \
    ENERGY_CORRECTED_COL

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

__username = None

FITTING_DATA_INFO_FILENAME = "fitting_data_info.csv"


def get_username():
    global __username
    if __username is None:
        try:
            import getpass
            __username = getpass.getuser()
            log.info("User name automatically identified: {}".format(__username))
            return __username
        except ImportError:
            log.info("Couldn't automatically identify user name")
    else:
        return __username


class GeneralACEFit:
    """
    Main fitting wrapper class

    :param potential_config:  specification of the potential
                    - configuration dictionary
                    - YAML with BBasisConfiguration potential configuration
                    - BBasisConfiguration
                    - ACEBBasisSet
    :param fit_config:  specification of fitting (loss function, number of iterations, weighting policy, ...)
    :param data_config:  training data specification
    :param backend_config: specification of potential evaluation and fitting backend (pyace / tensorpot)
                    - dict ['evaluator']
    """

    def __init__(self,
                 potential_config: Union[Dict, str, BBasisConfiguration, ACEBBasisSet],
                 fit_config: Dict,
                 data_config: Union[Dict, pd.DataFrame],
                 backend_config: Dict,
                 cutoff=None,
                 seed=None,
                 callbacks=None
                 ):
        self.seed = seed
        if self.seed is not None:
            log.info("Set numpy random seed to {}".format(self.seed))
            np.random.seed(self.seed)

        self.callbacks = [save_interim_potential_callback]
        if callbacks is not None:
            if isinstance(callbacks, (list, tuple)):
                for c in callbacks:
                    if isinstance(c, Callable):
                        self.callbacks.append(c)
                        log.info("{} callback added".format(c))
                    elif isinstance(c, str):
                        log.info("")
                        c = active_import(c)
                        self.callbacks.append(c)
                        log.info("{} callback added".format(c))
            else:
                raise ValueError(
                    "'callbacks' should be list/tuple of importable function name or function with signature: callback" +
                    "(coeffs, bbasisconfig: BBasisConfiguration, current_fit_cycle: int, current_ladder_step: int). " +
                    "But got: {}".format(callbacks)
                )

        self.current_fit_iteration = 0
        self.ladder_scheme = False
        self.ladder_type = 'body_order'
        self.initial_bbasisconfig = None
        if isinstance(potential_config, dict):
            self.target_bbasisconfig = construct_bbasisconfiguration(potential_config)
            log.info("Target potential shape constructed from dictionary, it contains {} functions".format(
                self.target_bbasisconfig.total_number_of_functions))
            if POTENTIAL_INITIAL_POTENTIAL_KW in potential_config:
                start_potential = potential_config[POTENTIAL_INITIAL_POTENTIAL_KW]

                if isinstance(start_potential, str):
                    self.initial_bbasisconfig = BBasisConfiguration(start_potential)
                elif isinstance(start_potential, BBasisConfiguration):
                    self.initial_bbasisconfig = start_potential
                else:
                    raise ValueError("potential_config[`{}`] is neither str nor BBasisConfiguration".format(
                        POTENTIAL_INITIAL_POTENTIAL_KW))
                self.ladder_scheme = True
                log.info("Initial potential provided: {}, it contains {} functions".format(start_potential,
                                                                                           self.initial_bbasisconfig.total_number_of_functions))
                log.info("Ladder-scheme fitting is ON")
            elif FIT_LADDER_STEP_KW in fit_config:
                self.ladder_scheme = True
                self.initial_bbasisconfig = self.target_bbasisconfig.copy()
                for block in self.initial_bbasisconfig.funcspecs_blocks:
                    block.lmaxi = 0
                    block.nradmaxi = 0
                    block.nradbaseij = 0
                    block.radcoefficients = [[[]]]
                    block.funcspecs = []
                log.info("Ladder-scheme fitting is ON")
                log.info("Initial potential is NOT provided, starting from empty potential")
        elif isinstance(potential_config, str):
            self.target_bbasisconfig = BBasisConfiguration(potential_config)
            log.info("Target potential loaded from file '{}'".format(potential_config))
        elif isinstance(potential_config, BBasisConfiguration):
            self.target_bbasisconfig = potential_config
            log.info("Target potential provided as `BBasisConfiguration` object")
        elif isinstance(potential_config, ACEBBasisSet):
            self.target_bbasisconfig = potential_config.to_BBasisConfiguration()
            log.info("Target potential provided as `ACEBBasisSet` object")
        else:
            raise ValueError(
                ("Non-supported type: {}. Only dictionary (configuration), " +
                 "str (YAML file name) or BBasisConfiguration are supported").format(
                    type(potential_config)))
        # TODO: hardcoded
        if cutoff is None:
            self.cutoff = self.target_bbasisconfig.funcspecs_blocks[0].rcutij
        else:
            self.cutoff = cutoff

        if self.ladder_scheme:
            if FIT_LADDER_TYPE_KW in fit_config:
                self.ladder_type = str(fit_config[FIT_LADDER_TYPE_KW])
            log.info("Ladder_type: {} is selected".format(self.ladder_type))

        self.fit_config = fit_config
        if FIT_OPTIMIZER_KW not in self.fit_config:
            self.fit_config[FIT_OPTIMIZER_KW] = "BFGS"
            log.warning("'{}' is not provided, switch to default value: {}".format(FIT_OPTIMIZER_KW,
                                                                                   self.fit_config[FIT_OPTIMIZER_KW]))
        if FIT_NITER_KW not in self.fit_config:
            self.fit_config[FIT_NITER_KW] = 100
            log.warning(
                "'{}' is not provided, switch to default value: {}".format(FIT_NITER_KW, self.fit_config[FIT_NITER_KW]))

        if FIT_OPTIONS_KW in self.fit_config:
            log.info(
                "optimizer options are provided: '{}'".format(self.fit_config[FIT_OPTIONS_KW]))

        self.data_config = data_config
        self.weighting_policy_spec = self.fit_config.get(FIT_WEIGHTING_KW)
        self.fit_backend = FitBackendAdapter(backend_config)
        self.evaluator_name = self.fit_backend.evaluator_name

        set_general_metadata(self.target_bbasisconfig)

        if isinstance(self.data_config, (dict, str)):
            self.fitting_data = get_fitting_dataset(evaluator_name=self.evaluator_name,
                                                    data_config=self.data_config,
                                                    weighting_policy_spec=self.weighting_policy_spec,
                                                    cutoff=self.cutoff
                                                    )
        elif isinstance(self.data_config, pd.DataFrame):
            self.fitting_data = self.data_config
        else:
            raise ValueError("'data-config' should be dictionary or pd.DataFrame")

        self.save_fitting_data_info()

        self.loss_spec = LossFunctionSpecification(**self.fit_config.get(FIT_LOSS_KW, {}))

    def save_fitting_data_info(self):
        # columns to save: w_energy, w_forces, NUMBER_OF_ATOMS, PROTOTYPE_NAME, prop_id,structure_id, gen_id, if any
        columns_to_save = ["PROTOTYPE_NAME", "NUMBER_OF_ATOMS", "prop_id", "structure_id", "gen_id", "pbc"] + \
                          [ENERGY_CORRECTED_COL, EWEIGHTS_COL, FWEIGHTS_COL]

        fitting_data_columns = self.fitting_data.columns

        columns_to_save = [col for col in columns_to_save if col in fitting_data_columns]

        self.fitting_data[columns_to_save].to_csv(FITTING_DATA_INFO_FILENAME, index=None, sep=",")
        log.info("Fitting dataset info saved into {}".format(FITTING_DATA_INFO_FILENAME))

    def fit(self) -> BBasisConfiguration:
        gc.collect()
        self.target_bbasisconfig.save(INITIAL_POTENTIAL_YAML)

        log.info("Fitting dataset size: {} structures / {} atoms".format(len(self.fitting_data),
                                                                         self.fitting_data["NUMBER_OF_ATOMS"].sum()))
        if not self.ladder_scheme:  # normal "non-ladder" fit
            log.info("'Single-shot' fitting")
            self.target_bbasisconfig = self.cycle_fitting(self.target_bbasisconfig, current_ladder_step=0)
        else:  # ladder scheme
            log.info("'Ladder-scheme' fitting")
            self.target_bbasisconfig = self.ladder_fitting(self.initial_bbasisconfig, self.target_bbasisconfig)

        log.info("Fitting done")
        return self.target_bbasisconfig

    def ladder_fitting(self, initial_config, target_config):
        total_number_of_funcs = target_config.total_number_of_functions
        ladder_step_param = self.fit_config.get(FIT_LADDER_STEP_KW, 0.1)

        current_bbasisconfig = initial_config.copy()
        current_ladder_step = 0
        while True:
            prev_func_num = current_bbasisconfig.total_number_of_functions
            log.info("Current basis set size: {} B-functions".format(prev_func_num))
            ladder_step = get_actual_ladder_step(ladder_step_param, prev_func_num, total_number_of_funcs)
            log.info("Ladder step size: {}".format(ladder_step))
            current_bbasisconfig = extend_basis(current_bbasisconfig, target_config, self.ladder_type, ladder_step)
            new_func_num = current_bbasisconfig.total_number_of_functions
            log.info("Extended basis set size: {} B-functions".format(new_func_num))

            if prev_func_num == new_func_num:
                log.info("No new function added after basis extension. Stopping")
                break

            current_bbasisconfig = self.cycle_fitting(current_bbasisconfig, current_ladder_step=current_ladder_step)

            if "_fit_cycles" in current_bbasisconfig.metadata:
                del current_bbasisconfig.metadata["_fit_cycles"]
            log.debug("Update metadata: {}".format(current_bbasisconfig.metadata))
            save_interim_potential(current_bbasisconfig, current_bbasisconfig.get_all_coeffs())
            current_ladder_step += 1
        return current_bbasisconfig

    def cycle_fitting(self, bbasisconfig: BBasisConfiguration, current_ladder_step: int = 0) -> BBasisConfiguration:
        current_bbasisconfig = bbasisconfig.copy()
        log.info('Cycle fitting loop')

        fit_cycles = int(self.fit_config.get(FIT_FIT_CYCLES_KW, 1))
        noise_rel_sigma = float(self.fit_config.get(FIT_NOISE_REL_SIGMA, 0))
        noise_abs_sigma = float(self.fit_config.get(FIT_NOISE_ABS_SIGMA, 0))

        if "_" + FIT_FIT_CYCLES_KW in current_bbasisconfig.metadata:
            finished_fit_cycles = int(current_bbasisconfig.metadata["_" + FIT_FIT_CYCLES_KW])
        else:
            finished_fit_cycles = 0

        if finished_fit_cycles >= fit_cycles:
            log.warning(
                ("Number of finished fit cycles ({}) >= number of expected fit cycles ({}). " +
                 "Use another potential or remove `{}` from potential metadata")
                    .format(finished_fit_cycles, fit_cycles, "_" + FIT_FIT_CYCLES_KW))
            return current_bbasisconfig

        fitting_attempts_list = []
        while finished_fit_cycles < fit_cycles:
            current_fit_cycle = finished_fit_cycles + 1
            log.info("Number of fit attempts: {}/{}".format(current_fit_cycle, fit_cycles))
            num_of_functions = current_bbasisconfig.total_number_of_functions
            num_of_parameters = len(current_bbasisconfig.get_all_coeffs())
            log.info("Total number of functions: {} / number of parameters: {}".format(num_of_functions,
                                                                                       num_of_parameters))
            log.info("Running fit backend")
            self.current_fit_iteration = 0
            current_bbasisconfig = self.fit_backend.fit(
                current_bbasisconfig,
                dataframe=self.fitting_data, loss_spec=self.loss_spec, fit_config=self.fit_config,
                callback=partial(self.callback_hook, basis_config=bbasisconfig, current_fit_cycle=current_fit_cycle,
                                 current_ladder_step=current_ladder_step)
            )

            log.info("Fitting cycle finished, final statistic:")
            self.fit_backend.print_detailed_metrics(prefix='Last iteration:')

            finished_fit_cycles = current_fit_cycle

            current_bbasisconfig.metadata["_" + FIT_FIT_CYCLES_KW] = str(finished_fit_cycles)
            current_bbasisconfig.metadata["_" + FIT_LOSS_KW] = str(self.fit_backend.res_opt.fun)
            log.debug("Update current_bbasisconfig.metadata = {}".format(current_bbasisconfig.metadata))

            fitting_attempts_list.append((np.sum(self.fit_backend.res_opt.fun), current_bbasisconfig.copy()))

            # select current_bbasisconfig as a best among all previous
            best_ind = np.argmin([v[0] for v in fitting_attempts_list])
            log.info(
                "Select best fit #{} among all available ({})".format(best_ind + 1, len(fitting_attempts_list)))
            current_bbasisconfig = fitting_attempts_list[best_ind][1].copy()

            if finished_fit_cycles < fit_cycles and (noise_rel_sigma > 0) or (noise_abs_sigma > 0):
                all_coeffs = current_bbasisconfig.get_all_coeffs()
                noisy_all_coeffs = all_coeffs
                if noise_rel_sigma > 0:
                    log.info(
                        "Applying Gaussian noise with relative sigma/mean = {:>1.4e} to all optimizable coefficients".format(
                            noise_rel_sigma))
                    noisy_all_coeffs = apply_noise(all_coeffs, noise_rel_sigma, relative=True)
                elif noise_abs_sigma > 0:
                    log.info(
                        "Applying Gaussian noise with sigma = {:>1.4e} to all optimizable coefficients".format(
                            noise_abs_sigma))
                    noisy_all_coeffs = apply_noise(all_coeffs, noise_abs_sigma, relative=False)
                current_bbasisconfig.set_all_coeffs(noisy_all_coeffs)

        # chose the best fit attempt among fitting_attempts_list
        best_fitting_attempts_ind = np.argmin([v[0] for v in fitting_attempts_list])
        log.info("Best fitting attempt is #{}".format(best_fitting_attempts_ind + 1))
        current_bbasisconfig = fitting_attempts_list[best_fitting_attempts_ind][1]
        save_interim_potential(current_bbasisconfig)
        return current_bbasisconfig

    def save_optimized_potential(self, potential_filename: str = "output_potential.yaml"):
        if "_" + FIT_FIT_CYCLES_KW in self.target_bbasisconfig.metadata:
            del self.target_bbasisconfig.metadata["_" + FIT_FIT_CYCLES_KW]

        log.debug("Update metadata: {}".format(self.target_bbasisconfig.metadata))
        self.target_bbasisconfig.save(potential_filename)
        log.info("Final potential is saved to {}".format(potential_filename))

    def callback_hook(self, coeffs, basis_config: BBasisConfiguration, current_fit_cycle: int,
                      current_ladder_step: int):
        # TODO add a list of callbacks

        basis_config = basis_config.copy()
        safely_update_bbasisconfiguration_coefficients(coeffs, basis_config)
        for callback in self.callbacks:
            callback(
                basis_config=basis_config,
                current_fit_iteration=self.current_fit_iteration,
                current_fit_cycle=current_fit_cycle,
                current_ladder_step=current_ladder_step,
            )
        self.current_fit_iteration += 1


def apply_noise(all_coeffs: Union[np.array, List], sigma: float, relative: bool = True) -> np.array:
    coeffs = np.array(all_coeffs)
    noise = np.random.randn(*coeffs.shape)
    if relative:
        base_coeffs = np.abs(coeffs)
        # clip minimal values
        base_coeffs[base_coeffs < 1e-2] = 1e-2
        coeffs = coeffs + noise * sigma * base_coeffs
    else:
        coeffs = coeffs + noise * sigma
    return coeffs


def set_general_metadata(bbasisconfig: BBasisConfiguration) -> None:
    bbasisconfig.metadata[METADATA_STARTTIME_KW] = str(datetime.now())
    if get_username() is not None:
        bbasisconfig.metadata[METADATA_USER_KW] = str(get_username())


def safely_update_bbasisconfiguration_coefficients(coeffs: np.array, config: BBasisConfiguration = None) -> None:
    current_coeffs = config.get_all_coeffs()
    for i, c in enumerate(coeffs):
        current_coeffs[i] = c
    config.set_all_coeffs(current_coeffs)


def save_interim_potential(basis_config: BBasisConfiguration, coeffs=None, potential_filename="interim_potential.yaml",
                           verbose=True):
    if coeffs is not None:
        basis_config = basis_config.copy()
        safely_update_bbasisconfiguration_coefficients(coeffs, basis_config)
    basis_config.metadata["intermediate_time"] = str(datetime.now())
    basis_config.save(potential_filename)
    if verbose:
        log.info('Intermediate potential saved in {}'.format(potential_filename))


def save_interim_potential_callback(basis_config: BBasisConfiguration, current_fit_iteration: int,
                                    current_fit_cycle: int,
                                    current_ladder_step: int):
    save_interim_potential(basis_config=basis_config,
                           potential_filename="interim_potential_{}.yaml".format(current_fit_cycle),
                           verbose=False)


def active_import(name):
    """
    This function will import the
    :param name:
    :type name:
    :return:
    :rtype:
    """
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
