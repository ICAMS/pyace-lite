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
import numpy as np
import pandas as pd
import time

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

from functools import partial
from typing import Union
from scipy.optimize import minimize

from pyace.basis import ACEBBasisSet, BBasisConfiguration
from pyace.evaluator import ACECTildeEvaluator, ACEBEvaluator
from pyace.calculator import ACECalculator
from pyace.paralleldataexecutor import ParallelDataExecutor, LOCAL_DATAFRAME_VARIALBE_NAME
from pyace.radial import *

import __main__

# Column names: string constants
ATOMIC_ENV_COL = "atomic_env"
ENERGY_CORRECTED_COL = "energy_corrected"
ENERGY_PRED_COL = "energy_pred"
EWEIGHTS_COL = "w_energy"
FORCES_COL = "forces"
FORCES_PRED_COL = "forces_pred"
FWEIGHTS_COL = "w_forces"

required_structures_dataframe_columns = [ATOMIC_ENV_COL, ENERGY_CORRECTED_COL, FORCES_COL]


def batch_compute_energy_forces_function_wrapper(batch_indices, cbasis):
    _local_df = getattr(__main__, LOCAL_DATAFRAME_VARIALBE_NAME)
    batch_df = _local_df.loc[batch_indices]

    ace = ACECalculator()
    evaluator = ACECTildeEvaluator()
    evaluator.set_basis(cbasis)
    ace.set_evaluator(evaluator)

    def pure_row_func(ae):
        ace.compute(ae)
        return ace.energy, ace.forces

    if isinstance(batch_df, pd.Series):
        return batch_df.map(pure_row_func)
    elif isinstance(batch_df, pd.DataFrame):
        return batch_df.apply(pure_row_func, axis=1)


def batch_compute_projections_function_wrapper(batch_indices, potential_params):
    _local_df = getattr(__main__, LOCAL_DATAFRAME_VARIALBE_NAME)
    batch_df = _local_df.loc[batch_indices]

    ace = ACECalculator()

    if isinstance(potential_params, BBasisConfiguration):
        bbasis = ACEBBasisSet(potential_params)
        evaluator = ACEBEvaluator()
        evaluator.set_basis(bbasis)
    elif isinstance(potential_params, ACECTildeBasisSet):
        evaluator = ACECTildeEvaluator()
        evaluator.set_basis(potential_params)
    elif isinstance(potential_params, ACEBBasisSet):
        evaluator = ACEBEvaluator()
        evaluator.set_basis(potential_params)
    else:
        raise ValueError("Unrecognized `potential_params` type: {}. Should be BBasisConfiguration, ACECTildeBasisSet "
                         "or ACEBBasisSet".format(type(potential_params)))

    ace.set_evaluator(evaluator)

    def pure_row_func(ae):
        ace.compute(ae)
        nat = ae.n_atoms_real
        proj1 = np.reshape(ace.basis_projections_rank1, (nat, -1))
        proj2 = np.reshape(ace.basis_projections, (nat, -1))
        return np.concatenate([proj1, proj2], axis=1)

    if isinstance(batch_df, pd.Series):
        return batch_df.map(pure_row_func)
    elif isinstance(batch_df, pd.DataFrame):
        return batch_df.apply(pure_row_func, axis=1)


class LossFunctionSpecification:

    def __init__(self, kappa=0.0,
                 L1_coeffs=0,
                 w1_coeffs=0,
                 L2_coeffs=0,
                 w2_coeffs=0,
                 w0_rad=0,
                 w1_rad=0,
                 w2_rad=0, **kwargs):
        # super(LossFunctionSpecification, self).__init__(self)
        self.kappa = kappa
        self.L1_coeffs = L1_coeffs
        self.w1_coeffs = w1_coeffs
        self.L2_coeffs = L2_coeffs
        self.w2_coeffs = w2_coeffs

        self.w0_rad = w0_rad
        self.w1_rad = w1_rad
        self.w2_rad = w2_rad

    def __str__(self):
        return ("LossFunctionSpecification(kappa={kappa}, L1={L1_coeffs}, " +
                "w1_coeffs={w1_coeffs}, " +
                "L2={L2_coeffs}, " +
                "w2_coeffs={w2_coeffs}, " +
                "DeltaRad=({w0_rad}, {w1_rad}, {w2_rad}))").format(
            kappa=self.kappa,
            L1_coeffs=self.L1_coeffs,
            w1_coeffs=self.w1_coeffs,
            L2_coeffs=self.L2_coeffs,
            w2_coeffs=self.w2_coeffs,
            w0_rad=self.w0_rad,
            w1_rad=self.w1_rad,
            w2_rad=self.w2_rad,
        )


class PyACEFit:
    """
    Create a class for fitting ACE potential

    :param basis (BBasisConfiguration, ACEBBasisSet) basis set specification
    :param loss_spec (LossFunctionSpecification)
    """

    def __init__(self, basis: Union[BBasisConfiguration, ACEBBasisSet] = None,
                 loss_spec: LossFunctionSpecification = None, seed=None, executors_kw_args=None, display_step=10):

        if basis is not None:
            if isinstance(basis, BBasisConfiguration):
                self.bbasis = ACEBBasisSet(basis)
            elif isinstance(basis, ACEBBasisSet):
                self.bbasis = basis
            else:
                raise ValueError(
                    "`basis` argument should be either 'BBasisConfiguration' or 'ACEBBasisSet', but got " + type(basis))
            self._init_params()
            self.cbasis = self.bbasis.to_ACECTildeBasisSet()
        else:
            self.bbasis = None
            self.cbasis = None
            self.params = None

        if loss_spec is None:
            loss_spec = LossFunctionSpecification()
        self.loss_spec = loss_spec
        self.seed = seed

        self._structures_dataframe = None

        self.current_func_eval = 0
        self.iter_num = 0

        self.best_loss = None
        self.best_params = None
        self.res_opt = None

        self.params_opt = None
        self.bbasis_opt = None
        self.cbasis_opt = None

        self.data_executor = None
        self.executors_kw_args = executors_kw_args or {}

        self.global_callback = None

        self.initial_loss = None
        self.last_loss = None
        self.last_epa_mae = None
        self.l1 = None
        self.l2 = None
        self.smooth_quad = None
        self.metrics = None
        self.eval_time = None
        self.display_step = display_step

    def _init_params(self):
        self.params = self.bbasis.all_coeffs

    @property
    def structures_dataframe(self):
        return self._structures_dataframe

    @structures_dataframe.setter
    def structures_dataframe(self, value):
        self._structures_dataframe = value  # .copy()
        # self.preprocess_dataframe(self._structures_dataframe)

    def preprocess_dataframe(self, structures_dataframe):
        # TODO: energies and forces weights are generated here, if columns not provided
        for col in required_structures_dataframe_columns:
            if col not in structures_dataframe.columns:
                raise ValueError("`structures_dataframe` doesn't contain column {}".format(col))

        if FORCES_COL in structures_dataframe.columns:
            structures_dataframe[FORCES_COL] = structures_dataframe[FORCES_COL].map(np.array)
        if FWEIGHTS_COL in structures_dataframe.columns:
            structures_dataframe[FWEIGHTS_COL] = structures_dataframe[FWEIGHTS_COL].map(np.array)

    def update_bbasis(self, params):
        self.bbasis.all_coeffs = params
        return self.bbasis

    def get_cbasis(self, params):
        self.bbasis = self.update_bbasis(params)
        self.cbasis = self.bbasis.to_ACECTildeBasisSet()
        return self.cbasis

    def loss(self, params, verbose=False):
        self.current_func_eval += 1
        t0 = time.time()
        energy_forces_pred_df = self.predict_energy_forces(params, keep_parallel_dataexecutor=True)

        total_na = self.structures_dataframe["NUMBER_OF_ATOMS"].values
        dE = (energy_forces_pred_df[ENERGY_PRED_COL] - self.structures_dataframe[ENERGY_CORRECTED_COL]).values
        dE_per_atom = dE / total_na
        dF = (self.structures_dataframe[FORCES_COL] - energy_forces_pred_df[FORCES_PRED_COL]).values

        self.last_epa_mae = np.mean(np.abs(dE_per_atom))

        # de = dE #np.hstack(dE.tolist())
        # de_pa = dE_per_atom #np.hstack(dE_per_atom.tolist())
        df = np.vstack(dF)  # np.vstack([v.reshape(-1, 3) for v in dF.tolist()])
        self.metrics.compute_metrics(dE.reshape(-1, 1), dE_per_atom.reshape(-1, 1), df,
                                     total_na, dataframe=self.structures_dataframe)

        if self.loss_spec.kappa < 1:
            # dEsqr = dE ** 2
            dEsqr = dE_per_atom ** 2
            if EWEIGHTS_COL in self.structures_dataframe.columns:
                dEsqr = dEsqr * np.vstack(self.structures_dataframe[EWEIGHTS_COL]).reshape(-1)  # structure-wise
            e_loss = np.sum(dEsqr)
        else:
            e_loss = 0

        if self.loss_spec.kappa > 0:  # forces have contribution to loss function
            dFsqr = (self.structures_dataframe[FORCES_COL] - energy_forces_pred_df[FORCES_PRED_COL])
            dFsqr = dFsqr.map(lambda f: np.sum(f ** 2, axis=1))
            if FWEIGHTS_COL in self.structures_dataframe.columns:
                dFsqr = dFsqr * self.structures_dataframe[FWEIGHTS_COL]
            dFsqr = dFsqr.map(np.sum)

            f_loss = np.sum(dFsqr)
        else:  # forces have no contribution to loss function
            f_loss = 0

        basis_coeffs = np.array(self.bbasis.basis_coeffs)

        self.l1 = np.sum(np.abs(basis_coeffs) * self.loss_spec.w1_coeffs)
        self.l2 = np.sum(basis_coeffs ** 2 * self.loss_spec.w2_coeffs)

        loss_coeff = \
            self.loss_spec.L1_coeffs * self.l1 + self.loss_spec.L2_coeffs * self.l2

        loss_crad = 0
        if self.loss_spec.w0_rad > 0 or self.loss_spec.w1_rad > 0 or self.loss_spec.w2_rad > 0:
            smothness = RadialFunctionSmoothness(RadialFunctionsValues(self.bbasis))
            self.smooth_quad = smothness.smooth_quad
            loss_crad = self.loss_spec.w0_rad * self.smooth_quad[0] + \
                        self.loss_spec.w1_rad * self.smooth_quad[1] + \
                        self.loss_spec.w2_rad * self.smooth_quad[2]

        self.last_loss = (1 - self.loss_spec.kappa) * e_loss + self.loss_spec.kappa * f_loss + \
                         loss_coeff + loss_crad

        if self.best_loss is None or self.last_loss < self.best_loss:
            self.best_loss = self.last_loss
            self.best_params = params

        if verbose:
            print("Eval {}: loss={}".format(self.current_func_eval, self.last_loss) + " " * 40 + "\r", end="")

        self.eval_time = time.time() - t0
        return self.last_loss

    def predict_energy_forces(self, params=None,
                              structures_dataframe=None,
                              keep_parallel_dataexecutor=False):
        if params is not None:
            if isinstance(params, (list, tuple, np.ndarray)):
                self.cbasis = self.get_cbasis(params)
            elif isinstance(params, ACECTildeBasisSet):
                self.cbasis = params
            elif isinstance(params, ACEBBasisSet):
                self.bbasis = params
                self.cbasis = self.bbasis.to_ACECTildeBasisSet()
            else:
                raise ValueError(
                    "Type of parameters could be only np.array, list, tuple, ACECTildeBasisSet, ACEBBasisSet" +
                    "but got {}".format(type(params)))
        if structures_dataframe is not None:
            self.structures_dataframe = structures_dataframe
        par = partial(batch_compute_energy_forces_function_wrapper, cbasis=self.cbasis)
        self._initialize_executor()
        energy_forces_pred = self.data_executor.map(wrapped_pure_func=par)
        if not keep_parallel_dataexecutor:
            self.data_executor.stop_executor()
        energy_forces_pred_df = pd.DataFrame({ENERGY_PRED_COL: energy_forces_pred.map(lambda d: d[0]),
                                              FORCES_PRED_COL: energy_forces_pred.map(lambda d: np.array(d[1]))},
                                             index=energy_forces_pred.index)

        return energy_forces_pred_df

    def predict(self):
        return self.predict_energy_forces()

    def predict_projections(self, params=None,
                            structures_dataframe=None,
                            keep_parallel_dataexecutor=False):
        if structures_dataframe is not None:
            self.structures_dataframe = structures_dataframe

        potential_params = None
        if params is not None:
            if isinstance(params, (list, tuple, np.ndarray)):
                self.cbasis = self.get_cbasis(params)
                potential_params = self.cbasis
            elif isinstance(params, ACECTildeBasisSet):
                self.cbasis = params
                potential_params = self.cbasis
            elif isinstance(params, ACEBBasisSet):
                self.bbasis = params
                potential_params = self.bbasis
            elif isinstance(params, BBasisConfiguration):
                potential_params = params
            else:
                raise ValueError(
                    "Type of parameters could be only np.array, list, tuple, ACECTildeBasisSet, ACEBBasisSet" +
                    "but got {}".format(type(params)))
        elif self.bbasis is not None:
            log.info("No 'params' provided to predict_projections, bbasis will be used")
            potential_params = self.bbasis
        else:
            raise ValueError(
                "Basis is not set. provide `params` argument or create PyACEFit with some predefined basis")

        par = partial(batch_compute_projections_function_wrapper, potential_params=potential_params)

        self._initialize_executor()
        projections_pred_df = self.data_executor.map(wrapped_pure_func=par)
        if not keep_parallel_dataexecutor:
            self.data_executor.stop_executor()

        return projections_pred_df

    def get_reg_components(self):
        if self.smooth_quad is not None:
            return [self.l1, self.l2, self.smooth_quad[0], self.smooth_quad[1], self.smooth_quad[2]]
        else:
            return [self.l1, self.l2, 0., 0., 0.]

    def get_reg_weights(self):
        return [self.loss_spec.L1_coeffs, self.loss_spec.L2_coeffs,
                self.loss_spec.w0_rad, self.loss_spec.w1_rad, self.loss_spec.w2_rad]

    def get_fitting_data(self):
        return self.structures_dataframe

    def print_detailed_metrics_old(self, prefix='Iteration:'):
        log.info('{:<12}'.format(prefix) +
                 "#{iter_num:<5}".format(iter_num=self.iter_num) +
                 '{:<14}'.format('({numeval} evals):'.format(numeval=self.current_func_eval)) +
                 '{:>10}'.format('Loss: ') + "{loss: >1.4e}".format(loss=float(self.last_loss)) +
                 '{str1:>16}{rmse_epa:>.2f} meV/at; {str2:>8}{rmse_f:>.2f} meV/A'.format(
                     str1=" | RMSE Energy: ", rmse_epa=1e3 * self.metrics.rmse_epa, str2="Forces: ",
                     rmse_f=1e3 * self.metrics.rmse_f) +
                 ' | Time/eval: {:>6.2f} mcs/at'.format(self.eval_time * 1e6 / self.metrics.nat))

    def print_detailed_metrics(self, prefix='Iteration:'):
        log.info('{:<12}'.format(prefix) +
                 "#{iter_num:<5}".format(iter_num=self.iter_num) +
                 '{:<14}'.format('({numeval} evals):'.format(numeval=self.current_func_eval)) +
                 '{:>10}'.format('Loss: ') + "{loss: >3.6f}".format(loss=float(self.last_loss)) +
                 '{str1:>21}{rmse_epa:>.2f} ({low_rmse_e:>.2f}) meV/at' \
                 .format(str1=" | RMSE Energy(low): ",
                         rmse_epa=1e3 * self.metrics.rmse_epa,
                         low_rmse_e=1e3 * self.metrics.low_rmse_e) +
                 '{str3:>16}{rmse_f:>.2f} ({low_rmse_f:>.2f}) meV/A' \
                 .format(str3=" | Forces(low): ",
                         rmse_f=1e3 * self.metrics.rmse_f,
                         low_rmse_f=1e3 * self.metrics.low_rmse_f) +
                 ' | Time/eval: {:>6.2f} mcs/at'.format(self.eval_time * 1e6 / self.metrics.nat))

    def callback(self, *args, **kwargs):
        # call global callback
        self.metrics.record_time(self.eval_time)
        if self.iter_num % self.display_step == 0:
            self.metrics.print_extended_metrics(self.iter_num, float(self.last_loss),
                                                self.get_reg_components(), self.get_reg_weights())
        else:
            self.print_detailed_metrics()
        self.iter_num += 1

        if self.global_callback is not None:
            self.global_callback(*args, **kwargs)

    def fit(self, structures_dataframe, method="Nelder-Mead",
            options={"maxiter": 100, "disp": True}, callback=None, verbose=True):

        if structures_dataframe is not None:
            self.preprocess_dataframe(structures_dataframe)
            self.structures_dataframe = structures_dataframe
        else:
            raise ValueError("structures_dataframe couldn't be None")

        self.global_callback = callback
        log.info("Data size:" + str(self.structures_dataframe.shape))
        # log.debug("self.structures_dataframe.columns = " + str(self.structures_dataframe.columns))
        log.info("Energy weights : " + str(EWEIGHTS_COL in self.structures_dataframe.columns))
        log.info("Forces weights : " + str(FWEIGHTS_COL in self.structures_dataframe.columns))
        self.current_func_eval = 0
        self.best_loss = None

        log.info('Number of parameters to optimize: {0}'.format(len(self.params)))

        w_e = np.hstack(self.structures_dataframe[EWEIGHTS_COL].tolist())
        w_f = np.hstack(self.structures_dataframe[FWEIGHTS_COL].tolist())
        self.metrics = FitMetrics(w_e.reshape(-1, 1), w_f.reshape(-1, 1), 1. - self.loss_spec.kappa,
                                  self.loss_spec.kappa, len(self.params))

        self._initialize_executor()
        if 'disp' not in options:
            options['disp'] = True
        if 'gtol' not in options:
            options['gtol'] = 1e-8
        log.info('Scipy minimize: method = {},  options = {}'.format(method, options))

        self.initial_loss = self.loss(self.bbasis)
        log.info('{:<32}'.format('Initial state:') + '{:>10}'.format('Loss: ') + "{loss: >3.6f}" \
                 .format(loss=float(self.initial_loss)) +
                 '{str:>21}{rmse_epa:>.2f} ({low_rmse_e:>.2f}) meV/at' \
                 .format(str=" | RMSE Energy(low): ",
                         rmse_epa=1e3 * float(self.metrics.rmse_epa),
                         low_rmse_e=1e3 * self.metrics.low_rmse_e))
        self.metrics.print_extended_metrics(step="Init", total_loss=self.initial_loss,
                                            reg_comps=self.get_reg_components(),
                                            reg_weights=self.get_reg_weights()
                                            )
        res_opt = minimize(self.loss, x0=self.params, args=(verbose,), method=method, options=options,
                           callback=self.callback)

        self.res_opt = res_opt

        self.params_opt = res_opt.x
        self.bbasis_opt = self.update_bbasis(self.params_opt)
        self.cbasis_opt = self.bbasis.to_ACECTildeBasisSet()

    def _initialize_executor(self):
        if self.data_executor is None:
            self.data_executor = ParallelDataExecutor(distributed_data=self.structures_dataframe[ATOMIC_ENV_COL],
                                                      **self.executors_kw_args)


class FitMetrics:
    def __init__(self, w_e, w_f, e_scale, f_scale, ncoefs, regs=None):
        self.w_e = w_e
        self.w_f = w_f
        self.e_scale = e_scale
        self.f_scale = f_scale
        self.regs = regs
        self.ncoefs = ncoefs
        self.time_history = []

    def record_time(self, time):
        self.time_history.append(time)

    def compute_metrics(self, de, de_pa, df, nat, dataframe=None, de_low=None):
        if de_low is None:
            de_low = 1.
        self.nat = np.sum(nat)
        self.rmse_epa = np.sqrt(np.mean(de_pa ** 2))
        self.rmse_e = np.sqrt(np.mean(de ** 2))
        self.rmse_f = np.sqrt(np.mean(np.sum(df ** 2, axis=1)))
        self.mae_epa = np.mean(np.abs(de_pa))
        self.mae_e = np.mean(np.abs(de))
        self.mae_f = np.mean(np.linalg.norm(df, axis=1))

        self.e_loss = np.float(np.sum(self.w_e * de_pa ** 2))
        self.f_loss = np.sum(self.w_f * df ** 2)
        self.max_abs_e = np.max(np.abs(de))
        self.max_abs_epa = np.max(np.abs(de_pa))
        self.max_abs_f = np.max(np.abs(df))

        if dataframe is not None:
            nrgs = dataframe['energy_corrected'].to_numpy().reshape(-1, ) / nat.reshape(-1, )
            emin = min(nrgs)
            mask = (nrgs <= (emin + de_low))
            mask_f = np.repeat(mask, nat.reshape(-1, ))
            self.low_rmse_e = np.sqrt(np.mean(de_pa[mask] ** 2))
            self.low_mae_e = np.mean(np.abs(de_pa[mask]))
            self.low_max_abs_e = np.max(np.abs(de_pa[mask]))
            self.low_rmse_f = np.sqrt(np.mean(np.sum(df[mask_f] ** 2, axis=1)))
            self.low_mae_f = np.mean(np.linalg.norm(df[mask_f], axis=1))
            self.low_max_abs_f = np.max(np.abs(df[mask_f]))
        else:
            self.low_rmse_e = 0
            self.low_mae_e = 0
            self.low_max_abs_e = 0
            self.low_rmse_f = 0
            self.low_mae_f = 0
            self.low_max_abs_f = 0

    def print_extended_metrics(self, step, total_loss, reg_comps, reg_weights):
        str0 = '\n' + '-' * 44 + 'FIT STATS' + '-' * 44 + '\n'
        str1 = '{prefix:<11} #{iter_num:<4}'.format(prefix='Iteration:', iter_num=step)
        str1 += '{prefix:<8}'.format(prefix='Loss:')
        str1 += '{prefix:>8} {tot_loss:>1.4e} ({fr:3.0f}%) '.format(prefix='Total: ', tot_loss=total_loss, fr=100)
        str1 += '\n'

        str2 = '{prefix:>33} {e_loss:>1.4e} ({fr:3.0f}%) '.format(prefix='Energy: ', e_loss=self.e_loss * self.e_scale,
                                                                  fr=self.e_loss * self.e_scale / total_loss * 100)
        str2 += '\n'

        str3 = '{prefix:>33} {f_loss:>1.4e} ({fr:3.0f}%) '.format(prefix='Force: ', f_loss=self.f_loss * self.f_scale,
                                                                  fr=self.f_loss * self.f_scale / total_loss * 100)
        str3 += '\n'

        l1 = float(reg_comps[0] * reg_weights[0])
        l2 = float(reg_comps[1] * reg_weights[1])

        str4 = '{prefix:>33} {l1:>1.4e} ({fr:3.0f}%) '.format(prefix='L1: ', l1=l1, fr=l1 / total_loss * 100)
        str4 += '\n'
        str4 += '{prefix:>33} {l2:>1.4e} ({fr:3.0f}%) '.format(prefix='L2: ', l2=l2, fr=l2 / total_loss * 100)
        str4 += '\n'

        str5 = ''
        for i in range(2, len(reg_comps)):
            str5 += '{prefix:>33} '.format(prefix='Smooth_w{}: '.format(i - 1))
            comp = float(reg_comps[i]) * float(reg_weights[i])
            str5 += '{s1:>1.4e} '.format(s1=comp)
            str5 += '({fr:3.0f}%) '.format(fr=comp / total_loss * 100)
            str5 += '\n'
        str6 = '{prefix:>20}'.format(prefix='Number of params.: ') + '{ncoefs:>6d}'.format(ncoefs=self.ncoefs) + \
               '{prefix:>22}'.format(prefix='Avg. time: ') + \
               '{avg_t:>10.2f} {un:<6}'.format(avg_t=np.mean(self.time_history) / self.nat * 1e6, un='mcs/at')
        str6 += '\n' + '-' * 97 + '\n'
        str_loss = str0 + str1 + str2 + str3 + str4 + str5 + str6
        ##############################
        er_str_h = '{:>9}'.format('') + \
                   '{:^22}'.format('Energy/at, meV/at') + \
                   '{:^22}'.format('Energy_low/at, meV/at') + \
                   '{:^22}'.format('Force, meV/A') + \
                   '{:^22}\n'.format('Force_low, meV/A')

        er_rmse = '{prefix:>9} '.format(prefix='RMSE: ')
        er_rmse += '{:>14.2f}'.format(self.rmse_epa * 1e3) + \
                   '{:>21.2f}'.format(self.low_rmse_e * 1e3) + \
                   '{:>21.2f}'.format(self.rmse_f * 1e3) + \
                   '{:>24.2f}\n'.format(self.low_rmse_f * 1e3)
        er_mae = '{prefix:>9} '.format(prefix='MAE: ')
        er_mae += '{:>14.2f}'.format(self.mae_epa * 1e3) + \
                  '{:>21.2f}'.format(self.low_mae_e * 1e3) + \
                  '{:>21.2f}'.format(self.mae_f * 1e3) + \
                  '{:>24.2f}\n'.format(self.low_mae_f * 1e3)
        er_max = '{prefix:>9} '.format(prefix='MAX_AE: ')
        er_max += '{:>14.2f}'.format(self.max_abs_epa * 1e3) + \
                  '{:>21.2f}'.format(self.low_max_abs_e * 1e3) + \
                  '{:>21.2f}'.format(self.max_abs_f * 1e3) + \
                  '{:>24.2f}\n'.format(self.low_max_abs_f * 1e3)
        er_str = er_str_h + er_rmse + er_mae + er_max + '-' * 97 + '\n'
        log.info(str_loss + er_str)
