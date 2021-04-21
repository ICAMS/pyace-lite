import logging

from typing import Dict, Union, Callable

import subprocess as sp
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

from pyace.const import *
from pyace.basis import BBasisConfiguration, ACEBBasisSet, ACEBBasisFunction
from pyace.pyacefit import PyACEFit, LossFunctionSpecification


class BackendConfig:
    def __init__(self, backend_config_dict: Dict):
        self.backend_config_dict = backend_config_dict
        self.validate()

    @property
    def evaluator_name(self):
        return self.backend_config_dict[BACKEND_EVALUATOR_KW]

    def __getattr__(self, item):
        return self.backend_config_dict[item]

    def validate(self):
        pass

    def get(self, item, default_value=None):
        return self.backend_config_dict.get(item, default_value)


class FitBackendAdapter:

    def __init__(self, backend_config: Union[Dict, BackendConfig], loss_spec: LossFunctionSpecification = None,
                 fit_config: Dict = None, callback: Callable = None):
        if isinstance(backend_config, dict):
            self.backend_config = BackendConfig(backend_config)
        else:
            self.backend_config = backend_config
        self.callback = callback
        self.loss_spec = loss_spec
        self.fit_config = fit_config
        self.res_opt = None
        self.fitter = None
        self.metrics = None

    @property
    def evaluator_name(self):
        return self.backend_config.evaluator_name

    def fit(self,
            bbasisconfig: BBasisConfiguration,
            dataframe: pd.DataFrame,
            loss_spec: LossFunctionSpecification = None,
            fit_config: Dict = None, callback: Callable = None) -> BBasisConfiguration:
        if loss_spec is None:
            loss_spec = self.loss_spec
        else:
            self.loss_spec = loss_spec
        if fit_config is None:
            fit_config = self.fit_config

        if callback is not None:
            self.callback = callback

        if self.backend_config.evaluator_name == TENSORPOT_EVAL:
            return self.run_tensorpot_fit(bbasisconfig, dataframe, loss_spec, fit_config)
        elif self.backend_config.evaluator_name == PYACE_EVAL:
            return self.run_pyace_fit(bbasisconfig, dataframe, loss_spec, fit_config)
        else:
            raise ValueError('{0} is not a valid evaluator'.format(self.backend_config.evaluator_name))

    @staticmethod
    def get_gpu_memory():
        _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

        MB2GB = 1./1024
        COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]

        return np.array(memory_free_values) * MB2GB

    def adjust_batch_size(self, dataframe, bbasisconfig, ini_batch_size):
        from tensorpotential.utils.utilities import batching_data

        try:
            gpu_mem = self.get_gpu_memory()
        except:
            gpu_mem = None

        if gpu_mem is None:
            print('There are no available GPUs, batch size adjustment will not be performed')

        batches = batching_data(dataframe, batch_size=ini_batch_size)
        max_A_tensor_size = 0
        max_nat = 0
        for b in batches:
            # TODO: single specie hardcoded
            nmax = bbasisconfig.funcspecs_blocks[0].nradmaxi
            lmax = bbasisconfig.funcspecs_blocks[0].lmaxi
            ml = (lmax +1) * (lmax + 1)
            nn = len(b['ind_i'])
            max_A_tensor_size = max(max_A_tensor_size, nn*nmax*ml)
            max_nat = max(max_nat, max(b['ind_i'])+1)

        rank_max = int.from_bytes(bytes(bbasisconfig.funcspecs_blocks[0].rankmax,
                                        encoding="raw_unicode_escape"), byteorder='little')
        ms_sizes = np.zeros([rank_max,])
        max_B_tensor_size = 0
        for f in bbasisconfig.funcspecs_blocks[0].funcspecs:
            func = ACEBBasisFunction(f)
            ms_sizes[func.rank-1] += len(func.ms_combs)
        for ms_size in ms_sizes:
            tens_size = ms_size*max_nat
            max_B_tensor_size = max(max_B_tensor_size, tens_size)

        max_tensor_gb = max(max_B_tensor_size, max_A_tensor_size) * 64 * 9.313225746154785e-10

        return max_tensor_gb

    def run_tensorpot_fit(self, bbasisconfig: BBasisConfiguration, dataframe: pd.DataFrame,
                          loss_spec: LossFunctionSpecification, fit_config: Dict) -> BBasisConfiguration:
        from tensorpotential.potentials.ace import ACE
        from tensorpotential.tensorpot import TensorPotential
        from tensorpotential.fit import FitTensorPotential
        from tensorpotential.utils.utilities import batching_data
        from tensorpotential.constants import (LOSS_TYPE, LOSS_FORCE_FACTOR, LOSS_ENERGY_FACTOR, L1_REG,
                                                L2_REG, AUX_LOSS_FACTOR)

        batch_size = self.backend_config.get(BACKEND_BATCH_SIZE_KW, 10)
        log.info("Loss function specification: " + str(loss_spec))
        log.info("Batch size: {}".format(batch_size))
        batches = batching_data(dataframe, batch_size=batch_size)
        # max_bytes = self.adjust_batch_size(dataframe, bbasisconfig, ini_batch_size=batch_size)
        n_batches = len(batches)
        if loss_spec.w1_coeffs != 1.0 or loss_spec.w2_coeffs != 1.0:
            log.warning("WARNING! 'w1_coeffs'={} and 'w2_coeffs'={}  in loss function will be ignored".
                        format(loss_spec.w1_coeffs,
                               loss_spec.w2_coeffs))
        loss_force_factor = loss_spec.kappa
        if (np.array([loss_spec.w0_rad, loss_spec.w1_rad, loss_spec.w2_rad]) != 0).any():
            ace_potential = ACE(bbasisconfig, compute_smoothness=True)
            tensorpotential = TensorPotential(ace_potential,loss_specs={
                                        LOSS_TYPE: 'per-atom',
                                        LOSS_FORCE_FACTOR: loss_force_factor,
                                        LOSS_ENERGY_FACTOR: (1-loss_force_factor),
                                        L1_REG: np.float64(loss_spec.L1_coeffs) / n_batches,
                                        L2_REG: np.float64(loss_spec.L2_coeffs) / n_batches,
                                        AUX_LOSS_FACTOR: [np.float64(loss_spec.w0_rad) / n_batches,
                                                            np.float64(loss_spec.w1_rad) / n_batches,
                                                            np.float64(loss_spec.w2_rad) / n_batches]})
        else:
            ace_potential = ACE(bbasisconfig, compute_smoothness=False)
            tensorpotential = TensorPotential(ace_potential, loss_specs={
                LOSS_TYPE: 'per-atom',
                LOSS_FORCE_FACTOR: loss_force_factor,
                LOSS_ENERGY_FACTOR: (1 - loss_force_factor),
                L1_REG: np.float64(loss_spec.L1_coeffs) / n_batches,
                L2_REG: np.float64(loss_spec.L2_coeffs) / n_batches})

        display_step = self.backend_config.get('display_step', 20)
        self.fitter = FitTensorPotential(tensorpotential, display_step=display_step)
        fit_options = fit_config.get(FIT_OPTIONS_KW, None)
        self.fitter.fit(dataframe, niter=fit_config[FIT_NITER_KW], optimizer=fit_config[FIT_OPTIMIZER_KW],
                        batch_size=batch_size, jacobian_factor=None,
                        callback=self._callback, options=fit_options)
        self.res_opt = self.fitter.res_opt
        coeffs = self.fitter.get_fitted_coefficients()
        bbasisconfig.set_all_coeffs(coeffs)

        return bbasisconfig

    def run_pyace_fit(self, bbasisconfig: BBasisConfiguration, dataframe: pd.DataFrame,
                      loss_spec: LossFunctionSpecification, fit_config: Dict) -> BBasisConfiguration:

        parallel_mode = self.backend_config.get(BACKEND_PARALLEL_MODE_KW) or "serial"
        batch_size = len(dataframe)

        log.info("Loss function specification: " + str(loss_spec))
        display_step = self.backend_config.get('display_step', 20)
        self.fitter = PyACEFit(basis=bbasisconfig,
                               loss_spec=loss_spec,
                               executors_kw_args=dict(parallel_mode=parallel_mode,
                                                      batch_size=batch_size,
                                                      n_workers=self.backend_config.get(BACKEND_NWORKERS_KW, None)
                                                      ),
                               seed=42,
                               display_step=display_step)

        maxiter = fit_config.get(FIT_NITER_KW, 100)

        fit_options = fit_config.get(FIT_OPTIONS_KW, {})
        options = {"maxiter": maxiter, "disp": True}
        options.update(fit_options)

        self.fitter.fit(structures_dataframe=dataframe, method=fit_config[FIT_OPTIMIZER_KW],
                        options=options,
                        callback=self._callback)

        # TODO: options=self.fit_config[FIT_OPTIMIZER_OPTIONS_KW]
        self.res_opt = self.fitter.res_opt
        new_bbasisconf = self.fitter.bbasis_opt.to_BBasisConfiguration()
        bbasisconfig.set_all_coeffs(new_bbasisconf.get_all_coeffs())

        return bbasisconfig

    def compute_metrics(self, energy_col='energy_corrected',
                        nat_column='NUMBER_OF_ATOMS', force_col='forces'):
        results = {}
        prediction = self.fitter.predict()
        l1, l2, smth1, smth2, smth3 = self.fitter.get_reg_components()
        datadf = self.fitter.get_fitting_data()

        datadf[force_col] = datadf[force_col].apply(np.array)
        datadf['w_forces'] = datadf['w_forces'].apply(np.reshape, newshape=[-1, 1])
        de = prediction['energy_pred'] - datadf[energy_col]
        df = prediction['forces_pred'] - datadf[force_col]
        e_loss = np.float(np.sum(datadf['w_energy'] * de ** 2))
        f_loss = np.sum((datadf['w_forces'] * df ** 2).map(np.sum))

        mae_pae = np.mean(np.abs(de / datadf[nat_column]))
        mae_e = np.mean(np.abs(de))
        mae_f   = np.mean(np.abs(df).map(np.mean))
        rmse_pae = np.sqrt(np.mean(de ** 2 / datadf[nat_column]))
        rmse_e = np.sqrt(np.mean(de ** 2))
        rmse_f = np.sqrt(np.mean((df ** 2).map(np.mean)))

        results['mae_pae'] = mae_pae
        results['mae_e'] = mae_e
        results['mae_f'] = mae_f
        results['rmse_pae'] = rmse_pae
        results['rmse_e'] = rmse_e
        results['rmse_f'] = rmse_f

        results['e_loss'] = e_loss
        results['f_loss'] = f_loss
        results['l1'] = l1
        results['l2'] = l2
        results['radial_smooth'] = [smth1, smth2, smth3]

        return results

    def print_detailed_metrics(self, prefix='Iteration:'):
        if self.fitter is not None:
            self.fitter.print_detailed_metrics(prefix=prefix)

    def _callback(self, coeffs):
        if self.callback is not None:
            self.callback(coeffs)

    @property
    def last_loss(self):
        if self.backend_config.evaluator_name == TENSORPOT_EVAL:
            return self.fitter.loss_history[-1]
        elif self.backend_config.evaluator_name == PYACE_EVAL:
            return self.fitter.last_loss
