#!/usr/bin/env python

# /*
# * Atomic cluster expansion
# *
# * Copyright 2021  (c) Yury Lysogorskiy, Anton Bochkarev, Ralf Drautz
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


import argparse
import pkg_resources
import ruamel.yaml as yaml
import sys

from pyace.generalfit import GeneralACEFit
from pyace.preparedata import get_reference_dataset

import logging

DEFAULT_SEED = 42

log = logging.getLogger()


# log.setLevel(logging.DEBUG)
# handler = logging.StreamHandler(sys.stdout)
# handler.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# log.addHandler(handler)


def main(args):
    parser = argparse.ArgumentParser(prog="pacemaker", description="Fitting utility for atomic cluster expansion " +
                                                                   "potentials")
    parser.add_argument("input", help="input YAML file", type=str)

    parser.add_argument("-o", "--output", help="output B-basis YAML file name, default: output_potential.yaml",
                        default="output_potential.yaml",
                        type=str)

    parser.add_argument("-p", "--potential",
                        help="input potential YAML file name, will override input file 'potential' section",
                        type=str,
                        default=argparse.SUPPRESS)

    parser.add_argument("-ip", "--initial-potential",
                        help="initial potential YAML file name, will override input file 'potential::initial_potential' section",
                        type=str,
                        default=argparse.SUPPRESS)

    parser.add_argument("-b", "--backend",
                        help="backend evaluator, will override section 'backend::evaluator' from input file",
                        type=str,
                        default=argparse.SUPPRESS)

    parser.add_argument("-d", "--data",
                        help="data file, will override section 'YAML:fit:filename' from input file",
                        type=str,
                        default=argparse.SUPPRESS)

    parser.add_argument("--query-data", help="query the training data from database, prepare and save them", dest="query_data", default=False,
                        action="store_true")

    parser.add_argument("--prepare-data", help="prepare and save training data only", dest="prepare_data", default=False,
                        action="store_true")

    parser.add_argument("-l", "--log", help="log filename", type=str, default="log.txt")

    default_bbasis_func_df_filename = pkg_resources.resource_filename('pyace.data',
                                                                      'pyace_selected_bbasis_funcspec.pckl.gzip')

    args_parse = parser.parse_args(args)
    input_yaml_filename = args_parse.input
    output_file_name = args_parse.output

    if "log" in args_parse:
        log_file_name = args_parse.log
        log.info("Redirecting log into file {}".format(log_file_name))
        fileh = logging.FileHandler(log_file_name, 'a')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fileh.setFormatter(formatter)
        # log = logging.getLogger()
        log.addHandler(fileh)

    log.info("Start pacemaker")
    log.info("Loading {}... ".format(input_yaml_filename))
    with open(input_yaml_filename) as f:
        args_yaml = yaml.safe_load(f)

    assert isinstance(args_yaml, dict)
    if "cutoff" in args_yaml:
        cutoff = args_yaml["cutoff"]
    else:
        log.warning("No 'cutoff' provided in YAML file, please specify it")
        raise ValueError("No 'cutoff' provided in YAML file, please specify it")

    if "seed" in args_yaml:
        seed = args_yaml["seed"]
    else:
        seed = DEFAULT_SEED
        log.warning("No 'seed' provided in YAML file, default value seed = {} will be used.".format(seed))
        # raise ValueError("No 'cutoff' provided in YAML file, please specify it")

    # data section
    if "data" in args_parse:
        data_config = {"filename": args_parse.data}
        log.info("Overwrite 'data' with " + str(data_config))
    elif "data" in args_yaml:
        data_config = args_yaml["data"]
        if isinstance(data_config, str):
            data_config = {"filename": data_config}
        if "seed" not in data_config:
            data_config["seed"] = seed
    else:
        raise ValueError("'data' section is not provided neither in input file nor in arguments")

    # backend section
    backend_config = {}
    if "backend" in args_parse:
        backend_config = {"evaluator": args_parse.backend}
        log.info("Backend settings is overwritten from arguments: ", backend_config)
    elif "backend" in args_yaml:
        backend_config = args_yaml["backend"]
    elif not args_parse.query_data:
        backend_config["evaluator"] = "pyace"
        backend_config["parallel_mode"] = "process"
        log.warning("'backend' is not specified, default settings will be used: {}".format(backend_config))
        # raise ValueError("'backend' section is not given")

    if 'evaluator' in backend_config:
        evaluator_name = backend_config['evaluator']
    else:
        backend_config['evaluator'] = 'pyace'
        evaluator_name = backend_config['evaluator']
        log.info("Couldn't find evaluator ('pyace' or 'tensorpot').")
        log.info("Default evaluator `{}` would be used, ".format(evaluator_name) +
                 " otherwise please specify in YAML:backend:evaluator or as -b <evaluator>")

    if args_parse.query_data:
        if isinstance(data_config, str):
            raise ValueError("Requires YAML input file with 'data' section")
        log.debug("data_config={}".format(str(data_config)))
        log.debug("evaluator_name={}".format(evaluator_name))
        log.debug("cutoff={}".format(cutoff))
        get_reference_dataset(evaluator_name=evaluator_name, data_config=data_config, cutoff=cutoff, force_query=True,
                              cache_ref_df=True)
        log.info("Done, now stopping")
        sys.exit(0)

    if args_parse.prepare_data:
        if isinstance(data_config, str):
            raise ValueError("Requires YAML input file with 'data:filename' section")
        log.debug("data_config={}".format(str(data_config)))
        log.debug("evaluator_name={}".format(evaluator_name))
        log.debug("cutoff={}".format(cutoff))
        get_reference_dataset(evaluator_name=evaluator_name, data_config=data_config, cutoff=cutoff, force_query=False,
                              cache_ref_df=True)
        log.info("Done, now stopping")
        sys.exit(0)

    # potential section
    if "potential" in args_parse:
        potential_config = args_parse.potential
        log.info("Potential settings is overwritten from arguments: " + str(potential_config))
    elif "potential" in args_yaml:
        potential_config = args_yaml["potential"]
        if isinstance(potential_config, dict):
            if "metadata" in args_yaml:
                potential_config["metadata"] = args_yaml["metadata"]
            if "basisdf" not in potential_config:
                potential_config["basisdf"] = default_bbasis_func_df_filename
                log.info("Using default BBasis functions list from " + default_bbasis_func_df_filename)
    elif not args_parse.query_data:
        raise ValueError("'potential' section is not given")

    if "initial_potential" in args_parse:
        if isinstance(potential_config, dict):
            potential_config["initial_potential"] = args_parse.initial_potential
        else:
            raise ValueError("Couldn't combine `initial_potential` setting with non-dictionary `potential` setting")

    # fit section
    fit_config = {}
    if "fit" in args_yaml:
        fit_config = args_yaml["fit"]
    callbacks=[]
    if "callbacks" in fit_config:
        callbacks = fit_config["callbacks"]
    # elif not args_parse.query_data:
    #     raise ValueError("'fit' section is not given")

    general_fit = GeneralACEFit(potential_config=potential_config, fit_config=fit_config, data_config=data_config,
                                backend_config=backend_config, seed=seed, callbacks=callbacks)

    general_fit.fit()
    general_fit.save_optimized_potential(output_file_name)


if __name__ == "__main__":
    main(sys.argv[1:])
