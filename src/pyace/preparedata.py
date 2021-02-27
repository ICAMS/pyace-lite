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
import multiprocessing
import numpy as np
import os
import pandas as pd
import time

from typing import Dict

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from pyace.atomicenvironment import aseatoms_to_atomicenvironment
from pyace.const import *

DMIN_COLUMN = "dmin"
ATOMIC_ENV_COLUMN = "atomic_env"
FORCES_COLUMN = "forces"
E_CORRECTED_PER_ATOM_COLUMN = "energy_corrected_per_atom"
WEIGHTS_FORCES_COLUMN = "w_forces"
WEIGHTS_ENERGY_COLUMN = "w_energy"
REF_ENERGY_KW = "ref_energy"

log = logging.getLogger(__name__)

# ## QUERY DATA
LATTICE_COLUMNS = ["_lat_ax", "_lat_ay", "_lat_az",
                   "_lat_bx", "_lat_by", "_lat_bz",
                   "_lat_cx", "_lat_cy", "_lat_cz"]


def import_parallel_processing_tools(nb_workers=None, progress_bar=False):
    log.info("Trying to setup pandas parallel processing")
    nb_workers = nb_workers or multiprocessing.cpu_count()
    log.info("CPU count: {}".format(multiprocessing.cpu_count()))
    try:
        import pandarallel
        log.info("pandarallel imported".format(pandarallel.__version__))
        pandarallel.pandarallel.initialize(nb_workers=nb_workers, progress_bar=progress_bar)

        log.info("Number of pandarallel workers: {}".format(nb_workers))
        parallel_pandas = True
    except ImportError:
        parallel_pandas = False
        log.warning("Couldn't import 'pandarallel' package for parallel dataframe processing " +
                    "use 'pip install pandarallel' to install")
        log.warning("Fallback to serial pandas map mode")

    log.info("Parallel pandas processing: " + str(parallel_pandas))
    return parallel_pandas


# ### preprocess and store


def create_ase_atoms(row):
    pbc = row["pbc"]
    if pbc:
        cell = row["cell"]
        if row['COORDINATES_TYPE'] == 'relative':
            atoms = Atoms(symbols=row["_OCCUPATION"], scaled_positions=row["_COORDINATES"], cell=cell, pbc=pbc)
        else:
            atoms = Atoms(symbols=row["_OCCUPATION"], positions=row["_COORDINATES"], cell=cell, pbc=pbc)
    else:
        atoms = Atoms(symbols=row["_OCCUPATION"], positions=row["_COORDINATES"], pbc=pbc)
    e = row["energy_corrected"]
    f = row["_VALUE"]['forces']
    calc = SinglePointCalculator(atoms, energy=np.array(e).reshape(-1, ), forces=np.array(f))
    atoms.set_calculator(calc)
    return atoms


# define safe_min function, that return None of input is empty
def safe_min(val):
    try:
        return min((v for v in val if v is not None))
    except (ValueError, TypeError):
        return None


# define function that compute minimal distance
def calc_min_distance(ae):
    atpos = np.array(ae.x)
    nlist = ae.neighbour_list
    return safe_min(safe_min(np.linalg.norm(atpos[nlist[nat]] - atpos[nat], axis=1)) for nat in range(ae.n_atoms_real))


def query_data(config: Dict, seed=None, query_limit=None, db_conn_string=None):
    from structdborm import StructSQLStorage, CalculatorType, StructureEntry, StaticProperty, GenericEntry, Property
    from sqlalchemy.orm.exc import NoResultFound

    # validate config
    if "calculator" not in config:
        raise ValueError("'calculator' is not in YAML:data:config, couldn't query")
    if "element" not in config:
        raise ValueError("'element' is not in YAML:data:config, couldn't query")

    log.info("Connecting to database")
    with StructSQLStorage(db_conn_string) as storage:
        log.info("Querying database -- please be patient")
        reference_calculator = storage.query(CalculatorType).filter(
            CalculatorType.NAME == config["calculator"]).one()

        structure_entry_cell = [StructureEntry._lat_ax, StructureEntry._lat_ay, StructureEntry._lat_az,
                                StructureEntry._lat_bx, StructureEntry._lat_by, StructureEntry._lat_bz,
                                StructureEntry._lat_cx, StructureEntry._lat_cy, StructureEntry._lat_cz]
        if REF_ENERGY_KW not in config:
            try:
                # TODO: generalize query of reference property
                REF_PROP_NAME = '1-body-000001:static'
                REF_GENERIC_PROTOTYPE_NAME = '1-body-000001'
                ref_prop = storage.query(StaticProperty).join(StructureEntry, GenericEntry).filter(
                    Property.CALCULATOR == reference_calculator,
                    Property.NAME == REF_PROP_NAME,
                    StructureEntry.COMPOSITION.like(config["element"] + "-%"),
                    StructureEntry.NUMBER_OF_ATOMS == 1,
                    GenericEntry.PROTOTYPE_NAME == REF_GENERIC_PROTOTYPE_NAME
                ).one()
                # free atom reference energy
                ref_energy = ref_prop.energy / ref_prop.n_atom
            except NoResultFound as e:
                log.error(("No reference energy for {} was found in database. " +
                           "Either add property named `{}` with generic named `{}` to database or use `{}` " +
                           "keyword in data config ").format(config["element"], REF_PROP_NAME,
                                                             REF_GENERIC_PROTOTYPE_NAME, REF_ENERGY_KW))
                raise e
        else:
            ref_energy = config[REF_ENERGY_KW]
        # TODO: join with query with generic-parent-absent structures/properties
        q = storage.query(StaticProperty.id.label("prop_id"),
                          StructureEntry.id.label("structure_id"),
                          GenericEntry.id.label("gen_id"),
                          GenericEntry.PROTOTYPE_NAME,
                          *structure_entry_cell,
                          StructureEntry.COORDINATES_TYPE,
                          StructureEntry._COORDINATES,
                          StructureEntry._OCCUPATION,
                          StructureEntry.NUMBER_OF_ATOMS,
                          StaticProperty._VALUE) \
            .join(StaticProperty.ORIGINAL_STRUCTURE).join(StructureEntry.GENERICPARENT) \
            .filter(Property.CALCULATOR == reference_calculator,
                    StructureEntry.NUMBER_OF_ATOMTYPES == 1,
                    StructureEntry.COMPOSITION.like(config["element"] + "-%"),
                    ).order_by(StaticProperty.id)
        if query_limit is not None:
            q = q.limit(query_limit)
        log.info("Querying entries with defined generic prototype...")
        tot_data = q.all()
        log.info("Queried: {} entries".format(len(tot_data)))

        q_none = storage.query(StaticProperty.id.label("prop_id"),
                               StructureEntry.id.label("structure_id"),
                               StaticProperty.NAME.label("property_name"),
                               *structure_entry_cell,
                               StructureEntry.COORDINATES_TYPE,
                               StructureEntry._COORDINATES,
                               StructureEntry._OCCUPATION,
                               StructureEntry.NUMBER_OF_ATOMS,
                               StaticProperty._VALUE) \
            .join(StaticProperty.ORIGINAL_STRUCTURE) \
            .filter(Property.CALCULATOR == reference_calculator,
                    StructureEntry.NUMBER_OF_ATOMTYPES == 1,
                    StructureEntry.COMPOSITION.like(config["element"] + "-%"),
                    StructureEntry.GENERICPARENT == None
                    ).order_by(StaticProperty.id)

        if query_limit is not None:
            q_none = q_none.limit(query_limit)

        log.info("Querying entries without defined generic prototype...")
        no_generic_tot_data = q_none.all()
        log.info("Queried: {} entries".format(len(no_generic_tot_data)))

        df = pd.DataFrame(tot_data)
        df_no_generic = pd.DataFrame(no_generic_tot_data)

        log.info("Combining both queries together")
        df_total = pd.concat([df, df_no_generic], axis=0)

        # shuffle notebook for randomizing parallel processing
        if seed is not None:
            log.info("set numpy random seed = {}".format(seed))
            np.random.seed(seed)
            log.info("Shuffle dataset")
            df_total = df_total.sample(frac=1).reset_index(drop=True)
        else:
            log.info("Seed is not provided, no shuffling")
        log.info("Total entries obtained from database:" + str(df_total.shape[0]))
        return df_total, ref_energy


class StructuresDatasetWeightingPolicy:
    def generate_weights(self, df):
        raise NotImplementedError


def save_dataframe(df: pd.DataFrame, filename: str, protocol: int = 4):
    filename = os.path.abspath(filename)
    log.info("Writing fit pickle file: {}".format(filename))
    if filename.endswith("gzip"):
        compression = "gzip"
    else:
        compression = "infer"
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)
    df.to_pickle(filename, protocol=protocol, compression=compression)
    log.info("Saved to file " + filename)


def load_dataframe(filename: str, compression: str = "infer") -> pd.DataFrame:
    log.info("Loading dataframe from pickle file: " + filename)
    if filename.endswith(".gzip"):
        compression = "gzip"
    df = pd.read_pickle(filename, compression=compression)
    return df


class StructuresDatasetSpecification:
    """
    Object to query or load from cache the fitting dataset

        :param config:  dictionary with "element" - the element for which the data will be collected
                                        "calculator" - calculator and
                                        "seed" - random seed
        :param cutoff:
        :param filename:
        :param datapath:
        :param db_conn_string:
        :param parallel:
        :param force_query:
        :param query_limit:
        :param seed:
        :param cache_ref_df:
    """

    FHI_AIMS_PBE_TIGHT = 'FHI-aims/PBE/tight'

    def __init__(self,
                 config: Dict = None,
                 cutoff: float = 10,
                 filename: str = None,
                 datapath: str = "",
                 db_conn_string: str = None,
                 parallel: bool = False,
                 force_query: bool = False,
                 ignore_weights: bool = False,
                 query_limit: int = None,
                 seed: int = None,
                 cache_ref_df: bool = True,
                 progress_bar: bool = False,
                 df: pd.DataFrame = None
                 ):
        """

        :param config:
        :param cutoff:
        :param filename:
        :param datapath:
        :param db_conn_string:
        :param parallel:
        :param force_query:
        :param query_limit:
        :param seed:
        :param cache_ref_df:
        """

        # data config
        self.query_limit = query_limit
        if config is None:
            config = {}

        self.config = config
        self.force_query = force_query
        self.ignore_weights = ignore_weights
        self.filename = filename

        # random seed
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)

        # neighbour list cutoff
        self.cutoff = cutoff

        # result column name : tuple(mapper function,  kwargs)
        self.ase_atoms_transformers = {}
        # default transformer: to ACEAtomicEnvironment
        # self.add_ase_atoms_transformer(ATOMIC_ENV_COLUMN, aseatoms_to_atomicenvironment, cutoff=self.cutoff)

        # ### Path where pickle files will be stored
        self.datapath = datapath
        self.db_conn_string = db_conn_string
        self.parallel = parallel
        self.progress_bar = progress_bar

        self.raw_df = df
        self.df = None
        self.ref_energy = None
        self.weights_policy = None

        self.ref_df_changed = False
        self.cache_ref_df = cache_ref_df

    def set_weights_policy(self, weights_policy):
        self.weights_policy = weights_policy

    def add_ase_atoms_transformer(self, result_column_name, transformer_func, **kwargs):
        self.ase_atoms_transformers[result_column_name] = (transformer_func, kwargs)

    def _get_default_name(self, suffix):
        try:
            return os.path.join(self.datapath,
                                "df-{calculator}-{element}-{suffix}.pckl.gzip".format(
                                    calculator=self.config["calculator"],
                                    element=self.config["element"],
                                    suffix=suffix).replace("/", "_"))
        except KeyError as e:
            log.warning("Couldn't generate default name: " + str(e))
            return None
        except Exception as e:
            raise

    def get_default_ref_filename(self):
        return self._get_default_name(suffix='ref')

    def process_ref_dataframe(self, ref_df: pd.DataFrame, e0_per_atom: float) -> pd.DataFrame:
        log.info("Setting up structures dataframe - please be patient...")
        if "NUMBER_OF_ATOMS" not in ref_df.columns:
            raise ValueError("Dataframe is corrupted: 'NUMBER_OF_ATOMS' column is missing")

        tot_atoms_num = ref_df["NUMBER_OF_ATOMS"].sum()
        mean_atoms_num = ref_df["NUMBER_OF_ATOMS"].mean()
        log.info("Processing structures dataframe. Shape: " + str(ref_df.shape))
        log.info("Total number of atoms: " + str(tot_atoms_num))
        log.info("Mean number of atoms per structure: " + str(mean_atoms_num))

        # Extract energies and forces into separate columns
        if "energy" not in ref_df.columns and "energy_corrected" not in ref_df.columns:
            log.info("'energy' columns extraction")
            ref_df["energy"] = ref_df["_VALUE"].map(lambda d: d["energy"])
            self.ref_df_changed = True
        else:
            log.info("'energy' columns found")

        if FORCES_COLUMN not in ref_df.columns:
            log.info("'forces' columns extraction")
            ref_df[FORCES_COLUMN] = ref_df["_VALUE"].map(lambda d: np.array(d[FORCES_COLUMN]))
            self.ref_df_changed = True
        else:
            log.info("'forces' columns found")

        if "pbc" not in ref_df.columns:
            log.info("'pbc' columns extraction")
            ref_df["pbc"] = (ref_df[LATTICE_COLUMNS] != 0).any(axis=1)  # check the periodicity of coordinates
            self.ref_df_changed = True
        else:
            log.info("'pbc' columns found")

        if "cell" not in ref_df.columns and "ase_atoms" not in ref_df.columns:
            log.info("'cell' column extraction")
            ref_df["cell"] = ref_df[LATTICE_COLUMNS].apply(lambda row: row.values.reshape(-1, 3), axis=1)
            ref_df.drop(columns=LATTICE_COLUMNS, inplace=True)
            ref_df.reset_index(drop=True, inplace=True)
            self.ref_df_changed = True
        else:
            log.info("'cell' column found")

        if "energy_corrected" not in ref_df.columns:
            log.info("'energy_corrected' column extraction")
            if e0_per_atom is not None:
                ref_df["energy_corrected"] = ref_df["energy"] - ref_df["NUMBER_OF_ATOMS"] * e0_per_atom
            else:
                raise ValueError("e0_per_atom is not specified, please re-query the data from database")
            self.ref_df_changed = True
        else:
            log.info("'energy_corrected' column found")

        if E_CORRECTED_PER_ATOM_COLUMN not in ref_df.columns:
            log.info("'energy_corrected_per_atom' column extraction")
            ref_df[E_CORRECTED_PER_ATOM_COLUMN] = ref_df["energy_corrected"] / ref_df["NUMBER_OF_ATOMS"]
            self.ref_df_changed = True
        else:
            log.info("'energy_corrected_per_atom' column found")

        log.info("Min energy per atom: {}".format(ref_df[E_CORRECTED_PER_ATOM_COLUMN].min()))
        log.info("Max energy per atom: {}".format(ref_df[E_CORRECTED_PER_ATOM_COLUMN].max()))
        log.info("Min abs energy per atom: {}".format(ref_df[E_CORRECTED_PER_ATOM_COLUMN].abs().min()))
        log.info("Max abs energy per atom: {}".format(ref_df[E_CORRECTED_PER_ATOM_COLUMN].abs().max()))

        if "ase_atoms" not in ref_df.columns:
            log.info("ASE Atoms construction...")
            start = time.time()
            self.apply_create_ase_atoms(ref_df)
            end = time.time()
            time_elapsed = end - start
            log.info("ASE Atoms construction...done within {} sec ({} ms/at)".
                     format(time_elapsed, time_elapsed / tot_atoms_num * 1e3))
            self.ref_df_changed = True
        else:
            log.info("ASE atoms ('ase_atoms' column) are already in dataframe")

        log.info("Atomic environment representation construction...")
        start = time.time()
        self.apply_ase_atoms_transformers(ref_df)
        end = time.time()
        time_elapsed = end - start
        log.info("Atomic environment representation construction...done within {} sec ({} ms/atom)".
                 format(time_elapsed, time_elapsed / tot_atoms_num * 1e3))
        return ref_df

    def apply_create_ase_atoms(self, df):
        self._setup_pandas_parallel()
        if self.parallel:
            df["ase_atoms"] = df.parallel_apply(create_ase_atoms, axis=1)
        else:
            if self.progress_bar:
                from tqdm import tqdm
                tqdm.pandas()
                df["ase_atoms"] = df.progress_apply(create_ase_atoms, axis=1)
            else:
                df["ase_atoms"] = df.apply(create_ase_atoms, axis=1)

    def _setup_pandas_parallel(self):
        if self.parallel:
            cpu_count = None
            if isinstance(self.parallel, int):
                cpu_count = self.parallel
            parallel_pandas = import_parallel_processing_tools(nb_workers=cpu_count, progress_bar=self.progress_bar)
            if not parallel_pandas:
                self.parallel = False
                log.error("Parallel pandas processing is not loaded, switching to serial mode")

    def apply_ase_atoms_transformers(self, df):
        self._setup_pandas_parallel()
        if self.parallel:
            apply_function = df["ase_atoms"].parallel_apply
        elif self.progress_bar:
            from tqdm import tqdm
            tqdm.pandas()
            apply_function = df["ase_atoms"].progress_apply
        else:
            apply_function = df["ase_atoms"].apply

        for res_column_name, (transformer, kwargs) in self.ase_atoms_transformers.items():
            if res_column_name not in df.columns:
                log.info("Building '{}'...".format(res_column_name))
                l1 = len(df)
                log.info("Dataframe size before transform: " + str(l1))
                df[res_column_name] = apply_function(transformer, **kwargs)
                df.dropna(subset=[res_column_name], inplace=True)
                df.reset_index(drop=True, inplace=True)
                l2 = len(df)
                log.info("Dataframe size after transform: " + str(l2))
                self.ref_df_changed = True
            else:
                log.info("'{}' already in dataframe, skipping...".format(res_column_name))

    def load_or_query_ref_structures_dataframe(self, force_query=None):
        self.ref_df_changed = False
        if force_query is None:
            force_query = self.force_query
        file_to_load = self.filename or self.get_default_ref_filename()
        log.info("Search for cache ref-file: " + str(file_to_load))
        ref_energy = None
        if self.raw_df is not None and not force_query:
            self.df = self.raw_df
        elif file_to_load is not None and os.path.isfile(file_to_load) and not force_query:
            log.info(file_to_load + " found, try to load")
            self.df = load_dataframe(file_to_load, compression="infer")
        else:  # if ref_df is still not loaded, try to query from DB
            if not force_query:
                log.info("Cache not found, querying database")
            else:
                log.info("Forcing query database")
            self.df, ref_energy = query_data(config=self.config, seed=self.seed, query_limit=self.query_limit,
                                             db_conn_string=self.db_conn_string)
            self.ref_df_changed = True

        # check, that all necessary columns are there
        self.df = self.process_ref_dataframe(self.df, ref_energy)

        return self.df

    def get_ref_dataframe(self, force_query=None, cache_ref_df=False):
        self.ref_df_changed = False
        if force_query is None:
            force_query = self.force_query
        if self.df is None:
            self.df = self.load_or_query_ref_structures_dataframe(force_query=force_query)
        if cache_ref_df or self.cache_ref_df:
            if self.ref_df_changed:
                filename = self.filename or self.get_default_ref_filename() or "df_ref.pckl.gzip"
                log.info("Saving processed raw dataframe into " + filename)
                save_dataframe(self.df, filename=filename)
            else:
                log.info("Reference dataframe was not changed, nothing to save")
        return self.df

    def get_fit_dataframe(self, force_query=None, weights_policy=None, ignore_weights=None):
        if force_query is None:
            force_query = self.force_query
        self.df = self.get_ref_dataframe(force_query=force_query)

        if ignore_weights is None:
            ignore_weights = self.ignore_weights

        # TODO: check if weights columns already in dataframe
        if WEIGHTS_ENERGY_COLUMN in self.df.columns and WEIGHTS_FORCES_COLUMN in self.df.columns and not ignore_weights:
            log.info("Both weighting columns ({} and {}) are found, no another weighting policy will be applied".format(
                WEIGHTS_ENERGY_COLUMN, WEIGHTS_FORCES_COLUMN))
        else:
            if ignore_weights and (
                    WEIGHTS_ENERGY_COLUMN in self.df.columns or WEIGHTS_FORCES_COLUMN in self.df.columns):
                log.info("Existing weights are ignored, weighting policy calculation is forced")

            if weights_policy is not None:
                self.set_weights_policy(weights_policy)

            if self.weights_policy is None:
                log.info("No weighting policy is specified, setting default weighting policy")
                self.set_weights_policy(UniformWeightingPolicy())

            log.info("Apply weights policy: " + str(self.weights_policy))
            self.df = self.weights_policy.generate_weights(self.df)
        return self.df


class EnergyBasedWeightingPolicy(StructuresDatasetWeightingPolicy):

    def __init__(self, nfit=20000, cutoff=None, DElow=1.0, DEup=10.0, DE=1.0, DF=1.0, wlow=0.75, reftype='all',
                 seed=None):
        # #### Data selection and weighting
        # number of structures to be used in fit
        self.nfit = nfit
        # lower threshold: all structures below lower threshold are used in the fit (if fewer than nfit)
        self.DElow = DElow
        # upper threshold: structures between lower and upper threshold are selected randomly (after DElow structures)
        self.DEup = DEup
        # Delta E: energy offset in energy weights
        self.DE = DE
        # Delta F: force offset in force weights
        self.DF = DF
        # relative fraction of structures below lower threshold in energy weights
        self.wlow = wlow
        # use all/bulk/cluster reference data
        self.reftype = reftype
        # random seed
        self.seed = seed

        self.cutoff = cutoff

    def __str__(self):
        return ("EnergyBasedWeightingPolicy(nfit={nfit}, cutoff={cutoff}, DElow={DElow}, DEUp={DEup}, DE={DE}, " + \
                "DF={DF}, wlow={wlow}, reftype={reftype},seed={seed})").format(nfit=self.nfit,
                                                                               cutoff=self.cutoff,
                                                                               DElow=self.DElow,
                                                                               DEup=self.DEup,
                                                                               DE=self.DE,
                                                                               DF=self.DF,
                                                                               wlow=self.wlow,
                                                                               reftype=self.reftype, seed=self.seed)

    def generate_weights(self, df):
        if self.reftype == "bulk":
            log.info("Reducing to bulk data")
            df = df[df.pbc].reset_index(drop=True)
        elif self.reftype == "cluster":
            log.info("Reducing to cluster data")
            df = df[~df.pbc].reset_index(drop=True)
        else:
            log.info("Keeping bulk and cluster data")

        if self.cutoff is not None:
            log.info("Collecting shortest bond lengths")
            if ATOMIC_ENV_COLUMN not in df.columns:
                df[ATOMIC_ENV_COLUMN] = df["ase_atoms"].apply(aseatoms_to_atomicenvironment, cutoff=self.cutoff)
            if DMIN_COLUMN not in df.columns:
                df[DMIN_COLUMN] = df[ATOMIC_ENV_COLUMN].map(calc_min_distance)
            valid_dmin_mask = (df[DMIN_COLUMN] < self.cutoff)
            non_valid_dmin_mask = ~valid_dmin_mask
            log.info("Valid structures: {} ({:.2f}%)".format(np.sum(valid_dmin_mask), 100 * np.mean(valid_dmin_mask)))
            log.info("Non-valid structures: {} ({:.2f}%)".format(np.sum(non_valid_dmin_mask),
                                                                 100 * np.mean(non_valid_dmin_mask)))

            log.info(
                "{} structures outside cutoff that will now be removed from dataframe".format(
                    non_valid_dmin_mask.sum()))

            df = df[valid_dmin_mask].reset_index(drop=True)
        else:
            log.info("No EnergyBasedWeightingPolicy(...cutoff=...) is provided, no structures outside cutoff that " +
                     "will now be removed")

        # #### structure selection
        log.info("Structure selection for fitting")
        emin = df[E_CORRECTED_PER_ATOM_COLUMN].min()

        # remove high energy structures
        df = df[df[E_CORRECTED_PER_ATOM_COLUMN] < (emin + self.DEup)].reset_index(drop=True)

        elow_mask = df[E_CORRECTED_PER_ATOM_COLUMN] < (emin + self.DElow)
        eup_mask = (df[E_CORRECTED_PER_ATOM_COLUMN] >= (emin + self.DElow))
        nlow = elow_mask.sum()
        nup = eup_mask.sum()
        log.info("{} structures below DElow={} eV/atom".format(nlow, self.DElow))
        log.info("{} structures between DElow={} eV/atom and DEup={} eV/atom".format(nup, self.DElow, self.DEup))
        log.info("all other structures were removed")

        lowlist = np.where(elow_mask)[0]
        uplist = np.where(eup_mask)[0]

        np.random.seed(self.seed)

        if nlow <= self.nfit:
            takelist = lowlist
        else:  # nlow >nfit
            takelist = np.random.choice(lowlist, self.nfit, replace=False)

        nremain = self.nfit - len(takelist)

        if nremain <= nup:
            takelist = np.hstack([takelist, np.random.choice(uplist, nremain, replace=False)])
        else:
            takelist = np.hstack([takelist, uplist])

        np.random.shuffle(takelist)

        df = df.loc[takelist].reset_index(drop=True)

        elow_mask = df[E_CORRECTED_PER_ATOM_COLUMN] < (emin + self.DElow)
        eup_mask = (df[E_CORRECTED_PER_ATOM_COLUMN] >= (
                emin + self.DElow))

        log.info(str(len(df)) + " structures were selected")
        assert elow_mask.sum() + eup_mask.sum() == len(df)
        if nremain == 0 and self.wlow != 1.0:
            log.warning(("All structures were taken from low-tier, but relative weight of low-tier (wlow={}) " +
                         "is less than one. It will be adjusted to one").format(self.wlow))
            self.wlow = 1.0
        # ### energy weights
        log.info("Setting up energy weights")
        DE = abs(self.DE)

        df[WEIGHTS_ENERGY_COLUMN] = 1 / (df[E_CORRECTED_PER_ATOM_COLUMN] - emin + DE) ** 2
        df[WEIGHTS_ENERGY_COLUMN] = df[WEIGHTS_ENERGY_COLUMN] / df[WEIGHTS_ENERGY_COLUMN].sum()
        # log.info('df["w_energy"].sum()={}'.format(df["w_energy"].sum()))
        assert np.allclose(df[WEIGHTS_ENERGY_COLUMN].sum(), 1)
        #  ### relative weights of structures below and above threshold DElow
        wlowcur = df.loc[elow_mask, WEIGHTS_ENERGY_COLUMN].sum()
        wupcur = df.loc[eup_mask, WEIGHTS_ENERGY_COLUMN].sum()

        log.info("Current relative energy weights: {}/{}".format(wlowcur, wupcur))
        if wlowcur < 1.0 and wlowcur > 0.:
            log.info("Will be adjusted to            : {}/{}".format(self.wlow, 1 - self.wlow))
            flow = self.wlow / wlowcur
            if wlowcur == 1:
                fup = 0
            else:
                fup = (1 - self.wlow) / (1 - wlowcur)

            df.loc[elow_mask, WEIGHTS_ENERGY_COLUMN] = flow * df.loc[elow_mask, WEIGHTS_ENERGY_COLUMN]
            df.loc[eup_mask, WEIGHTS_ENERGY_COLUMN] = fup * df.loc[eup_mask, WEIGHTS_ENERGY_COLUMN]
            # log.info('df["w_energy"].sum() after = {}'.format(df["w_energy"].sum()))
            energy_weights_sum = df[WEIGHTS_ENERGY_COLUMN].sum()
            assert np.allclose(energy_weights_sum, 1), "Energy weights sum differs from one and equal to {}".format(
                energy_weights_sum)
            wlowcur = df.loc[elow_mask, WEIGHTS_ENERGY_COLUMN].sum()
            wupcur = df.loc[eup_mask, WEIGHTS_ENERGY_COLUMN].sum()
            log.info("After adjustment: relative energy weights: {}/{}".format(wlowcur, wupcur))
            assert np.allclose(wlowcur, self.wlow)
            assert np.allclose(wupcur, 1 - self.wlow)
        else:
            log.warning("No weights adjustment possible")

        # ### force weights
        log.info("Setting up force weights")
        DF = abs(self.DF)
        df[FORCES_COLUMN] = df[FORCES_COLUMN].map(np.array)
        df[WEIGHTS_FORCES_COLUMN] = df[FORCES_COLUMN].map(lambda forces: 1 / (np.sum(forces ** 2, axis=1) + DF))
        assert (df[WEIGHTS_FORCES_COLUMN].map(len) == df["NUMBER_OF_ATOMS"]).all()
        df[WEIGHTS_FORCES_COLUMN] = df[WEIGHTS_FORCES_COLUMN] * df[WEIGHTS_ENERGY_COLUMN]
        w_forces_norm = df[WEIGHTS_FORCES_COLUMN].map(sum).sum()
        df[WEIGHTS_FORCES_COLUMN] = df[WEIGHTS_FORCES_COLUMN] / w_forces_norm

        energy_weights_sum = df[WEIGHTS_ENERGY_COLUMN].sum()
        assert np.allclose(energy_weights_sum, 1), "Energy weights sum differs from one and equal to {}".format(
            energy_weights_sum)
        forces_weights_sum = df[WEIGHTS_FORCES_COLUMN].map(sum).sum()
        assert np.allclose(forces_weights_sum, 1), "Forces weights sum differs from one and equal to {}".format(
            forces_weights_sum)
        return df

    def plot(self, df):
        import matplotlib.pyplot as plt
        elist = df[E_CORRECTED_PER_ATOM_COLUMN]
        dminlist = df["dmin"]
        print("Please check that your cutoff makes sense in the following graph")
        xh = [0, self.cutoff + 1]
        yh = [10 ** -3, 10 ** -3]
        yv = [10 ** -10, 10 ** 3]
        xv = [self.cutoff, self.cutoff]
        fig, ax = plt.subplots(1)

        ax.semilogy(dminlist, elist.abs(), '+', label="data")
        ax.semilogy(xh, yh, '--', label="1 meV")
        ax.semilogy(xv, yv, '-', label="cutoff")

        plt.ylabel(r"| cohesive energy | / eV")
        plt.xlabel("dmin / ${\mathrm{\AA}}$")
        plt.title("Reference data overview")
        plt.legend()
        plt.xlim(1, self.cutoff + 0.5)
        plt.ylim(10 ** -4, 10 ** 2)
        plt.show()


class UniformWeightingPolicy(StructuresDatasetWeightingPolicy):

    def __init__(self):
        pass

    def __str__(self):
        return "UniformWeightingPolicy()"

    def generate_weights(self, df):
        df[WEIGHTS_ENERGY_COLUMN] = 1. / len(df)

        df[WEIGHTS_FORCES_COLUMN] = df[FORCES_COLUMN].map(lambda forces: np.ones(len(forces)))

        # assert (df[WEIGHTS_FORCES_COLUMN].map(len) == df["NUMBER_OF_ATOMS"]).all()
        df[WEIGHTS_FORCES_COLUMN] = df[WEIGHTS_FORCES_COLUMN] * df[WEIGHTS_ENERGY_COLUMN]

        # w_forces_norm = df[WEIGHTS_FORCES_COLUMN].map(sum).sum()
        # df[WEIGHTS_FORCES_COLUMN] = df[WEIGHTS_FORCES_COLUMN] / w_forces_norm

        # assert np.allclose(df[WEIGHTS_ENERGY_COLUMN].sum(), 1)
        # assert np.allclose(df[WEIGHTS_FORCES_COLUMN].map(sum).sum(), 1)
        normalize_energy_forces_weights(df)
        return df


def normalize_energy_forces_weights(df: pd.DataFrame) -> pd.DataFrame:
    if WEIGHTS_ENERGY_COLUMN not in df.columns:
        raise ValueError("`{}` column not in dataframe".format(WEIGHTS_ENERGY_COLUMN))
    if WEIGHTS_FORCES_COLUMN not in df.columns:
        raise ValueError("`{}` column not in dataframe".format(WEIGHTS_FORCES_COLUMN))

    assert (df[WEIGHTS_FORCES_COLUMN].map(len) == df[FORCES_COLUMN].map(len)).all()

    df[WEIGHTS_ENERGY_COLUMN] = df[WEIGHTS_ENERGY_COLUMN] / df[WEIGHTS_ENERGY_COLUMN].sum()
    # df[WEIGHTS_FORCES_COLUMN] = df[WEIGHTS_FORCES_COLUMN] * df[WEIGHTS_ENERGY_COLUMN]
    w_forces_norm = df[WEIGHTS_FORCES_COLUMN].map(sum).sum()
    df[WEIGHTS_FORCES_COLUMN] = df[WEIGHTS_FORCES_COLUMN] / w_forces_norm

    assert np.allclose(df[WEIGHTS_ENERGY_COLUMN].sum(), 1)
    assert np.allclose(df[WEIGHTS_FORCES_COLUMN].map(sum).sum(), 1)
    return df

def get_weighting_policy(weighting_policy_spec: Dict) -> StructuresDatasetWeightingPolicy:
    weighting_policy = None
    if weighting_policy_spec is not None:
        if isinstance(weighting_policy_spec, dict):
            weighting_policy_spec = weighting_policy_spec.copy()
            log.debug("weighting_policy_spec: " + str(weighting_policy_spec))

            if WEIGHTING_TYPE_KW not in weighting_policy_spec:
                raise ValueError("Weighting 'type' is not specified")

            if weighting_policy_spec[WEIGHTING_TYPE_KW] == ENERGYBASED_WEIGHTING_POLICY:
                del weighting_policy_spec[WEIGHTING_TYPE_KW]
                weighting_policy = EnergyBasedWeightingPolicy(**weighting_policy_spec)
            else:
                raise ValueError("Unknown weighting 'type': " + weighting_policy_spec[WEIGHTING_TYPE_KW])
        elif isinstance(weighting_policy_spec, StructuresDatasetWeightingPolicy):
            return weighting_policy
        else:
            raise ValueError("Unknown 'weighting' option type: " + str(type(weighting_policy_spec)))
    return weighting_policy


def get_dataset_specification(evaluator_name, data_config: Dict,
                              cutoff=10) -> StructuresDatasetSpecification:
    if isinstance(data_config, str):
        spec = StructuresDatasetSpecification(filename=data_config, cutoff=cutoff)
    elif isinstance(data_config, dict):
        spec = StructuresDatasetSpecification(**data_config, cutoff=cutoff)
    else:
        raise ValueError("Unknown data specification type: " + str(type(data_config)))

    # add the transformer depending on the evaluator
    if evaluator_name == PYACE_EVAL:
        spec.add_ase_atoms_transformer(ATOMIC_ENV_DF_COLUMN, aseatoms_to_atomicenvironment, cutoff=cutoff)
    elif evaluator_name == TENSORPOT_EVAL:
        from tensorpotential.utils.utilities import generate_tp_atoms
        spec.add_ase_atoms_transformer(TP_ATOMS_DF_COLUMN, generate_tp_atoms, cutoff=cutoff)

    return spec


def get_reference_dataset(evaluator_name, data_config: Dict, cutoff=10, force_query=False, cache_ref_df=True):
    spec = get_dataset_specification(evaluator_name=evaluator_name, data_config=data_config,
                                     cutoff=cutoff)
    return spec.get_ref_dataframe(force_query=force_query, cache_ref_df=cache_ref_df)


def get_fitting_dataset(evaluator_name, data_config: Dict, weighting_policy_spec: Dict = None,
                        cutoff=10, force_query=False, force_weighting=None) -> pd.DataFrame:
    spec = get_dataset_specification(evaluator_name=evaluator_name, data_config=data_config,
                                     cutoff=cutoff)
    spec.set_weights_policy(get_weighting_policy(weighting_policy_spec))

    df = spec.get_fit_dataframe(force_query=force_query, ignore_weights=force_weighting)
    normalize_energy_forces_weights(df)
    return df
