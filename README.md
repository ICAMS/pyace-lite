# pyace

`pyace` is the python implementation of Atomic Cluster Expansion.
It provides the basis functionality for analysis, potentials conversion and fitting.
!!! THIS IS LIMITED FUNCTIONALITY VERSION OF `pyace` !!! 

Please, contact us by email yury.lysogorskiy@rub.de if you want to have fully-functional version

## Installation

```
pip install pyace-lite
```
 
### (optional) Installation of `tensorpotential`  
If you want to use `TensorFlow` implementation of atomic cluster expansion 
(made by *Dr. Anton Bochkarev*), then contact us by email.

### (!) Known issues
If you will encounter `segmentation fault` errors,  then try to upgrade the `numpy` package with the command:
```
pip install --upgrade numpy --force 
```

## Directory structure

- **lib/**: contains the extra libraries for `pyace`
- **src/pyace/**: bindings

# Utilities
## Potential conversion

There are **two** basic formats ACE potentials:

1. **B-basis set** in YAML format, i.e. 'Al.pbe.yaml'. This is an internal developers *complete* format 
2. **Ctilde-basis set** in plain text format, i.e. 'Al.pbe.ace'. This format is *irreversibly* converted from *B-basis set* for
public potentials distribution and is used by LAMMPS.

To convert potential you can use following utilities, that are installed together with `pyace` package into you executable paths:
  * `YAML` to `ace` : `pace_yaml2ace`. Usage:
```
  pace_yaml2ace [-h] [-o OUTPUT] input
```

## Pacemaker

`pacemaker` is an utility for fitting the atomic cluster expansion potential. Usage:

```
pacemaker [-h] [-o OUTPUT] [-p POTENTIAL] [-ip INITIAL_POTENTIAL]
                 [-b BACKEND] [-d DATA] [--query-data] [--prepare-data]
                 [-l LOG]
                 input

Fitting utility for atomic cluster expansion potentials

positional arguments:
  input                 input YAML file

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        output B-basis YAML file name, default:
                        output_potential.yaml
  -p POTENTIAL, --potential POTENTIAL
                        input potential YAML file name, will override input
                        file 'potential' section
  -ip INITIAL_POTENTIAL, --initial-potential INITIAL_POTENTIAL
                        initial potential YAML file name, will override input
                        file 'potential::initial_potential' section
  -b BACKEND, --backend BACKEND
                        backend evaluator, will override section
                        'backend::evaluator' from input file
  -d DATA, --data DATA  data file, will override section 'YAML:fit:filename'
                        from input file
  --query-data          query the training data from database, prepare and
                        save them
  --prepare-data        prepare and save training data only
  -l LOG, --log LOG     log filename (default log.txt)
``` 

The required settings are provided by input YAML file. The main sections
#### 1. Cutoff and  (optional) metadata

* Global cutoff for the fitting is setup as:

```YAML
cutoff: 10.0
```

* Metadata (optional)

This is arbitrary key (string)-value (string) pairs that would be added to the potential YAML file: 
```YAML
metadata:
  info: some info
  comment: some comment
  purpose: some purpose
```
Moreover, `starttime` and `user` fields would be added automatically

#### 2.Dataset specification section
Fitting dataset could be queried automatically from `structdb` (if corresponding `structdborm` package is installed and 
connection to database is configured, see `structdb.ini` file in home folder). Alternatively, dataset could be saved into
file as a pickled `pandas` dataframe with special names for columns: #TODO: add columns names
 
Example:
```YAML
data: # dataset specification section
  # data configuration section
  config:
    element: Al                    # element name
    calculator: FHI-aims/PBE/tight # calculator type from `structdb` 
    # ref_energy: -1.234           # single atom reference energy
                                   # if not specified, then it will be queried from database

  # seed: 42                       # random seed for shuffling the data  
  # query_limit: 1000              # limiting number of entries to query from `structdb`
                                   # ignored if reading from cache
  
  # parallel: 3                    # number of parallel workers to preprocess the data, `pandarallel` package required
                                   # if not specified, serial mode will be used 
  # cache_ref_df: True             # whether to store the queried or modified dataset into file, default - True
  # filename: some.pckl.gzip       # force to read reference pickled dataframe from given file
  # ignore_weights: False          # whether to ignore energy and force weighting columns in dataframe
  # datapath: ../data              # path to folder with cache files with pickled dataframes 
```
Alternatively, instead of `data::config` section, one can specify just the cache file 
with pickled dataframe as `data::filename`:
```YAML
data: 
  filename: small_df_tf_atoms.pckl
  datapath: ../tests/
```

Example of creating the **subselection of fitting dataframe** and saving it is given in `notebooks/data_preprocess.ipynb`

Example of generating **custom energy/forces weights** is given in `notebooks/data_custom_weights.ipynb`

##### Querying data
You can just query and preprocess data, without running potential fitting.
Here is the minimalistic input YAML:

```YAML
# input.yaml file

cutoff: 10.0  # use larger cutoff to have excess neighbour list
data: # dataset specification section
  config:
    element: Al                    # element name
    calculator: FHI-aims/PBE/tight # calculator type from `structdb`
  seed: 42
  parallel: 3                      # parallel data processing. WARNING! higher memory usage is possible
  datapath: ../data                # path to the directory with cache files
  # query_limit: 100               # number of entries to query  
```

Then execute `pacemaker --query-data input.yaml` to query and build the dataset with `pyace` neighbour lists.
For building *both* `pyace` and `tensorpot` neighbour lists use `pacemaker --query-data input.yaml -b tensorpot`

##### Preparing the data / constructing neighbourlists
You can use existing `.pckl.gzip` dataset and generate all necessary columns for that, including neighbourlists
Here is the minimalistic input YAML:

```YAML
# input.yaml file

cutoff: 10.

data:
  filename: my_dataset.pckl.gzip

backend:
  evaluator: tensorpot  # pyace, tensorpot

```

Then execute `pacemaker --prepare-data input.yaml`
Generation of the `my_dataset.pckl.gzip` from, for example, *pyiron* is shown in `notebooks/convert-pyiron-to-pacemaker.ipynb` 

#### 3. Interatomic potential (or B-basis) configuration
One could define the initial interatomic potential configuration as:
```YAML
potential:
  deltaSplineBins: 0.001
  element: Al
  fs_parameters: [1, 1, 1, 0.5]
  npot: FinnisSinclair
  NameOfCutoffFunction: cos

  rankmax: 3
  nradmax: [ 4, 3, 3 ]  # per-rank values of nradmax
  lmax: [ 0, 1, 1 ]     # per-rank values of lmax,  lmax=0 for first rank always!

  ndensity: 2
  rcut: 8.7
  dcut: 0.01
  radparameters: [ 5.25 ]
  radbase: ChebExpCos

 ##hard-core repulsion (optional)
 # core-repulsion: [500, 10]
 # rho_core_cut: 50
 # drho_core_cut: 20

 # basisdf:  /some/path/to/pyace_bbasisfunc_df.pckl      # path to the dataframe with "white list" of basis functions to use in fit
 # initial_potential: whatever.yaml                      # in "ladder" fitting scheme, potential from with to start fit
```
If you want to continue fitting of the existing potential in `potential.yaml` file, then specify
```YAML
potential: potential yaml
```

Alternatively, one could use `pacemaker ... -p potential.yaml ` option


#### 4. Fitting settings
Example of `fit` section is:
```YAML
fit:
  loss: { kappa: 0, L1_coeffs: 0,  L2_coeffs: 0,  w1_coeffs: 0, w2_coeffs: 0,
          w0_rad: 0, w1_rad: 0, w2_rad: 0 }

  weighting:
   type: EnergyBasedWeightingPolicy
    nfit: 10000
    cutoff: 10
    DElow: 1.0
    DEup: 10.0
    DE: 1.0
    DF: 1.0
    wlow: 0.75
   seed: 42

  optimizer: BFGS # L-BFGS-B # Nelder-Mead
  maxiter: 1000

  # fit_cycles: 2               # (optional) number of consequentive runs of fitting algorithm,
                                # that helps convergence 
  # noise_relative_sigma: 1e-2   # applying Gaussian noise with specified relative sigma/mean ratio to all potential optimizable coefficients
  # noise_absolute_sigma: 1e-3   # applying Gaussian noise with specified absolute sigma to all potential optimizable coefficients
  # ladder_step: [10, 0.02]     # Possible values:
                                #  - integer >= 1 - number of basis functions to add in ladder scheme,
                                #  - float between 0 and 1 - relative ladder step size wrt. current basis step
                                #  - list of both above values - select maximum between two possibilities on each iteration 
                                # see. Ladder scheme fitting for more info 
  # ladder_type: body_order     # default
                                # Possible values:
                                # body_order  -  new basis functions are added according to the body-order, i.e., a function with higher body-order
                                #                will not be added until the list of functions of the previous body-order is exhausted
                                # power_order -  the order of adding new basis functions is defined by the "power rank" p of a function.
                                #                p = len(ns) + sum(ns) + sum(ls). Functions with the smallest p are added first  
```
If not specified, then *uniform weight* and *energy-only* fit (kappa=0),
 *fit_cycles*=1, *noise_relative_sigma* = 0 settings will be used. 
 
#### 5. Backend specification
```YAML
backend:
  evaluator: pyace  # pyace, tensorpot

  ## for `pyace` evaluator, following options are available:
  # parallel_mode: process    # process, serial  - parallelization mode for `pyace` evaluator
  # n_workers: 4              # number of parallel workers for `process` parallelization mode

  ## for `tensorpot` evaluator, following options are available:
  # batch_size: 10            # batch size for loss function evaluation, default is 10 
  # display_step: 20          # frequency of detailed metric calculation and printing  
```
Alternatively, backend could be selected as `pacemaker ... -b tensorpot` 

##  Ladder scheme fitting 
In a ladder scheme potential fitting happens by adding new portion of basis functions step-by-step,
to form a "ladder" from *initial potential* to *final potential*. Following settings should be added to
the input YAML file:

* Specify *final potential* shape by providing `potential` section:
```yaml
potential:
  deltaSplineBins: 0.001
  element: Al
  fs_parameters: [1, 1, 1, 0.5]
  npot: FinnisSinclair
  NameOfCutoffFunction: cos
  rankmax: 3

  nradmax: [4, 1, 1]
  lmax: [0, 1, 1]

  ndensity: 2
  rcut: 8.7
  dcut: 0.01
  radparameters: [5.25]
  radbase: ChebExpCos 
```

* Specify *initial or intermediate potential* by providing `initial_potential` option in `potential` section: 
```yaml
potential:

    ...

    initial_potential: some_start_or_interim_potential.yaml    # potential to start fit from
```
If *initial or intermediate potential* is not specified, then fit will start from empty potential. 
Alternatively, you can specify *initial or intermediate potential* by command-line option

`pacemaker ... -ip some_start_or_interim_potential.yaml `

* Specify `ladder_step` in `fit` section:
```yaml
fit:

    ...

  ladder_step: [10, 0.02]       # Possible values:
                                #  - integer >= 1 - number of basis functions to add in ladder scheme,
                                #  - float between 0 and 1 - relative ladder step size wrt. current basis step
                                #  - list of both above values - select maximum between two possibilities on each iteration 
```

