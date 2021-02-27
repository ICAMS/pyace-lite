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

import numpy as np

from ase.calculators.calculator import Calculator, all_changes

from pyace.basis import ACEBBasisSet, ACECTildeBasisSet, BBasisConfiguration
from pyace.calculator import ACECalculator
from pyace.catomicenvironment import ACEAtomicEnvironment
from pyace.atomicenvironment import aseatoms_to_atomicenvironment
from pyace.evaluator import ACEBEvaluator, ACECTildeEvaluator


class PyACECalculator(Calculator):
    """
    PyACE ASE calculator
    :param basis_set - specification of ACE potential, could be in following forms:
                      ".ace" potential filename
                      ".yaml" potential filename
                      ACEBBasisSet object
                      ACECTildeBasisSet object
                      BBasisConfiguration object
    """
    implemented_properties = ['energy', 'forces', 'stress', 'energies', 'free_energy']

    def __init__(self, basis_set, **kwargs):
        """
PyACE ASE calculator
:param basis_set - specification of ACE potential, could be in following forms:
                  ".ace" potential filename
                  ".yaml" potential filename
                  ACEBBasisSet object
                  ACECTildeBasisSet object
                  BBasisConfiguration object
"""
        Calculator.__init__(self, basis_set=basis_set, **kwargs)
        self.nl = None
        self.skin = 0.
        # self.reset_nl = True  # Set to False for MD simulations
        self.ae = ACEAtomicEnvironment()

        self._create_evaluator()

        self.cutoff = self.basis.radial_functions.cutoff  # self.parameters.basis_config.funcspecs_blocks[0].rcutij

        self.energy = None
        self.energies = None
        self.forces = None
        self.virial = None
        self.stress = None
        self.projections = None

        self.ace = ACECalculator()
        self.ace.set_evaluator(self.evaluator)

    def _create_evaluator(self):

        basis_set = self.parameters.basis_set
        if isinstance(basis_set, BBasisConfiguration):
            self.basis = ACEBBasisSet(self.parameters.basis_set)
        elif isinstance(basis_set, (ACEBBasisSet, ACECTildeBasisSet)):
            self.basis = basis_set
        elif isinstance(basis_set, str):
            if basis_set.endswith(".yaml"):
                self.basis = ACEBBasisSet(basis_set)
            elif basis_set.endswith(".ace"):
                self.basis = ACECTildeBasisSet(basis_set)
            else:
                raise ValueError("Unrecognized file format: " + basis_set)
        else:
            raise ValueError("Unrecognized basis set specification")

        self.elements_name = np.array(self.basis.elements_name).astype(dtype="S2")
        self.elements_mapper_dict = {el: i for i, el in enumerate(self.elements_name)}

        if isinstance(self.basis, ACEBBasisSet):
            self.evaluator = ACEBEvaluator(self.basis)
        elif isinstance(self.basis, ACECTildeBasisSet):
            self.evaluator = ACECTildeEvaluator(self.basis)

    def get_atomic_env(self, atoms):
        try:
            self.ae = aseatoms_to_atomicenvironment(atoms, cutoff=self.cutoff,
                                                    skin=self.skin,
                                                    elements_mapper_dict=self.elements_mapper_dict)
        except KeyError as e:
            raise ValueError("Unsupported species type: " + str(e))
        return self.ae

    def calculate(self, atoms=None, properties=['energy', 'forces', 'stress', 'energies'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        self.energy = 0.0
        self.energies = np.zeros(len(atoms))
        self.forces = np.empty((len(atoms), 3))

        self.get_atomic_env(atoms)
        self.ace.compute(self.ae)

        self.energy, self.forces = np.array(self.ace.energy), np.array(self.ace.forces)
        nat = len(atoms)
        proj1 = np.reshape(self.ace.basis_projections_rank1, (nat, -1))
        proj2 = np.reshape(self.ace.basis_projections, (nat, -1))
        self.projections = np.concatenate([proj1, proj2], axis=1)

        self.energies = np.array(self.ace.energies)

        self.results = {
            'energy': np.float64(self.energy.reshape(-1, )),
            'free_energy': np.float64(self.energy.reshape(-1, )),
            'forces': self.forces.astype(np.float64),
            'energies': self.energies.astype(np.float64)
        }
        if self.atoms.number_of_lattice_vectors == 3:
            self.volume = atoms.get_volume()
            self.virial = np.array(self.ace.virial)  # order is: xx, yy, zz, xy, xz, yz
            # swap order of the virials to fullfill ASE Voigt stresses order:  (xx, yy, zz, yz, xz, xy)
            self.stress = self.virial[[0, 1, 2, 5, 4, 3]] / self.volume
            self.results["stress"] = self.stress


if __name__ == '__main__':
    from ase.build import bulk
    from pyace.basis import BBasisConfiguration, BBasisFunctionSpecification, BBasisFunctionsSpecificationBlock

    block = BBasisFunctionsSpecificationBlock()

    block.block_name = "Al"
    block.nradmaxi = 1
    block.lmaxi = 0
    block.npoti = "FinnisSinclair"
    block.fs_parameters = [1, 1, 1, 0.5]
    block.rcutij = 8.7
    block.dcutij = 0.01
    block.NameOfCutoffFunctionij = "cos"
    block.nradbaseij = 1
    block.radbase = "ChebExpCos"
    block.radparameters = [3.0]
    block.radcoefficients = [1]

    block.funcspecs = [
        BBasisFunctionSpecification(["Al", "Al"], ns=[1], ls=[0], LS=[], coeffs=[1.]),
        # BBasisFunctionSpecification(["Al", "Al", "Al"], ns=[1, 1], ls=[0, 0], LS=[], coeffs=[2])
    ]

    basisConfiguration = BBasisConfiguration()
    basisConfiguration.deltaSplineBins = 0.001
    basisConfiguration.funcspecs_blocks = [block]

    a = bulk('Al', 'fcc', a=4, cubic=True)
    a.pbc = False
    print(a)
    calc = PyACECalculator(basis_set=basisConfiguration)
    a.set_calculator(calc)
    e1 = (a.get_potential_energy())
    f1 = a.get_forces()
    print(e1)
    print(f1)

    calc2 = PyACECalculator(basis_set=ACEBBasisSet(basisConfiguration))
    a2 = bulk('Al', 'fcc', a=4, cubic=True)
    a2.set_calculator(calc2)
    e2 = (a2.get_potential_energy())
    f2 = a2.get_forces()
    print(e2)
    print(f2)
