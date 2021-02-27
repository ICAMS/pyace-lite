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
#get everything from atomic environment
from pyace.catomicenvironment import ACEAtomicEnvironment
from pyace.pyneighbor import ACENeighborList
import warnings
from ase import Atoms

def create_cube(dr, cube_side_length):
    """
    Create a simple cube without pbc. Test function
    """
    warnings.warn("This function is retained for testing purposes only")
    posx = np.arange(-cube_side_length/2, cube_side_length/2 + dr/2, dr)
    posy = np.arange(-cube_side_length/2, cube_side_length/2 + dr/2, dr)
    posz = np.arange(-cube_side_length/2, cube_side_length/2 + dr/2, dr)
    n_atoms = len(posx)*len(posy)*len(posz)

    positions = []
    for x in posx:
        for y in posy:
            for z in posz:
                positions.append([x, y, z])

    atoms = Atoms(positions=positions, symbols=["W"]*len(positions))
    nl=ACENeighborList()
    nl.make_neighborlist(atoms)
    ae = ACEAtomicEnvironment()
    ae.x = nl.x
    ae.species_type = nl.species_type
    ae.neighbour_list = nl.jlists
    return ae

def create_linear_chain(natoms, axis=2):
    """
    Create a linear chain along partcular axis

    Parameters
    ----------
    natoms: int
        number of atoms

    axis : int, optional
        default 2. Axis along which linear chain is created
    """
    warnings.warn("This function is retained for testing purposes only")

    positions = []
    for i in range(natoms):
        pos = [0,0,0]
        pos[axis] = (i - natoms/2)
        positions.append(pos)
    atoms = Atoms(positions=positions, symbols=["W"]*len(positions))
    nl=ACENeighborList()
    nl.make_neighborlist(atoms)
    ae = ACEAtomicEnvironment()
    ae.x = nl.x
    ae.species_type = nl.species_type
    ae.neighbour_list = nl.jlists
    return ae


def aseatoms_to_atomicenvironment(atoms, cutoff=9, skin=0, elements_mapper_dict=None):
    """
    Function to read from a ASE atoms objects

    Parameters
    ----------
    atoms : ASE Atoms object
        name of the ASE atoms object

    is_triclinic : bool, optional
        True if the configuration is triclinic

    periodic : list of ints, optional
        periodicity of the system can be indicated using indices such as
        [1,1,1] or [1,0,0] which stands for [x,y,z]. By default [1,1,1]
        is taken.

    cutoff: float
        cutoff value for calculating neighbors
    """

    nl = ACENeighborList(cutoff=cutoff, skin=skin)
    if elements_mapper_dict is not None:
        nl.types_mapper_dict = elements_mapper_dict
    nl.make_neighborlist(atoms)
    ae = ACEAtomicEnvironment()
    ae.x = nl.x
    ae.species_type = nl.species_type
    ae.neighbour_list = nl.jlists
    ae.origins = nl.origins
    return ae
