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


import os
from pyace.basis import ACE_B_to_CTildeBasisSet

def convert_yaml_to_ace(infile, outfile):
    """
    Utility to convert yaml format to ace format

    Parameters
    ----------
    infile: string
        input file name

    outfile: string
        output file name
    """
    if not os.path.exists(infile):
        raise FileNotFoundError("input file does not exist")

    basis = ACE_B_to_CTildeBasisSet()
    basis.load_yaml(infile)
    basis.save(outfile)
    
