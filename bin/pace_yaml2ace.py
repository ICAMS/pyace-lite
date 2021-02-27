# /*
# * Atomic cluster expansion
# *
# * Copyright 2021  (c) Yury Lysogorskiy
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

from pyace import ACEBBasisSet

parser = argparse.ArgumentParser(prog="pace_yaml2ace",
                                 description="Conversion utility from B-basis (.yaml file) to Ctilde-basis (.ace file)")
parser.add_argument("input", help="input B-basis file name (.yaml)", type=str)
parser.add_argument("-o", "--output", help="output Ctilde-basis file name (.ace)", type=str, default="")

args_parse = parser.parse_args()
input_yaml_filename = args_parse.input
output_ace_filename = args_parse.output

if output_ace_filename == "":
    if input_yaml_filename.endswith("yaml"):
        output_ace_filename = input_yaml_filename.replace("yaml", "ace")
    elif input_yaml_filename.endswith("yml"):
        output_ace_filename = input_yaml_filename.replace("yml", "ace")
    else:
        output_ace_filename = input_yaml_filename + ".ace"

print("Loading B-basis from '{}'".format(input_yaml_filename))
bbasis = ACEBBasisSet(input_yaml_filename)
print("Converting to Ctilde-basis")
cbasis = bbasis.to_ACECTildeBasisSet()
print("Saving Ctilde-basis to '{}'".format(output_ace_filename))
cbasis.save(output_ace_filename)
