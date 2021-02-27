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
