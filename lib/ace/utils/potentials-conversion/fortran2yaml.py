#!/usr/bin/env python
import sys
from collections import OrderedDict, defaultdict

import numpy as np

try:
    from ruamel import yaml
except ImportError as e:
    print("Error: ", e.__class__.__name__ + ": " + e.message)
    print("ruamel package is not installed.")
    print("Run: 'pip install ruamel.yaml' or 'conda install -c conda-forge ruamel.yaml' ")
    raise e


# Fortran ACE file format conversion algebra
# Clu 2: rank = 1 => 1 ns, 0ls, 0 LS (ls=0, LS = .)
#
# Clu 3: rank = 2 => 2 ns, 1ls , 0LS
#
# clu 4: rank = 3 => 3 ns, 3 ls,  0 LS
#
# clu 5: rank = 4 => 4 ns, 4 ls, 1 LS
# clu 6: rank = 5 => 5ns, 5ls, 2 LS
#
# clu 7:  Rank = 6 => 6 ns, 6 ls, 3 LS
#
#
# rank = clu-1
#
# num_ns = rank
#
# num_ls = rank if(rank>=3) else  rank-1
# num_LS = rank-3 if(rank>=3) else  0
#
# num_ls_cpp = rank
# num_LS_cpp = max(rank-2,0)
#
# Rank 	num_ls	num_LS	num_ls_cpp	num_LS_cpp
# 1		0		0		1 (ls=0)		0
# 2		1		0		2 (l1=l2=l)	0
# 3		3		0		3			1   L(-1) = l(-1)
# 4		4		1		4			2   L(-1) = L(-2)
# 5		5		2		5			3   L(-1) = l(-1)
# 6		6		3		6			4.  L(-1) = L(-2)
# 7		7		4		7			5.  L(-1) = L(-2)
# 8		8		5		8			6   L(-1) = L(-2)

class BasisFunctionSignature(OrderedDict):
    def get_key(self):
        return self["type"], tuple(self["nr"]), tuple(self["nl"]), tuple(self.get("lint"))


class Global(OrderedDict): pass


def basis_func_rep(dumper, data):
    return dumper.represent_mapping('tag:yaml.org,2002:map', data.items(), flow_style=True)


def global_rep(dumper, data):
    return dumper.represent_mapping('tag:yaml.org,2002:map', data.items(), flow_style=False)


def setup_yaml():
    """ https://stackoverflow.com/a/8661021 """
    represent_dict_order = lambda self, data: self.represent_mapping('tag:yaml.org,2002:map', data.items())
    yaml.add_representer(OrderedDict, represent_dict_order)
    yaml.add_representer(BasisFunctionSignature, basis_func_rep)
    yaml.add_representer(Global, global_rep)


setup_yaml()


def main(filename_in="Al.pbe.in", filename_out="Al.pbe.yaml", DEBUG=False):
    print("Reading ", filename_in)
    with open(filename_in) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines if (len(l.strip()) > 0 and not l.strip().startswith("!"))]

    comments = lines[0:2]
    comments = [c.strip() for c in comments]
    metadata_comments = ";".join(comments)
    metadata_dict = {
        "comment": "Converted from FORTRAN mace file `{}`".format(filename_in),
        "original_comments": metadata_comments
    }
    elements_line = lines[2]
    kw, element = elements_line.split()
    if DEBUG:
        print(kw, "-", element)
    assert kw == "Elements", "Elements is missing" + " at line " + str(i)
    comments.append(lines[3])
    comments.append(lines[4])

    kw, lamb = lines[5].split()
    assert kw == "lambda", "Lambda is missing" + " at line " + str(i)
    lamb = float(lamb)
    if DEBUG:
        print("lamb={lamb}".format(lamb=lamb))
    line_split = lines[6].split()
    # print(line_split)
    if len(line_split) == 4:
        kw1, kw2, rcut, dcut = lines[6].split()
        rcut = float(rcut)
        dcut = float(dcut)
        ecut = None
        decut = None
        assert kw1 == "rcut", "rcut is missing" + " at line " + str(6)
        assert kw2 == "dcut", "dcut is missing" + " at line " + str(6)
    elif len(line_split) == 8:
        kw1, kw2, kw3, kw4, rcut, dcut, ecut, decut = lines[6].split()
        rcut = float(rcut)
        dcut = float(dcut)
        ecut = float(ecut)
        decut = float(decut)
        assert kw1 == "rcut", "rcut is missing" + " at line " + str(6)
        assert kw2 == "dcut", "dcut is missing" + " at line " + str(6)
        assert kw3 == "ecut", "ecut is missing" + " at line " + str(6)
        assert kw4 == "decut", "decut is missing" + " at line " + str(6)
    else:
        raise ValueError("Unrecognized format for cutoffs at line #6:" + str(lines[6]))
    if DEBUG:
        print("rcut={rcut}, dcut={dcut}".format(rcut=rcut, dcut=dcut))

    kw, nradbase = lines[7].split()
    assert kw == "nradbase", "nradbase is missing" + " at line " + str(i)
    nradbase = int(nradbase)
    if DEBUG:
        print("nradbase={nradbase}".format(nradbase=nradbase))

    i = 8
    nradials_dict = {}
    lmax_dict = {}
    while "nradial" in lines[i] and "max" in lines[i]:
        nrad_kw, lmax_kw, nrad_i, lmax_i = lines[i].split()
        nrad_i = int(nrad_i)
        lmax_i = int(lmax_i)
        rank = int(nrad_kw.split("nradial")[-1])
        nradials_dict[rank] = nrad_i
        lmax_dict[rank] = lmax_i
        i += 1

    nradmax = max(nradials_dict.values())
    lmax = max(lmax_dict.values())
    if DEBUG:
        print("nradials_dict=", nradials_dict)
        print("lmax_dict=", lmax_dict)
        print("nradmax=", nradmax)
        print("lmax=", lmax)

    kw, ndensity = lines[i].split()
    assert kw == "ndensity", "ndensity is missing" + " at line " + str(i)
    ndensity = int(ndensity)
    if DEBUG:
        print("ndensity={ndensity}".format(ndensity=ndensity))

    # density function form
    i += 1
    kw0 = "npot-nfemb-radtype"
    kw, npot, nfemb, radtype = lines[i].split()
    assert kw == kw0, kw0 + " is missing" + " at line " + str(i)
    npot = int(npot)
    nfemb = int(nfemb)
    radtype = int(radtype)
    if DEBUG:
        print("npot={}, nfemb={}, radtype={}".format(npot, nfemb, radtype))
    if (npot != 5 or nfemb not in [1, 5] or radtype not in [3, 5]):
        if (not (npot == 7 and ndensity < 3)):
            print("invalid npot-nfemb-radtype:", lines[i])
            print("only 'npot-nfemb-radtype 5 1 3' or 'npot-nfemb-radtype 7 1 3' with ndensity<=2 is valid")
            sys.exit(1)
    # par 1 1 ..
    # Al   1.0000000000000000      F
    i += 1
    density_params = []
    while lines[i].strip().startswith("par"):
        kw, i1, i2 = lines[i].split()
        ele, par, flag = lines[i + 1].split()
        density_params.append(float(par))
        i += 2
    if DEBUG:
        print("density_params=", density_params)

    # core repulsion:
    # core            1           0
    # Al Al   0.0000000000000000      F
    # core            2           0
    # Al Al   0.0000000000000000      F
    core_rep_params = []
    while lines[i].startswith("core"):
        kw, i1, i2 = lines[i].split()
        ele1, ele2, par, flag = lines[i + 1].split()
        core_rep_params.append(float(par))
        i += 2

    if DEBUG:
        print("core_rep_params=", core_rep_params)

    # radial coefficients
    # rad            0           0
    # Al Al           1           1           0  0.90386750010651951      T
    # crad=[]
    crad = np.zeros((nradbase, nradmax, lmax + 1))
    while lines[i].startswith("rad"):
        kw, i1, i2 = lines[i].split()
        # print("l=",lines[i+1])
        ele1, ele2, nradbase_i, nrad_i, l_i, coeff, flag = lines[i + 1].split()
        nradbase_i, nrad_i, l_i = list(map(int, (nradbase_i, nrad_i, l_i)))
        coeff = float(coeff)
        # print("(nradbase_i, nrad_i, l_i)=",(nradbase_i, nrad_i, l_i)," coeff=",coeff)
        # crad.append(list(map(float, (nradbase_i, nrad_i, k_i, coeff))))
        crad[nradbase_i - 1, nrad_i - 1, l_i] = coeff
        i += 2
    # print("crad=",crad.reshape(-1))

    while True:
        # print("l=",lines[i])
        kw, clu_size, dens_ind = lines[i].split()
        clu_size = int(clu_size)
        if (clu_size != 1):
            break
        dens_ind = int(dens_ind)
        i += 2

    nradmax = 0
    basis_functions_dens_dict = defaultdict(list)
    while i < len(lines):
        try:
            # print("cur l=",lines[i])
            kw, clu_size, dens_ind = lines[i].split()
            clu_size = int(clu_size)
            rank = clu_size - 1  # clu_size = rank + 1
            dens_ind = int(dens_ind)
            # print("next l=",lines[i+1])
            args = lines[i + 1].split()
            elements = args[:clu_size]
            ns = args[clu_size:2 * clu_size - 1]
            # print("ns=",ns)
            ls_num = rank if rank >= 3 else rank - 1
            # print("ls_num=",ls_num)
            ls = args[2 * clu_size - 1:2 * clu_size + ls_num - 1]
            if rank == 1:
                ls += [0]
            if rank == 2:  # for Al-Al-Al => nl=[0,0]
                ls.append(ls[-1])  # for rank = 2: l1=l2=l
            # print("ls=",ls)
            LS_num = max(rank - 3, 0)
            LS = args[2 * clu_size + ls_num - 1:2 * clu_size + ls_num + LS_num - 1]
            if rank == 3 or rank == 5:
                LS.append(ls[-1])
            elif rank >= 4:
                LS.append(LS[-1])
            # print("LS=",LS)
            coeff = args[-2]
            coeff = float(coeff)
            ns = list(map(int, ns))
            if clu_size > 2:
                nradmax = max(nradmax, max(ns))
            ls = list(map(int, ls))
            basis_func = BasisFunctionSignature({"type": " ".join(elements), "nr": ns, "nl": ls})
            if len(LS) > 0:
                LS = list(map(int, LS))
                basis_func["lint"] = LS
                key = (len(elements) - 1, basis_func["type"], tuple(basis_func["nr"]), tuple(basis_func["nl"]),
                       tuple(basis_func["lint"]))
            else:
                key = (len(elements) - 1, basis_func["type"], tuple(basis_func["nr"]), tuple(basis_func["nl"]))
            # print("key=",key)
            if key in basis_functions_dens_dict:
                # basis_functions_dens_dict[key]["c"].append(coeff)
                basis_func = basis_functions_dens_dict[key]
                try:
                    basis_func["c"][dens_ind - 1] = coeff
                except KeyError as e:
                    print("Key error: ", e)
                    print("key=", key)
                    print("basis_func=", basis_func)
                    raise
            else:
                basis_func["c"] = [0] * ndensity
                basis_func["c"][dens_ind - 1] = coeff
                # print("key=", key)
                # print("basis_func=", basis_func)
                basis_functions_dens_dict[key] = basis_func
                # print("basis_func_aasigned=", basis_functions_dens_dict[key])

            # basis_functions_dens_dict[dens_ind].append(basis_func)
            if DEBUG:
                print(basis_functions_dens_dict[key])
            i += 2
            # break
        except ValueError as e:
            print("Error while parsing line #{}: {}".format(i, lines[i]))
            raise e
    crad = np.transpose(crad, (1, 2, 0)).tolist()  # n,l,k
    # crad = list(map(float, crad.reshape(-1)))

    basis_func_out = [w for k, w in sorted(basis_functions_dens_dict.items())]
    if npot == 7 and ndensity > 2:
        print("ERROR: npot=7 is supported only with ndensity<=2")
        sys.exit(1)

    species_ord_dict = OrderedDict({"speciesblock": element,
                                    "nradmaxi": nradmax,
                                    "lmaxi": lmax,
                                    "ndensityi": ndensity,
                                    "npoti": npot_dict[(npot, nfemb)],
                                    "parameters": density_params,
                                    "core-repulsion": core_rep_params,
                                    "rcutij": rcut,
                                    "dcutij": dcut,
                                    "NameOfCutoffFunctionij": "cos",
                                    "nradbaseij": nradbase,

                                    "radbase": radbase_dict[radtype],
                                    "radparameters": [lamb],
                                    "radcoefficients": crad,

                                    "nbody": basis_func_out
                                    })
    if ecut is not None:
        species_ord_dict["rho_core_cut"] = ecut
        species_ord_dict["drho_core_cut"] = decut

    out = {"metadata": metadata_dict,
           "global": global_val,
           "species": [species_ord_dict]
           }

    print("Writing to ", filename_out)
    with open(filename_out, "w") as f:
        yaml.dump(out, f)


global_val = Global({"DeltaSplineBins": 0.001})

# npot-nfemb combinations:
npot_dict = {(5, 1): "FinnisSinclair", (7, 1): "FinnisSinclair",
             (5, 5): "FinnisSinclairShiftedScaled", (7, 5): "FinnisSinclairShiftedScaled"}

radbase_dict = {3: "ChebExpCos", 5: "ChebPow"}

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("""Usage:
        {} <ace_fortran_potential> <output.yaml> [--verbose]""".format(sys.argv[0]))
        sys.exit(0)

    if "--verbose" in sys.argv[3:]:
        DEBUG = True
    else:
        DEBUG = False
    main(sys.argv[1], sys.argv[2], DEBUG)
