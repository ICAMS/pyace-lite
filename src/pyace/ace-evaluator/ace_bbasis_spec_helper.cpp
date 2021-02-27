/*
 * pyace: atomic cluster expansion and its python bindings
 *
 * Copyright 2021  (c) Yury Lysogorskiy, Sarath Menon,
 * Anton Bochkarev, Ralf Drautz
 *
 * Ruhr-University Bochum, Bochum, Germany
 *
 * See the LICENSE file.
 * This FILENAME is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

//
// Created by Lysogorskiy Yury on 11.05.2020.
//
#include "ace_bbasis_spec_helper.h"

string radcoefficients_to_str(vector<vector<vector<DOUBLE_TYPE>>> radcoefficients) {

    vector<string> string_vec2;
    for (const auto &c_lk: radcoefficients) {
        vector<string> string_vec;
        for (const auto &c_k: c_lk) {
            string_vec.emplace_back("[" + join(c_k, ", ") + "]");
        }
        string_vec2.emplace_back("[" + join(string_vec, ", ") + "]");
    }
    return "[" + join(string_vec2, ", ") + "]";
}


string BBasisFunctionsSpecificationBlock_repr_(const BBasisFunctionsSpecificationBlock &block) {
    stringstream s;

    s << "BBasisFunctionsSpecificationBlock(name='" << block.block_name << "', " \
 << "nradmaxi=" << block.nradmaxi << ", "\
 << "lmaxi=" << block.lmaxi << ", "\
 << "npoti=" << block.npoti << ", "\
 << "fs_parameters=[" << join(block.fs_parameters, ", ") << "],  "\
 << "rcutij=" << block.rcutij << ", "\
 << "dcutij=" << block.dcutij << ", "\
 << "NameOfCutoffFunctionij=" << block.NameOfCutoffFunctionij << ", "\
 << "nradbaseij=" << block.nradbaseij << ", "\
 << "radbase=" << block.radbase << ", "\
 << "radparameters=[" << join(block.radparameters, ", ") << "], "\
 << "radcoefficients=[" << radcoefficients_to_str(block.radcoefficients);


    if (!block.funcspecs.empty())
        s << ", bfuncspec=[" << block.funcspecs.size() << " funcs ]";
    s << ")";
    return s.str();
}

string BBasisConfiguration_repr(BBasisConfiguration &config) {
    stringstream s;

    s << "BBasisConfiguration(" \
 << "deltaSplineBins=" << config.deltaSplineBins;

    if (!config.funcspecs_blocks.empty())
        s << ", bbasis_func_spec_blocks=[" << config.funcspecs_blocks.size() << " blocks ]";
    s << ")";
    return s.str();
}


py::tuple BBasisFunctionSpecification_getstate(const BBasisFunctionSpecification &spec) {
    return py::make_tuple(
            spec.elements, //0
            spec.ns, //1
            spec.ls, //2
            spec.LS,//3
            spec.coeffs //4
    );
}

BBasisFunctionSpecification BBasisFunctionSpecification_setstate(const py::tuple &tuple) {
    if (tuple.size() != 5)
        throw std::runtime_error(
                "Invalid state of BBasisFunctionSpecification-tuple, probably format has been changed");
    BBasisFunctionSpecification spec(
            tuple[0].cast<vector<string>>(),
            tuple[1].cast<vector<NS_TYPE>>(),
            tuple[2].cast<vector<LS_TYPE>>(),
            tuple[3].cast<vector<LS_TYPE>>(),
            tuple[4].cast<vector<DOUBLE_TYPE>>()
    );
    return spec;
}


py::tuple BBasisFunctionsSpecificationBlock_getstate(const BBasisFunctionsSpecificationBlock &block) {
    return py::make_tuple(
            block.block_name,           //0
            block.number_of_species,    //1
            block.mu0,                  //2
            block.elements_vec,         //3

            block.rankmax,              //4

            block.nradmaxi,             //5
            block.lmaxi,                //6
            block.nradbaseij,           //7
            block.radbase,              //8
            block.radparameters,        //9
            block.radcoefficients,      //10

            block.rcutij,               //11
            block.dcutij,               //12

            block.ndensityi,            //13
            block.npoti,                //14
            block.fs_parameters,        //15

            block.core_rep_parameters,  //16
            block.rho_cut,              //17
            block.drho_cut,             //18

            block.NameOfCutoffFunctionij,//19

            block.funcspecs             //20
    );
}

BBasisFunctionsSpecificationBlock BBasisFunctionsSpecificationBlock_setstate(const py::tuple &tuple) {
    if (tuple.size() != 21)
        throw std::runtime_error(
                "Invalid state of BBasisFunctionsSpecificationBlock-tuple, probalby format has been changed");
    BBasisFunctionsSpecificationBlock block;
    block.block_name = tuple[0].cast<string>();           //0
    block.number_of_species = tuple[1].cast<SPECIES_TYPE>();    //1
    block.mu0 = tuple[2].cast<string>();                  //2
    block.elements_vec = tuple[3].cast<vector<string>>();         //3

    block.rankmax = tuple[4].cast<RANK_TYPE>();              //4

    block.nradmaxi = tuple[5].cast<NS_TYPE>();              //5
    block.lmaxi = tuple[6].cast<LS_TYPE>();                //6
    block.nradbaseij = tuple[7].cast<NS_TYPE>();           //7
    block.radbase = tuple[8].cast<string>();              //8
    block.radparameters = tuple[9].cast<vector<DOUBLE_TYPE>>();        //9
    block.radcoefficients = tuple[10].cast<vector<vector<vector<DOUBLE_TYPE>>>>();      //10

    block.rcutij = tuple[11].cast<DOUBLE_TYPE>();               //11
    block.dcutij = tuple[12].cast<DOUBLE_TYPE>();               //12

    block.ndensityi = tuple[13].cast<DENSITY_TYPE>();            //13
    block.npoti = tuple[14].cast<string>();                //14
    block.fs_parameters = tuple[15].cast<vector<DOUBLE_TYPE>>();        //15

    block.core_rep_parameters = tuple[16].cast<vector<DOUBLE_TYPE>>();  //16
    block.rho_cut = tuple[17].cast<DOUBLE_TYPE>();              //17
    block.drho_cut = tuple[18].cast<DOUBLE_TYPE>();             //18

    block.NameOfCutoffFunctionij = tuple[19].cast<string>();//19

    block.funcspecs = tuple[20].cast<vector<BBasisFunctionSpecification>>();             //20

    return block;
}


py::tuple BBasisConfiguration_getstate(const BBasisConfiguration &config) {
    return py::make_tuple(
            config.deltaSplineBins,           //0
            config.funcspecs_blocks,          //1
            config.metadata                  //2
    );
}

BBasisConfiguration BBasisConfiguration_setstate(const py::tuple &tuple) {
    if (tuple.size() != 3)
        throw std::runtime_error(
                "Invalid state of BBasisFunctionsSpecificationBlock-tuple, probalby format has been changed");
    BBasisConfiguration config;
    config.deltaSplineBins = tuple[0].cast<DOUBLE_TYPE>();           //0
    config.funcspecs_blocks = tuple[1].cast<vector<BBasisFunctionsSpecificationBlock>>();    //1
    config.metadata = tuple[2].cast<map<string, string>>();                  //2
    return config;
}


// ACEBBasisSet pickling
py::tuple ACEBBasisSet_getstate(const ACEBBasisSet &bbasisSet) {
    auto bbasis_config = bbasisSet.to_BBasisConfiguration();
    auto tuple = BBasisConfiguration_getstate(bbasis_config);
    return tuple;
}

ACEBBasisSet ACEBBasisSet_setstate(const py::tuple &tuple) {
    BBasisConfiguration bbasis_config = BBasisConfiguration_setstate(tuple);
    ACEBBasisSet bbasis_set = ACEBBasisSet(bbasis_config);
    return bbasis_set;
}

