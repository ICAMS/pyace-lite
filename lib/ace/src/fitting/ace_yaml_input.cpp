#include "ace_yaml_input.h"

#include <algorithm>
#include <iostream>
#include <set>
#include <string>

#include "yaml-cpp/yaml.h"

#include "ace_couplings.h"

using namespace std;


/**
Split a given string by tabs or space

@param mainkey, string - input string

@returns splitted, string - the string after splitting
*/
vector<string> Input::split_key(string mainkey) {

    vector<string> splitted;
    istringstream ins(mainkey);

    for (string mainkey; ins >> mainkey;)
        splitted.emplace_back(mainkey);

    return splitted;
}


/**
Main function to parse the yaml file and read the input data
*/
void Input::parse_input(const string &ff) {
    //set the input file - first thing to do
    inputfile = ff;
    if (!if_file_exist(inputfile)) {
        stringstream s;
        s << "Potential file " << inputfile << " doesn't exists";
        cerr << "Exception: " << s.str();
        throw invalid_argument(s.str());
    }

    //load the file with yaml
    YAML::Node YAML_input = YAML::LoadFile(inputfile);
    //all the raw data is now available in rawinput

    //first step - parse the global data
    global.DeltaSplineBins = YAML_input["global"]["DeltaSplineBins"].as<DOUBLE_TYPE>();
    auto YAML_input_species = YAML_input["species"];
    //now get the number of species blocks
    number_of_species_block = static_cast<unsigned short>(YAML_input_species.size());
    global.nblocks = number_of_species_block;

    if (YAML_input["metadata"]) {
        global.metadata = YAML_input["metadata"].as<map<string, string>>();
    }

    //find the names of all species block, and use that to count
    //single elements, pairs and so on
    vector<vector<string>> all_split_species;
    vector<string> _splitted;

    global.nelements = 0;
    set<string> elements_set;
    //loop over each block and find elements.
    for (unsigned short int i = 0; i < number_of_species_block; i++) {
        string tosplit = YAML_input_species[i]["speciesblock"].as<string>();
        _splitted = split_key(tosplit);
//        _splitted  = YAML_input_species[i]["speciesblock"].as<vector<string>>();
        for (const auto &val: _splitted) {
            if (elements_set.count(val) == 0)
                elements_set.insert(val);
        }
        //add it to main vector
        all_split_species.emplace_back(_splitted);
    }

    //now check
    global.nelements = elements_set.size();
    std::copy(elements_set.begin(), elements_set.end(), std::back_inserter(global.element_names));
    std::sort(global.element_names.begin(), global.element_names.end());
#ifdef DEBUG_READ_YAML
    for (auto element: global.element_names) {
        cout << element << endl;
    }
#endif


    for (int species_block_ind = 0; species_block_ind < number_of_species_block; species_block_ind++) {
        //create temp block
        //vector<SpeciesBlock> temp_species;
        auto YAML_input_species_block = YAML_input_species[species_block_ind];
        string block_name = YAML_input_species_block["speciesblock"].as<string>();
#ifdef DEBUG_READ_YAML
        cout << block_name << endl;
#endif
        //create a species_block
        BBasisFunctionsSpecificationBlock b_basisfunc_spec_block = BBasisFunctionsSpecificationBlock();
        b_basisfunc_spec_block.block_name = block_name;
        b_basisfunc_spec_block.elements_vec = split_key(block_name);
        b_basisfunc_spec_block.mu0 = b_basisfunc_spec_block.elements_vec[0];

        // ! if only one element in species, then duplicate it
        if (b_basisfunc_spec_block.elements_vec.size() == 1)
            b_basisfunc_spec_block.elements_vec.emplace_back(b_basisfunc_spec_block.elements_vec[0]);

        b_basisfunc_spec_block.number_of_species = b_basisfunc_spec_block.elements_vec.size();


        //push this species_block to a list of blocks
        //input values that exist only for individual elements
        b_basisfunc_spec_block.nradmaxi = YAML_input_species_block["nradmaxi"].as<NS_TYPE>();
        b_basisfunc_spec_block.lmaxi = YAML_input_species_block["lmaxi"].as<LS_TYPE>();
        b_basisfunc_spec_block.npoti = YAML_input_species_block["npoti"].as<string>();
        if (YAML_input_species_block["ndensityi"]) {
            b_basisfunc_spec_block.ndensityi = YAML_input_species_block["ndensityi"].as<DENSITY_TYPE>();
        }
        if (YAML_input_species_block["parameters"]) {
            b_basisfunc_spec_block.fs_parameters = YAML_input_species_block["parameters"].as<vector<DOUBLE_TYPE>>();
        }

        //hard-core repulsion parameters
        if (YAML_input_species_block["core-repulsion"]) {
            b_basisfunc_spec_block.core_rep_parameters = YAML_input_species_block["core-repulsion"].as<vector<DOUBLE_TYPE>>();
        } else {
            b_basisfunc_spec_block.core_rep_parameters.resize(2, 0);
        }


        //energy cutoff parameters
        if (YAML_input_species_block["rho_core_cut"]) {
            b_basisfunc_spec_block.rho_cut = YAML_input_species_block["rho_core_cut"].as<DOUBLE_TYPE>();
        } else {
            b_basisfunc_spec_block.rho_cut = 100000.0;
        }

        if (YAML_input_species_block["drho_core_cut"]) {
            b_basisfunc_spec_block.drho_cut = YAML_input_species_block["drho_core_cut"].as<DOUBLE_TYPE>();
        } else {
            b_basisfunc_spec_block.drho_cut = 250.0;
        }

        if (b_basisfunc_spec_block.number_of_species <= 2) {
            b_basisfunc_spec_block.rcutij = YAML_input_species_block["rcutij"].as<DOUBLE_TYPE>();
            b_basisfunc_spec_block.dcutij = YAML_input_species_block["dcutij"].as<DOUBLE_TYPE>();
            b_basisfunc_spec_block.NameOfCutoffFunctionij = YAML_input_species_block["NameOfCutoffFunctionij"].as<string>();
            b_basisfunc_spec_block.nradbaseij = YAML_input_species_block["nradbaseij"].as<NS_TYPE>();

            b_basisfunc_spec_block.radbase = YAML_input_species_block["radbase"].as<string>();
            b_basisfunc_spec_block.radparameters = YAML_input_species_block["radparameters"].as<vector<DOUBLE_TYPE>>();
            try {
                b_basisfunc_spec_block.radcoefficients = YAML_input_species_block["radcoefficients"].as<vector<vector<vector<DOUBLE_TYPE>>>>();
            } catch (YAML::RepresentationException &exc) {
                cout
                        << "DEPRECATION WARNING!!! Old (flatten) radcoefficients parameter encounterd, whereas it should be three-dimensional with [nradmax][lmax+1][nradbase] shape."
                        << endl;
                cout << "Automatic reshaping will be done" << endl;
                auto radcoefficients = YAML_input_species_block["radcoefficients"].as<vector<DOUBLE_TYPE>>();
                size_t j = 0;

                //initialize array
                b_basisfunc_spec_block.radcoefficients = vector<vector<vector<DOUBLE_TYPE>>>(
                        b_basisfunc_spec_block.nradmaxi,
                        vector<vector<DOUBLE_TYPE>>(b_basisfunc_spec_block.lmaxi + 1,
                                                    vector<DOUBLE_TYPE>(b_basisfunc_spec_block.nradbaseij)
                        )
                );

                for (NS_TYPE k = 0; k < b_basisfunc_spec_block.nradbaseij; k++) {
                    for (NS_TYPE n = 0; n < b_basisfunc_spec_block.nradmaxi; n++) {
                        for (LS_TYPE l = 0; l <= b_basisfunc_spec_block.lmaxi; l++, j++) {
                            b_basisfunc_spec_block.radcoefficients.at(n).at(l).at(k) = radcoefficients.at(j);
                        }
                    }
                }

            }


            if (b_basisfunc_spec_block.radcoefficients.size() != b_basisfunc_spec_block.nradmaxi) {
                stringstream s;
                s << "Input file error: species block " << b_basisfunc_spec_block.block_name
                  << " has insufficient radcoefficients.shape(0)=("
                  << b_basisfunc_spec_block.radcoefficients.size()
                  << ") whereas it should be nradbaseij (" << b_basisfunc_spec_block.nradbaseij << ")";
                cerr << "Exception: " << s.str();
                throw std::invalid_argument(s.str());
            } else {
                for (NS_TYPE n = 0; n < b_basisfunc_spec_block.nradmaxi; n++)
                    if (b_basisfunc_spec_block.radcoefficients.at(n).size() != b_basisfunc_spec_block.lmaxi + 1) {
                        stringstream s;
                        s << "Input file error: species block " << b_basisfunc_spec_block.block_name
                          << " has insufficient radcoefficients[" << n + 1 << "].size=("
                          << b_basisfunc_spec_block.radcoefficients.at(n).size()
                          << ") whereas it should be lmaxi+1 = (" << b_basisfunc_spec_block.lmaxi + 1 << ")";
                        cerr << "Exception: " << s.str();
                        throw std::invalid_argument(s.str());
                    } else {
                        for (LS_TYPE l = 0; l <= b_basisfunc_spec_block.lmaxi; l++)
                            if (b_basisfunc_spec_block.radcoefficients.at(n).at(l).size() !=
                                b_basisfunc_spec_block.nradbaseij) {
                                stringstream s;
                                s << "Input file error: species block " << b_basisfunc_spec_block.block_name
                                  << " has insufficient radcoefficients[" << n + 1 << "][" << l << "].size=("
                                  << b_basisfunc_spec_block.radcoefficients.at(n).at(l).size()
                                  << ") whereas it should be nradbase = (" << b_basisfunc_spec_block.nradbaseij << ")";
                                cerr << "Exception: " << s.str();
                                throw std::invalid_argument(s.str());
                            }
                    }


            }

        }
        //SHORT_INT_TYPE num_of_densities = raw_input_species_block["density"].size();
        SHORT_INT_TYPE num_of_densities = b_basisfunc_spec_block.ndensityi;

        //time to read in nbody terms
        //species_block.rank_bfuncspec_vector.resize(num_of_densities);

        NS_TYPE actual_nradmax = 0;
        LS_TYPE actual_lmaxi = 0;
        LS_TYPE actual_LSmaxi = 0;
        //now loop over densities


        RANK_TYPE rankmax = 0;


        //this is just the number of entries - it can have multiple terms of rank 1 and so on
        auto num_of_basis_functions = static_cast<SHORT_INT_TYPE>(YAML_input_species_block["nbody"].size());
        if (num_of_basis_functions == 0) {
            stringstream s;
            s << "Potential yaml file '" << inputfile << "' has no <nbody> section in <speciesblock>";
            if (YAML_input_species_block["density"].size() > 0)
                s << "Section <density> is presented. It seems that this is old file format.";
            cerr << "Exception: " << s.str();
            throw std::invalid_argument(s.str());
//            exit(EXIT_FAILURE);
        }


        //this is a vector to store the nbodys temporarily
        vector<BBasisFunctionSpecification> temp_bbasisfunc_spec_vector;
        for (auto YAML_input_current_basisfunc_spec:  YAML_input_species_block["nbody"]) {
            BBasisFunctionSpecification bBasisFunctionSpec = BBasisFunctionSpecification();

            string basisfunc_species_str = YAML_input_current_basisfunc_spec["type"].as<string>();
            vector<string> curr_basisfunc_species_vec = split_key(basisfunc_species_str);
            //this is the rank
            RANK_TYPE rank = curr_basisfunc_species_vec.size() - 1;
            //if rank is greater than maxrank, increment maxrank
            if (rank > rankmax) rankmax = rank;

            bBasisFunctionSpec.rank = rank;
            bBasisFunctionSpec.elements = curr_basisfunc_species_vec;

            if (rank > 0) {
                bBasisFunctionSpec.ns = YAML_input_current_basisfunc_spec["nr"].as<vector<NS_TYPE>>();
                bBasisFunctionSpec.ls = YAML_input_current_basisfunc_spec["nl"].as<vector<LS_TYPE>>();
                if (YAML_input_current_basisfunc_spec["c"].Type() == YAML::NodeType::Sequence)
                    bBasisFunctionSpec.coeffs = YAML_input_current_basisfunc_spec["c"].as<vector<DOUBLE_TYPE>>();
                else if (YAML_input_current_basisfunc_spec["c"].Type() == YAML::NodeType::Scalar) {
                    vector<DOUBLE_TYPE> c_vec(1);
                    c_vec[0] = YAML_input_current_basisfunc_spec["c"].as<DOUBLE_TYPE>();
                    bBasisFunctionSpec.coeffs = c_vec;
                }

                if (bBasisFunctionSpec.coeffs.size() != num_of_densities) {
                    stringstream s;
                    s << " Number of coefficients in " << YAML_input_current_basisfunc_spec << " (" <<
                      bBasisFunctionSpec.coeffs.size() << "), does not match specified num_of_densities=" <<
                      num_of_densities;
                    cerr << "Exception: " << s.str();
                    throw invalid_argument(s.str());
//                    exit(EXIT_FAILURE);
                }

                NS_TYPE current_nradmax = *max_element(bBasisFunctionSpec.ns.begin(), bBasisFunctionSpec.ns.end());

                if (rank > 1) {
                    if (current_nradmax > actual_nradmax)
                        actual_nradmax = current_nradmax;

                    if (b_basisfunc_spec_block.nradmaxi < actual_nradmax) {
                        stringstream s;
                        s << "Given nradmaxi = " << b_basisfunc_spec_block.nradmaxi << " is less than the max(nr) = "
                          << actual_nradmax;
                        cerr << "Exception: " << s.str();
                        throw invalid_argument(s.str());
//                        exit(EXIT_FAILURE);
                    }
                } else {//rank==1
                    if (b_basisfunc_spec_block.nradbaseij < current_nradmax) {
                        stringstream s;
                        s << "Given nradbaseij = " << b_basisfunc_spec_block.nradbaseij <<
                          " is less than the max(nr) for rank=1, which is actual_nradmax = "
                          << actual_nradmax;
                        cerr << "Exception: " << s.str();
                        throw invalid_argument(s.str());
//                        exit(EXIT_FAILURE);
                    }
                }

                LS_TYPE current_lmaxi = *max_element(bBasisFunctionSpec.ls.begin(), bBasisFunctionSpec.ls.end());
                if (current_lmaxi > actual_lmaxi)
                    actual_lmaxi = current_lmaxi;
                if (b_basisfunc_spec_block.lmaxi < actual_lmaxi) {
                    stringstream s;
                    s << "Given lmaxi = " << b_basisfunc_spec_block.lmaxi << " is less than the max(nl) = "
                      << actual_lmaxi;
                    cerr << "Exception: " << s.str();
                    throw invalid_argument(s.str());
//                    exit(EXIT_FAILURE);
                }
            }

            if (rank > 2) {
                bBasisFunctionSpec.LS = YAML_input_current_basisfunc_spec["lint"].as<vector<LS_TYPE>>();
                LS_TYPE current_LSmax = *max_element(bBasisFunctionSpec.LS.begin(), bBasisFunctionSpec.LS.end());
                if (current_LSmax > actual_LSmaxi)
                    actual_LSmaxi = current_LSmax;
                if (2 * b_basisfunc_spec_block.lmaxi < actual_LSmaxi) {
                    stringstream s;
                    s << "Given 2*lmaxi = " << 2 * b_basisfunc_spec_block.lmaxi << " is less than the max(lint) = "
                      << actual_LSmaxi;
                    cerr << "Exception: " << s.str();
                    throw invalid_argument(s.str());
//                    exit(EXIT_FAILURE);
                }
            }

            //validation according to ls-LS relations

            try {
                validate_ls_LS(bBasisFunctionSpec.ls, bBasisFunctionSpec.LS);
            } catch (invalid_argument e) {
                cerr << "Exception: " << e.what();
                throw e;
            }

            temp_bbasisfunc_spec_vector.emplace_back(bBasisFunctionSpec);
        }//end loop over basis functions {}

        //now maxrank would be update
        b_basisfunc_spec_block.rankmax = rankmax;
        //now give maxrank
        //species_block.rank_bfuncspec_vector.resize(rankmax);

        //now loop through temp_nbodies_vector and assign based on rank
        for (auto &b_basisfunc_spec : temp_bbasisfunc_spec_vector) {
            b_basisfunc_spec_block.funcspecs.emplace_back(b_basisfunc_spec);
        }
        //always start from 2-species blocks, etc.
        //species_blocks_vector[species_block.number_of_species - 2].emplace_back(species_block);
        bbasis_func_spec_blocks_vector.emplace_back(b_basisfunc_spec_block);

    }


    global.lmax = 0;
    global.nradmax = 0;
    global.nradbase = 0;
    global.rankmax = 0;
    global.ndensitymax = 0;
    global.cutoffmax = 0;

    for (const auto &pair_block: bbasis_func_spec_blocks_vector) {
        if (pair_block.number_of_species != 2) continue;
        if (pair_block.lmaxi > global.lmax) global.lmax = pair_block.lmaxi;
        if (pair_block.nradmaxi > global.nradmax) global.nradmax = pair_block.nradmaxi;
        if (pair_block.nradbaseij > global.nradbase) global.nradbase = pair_block.nradbaseij;
        if (pair_block.rankmax > global.rankmax) global.rankmax = pair_block.rankmax;
        if (pair_block.ndensityi > global.ndensitymax) global.ndensitymax = pair_block.ndensityi;
        if (pair_block.rcutij > global.cutoffmax) global.cutoffmax = pair_block.rcutij;
    }

}


