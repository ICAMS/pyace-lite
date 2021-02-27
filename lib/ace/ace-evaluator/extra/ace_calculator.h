/*
 * Performant implementation of atomic cluster expansion and interface to LAMMPS
 *
 * Copyright 2021  (c) Yury Lysogorskiy^1, Cas van der Oord^2, Anton Bochkarev^1,
 * Sarath Menon^1, Matteo Rinaldi^1, Thomas Hammerschmidt^1, Matous Mrovec^1,
 * Aidan Thompson^3, Gabor Csanyi^2, Christoph Ortner^4, Ralf Drautz^1
 *
 * ^1: Ruhr-University Bochum, Bochum, Germany
 * ^2: University of Cambridge, Cambridge, United Kingdom
 * ^3: Sandia National Laboratories, Albuquerque, New Mexico, USA
 * ^4: University of British Columbia, Vancouver, BC, Canada
 *
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
// Created by Yury Lysogorskiy on 13.03.2020.
//

#ifndef ACE_CALCULATOR_H
#define ACE_CALCULATOR_H

#include "ace_evaluator.h"
#include "ace_atoms.h"

class ACECalculator {
    ACEEvaluator *evaluator = nullptr;
public:
    //total energy of ACEAtomicEnvironment
    DOUBLE_TYPE energy = 0;
    //total forces array
    //forces(i,3), i = 0..num_of_atoms-1
    Array2D<DOUBLE_TYPE> forces = Array2D<DOUBLE_TYPE>("forces");

    //stresses
    Array1D<DOUBLE_TYPE> virial = Array1D<DOUBLE_TYPE>(6, "virial");

    //Per-atom energies
    //energies(i), i = 0..num_of_atoms-1
    Array1D<DOUBLE_TYPE> energies = Array1D<DOUBLE_TYPE>("energies");

    ACECalculator() = default;

    ACECalculator(ACEEvaluator &aceEvaluator) {
        set_evaluator(aceEvaluator);
    }
    void set_evaluator(ACEEvaluator &aceEvaluator);

    //compute the energies and forces for each atoms in atomic_environment
    //results are stored in forces and energies arrays
    void compute(ACEAtomicEnvironment &atomic_environment, bool verbose = false);
#ifdef EXTRA_C_PROJECTIONS
    vector<vector<vector<DOUBLE_TYPE>>> basis_peratom_projections_rank1;
    vector<vector<vector<DOUBLE_TYPE>>> basis_peratom_projections;
#endif
};


#endif //ACE_CALCULATOR_H
