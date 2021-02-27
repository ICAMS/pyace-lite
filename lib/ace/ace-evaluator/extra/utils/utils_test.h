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
// Created by Yury Lysogorskiy on 22.02.20.
//

#ifndef ACE_UTILS_TEST_H
#define ACE_UTILS_TEST_H

#include <cmath>
#include "ace_types.h"
#include "ace_atoms.h"
#include "ace_evaluator.h"
#include "ace_utils.h"
#include "ace_calculator.h"


void print_input_structure_for_fortran(ACEAtomicEnvironment &atomic_environment);

void check_sum_of_forces(ACEAtomicEnvironment &ae, ACECalculator &aceCalculator, DOUBLE_TYPE threshold = 1e-10);

void check_cube_diagonal_forces_symmetry(ACEAtomicEnvironment &ae, ACECalculator &ace);

void compare_forces(DOUBLE_TYPE analytic_force, DOUBLE_TYPE numeric_force, DOUBLE_TYPE rel_threshold);

void check_numeric_force(ACEAtomicEnvironment &ae, ACECalculator &ace, DOUBLE_TYPE rel_threshold = 1e-5,
                         DOUBLE_TYPE dr = 1e-8,
                         int atom_ind_freq = 10);

#endif //ACE_UTILS_TEST_H
