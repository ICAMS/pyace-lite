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

#ifndef PYACE_ACE_RADIAL_HELPER_H
#define PYACE_ACE_RADIAL_HELPER_H

#include <pybind11/pytypes.h>

#include "ace_radial.h"

pybind11::tuple ACERadialFunctions_getstate(const AbstractRadialBasis *radial_functions);

ACERadialFunctions *ACERadialFunctions_setstate(const pybind11::tuple &t);

#endif //PYACE_ACE_RADIAL_HELPER_H
