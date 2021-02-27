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
// Created by Yury Lysogorskiy on 27.02.20.
//

#ifndef ACE_UTILS_H
#define ACE_UTILS_H
#include <cmath>
#include <string>
#include <sstream>

#include "ace_types.h"

using namespace std;

inline int sign(DOUBLE_TYPE x) {
    if (x < 0) return -1;
    else if (x > 0) return +1;
    else return 0;
}

inline double absolute_relative_error(double x, double y, double zero_threshold = 5e-6) {
    if (x == 0 && y == 0) return 0;
    else if (x == 0 || y == 0) return (abs(x + y) < zero_threshold ? 0 : 2);
    else return 2 * abs(x - y) / (abs(x) + abs(y));
}

//https://stackoverflow.com/questions/9277906/stdvector-to-string-with-custom-delimiter
template <typename T>
string join(const T& v, const string& delim) {
    stringstream s;
    for (const auto& i : v) {
        if (&i != &v[0]) {
            s << delim;
        }
        s << i;
    }
    return s.str();
}


#endif //ACE_UTILS_H
