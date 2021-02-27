/*
 * Atomic cluster expansion
 *
 * Copyright 2021  (c) Yury Lysogorskiy, Sarath Menon,
 * Matteo Rinaldi, Thomas Hammerschmidt, Matous Mrovec,
 * Ralf Drautz
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

#ifndef ACE_SPHERICAL_POLAR_H
#define ACE_SPHERICAL_POLAR_H

#include <cmath>

#include "ace_arraynd.h"
#include "ace_array2dlm.h"
#include "ace_complex.h"
#include "ace_types.h"


using namespace std;

/**
Class to store spherical harmonics and their associated functions. \n
All the associated members such as \f$ P_{lm}, Y_{lm}\f$ etc are one dimensional arrays of length (L+1)*(L+2)/2. \n
The value that corresponds to a particular l, m configuration can be accessed through a preprocessor directive as \n
\code ylm[at(l,m)] \endcode \n
which can access the (m+(l*(l+1))/2) value from the one dimensional array.
*/
class ACESHarmonics {
public:

    constexpr static const DOUBLE_TYPE EPS = 1e-6;
    /**
    int, the number of spherical harmonics to be found
    */
    LS_TYPE lmax;

    ACESHarmonics() = default;

    explicit ACESHarmonics(LS_TYPE lmax);

    void init(LS_TYPE lm);

    ~ACESHarmonics();

    void pre_compute();

    void compute_plm(DOUBLE_TYPE costheta, DOUBLE_TYPE sintheta);

    void compute_plm_2(DOUBLE_TYPE costheta, DOUBLE_TYPE sintheta);

    void compute_ylm(DOUBLE_TYPE costheta, DOUBLE_TYPE sintheta, DOUBLE_TYPE cosphi, DOUBLE_TYPE sinphi, LS_TYPE lmaxi);

    /**
    Array to store the precomputed prefactor values. \f$ a_{lm} \f$ is defined by, \n
    \f$ a_{lm} = \sqrt{\frac{4l^2-1}{l^2-m^2}} \f$.
    See @link https://arxiv.org/pdf/1410.1748.pdf @endlink for more information. \n
    */
    Array2DLM<DOUBLE_TYPE> alm;
    /**
    Array to store the precomputed prefactor values. \f$ b_{lm} \f$ is defined by, \n
    \f$ a_{lm} = \sqrt{\frac{(l-1)^2-m^2}{4(l-1)^2-1}} \f$.
    See @link https://arxiv.org/pdf/1410.1748.pdf @endlink for more information. \n
    */
    Array2DLM<DOUBLE_TYPE> blm;
    Array2DLM<DOUBLE_TYPE> clm;
    //double* dl;
    Array1D<DOUBLE_TYPE> dl;
    //double* el;
    Array1D<DOUBLE_TYPE> el;
    /**
    Array to store \f$ P_{lm} \f$ values.
    */
    Array2DLM<DOUBLE_TYPE> plm;
    Array2DLM<DOUBLE_TYPE> splm;
    Array2DLM<DOUBLE_TYPE> dplm;
    /**
    Array to store \f$ Y_{lm} \f$ values. Each component of this is a complex number
    which is stored using the ACEComplex structure.
    */

    Array2DLM<ACEComplex> ylm;
    Array2DLM<ACEDYcomponent> dylm;

};


#endif
