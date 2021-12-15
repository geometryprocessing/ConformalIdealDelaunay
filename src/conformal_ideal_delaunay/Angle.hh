/*********************************************************************************
*  This file is part of reference implementation of SIGGRAPH Asia 2021 Paper     *
*  `Efficient and Robust Discrete Conformal Equivalence with Boundary`           *
*  v1.0                                                                          *
*                                                                                *
*  The MIT License                                                               *
*                                                                                *
*  Permission is hereby granted, free of charge, to any person obtaining a       *
*  copy of this software and associated documentation files (the "Software"),    *
*  to deal in the Software without restriction, including without limitation     *
*  the rights to use, copy, modify, merge, publish, distribute, sublicense,      *
*  and/or sell copies of the Software, and to permit persons to whom the         *
*  Software is furnished to do so, subject to the following conditions:          *
*                                                                                *
*  The above copyright notice and this permission notice shall be included in    *
*  all copies or substantial portions of the Software.                           *
*                                                                                *
*  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR    *
*  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,      *
*  FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE  *
*  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER        *
*  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING       *
*  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS  *
*  IN THE SOFTWARE.                                                              *
*                                                                                *
*  Author(s):                                                                    *
*  Marcel Campen, Institute for Computer Science, Osnabrück University, Germany. *
*  Ryan Capouellez, Hanxiao Shen, Leyi Zhu, Daniele Panozzo, Denis Zorin,        *
*  Courant Institute of Mathematical Sciences, New York University, USA          *
*                                          *                                     *
*********************************************************************************/

/** @file Angle.hh
 *  @brief Angle computation based on edge length.
 *  This contains a function for computing interior angle given edge lengths of a triangle.
 */
#ifndef ANGLE_H
#define ANGLE_H

#include <cmath>

#ifdef WITH_MPFR
#include <unsupported/Eigen/MPRealSupport>
#endif

/**
 * Triangle interior angle computation function that takes edge length as iput and produce angle in radian.
 * When the input lengths do not satisify triangle inequality function returns M_PI when l12 + l23 <= l31 or 0 otherwise.
 * 
 * @param l12 edge length between corner 1 and corner 2
 * @param l23 edge length between corner 2 and corner 3
 * @param l31 edge length between corner 3 and corner 1
 * @return Scalar interior angle at 2 in radian
 */

template <typename Scalar>
Scalar angle(Scalar l12, Scalar l23, Scalar l31){
  const Scalar t31 = +l12+l23-l31,
               t23 = +l12-l23+l31,
               t12 = -l12+l23+l31;
  // valid triangle
  if( t31 > 0 && t23 > 0 && t12 > 0 ){
    const Scalar l123 = l12+l23+l31;
    const Scalar denom = sqrt(t12*t23*t31*l123);
    return 2*atan2(t23*t12,denom); // angle at corner2, opposite to edge l31
  }else if( t31 <= 0 ) 
    return M_PI;
  else
    return 0;
}

template double angle<double>(double, double, double);
#ifdef WITH_MPFR
template mpfr::mpreal angle<mpfr::mpreal>(mpfr::mpreal, mpfr::mpreal, mpfr::mpreal);
#endif

#endif