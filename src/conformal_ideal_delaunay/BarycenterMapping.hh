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
#ifndef BC_MAPPING_HH
#define BC_MAPPING_HH
#include <igl/barycentric_coordinates.h>
#include <vector>

template <typename Scalar>
struct Pt
{
    int f_id;
    Eigen::Matrix<Scalar, 1, 3> bc;
};

template <typename Scalar>
Scalar compute_tan_half(Scalar a, Scalar b, Scalar c);

template <typename Scalar, typename Scalar_pt>
void recompute_bc_hyper(int _h,
                        const std::vector<int> &n,
                        const std::vector<int> &h,
                        const std::vector<int> &f,
                        const std::vector<int> &opp,
                        const std::vector<Scalar> &l,
                        std::vector<Pt<Scalar_pt>> &pts,
                        std::vector<std::vector<int>> &pt_in_f);

template <typename Scalar, typename Scalar_pt>
void original_to_equilateral(
    std::vector<Pt<Scalar_pt>> &pts,
    std::vector<std::vector<int>> &pt_in_f,
    const std::vector<int> &n,
    const std::vector<int> &h,
    const std::vector<Scalar> &l);

template <typename Scalar, typename Scalar_pt>
void equilateral_to_scaled(
    std::vector<Pt<Scalar_pt>> &pts,
    std::vector<std::vector<int>> &pt_in_f,
    const std::vector<int> &n,
    const std::vector<int> &h,
    const std::vector<int> &to,
    const std::vector<Scalar> &l,
    const Eigen::Matrix<Scalar, -1, 1> &u);

template <typename Scalar, typename Scalar_pt>
void recompute_bc_original(int _h,
                        const std::vector<int> &n,
                        const std::vector<int> &h,
                        const std::vector<int> &f,
                        const std::vector<int> &opp,
                        const std::vector<Scalar> &l,
                        std::vector<Pt<Scalar_pt>> &pts,
                        std::vector<std::vector<int>> &pt_in_f);
#endif
