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
/** @file plot.h
 *  @brief Libigl viewer extention functions for better visualizing spheres and line segments.
 */

#ifndef VIS_H
#define VIS_H

#include <Eigen/Core>
#include <igl/opengl/glfw/Viewer.h>
#include <set>

/**
 * Given the viewer and mesh, render spheres at positions indicated by vector S.
 * @param viewer, libigl viewer
 * @param V, #v*3 list of 3d positions of mesh vertices.
 * @param F, #f*3 list of vertex id in each face, three per row.
 * @param S, #v*1 list of singularity indices for each vertex, will render a green sphere at the vertex
 *           where S(i) < 0, and will render a red sphere at the vertex where S(i) > 0.
 * @param point_size, Scalar, controling size of spheres.
 * @return void.
 */
void plot_singularity_sphere(igl::opengl::glfw::Viewer& viewer, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::VectorXd& S, double point_scale = 1.0);


/**
 * Given the mesh, building a mesh of line-segments (cuboids) that consists of line segments for the given edges in list E.
 * @param V_s, #v*3 list of 3d positions of mesh vertices.
 * @param F, #f*3 list of vertex id in each face, three per row.
 * @param E, list of pairs of vertex indices in mesh.
 * @param thick, the thickness of line segments.
 * @return pair of matrix V, F represent the mesh of the line-segments.
 */
std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> get_edges(const Eigen::MatrixXd& V_s, const Eigen::MatrixXi& F, const std::vector<std::vector<int>>& E, float thick);


/**
 * Given the viewer and mesh, render a line segment (approximated by cuboids) for all edges in list E.
 * @param viewer, libigl viewer
 * @param V, #v*3 list of 3d positions of mesh vertices.
 * @param F, #f*3 list of vertex id in each face, three per row.
 * @param c, the RGB color for the line segments.
 * @param E, list of pairs of indices, could be face id or vertex id.
 * @param thick, the thickness of line segments.
 * @param is_dual, whether the given list E contains index of faces (true) or vertices (false).
 * @return void.
 */
void plot_edges(igl::opengl::glfw::Viewer& viewer, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::RowVector3d& c, std::set<std::pair<int,int>> & E, float thick = 1.0,  bool is_dual = false);

/**
 * Given the viewer and mesh, render line segments along the boundaries of mesh.
 * @param viewer, libigl viewer
 * @param V, #v*3 list of 3d positions of mesh vertices.
 * @param F, #f*3 list of vertex id in each face, three per row.
 * @param w, the thickness of line segments.
 * @return void.
 */
void show_boundary(igl::opengl::glfw::Viewer& viewer, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, double w = 1.0);

#endif
