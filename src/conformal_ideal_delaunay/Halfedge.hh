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

#ifndef HALFEDGE_HH
#define HALFEDGE_HH
#include <vector>
#include <Eigen/Sparse>

struct Connectivity
{
    std::vector<int> n;    // #h, next halfedge id
    std::vector<int> prev; // #h, previous halfedge id
    std::vector<int> to;   // #h, vertex id the current halfedge is pointing to
    std::vector<int> f;    // #h, face id on the left of the halfedge
    std::vector<int> h;    // #f, one halfedge that's inside the current face
    std::vector<int> out;  // #v, one halfedge that's pointing from the current vertex
    std::vector<int> opp;  // #h, opposite halfedge id
};

/**
* Convert from matrix (F) mesh representation to NOB data structure
* 
* @param F, #f*n, each row represents the vertex id (ccw) of current face
* @param (N)next_he, size #h vector, next halfedge id
* @param (O)opp, size #h vector, opposite halfedge id
* @param (B)bnd_loops, collection of boundary face ids.
* @param vtx_reindex, map from new vertices to old (vertex id in NOB is different from F)
* @return void
*/
void FV_to_NOB(const std::vector<std::vector<int>> &F,
               std::vector<int> &next_he,
               std::vector<int> &opp,
               std::vector<int> &bnd_loops,
               std::vector<int> &vtx_reindex);

void FV_to_NOB(const Eigen::MatrixXi &F,
               std::vector<int> &next_he,
               std::vector<int> &opp,
               std::vector<int> &bnd_loops,
               std::vector<int> &vtx_reindex);

/**
* Extend next_he and opp to add extra halfedges along the boundaries.
* 
* @param next_he, next-halfedge map same length as opp
* @param opp, halfedge map, for boundary halfedges -1
* @param next_he_ext, next_halfedge map, same length as opp_ext; newly added halfedges are linked into boundary loops
* @param opp_ext, opp with an extra coupled halfedge added at the end for each boundary halfedge
*                 all halfedges have a pair 
* @return void
*/
void build_boundary_loops(const std::vector<int> &next_he,
                          const std::vector<int> &opp,
                          std::vector<int> &next_he_ext,
                          std::vector<int> &opp_ext);

/**
* Build orbits following next id recorded in perm.
* 
* @param perm, a permutation  given by a list of non-repeating integers in the range 0..len(perm)
* @param cycles, a list of lists, each list represents a cycle of perm
* @return void
*/
void build_orbits(const std::vector<int> &perm,
                  std::vector<std::vector<int>> &cycles);


void bnd_face_list(const std::vector<int> &next_he,
                   const int n_he,
                   std::vector<int> &bnd_loops);


template <class T>
void flatten(const std::vector<std::vector<T>> &arr,
             std::vector<T> &arr_flat)
{
    for (int i = 0; i < arr.size(); ++i)
    {
        for (int j = 0; j < arr[i].size(); ++j)
        {
            arr_flat.push_back(arr[i][j]);
        }
    }
}

/**
 * Given next_he, opp, bnd_loops as defined in Connectivity, infer the rest fields.
 * @param next_he(n), opp, bnd_loops (NOB)
 * @param Connectivity structure C, with fields above plus inferred from these: 
 *    f:    index of face for each halfedge (faces ordered as next_he orbits, i.e., by minimal halfedge index
 *    prev: inverse of next_he
 *    to:   index of the vertex halfedge is pointing to (vertices ordered as circulator orbits, i.e., by minimal index 
 *          of the halfedge pointing from v
 *    h:    an index of a halfedge of f (initially set to the min index halfedge in the face)
 *    out:  an index of a halfedge pointing from v (initially set to min index)
*/
void NOB_to_connectivity(const std::vector<int> &next_he,
                         const std::vector<int> &opp,
                         const std::vector<int> &bnd_loops,
                         Connectivity &C,
                         const std::vector<int> &diag_he = std::vector<int>());
/**
 * Creates a doubled mesh with reflectional symmetry map and corresponding labels froma   mesh connectivity with (possibly empty) boundary, specified by next_he, opp, bnd_loops
 * ASSUMES that the boundary loop halfedges are at the end of indices so boundary loops are at the end 
 * construct a mesh with double number of faces and glued to the original mesh along the boundary
 * @params: next_he, opp, bnd_loops 
 * @params: Connectivity and Reflection structures for the double mesh
 * @return: void
 */
void NOB_to_double(const std::vector<int> &next_he_in,
                   const std::vector<int> &opp_in,
                   const std::vector<int> &bnd_loops_in,
                   Connectivity &C,
                   std::vector<char> &type,
                   std::vector<int> &R);


/**
 * Partition vertices in a doubled mesh with reflection map into independent and dependent
   vertices for symmetric functions on the vertices of the mesh, and create a map from the
   vertices to their corresponding independent vertices.
 * @params: out, to, and Reflection structures for the double mesh
 * @params: indep_vtx, dep_vtx partitioned sets of vertices
 * @params: v_rep map of vertices to independent vertex
 * @return: void
 */
void find_indep_vertices(const std::vector<int> &out,
                         const std::vector<int> &to,
                         const std::vector<char> &type,
                         const std::vector<int> &R,
                         std::vector<int> &indep_vtx,
                         std::vector<int> &dep_vtx,
                         std::vector<int> &v_rep);

/**
 * Dependent vertices of a doubled mesh are identified with their corresponding independent vertex 
 * @params: to, out, and Reflection structures for the double mesh
 * @return: void
 */
void create_tufted_cover(const std::vector<char> &type,
                         const std::vector<int> &R,
                         const std::vector<int> &indep_vtx,
                         const std::vector<int> &dep_vtx,
                         const std::vector<int> &v_rep,
                         std::vector<int> &out,
                         std::vector<int> &to);


template <typename T>
void permute(std::vector<T> &arr,
             const std::vector<int> &perm)
{
    std::vector<T> arr_copy = arr;
    for (int i = 0; i < arr.size(); ++i)
    {
        arr[perm[i]] = arr_copy[i];
    }
}

/**
 * Infer the intial edge lengths from 3d vertices positions defined in matrix V.
 * @params: Connectivity defined on the top
 * @params: V, #v*3 matrix, represents 3d-coordinates for each vertex
 * @params: vtx_reindex, map between new mesh indices (in Connectivity) to old (in V)
 * @return: void
 */
template <typename Scalar>
void compute_l_from_vertices(const Connectivity &C,
                             const Eigen::MatrixXd &V,
                             const std::vector<int> &vtx_reindex,
                             std::vector<Scalar> &l)
{
    int n_he = C.to.size();
    l.resize(n_he);

    for (int he = 0; he < n_he; ++he)
    {
        // Get the verticess at the tip and tail of the halfedge
        Eigen::Vector3d v_to = V.row(vtx_reindex[C.to[he]]);
        Eigen::Vector3d v_fr = V.row(vtx_reindex[C.to[C.opp[he]]]);
        
        // Compute the length of the halfedge from the displacement vector
        Eigen::Vector3d vec_disp = v_to - v_fr;
        l[he] = sqrt(vec_disp.dot(vec_disp));
    }
}

std::vector<int> range(int start, int end);

#endif
