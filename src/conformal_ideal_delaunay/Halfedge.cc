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

#include "Halfedge.hh"
#include <Eigen/Sparse>

void FV_to_NOB(const std::vector<std::vector<int>> &F,
               std::vector<int> &next_he,
               std::vector<int> &opp,
               std::vector<int> &bnd_loops,
               std::vector<int> &vtx_reindex)
{
    // Get the cumulative sum of the number of halfedges per face
    int n_f = F.size();
    int n_he = 0;
    std::vector<int> F_he_cumsum(n_f);
    for (int i = 0; i < n_f; ++i)
    {
        n_he += F[i].size();
        F_he_cumsum[i] = n_he;
    }

    // Create a list of indices of halfedges per face, sequentially numbered, not including
    // boundary-loop faces
    std::vector<std::vector<int>> F_he(n_f);
    F_he[0] = range(0, F_he_cumsum[0]);
    for (int i = 1; i < n_f; ++i)
    {
        F_he[i] = range(F_he_cumsum[i-1], F_he_cumsum[i]);
    }
    // Create the per face next halfedge map
    std::vector<std::vector<int>> F_n(n_f);
    for (int i = 0; i < n_f; ++i)
    {
        F_n[i] = std::vector<int>(F_he[i].size());
        for (int j = 0; j < F_he[i].size()-1; ++j)
        {
            F_n[i][j] = F_he[i][j+1];
        }
        F_n[i][F_he[i].size()-1] = F_he[i][0];
    }


    // Create the next halfedge map (without boundary-loop halfedges)
    next_he.clear();
    next_he.reserve(n_he);
    flatten<int>(F_n, next_he);

    // Get the indices of tail vertices of halfedges
    std::vector<int> tail;
    tail.reserve(n_he);
    flatten<int>(F, tail);

    // Get the per face indices of head vertices of halfedges
    std::vector<std::vector<int>> F_head(n_f);
    for (int i = 0; i < n_f; ++i)
    {
        F_head[i] = std::vector<int>(F[i].size());
        for (int j = 0; j < F[i].size()-1; ++j)
        {
            F_head[i][j] = F[i][j+1];
        }
        F_head[i][F[i].size()-1] = F[i][0];
    }

    // Get the indices of head vertices of halfedges
    std::vector<int> head;
    tail.reserve(n_he);
    flatten<int>(F_head, head);

    // Shift indices by 1 to distinguish from 0 entries in the matrix
    std::vector<int> he_index = range(1, n_he + 1);

    // Get the number of vertices
    int n_v = 0;
    for (int i = 0; i < tail.size(); ++i)
    {
        n_v = std::max<int>(n_v, tail[i] + 1);
    }

    // Create the adjacency matrix (tail, head) -> halfedge index+1
    Eigen::SparseMatrix<int> vv2he(n_v, n_v);
    typedef Eigen::Triplet<int> Trip;
    std::vector<Trip> trips(n_he);
    for (int he = 0; he < n_he; ++he)
    {
        trips[he] = Trip(tail[he], head[he], he_index[he]);
    }
    vv2he.setFromTriplets(trips.begin(), trips.end());

    // Create opp array
    opp.clear();
    opp.resize(n_he);
    for (int he = 0; he < n_he; ++he)
    {
        opp[he] = vv2he.coeffRef(head[he], tail[he]) - 1;
    }

    // Add boundary loop halfedges
    std::vector<int> next_he_ext;
    std::vector<int> opp_ext;
    build_boundary_loops(next_he, opp, next_he_ext, opp_ext);
    next_he = next_he_ext;
    opp = opp_ext;
    
    // Create previous halfedge array
    std::vector<int> prev_he(next_he.size(), -1);
    for (int he = 0; he < next_he.size(); ++he)
    {
        prev_he[next_he[he]] = he;
    }

    // Create circulator function array
    std::vector<int> circ(next_he.size());
    for (int he = 0; he < next_he.size(); ++he)
    {
        circ[he] = prev_he[opp[he]];
    }

    // Create vertex array from the circulator array
    std::vector<std::vector<int>> vert;
    build_orbits(circ, vert);
    
    // Map orbit indices to indices of input vertices
    vtx_reindex.clear();
    vtx_reindex.resize(n_v);
    for (int i = 0; i < n_v; ++i)
    {
        vtx_reindex[i] = head[vert[i][0]];
    }

    // Create boundary loops
    bnd_face_list(next_he, n_he, bnd_loops);
}   

void FV_to_NOB(const Eigen::MatrixXi &F,
               std::vector<int> &next_he,
               std::vector<int> &opp,
               std::vector<int> &bnd_loops,
               std::vector<int> &vtx_reindex)
{
    // Convert eigen matrix to vector of vectors
    std::vector<std::vector<int>> F_vec(F.rows(),std::vector<int>(F.cols()));
    for (int i = 0; i < F.rows(); ++i)
    {
        for (int j = 0; j < F.cols(); ++j)
        {
            F_vec[i][j] = F(i,j);
        }
    }

    // Run FV_to_NOB method
    FV_to_NOB(F_vec, next_he, opp, bnd_loops, vtx_reindex);
}

void build_boundary_loops(const std::vector<int> &next_he,
                          const std::vector<int> &opp,
                          std::vector<int> &next_he_ext,
                          std::vector<int> &opp_ext)
{
    int n_he = opp.size();

    // Get halfedges on the boundary
    std::vector<int> bnd_he;
    for (int he = 0; he < n_he; ++he)
    {
        if (opp[he] == -1)
        {
            bnd_he.push_back(he);
        }
    }
    int n_bnd_he = bnd_he.size();

    // Add boundary halfedges to the extended opp array
    opp_ext = opp;
    std::vector<int> opp_ext_tail(n_bnd_he, 0);
    opp_ext.insert(opp_ext.end(), opp_ext_tail.begin(), opp_ext_tail.end());
    for (int i = 0; i < n_bnd_he; ++i)
    {
        opp_ext[bnd_he[i]] = n_he + i;
        opp_ext[n_he + i] = bnd_he[i];
    }

    // Add boundary halfedges to the extended next halfedge array
    next_he_ext = next_he;
    std::vector<int> next_he_ext_tail(n_bnd_he, -1);
    next_he_ext.insert(next_he_ext.end(), next_he_ext_tail.begin(), next_he_ext_tail.end());
    for (int i = 0; i < n_bnd_he; ++i)
    {
        int he = bnd_he[i];
        int he_it = next_he[he];
        while (opp[he_it] != -1)
        {
            he_it = next_he[opp[he_it]];
        }
        next_he_ext[opp_ext[he_it]] = opp_ext[he];
    }
}

void build_orbits(const std::vector<int> &perm,
                  std::vector<std::vector<int>> &cycles)
{
    int n_perm = perm.size();

    // Get the maximum value in perm
    int max_perm = 0;
    for (int i = 0; i < n_perm; ++i)
    {
        max_perm = std::max(perm[i], max_perm);
    }
    std::vector<bool> visited(max_perm+1, false);

    // Build the cycles of the permutation
    cycles.clear();
    for (int i = 0; i < max_perm + 1; ++i)
    {
        if (!visited[i])
        {
            cycles.push_back(std::vector<int>());
            int i_it = i;
            while (true)
            {
                visited[i_it] = true;
                cycles.back().push_back(i_it);
                i_it = perm[i_it];
                if (i_it == i)
                {
                    break;
                }
            }
        }
    }
}

void bnd_face_list(const std::vector<int> &next_he,
                   const int n_he,
                   std::vector<int> &bnd_loops)
{
    std::vector<std::vector<int>> faces;
    build_orbits(next_he, faces);
    int n_f = faces.size();
    bnd_loops.clear();
    for (int i = 0; i < n_f; ++i)
    {
        if (faces[i][0] >= n_he)
        {
           bnd_loops.push_back(i);
        }
    }
}


std::vector<int> range(int start, int end)
{
    std::vector<int> arr(end-start);
    for (int i = 0; i < end - start; ++i)
    {
        arr[i] = start + i;
    }

    return arr;
}

void NOB_to_connectivity(const std::vector<int> &next_he,
                         const std::vector<int> &opp,
                         const std::vector<int> &bnd_loops,
                         Connectivity &C,
                         const std::vector<int> &diag_he)
{
    // Build previous halfedge array
    int n_he = next_he.size();
    std::vector<int> prev_he = std::vector<int>(n_he, -1);
    for (int he = 0; he < n_he; ++he)
    {
        prev_he[next_he[he]] = he;
    }
    
    // Construct face loops
    std::vector<std::vector<int>> faces;
    build_orbits(next_he, faces);

    // Create a boolean array marking the diagonal halfedge indices
    std::vector<bool> reserved_he(n_he, false);
    for (int i = 0; i < diag_he.size(); ++i)
    {
        reserved_he[diag_he[i]] = true;
    }

    // Map faces to attached halfedges
    int n_f = faces.size();
    std::vector<int> f2he(n_f);
    for (int i = 0; i < n_f; ++i)
    {
        f2he[i] = faces[i][0];
    }

    // Map halfedges to faces
    std::vector<int> he2f_perm;
    he2f_perm.reserve(n_he);
    for (int i = 0; i < n_f; ++i)
    {
        for (int j = 0; j < faces[i].size(); ++j)
        {
            he2f_perm.push_back(i);
        }
    }
    std::vector<int> Fhe;
    Fhe.reserve(n_he);
    flatten<int>(faces, Fhe);
    std::vector<int> he2f(n_he);
    for (int i = 0; i < n_he; ++i)
    {
        he2f[Fhe[i]] = he2f_perm[i];
    }

    // Create vertex list
    std::vector<int> circ(next_he.size());
    for (int he = 0; he < next_he.size(); ++he)
    {
        circ[he] = prev_he[opp[he]];
    }
    std::vector<std::vector<int>> vert_all;
    build_orbits(circ, vert_all);
    std::vector<std::vector<int>> vert;
    vert.reserve(vert_all.size());
    for (int i = 0; i < vert_all.size(); ++i)
    {
        // Skip vertices originating from diagonal halfedges
        if (reserved_he[vert_all[i][0]])
        {
            continue;
        }
        vert.push_back(vert_all[i]);
    }

    // Create out array
    int n_v = vert.size();
    std::vector<int> out(n_v);
    for (int i = 0; i < n_v; ++i)
    {
        out[i] = next_he[vert[i][0]];
    }


    // Create to array
    std::vector<int> to(n_he);
    std::vector<int> vind;
    vind.reserve(n_he);
    for (int i = 0; i < n_v; ++i)
    {
        for (int j = 0; j < vert[i].size(); ++j)
        {
            vind.push_back(i);
        }
    }
    std::vector<int> vhe;
    vhe.reserve(n_he);
    flatten(vert, vhe);
    for (int i = 0; i < n_he; ++i)
    {
        to[vhe[i]] = vind[i];
    }

    // Copy arrays to connectivity struct
    C.n = next_he;
    C.prev = prev_he;
    C.to = to;
    C.f = he2f;
    C.h = f2he;
    C.out = out;
    C.opp = opp;
}

void NOB_to_double(const std::vector<int> &next_he_in,
                   const std::vector<int> &opp_in,
                   const std::vector<int> &bnd_loops_in,
                   Connectivity &C,
                   std::vector<char> &type,
                   std::vector<int> &R)
{
    int n_f = bnd_loops_in[0];

    // Build faces of the input mesh
    std::vector<std::vector<int>> faces_in;
    build_orbits(next_he_in, faces_in);

    // Construct list of halfedges of boundary faces
    std::vector<int> bnd_loop_he;
    for (int i = 0; i < bnd_loops_in.size(); ++ i)
    {
        int l = bnd_loops_in[i];
        bnd_loop_he.insert(bnd_loop_he.end(), faces_in[l].begin(), faces_in[l].end());
    }

    // Get the number of halfedges belonging to faces, which is the first boundary loop halfedge
    int n_he_nb = bnd_loop_he[0];
    for (int i = 0; i < bnd_loop_he.size(); ++i)
    {
        n_he_nb = std::min(bnd_loop_he[i], n_he_nb);
    }

    // Create truncated arrays excluding boundary loop halfedges
    std::vector<int> opp_nb(opp_in.begin(), opp_in.begin() + n_he_nb);
    std::vector<std::vector<int>> faces_nb(faces_in.begin(), faces_in.begin() + bnd_loops_in[0]);

    // The symmetry map simply reverses the order of halfedges
    R.clear();
    R.resize(2*n_he_nb);
    for (int i = 0; i < 2*n_he_nb; ++i)
    {
        R[i] = 2*n_he_nb - i - 1;
    }

    // The faces as a result of the above symmetry map definition also reverse order
    std::vector<std::vector<int>> faces_new(n_f, std::vector<int>());
    for (int i = 0; i < n_f; ++i)
    {
        int face_size = faces_nb[n_f-i-1].size();
        faces_new[i].resize(face_size);
        for (int j = 0; j < face_size; ++j)
        {
            faces_new[i][j] = R[faces_nb[n_f-i-1][face_size-j-1]];
        }
    }
    std::vector<std::vector<int>> faces_double(faces_nb);
    faces_double.insert(faces_double.end(), faces_new.begin(), faces_new.end());

    // Construct the next halfedge array for the doubled mesh
    std::vector<int> he_double;
    he_double.reserve(2*n_he_nb);
    flatten<int>(faces_double, he_double);
    std::vector<std::vector<int>> next_he_double_faces(2*n_f, std::vector<int>());
    for (int i = 0; i < 2*n_f; ++i)
    {
        next_he_double_faces[i].reserve(faces_double[i].size());
        next_he_double_faces[i].insert(next_he_double_faces[i].end(),
                                       faces_double[i].begin()+1,
                                       faces_double[i].end());
        next_he_double_faces[i].push_back(faces_double[i][0]);
    }
    std::vector<int> next_he_double;
    next_he_double.reserve(2*n_he_nb);
    flatten<int>(next_he_double_faces, next_he_double);
    permute<int>(next_he_double, he_double);
    
    // Rebuild the faces in the the canonical oder inferred from faces
    std::vector<std::vector<int>> ordered_faces_double;
    build_orbits(next_he_double, ordered_faces_double);

    // Rebuild the halfedges of the faces enumerated sequentially
    //std::vector<int> he_double_ordered;
    //he_double_ordered.reserve(2*n_he_nb);
    //flatten<int>(ordered_faces_double, he_double_ordered);

    // Construct opp for the doubled mesh
    std::vector<int> bnd_he;
    std::vector<int> inter_he;
    for (int i = 0; i < opp_nb.size(); ++i)
    {
        if (opp_nb[i] >= n_he_nb)
        {
            bnd_he.push_back(i);
        }
        else
        {
            inter_he.push_back(i);
        }
    }
    std::vector<int> bnd_he_new(bnd_he.size());
    for (int i = 0; i < bnd_he.size(); ++i)
    {
        bnd_he_new[i] = R[bnd_he[i]];
    }
    for (int i = 0; i < bnd_he.size(); ++i)
    {
        opp_nb[bnd_he[i]] = bnd_he_new[i];
    }
    std::vector<int> opp_new(n_he_nb);
    for (int i = 0; i < n_he_nb; ++i)
    {
        opp_new[i] = R[opp_nb[R[n_he_nb+i]]];
    }
    std::vector<int> opp_double(opp_nb);
    opp_double.insert(opp_double.end(), opp_new.begin(), opp_new.end());

    // There is no boundary in the doubled mesh
    std::vector<int> bnd_loops_double;

    // Create the connectivity arrays for the doubled mesh
    NOB_to_connectivity(next_he_double, opp_double, bnd_loops_double, C);
    
    // Label the halfedges from the original mesh with 1 and the new mesh with 2
    type.clear();
    type.resize(2*n_he_nb);
    for (int i = 0; i < n_he_nb; ++i)
    {
        type[i] = 1;
    }
    for (int i = n_he_nb; i < 2*n_he_nb; ++i)
    {
        type[i] = 2;
    }
    for (int i = 0; i < bnd_he.size(); ++i)
    {
        type[bnd_he[i]] = 1;
    }
    for (int i = 0; i < bnd_he_new.size(); ++i)
    {
        type[bnd_he_new[i]] = 2;
    }
}


void find_indep_vertices(const std::vector<int> &out,
                         const std::vector<int> &to,
                         const std::vector<char> &type,
                         const std::vector<int> &R,
                         std::vector<int> &indep_vtx,
                         std::vector<int> &dep_vtx,
                         std::vector<int> &v_rep)
{
    indep_vtx.clear();
    dep_vtx.clear();
    v_rep.clear();
    int n_v = out.size();
    int n_he = to.size();

    // Create a reflection map for the vertices
    std::vector<int> refl_v(n_v);
    for (int i = 0; i < n_v; ++i)
    {
        refl_v[i] = to[R[out[i]]];
    }

    // Label the vertices as type 1=D1, 2=D2, or 3=S (on the symmetry line)
    std::vector<char> vtype(n_v);
    for (int he = 0; he < n_he; ++he)
    {
        if ((type[he] == 1) || (type[he] == 2))
        {
            vtype[to[he]] = type[he];
        }
    }
    for (int i = 0; i < n_v; ++i)
    {
        if (refl_v[i] == i)
        {
            vtype[i] = 3;
        }
    }

    // Partition vertices of type 1 and 3 (arising from the original mesh) and vertices
    // of type 2 (created in the double)
    indep_vtx.reserve(n_v);
    dep_vtx.reserve(n_v);
    for (int i = 0; i < n_v; ++i)
    {
        if (vtype[i] == 2)
        {
            dep_vtx.push_back(i);
        }
        else
        {
            indep_vtx.push_back(i);
        }
    }

    // Map independent vertices to unique indices and dependent vertices to their reflection's
    // index
    v_rep.resize(n_v);
    for (int i = 0; i < indep_vtx.size(); ++i)
    {
        v_rep[indep_vtx[i]] = i;
    }
    for (int i = 0; i < dep_vtx.size(); ++i)
    {
        v_rep[dep_vtx[i]] = v_rep[refl_v[dep_vtx[i]]];
    }
}

void create_tufted_cover(const std::vector<char> &type,
                         const std::vector<int> &R,
                         const std::vector<int> &indep_vtx,
                         const std::vector<int> &dep_vtx,
                         const std::vector<int> &v_rep,
                         std::vector<int> &out,
                         std::vector<int> &to)
{
    int n_v = out.size();
    int n_he = to.size();

    // Modify the to and out arrays to identify dependent vertices with their reflection
    std::vector<int> out_tufted(indep_vtx.size());
    for (int i = 0; i < n_he; ++i)
    {
        to[i] = v_rep[to[i]];
    }
    for (int i = 0; i < indep_vtx.size(); ++i)
    {
        out_tufted[i] = out[indep_vtx[i]];
    }
    out = out_tufted;
}
