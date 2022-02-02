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

/** @file ConformalInterface.hh
 *  @brief Top level definition of interface for conformal ideal delaunay algorithm.
 */

#ifndef CONFORMAL_HH 
#define CONFORMAL_HH

#include "ConformalIdealDelaunayMapping.hh"
#include "Halfedge.hh"
#include "Layout.hh"
#include <igl/boundary_loop.h>
#include <igl/is_border_vertex.h>
#include <igl/edges.h>
#include <igl/writeOBJ.h>

/**
 * Convert triangle mesh in V, F format to halfedge structure.
 * 
 * @param V dim #v*3 matrix, each row corresponds to mesh vertex coordinates
 * @param F dim #f*3 matrix, each row corresponds to three vertex ids of each facet
 * @param Theta_hat dim #v vector, each element is the prescribed angle sum at each vertex
 * @param vtx_reindex, dim #v int-vector, stores the correspondence of new vertex id in mesh m to old index in V
 * @param indep_vtx, int-vector, stores index of identified independent vertices in the original copy
 * @param dep_vtx, int-vector, stores index of new added vertex copies of the double cover
 * @param v_rep, dim #v int-vector, map independent vertices to unique indices and dependent vertices to their reflection's index
 * @return m, Mesh data structure, for details check OverlayMesh.hh
 */
template <typename Scalar>
Mesh<Scalar>
FV_to_double(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const std::vector<Scalar> &Theta_hat, std::vector<int>& vtx_reindex, std::vector<int>& indep_vtx, std::vector<int>& dep_vtx, std::vector<int>& v_rep, std::vector<int>& bnd_loops){
    Mesh<Scalar> m;
    // Build the NOB representation from the input connectivity
    std::vector<int> next_he;
    std::vector<int> opp;
    FV_to_NOB(F, next_he, opp, bnd_loops, vtx_reindex);

    // Build the connectivity arrays from the NOB arrays
    Connectivity C;
    NOB_to_connectivity(next_he, opp, bnd_loops, C);

    // Build the edge length array from the vertex positions
    std::vector<Scalar> l;
    compute_l_from_vertices<Scalar>(C, V, vtx_reindex, l);

    // Permute the target angles to match the new vertex indices of the halfedge mesh
    std::vector<Scalar> Theta_hat_perm(Theta_hat.size());
    for (int i = 0; i < Theta_hat_perm.size(); ++i)
    {
        Theta_hat_perm[i] = Theta_hat[vtx_reindex[i]];
    }
    
    // If there is no boundary, create a mesh with trivial reflection information
    if (bnd_loops.size() == 0)
    {
        int n_v = C.to.size();
        int n_he = C.n.size();

        // Create trivial reflection information
        std::vector<char> type(n_he, 0);
        std::vector<int> R(n_he, 0);

        // Create a halfedge structure for the mesh
        m.n = C.n;
        m.to = C.to;
        m.f = C.f;
        m.h = C.h;
        m.out = C.out;
        m.opp = C.opp;
        m.type = type;
        m.type_input = type;
        m.R = R;
        m.l = l;
        m.Th_hat = Theta_hat_perm;
        m.v_rep = range(0, n_v);
    }
    // If there is boundary, create a double tufted cover with a reflection map
    else
    {
        // Create the doubled mesh connectivity information
        Connectivity C_double;
        std::vector<char> type;
        std::vector<int> R;
        NOB_to_double(next_he, opp, bnd_loops, C_double, type, R);
        find_indep_vertices(C_double.out, C_double.to, type, R, indep_vtx, dep_vtx, v_rep);
        int n_v = C.out.size();
        int n_v_double = C_double.out.size();

        // Double the target angle array
        std::vector<Scalar> Theta_hat_double(n_v);
        for (int i = 0; i < n_v; ++i)
        {
            Theta_hat_double[i] = 2*Theta_hat_perm[C.to[C.opp[C_double.out[indep_vtx[i]]]]];
        }

        // Double the length array FIXME Only works for double tufted cover
        std::vector<Scalar> l_double;
        std::vector<int> vtx_reindex_double(n_v_double);
        for (int i = 0; i < n_v_double; ++i)
        {
            vtx_reindex_double[i] = vtx_reindex[v_rep[i]];
        }
        compute_l_from_vertices<Scalar>(C_double, V, vtx_reindex_double, l_double);

        // Create the halfedge structure for the doubled mesh
        m.n = C_double.n;
        m.to = C_double.to;
        m.f = C_double.f;
        m.h = C_double.h;
        m.out = C_double.out;
        m.opp = C_double.opp;
        m.type = type;
        m.type_input = type;
        m.R = R;
        m.l = l_double;
        m.Th_hat = Theta_hat_double;
        m.v_rep = v_rep;
    }
    return m;
}

/**
 * Helper function for compute the topological info (#bd, #genus) of the input mesh
 * @param V dim #v*3 matrix, each row corresponds to mesh vertex coordinates
 * @param F dim #f*3 matrix, each row corresponds to three vertex ids of each facet
 * @return n_genus, int, genus of the input mesh
 * @return n_boundary, int, number of boundary loops of the input mesh
 */ 
std::pair<int,int> count_genus_and_boundary(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F){
  int nv = V.rows();
  int nf = F.rows();
  Eigen::MatrixXi E;
  igl::edges(F, E);

  std::vector<std::vector<int>> bds;
  igl::boundary_loop(F, bds);

  int n_bd = bds.size();
  int ne = E.rows();
  int eu = nv - ne + nf + n_bd;
  int n_genus = (2-eu)/2;
  return std::make_pair(n_genus, n_bd);
}

/**
 * Top level c++ interface of conformal ideal delaunay algorithm.
 * This function computes a conformal metric for the input triangle mesh 
 * wrt the given curvature and with zero initial scale factors, also supports 
 * mapping a collection of sample points on original mesh onto the final mesh
 * 
 * @param V dim #v*3 matrix, each row corresponds to mesh vertex coordinates
 * @param F dim #f*3 matrix, each row corresponds to three vertex ids of each facet
 * @param Theta_hat dim #v vector, each element is the prescribed angle sum at each vertex
 * @param pt_fids, dim #s int-vector, optional, contains face id of each sample points on original mesh
 * @param pt_bcs, dim #s vector, optional, contains barycentric-coordinates of each sample point on original mesh
 * @param alg_params, optional, algorithm parameters, for details check ConformalIdealDelaunayMapping.hh
 * @param ls_params, optional, line search parameters, for details check ConformalIdealDelaunayMapping.hh
 * @param stats_params, optional, statistic parameters, for details check ConformalIdealDelaunayMapping.hh
 * @return m, Mesh data structure, for details check OverlayMesh.hh
 * @return u, dim #v vector, final scale factors assigned to each vertex
 * @return pt_fids, dim #s int-vector, contains updated face id of each sample points on original mesh
 * @return pt_bcs, dim #s vector, contains updated barycentric-coordinates of each sample point on original mesh
 * @return vtx_reindex, dim #v int-vector, stores the correspondence of new vertex id in mesh m to old index in V
 */
template<typename Scalar>
std::tuple<
           OverlayMesh<Scalar>,                     // m
           std::vector<Scalar>,                     // u
           std::vector<int>,                        // pt_fids
           std::vector<std::vector<Scalar>>,        // pt_bcs
           std::vector<int>,                        // vtx_reindex
           std::vector<std::vector<Scalar>>>        // V_overlay
conformal_metric(const Eigen::MatrixXd &V,
                    const Eigen::MatrixXi &F,
                    const std::vector<Scalar> &Theta_hat,
                    std::vector<int>& pt_fids,
                    std::vector<Eigen::Matrix<double, 3, 1>>& pt_bcs,
                    std::shared_ptr<AlgorithmParameters> alg_params=nullptr,
                    std::shared_ptr<LineSearchParameters> ls_params=nullptr,
                    std::shared_ptr<StatsParameters> stats_params=nullptr
){
    
    if(alg_params == nullptr) alg_params = std::make_shared<AlgorithmParameters>();
    if(ls_params    == nullptr) ls_params    = std::make_shared<LineSearchParameters>();
    if(stats_params == nullptr) stats_params = std::make_shared<StatsParameters>();

#ifdef WITH_MPFR
    mpfr::mpreal::set_default_prec(alg_params->MPFR_PREC);
    mpfr::mpreal::set_emax(mpfr::mpreal::get_emax_max());
    mpfr::mpreal::set_emin(mpfr::mpreal::get_emin_min());
#endif

    std::vector<Scalar> u;
    std::vector<int> vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops;
    Mesh<Scalar> m = FV_to_double(V, F, Theta_hat, vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops);

    OverlayMesh<Scalar> mo(m);

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> u0; u0.setZero(m.n_ind_vertices());
    std::vector<Eigen::Matrix<Scalar, 3, 1>> pt_bcs_scalar(pt_bcs.size());
    for (int i = 0; i < pt_bcs.size(); i++)
    {
        pt_bcs_scalar[i] = pt_bcs[i].template cast<Scalar>(); 
    }
    auto conformal_out = ConformalIdealDelaunay<Scalar>::FindConformalMetric(mo, u0, pt_fids, pt_bcs_scalar, *alg_params, *ls_params, *stats_params);
    auto u_o = std::get<0>(conformal_out);
    auto flip_seq = std::get<1>(conformal_out);
    u.resize(u_o.rows());
    for(int i = 0; i < u_o.rows(); i++)
        u[i] = u_o[i];
    mo.garbage_collection();
    std::vector<std::vector<Scalar>> V_reindex(3);
    for (int i = 0; i < 3; i++)
    {
        V_reindex[i].resize(mo._m.out.size(), 0);
        for (int j = 0; j < V.rows(); j++)
        {
            V_reindex[i][j] = Scalar(V(vtx_reindex[j], i));
        }
    }
    std::vector<std::vector<Scalar>> V_overlay;
    if(!mo.bypass_overlay)
        V_overlay = ConformalIdealDelaunay<Scalar>::Interpolate_3d(mo, flip_seq, V_reindex);
    if(mo.bypass_overlay)
        spdlog::warn("overlay bypassed due to numerical issue or as instructed.");
    if (bnd_loops.size() != 0)
    {
        int n_v = V.rows();
        auto mc = mo.cmesh();
        create_tufted_cover(mo._m.type, mo._m.R, indep_vtx, dep_vtx, v_rep, mo._m.out, mo._m.to);
        mo._m.v_rep = range(0, n_v);
    }
    
    // Eigen::Vector to std::vector for pybind
    std::vector<std::vector<Scalar>> pt_bcs_out(pt_bcs_scalar.size());
    for (int i = 0; i < pt_bcs_scalar.size(); i++)
    {
        for (int j = 0; j < 3; j++)
        {
            pt_bcs_out[i].push_back(pt_bcs_scalar[i](j));
        }
    }
    return std::make_tuple(mo, u, pt_fids, pt_bcs_out, vtx_reindex, V_overlay); 

}

/**
 * Helper function to (1) split doubled-OverlayMesh into one part(if necessary) and triangulate it by adding diagonals then (2) update the per corner layout result accordingly.
 * @param mo OverlayMesh data structure
 * @param f_labels dim #faces vector, indicate which part a face is beong to
 * @param u_o per corner layout (u coordinates)
 * @param v_o per corner layout (v coordinates)
 * @return n_new next array of the result half-edge structure
 * @return opp_new opp array of the result half-edge structure
 * @return u_new updated per corner layout (u coordinates)
 * @return v_new updated per corner layout (v coordinates)
 */
template<typename Scalar>
void split_overlay(const OverlayMesh<Scalar> &mo, 
              const std::vector<int> &f_labels,
              const std::vector<Scalar> &u_o,
              const std::vector<Scalar> &v_o,
              std::vector<int> &n_new,
              std::vector<int> &opp_new,
              std::vector<Scalar> &u_new,
              std::vector<Scalar> &v_new
              )
{
    int cnt = 0;
    std::vector<int> he_d2s(mo.n.size(), -1);
    for (int i = 0; i < mo.n.size(); i++)
    {
        if (f_labels[mo.f[i]] != 2)
        {
            he_d2s[i] = cnt;
            cnt++;
        }
    }
    
    n_new.resize(cnt);
    opp_new.resize(cnt);
    for (int i = 0; i < mo.n.size(); i++)
    {
        if (he_d2s[i] == -1) continue;
        n_new[he_d2s[i]] = he_d2s[mo.n[i]];
        opp_new[he_d2s[i]] = he_d2s[mo.opp[i]];
    }

    // reindex u_new, v_new from uo, vo
    u_new.resize(cnt);
    v_new.resize(cnt);
    for (int i = 0; i < mo.n.size(); i++)
    {
        if (he_d2s[i] == -1) continue;
        u_new[he_d2s[i]] = u_o[i];
        v_new[he_d2s[i]] = v_o[i];
    }

    // triangulate polygons
    for (int i = 0; i < mo.h.size(); i++)
    {
        if (f_labels[mo.f[i]] == 2) continue;
        int h0 = he_d2s[i];
        int hc = n_new[h0];
        std::vector<int> hf;
        hf.push_back(h0);
        while (hc != h0)
        {
            hf.push_back(hc);
            hc = n_new[hc];
        }
        if (hf.size() == 3) continue;
        for (int j = 0; j < hf.size() - 3; j++)
        {
            int cur_size = n_new.size();
            opp_new.push_back(cur_size + 1);
            opp_new.push_back(cur_size);
            u_new.push_back(u_new[hf[0]]);
            u_new.push_back(u_new[hf[j + 1]]);
            v_new.push_back(v_new[hf[0]]);
            v_new.push_back(v_new[hf[j + 1]]);

            if (j == 0)
            {
                n_new.push_back(hf[0]);
                n_new.push_back(hf[j + 2]);
                n_new[hf[j + 1]] = cur_size;
            }
            else
            {
                n_new.push_back(cur_size - 1); // previous diagonal
                n_new.push_back(hf[j + 2]);
                n_new[hf[j + 1]] = cur_size;
            }
            
            // last diagonal
            if (j == hf.size() - 4)
            {
                n_new[hf.back()] = cur_size + 1; // last diagonal
            }
        }
    }
    assert(n_new.size() == u_new.size() && n_new.size() == v_new.size() && n_new.size() == opp_new.size());
}

/**
 * Helper function to get the V,F,uv output from half_edge structure an per corner layout
 * @param mo OverlayMesh data structure
 * @param is_cut_o cut_to_singularity label
 * @param u_o per corner layout (u coordinates)
 * @param v_o per corner layout (v coordinates)
 * @return v3d_out dim #v*3 vector, each row corresponds to mesh vertex coordinates
 * @return u_o_out dim #v vector, u coordinates of layout
 * @return v_o_out dim #v vector, v coordniates of layout
 * @return F_out dim #f*3 vector, each row corresponds to three vertex ids of each facet
 */ 
template<typename Scalar>
std::tuple<
        std::vector<std::vector<Scalar>>,   // v3d_out
        std::vector<Scalar>,                // u_o_out
        std::vector<Scalar>,                // v_o_out
        std::vector<std::vector<int>>       // F_out
> 
get_FV(OverlayMesh<Scalar> &mo,
            std::vector<bool> &is_cut_o,
            const std::vector<std::vector<Scalar>> &v3d, 
            const std::vector<Scalar> &u_o,
            const std::vector<Scalar> &v_o)
{
    // get h_group and to_map
    std::vector<int> f_labels = get_overlay_face_labels(mo);
    
    // modify is_cut_o
    for (int i = 0; i < is_cut_o.size(); i++)
    {
        if (f_labels[mo.f[i]] != f_labels[mo.f[mo.opp[i]]])
        {
            is_cut_o[i] = true;
        }
    }   
    
    int origin_size = mo.cmesh().out.size();
    std::vector<int> h_group(mo.n.size(), -1);
    std::vector<int> to_map(origin_size, -1);
    for (int i = 0; i < mo.n.size(); i++)
    {
        if (h_group[i] != -1 || f_labels[mo.f[i]] == 2) continue;
        if (mo.to[i] < origin_size && to_map[mo.to[i]] == -1)
        {
            h_group[i] = mo.to[i];
            to_map[mo.to[i]] = i;
        }    
        else
        {
            h_group[i] = to_map.size();
            to_map.push_back(i);
        }
        int cur = mo.n[i];
        while (is_cut_o[cur] == false && mo.opp[cur] != i)
        {
            cur = mo.opp[cur];
            h_group[cur] = h_group[i];
            cur = mo.n[cur];
        }
        cur = mo.opp[i];
        while (is_cut_o[cur] == false && mo.prev[cur] != i)
        {
            cur = mo.prev[cur];
            h_group[cur] = h_group[i];
            cur = mo.opp[cur];
        }
    }

    spdlog::debug("to_map size: {}", to_map.size());

    // reindex V and uv
    std::vector<std::vector<Scalar>> v3d_out(3);
    v3d_out[0].resize(to_map.size());
    v3d_out[1].resize(to_map.size());
    v3d_out[2].resize(to_map.size());
    std::vector<Scalar> u_o_out(to_map.size());
    std::vector<Scalar> v_o_out(to_map.size());
    for (int i = 0; i < to_map.size(); i++)
    {
        v3d_out[0][i] = v3d[0][mo.to[to_map[i]]];
        v3d_out[1][i] = v3d[1][mo.to[to_map[i]]];
        v3d_out[2][i] = v3d[2][mo.to[to_map[i]]];
        u_o_out[i] = u_o[to_map[i]];
        v_o_out[i] = v_o[to_map[i]];
    }

    // get F
    std::vector<std::vector<int>> F_out;
    for (int i = 0; i < mo.h.size(); i++)
    {
        if (f_labels[i] == 2) continue;
        int h0 = mo.h[i];
        int hc = mo.n[h0];
        std::vector<int> hf;
        hf.push_back(h0);
        while (hc != h0)
        {
            hf.push_back(hc);
            hc = mo.n[hc];
        }
        for (int j = 0; j < hf.size() - 2; j++)
        {
            F_out.push_back(std::vector<int>{h_group[h0], h_group[hf[j + 1]], h_group[hf[j + 2]]});
        }
    }

    return std::make_tuple(v3d_out, u_o_out, v_o_out, F_out);
}


/**
 * Compute the map between faces in overlay mesh and original mesh
 * @param mo OverlayMesh data structure
 * @param Fn_to_F map from each overlay mesh face to the id of original mesh face id that contains it
 */ 
template <typename Scalar>
void build_face_maps(OverlayMesh<Scalar>& mo, std::vector<int>& Fn_to_F){
    
    Fn_to_F = std::vector<int>(mo.n_faces(), -1);
    
    // start from the face of halfedge h and do a breath first search to reach as
    // many faces as possible while only going across CURRENT_EDGE
    // the vector `compo` contains all sub-faces that correspoinding to the same original face
    auto flood_fill = [](OverlayMesh<Scalar>& mo, int h, std::vector<int>& compo){
        compo.clear();
        auto done = std::vector<bool>(mo.n_faces(), false);
        std::queue<int> Q; Q.push(h);
        done[mo.f[h]] = true;
        while (!Q.empty())
        {
            h = Q.front();
            Q.pop();
            // traverse all adjacent faces while only going through CURRENT_EDGE
            int hi = h;
            do{
                int hi_opp = mo.opp[hi];
                if (mo.f[hi_opp] != -1 && !done[mo.f[hi_opp]] && mo.edge_type[hi] == CURRENT_EDGE)
                {
                    done[mo.f[hi_opp]] = true;
                    Q.push(hi_opp);
                }
                hi = mo.n[hi];
            }while(h != hi);
        }
        for(int f = 0; f < done.size(); f++)
            if(done[f])
                compo.push_back(f);
    };
    
    for(int i = 0; i < mo.n_halfedges(); i++){
        if(mo.edge_type[i] == ORIGINAL_EDGE){
            if(mo.origin[i] != mo.origin_of_origin[i]){
                mo.origin_of_origin[i] = mo.origin[i];
            }
        }
    }
    // go through all faces and pick any halfedge in that face
    // find all 'sibling' faces that belong to the same original face
    for(int f = 0; f < mo.n_faces(); f++){
        int h0 = mo.h[f];
        std::vector<int> compo;
        flood_fill(mo, h0, compo);
        for(int fx: compo){
            int h1 = mo.h[fx];
            int hi = h1;
            do{
                if(mo.edge_type[hi] != CURRENT_EDGE) break;
                hi = mo.n[hi];
            }while(hi != h1);
            int fi = mo.m0.f[mo.origin_of_origin[hi]];
            Fn_to_F[fx] = fi;
        }
    }

}

template<typename Scalar>
std::tuple<
        std::vector<std::vector<Scalar>>,   // v3d_out
        std::vector<Scalar>,                // u_out
        std::vector<Scalar>,                // v_out
        std::vector<std::vector<int>>,      // F_out
        std::vector<std::vector<int>>,      // FT_out
        std::vector<int>>                   // Fn_to_F
get_FV_FTVT(OverlayMesh<Scalar> &mo,
            std::vector<bool> &is_cut_o,
            const std::vector<std::vector<Scalar>> v3d, 
            const std::vector<Scalar> u_o,
            const std::vector<Scalar> v_o)
{
    // get h_group and to_map
    std::vector<int> f_labels = get_overlay_face_labels(mo);
    
    int origin_size = mo.cmesh().out.size();
    std::vector<int> h_group(mo.n.size(), -1);
    std::vector<int> to_map(origin_size, -1);
    std::vector<int> to_group(origin_size, -1);
    std::vector<int> which_to_group(mo.out.size(), -1);
    for (int i = 0; i < to_group.size(); i++)
    {
        to_group[i] = i;
        which_to_group[i] = i;
    }
    for (int i = 0; i < mo.n.size(); i++)
    {
        if (h_group[i] != -1 || f_labels[mo.f[i]] == 2) continue;
        if (which_to_group[mo.to[i]] == -1)
        {
            which_to_group[mo.to[i]] = to_group.size();
            to_group.push_back(mo.to[i]);
        }
        if (mo.to[i] < origin_size && to_map[mo.to[i]] == -1)
        {
            h_group[i] = mo.to[i];
            to_map[mo.to[i]] = i;
        }    
        else
        {
            h_group[i] = to_map.size();
            to_map.push_back(i);
        }
        int cur = mo.n[i];
        while (is_cut_o[cur] == false && mo.opp[cur] != i)
        {
            cur = mo.opp[cur];
            h_group[cur] = h_group[i];
            cur = mo.n[cur];
        }
        cur = mo.opp[i];
        while (is_cut_o[cur] == false && mo.prev[cur] != i)
        {
            cur = mo.prev[cur];
            h_group[cur] = h_group[i];
            cur = mo.opp[cur];
        }
    }

    std::vector<std::vector<Scalar>> v3d_out(3);
    v3d_out[0].resize(to_group.size());
    v3d_out[1].resize(to_group.size());
    v3d_out[2].resize(to_group.size());
    std::vector<Scalar> u_o_out(to_map.size());
    std::vector<Scalar> v_o_out(to_map.size());
    for (int i = 0; i < to_map.size(); i++)
    {
        u_o_out[i] = u_o[to_map[i]];
        v_o_out[i] = v_o[to_map[i]];
    }
    for (int i = 0; i < to_group.size(); i++)
    {
        v3d_out[0][i] = v3d[0][to_group[i]];
        v3d_out[1][i] = v3d[1][to_group[i]];
        v3d_out[2][i] = v3d[2][to_group[i]];
    }
    std::vector<int> Fn_to_F;
    build_face_maps(mo, Fn_to_F);

    std::vector<std::vector<int>> F_out;
    std::vector<std::vector<int>> FT_out;
    std::vector<int> f_remap; // map each face in F_out to mo
    for (int i = 0; i < mo.h.size(); i++)
    {
        if (f_labels[i] == 2) continue;
        int h0 = mo.h[i];
        int hc = mo.n[h0];
        std::vector<int> hf;
        hf.push_back(h0);
        while (hc != h0)
        {
            hf.push_back(hc);
            hc = mo.n[hc];
        }
        Scalar area = 0;
        Scalar area1 = 0;
        for (int j = 0; j < hf.size(); j++)
        {
            area += u_o_out[h_group[hf[j]]] * v_o_out[h_group[hf[(j+1) % hf.size()]]] - v_o_out[h_group[hf[j]]] * u_o_out[h_group[hf[(j+1) % hf.size()]]];
            area1 += u_o[hf[j]] * v_o[hf[(j+1) % hf.size()]] - v_o[hf[j]] * u_o[hf[(j+1) % hf.size()]];
        }
        if (area <= 0)
        {
            spdlog::debug("overlay face {} flipped, area: {}, {}", i, area/2, area1/2);
        }
        for (int j = 0; j < hf.size() - 2; j++)
        {
            f_remap.push_back(i);
            FT_out.push_back(std::vector<int>{h_group[h0], h_group[hf[j + 1]], h_group[hf[j + 2]]});
            F_out.push_back(std::vector<int>{which_to_group[mo.to[h0]], which_to_group[mo.to[hf[j + 1]]], which_to_group[mo.to[hf[j + 2]]]});
        }
    }

    // after drop the second copy - update face map
    std::vector<int> _Fn_to_F(F_out.size(), -1);
    for(int i = 0; i < F_out.size(); i++)
         _Fn_to_F[i] = Fn_to_F[f_remap[i]];
    Fn_to_F = _Fn_to_F;

    return std::make_tuple(v3d_out, u_o_out, v_o_out, F_out, FT_out, Fn_to_F);

}


// New Interfaces with 4 choices

/**
 * 1(a) halfedge + lengths per halfedge for OverlayMesh
 * @param V dim #v*3 matrix, each row corresponds to mesh vertex coordinates
 * @param F dim #f*3 matrix, each row corresponds to three vertex ids of each facet
 * @param Theta_hat dim #v vector, each element is the prescribed angle sum at each vertex
 * @param alg_params, optional, algorithm parameters, for details check ConformalIdealDelaunayMapping.hh
 * @param ls_params, optional, line search parameters, for details check ConformalIdealDelaunayMapping.hh
 * @param stats_params, optional, statistic parameters, for details check ConformalIdealDelaunayMapping.hh
 * @return n, dim #he next array of the result half-edge structure
 * @return opp, dim #he opp array of the result half-edge structure
 * @return l, dim #he metric(length) of each halfedge
 */ 
 
template<typename Scalar>
std::tuple<
        std::vector<int>,                   // n
        std::vector<int>,                   // opp
        std::vector<Scalar>>                // l(metric)
conformal_metric_CL(const Eigen::MatrixXd &V,
                    const Eigen::MatrixXi &F,
                    const std::vector<Scalar> &Theta_hat,
                    std::shared_ptr<AlgorithmParameters> alg_params=nullptr,
                    std::shared_ptr<LineSearchParameters> ls_params=nullptr,
                    std::shared_ptr<StatsParameters> stats_params=nullptr)
{
    // get cones and bd
    std::vector<int> cones, bd;
    std::vector<bool> is_bd = igl::is_border_vertex(F);
    for (int i = 0; i < is_bd.size(); i++)
    {
        if (is_bd[i])
        {
            bd.push_back(i);
        }
    }
    bool do_trim = false;
    auto gb = count_genus_and_boundary(V, F);
    int n_genus = gb.first, n_bd = gb.second;
    if((n_genus >= 1 && n_bd != 0) || n_bd > 1){
        do_trim = true;
    }
    for (int i = 0; i < Theta_hat.size(); i++)
    {
        if ((!is_bd[i]) && abs(Theta_hat[i] -  2 * M_PI) > 1e-15)
        {
            cones.push_back(i);
        }
    }

    // do conformal_metric
    std::vector<int> pt_fids_placeholder;
    std::vector<Eigen::Matrix<double, 3, 1>> pt_bcs_placeholder;
    auto conformal_out = conformal_metric(V, F, Theta_hat, pt_fids_placeholder, pt_bcs_placeholder, alg_params, ls_params, stats_params);
    OverlayMesh<Scalar> mo = std::get<0>(conformal_out);
    std::vector<Scalar> u = std::get<1>(conformal_out);
    std::vector<int> vtx_reindex = std::get<4>(conformal_out);
    if(mo.bypass_overlay){
        spdlog::warn("overlay bypassed due to numerical issue or as instructed.");
        return std::make_tuple(std::vector<int>(), std::vector<int>(), std::vector<Scalar>());
    }
    std::vector<int> f_labels = get_overlay_face_labels(mo);

    // reindex cones and bd
    std::vector<int> vtx_reindex_rev(vtx_reindex.size());
    for (int i = 0; i < vtx_reindex.size(); i++)
    {
        vtx_reindex_rev[vtx_reindex[i]] = i;
    }
    for (int i = 0; i < cones.size(); i++)
    {
        cones[i] = vtx_reindex_rev[cones[i]];
    }
    for (int i = 0; i < bd.size(); i++)
    {
        bd[i] = vtx_reindex_rev[bd[i]];
    }

    int root = -1;
    if(alg_params->layout_root != -1){
        root = vtx_reindex_rev[alg_params->layout_root];
    }

    // get layout
    auto layout_res = get_layout(mo, u, bd, cones, do_trim, root);
    auto u_o = std::get<3>(layout_res);
    auto v_o = std::get<4>(layout_res);

    // get output connectivity and metric
    std::vector<int> n_new, opp_new;
    std::vector<Scalar> u_new, v_new;
    split_overlay(mo, f_labels, u_o, v_o, n_new, opp_new, u_new, v_new);
    std::vector<Scalar> l_new(n_new.size());
    for (int i = 0; i < n_new.size(); i++)
    {
        int i_prev = n_new[n_new[i]];
        l_new[i] = sqrt((u_new[i] - u_new[i_prev]) * (u_new[i] - u_new[i_prev]) + (v_new[i] - v_new[i_prev]) * (v_new[i] - v_new[i_prev]));
    }

    return std::make_tuple(n_new, opp_new, l_new);
}

/**
 * 1(b) (V,F) + lenghts per edge for OverlayMesh
 * @param V dim #v*3 matrix, each row corresponds to mesh vertex coordinates
 * @param F dim #f*3 matrix, each row corresponds to three vertex ids of each facet
 * @param Theta_hat dim #v vector, each element is the prescribed angle sum at each vertex
 * @param alg_params, optional, algorithm parameters, for details check ConformalIdealDelaunayMapping.hh
 * @param ls_params, optional, line search parameters, for details check ConformalIdealDelaunayMapping.hh
 * @param stats_params, optional, statistic parameters, for details check ConformalIdealDelaunayMapping.hh
 * @return V_out, #v'*3 vector, vertex coordinates
 * @return F_out, #f'*3 vector, each row corresponds to three vertex ids of each facet
 * @return l, #f'*3 vector, l[i][j] = length of edge (F[i][j], F[i][(j+1)%3])
 */ 
template<typename Scalar>
std::tuple<
        std::vector<std::vector<Scalar>>,       // V_out
        std::vector<std::vector<int>>,          // F_out
        std::vector<std::vector<Scalar>>>       // l(metric)
conformal_metric_VL(const Eigen::MatrixXd &V,
                    const Eigen::MatrixXi &F,
                    const std::vector<Scalar> &Theta_hat,
                    std::shared_ptr<AlgorithmParameters> alg_params=nullptr,
                    std::shared_ptr<LineSearchParameters> ls_params=nullptr,
                    std::shared_ptr<StatsParameters> stats_params=nullptr)
{
    // get cones and bd
    std::vector<int> cones, bd;
    std::vector<bool> is_bd = igl::is_border_vertex(F);
    for (int i = 0; i < is_bd.size(); i++)
    {
        if (is_bd[i])
        {
            bd.push_back(i);
        }
    }
    bool do_trim = false;
    auto gb = count_genus_and_boundary(V, F);
    int n_genus = gb.first, n_bd = gb.second;
    if((n_genus >= 1 && n_bd != 0) || n_bd > 1){
        do_trim = true;
    }
    for (int i = 0; i < Theta_hat.size(); i++)
    {
        if ((!is_bd[i]) && abs(Theta_hat[i] -  2 * M_PI) > 1e-15)
        {
            cones.push_back(i);
        }
    }

    // do conformal_metric
    std::vector<int> pt_fids_placeholder;
    std::vector<Eigen::Matrix<double, 3, 1>> pt_bcs_placeholder;
    auto conformal_out = conformal_metric(V, F, Theta_hat, pt_fids_placeholder, pt_bcs_placeholder, alg_params, ls_params, stats_params);
    OverlayMesh<Scalar> mo = std::get<0>(conformal_out);
    std::vector<Scalar> u = std::get<1>(conformal_out);
    std::vector<int> vtx_reindex = std::get<4>(conformal_out);
    auto V_overlay = std::get<5>(conformal_out);

    if(mo.bypass_overlay){
        spdlog::warn("overlay bypassed due to numerical issue or as instructed.");
        return std::make_tuple(std::vector<std::vector<Scalar>>(), std::vector<std::vector<int>>(), std::vector<std::vector<Scalar>>());
    }

    std::vector<int> f_labels = get_overlay_face_labels(mo);

    // reindex cones and bd
    std::vector<int> vtx_reindex_rev(vtx_reindex.size());
    for (int i = 0; i < vtx_reindex.size(); i++)
    {
        vtx_reindex_rev[vtx_reindex[i]] = i;
    }
    for (int i = 0; i < cones.size(); i++)
    {
        cones[i] = vtx_reindex_rev[cones[i]];
    }

    int root = -1;
    if(alg_params->layout_root != -1)
        root = vtx_reindex_rev[alg_params->layout_root];

    for (int i = 0; i < bd.size(); i++)
    {
        bd[i] = vtx_reindex_rev[bd[i]];
    }

    // get layout
    auto layout_res = get_layout(mo, u, bd, cones, do_trim, root);
    auto u_o = std::get<3>(layout_res);
    auto v_o = std::get<4>(layout_res);
    auto is_cut_o = std::get<5>(layout_res);

    // get output VF and metric
    auto FV_res = get_FV(mo, is_cut_o, V_overlay, u_o, v_o);
    auto v3d = std::get<0>(FV_res);
    auto u_o_out = std::get<1>(FV_res);
    auto v_o_out = std::get<2>(FV_res);
    auto F_out = std::get<3>(FV_res);
    std::vector<std::vector<Scalar>> v3d_out(v3d[0].size());
    for (int i = 0; i < v3d[0].size(); i++)
    {
        v3d_out[i].resize(3);
        for (int j = 0; j < 3; j++)
        {
            v3d_out[i][j] = v3d[j][i];
        }
    }

    std::vector<std::vector<Scalar>> l(F_out.size());
    for (int i = 0; i < F_out.size(); i++)
    {
        l[i].resize(3);
        for (int j = 0; j < 3; j++)
        {
            int v0 = F_out[i][j];
            int v1 = F_out[i][(j + 1) % 3];
            l[i][j] = sqrt( (u_o_out[v0] - u_o_out[v1]) * (u_o_out[v0] - u_o_out[v1]) +  (v_o_out[v0] - v_o_out[v1]) * (v_o_out[v0] - v_o_out[v1]) );
        }
    }

    return std::make_tuple(v3d_out, F_out, l);
}

/**
 * 2(a) halfedge + per corner layout
 * @param V dim #v*3 matrix, each row corresponds to mesh vertex coordinates
 * @param F dim #f*3 matrix, each row corresponds to three vertex ids of each facet
 * @param Theta_hat dim #v vector, each element is the prescribed angle sum at each vertex
 * @param alg_params, optional, algorithm parameters, for details check ConformalIdealDelaunayMapping.hh
 * @param ls_params, optional, line search parameters, for details check ConformalIdealDelaunayMapping.hh
 * @param stats_params, optional, statistic parameters, for details check ConformalIdealDelaunayMapping.hh
 * @return n, dim #he next array of the result half-edge structure
 * @return opp, dim #he opp array of the result half-edge structure
 * @return u, dim #he per corner u coordinates of the layout
 * @return v, dim #he per corner v coordinates of the layout
 */ 
template<typename Scalar>
std::tuple<
        std::vector<int>,                   // n
        std::vector<int>,                   // opp
        std::vector<Scalar>,                // layout u (per corner)
        std::vector<Scalar>>                // layout v (per corner) 
conformal_parametrization_CL(const Eigen::MatrixXd &V,
                    const Eigen::MatrixXi &F,
                    const std::vector<Scalar> &Theta_hat,
                    std::shared_ptr<AlgorithmParameters> alg_params=nullptr,
                    std::shared_ptr<LineSearchParameters> ls_params=nullptr,
                    std::shared_ptr<StatsParameters> stats_params=nullptr)
{
    // get cones and bd
    std::vector<int> cones, bd;
    std::vector<bool> is_bd = igl::is_border_vertex(F);
    for (int i = 0; i < is_bd.size(); i++)
    {
        if (is_bd[i])
        {
            bd.push_back(i);
        }
    }
    bool do_trim = false;
    auto gb = count_genus_and_boundary(V, F);
    int n_genus = gb.first, n_bd = gb.second;
    if((n_genus >= 1 && n_bd != 0) || n_bd > 1){
        do_trim = true;
    }
    for (int i = 0; i < Theta_hat.size(); i++)
    {
        if ((!is_bd[i]) && abs(Theta_hat[i] -  2 * M_PI) > 1e-15)
        {
            cones.push_back(i);
        }
    }

    // do conformal_metric
    std::vector<int> pt_fids_placeholder;
    std::vector<Eigen::Matrix<double, 3, 1>> pt_bcs_placeholder;
    auto conformal_out = conformal_metric(V, F, Theta_hat, pt_fids_placeholder, pt_bcs_placeholder, alg_params, ls_params, stats_params);
    OverlayMesh<Scalar> mo = std::get<0>(conformal_out);
    std::vector<Scalar> u = std::get<1>(conformal_out);
    std::vector<int> vtx_reindex = std::get<4>(conformal_out);
    if(mo.bypass_overlay){
        spdlog::warn("overlay bypassed due to numerical issue or as instructed.");
        return std::make_tuple(std::vector<int>(), std::vector<int>(), std::vector<Scalar>(), std::vector<Scalar>());
    }
    std::vector<int> f_labels = get_overlay_face_labels(mo);

    // reindex cones and bd
    std::vector<int> vtx_reindex_rev(vtx_reindex.size());
    for (int i = 0; i < vtx_reindex.size(); i++)
    {
        vtx_reindex_rev[vtx_reindex[i]] = i;
    }
    for (int i = 0; i < cones.size(); i++)
    {
        cones[i] = vtx_reindex_rev[cones[i]];
    }
    int root = -1;
    if(alg_params->layout_root != -1)
        root = vtx_reindex_rev[alg_params->layout_root];
    for (int i = 0; i < bd.size(); i++)
    {
        bd[i] = vtx_reindex_rev[bd[i]];
    }

    // get layout
    auto layout_res = get_layout(mo, u, bd, cones, do_trim, root);
    auto u_o = std::get<3>(layout_res);
    auto v_o = std::get<4>(layout_res);

    // get output connectivity and metric
    std::vector<int> n_new, opp_new;
    std::vector<Scalar> u_new, v_new;
    split_overlay(mo, f_labels, u_o, v_o, n_new, opp_new, u_new, v_new);

    return std::make_tuple(n_new, opp_new, u_new, v_new);
}


/**
 * 2(b) (V,F,u,v)
 * @param V dim #v*3 matrix, each row corresponds to mesh vertex coordinates
 * @param F dim #f*3 matrix, each row corresponds to three vertex ids of each facet
 * @param Theta_hat dim #v vector, each element is the prescribed angle sum at each vertex
 * @param alg_params, optional, algorithm parameters, for details check ConformalIdealDelaunayMapping.hh
 * @param ls_params, optional, line search parameters, for details check ConformalIdealDelaunayMapping.hh
 * @param stats_params, optional, statistic parameters, for details check ConformalIdealDelaunayMapping.hh
 * @return V_out, #v'*3 vector, vertex coordinates
 * @return F_out, #f'*3 vector, each row corresponds to three vertex ids of each facet
 * @return u, #v' vector, per vertex u coordinates of the layout
 * @return v, #v' vector, per vertex v coordinates of the layout
 */ 
template<typename Scalar>
std::tuple<
        std::vector<std::vector<Scalar>>,       // V_out
        std::vector<std::vector<int>>,          // F_out
        std::vector<Scalar>,                    // layout u (per vertex)
        std::vector<Scalar>,                    // layout v (per vertex)
        std::vector<std::vector<int>>,          // FT_out
        std::vector<int>>                       // Fn_to_F
conformal_parametrization_VL(const Eigen::MatrixXd &V,
                    const Eigen::MatrixXi &F,
                    const std::vector<Scalar> &Theta_hat,
                    std::shared_ptr<AlgorithmParameters> alg_params=nullptr,
                    std::shared_ptr<LineSearchParameters> ls_params=nullptr,
                    std::shared_ptr<StatsParameters> stats_params=nullptr)
{
    // get cones and bd
    std::vector<int> cones, bd;
    std::vector<bool> is_bd = igl::is_border_vertex(F);
    for (int i = 0; i < is_bd.size(); i++)
    {
        if (is_bd[i])
        {
            bd.push_back(i);
        }
    }
    bool do_trim = false;
    auto gb = count_genus_and_boundary(V, F);
    int n_genus = gb.first, n_bd = gb.second;
    if((n_genus >= 1 && n_bd != 0) || n_bd > 1){
        do_trim = true;
    }
    for (int i = 0; i < Theta_hat.size(); i++)
    {
        if ((!is_bd[i]) && abs(Theta_hat[i] -  2 * M_PI) > 1e-15)
        {
            cones.push_back(i);
        }
    }

    // do conformal_metric
    std::vector<int> pt_fids_placeholder;
    std::vector<Eigen::Matrix<double, 3, 1>> pt_bcs_placeholder;
    auto conformal_out = conformal_metric(V, F, Theta_hat, pt_fids_placeholder, pt_bcs_placeholder, alg_params, ls_params, stats_params);
    OverlayMesh<Scalar> mo = std::get<0>(conformal_out);
    std::vector<Scalar> u = std::get<1>(conformal_out);
    std::vector<int> vtx_reindex = std::get<4>(conformal_out);
    auto V_overlay = std::get<5>(conformal_out);

    if(mo.bypass_overlay){
        spdlog::warn("overlay bypassed due to numerical issue or as instructed.");
        return std::make_tuple(std::vector<std::vector<Scalar>>(), std::vector<std::vector<int>>(), std::vector<Scalar>(), std::vector<Scalar>(), std::vector<std::vector<int>>(), std::vector<int>());
    }

    std::vector<int> f_labels = get_overlay_face_labels(mo);

    // reindex cones and bd
    std::vector<int> vtx_reindex_rev(vtx_reindex.size());
    for (int i = 0; i < vtx_reindex.size(); i++)
    {
        vtx_reindex_rev[vtx_reindex[i]] = i;
    }
    for (int i = 0; i < cones.size(); i++)
    {
        cones[i] = vtx_reindex_rev[cones[i]];
    }

    int root = -1;
    if(alg_params->layout_root != -1)
        root = vtx_reindex_rev[alg_params->layout_root];

    for (int i = 0; i < bd.size(); i++)
    {
        bd[i] = vtx_reindex_rev[bd[i]];
    }
    spdlog::info("#bd_vt: {}", bd.size());
    spdlog::info("#cones: {}", cones.size());
    spdlog::info("vtx reindex size: {}", vtx_reindex.size());
    spdlog::info("mc.out size: {}", mo.cmesh().out.size());

    // get layout
    auto layout_res = get_layout(mo, u, bd, cones, do_trim, root);
    auto u_o = std::get<3>(layout_res);
    auto v_o = std::get<4>(layout_res);
    auto is_cut_o = std::get<5>(layout_res);

    // get output VF and metric
    auto FVFT_res = get_FV_FTVT(mo, is_cut_o, V_overlay, u_o, v_o);
    auto v3d = std::get<0>(FVFT_res); 
    auto u_o_out = std::get<1>(FVFT_res);
    auto v_o_out = std::get<2>(FVFT_res);
    auto F_out = std::get<3>(FVFT_res);
    auto FT_out = std::get<4>(FVFT_res);
    auto Fn_to_F = std::get<5>(FVFT_res);

    // v3d_out = v3d^T
    std::vector<std::vector<Scalar>> v3d_out(v3d[0].size());
    for (int i = 0; i < v3d[0].size(); i++)
    {
        v3d_out[i].resize(3);
        for (int j = 0; j < 3; j++)
        {
            v3d_out[i][j] = v3d[j][i];
        }
    }

    // reindex back
    auto u_o_out_copy = u_o_out;
    auto v_o_out_copy = v_o_out;
    auto v3d_out_copy = v3d_out;
    for (int i = 0; i < F_out.size(); i++)
    {
        for (int j = 0; j < 3; j++)
        {
            if (F_out[i][j] < vtx_reindex.size())
            {
                F_out[i][j] = vtx_reindex[F_out[i][j]];
            }
            if (FT_out[i][j] < vtx_reindex.size())
            {
                FT_out[i][j] = vtx_reindex[FT_out[i][j]];
            }
        }
    }
    for (int i = 0; i < vtx_reindex.size(); i++)
    {
        u_o_out[vtx_reindex[i]] = u_o_out_copy[i];
        v_o_out[vtx_reindex[i]] = v_o_out_copy[i];
        v3d_out[vtx_reindex[i]] = v3d_out_copy[i];
    }

    return std::make_tuple(v3d_out, F_out, u_o_out, v_o_out, FT_out, Fn_to_F);
}

/**
 * Helper function for save mesh data with texture coordinates as obj file
 * @param fname, the complete path and filename for writing obj file
 * @param V dim #v*3 matrix, each row corresponds to mesh vertex coordinates
 * @param F dim #f*3 matrix, each row corresponds to three vertex ids of each facet
 * @param u dim #f*3 scalar matrix, u-coordinates
 * @param v dim #f*3 scalar matrix, v-coordinates
 * @param Ft, dim #f*3 int matrix, stores the vertex index of texture coordinates for each face
 */ 
template <typename Scalar>               
void write_texture_obj(
    std::string fname, 
    const std::vector<std::vector<Scalar>>& V, 
    const std::vector<std::vector<int>>& F, 
    const std::vector<Scalar>& u, 
    const std::vector<Scalar>& v, 
    const std::vector<std::vector<int>>& Ft)
{
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> V_mat(V.size(), 3), uv(u.size(), 2), CN;
    Eigen::MatrixXi F_mat(F.size(), 3), FN, Ft_mat(Ft.size(), 3);
    for(int i = 0; i < u.size(); i++)
        uv.row(i) << u[i], v[i];
    for(int i = 0; i < V.size(); i++)
        V_mat.row(i) << V[i][0], V[i][1], V[i][2];
    for(int i = 0; i < F.size(); i++)
        F_mat.row(i) << F[i][0], F[i][1], F[i][2];
    for(int i = 0; i < Ft.size(); i++)
        Ft_mat.row(i) << Ft[i][0], Ft[i][1], Ft[i][2];
    
    igl::writeOBJ(fname, V_mat, F_mat, CN, FN, uv, Ft_mat);
}      

#endif
