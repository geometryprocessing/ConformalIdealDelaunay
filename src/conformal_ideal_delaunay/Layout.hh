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

#ifndef LAYOUT_HH
#define LAYOUT_HH

#include <queue>
#include <vector>
#include <set>
#include "OverlayMesh.hh"
#include <igl/writeOBJ.h>
#ifdef WITH_MPFR
#include <unsupported/Eigen/MPRealSupport>
#endif


// get the number of cut_edges touching to[h0]
int count_valence(const std::vector<int> &n, const std::vector<int> &opp, int h0, std::vector<bool> is_cut)
{
  int valence = 0;
  int hi = opp[n[h0]];
  if (is_cut[h0]) 
    valence = 1;
  while (hi != h0)
  {
    if (is_cut[hi])
      valence++;
    hi = opp[n[hi]];
  }
  return valence;
}
/**
* Given a metric defined by original edge lengths and scale factor u, do a bfs on dual graph of mesh or 
* using given cuts to singularities defined in is_cut_h to compute per-corner u, v coordinates
* 
* @param m, mesh data structure
* @param u, per-vertex scale factor 
* @param is_cut_h, (optional) pre-defined cut to singularities
* @param start_h, the first halfedge to be laid out, can be used to control the axis-alignment for the whole patch
* @return _u, size #h vector, the u-coordinate of the vertex that current halfedge is pointing to 
* @return _v, size #h vector, the v-coordinate of the vertex that current halfedge is pointing to 
* @return is_cut_h #h vector, mark whether the current halfedge is part of cut-to-singularity(when true)
*/
template <typename Scalar>
std::tuple<std::vector<Scalar>, std::vector<Scalar>, std::vector<bool>>
compute_layout(Mesh<Scalar> &m, const std::vector<Scalar> &u, std::vector<bool>& is_cut_h, int start_h = -1)
{

  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorX;

  auto _u = std::vector<Scalar>(m.n_halfedges(), 0.0);
  auto _v = std::vector<Scalar>(m.n_halfedges(), 0.0);

  bool cut_given = !is_cut_h.empty();
  auto cut_final = std::vector<bool>(m.n_halfedges(), false);
  auto is_cut_h_gen = std::vector<bool>(m.n_halfedges(), false);

  auto phi = std::vector<Scalar>(m.n_halfedges(), 0.0);
  auto xi = std::vector<Scalar>(m.n_halfedges(), 0.0);
  for (int i = 0; i < m.n_halfedges(); i++)
  {
    xi[i] = u[m.to[i]] - u[m.to[m.opp[i]]];
  }

  // set starting point - use a boundary edge
  int h = 0;
  if (start_h == -1)
  {
    for (int i = 0; i < m.n_halfedges(); i++)
    {
      if (m.f[i] != -1 && m.f[m.opp[i]] == -1)
        h = m.n[m.n[i]];
    }
  }
  else
  {
    assert(m.f[start_h] != -1);
    h = m.n[m.n[start_h]];
  }

  _u[h] = 0.0;
  _v[h] = 0.0;
  phi[h] = 0.0;
  h = m.n[h];
  assert(m.f[h] != -1);

  phi[h] = xi[h];
  _u[h] = m.l[m.e(h)] * exp(phi[h] / 2);
  _v[h] = 0.0;
  auto done = std::vector<bool>(m.n_faces(), false);

  // discard part 2
  for (int i = 0; i < done.size(); i++)
  {
    int hh = m.h[i];
    if (m.type[hh] == 2 && m.type[m.n[hh]] == 2 && m.type[m.n[m.n[hh]]] == 2)
    {
      done[i] = true;
    }
  }
  // set edge type 2 as cut
  for (int i = 0; i < is_cut_h.size(); i++)
  {
    if (m.type[i] == 2)
    {
      is_cut_h[i] = true;
    }
  }

  std::queue<int> Q;
  Q.push(h);
  done[m.f[h]] = true;

  auto perp = [](Eigen::Matrix<Scalar, 1, 2> a)
  {
    Eigen::Matrix<Scalar, 1, 2> b;
    b[0] = -a[1];
    b[1] = a[0];
    return b;
  };

  auto area_from_len = [](Scalar l1, Scalar l2, Scalar l3)
  {
    auto s = 0.5 * (l1 + l2 + l3);
    return sqrt(s * (s - l1) * (s - l2) * (s - l3));
  };

  auto square = [](Scalar x)
  { return x * x; };

  while (!Q.empty())
  {
    h = Q.front();
    Q.pop();
    int hn = m.n[h];
    int hp = m.n[hn];
    phi[hn] = phi[h] + xi[hn];
    Eigen::Matrix<Scalar, 1, 2> p1;
    p1[0] = _u[hp];
    p1[1] = _v[hp];
    Eigen::Matrix<Scalar, 1, 2> p2;
    p2[0] = _u[h];
    p2[1] = _v[h];
    assert(m.l[m.e(h)] != 0.0);
    Scalar l0 = Scalar(1.0);
    Scalar l1 = exp((phi[hn] - phi[hp]) / 2) * (m.l[m.e(hn)] / m.l[m.e(h)]);
    Scalar l2 = exp((phi[hn] - phi[h]) / 2) * (m.l[m.e(hp)] / m.l[m.e(h)]);
    Eigen::Matrix<Scalar, 1, 2> pn = p1 + (p2 - p1) * (1 + square(l2 / l0) - square(l1 / l0)) / 2 + perp(p2 - p1) * 2 * area_from_len(1.0, l1 / l0, l2 / l0);
    _u[hn] = pn[0];
    _v[hn] = pn[1];
    int hno = m.opp[hn];
    int hpo = m.opp[hp];
    int ho = m.opp[h];

    if (m.f[hno] != -1 && !done[m.f[hno]] && !(cut_given && is_cut_h[hn]))
    {
      done[m.f[hno]] = true;
      phi[hno] = phi[h];
      phi[m.n[m.n[hno]]] = phi[hn];
      _u[hno] = _u[h];
      _v[hno] = _v[h];
      _u[m.n[m.n[hno]]] = _u[hn];
      _v[m.n[m.n[hno]]] = _v[hn];
      Q.push(hno);
    }
    else
    {
      is_cut_h_gen[hn] = true;
      is_cut_h_gen[m.opp[hn]] = true;
    }

    if (m.f[hpo] != -1 && !done[m.f[hpo]] && !(cut_given && is_cut_h[hp]))
    {
      done[m.f[hpo]] = true;
      phi[hpo] = phi[hn];
      phi[m.n[m.n[hpo]]] = phi[hp];
      _u[hpo] = _u[hn];
      _v[hpo] = _v[hn];
      _u[m.n[m.n[hpo]]] = _u[hp];
      _v[m.n[m.n[hpo]]] = _v[hp];
      Q.push(hpo);
    }
    else
    {
      is_cut_h_gen[hp] = true;
      is_cut_h_gen[m.opp[hp]] = true;
    }

    if (m.f[ho] != -1 && !done[m.f[ho]] && !(cut_given && is_cut_h[ho]))
    {
      done[m.f[ho]] = true;
      phi[ho] = phi[hp];
      phi[m.n[m.n[ho]]] = phi[h];
      _u[ho] = _u[hp];
      _v[ho] = _v[hp];
      _u[m.n[m.n[ho]]] = _u[h];
      _v[m.n[m.n[ho]]] = _v[h];
      Q.push(ho);
    }
  }

  return std::make_tuple(_u, _v, is_cut_h_gen);
  
};

/**
 * @brief Given overlay mesh (doubled) and list of bd vertices and list of singularity ids, connect singularities to boundary vertices using shortest path
 * 
 * @param m_o, overlay mesh
 * @param f_labels, list of integer label marking which copy(1/2) of the double does the current face belong
 * @param bd, list of boundary vertex id
 * @param singularities, list of singularity vertex id
 * @param is_cut_o, (out) marked cut halfedges
 * @param root, (optional) index of a boundary vertex, when not -1, this will be the only intersection of the cut to singularity edges with boundary
 */
template <typename Scalar>
void connect_to_singularities(OverlayMesh<Scalar>& m_o, const std::vector<int>& f_labels, const std::vector<int>& bd, const std::vector<int>& singularities, std::vector<bool>& is_cut_o, int root = -1){
    
    int n_v = m_o.out.size();
    int n_e = m_o.n.size();

    spdlog::debug("n_v: {}, n_f: {}", m_o.out.size(), m_o.n_faces());
    std::vector<std::vector<int>> v2e(m_o.out.size(), std::vector<int>());
    for (int i = 0; i < m_o.n.size(); i++)
    {
        if (f_labels[m_o.f[i]] == 1)
        {
            v2e[m_o.v0(i)].push_back(i);
        }
    }

    std::vector<int> min_distance(n_v, n_e);
    std::vector<int> T(n_v, -1);
    std::set<std::pair<int, int>> vertex_queue;
    std::vector<bool> is_cone(n_v, false);
    std::vector<bool> is_border(n_v, false);
    for(int v: bd) is_border[v] = true;
    for(int v: singularities) is_cone[v] = true;

    // put boundary to queue
    if(root != -1){
      assert(std::find(bd.begin(), bd.end(), root) != bd.end() && "selected root not on boundary");
      spdlog::info("select root {} for layout", root);
      vertex_queue.insert(std::make_pair(0, root));
      min_distance[root] = 0;
    }else{
      for (int i = 0; i < bd.size(); i++)
      {
        vertex_queue.insert(std::make_pair(0, bd[i]));
        min_distance[bd[i]] = 0;
      }
    }

    spdlog::debug("start dijkstra");

    // do dijkstra
    int n_visited = 0;
    while (!vertex_queue.empty())
    {
      int dist_u = vertex_queue.begin()->first;
      int u = vertex_queue.begin()->second;
      // end earlier if all targets(singularities) are visited
      if (is_cone[u])
      {
        n_visited++;
        spdlog::debug("path to cone {}: len({})", u, dist_u);
      }
      if (n_visited == singularities.size())
        break;
      vertex_queue.erase(vertex_queue.begin());
      if (root != -1 && u != root && is_border[u]) continue;
      int h0 = m_o.out[u];
      if(f_labels[m_o.f[h0]] == 2){
        if(!v2e[u].empty()) // pick a type 1 edge if exist
            h0 = v2e[u][0];
      }

      int h = h0;
      do
      {
        if (m_o.edge_type[h] != ORIGINAL_EDGE) 
        {
          int v = m_o.to[h];
          int dist_v = dist_u + 1;
          // update queue
          if (min_distance[v] > dist_v)
          {
            vertex_queue.erase(std::make_pair(min_distance[v], v));
            min_distance[v] = dist_v;
            T[v] = h;
            vertex_queue.insert(std::make_pair(min_distance[v], v));
          }
        }
        h = m_o.next_out(h);
      } while (h != h0);
    }

    spdlog::info("dijsktra done, connected cones: {}/{}", n_visited, singularities.size());

    // get cut_to_sin
    is_cut_o = std::vector<bool>(n_e, false);
    for (int s : singularities)
    {
      int h = T[s];
      std::set<int> seq_v;
      while (h != -1)
      {
        is_cut_o[h] = true;
        is_cut_o[m_o.opp[h]] = true;
        h = T[m_o.v0(h)];
      }
    }
}

/**
 * @brief Helper function for layout overlay mesh. Given polygon mesh m and per-corner u/v assignment, triangulate the polygon faces
 * 
 * @param m, mesh structure, possibly with polygon faces
 * @param u, per corner assignment u values (e.g. produced by compute_layout)
 * @param v, per corner assignment v values (e.g. produced by compute_layout)
 * @param f_labels, list of integer label marking which copy(1/2) of the double does the current face belong
 */

template <typename Scalar>
void triangulate_polygon_mesh(Mesh<Scalar>& m, const std::vector<Scalar>& u, const std::vector<Scalar>& v, std::vector<int>& f_labels){
    int n_f0 = m.n_faces();
    spdlog::info("initial f size: {}", n_f0);
    spdlog::info("initial he size: {}", m.n.size());
    for(int f = 0; f < n_f0; f++){
        int n_f = m.n_faces();
        int h0 = m.h[f];
        int hc = h0;
        std::vector<int> hf;
        do{
            hf.push_back(hc);
            hc = m.n[hc];
        }while(h0 != hc);
        if(hf.size() == 3) continue;
        spdlog::debug("triangulate face {}, #e {}", f, hf.size());
        int N = hf.size();
        // new faces: N-3
        // new halfedges 2*(N-3)
        int n_new_f = N-3;
        int n_new_he = 2*(N-3);
        int n_he = m.n.size();
        m.n.resize(n_he   + n_new_he);
        m.to.resize(n_he  + n_new_he);
        m.opp.resize(n_he + n_new_he);
        m.l.resize(n_he + n_new_he);
        m.f.resize(n_he + n_new_he);
        m.h.resize(n_f + n_new_f);
        f_labels.resize(n_f + n_new_f);
        for(int k = 0; k < n_new_f; k++)
            f_labels[n_f+k] = f_labels[f];
        m.n[n_he] = hf[0];
        m.n[hf[1]] = n_he;
        m.opp[n_he] = n_he+1;
        m.to[n_he] = m.to[hf.back()];
        m.h[f] = n_he;
        m.f[n_he] = f;
        assert(hf.back() < m.to.size() && hf[0] < m.to.size());
        m.l[n_he] = sqrt((u[hf.back()]-u[hf[1]])*(u[hf.back()]-u[hf[1]]) + (v[hf.back()]-v[hf[1]])*(v[hf.back()]-v[hf[1]]));
        for(int k = 1; k < 2*(N-3); k++){
            if(k%2 == 0){
                m.n[n_he+k] = n_he+k-1;
                m.opp[n_he+k] = n_he+k+1;
                m.to[n_he+k] = m.to[hf.back()];
                m.l[n_he+k] = sqrt((u[m.n[m.n[n_he+k]]]-u[hf.back()])*(u[m.n[m.n[n_he+k]]]-u[hf.back()])+(v[m.n[m.n[n_he+k]]]-v[hf.back()])*(v[m.n[m.n[n_he+k]]]-v[hf.back()]));
                m.f[n_he+k] = n_f+k/2-1;
                m.h[n_f+k/2-1] = n_he+k;
            }else{
                m.n[n_he+k] = hf[(k-1)/2+2];
                if((k-1)/2+2 != hf.size()-2)
                    m.n[hf[(k-1)/2+2]] = n_he+k+1;
                m.opp[n_he+k] = n_he+k-1;
                m.to[n_he+k] = m.to[m.n[m.n[m.opp[n_he+k]]]];
                m.l[n_he+k] = sqrt((u[m.n[m.n[m.opp[n_he+k]]]]-u[hf.back()])*(u[m.n[m.n[m.opp[n_he+k]]]]-u[hf.back()]) + (v[m.n[m.n[m.opp[n_he+k]]]]-v[hf.back()])*(v[m.n[m.n[m.opp[n_he+k]]]]-v[hf.back()]));
                m.f[n_he+k] = n_f+(k+1)/2-1;
                m.h[n_f+(k+1)/2-1] = n_he+k;
                m.f[m.n[n_he+k]] = n_f+(k+1)/2-1;
            }                
        }
        m.n[hf.back()] = n_he + n_new_he - 1;
        m.f[hf.back()] = n_f + n_new_f - 1;

    }

    for(int f = 0; f < m.h.size(); f++){
        int n_f = m.n_faces();
        int h0 = m.h[f];
        int hc = h0;
        std::vector<int> hf;
        do{
            hf.push_back(hc);
            hc = m.n[hc];
            if(hf.size() > 3){
                spdlog::error("face {} has {} he!!!", f, hf.size());
                for(auto x: hf)
                    std::cout<<x<<" ";
                std::cout<<std::endl;
                exit(0);
            }
        }while(h0 != hc);
    }

}

/**
 * @brief Helper function to remove redundent cut edges from cutgraph, it recursively remove degree-1 cut edges as long as it's not touching singularities
 * 
 * @param m_o Overlay mesh structure
 * @param f_labels list of integer label marking to which copy(1/2) the double the current face belongs
 * @param singularities list of singularity vertex ids
 * @param is_cut (in/out) cut edge marks
 */
template <typename Scalar>
void trim_open_branch(OverlayMesh<Scalar>& m_o, std::vector<int>& f_labels, std::vector<int>& singularities, std::vector<bool>& is_cut){
  bool any_trimmed = true;
  int n_trimmed = 0;
  while (any_trimmed)
  {
    any_trimmed = false;
    for (int hi = 0; hi < m_o.n.size(); hi++)
    {
      if (!is_cut[hi] || f_labels[m_o.f[hi]] == 2) 
        continue;
      int v0 = m_o.to[hi];
      int v1 = m_o.to[m_o.opp[hi]];
      if (std::find(singularities.begin(), singularities.end(), v0) != singularities.end() || 
          std::find(singularities.begin(), singularities.end(), v1) != singularities.end())
        continue;
      if (count_valence(m_o.n, m_o.opp, hi, is_cut) == 1 || count_valence(m_o.n, m_o.opp, m_o.opp[hi], is_cut) == 1)
      {
        is_cut[hi] = false;
        is_cut[m_o.opp[hi]] = false;
        any_trimmed = true;
        n_trimmed++;
      }
    }
  }
  spdlog::info("#trimmed: {}", n_trimmed);
}

template <typename Scalar>
std::tuple<std::vector<Scalar>, std::vector<Scalar>, std::vector<bool>,
           std::vector<Scalar>, std::vector<Scalar>, std::vector<bool>> get_layout_x(OverlayMesh<Scalar> &m_o, const std::vector<Scalar> &u_vec, std::vector<int> bd, std::vector<int> singularities, bool do_trim = false, int root=-1)
{
  auto m = m_o.cmesh();
  m_o.garbage_collection();

  if (bd.size() != 0)
  {
    std::vector<int> f_type = get_overlay_face_labels(m_o);

    int n_v = m_o.out.size();
    int n_e = m_o.n.size();

    std::vector<int> min_distance(n_v, n_e);
    std::vector<int> T(n_v, -1);
    std::set<std::pair<int, int>> vertex_queue;

    // put boundary to queue
    if(root != -1){
      assert(std::find(bd.begin(), bd.end(), root) != bd.end() && "selected root not on boundary");
      spdlog::info("select root {} for layout", root);
      vertex_queue.insert(std::make_pair(0, root));
      min_distance[root] = 0;
    }else{
      for (int i = 0; i < bd.size(); i++)
      {
        vertex_queue.insert(std::make_pair(0, bd[i]));
        min_distance[bd[i]] = 0;
      }
    }

    // do dijkstra
    int n_visited = 0;
    while (!vertex_queue.empty())
    {
      int dist_u = vertex_queue.begin()->first;
      int u = vertex_queue.begin()->second;
      // end earlier if all targets(singularities) are visited
      if (std::find(singularities.begin(), singularities.end(), u) != singularities.end())
      {
        n_visited++;
        spdlog::debug("path to cone {}: len({})", u, dist_u);
      }
      if (n_visited == singularities.size())
        break;
      vertex_queue.erase(vertex_queue.begin());
      if (root != -1 && u != root && std::find(bd.begin(), bd.end(), u) != bd.end()) continue;
      int h0 = m_o.out[u];
      if (f_type[m_o.f[h0]] == 2)
      {
        for (int i = 0; i < m_o.n.size(); i++)
        {
          if (f_type[m_o.f[i]] == 1 && m_o.v0(i) == u && m_o.edge_type[i] == ORIGINAL_AND_CURRENT_EDGE)
          {
            h0 = i;
            break;
          }
        }
      }
      int h = h0;

      do
      {
        if (m_o.edge_type[h] == ORIGINAL_AND_CURRENT_EDGE && f_type[m_o.f[h]] == 1)
        {
          int v = m_o.to[h];
          int dist_v = dist_u + 1;
          // update queue
          if (min_distance[v] > dist_v)
          {
            vertex_queue.erase(std::make_pair(min_distance[v], v));
            min_distance[v] = dist_v;
            T[v] = h;
            vertex_queue.insert(std::make_pair(min_distance[v], v));
          }
        }
        h = m_o.next_out(h);
      } while (h != h0);
    }

    // get cut_to_sin
    std::vector<bool> is_cut_o(n_e, false);
    for (int s : singularities)
    {
      int h = T[s];
      while (h != -1)
      {
        is_cut_o[h] = true;
        is_cut_o[m_o.opp[h]] = true;
        h = T[m_o.v0(h)];
      }
    }

    std::vector<bool> is_cut(m.n.size(), false);
    for (int i = 0; i < is_cut_o.size(); i++)
    {
      if (is_cut_o[i])
      {
        if (m_o.edge_type[i] == ORIGINAL_EDGE)
        {
          spdlog::error("original cut");
        }
        else
        {
          is_cut[m_o.origin[i]] = true;
        }
      }
    }

    // get start_h
    int start_h = 0;
    while ((m.type[start_h] != 1 || m.type[m.opp[start_h]] != 2) && start_h < m.opp.size())
    {
      start_h++;
    }
    start_h = start_h % m.opp.size();
    spdlog::info("layout start_h: {}", start_h);

    std::vector<Scalar> u_scalar, v_scalar;
    auto layout_res = compute_layout(m, u_vec, is_cut, start_h);
    u_scalar = std::get<0>(layout_res);
    v_scalar = std::get<1>(layout_res);

    spdlog::info("check layout in current mesh.");
    for (int h = 0; h < m.n.size(); h++)
    {
      int h_prev = m.n[m.n[h]];
      int h_opp = m.opp[h];
      int h_opp_prev = m.n[m.n[h_opp]];

      if (m.type[h] == 2 && m.type[m.n[h]] == 2 && m.type[m.n[m.n[h]]] == 2)
        continue;
      if (m.type[h_opp] == 2 && m.type[m.n[h_opp]] == 2 && m.type[m.n[m.n[h_opp]]] == 2)
        continue;
      Scalar l1 = (u_scalar[h] - u_scalar[h_prev]) * (u_scalar[h] - u_scalar[h_prev]) + (v_scalar[h] - v_scalar[h_prev]) * (v_scalar[h] - v_scalar[h_prev]);
      Scalar l2 = (u_scalar[h_opp] - u_scalar[h_opp_prev]) * (u_scalar[h_opp] - u_scalar[h_opp_prev]) + (v_scalar[h_opp] - v_scalar[h_opp_prev]) * (v_scalar[h_opp] - v_scalar[h_opp_prev]);

      if (abs(l1 - l2) > 1e-8)
      {
        spdlog::debug("mismatched len for he pairs({}, {}): {}, {}, {}", h, h_opp, l1, l2, l1-l2);
      }
      if (is_cut[h] == false)
      {
        if ((abs(u_scalar[h] - u_scalar[h_opp_prev]) + abs(v_scalar[h] - v_scalar[h_opp_prev]) > 1e-8) || (abs(u_scalar[h_prev] - u_scalar[h_opp]) + abs(v_scalar[h_prev] - v_scalar[h_opp]) > 1e-8))
        {
          spdlog::debug("mismatch corner coordinates: to({}), to({})", h, h_opp);
          spdlog::debug("halfedge types: h{}({}), h{}({})", h, h_opp, int(m.type[h]), int(m.type[h_opp]));
          spdlog::debug("u({}): {}, v({}): {}", h_prev, u_scalar[h_prev], h_prev, v_scalar[h_prev]);
          spdlog::debug("u({}): {}, v({}): {}", h, u_scalar[h], h, v_scalar[h]);
          spdlog::debug("u({}): {}, v({}): {}", h_opp, u_scalar[h_opp], h_opp, v_scalar[h_opp]);
          spdlog::debug("u({}): {}, v({}): {}", h_opp_prev, u_scalar[h_opp_prev], h_opp_prev, v_scalar[h_opp_prev]);
        }
      }
    }

    Eigen::Matrix<Scalar, -1, 1> u_eig;
    u_eig.resize(u_vec.size());
    for (int i = 0; i < u_vec.size(); i++)
    {
      u_eig(i) = u_vec[i];
    }

    m_o.bc_eq_to_scaled(m.n, m.to, m.l, u_eig);

    auto u_o = m_o.interpolate_along_c_bc(m.n, m.f, u_scalar);
    auto v_o = m_o.interpolate_along_c_bc(m.n, m.f, v_scalar);

    // mark boundary as cut
    auto f_labels = get_overlay_face_labels(m_o);
    for (int i = 0; i < is_cut_o.size(); i++)
    {
        if (f_labels[m_o.f[i]] != f_labels[m_o.f[m_o.opp[i]]])
        {
            is_cut_o[i] = true;
        }
    }
    if (do_trim)
    {
      is_cut = std::get<2>(layout_res);
      is_cut_o = m_o.interpolate_is_cut_h(is_cut);
      // trim_open_branch(m_o, f_labels, singularities, is_cut_o);
    } // end of do trim
    return std::make_tuple(u_scalar,v_scalar,is_cut,u_o,v_o,is_cut_o);
  }
  else // closed mesh
  {
    std::vector<bool> is_cut;
    auto layout_res = compute_layout(m, u_vec, is_cut);
    std::vector<Scalar> u_scalar = std::get<0>(layout_res);
    std::vector<Scalar> v_scalar = std::get<1>(layout_res);
    is_cut = std::get<2>(layout_res);

    Eigen::Matrix<Scalar, -1, 1> u_eig;
    u_eig.resize(u_vec.size());
    for (int i = 0; i < u_vec.size(); i++)
    {
      u_eig(i) = u_vec[i];
    }
    m_o.bc_eq_to_scaled(m.n, m.to, m.l, u_eig);

    auto u_o = m_o.interpolate_along_c_bc(m.n, m.f, u_scalar);
    auto v_o = m_o.interpolate_along_c_bc(m.n, m.f, v_scalar);

    // trim cut
    bool any_trimmed = true;
    while (any_trimmed)
    {
      any_trimmed = false;
      for (int hi = 0; hi < m.n.size(); hi++)
      {
        if (!is_cut[hi]) 
          continue;
        int v0 = m.to[hi];
        int v1 = m.to[m.opp[hi]];
        if (std::find(singularities.begin(), singularities.end(), v0) != singularities.end() || std::find(singularities.begin(), singularities.end(), v1) != singularities.end())
          continue;
        if (count_valence(m.n, m.opp, hi, is_cut) == 1 || count_valence(m.n, m.opp, m.opp[hi], is_cut) == 1)
        {
          is_cut[hi] = false;
          is_cut[m.opp[hi]] = false;
          any_trimmed = true;
        }
      }
    }
    auto is_cut_o = m_o.interpolate_is_cut_h(is_cut);
  
    return std::make_tuple(u_scalar,v_scalar,is_cut,u_o,v_o,is_cut_o);

  }
  
};

/**
 * @brief Given overlay mesh with associated flat metric compute the layout
 * 
 * @tparam Scalar double/mpfr::mpreal
 * @param m_o, overlay mesh
 * @param u_vec, per-vertex scale factor
 * @param bd, list of boundary vertex ids
 * @param singularities, list of singularity vertex ids
 * @param root (optional) index of a boundary vertex, when not -1, this will be the only intersection of the cut to singularity edges with boundary
 * @return _u_c, _v_c, is_cut_c (per-corner u/v assignment of current mesh and marked cut edges) 
 *         _u_o, _v_o, is_cut_h (per-corner u/v assignment of overlay mesh and marked cut edges)
 */
template <typename Scalar>
std::tuple<std::vector<Scalar>, std::vector<Scalar>, std::vector<bool>,
           std::vector<Scalar>, std::vector<Scalar>, std::vector<bool>> get_layout(OverlayMesh<Scalar> &m_o, const std::vector<Scalar> &u_vec, std::vector<int> bd, std::vector<int> singularities, bool do_trim = false, int root=-1)
{
    
  auto f_labels = get_overlay_face_labels(m_o);
  
  // layout the current mesh with arbitrary cut
  std::vector<bool> _is_cut_place_holder;
  auto mc = m_o.cmesh();
  m_o.garbage_collection();

  mc.type = std::vector<char>(mc.n_halfedges(), 0);
  auto layout_res = compute_layout(mc, u_vec, _is_cut_place_holder, 0);
  auto _u_c = std::get<0>(layout_res);
  auto _v_c = std::get<1>(layout_res);
  auto is_cut_c = std::get<2>(layout_res);
  std::vector<Scalar> _u_o, _v_o;
  std::vector<bool> is_cut_o;

  Eigen::Matrix<Scalar, -1, 1> u_eig;
  u_eig.resize(u_vec.size());
  for (int i = 0; i < u_vec.size(); i++)
  {
    u_eig(i) = u_vec[i];
  }

  m_o.bc_eq_to_scaled(mc.n, mc.to, mc.l, u_eig);

  auto u_o = m_o.interpolate_along_c_bc(mc.n, mc.f, _u_c);
  auto v_o = m_o.interpolate_along_c_bc(mc.n, mc.f, _v_c);
  spdlog::info("Interpolate on overlay mesh done.");

  if(!bd.empty()){ 
    // compute edge lengths of overlay mesh and triangulate it
    Mesh<Scalar> m;
    m.n = m_o.n;
    m.opp = m_o.opp;
    m.f = m_o.f;
    m.h = m_o.h;
    m.out = m_o.out;
    m.to = m_o.to;
    m.l = std::vector<Scalar>(m.n.size(), 0.0);
    for(int i = 0; i < m.n.size(); i++){
        int h0 = i; 
        int h1 = h0;
        do{
            if(m.n[h1] == h0)
                break;
            h1 = m.n[h1];
        }while(h0 != h1);
        if(m.to[m.opp[h0]] != m.to[h1]){
            spdlog::error("h0 h1 picked wrong.");
            exit(0);
        }
        m.l[h0] = sqrt((u_o[h0]-u_o[h1])*(u_o[h0]-u_o[h1]) + (v_o[h0]-v_o[h1])*(v_o[h0]-v_o[h1]));
    }
    triangulate_polygon_mesh(m, u_o, v_o, f_labels);

    m.type = std::vector<char>(m.n.size(), 0);
    m.type_input = m.type;
    m.R = std::vector<int>(m.n.size(), 0);
    m.v_rep = range(0, m.out.size());
    m.Th_hat = std::vector<Scalar>(m.out.size(), 0.0);
    
    // try to connect to singularties again with overlay mesh edges
    spdlog::info("try to connect to singularities using a tree rooted at root");
  
    OverlayMesh<Scalar> m_o_tri(m);
    for(int i = m_o.n.size(); i < m_o_tri.n.size(); i++){
        m_o_tri.edge_type[i] = ORIGINAL_EDGE; // make sure do not use the new diagonal
    }
    spdlog::info("root = {}", root);
    connect_to_singularities(m_o_tri, f_labels, bd, singularities, is_cut_o, root);
    
    int start_h = 0;
    for(int i = 0; i < m_o_tri.n.size(); i++){
        if(f_labels[m_o_tri.f[i]] == 1 && f_labels[m_o_tri.f[m_o_tri.opp[i]]] == 2){
            start_h = i; break;
        }
    }
    spdlog::info("selected start h: {}, left: {}, right: {}", start_h, f_labels[m_o_tri.f[start_h]], f_labels[m_o_tri.f[m_o_tri.opp[start_h]]]);

    // sanity check for the input of compute layout
    // - opposite halfedges should have same edge lenghts (up to numerical error)
    // - all halfedges that belongs to a face with type 1 should have non-zero edge lengths
    for(int i = 0; i < m_o_tri.n.size(); i++){
        int h0 = i, h1 = m_o_tri.opp[h0];
        int ft0 = f_labels[m_o_tri.f[h0]];
        int ft1 = f_labels[m_o_tri.f[h1]];
        if(std::abs<Scalar>(m_o_tri._m.l[h0]-m_o_tri._m.l[h1]) > 1e-8 && ft0 == ft1 && ft0 == 1){
            spdlog::error("halfedge lengths mismatch, {}: {}, {}: {}; {}/{}", h0, m_o_tri._m.l[h0], h1, m_o_tri._m.l[h1], ft0, ft1);
        }
        int f0 = m_o_tri.f[h0];
        int f1 = m_o_tri.f[h1];
        if(f_labels[f0] == 1 && m_o_tri._m.l[h0] == 0)
            spdlog::error("copy 1 has zero edge at {}, f{}", h0, f0);
        if(f_labels[f1] == 1 && m_o_tri._m.l[h1] == 0)
            spdlog::error("copy 1 has zero edge at {}, f{}", h1, f1);
    }
    spdlog::info("sanity check done.");

    // mark boundary as cut
    for (int i = 0; i < is_cut_o.size(); i++)
    {
        if (f_labels[m_o_tri.f[i]] != f_labels[m_o_tri.f[m_o_tri.opp[i]]])
        {
            is_cut_o[i] = true;
        }
    }

    // now directly do layout on overlay mesh
    for(int f = 0; f < f_labels.size(); f++){
        int h0 = m_o_tri.h[f];
        int h1 = m_o_tri.n[h0];
        int h2 = m_o_tri.n[h1];
        m_o_tri._m.type[h0] = f_labels[f];
        m_o_tri._m.type[h1] = f_labels[f];
        m_o_tri._m.type[h2] = f_labels[f];
    }

    // get output connectivity and metric
    std::vector<Scalar> phi(m_o_tri.out.size(), 0.0);
    auto overlay_layout_res = compute_layout(m_o_tri._m, phi, is_cut_o, start_h);
    _u_o = std::get<0>(overlay_layout_res);
    _v_o = std::get<1>(overlay_layout_res);

    // recursively remove degree-1 edges unless it's connected to a singularity
    if (do_trim){
      is_cut_o = std::get<2>(overlay_layout_res);
      Eigen::MatrixXd debug_uv(m_o_tri.h.size()*3, 3);
      Eigen::MatrixXi debug_Fuv(m_o_tri.h.size(), 3);
      for(int f = 0; f < m_o_tri.h.size(); f++){
        int h0 = m_o_tri.h[f];
        int h1 = m_o_tri.n[h0];
        int h2 = m_o_tri.n[h1];
        debug_uv.row(f*3) <<   double(_u_o[h0]), double(_v_o[h0]), 0;
        debug_uv.row(f*3+1) << double(_u_o[h1]), double(_v_o[h1]), 0;
        debug_uv.row(f*3+2) << double(_u_o[h2]), double(_v_o[h2]), 0;
        debug_Fuv.row(f) << f*3, f*3+1, f*3+2;
      }
      igl::writeOBJ("debug_uv.obj", debug_uv, debug_Fuv);
      // trim_open_branch(m_o_tri, f_labels, singularities, is_cut_o);
    }

    _u_o.resize(m_o.n.size());
    _v_o.resize(m_o.n.size());
    is_cut_o.resize(m_o.n.size()); 

  }else{
    // for closed mesh directly copy interpolated results
    _u_o = u_o;
    _v_o = v_o;
    is_cut_o = m_o.interpolate_is_cut_h(is_cut_c);
    if (do_trim)
      trim_open_branch(m_o, f_labels, singularities, is_cut_o);
  }

  return std::make_tuple(_u_c, _v_c, is_cut_c, _u_o, _v_o, is_cut_o);
  
}


#endif
