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

template <typename Scalar>
std::tuple<std::vector<Scalar>, std::vector<Scalar>, std::vector<bool>,
           std::vector<Scalar>, std::vector<Scalar>, std::vector<bool>> get_layout(OverlayMesh<Scalar> &m_o, const std::vector<Scalar> &u_vec, std::vector<int> bd, std::vector<int> singularities, bool do_trim = false, int root=-1)
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
          if (f_type[m_o.f[i]] == 1 && m_o.v0(i) == u)
          {
            h0 = i;
            break;
          }
        }
      }
      int h = h0;

      do
      {
        if (m_o.edge_type[h] != ORIGINAL_EDGE && f_type[m_o.f[h]] == 1)
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
      bool any_trimmed = true;
      while (any_trimmed)
      {
        any_trimmed = false;
        for (int hi = 0; hi < m_o.n.size(); hi++)
        {
          if (!is_cut_o[hi] || f_labels[m_o.f[hi]] == 2) 
            continue;
          int v0 = m_o.to[hi];
          int v1 = m_o.to[m_o.opp[hi]];
          if (std::find(singularities.begin(), singularities.end(), v0) != singularities.end() || std::find(singularities.begin(), singularities.end(), v1) != singularities.end())
            continue;
          if (count_valence(m_o.n, m_o.opp, hi, is_cut_o) == 1 || count_valence(m_o.n, m_o.opp, m_o.opp[hi], is_cut_o) == 1)
          {
            is_cut_o[hi] = false;
            is_cut_o[m_o.opp[hi]] = false;
            any_trimmed = true;
          }
        }
      }
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

#endif
