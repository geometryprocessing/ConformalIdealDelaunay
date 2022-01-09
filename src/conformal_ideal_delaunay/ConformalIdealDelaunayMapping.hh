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

#ifndef CONFORMAL_IDEAL_DELAUNAY_MAPPING_HH
#define CONFORMAL_IDEAL_DELAUNAY_MAPPING_HH

#include <set>
#include <iomanip>
#include <fstream>
#include <ctime>
#include <Eigen/Sparse>
#include "Angle.hh"
#include "Claussen.hh"
#include "OverlayMesh.hh"

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

using namespace OverlayProblem;

struct DelaunayStats {
// Delaunay
  int n_flips = 0, n_flips_s = 0, n_flips_t = 0, n_flips_q = 0, n_flips_12 = 0, n_nde = -1;
   bool flip_count = false;          // when true: write out stats for different kinds of flips
   std::vector<int> flip_seq;
};

template <typename Scalar>
struct SolveStats { 
  int n_solves = 0, n_g = 0, n_checks = 0;
  Scalar cetm_energy = 0;
};

struct StatsParameters{
  bool flip_count = false;      // when true: collect stats on different types of edge flips
  std::string name = "";        // name of the model that's been tested - for logging purpose
  std::string output_dir = "";  // directory name for genearting all stats
  bool error_log = false;       // when true: write out per-newton iterations stats
  bool print_summary = false;   // when true: add final stats of optimization to summary file
  int log_level = 2;            // controlling detail of console logging
};

struct LineSearchParameters { 
  double c1 = 1e-4;                  // c1 for armijo condition
  double c2 = 0.9;                   // c2 for curvature condition
  bool energy_samples = false;       // This boolean is only used for generating figure 4 in paper
  bool energy_cond = false;          // when true: use energy decrease as line search stop criterion
  bool do_reduction = false;         // when true: reduce step, if the components of descent direction vary too much 
  double descent_dir_max_variation = 1e-10; // threshold for descent direction component max difference to decrease step
  bool do_grad_norm_decrease = true; // when true: require gradient norm to decrease at each iteration
  double bound_norm_thres = 1e-10;   // threshold to drop gradient decrease requirement when step lambda is below this
  double lambda0 = 1.0;              // starting lambda value for the line search, normally 1
  bool reset_lambda = true;          // when true: start with lambda = lambda0 for each newton iteration; if false, start with lambda from the previous 
};

struct AlgorithmParameters {
  int MPFR_PREC = 100;           // precision if done in multiprecision
  bool initial_ptolemy = false;  // when true: use ptolemey flips for the first MakeDelaunay  Do we really need this?
  // termination
  double error_eps = 0;          // max angle error tolerance, terminate if below
  double min_lambda = 1e-16;     // terminate if lambda drops below this threshold
  double newton_decr_thres = 0;  // terminate if the newton decrement is above this threshold (it is negative)
  int max_itr = 500;             // upper bound for newton iterations
  bool bypass_overlay = false;   // avoid overlay computation
  int layout_root = -1;          // select vertex on boundary as root for constructing spanning tree connecting cones
 };

// Scalar: a floating point type, either double or MPFR
template <typename Scalar>
class ConformalIdealDelaunay
{
public:
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorX;

  /**
   * Interior angle and its cotangent computed for the whole mesh given decorated per-vertex u values,
   * the angles are computed via rescaled conformal edge lengths.
   * 
   * @param m Mesh data structure
   * @param u vector of Scalar size equal to number of vertices of mesh, the per-vertex logarithmic scale factors
   * @param alpha vector of Scalar size equal to number of halfedges of mesh,
   *              ith entry corresponds to the interior angle opposite to
   *              ith halfedge.
   * @param cot_alpha vector of Scalar size equal to number of halfedges of mesh,
   *              e.g. ith entry corresponds to the cotangent of interior angle opposite to
   *              ith halfedge.
   * @return void
   */
  static void ComputeAngles(const Mesh<Scalar>& m, const VectorX& u, VectorX& alpha, VectorX& cot_alpha)
  {
    
    alpha.setZero(m.n_halfedges());
    cot_alpha.setZero(m.n_halfedges());

    const Scalar cot_infty = Scalar(1e10);

    Scalar pi;
#ifdef WITH_MPFR
    if (std::is_same<Scalar, mpfr::mpreal>::value)
      pi = Scalar(mpfr::const_pi());
    else
      pi = Scalar(M_PI);
#else
      pi = Scalar(M_PI);
#endif

#pragma omp parallel for
    for (int f = 0; f < m.n_faces(); f++)
    {
      int hi = m.h[f];
      int hj = m.n[hi];
      int hk = m.n[hj];
      int i = m.v_rep[m.to[hj]];
      int j = m.v_rep[m.to[hk]];
      int k = m.v_rep[m.to[hi]];
      Scalar ui = u[i];
      Scalar uj = u[j];
      Scalar uk = u[k];
      Scalar uijk_avg = (ui + uj + uk)/3.0; // Scale lengths for numerical stability
      Scalar li = ell(m.l[m.e(hi)], uj, uk, uijk_avg);
      Scalar lj = ell(m.l[m.e(hj)], uk, ui, uijk_avg);
      Scalar lk = ell(m.l[m.e(hk)], ui, uj, uijk_avg);
      // (following "A Cotangent Laplacian for Images as Surfaces")
      Scalar s = (li + lj + lk) / 2.0;
      Scalar Aijk4 = 4.0 * sqrt(std::max<Scalar>(s * (s - li) * (s - lj) * (s - lk), 0.0));
      Scalar Ijk = (-li * li + lj * lj + lk * lk);
      Scalar iJk = (li * li - lj * lj + lk * lk);
      Scalar ijK = (li * li + lj * lj - lk * lk);
      cot_alpha[hi] = Aijk4 == 0.0 ? copysign(cot_infty, Ijk) : (Ijk / Aijk4);
      cot_alpha[hj] = Aijk4 == 0.0 ? copysign(cot_infty, iJk) : (iJk / Aijk4);
      cot_alpha[hk] = Aijk4 == 0.0 ? copysign(cot_infty, ijK) : (ijK / Aijk4);

#define USE_ACOS
#ifdef USE_ACOS
      alpha[hi] = acos(std::min<Scalar>(std::max<Scalar>(Ijk / (2.0 * lj * lk), -1.0), 1.0));
      alpha[hj] = acos(std::min<Scalar>(std::max<Scalar>(iJk / (2.0 * lk * li), -1.0), 1.0));
      alpha[hk] = acos(std::min<Scalar>(std::max<Scalar>(ijK / (2.0 * li * lj), -1.0), 1.0));
#else
      // atan2 is prefered for stability
      alpha[hi] = 0.0, alpha[hj] = 0.0, alpha[hk] = 0.0;
      // li: l12, lj: l23, lk: l31
      Scalar l12 = li, l23 = lj, l31 = lk;
      const Scalar t31 = +l12+l23-l31,
                   t23 = +l12-l23+l31,
                   t12 = -l12+l23+l31;
      // valid triangle
      if( t31 > 0 && t23 > 0 && t12 > 0 ){
        const Scalar l123 = l12+l23+l31;
        const Scalar denom = sqrt(t12*t23*t31*l123);
        alpha[hj] = 2*atan2(t12*t31,denom); // a1 l23
        alpha[hk] = 2*atan2(t23*t12,denom); // a2 l31
        alpha[hi] = 2*atan2(t31*t23,denom); // a3 l12
      }else if( t31 <= 0 ) alpha[hk] = pi;
       else if( t23 <= 0 ) alpha[hj] = pi;
       else if( t12 <= 0 ) alpha[hi] = pi;
       else alpha[hj] = pi;
#endif
    }
  }
  
  /**
   * Milnor’s Lobachevsky function, see appendix A in http://www.multires.caltech.edu/pubs/ConfEquiv.pdf
   */
  static Scalar Lob(const Scalar angle)
  {
    return 0 <= angle && angle <= M_PI ? claussen(double(2 * angle)) / 2 : 0;
  }

  /**
   * Compute angle sums at each vertex of given mesh according to per-corner angles
   * 
   * @param m Mesh data structure
   * @param alpha vector of Scalar size equal to number of halfedges of mesh,
   *              ith entry corresponds to the interior angle opposite to
   *              ith halfedge, e.g. the one computed by function `ComputeAngles`.
   * @return VectorX vector of Scalar size equal to number of vertices of mesh,
   *         each entry is the total angle sum at each vertex.
   */
  static VectorX Theta(const Mesh<Scalar>& m, const VectorX& alpha)
  {
    VectorX t(m.n_ind_vertices());
    t.setZero();
    for (int h = 0; h < m.n_halfedges(); h++)
    {
      t[m.v_rep[m.to[m.n[h]]]] += alpha[h];
    }
    return t;
  }

  /**
   * Compute conformal-equivalence-energy (see https://cims.nyu.edu/gcl/papers/2021-Conformal.pdf section 4) 
   * of a given mesh with per-vertex logarithmic scale factors
   * 
   * @param m Mesh data structure
   * @param angles vector of Scalar with size equal to number of halfedges, produced by ComputeAngles function
   * @param u vector of Scalar size equal to number of vertices of mesh, the per-vertex logarithmic scale factors
   * @return Scalar Energy of the mesh with given conformal metric
   */
  static Scalar ConformalEquivalenceEnergy(Mesh<Scalar> &m, const VectorX& angles, const VectorX &u)
  {

    auto func_f = [](
      const Scalar l12, const Scalar l23, const Scalar l31,  
      const Scalar u1,  const Scalar u2,  const Scalar u3, 
      const Scalar a1, const Scalar a2, const Scalar a3
    ){
      // h1->hi, h2->hj, h3->hk
      Scalar s12 = u1 + u2 - 2 * u3;
      Scalar s23 = u2 + u3 - 2 * u1;
      Scalar s31 = u3 + u1 - 2 * u2;
      Scalar lt12 = l12 * exp(1.0 / 6.0 * s12);
      Scalar lt23 = l23 * exp(1.0 / 6.0 * s23);
      Scalar lt31 = l31 * exp(1.0 / 6.0 * s31);
      Scalar lambda12 = 2 * log(l12);
      Scalar lambda23 = 2 * log(l23);
      Scalar lambda31 = 2 * log(l31);
      Scalar lambdat12 = lambda12 + u1 + u2;
      Scalar lambdat23 = lambda23 + u2 + u3;
      Scalar lambdat31 = lambda31 + u3 + u1;
      Scalar T1 = a1 * lambdat23 + a2 * lambdat31 + a3 * lambdat12;
      Scalar T2 = Lob(a1) + Lob(a2) + Lob(a3);
      return 0.5 * T1 + T2;
    };

    Scalar E = 0;

    // first part of the energy on faces
    Scalar total_f = 0.0;
    for (int _f = 0; _f < m.n_faces(); _f++)
    {
      int h1 = m.h[_f], h2 = m.n[h1], h3 = m.n[h2];
      int v1 = m.v_rep[m.to[h3]], v2 = m.v_rep[m.to[h1]], v3 = m.v_rep[m.to[h2]];
      Scalar l12 = m.l[m.e(h1)], l23 = m.l[m.e(h2)], l31 = m.l[m.e(h3)];
      Scalar u2 = u(m.v_rep[m.to[h1]]), u3 = u(m.v_rep[m.to[h2]]), u1 = u(m.v_rep[m.to[h3]]);
      Scalar val_f = func_f(l12, l23, l31, u1, u2, u3, angles[h2], angles[h3], angles[h1]);
      Scalar td_lambda_h1 = 2 * log(m.l[m.e(h1)]) + u[m.v_rep[m.to[h1]]] + u[m.v_rep[m.to[m.opp[h1]]]];
      Scalar td_lambda_h2 = 2 * log(m.l[m.e(h2)]) + u[m.v_rep[m.to[h2]]] + u[m.v_rep[m.to[m.opp[h2]]]];
      Scalar td_lambda_h3 = 2 * log(m.l[m.e(h3)]) + u[m.v_rep[m.to[h3]]] + u[m.v_rep[m.to[m.opp[h3]]]];
      auto e_tri = val_f - (M_PI / 4) * (td_lambda_h1 + td_lambda_h2 + td_lambda_h3);
      E += e_tri;
      total_f += val_f;
    }

    // second part of the energy on vertices
    Scalar Ex = 0;
    for (int _v = 0; _v < m.n_ind_vertices(); _v++)
      Ex += m.Th_hat[_v] * u(_v);

    E = E + 0.5 * Ex;

    return E;
  }

  /**
   * Compute the gradient of conformal-equivalence-energy, which is equal to the per-vertex angle defects 
   * 
   * @param m Mesh data structure
   * @param angles vector of Scalar with size equal to number of halfedges, produced by ComputeAngles function
   * @param g vector of size equal to number of vertices, the gradient.
   * @param solve_stats struct collecting info for solvings through out the algorithm
   * @return void
   */
  static void Gradient(const Mesh<Scalar>& m, const VectorX& angles, VectorX& g, SolveStats<Scalar>& solve_stats)
  {
    solve_stats.n_g++;
    g.setZero(m.n_ind_vertices());
    auto angle_sums = Theta(m, angles);
    for(int i = 0; i < g.rows(); i++)
      g[i] = m.Th_hat[i] - angle_sums(i);
  }

  /**
   * Compute the Hessian of conformal-equivalence-energy, which is the cotangent laplacian.
   * 
   * @param m Mesh data structure
   * @param cot_alpha vector of Scalar with size equal to number of halfedges, produced by ComputeAngles function
   * @param H (output), Sparse Matrix with size #v*#v.
   * @return void
   */
  static void Hessian(const Mesh<Scalar>& m, const VectorX& cot_alpha, Eigen::SparseMatrix<Scalar>& H)
  {
    H.resize(m.n_ind_vertices(), m.n_ind_vertices());
    typedef Eigen::Triplet<Scalar> Trip;
    std::vector<Trip> trips;
    trips.clear();
    trips.resize(m.n_halfedges() * 2);
#pragma omp parallel for
    for (int h = 0; h < m.n_halfedges(); h++)
    {
      int v0 = m.v_rep[m.v0(h)];
      int v1 = m.v_rep[m.v1(h)];
      Scalar w = (cot_alpha[h] + cot_alpha[m.opp[h]]) / 2;
      trips[h * 2] = Trip(v0, v1, -w);
      trips[h * 2 + 1] = Trip(v0, v0, w);
    }

    H.setFromTriplets(trips.begin(), trips.end());
  }

  /**
   * Given original edge length and two scale factors defined on two endpoints, compute the rescaled edge lengths.
   * 
   * @param l, Scalar, original edge length
   * @param u0, Scalar, first scale factor 
   * @param u1, Scalar, second scale factor
   * @param offset, Scalar,  a common factor to be subtracted from the total scale, added for numerical stability.
   * @return Scalar rescaled edge length
   */
  static Scalar ell(Scalar l, Scalar u0, Scalar u1, Scalar offset = 0)
  {
    return l * exp((u0 + u1) / 2 - offset);
  }

  /**
   * Predicate, checking whether the two neighboring triangles of given halfedge in the mesh 
   * with given scale factor satisfying delaunay condition after rescaling.
   * 
   * @param m, mesh data structure
   * @param u, #v vector, per-vertex scale factors
   * @param e, int, halfedge id
   * @param solve_stats struct collecting info for solvings through out the algorithm
   * @return bool, true indicates delaunay condition is violated.
   */
  static bool NonDelaunay(Mesh<Scalar>& m, const VectorX& u, int e, SolveStats<Scalar>& solve_stats)
  {
    if (m.type[m.h0(e)] == 4)
      return false; //virtual diagonal of symmetric trapezoid
    solve_stats.n_checks++;
    int hij = m.h0(e);
    int hjk = m.n[hij];
    int hki = m.n[hjk];
    int hji = m.h1(e);
    int him = m.n[hji];
    int hmj = m.n[him];
    int i = m.v_rep[m.to[hji]];
    int j = m.v_rep[m.to[hij]];
    int k = m.v_rep[m.to[hjk]];
    int n = m.v_rep[m.to[him]];
    Scalar ui = u[i];
    Scalar uj = u[j];
    Scalar uk = u[k];
    Scalar um = u[n];
    Scalar uijk_avg = (ui + uj + uk)/3.0;
    Scalar ujim_avg = (uj + ui + um)/3.0;
    Scalar ljk = ell(m.l[m.e(hjk)], uj, uk, uijk_avg);
    Scalar lki = ell(m.l[m.e(hki)], uk, ui, uijk_avg);
    Scalar lij = ell(m.l[m.e(hij)], ui, uj, uijk_avg);
    Scalar lji = ell(m.l[m.e(hji)], uj, ui, ujim_avg);
    Scalar lmj = ell(m.l[m.e(hmj)], um, uj, ujim_avg);
    Scalar lim = ell(m.l[m.e(him)], ui, um, ujim_avg);
    
    bool pre_flip_check = (ljk / lki + lki / ljk - (lij / ljk) * (lij / lki)) + (lmj / lim + lim / lmj - (lji / lmj) * (lji / lim)) < 0;
    
    // additionally check whether delaunay is violated after flip
    // we consider the configuration to 'violate delaunay condition' only if 
    // it does not satisfy delaunay check AND post-flip configuration satisfies delaunay condition.
    Scalar umki_avg = (um + uk + ui)/3.0;
    Scalar ukmj_avg = (uk + um + uj)/3.0;
    Scalar _lkm_non_scaled = (m.l[m.e(hjk)] * m.l[m.e(him)] + m.l[m.e(hki)] * m.l[m.e(hmj)]) / m.l[m.e(hij)];
    Scalar _lkm = ell(_lkm_non_scaled , uk, um, ukmj_avg);
    Scalar _lmj = ell(m.l[m.e(hmj)], um, uj, ukmj_avg);
    Scalar _ljk = ell(m.l[m.e(hjk)], uj, uk, ukmj_avg);
    Scalar _lmk = ell(_lkm_non_scaled , um, uk, umki_avg);
    Scalar _lki = ell(m.l[m.e(hki)] , uk, ui, umki_avg);
    Scalar _lim = ell(m.l[m.e(him)] , ui, um, umki_avg);
    bool post_flip_check = (_lki / _lim + _lim / _lki - (_lmk / _lki) * (_lmk / _lim)) + (_ljk / _lmj + _lmj / _ljk - (_lkm / _ljk) * (_lkm / _lmj)) < 0;
    return pre_flip_check && !post_flip_check;
  }

  /**
   * Flip the given halfedge in mesh and update the edge length accordingly.
   * 
   * @param m, mesh data structure
   * @param u, #v vector, per-vertex scale factors
   * @param e, int, halfedge id
   * @param delaunay_stats struct collecting info for delaunay flips through out the algorithm
   * @param Ptolemy, bool, when true the edge length is updated via ptolemy formula, otherwise using law of cosine.
   * @return bool, true indicates flip succeeds.
   */
  static bool EdgeFlip(Mesh<Scalar>& m, const VectorX& u, int e, int tag, DelaunayStats& delaunay_stats, bool Ptolemy = true)
  {
    Mesh<Scalar>& mc = m.cmesh();

    int hij = mc.h0(e);
    int hjk = mc.n[hij];
    int hki = mc.n[hjk];
    int hji = mc.h1(e);
    int him = mc.n[hji];
    int hmj = mc.n[him];

    std::vector<char> &type = mc.type;

    std::vector<int> to_flip;
    if (type[hij] > 0) // skip in non-symmetric mode for efficiency
    {
      int types;
      bool reverse = true;
      if (type[hki] <= type[hmj])
      {
        types = type[hki] * 100000 + type[hjk] * 10000 + type[hij] * 1000 + type[hji] * 100 + type[him] * 10 + type[hmj];
        reverse = false;
      }
      else
        types = type[hmj] * 100000 + type[him] * 10000 + type[hji] * 1000 + type[hij] * 100 + type[hjk] * 10 + type[hki];

      if (types == 231123 || types == 231132 || types == 321123)
        return false; // t1t irrelevant
      if (types == 132213 || types == 132231 || types == 312213)
        return false; // t2t irrelevant
      if (types == 341143)
        return false; // q1q irrelevant
      if (types == 342243)
        return false; // q2q irrelevant

      if (types == 111222 || types == 123312)
        delaunay_stats.n_flips_s++;
      if (types == 111123 || types == 111132)
        delaunay_stats.n_flips_t++;
      if (types == 213324 || types == 123314 || types == 111143 || types == 413324 || types == 23314)
        delaunay_stats.n_flips_q++;
      if (types == 111111)
        delaunay_stats.n_flips_12++;
      switch (types)
      {
      case 111222: // (1|2)
        type[hij] = type[hji] = 3;
        mc.R[hij] = hij;
        mc.R[hji] = hji;
        break;
      case 123312: // (t,_,t)
        type[hij] = type[hki];
        type[hji] = type[hmj];
        mc.R[hij] = hji;
        mc.R[hji] = hij;
        break;
      case 111123: // (1,1,t)
        type[hij] = type[hji] = 4;
        mc.R[hij] = hij;
        mc.R[hji] = hji;
        break;
      case 111132: // (1,1,t) mirrored
        type[hij] = type[hji] = 4;
        mc.R[hij] = hij;
        mc.R[hji] = hji;
        break;
      case 222214: // (2,2,t) following (1,1,t) mirrored
        type[hij] = type[hji] = 3;
        to_flip.push_back(6); // to make sure all fake diagonals are top left to bottom right
        mc.R[hij] = hij;
        mc.R[hji] = hji;
        break;
      case 142222: // (2,2,t) following (1,1,t)
        type[hij] = type[hji] = 3;
        mc.R[hij] = hij;
        mc.R[hji] = hji;
        break;
      case 213324: // (t,_,q)
        type[hij] = type[hji] = 2;
        to_flip.push_back(6);
        break;
      case 134412: // (t,_,q) 2nd
        type[hij] = type[hji] = 1;
        if (!reverse)
        {
          mc.R[hji] = hmj;
          mc.R[hmj] = hji;
          mc.R[mc.opp[hji]] = mc.opp[hmj];
          mc.R[mc.opp[hmj]] = mc.opp[hji];
        }
        else
        {
          mc.R[hij] = hki;
          mc.R[hki] = hij;
          mc.R[mc.opp[hij]] = mc.opp[hki];
          mc.R[mc.opp[hki]] = mc.opp[hij];
        }
        break;
      case 123314: // (q,_,t)
        type[hij] = type[hji] = 1;
        to_flip.push_back(6);
        break;
      case 124432: // (q,_,t) 2nd
        type[hij] = type[hji] = 2;
        if (!reverse)
        {
          mc.R[hki] = hij;
          mc.R[hij] = hki;
          mc.R[mc.opp[hki]] = mc.opp[hij];
          mc.R[mc.opp[hij]] = mc.opp[hki];
        }
        else
        {
          mc.R[hmj] = hji;
          mc.R[hji] = hmj;
          mc.R[mc.opp[hmj]] = mc.opp[hji];
          mc.R[mc.opp[hji]] = mc.opp[hmj];
        }
        break;
      case 111143: // (1,1,q)
        type[hij] = type[hji] = 4;
        mc.R[hij] = hij;
        mc.R[hji] = hji;
        break;
      case 222243: // (2,2,q) following (1,1,q)
        type[hij] = type[hji] = 4;
        to_flip.push_back(5);
        mc.R[hij] = hij;
        mc.R[hji] = hji;
        break;
      case 144442: // (1,1,q)+(2,2,q) 3rd
        type[hij] = type[hji] = 3;
        mc.R[hij] = hij;
        mc.R[hji] = hji;
        break;
      case 413324: // (q,_,q)
        type[hij] = type[hji] = 4;
        to_flip.push_back(6);
        to_flip.push_back(1);
        mc.R[hij] = hij;
        mc.R[hji] = hji;
        break;
      case 423314: // (q,_,q) opp
        type[hij] = type[hji] = 4;
        to_flip.push_back(1);
        to_flip.push_back(6);
        mc.R[hij] = hij;
        mc.R[hji] = hji;
        break;
      case 134414: // (q,_,q) 2nd
        type[hij] = type[hji] = 1;
        break;
      case 234424: // (q,_,q) 3rd
        type[hij] = type[hji] = 2;
        if (!reverse)
        {
          mc.R[hji] = mc.n[mc.n[mc.opp[mc.n[mc.n[hji]]]]]; // attention: hji is not yet flipped here, hence twice .n[]
          mc.R[mc.n[mc.n[mc.opp[mc.n[mc.n[hji]]]]]] = hji;
          mc.R[mc.opp[hji]] = mc.opp[mc.R[hji]];
          mc.R[mc.opp[mc.R[hji]]] = mc.opp[hji];
        }
        else
        {
          mc.R[hij] = mc.n[mc.n[mc.opp[mc.n[mc.n[hij]]]]];
          mc.R[mc.n[mc.n[mc.opp[mc.n[mc.n[hij]]]]]] = hij;
          mc.R[mc.opp[hij]] = mc.opp[mc.R[hij]];
          mc.R[mc.opp[mc.R[hij]]] = mc.opp[hij];
        }
        break;
      case 314423: // fake diag switch following (2,2,t) following (1,1,t) mirrored
        break;
      case 324413: // fake diag switch (opp) following (2,2,t) following (1,1,t) mirrored
        break;
      case 111111:
        break;
      case 222222:
        break;
      case 000000:
        type[hij] = type[hji] = 0; // for non-symmetric mode
        break;
      default:
        spdlog::error(" (attempted to flip edge that should never be non-Delaunay (type{})).", types);
        return false;
      }

      if (reverse)
      {
        for (int i = 0; i < to_flip.size(); i++)
          to_flip[i] = 7 - to_flip[i];
      }
    }

    delaunay_stats.n_flips++;
    if (Ptolemy)
    {
      delaunay_stats.flip_seq.push_back(hij);
    }
    else
    {
      delaunay_stats.flip_seq.push_back(-hij-1);
    }
    if (!m.flip_ccw(hij, Ptolemy))
    {
      spdlog::error(" EDGE COULD NOT BE FLIPPED! ");
    }
    if (tag == 1)
    {
      m.flip_ccw(hij, Ptolemy);
      m.flip_ccw(hij, Ptolemy);
      if (Ptolemy)
      {
        delaunay_stats.flip_seq.push_back(hij);
        delaunay_stats.flip_seq.push_back(hij);
      }
      else
      {
        delaunay_stats.flip_seq.push_back(-hij-1);
        delaunay_stats.flip_seq.push_back(-hij-1);
      }
    } // to make it cw on side 2

    for (int i = 0; i < to_flip.size(); i++)
    {
      if (to_flip[i] == 1)
        EdgeFlip(m, u, mc.e(hki), 2, delaunay_stats, Ptolemy);
      if (to_flip[i] == 2)
        EdgeFlip(m, u, mc.e(hjk), 2, delaunay_stats, Ptolemy);
      if (to_flip[i] == 5)
        EdgeFlip(m, u, mc.e(him), 2, delaunay_stats, Ptolemy);
      if (to_flip[i] == 6)
        EdgeFlip(m, u, mc.e(hmj), 2, delaunay_stats, Ptolemy);
    }

    return true;
  }
  
  /**
   * Repeatedly perform edge flip operations until the rescaled triangles edges satisfying delaunay condition for all.
   * 
   * @param m, mesh data structure
   * @param u, #v vector, per-vertex scale factors
   * @param delaunay_stats struct collecting info for delaunay flips through out the algorithm
   * @param solve_stats struct collecting info for solvings through out the algorithm
   * @param Ptolemy, bool, when true the edge length is updated via ptolemy formula, otherwise using law of cosine.
   * @return void.
   */
  static void MakeDelaunay(Mesh<Scalar>& m, const VectorX& u, DelaunayStats& delaunay_stats, SolveStats<Scalar>& solve_stats, bool Ptolemy = true)
  {
    Mesh<Scalar>& mc = m.cmesh();
    std::set<int> q;
    for (int i = 0; i < mc.n_halfedges(); i++)
    {
      if (mc.opp[i] < i) // Only consider halfedges with lower index to prevent duplication
        continue;
      int type0 = mc.type[mc.h0(i)];
      int type1 = mc.type[mc.h1(i)];
      if (type0 == 0 || type0 == 1 || type1 == 1 || type0 == 3) //type 22 edges are flipped below; type 44 edges (virtual diagonals) are never flipped.
        q.insert(i);
    }
    while (!q.empty())
    {
      int e = *(q.begin());
      q.erase(q.begin());
      int type0 = mc.type[mc.h0(e)];
      int type1 = mc.type[mc.h1(e)];
      if (!(type0 == 2 && type1 == 2) && !(type0 == 4) && NonDelaunay(mc, u, e, solve_stats))
      {
        int Re = -1;
        if (type0 == 1 && type1 == 1)
          Re = mc.e(mc.R[mc.h0(e)]);
        if (!EdgeFlip(m, u, e, 0, delaunay_stats, Ptolemy))
          continue;
        int hn = mc.n[mc.h0(e)];
        q.insert(mc.e(hn));
        q.insert(mc.e(mc.n[hn]));
        hn = mc.n[mc.h1(e)];
        q.insert(mc.e(hn));
        q.insert(mc.e(mc.n[hn]));
        if (type0 == 1 && type1 == 1) // flip mirror edge on sheet 2
        {
          int e = Re;
          if (Re == -1)
            spdlog::info("Negative index");
          if (!EdgeFlip(m, u, e, 1, delaunay_stats, Ptolemy))
            continue;
          int hn = mc.n[mc.h0(e)];
          q.insert(mc.e(hn));
          q.insert(mc.e(mc.n[hn]));
          hn = mc.n[mc.h1(e)];
          q.insert(mc.e(hn));
          q.insert(mc.e(mc.n[hn]));
        }
        // checkR();
      }
    }
  }

  static VectorX DescentDirection(const Eigen::SparseMatrix<Scalar>& hessian, const VectorX& grad, int fixed_dof, SolveStats<Scalar>& solve_stats)
  {

    static Scalar a = 0.0; // Parameter for interpolating from the Newton direction to steepest descent

    auto grad_dof_fixed = grad;
    auto hessian_dof_fixed = hessian;

    // Set fixed degree of freedom in the gradient and hessian
    grad_dof_fixed[fixed_dof] = 0;
    for (int k = 0; k < hessian_dof_fixed.outerSize(); ++k)
    {
      for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(hessian_dof_fixed, k); it; ++it)
      {
        if ((it.row() == fixed_dof) || (it.col() == fixed_dof))
        {
          it.valueRef() = 0;
        }
      }
    }
    hessian_dof_fixed.coeffRef(fixed_dof,fixed_dof) = 1;
 
    // Compute corrected descent direction
    while (true)
    {
      Eigen::SparseMatrix<Scalar> mat;
      if (a == 0)
      {
        mat = hessian_dof_fixed; // Use newton step
      }
      else 
      {     
        // Create identity
        typedef Eigen::Triplet<Scalar> T;
        std::vector<T> tripletList;
        tripletList.reserve(grad.rows());
        for(int i = 0; i < grad.rows(); ++i)
        {
          tripletList.push_back(T(i,i,1));
        }
        Eigen::SparseMatrix<Scalar> id(grad.rows(), grad.rows());
        id.setFromTriplets(tripletList.begin(), tripletList.end());
        
        // Create matrix with correction
        mat = hessian_dof_fixed + a*id;
      }

      Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>> solver;
      solver.compute(mat);
      VectorX d = -solver.solve(grad_dof_fixed);
      Scalar newton_decr = d.dot(grad_dof_fixed);
      if (solver.info() == Eigen::Success && newton_decr < 0)
      {
        a *= 0.5; // start from lower a on the next step
        solve_stats.n_solves++;
        return d;
      }
      else if (a == 0)
      {
        a = 1; // We did not try the correction yet, start from arbitrary value 1
        spdlog::info(" Starting correction.");
      }
      else
      {
        a *= 2; // Correction was not enough, increase weight of id
      }
    }
  }

  /**
   * Backtracking line search function, checking the sign of projected gradient.
   * @param m, mesh data structure
   * @param u0, #v vector, per-vertex scale factors
   * @param d0, #v vector, descent direction
   * @param lambda, initial step size, will be updated when exit line-search
   * @param currentg, gradient computed before start doing line-search, will be updated when exit line-search
   * @param bound_norm, when true: require gradient norm to decrease at each iteration
   * @param delaunay_stats struct collecting info for delaunay flips through out the algorithm
   * @param solve_stats struct collecting info for solvings through out the algorithm
   * @param alg_params, algorithm parameters, for details check the struct definitions on the top
   * @param ls_params, line search parameters, for details check the struct definitions on the top
   * @param stats_params, statistics parameters, for details check the struct definitions on the top
   * @return VectorX, updated per-vertex scale factor along descent direction.
   */
  static VectorX LineSearchNewtonDecr(Mesh<Scalar>& m, const VectorX& u0, const VectorX& d0, Scalar& lambda, VectorX& currentg, bool& bound_norm, DelaunayStats& delaunay_stats, SolveStats<Scalar>& solve_stats, const AlgorithmParameters& alg_params, const LineSearchParameters& ls_params, const StatsParameters& stats_params){
    
    Mesh<Scalar> &mc = m.cmesh();
    auto d = d0;
    auto u = u0;
    auto newton_decr = d.dot(currentg);

    // Scale the search direction vector by lambda
    d *= lambda;  

    // To avoid nans/infs
    if(ls_params.do_reduction){
      while(d.maxCoeff() - d.minCoeff() > 10)
      {
        d /= 2;
        lambda /=2;
      }
    }

    Scalar init_e = 0.0; VectorX init_g = currentg;
    Scalar l2_g0_sq = currentg.dot(currentg);
    VectorX alpha, cot_alpha;

    // Line search
    u += d;
    MakeDelaunay(m, u, delaunay_stats, solve_stats);
    ComputeAngles(mc, u, alpha, cot_alpha);

    int count = 0;
    Gradient(mc, alpha, currentg, solve_stats); // Current gradient value
    Scalar l2_g_sq = currentg.dot(currentg); // Squared norm of the gradient
    Scalar proj_grad = d.dot(currentg);  // Projected gradient
    while ((proj_grad > 0) || (l2_g_sq > l2_g0_sq && bound_norm))
    {
      // Backtrack one step
      d /= 2;
      lambda /= 2; // record changes in lambda as well
      u -= d;
      MakeDelaunay(m, u, delaunay_stats, solve_stats);
      ComputeAngles(mc, u, alpha, cot_alpha);
      Gradient(mc, alpha, currentg, solve_stats); // update gradient

      // Line search condition to ensure quadratic convergence
      if (   (count == 0)
          && ((l2_g_sq <= l2_g0_sq) || (!bound_norm))
          && (0.5 * (d.dot(currentg) + proj_grad) <= 0.1 * newton_decr))
      {
        u += d; // Use full line step
        lambda *= 2;
        MakeDelaunay(m, u, delaunay_stats, solve_stats);
        ComputeAngles(mc, u, alpha, cot_alpha);
        Gradient(mc, alpha, currentg, solve_stats); // update gradient
        break;
      }

      // Update squared gradient norm and projected gradient
      l2_g_sq = currentg.dot(currentg);
      proj_grad = d.dot(currentg);

      count++;

      // Check if gradient norm is below the threshold to drop the bound
      if ((bound_norm) && (lambda <= ls_params.bound_norm_thres))
      {
        bound_norm = false;
        spdlog::debug("Dropping norm bound.");
      }

      // Check if lambda is below the termination threshold
      if (lambda < alg_params.min_lambda) 
        break;
    }
    spdlog::debug("Used lambda {} ", lambda);
    return u;
  }

  /**
   * Backtracking line search function checking the conformal-equivalence-energy and with armijo condition
   * @param m, mesh data structure
   * @param u0, #v vector, per-vertex scale factors
   * @param d0, #v vector, descent direction
   * @param lambda, initial step size, will be updated when exit line-search
   * @param currentg, gradient computed before start doing line-search, will be updated when exit line-search
   * @param bound_norm, when true: require gradient norm to decrease at each iteration
   * @param delaunay_stats struct collecting info for delaunay flips through out the algorithm
   * @param solve_stats struct collecting info for solvings through out the algorithm
   * @param alg_params, algorithm parameters, for details check the struct definitions on the top
   * @param ls_params, line search parameters, for details check the struct definitions on the top
   * @param stats_params, statistics parameters, for details check the struct definitions on the top
   * @return VectorX, updated per-vertex scale factor along descent direction.
   */
  static VectorX LineSearchCETMEnergy(Mesh<Scalar>& m, const VectorX& u0, const VectorX& d0, Scalar& lambda, VectorX& currentg, bool& bound_norm, DelaunayStats& delaunay_stats, SolveStats<Scalar>& solve_stats, const AlgorithmParameters& alg_params, const LineSearchParameters& ls_params, const StatsParameters& stats_params){
    
    Mesh<Scalar> &mc = m.cmesh();
    auto d = d0;
    auto u = u0;
    auto newton_decr = d.dot(currentg);

    // Scale the search direction vector by lambda
    d *= lambda;  

    // To avoid nans/infs
    if(ls_params.do_reduction){
      while(d.maxCoeff() - d.minCoeff() > 10)
      {
        d /= 2;
        lambda /=2;
      }
    }

    Scalar init_e = 0.0; VectorX init_g = currentg;
    Scalar l2_g0_sq = currentg.dot(currentg);
    VectorX alpha, cot_alpha;
    ComputeAngles(mc, u, alpha, cot_alpha);
    
    if(ls_params.energy_samples){
      DelaunayStats d_stats_placeholder;
      SolveStats<Scalar> s_stats_placeholder;
      SampleEnergyAlongDirection(mc, u, stats_params.output_dir+"/"+"energy_sample.csv", d, 2.0, 200, d_stats_placeholder, s_stats_placeholder, true);
      SampleNewtonDecrement(mc, u, stats_params.output_dir+"/"+"newton_decrement.csv", d, 0.0, 2.0, 200, d_stats_placeholder, s_stats_placeholder);
    }
    
    // init energy before line search start
    init_e = ConformalEquivalenceEnergy(mc, alpha, u);

    // Line search
    u += d;
    MakeDelaunay(m, u, delaunay_stats, solve_stats);
    ComputeAngles(mc, u, alpha, cot_alpha);

    Gradient(mc, alpha, currentg, solve_stats); // Current gradient value
    Scalar new_e = ConformalEquivalenceEnergy(mc, alpha, u);
    bool armijo_cond = false, curvature_cond = false;
    do{
      
      armijo_cond = new_e <= (init_e + ls_params.c1 * lambda * init_g.dot(d));
      curvature_cond = currentg.dot(d) >= ls_params.c2 * init_g.dot(d);
      
      if((bound_norm) && (lambda <= ls_params.bound_norm_thres))
      {
        bound_norm = false;
        spdlog::debug("Dropping norm bound.");
      }
      if (lambda < alg_params.min_lambda)
        break;

      if(new_e < init_e && armijo_cond && curvature_cond)
        break;
      
      d /= 2;
      lambda /= 2; // record backtrack changes in lambda
      u -= d; // Backtrack
      MakeDelaunay(m, u, delaunay_stats, solve_stats);
      ComputeAngles(mc, u, alpha, cot_alpha);
      Gradient(mc, alpha, currentg, solve_stats);
      new_e = ConformalEquivalenceEnergy(mc, alpha, u);

    }while(true);

    solve_stats.cetm_energy = new_e;

    spdlog::debug("Used lambda {} ", lambda);
    return u;
  }

  /**
   * The top-level conformal-hyperblic-delaunay algorithm
   * 
   * @param m Mesh data structure
   * @param u0 vector of Scalar size equal to number of vertices of mesh, the initial values of per-vertex logarithmic scale factors
   * @param pt_fids list of face ids per sample point on the original mesh surface, will be updated through out the whole algorithm
   * @param pt_bcs list of barycentric coordinates of each sample point in the correspoinding face, will be updated through out the whole algorithm
   * @param alg_params, algorithm parameters, for details check the struct definitions on the top
   * @param ls_params, line search parameters, for details check the struct definitions on the top
   * @param stats_params, statistics parameters, for details check the struct definitions on the top
   * @return flip sequence
   */
  static std::tuple<VectorX, std::vector<int>> FindConformalMetric(OverlayMesh<Scalar>& m, const VectorX& u0, std::vector<int>& pt_fids, std::vector<Eigen::Matrix<Scalar, 3, 1>>& pt_bcs, const AlgorithmParameters& alg_params, const LineSearchParameters& ls_params, const StatsParameters& stats_params)
  {
    switch (stats_params.log_level){
      case 0: spdlog::set_level(spdlog::level::trace);    break;
      case 1: spdlog::set_level(spdlog::level::debug);    break;
      case 2: spdlog::set_level(spdlog::level::info);     break;
      case 3: spdlog::set_level(spdlog::level::warn);     break;
      case 4: spdlog::set_level(spdlog::level::err);      break;
      case 5: spdlog::set_level(spdlog::level::critical); break;
      default:
      case 6: spdlog::set_level(spdlog::level::off);      break;
    }
    m.bypass_overlay = alg_params.bypass_overlay;
    Mesh<Scalar>& mc = m.cmesh(); 
    mc.init_pts(pt_fids, pt_bcs);

    DelaunayStats delaunay_stats;
    SolveStats<Scalar> solve_stats;

    std::clock_t start;
    start = std::clock();

    // Initialize u to the zero vector
    VectorX u = u0;
    VectorX cot_alpha(mc.n_halfedges());
    VectorX alpha(mc.n_halfedges());

    // Degree of freedom to eliminate to make the Hessian positive definite
    // Choose first vertex arbitrarily for the fixed_dof for regular meshes
    int fixed_dof = 0;
    if (mc.R[0] == 0)
    {
      fixed_dof = 0;
    }
    // Set the fixed_dof to the first boundary halfedge for symmetric meshes
    else
    {
      for (int i = 0; i < mc.n_vertices(); ++i)
      {
        if (mc.v_rep[mc.to[mc.R[mc.out[i]]]] == mc.v_rep[i])
        {
          fixed_dof = mc.v_rep[i];
          break;
        }
      }
    }

    bool bound_norm = (ls_params.lambda0 > ls_params.bound_norm_thres); // prevents the grad norm from increasing
    if(bound_norm) spdlog::debug("Using norm bound.");
    
    double max_curr = 0.0;
    Scalar pi;
#ifdef WITH_MPFR
    if (std::is_same<Scalar, mpfr::mpreal>::value)
        pi = Scalar(mpfr::const_pi());
    else
        pi = Scalar(M_PI);
#else
    pi = Scalar(M_PI);
#endif
    if (stats_params.flip_count){
      // need to also collect max boundary curvature error
      for(int i = 0; i < mc.R.size(); i++){
        if(mc.R[i] == mc.opp[i]){
          int v0 = mc.v_rep[mc.to[i]];
          if(max_curr < std::abs(double(mc.Th_hat[v0])/2-M_PI))
            max_curr = std::abs(double(mc.Th_hat[v0])/2-M_PI);
        }
      }
    }

    Scalar lambda = ls_params.lambda0;

    // Optionally use Euclidean flips instead of Ptolemy flips for the initial MakeDelaunay
    if (!alg_params.initial_ptolemy){
      MakeDelaunay(m, u, delaunay_stats, solve_stats, false);
      spdlog::debug("Finish first delaunay non_ptolemy");
      m.garbage_collection();
      m.bc_original_to_eq(mc.n, mc.to, mc.l);
    }

    // step1 apply per triangle the bc map to unit equilateral triangle
    original_to_equilateral(mc.pts, mc.pt_in_f, mc.n, mc.h, mc.l);
    if (alg_params.initial_ptolemy) {
      MakeDelaunay(m, u, delaunay_stats, solve_stats, true);
      spdlog::debug("Finish first delaunay ptolemy");
    } 
    ComputeAngles(mc, u, alpha, cot_alpha);
    std::ofstream mf;
    if(stats_params.error_log){
      mf.open(stats_params.output_dir+"/"+stats_params.name+".csv",std::ios_base::out);
      mf << "itr, max error, min_u, max_u, lambda, newton_dec, do_reduction, cetm_e\n";
    }

    VectorX currentg;
    Gradient(mc, alpha, currentg, solve_stats);
    while (currentg.cwiseAbs().maxCoeff() >= alg_params.error_eps)
    {
      // Compute gradient and descent direction from Hessian (with efficient solver)
      Eigen::SparseMatrix<Scalar> hessian;
      Hessian(mc, cot_alpha, hessian);
      VectorX d = DescentDirection(hessian, currentg, fixed_dof, solve_stats);

      // Terminate if newton decrement sufficiently smalll      
      Scalar newton_decr = d.dot(currentg);

      if(stats_params.error_log){
        solve_stats.cetm_energy = ConformalEquivalenceEnergy(mc, alpha, u);
        mf << solve_stats.n_solves << "," << std::setprecision(17) << currentg.cwiseAbs().maxCoeff() << "," <<u.minCoeff() << "," << u.maxCoeff() << "," << lambda << "," << newton_decr << "," << ls_params.do_reduction <<" , "<<solve_stats.cetm_energy<< std::endl;
      }
      // Alternative termination conditons to error threshold
      if (lambda < alg_params.min_lambda)
        break;
      if (solve_stats.n_solves >= alg_params.max_itr)
        break;
      if (newton_decr > alg_params.newton_decr_thres)
        break;
      
      // Determine initial lambda for line search based on method parameters
      if (ls_params.energy_cond || ls_params.reset_lambda)
      {
        lambda = ls_params.lambda0; 
      }
      else
      {
        lambda = std::min<Scalar>(1, 2 * lambda); // adaptive step length
      }
      
      // reset lambda when it goes above norm bound threshold
      if ((lambda > ls_params.bound_norm_thres) && (!bound_norm))
      {
        bound_norm = true;
        lambda = ls_params.lambda0;
        spdlog::debug("Using norm bound.");
      }
      if(ls_params.energy_cond)
        u = LineSearchCETMEnergy(m, u, d, lambda, currentg, bound_norm, delaunay_stats, solve_stats, alg_params, ls_params, stats_params);
      else
        u = LineSearchNewtonDecr(m, u, d, lambda, currentg, bound_norm, delaunay_stats, solve_stats, alg_params, ls_params, stats_params);

      // Display current iteration information
      if(ls_params.energy_cond)
        spdlog::info("itr({}) lm({}) flips({}) newton_decr({}) max_error({}), cetm_e({}))", solve_stats.n_solves, lambda, delaunay_stats.n_flips, newton_decr, currentg.cwiseAbs().maxCoeff(), solve_stats.cetm_energy);
      else
        spdlog::info("itr({}) lm({}) flips({}) newton_decr({}) max_error({}))", solve_stats.n_solves, lambda, delaunay_stats.n_flips, newton_decr, currentg.cwiseAbs().maxCoeff());

      ComputeAngles(mc, u, alpha, cot_alpha);

    }

    // Output flip stats
    if(stats_params.error_log) mf.close();
    auto total_time = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    if(stats_params.flip_count){
      auto fname = stats_params.output_dir+"/flips_stats.csv";
      auto header = "name, flips12, flipsq, flipss,flipst, n_flips, fac, time";
      std::stringstream ss;
      ss << stats_params.name << ", " << delaunay_stats.n_flips_12 << ", " << delaunay_stats.n_flips_q << ", " << delaunay_stats.n_flips_s << ", " << delaunay_stats.n_flips_t << ", " << delaunay_stats.n_flips  << ", " << max_curr/pi <<", "<< total_time;
      std::vector<std::string> content = {ss.str()};
      WriteLog(fname, content, header, true);
    }

    if(stats_params.print_summary){
      auto fname = stats_params.output_dir+"/summary_delaunay.csv";
      auto header = "name, n_flips, max_error, min_u, max_u, time";
      VectorX currentg;
      Gradient(mc, alpha, currentg, solve_stats);
      std::stringstream ss;
      ss << stats_params.name << ", " <<delaunay_stats.n_flips << "," << currentg.cwiseAbs().maxCoeff()  << ", " << u.minCoeff() << "," << u.maxCoeff() << "," << total_time;
      std::vector<std::string> content = {ss.str()};
      WriteLog(fname, content, header, true);
    }

    // map barycentric coordinates from equilateral to scaled triangle
    equilateral_to_scaled(mc.pts, mc.pt_in_f, mc.n, mc.h, mc.to, mc.l, u);

    int cnt = 0;
    for (auto pt : mc.pts)
    {
      pt_fids[cnt] = pt.f_id;
      pt_bcs[cnt] = pt.bc;
      cnt++;
    }

    return std::make_tuple(u, delaunay_stats.flip_seq);

  }

   /**
   * Get the Reverse Map of the FindConformalMetric using the Halfedge-Flip Sequence
   * @param m_o OverlayMesh computed from FindConformalMetric
   * @param flip_seq Flip_ccw Sequence used in FindConformalMetric
   * @return Reverse Overlaymesh m_o_rev and vertices-id map between m_o_rev and m_o
   */
  static std::tuple<OverlayMesh<Scalar>, std::vector<int>> GetReverseMap(OverlayMesh<Scalar> & m_o, const std::vector<int> &flip_seq)
  {
    auto mc = m_o.cmesh();
    OverlayMesh<Scalar> m_o_rev(mc);
    bool do_Ptolemy = true;
    // do reverse flips
    for (int ii = flip_seq.size() - 1; ii >= 0; ii--)
    {
      if (do_Ptolemy && flip_seq[ii] < 0)
      {
        do_Ptolemy = false;
        m_o_rev.garbage_collection();
        Eigen::Matrix<Scalar, -1, 1> u_0(m_o_rev.cmesh().out.size());
        u_0.setZero();
        m_o_rev.bc_eq_to_scaled(m_o_rev.cmesh().n, m_o_rev.cmesh().to, m_o_rev.cmesh().l,u_0);
      }
      if (do_Ptolemy)
      {
        m_o_rev.flip_ccw(flip_seq[ii], true);
        m_o_rev.flip_ccw(flip_seq[ii], true);
        m_o_rev.flip_ccw(flip_seq[ii], true);
      }
      else
      {
        m_o_rev.flip_ccw(-flip_seq[ii]-1, false);
        m_o_rev.flip_ccw(-flip_seq[ii]-1, false);
        m_o_rev.flip_ccw(-flip_seq[ii]-1, false);
      }
      
    }
    if(m_o_rev.bypass_overlay){
      m_o.bypass_overlay = true;
      return std::make_tuple(m_o_rev, std::vector<int>());
    }
    m_o.garbage_collection();
    m_o_rev.garbage_collection();

    if (do_Ptolemy == false)
    {
      m_o_rev.bc_original_to_eq(m_o_rev.cmesh().n, m_o_rev.cmesh().to, m_o_rev.cmesh().l);
    }
    spdlog::debug("#m_o.out: {}, #m_o_rev.out: {}", m_o.out.size(), m_o_rev.out.size());
    spdlog::debug("#m_o.n: {}, #m_o_rev.n: {}", m_o.n.size(), m_o_rev.n.size());

    // get the v_map
    std::vector<int> v_map(m_o.out.size());
    // init the original vertices part with Id
    for (int i = 0; i < mc.out.size(); i++)
    {
      v_map[i] = i;
    }
    // init the segment vertices part with -1
    for (int i = mc.out.size(); i < v_map.size(); i++)
    {
      v_map[i] = -1;
    }

    for (int v_start = 0; v_start < mc.out.size(); v_start++)
    {
      int h_out0 = m_o.out[v_start];
      int h_out0_copy = h_out0;
      int v_end = m_o.find_end_origin(h_out0);

      int h_out0_rev = m_o_rev.out[v_start];
      bool flag = false;
      int while_cnt = 0;
      int caseid = 0;

      while (true)
      {
        if (m_o_rev.find_end_origin(h_out0_rev) == v_end && m_o.dist_to_next_origin(h_out0) == m_o_rev.dist_to_next_origin(h_out0_rev))
        {
          // test first segment vertex
          // case 1, no segment vertex
          if (m_o_rev.to[h_out0_rev] == v_end)
          {
            caseid = 0;
            if (m_o.next_out(h_out0) != h_out0_copy)
            {
              h_out0 = m_o.next_out(h_out0);
              v_end = m_o.find_end_origin(h_out0);
            }
            else
            {
              flag = true;
            }

          }
          else
          {
            int h_first = m_o.n[h_out0];
            int h_first_rev = m_o_rev.n[h_out0_rev];

            if (m_o.find_end_origin(h_first) == m_o_rev.find_end_origin(h_first_rev) && m_o.find_end_origin(m_o.opp[h_first]) == m_o_rev.find_end_origin(m_o_rev.opp[h_first_rev]) && m_o.dist_to_next_origin(h_first) == m_o_rev.dist_to_next_origin(h_first_rev))
            {
              caseid = 1;
              flag = true;
            }
          }
        }
        
        if (flag) break;

        h_out0_rev = m_o_rev.next_out(h_out0_rev);
        while_cnt++;

        if (while_cnt > 99999)
        {
          spdlog::error("infi loop in finding first match");
          break;
        }
      }

      int h_out = h_out0;
      int h_out_rev = h_out0_rev;

      do
      {
        int h_current = h_out;
        int h_current_rev = h_out_rev;
        
        while (m_o.vertex_type[m_o.to[h_current]] != ORIGINAL_VERTEX)
        {
          
          if (m_o_rev.vertex_type[m_o_rev.to[h_current_rev]] == ORIGINAL_VERTEX)
          {
            spdlog::error("out path not matching, case: {}", caseid);
            break;  
          }
          int v_current = m_o.to[h_current];
          int v_current_rev = m_o_rev.to[h_current_rev];
          if (v_map[v_current] == -1)
          {
            v_map[v_current] = v_current_rev;
          }
          else if (v_map[v_current] != v_current_rev)
          {
            spdlog::error("the mapping is wrong, case: {}", caseid);
          }
          h_current = m_o.n[m_o.opp[m_o.n[h_current]]];
          h_current_rev = m_o_rev.n[m_o_rev.opp[m_o_rev.n[h_current_rev]]];
        }
        h_out = m_o.next_out(h_out);
        h_out_rev = m_o_rev.next_out(h_out_rev);
      } while (h_out != h_out0);
      
    }
    
    return std::make_tuple(m_o_rev, v_map);
  }

  /**
   * Interpolate 3d coordinates to get the OverlayMesh in 3d
   * @param m_o OverlayMesh computed from FindConformalMetric
   * @param flip_seq Flip_ccw Sequence used in FindConformalMetric
   * @param x 3d coordinat of the Original Mesh
   * @return interpolated OverlayMesh Coordinates
   */
  static std::vector<std::vector<Scalar>> Interpolate_3d(OverlayMesh<Scalar> & m_o, const std::vector<int> &flip_seq, const std::vector<std::vector<Scalar>> &x, bool uniform = false)
  {
    std::vector<std::vector<Scalar>> z(3);

    if (uniform)
    {
      for (int j = 0; j < 3; j++)
      {
        z[j] = m_o.interpolate_along_o(x[j]);
      }
      return z;
    }
    auto rev_map = GetReverseMap(m_o, flip_seq);
    if(m_o.bypass_overlay) return std::vector<std::vector<Scalar>>();
    auto m_o_rev = std::get<0>(rev_map);
    auto v_map = std::get<1>(rev_map);

    Eigen::Matrix<Scalar, -1, 1> u_0(m_o_rev.cmesh().out.size());
    u_0.setZero();

    m_o_rev.bc_eq_to_scaled(m_o_rev.cmesh().n, m_o_rev.cmesh().to, m_o_rev.cmesh().l, u_0);

    std::vector<std::vector<Scalar>> z_rev(3);
    for (int j = 0; j < 3; j++)
    {
      z_rev[j] = m_o_rev.interpolate_along_o_bc(m_o_rev.cmesh().opp, m_o_rev.cmesh().to, x[j]);
    }

    for (int j = 0; j < 3; j++)
    {
      z[j].resize(z_rev[j].size());
      for (int i = 0; i < z[j].size(); i++)
      {
        z[j][i] = z_rev[j][v_map[i]];
      }
    }
    
    return z;
  }

  /**
   * Start at any configuration that all scaled elements in mesh satisfies delaunay condition, evenly evaluate newton-decrement along 
   * certain direction for a number of samples the series of sampled values will be written to file `fname`.
   * 
   * @param m0 Mesh data structure
   * @param u0 vector of Scalar size equal to number of vertices of mesh, the initial values of per-vertex logarithmic scale factors
   * @param d0 vector of Scalar size eqaul to number of vertices of mesh, a delta vector on u0
   * @param lambda_max Scalar controlling the maximum step size along the direction d0
   * @param n_samples Total number of evenly distributed sample points along d0
   * @param delaunay_stats struct collecting stats for delaunay flips through out the algorithm
   * @param solve_stats struct collecting info for solvings through out the algorithm
   * @param rescale Scale (and record) the lambda values by |d0| and the newton decr by 1/|d0|
   * @return void
   */
  static void SampleNewtonDecrement(const Mesh<Scalar>& m0, const VectorX& u0, std::string fname, const VectorX &d, Scalar lambda_min, Scalar lambda_max, int n_sample, DelaunayStats& delaunay_stats, SolveStats<Scalar>& solve_stats, bool rescale=false){
    
    if (n_sample == 0)
      return;

    std::vector<std::string> newton_dec;
    Scalar step_size = (lambda_max - lambda_min) / n_sample;
    VectorX alpha, cot_alpha;
    auto m = m0;
    auto u = u0;

    spdlog::info("n_sample = {}", n_sample);

    for (int i = 0; i < n_sample; i++)
    {
      m = m0;
      u = u0 + (lambda_min + i * step_size) * d;
      MakeDelaunay(m, u, delaunay_stats, solve_stats);
      ComputeAngles(m, u, alpha, cot_alpha);
      VectorX g;
      Gradient(m, alpha, g, solve_stats);
      std::stringstream ss;
      if (rescale)
      {
          ss << std::setprecision(17) << sqrt(d.dot(d))*(lambda_min + i * step_size)
             << "," << d.dot(g)/sqrt(d.dot(d));
      }
      else
      {
          ss << std::to_string(i) << "," <<std::setprecision(17) << d.dot(g);
      }
      newton_dec.push_back(ss.str());
    }

    WriteLog(fname, newton_dec, "itr, newton_decrement");

  }
  /**
   * @brief Overloaded version of sampleNewtonDecrement function above. Used for python binding. Unlike the above method, this method always computes and uses the Newton descent direction.
   * @param m0 Mesh data structure
   * @param u0 vector of Scalar size equal to number of vertices of mesh, the initial values of per-vertex logarithmic scale factors
   * @param lambda_min Scalar controlling the minimum step size along the direction d0
   * @param lambda_max Scalar controlling the maximum step size along the direction d0
   * @param n_samples Total number of evenly distributed sample points along d0
   * @return void
   */
  static void SampleNewtonDecrementStl(Mesh<Scalar>& m0,
                                       std::vector<Scalar>& u0_vec,
                                       std::string fname,
                                       Scalar lambda_min,
                                       Scalar lambda_max,
                                       int n_sample) {
    VectorX u0(u0_vec.size()); 
    for(int i = 0; i < u0_vec.size(); i++)
      u0(i) = u0_vec[i];

    // Create placeholder stat structures
    DelaunayStats d_stats_placeholder;
    SolveStats<Scalar> s_stats_placeholder;

    // Degree of freedom to eliminate to make the Hessian positive definite
    // Choose first vertex arbitrarily for the fixed_dof for regular meshes
    int fixed_dof = 0;
    if (m0.R[0] == 0)
    {
      fixed_dof = 0;
    }
    // Set the fixed_dof to the first boundary halfedge for symmetric meshes
    else
    {
      for (int i = 0; i < m0.n_vertices(); ++i)
      {
        if (m0.to[m0.R[m0.out[i]]] == i)
        {
          fixed_dof = i;
          break;
        }
      }
    }

    // Compute angles and cotangents of angles
    VectorX cot_alpha(m0.n_halfedges());
    VectorX alpha(m0.n_halfedges());
    MakeDelaunay(m0, u0, d_stats_placeholder, s_stats_placeholder, true);
    ComputeAngles(m0, u0, alpha, cot_alpha);

    // Compute descent direction from gradient and hessian
    VectorX currentg;
    Gradient(m0, alpha, currentg, s_stats_placeholder);
    Eigen::SparseMatrix<Scalar> hessian;
    Hessian(m0, cot_alpha, hessian);
    VectorX d = DescentDirection(hessian, currentg, fixed_dof, s_stats_placeholder);

    // Sample newton decrement
    SampleNewtonDecrement(m0,
                          u0,
                          fname,
                          d,
                          lambda_min,
                          lambda_max,
                          n_sample,
                          d_stats_placeholder,
                          s_stats_placeholder,
                          true);
  }

  /**
   * Start at any configuration that all scaled elements in mesh satisfies delaunay condition, evenly evaluate conformal-equivalence-energy along 
   * certain direction for a number of samples the series of sampled values will be written to file `fname`.
   * 
   * @param m0 Mesh data structure
   * @param u0 vector of Scalar size equal to number of vertices of mesh, the initial values of per-vertex logarithmic scale factors
   * @param d0 vector of Scalar size eqaul to number of vertices of mesh, a delta vector on u0
   * @param lambda_max Scalar controlling the maximum step size along the direction d0
   * @param n_samples Total number of evenly distributed sample points along d0
   * @param delaunay_stats struct collecting stats for delaunay flips through out the algorithm
   * @param solve_stats struct collecting info for solvings through out the algorithm
   * @return void
   */
  static void SampleEnergyAlongDirection(const Mesh<Scalar>& m0, const VectorX& u0, std::string fname, const VectorX &d, Scalar lambda_max, int n_sample, DelaunayStats& delaunay_stats, SolveStats<Scalar>& solve_stats, bool subtract_avg=false){

    if (n_sample == 0) return;

    VectorX alpha, cot_alpha;
    Scalar step_size = lambda_max / n_sample;
    Scalar avg_e = 0.0;
    auto m = m0; auto u = u0;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> E; E.setZero(n_sample);
    for (int i = 0; i < n_sample; i++){
      m = m0;
      u = u0 + i * step_size * d;
      MakeDelaunay(m, u, delaunay_stats, solve_stats);
      ComputeAngles(m, u, alpha, cot_alpha);
      E[i] = ConformalEquivalenceEnergy(m, alpha, u);
    }
    if(subtract_avg) avg_e = E.sum() / n_sample;
    
    std::vector<std::string> e_samples;
    for(int i = 0; i < E.size(); i++){
      std::stringstream ss;
      ss << std::to_string(i) << "," << std::setprecision(17) << E[i]-avg_e;
      e_samples.push_back(ss.str());
    }

    std::fstream nf(fname,std::ios::in | std::ios::out);
    WriteLog(fname, e_samples, "itr, e-avg_e");

  }

  /**
   * Given the prescribed per-vertex angle sum, modify the angle sum at first vertex, 
   * to make sure Gauss-Bonnet is respected up to numerical error.
   * @param m Mesh data structure
   * @return void
   */
  static void GaussBonnetCorrection(Mesh<Scalar>& m)
  {
    
    Scalar pi;
#ifdef WITH_MPFR
    if (std::is_same<Scalar, mpfr::mpreal>::value)
      pi = Scalar(mpfr::const_pi());
    else
      pi = Scalar(M_PI);
#else
      pi = Scalar(M_PI);
#endif
    int double_genus = 2 - (m.n_vertices() - m.n_edges() + m.n_faces());
    Scalar targetsum = pi * (2 * m.n_vertices() - 2 * (2 - double_genus));
    double th_hat_sum = 0.0;
    for(auto t: m.Th_hat)
      th_hat_sum += t;
    m.Th_hat[0] -= (th_hat_sum - targetsum);
  }

  /**
   * Logging function, to write list of strings to given file, possibly with header if the file is empty.
   * @param fname, the filename to write log to.
   * @param v, vector of strings to be written to file.
   * @param header, will be written as first line to the file if it's empty.
   * @param append, toggle between append (true) and out mode (false).
   * @return void
   */
  static void WriteLog(std::string fname, std::vector<std::string>& v, std::string header="", bool append=false){
    std::fstream mf, nf; nf.open(fname, std::ios_base::in);
    if(append)
      mf.open(fname, std::ios_base::app);
    else
      mf.open(fname, std::ios_base::out);
    
    if(!(append && nf.peek() != std::ifstream::traits_type::eof()))
      mf << header << "\n";
    for(int i = 0; i < v.size(); i++){
      mf << v[i] << "\n";
    }
    mf.close();
  }

};
#endif
