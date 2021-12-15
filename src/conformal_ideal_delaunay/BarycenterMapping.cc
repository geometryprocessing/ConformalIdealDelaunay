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
#include "BarycenterMapping.hh"
#ifdef WITH_MPFR
#include <unsupported/Eigen/MPRealSupport>
#endif
template <typename Scalar>
Scalar compute_tan_half(Scalar a, Scalar b, Scalar c)
{
    return sqrt((a-b+c) * (a+b-c) / (a+b+c) / (-a+b+c));
}

template <typename Scalar, typename Scalar_pt>
void recompute_bc_hyper(int _h,
                        const std::vector<int> &n,
                        const std::vector<int> &h,
                        const std::vector<int> &f,
                        const std::vector<int> &opp,
                        const std::vector<Scalar> &l,
                        std::vector<Pt<Scalar_pt>> &pts,
                        std::vector<std::vector<int>> &pt_in_f)
{
    int ha = _h;
    int hb = opp[_h];
    int h2 = n[ha];
    int h3 = n[h2];
    int h4 = n[hb];
    int h5 = n[h4];

    int f0 = f[ha];
    int f1 = f[hb];

    Scalar lil = l[h4];
    Scalar ljk = l[h2];
    Scalar llj = l[h5];
    Scalar lki = l[h3];

    // step 2(a)
    Scalar exp_zij = (llj * lki) / (lil * ljk);
    Scalar z = (exp_zij - 1) / (exp_zij + 1);


    // step 2(bc)
    Eigen::Matrix<Scalar, 1, 3> A, B, C, D;

    A << -1, 0, 0;
    B << 1, 0, 0;
    C << z, sqrt(1 - z * z), 0;
    D << -z, -sqrt(1 - z * z), 0;

    Scalar lab = 2;
    Scalar lba = lab;
    Scalar lbc = (B - C).norm();
    Scalar lca = (C - A).norm();
    Scalar lad = (A - D).norm();
    Scalar ldb = (D - B).norm();
    Scalar lcd = (C - D).norm();
    Scalar ldc = lcd;
    assert(lbc == lad && ldb == lca);

    int it0 = 0, it1 = 0;
    int h_tmp = h[f0];
    while (h_tmp != ha)
    {
        h_tmp = n[h_tmp];
        it0++;
    }
    h_tmp = h[f1];
    while (h_tmp != hb)
    {
        h_tmp = n[h_tmp];
        it1++;
    }
    std::vector<int> pts_new_f0, pts_new_f1;

    for (auto pt_id : pt_in_f[f0])
    {
        auto pt = pts[pt_id].bc;
        // eqn 8
        pt(it0) *= Scalar_pt(lbc / (lab * lca));           // A
        pt((it0 + 1) % 3) *= Scalar_pt(lca / (lab * lbc)); // B
        pt((it0 + 2) % 3) *= Scalar_pt(lab / (lbc * lca)); // C
        pt /= pt.sum();

        if (pt(it0) >= pt((it0 + 1) % 3))
        {
            Eigen::Matrix<Scalar_pt, 1, 3> bc_tmp;
            bc_tmp << pt(it0) - pt((it0 + 1) % 3), pt((it0 + 1) % 3), pt((it0 + 1) % 3) + pt((it0 + 2) % 3);
            // eqn 8 inv
            bc_tmp[0] *= Scalar_pt(lad * lca / ldc); // A
            bc_tmp[1] *= Scalar_pt(lad * ldc / lca); // D
            bc_tmp[2] *= Scalar_pt(lca * ldc / lad); // C

            bc_tmp /= bc_tmp.sum();

            pts[pt_id].bc = bc_tmp;
            pts[pt_id].f_id = f0;
            pts_new_f0.push_back(pt_id);
        }
        else
        {
            Eigen::Matrix<Scalar_pt, 1, 3> bc_tmp;
            bc_tmp << pt((it0 + 1) % 3) - pt(it0), pt(it0) + pt((it0 + 2) % 3), pt(it0);
            // eqn 8 inv
            bc_tmp[0] *= Scalar_pt(ldb * lbc / lcd); // B
            bc_tmp[1] *= Scalar_pt(lbc * lcd / ldb); // C
            bc_tmp[2] *= Scalar_pt(ldb * lcd / lbc); // D
            bc_tmp /= bc_tmp.sum();

            pts[pt_id].bc = bc_tmp;
            pts[pt_id].f_id = f1;
            pts_new_f1.push_back(pt_id);
        }
    }

    for (auto pt_id : pt_in_f[f1])
    {
        auto pt = pts[pt_id].bc;
        // eqn 8
        pt(it1) *= Scalar_pt(lad / (ldb * lba));           // B
        pt((it1 + 1) % 3) *= Scalar_pt(ldb / (lad * lba)); // A
        pt((it1 + 2) % 3) *= Scalar_pt(lba / (ldb * lad)); // D
        pt /= pt.sum();

        if (pt((it1 + 1) % 3) >= pt(it1))
        {
            Eigen::Matrix<Scalar_pt, 1, 3> bc_tmp;
            bc_tmp << -pt(it1) + pt((it1 + 1) % 3), pt(it1) + pt((it1 + 2) % 3), pt(it1);

            // eqn 8 inv
            bc_tmp[0] *= Scalar_pt(lad * lca / ldc); // A
            bc_tmp[1] *= Scalar_pt(lad * ldc / lca); // D
            bc_tmp[2] *= Scalar_pt(lca * ldc / lad); // C
            bc_tmp /= bc_tmp.sum();

            pts[pt_id].bc = bc_tmp;
            pts[pt_id].f_id = f0;
            pts_new_f0.push_back(pt_id);
        }
        else
        {
            Eigen::Matrix<Scalar_pt, 1, 3> bc_tmp;
            bc_tmp << pt(it1) - pt((it1 + 1) % 3), pt((it1 + 1) % 3), pt((it1 + 1) % 3) + pt((it1 + 2) % 3);

            // eqn 8 inv
            bc_tmp[0] *= Scalar_pt(ldb * lbc / lcd); // B
            bc_tmp[1] *= Scalar_pt(lbc * lcd / ldb); // C
            bc_tmp[2] *= Scalar_pt(ldb * lcd / lbc); // D
            bc_tmp /= bc_tmp.sum();

            pts[pt_id].bc = bc_tmp;
            pts[pt_id].f_id = f1;
            pts_new_f1.push_back(pt_id);
        }
    }

    pt_in_f[f0] = pts_new_f0;
    pt_in_f[f1] = pts_new_f1;
}

template <typename Scalar, typename Scalar_pt>
void original_to_equilateral(
    std::vector<Pt<Scalar_pt>> &pts,
    std::vector<std::vector<int>> &pt_in_f,
    const std::vector<int> &n,
    const std::vector<int> &h,
    const std::vector<Scalar> &l)
{
    for (int i = 0; i < pts.size(); i++)
    {
        int fid = pts[i].f_id;
        int hij = h[fid];
        int hjk = n[hij];
        int hki = n[hjk];
        Scalar lij = l[hij];
        Scalar ljk = l[hjk];
        Scalar lki = l[hki];
        Scalar Si = (lij * lki) / ljk;
        Scalar Sj = (lij * ljk) / lki;
        Scalar Sk = (lki * ljk) / lij;
        pts[i].bc(0) *= Scalar_pt(Si);
        pts[i].bc(1) *= Scalar_pt(Sj);
        pts[i].bc(2) *= Scalar_pt(Sk); 
        pts[i].bc /= pts[i].bc.sum();
    }
}

template <typename Scalar, typename Scalar_pt>
void equilateral_to_scaled(
    std::vector<Pt<Scalar_pt>> &pts,
    std::vector<std::vector<int>> &pt_in_f,
    const std::vector<int> &n,
    const std::vector<int> &h,
    const std::vector<int> &to,
    const std::vector<Scalar> &l,
    const Eigen::Matrix<Scalar, -1, 1> &u)
{
    for (int i = 0; i < pts.size(); i++)
    {
      int fid = pts[i].f_id;
      int hij = h[fid];
      int hjk = n[hij];
      int hki = n[hjk];
      Scalar lij = l[hij];
      Scalar ljk = l[hjk];
      Scalar lki = l[hki];
      if(to[hki] < u.rows() && to[hij] < u.rows() && to[hjk] < u.rows()){
        Scalar u1 = u[to[hki]];
        Scalar u2 = u[to[hij]];
        Scalar u3 = u[to[hjk]];
        Scalar u_avg = (u1 + u2 + u3) / 3;
        Scalar Si = ljk / (lij * lki) * exp(-u1 + u_avg);
        Scalar Sj = lki / (lij * ljk) * exp(-u2 + u_avg);
        Scalar Sk = lij / (lki * ljk) * exp(-u3 + u_avg);
        pts[i].bc(0) *= Scalar_pt(Si);
        pts[i].bc(1) *= Scalar_pt(Sj);
        pts[i].bc(2) *= Scalar_pt(Sk);
        pts[i].bc /= pts[i].bc.sum();
      }
    }
}

template <typename Scalar, typename Scalar_pt>
void recompute_bc_original(int _h,
                        const std::vector<int> &n,
                        const std::vector<int> &h,
                        const std::vector<int> &f,
                        const std::vector<int> &opp,
                        const std::vector<Scalar> &l,
                        std::vector<Pt<Scalar_pt>> &pts,
                        std::vector<std::vector<int>> &pt_in_f)
{
    int ha = _h;
    int hb = opp[_h];
    int h2 = n[ha];
    int h3 = n[h2];
    int h4 = n[hb];
    int h5 = n[h4];

    int f0 = f[ha];
    int f1 = f[hb];
    
    Scalar l_ab = l[ha];
    Scalar l_ba = l_ab;
    Scalar l_bc = l[h2];
    Scalar l_ca = l[h3];
    Scalar l_ad = l[h4];
    Scalar l_db = l[h5];

    Eigen::Matrix<Scalar, 1, 3> A, B, C, D;

    A << 0, 0, 0;
    B << l_ab, 0, 0;
    Scalar cos_bac = (l_ab * l_ab + l_ca * l_ca - l_bc * l_bc) / (2 * l_ab * l_ca);
    Scalar sin_bac = sqrt(1 - cos_bac * cos_bac);
    C << l_ca * cos_bac, l_ca * sin_bac, 0;
    Scalar cos_bad = (l_ab * l_ab + l_ad * l_ad - l_db * l_db) / (2 * l_ab * l_ad);
    Scalar sin_bad = sqrt(1 - cos_bad * cos_bad);
    D << l_ad * cos_bad, -l_ad * sin_bad, 0;

    // select correct order for computing the bc
    int it0 = 0, it1 = 0;
    int h_tmp = h[f0];
    while (h_tmp != ha)
    {
        h_tmp = n[h_tmp];
        it0++;
    }
    h_tmp = h[f1];
    while (h_tmp != hb)
    {
        h_tmp = n[h_tmp];
        it1++;
    }
    std::vector<int> pts_new_f0, pts_new_f1;
    Eigen::Matrix<Scalar, -1, -1> new_bc;
    for (auto pt_id : pt_in_f[f0])
    {
        auto pt = pts[pt_id].bc;
        auto P = pt(it0) * A + pt((it0 + 1) % 3) * B + pt((it0 + 2) % 3) * C;
        igl::barycentric_coordinates(P, A, D, C, new_bc);

        if (new_bc.row(0).minCoeff() >= -1e-7)
        {
            pts[pt_id].bc = new_bc.row(0).template cast<Scalar_pt>();
            pts[pt_id].f_id = f0;
            pts_new_f0.push_back(pt_id);
        }
        else
        {
            new_bc.resize(0, 0);
            igl::barycentric_coordinates(P, B, C, D, new_bc);
         
            pts[pt_id].bc = new_bc.row(0).template cast<Scalar_pt>();
            pts[pt_id].f_id = f1;
            pts_new_f1.push_back(pt_id);
        }
    }

    for (auto pt_id : pt_in_f[f1])
    {
        auto pt = pts[pt_id].bc;
        auto P = pt(it1) * B + pt((it1 + 1) % 3) * A + pt((it1 + 2) % 3) * D;
        igl::barycentric_coordinates(P, A, D, C, new_bc);

        if (new_bc.row(0).minCoeff() >= -1e-7)
        {
            pts[pt_id].bc = new_bc.row(0).template cast<Scalar_pt>();
            pts[pt_id].f_id = f0;
            pts_new_f0.push_back(pt_id);
        }
        else
        {
            new_bc.resize(0, 0);
            igl::barycentric_coordinates(P, B, C, D, new_bc);
      
            pts[pt_id].bc = new_bc.row(0).template cast<Scalar_pt>();
            pts[pt_id].f_id = f1;
            pts_new_f1.push_back(pt_id);
        }
    }

    pt_in_f[f0] = pts_new_f0;
    pt_in_f[f1] = pts_new_f1;
}

template void equilateral_to_scaled<double, double>(std::vector<Pt<double>> &, std::vector<std::vector<int>> &, const std::vector<int> &, const std::vector<int> &, const std::vector<int> &, const std::vector<double> &, const Eigen::Matrix<double, -1, 1> &);
template double compute_tan_half<double>(double a, double b, double c);
template void recompute_bc_hyper<double, double>(int, const std::vector<int> &, const std::vector<int> &, const std::vector<int> &, const std::vector<int> &, const std::vector<double> &, std::vector<Pt<double>> &, std::vector<std::vector<int>> &);
template void original_to_equilateral<double, double>(std::vector<Pt<double>> &, std::vector<std::vector<int>> &, const std::vector<int> &, const std::vector<int> &, const std::vector<double> &);
template void recompute_bc_original<double, double>(int, const std::vector<int> &, const std::vector<int> &, const std::vector<int> &, const std::vector<int> &, const std::vector<double> &, std::vector<Pt<double>> &, std::vector<std::vector<int>> &);

#ifdef WITH_MPFR
template mpfr::mpreal compute_tan_half<mpfr::mpreal>(mpfr::mpreal a, mpfr::mpreal b, mpfr::mpreal c);
template void equilateral_to_scaled<mpfr::mpreal, mpfr::mpreal>(std::vector<Pt<mpfr::mpreal>>&, std::vector<std::vector<int>>&, const std::vector<int>&, const std::vector<int>&, const std::vector<int>&, const std::vector<mpfr::mpreal>&, const Eigen::Matrix<mpfr::mpreal, -1, 1>&);
template void recompute_bc_hyper<mpfr::mpreal, mpfr::mpreal>(int, const std::vector<int>&, const std::vector<int>&, const std::vector<int>&, const std::vector<int>&, const std::vector<mpfr::mpreal>&, std::vector<Pt<mpfr::mpreal>>&, std::vector<std::vector<int>>&);
template void original_to_equilateral<mpfr::mpreal, mpfr::mpreal>(std::vector<Pt<mpfr::mpreal>>&, std::vector<std::vector<int>>&, const std::vector<int>&, const std::vector<int>&, const std::vector<mpfr::mpreal>&);
template void recompute_bc_original<mpfr::mpreal, mpfr::mpreal>(int, const std::vector<int> &, const std::vector<int> &, const std::vector<int> &, const std::vector<int> &, const std::vector<mpfr::mpreal> &, std::vector<Pt<mpfr::mpreal>> &, std::vector<std::vector<int>> &);
#endif