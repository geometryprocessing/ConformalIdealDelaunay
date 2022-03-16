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
#ifndef SAMPLE_HH
#define SAMPLE_HH

#ifdef USE_EMBREE
#include <igl/embree/unproject_onto_mesh.h>
#else
#include <igl/unproject_onto_mesh.h>
#endif
#include <spdlog/spdlog.h>
#include <vector>
#include <igl/cat.h>
#include <igl/boundary_loop.h>
#include "../util/plot.hh"

typedef std::tuple< Eigen::Matrix4f,
                    Eigen::Matrix4f,
                    Eigen::Vector4f > camera_info;

std::tuple<std::vector<std::vector<int>>,
           std::vector<std::vector<Eigen::Vector3d>>>
get_pt_mat(camera_info cam,
        const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
        const Eigen::MatrixXd &V_edges, const Eigen::MatrixXi &F_edges, int red_size, int blue_size,
        int W, int H)
{
  auto view = std::get<0>(cam);
  auto proj = std::get<1>(cam);
  auto vp = std::get<2>(cam);

  int Vsize = V.rows();
  int Fsize = F.rows();

  Eigen::MatrixXd V_all;
  Eigen::MatrixXi F_all;

  igl::cat(1, V, V_edges, V_all);
  igl::cat(1, F, F_edges, F_all);

  for (int i = Fsize; i < F_all.rows(); i++)
  {
    for (int j = 0; j < 3; j++)
    {
      F_all(i, j) += Vsize;
    }
  }

  std::vector<std::vector<int>> fid_mat(H);
  std::vector<std::vector<Eigen::Vector3d>> bc_mat(H);
  for (int i = 0; i < H; i++)
  {
    fid_mat[i].resize(W);
    std::fill(fid_mat[i].begin(), fid_mat[i].end(), -1);
    bc_mat[i].resize(W);
    std::fill(bc_mat[i].begin(), bc_mat[i].end(), Eigen::Vector3d(0,0,0));
  }

#ifdef USE_EMBREE
  igl::embree::EmbreeIntersector ei;
  ei.init(V_all.cast<float>(), F_all);
#endif
  spdlog::debug("ray tracing start.");
  for (int i = 0; i < H; i++)
  {
    for (int j = 0; j < W; j++)
    {
      int fid;
      Eigen::Vector3f bc;
      double x = vp(2) / (double)W * (j + 0.5);
      double y = vp(3) / (double)H * (H - i - 0.5);
#ifdef USE_EMBREE
      if (igl::embree::unproject_onto_mesh(Eigen::Vector2f(x, y), F_all, view, proj, vp, ei, fid, bc))
#else
      if (igl::unproject_onto_mesh(Eigen::Vector2f(x, y), view, proj, vp, V_all, F_all, fid, bc))
#endif
      {
        if (fid < Fsize)
        {
          fid_mat[i][j] = fid;
        }
        else if (fid < Fsize + red_size)
        {
          fid_mat[i][j] = -2;
        }
        else if (fid < Fsize + red_size + blue_size)
        {
          fid_mat[i][j] = -3;
        }
        else
        {
          fid_mat[i][j] = -4;
        }
        bc_mat[i][j] = bc.cast<double>();
      }
    }
  }
  spdlog::debug("ray tracing end.");

  return std::make_tuple(fid_mat, bc_mat);
}

std::tuple<camera_info,
           Eigen::MatrixXd,
           Eigen::MatrixXi,
           int, int,
           std::vector<std::vector<int>>,
           std::vector<std::vector<Eigen::Vector3d>>>
cpp_viewer(Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::VectorXd &Th_hat,
           double pt_scale, double bd_scale, int W = 256, int H = 256, bool show_bd = false)
{
  igl::opengl::glfw::Viewer viewer;
  
  Eigen::VectorXd S(Th_hat.rows());
  for (int i = 0; i < Th_hat.rows(); i++)
  {
    if (Th_hat(i) - 2 * M_PI > 1e-5)
      S(i) = -1;
    else if (Th_hat(i) - 2 * M_PI < -1e-5)
      S(i) = 1;
    else
      S(i) = 0;
  }
  std::vector<std::vector<int>> bd_loops;
  igl::boundary_loop(F, bd_loops);
  for (auto bd : bd_loops)
  {
    for (auto bd_pt : bd)
    {
      S(bd_pt) = 0;
    }
  }
  plot_singularity_sphere(viewer, V, F, S, pt_scale);
  if (show_bd)
  {
    show_boundary(viewer, V, F, bd_scale);
  }
  viewer.append_mesh();
  viewer.data().set_mesh(V, F);

  viewer.launch();
  spdlog::info("#v: {}, #f: {}", V.rows(), F.rows());
  spdlog::info("mesh size in viewer: #v({}), #f({})", viewer.data().V.rows(), viewer.data().F.rows());
  spdlog::debug("viewer.core().view = {}", viewer.core().view);
  spdlog::debug("viewer.core().proj = {}", viewer.core().proj);
  spdlog::debug("viewer.core().norm = {}", viewer.core().norm);
  spdlog::debug("viewer.core().viewport = {}", viewer.core().viewport);
  spdlog::debug("viewer.core().light_position = {}", viewer.core().light_position);
  spdlog::debug("viewer.core().camera_eye = {}", viewer.core().camera_eye);

  camera_info camera = std::make_tuple(viewer.core().view, viewer.core().proj, viewer.core().viewport);

  Eigen::MatrixXd V_edges;
  Eigen::MatrixXi F_edges;

  Eigen::MatrixXd V_sin;
  Eigen::MatrixXi F_sin;

  int red_size = 0;
  int blue_size = 0;
  if (viewer.data_list.size() > 1)
  {
    spdlog::info("adding singularities.");
    auto v_red = viewer.data(1).V;
    auto f_red = viewer.data(1).F;
    auto v_blue = viewer.data(2).V;
    auto f_blue = viewer.data(2).F;

    igl::cat(1, v_red, v_blue, V_sin);
    igl::cat(1, f_red, f_blue, F_sin);
    for (int i = f_red.rows(); i < F_sin.rows(); i++)
    {
      for (int j = 0; j < 3; j++)
      {
        F_sin(i, j) += v_red.rows();
      }
    }
    red_size = f_red.rows();
    blue_size = f_blue.rows();
    spdlog::info("red_size: {}, blue_size: {}", red_size, blue_size);
    if (show_bd && viewer.data_list.size() > 3)
    {
      spdlog::info("adding cuts for display.");
      auto v_bd = viewer.data(3).V;
      auto f_bd = viewer.data(3).F;
      igl::cat(1, V_sin, v_bd, V_edges);
      igl::cat(1, F_sin, f_bd, F_edges);
      for (int i = F_sin.rows(); i < F_edges.rows(); i++)
      {
        for (int j = 0; j < 3; j++)
        {
          F_edges(i, j) += V_sin.rows();
        }
      }
    }
    else // no boundary meshes
    {
      V_edges = V_sin;
      F_edges = F_sin;
    }
  }

  auto mats = get_pt_mat(camera, V, F, V_edges, F_edges, red_size, blue_size, W, H);

  return(std::make_tuple(camera, V_edges, F_edges, red_size, blue_size, std::get<0>(mats), std::get<1>(mats)));

}

#endif
