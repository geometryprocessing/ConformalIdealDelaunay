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

#include "plot.h"
#include <igl/avg_edge_length.h>
#include <igl/boundary_loop.h>
#include <igl/doublearea.h>
#include <cstdlib>

void plot_singularity_sphere(
  igl::opengl::glfw::Viewer& viewer,
  const Eigen::MatrixXd& V_s,
  const Eigen::MatrixXi& F,
  const Eigen::VectorXd& S,
  double point_scale
)
{
  Eigen::MatrixXd v_sphere;
  Eigen::MatrixXd f_sphere;

  igl::readOBJ("./sphere_6.obj", v_sphere, f_sphere);

  auto V = V_s;
  if(V.cols() == 2){
    V.conservativeResize(V.rows(),3);
    V.col(2).setZero();
  }
  int n_pos = 0, n_neg = 0;
  for(int i=0;i<S.rows();i++){
    if(S(i) > 0)
      n_pos++;
    else if(S(i) < 0)
      n_neg++;
  }
  
  double avg = igl::avg_edge_length(V,F);
  
  Eigen::MatrixXd point_mesh_V_red(n_pos*v_sphere.rows(), 3);
  Eigen::MatrixXi point_mesh_F_red(n_pos*f_sphere.rows(), 3);
  Eigen::MatrixXd point_mesh_V_blue(n_neg*v_sphere.rows(), 3);
  Eigen::MatrixXi point_mesh_F_blue(n_neg*f_sphere.rows(), 3);

  int index_red_v = 0, index_red_f = 0;
  int index_blue_v = 0, index_blue_f = 0;
  double r = avg * 0.2 * point_scale;

  for(int i=0;i<S.rows();i++){
    if(S(i) > 0){
      Eigen::RowVector3d pos = V.row(i);
      for (int it = 0; it < v_sphere.rows(); it++)
      {
        point_mesh_V_red.row(index_red_v + it) = pos + v_sphere.row(it) * r;
      }
      for (int it = 0; it < f_sphere.rows(); it++)
      {
        for (int j = 0; j < 3; j++)
        {
          point_mesh_F_red(index_red_f + it, j) = f_sphere(it, j) + index_red_v;
        }
      }
      index_red_v += v_sphere.rows();
      index_red_f += f_sphere.rows();
    }else if(S(i) < 0){
      Eigen::RowVector3d pos = V.row(i);
      for (int it = 0; it < v_sphere.rows(); it++)
      {
        point_mesh_V_blue.row(index_blue_v + it) = pos + v_sphere.row(it) * r;
      }
      for (int it = 0; it < f_sphere.rows(); it++)
      {
        for (int j = 0; j < 3; j++)
        {
          point_mesh_F_blue(index_blue_f + it, j) = f_sphere(it, j) + index_blue_v;
        }
      }
      index_blue_v += v_sphere.rows();
      index_blue_f += f_sphere.rows();
    }
  }
  viewer.append_mesh();
  viewer.data().set_mesh(point_mesh_V_red,point_mesh_F_red);
  viewer.data().show_lines = false;
  Eigen::MatrixXd color_red(point_mesh_F_red.rows(),3);
  for(int i=0;i<color_red.rows();i++)
    color_red.row(i) << 1,0,0;
  viewer.data().set_colors(color_red);
  
  viewer.append_mesh();
  viewer.data().set_mesh(point_mesh_V_blue,point_mesh_F_blue);
  viewer.data().show_lines = false;
  Eigen::MatrixXd color_blue(point_mesh_F_blue.rows(),3);
  for(int i=0;i<color_blue.rows();i++)
    color_blue.row(i) << 0,0,1;
  viewer.data().set_colors(color_blue);

}

void cylinder(
  Eigen::RowVector3d& v0,
  Eigen::RowVector3d& v1,
  Eigen::MatrixXd& V0,
  Eigen::MatrixXi& F0,
  double radius_top,
  double radius_bot,
  int index
){
  Eigen::RowVector3d n1;
  Eigen::RowVector3d v1v0 = v0-v1;
  if(v1v0(0) != 0)
    n1 << (-v1v0(1)-v1v0(2))/v1v0(0), 1, 1;
  else if(v1v0(1) != 0)
    n1 << 1, (-v1v0(0)-v1v0(2))/v1v0(1), 1;
  else
    n1 << 1, 1, (-v1v0(0)-v1v0(1))/v1v0(2);
  n1.normalize();
  
  Eigen::RowVector3d n2 = ((v0-v1).cross(n1)).normalized();
  Eigen::RowVector3d p1 = v0 + n1 * radius_top;
  Eigen::RowVector3d q1 = v1 + n1 * radius_bot;
  Eigen::RowVector3d p2 = v0 + n2 * radius_top;
  Eigen::RowVector3d q2 = v1 + n2 * radius_bot;
  Eigen::RowVector3d p3 = v0 - n1 * radius_top;
  Eigen::RowVector3d q3 = v1 - n1 * radius_bot;
  Eigen::RowVector3d p4 = v0 - n2 * radius_top;
  Eigen::RowVector3d q4 = v1 - n2 * radius_bot;
  V0.block(index,0,8,3)<<p1,p2,p3,p4,q1,q2,q3,q4;
  F0.block(index,0,8,3)<<1,0,5,0,4,5,0,3,7,0,7,4,3,2,7,7,2,6,2,1,5,2,5,6;
  F0.block(index,0,8,3)<<F0.block(index,0,8,3).array()+index;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> get_edges(
  const Eigen::MatrixXd& V_s,
  const Eigen::MatrixXi& F,
  const std::vector<std::vector<int>>& E,
  float thick
){
  Eigen::MatrixXd V = V_s;
  if(V.cols() == 2){
    V.conservativeResize(V.rows(),3);
    V.col(2).setZero();
  }
  Eigen::MatrixXd edge_mesh_V(E.size()*8,3);
  Eigen::MatrixXi edge_mesh_F(E.size()*8,3);

  Eigen::VectorXd A;
  igl::doublearea(V_s, F, A);
  assert(A.minCoeff() > 0.0 && "expecting non-zero area for getting face normals");
  
  double factor = 10;
  double avg = factor*igl::avg_edge_length(V,F);
  
  int index = 0; // ptr to bottom of matrix
  if(true){
    for(auto e: E){
      int u = e[0];
      int v = e[1];
      Eigen::RowVector3d v0 = V.row(u);
      Eigen::RowVector3d v1 = V.row(v);
      cylinder(v0, v1, edge_mesh_V, edge_mesh_F, avg*0.01*thick, avg*0.01*thick, index);
      index += 8;
    }
  }
  
  return std::make_tuple(edge_mesh_V,edge_mesh_F);

}

void plot_edges(
  igl::opengl::glfw::Viewer& viewer,
  const Eigen::MatrixXd& V_s,
  const Eigen::MatrixXi& F,
  const Eigen::RowVector3d& c,
  std::set<std::pair<int,int>> & E,
  float thick, 
  bool is_dual
){
  Eigen::MatrixXd V = V_s;
  if(V.cols() == 2){
    V.conservativeResize(V.rows(),3);
    V.col(2).setZero();
  }
  Eigen::MatrixXd edge_mesh_V(E.size()*8,3);
  Eigen::MatrixXi edge_mesh_F(E.size()*8,3);

  Eigen::VectorXd A;
  igl::doublearea(V_s, F, A);
  assert(A.minCoeff() > 0.0 && "expecting non-zero area for getting face normals");
  
  double factor = 10;
  double avg = factor*igl::avg_edge_length(V,F);
  
  int index = 0; // ptr to bottom of matrix
  if(!is_dual){
    for(auto e: E){
      int u = e.first;
      int v = e.second;
      Eigen::RowVector3d v0 = V.row(u);
      Eigen::RowVector3d v1 = V.row(v);
      cylinder(v0, v1, edge_mesh_V, edge_mesh_F, avg*0.01*thick, avg*0.01*thick, index);
      index += 8;
    }
  }else{
    for(auto e: E){
      int f0 = e.first;
      int f1 = e.second;
      Eigen::RowVector3d v0 = (V.row(F(f0,0)) + V.row(F(f0,1)) + V.row(F(f0,2)))/3;
      Eigen::RowVector3d v1 = (V.row(F(f1,0)) + V.row(F(f1,1)) + V.row(F(f1,2)))/3;
      cylinder(v0, v1, edge_mesh_V, edge_mesh_F, avg*0.001, avg*0.001,index);
      index += 8;
    }
  }
  viewer.append_mesh();
  Eigen::MatrixXd color(edge_mesh_F.rows(),3);
  for(int i=0;i<color.rows();i++)
    color.row(i) << c;
  viewer.data().set_mesh(edge_mesh_V,edge_mesh_F);
  viewer.data().show_lines = false;
  viewer.data().set_colors(color);
}

void show_boundary(
  igl::opengl::glfw::Viewer& viewer,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  double w
){
  
  std::vector<std::vector<int>> bds;
  igl::boundary_loop(F, bds);
  
  for(auto bd : bds){
    if(bd.size() <= 1){ continue; }
    std::set<std::pair<int,int>> bd_e;
    for(int i=0;i<bd.size();i++){
      int v0 = bd[i];
      int v1 = bd[(i+1)%bd.size()];
      bd_e.insert(std::make_pair(v0, v1));
      assert(v0 < V.rows() && v1 < V.rows());
    }
    if(!bd_e.empty())
      plot_edges(viewer, V, F, Eigen::RowVector3d(0,0,0), bd_e, w);
    viewer.data().show_lines = true;
  }
}