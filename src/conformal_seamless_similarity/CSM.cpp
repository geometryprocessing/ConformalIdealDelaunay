#include <igl/readOBJ.h>
#include <igl/readCSV.h>
#include <igl/matrix_to_list.h>
#include <igl/edge_topology.h>
#include "ConformalSeamlessSimilarityMapping.hh"
#include "../util/argh.h"

void obj_to_mesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Mesh& m){

  Eigen::MatrixXi EV,FE,EF;
  igl::edge_topology(V, F, EV, FE, EF);

  int n_v = V.rows();
  int n_f = F.rows();
  int n_h = EV.rows()*2;
  
  m.out.resize(n_v);  // one halfedge out from vt_i
  m.n.resize(n_h); // next halfedge of he_i
  m.to.resize(n_h);// vertex that he_i is pointing to
  m.f.resize(n_h); // face of he_i
  m.h.resize(n_f);    // one he_i for f_i (first)
  m.l.resize(n_h/2);   // length of edge ue_i

  for(int i=0;i<EV.rows();i++){
    // row_i un-directed edge with EV(i,0) < EV(i,1)
    int v0 = EV(i,0);
    int v1 = EV(i,1);
    int h0 = i*2;   // halfedge (u,v)
    int h1 = i*2+1; // halfedge (v,u)
    m.out[v0] = h0;
    m.out[v1] = h1;
    m.to[h0] = v1;
    m.to[h1] = v0;
    m.f[h0] = EF(i,0);
    m.f[h1] = EF(i,1);
    if(EF(i,0) != -1)
      m.h[EF(i,0)] = h0;
    if(EF(i,1) != -1)
      m.h[EF(i,1)] = h1;
    m.l[i] = (V.row(v0)-V.row(v1)).norm();
  }

  auto ue_he_match_dir = [](const Eigen::MatrixXi& FE, const Eigen::MatrixXi& EV, const Eigen::MatrixXi& F, int f, int k){
    assert(f >= 0);
    int ue = FE(f,k);
    assert(ue >= 0 && ue < EV.rows());
    return (EV(ue,0) == F(f,k) && EV(ue,1) == F(f,(k+1)%3));
  };
  
  // compute the halfedge id for every edge in F
  Eigen::MatrixXi D;
  D.setZero(F.rows(),3);
  for(int i=0;i<F.rows();i++){
    for(int k=0;k<3;k++){
      if(ue_he_match_dir(FE, EV, F, i, k)){ // if the halfedge and undirected edge are in the same direction
        D(i,k) = FE(i,k)*2;
      }else
        D(i,k) = FE(i,k)*2+1;
    }
  }
  
  // assign next halfedges relations
  for(int f0=0;f0<F.rows();f0++){
    for(int k0=0;k0<3;k0++){
      int curr = D(f0,k0);
      int next = D(f0,(k0+1)%3);
      m.n[curr] = next;
    }
  }

}

int main(int argc, char* argv[]){

  auto cmdl = argh::parser(argc, argv, argh::parser::PREFER_PARAM_FOR_UNREG_OPTION);

  std::string data_dir, log_dir;
  std::string model, name;
  std::string th_file;
  int postfix;
  int factor = -10;
  cmdl("-m") >> model;
  cmdl("-d") >> data_dir;
  cmdl("-o") >> log_dir;
  cmdl("-t") >> th_file;
  cmdl("-f") >> factor;
  cmdl("-p") >> postfix;
  double eps = std::pow(10, factor);

  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  igl::readOBJ(data_dir + "/" + model, V, F);
  name = model.substr(0, model.find_last_of('.'));
  name = name.substr(name.find_last_of('/')+1);

  Mesh m;
  obj_to_mesh(V, F, m);

  Eigen::MatrixXd Theta_hat_vec;
  igl::readCSV(th_file, Theta_hat_vec);
  
  std::vector<double> Th_hat;
  igl::matrix_to_list(Theta_hat_vec, Th_hat);

  std::vector<double> kappa_hat;
  std::vector<std::vector<int>> gamma;
  auto cssm = ConformalSeamlessSimilarityMapping(m, Th_hat, kappa_hat, gamma, log_dir, name+"_"+std::to_string(postfix)+"_float", eps);

  cssm.compute_metric();

}
