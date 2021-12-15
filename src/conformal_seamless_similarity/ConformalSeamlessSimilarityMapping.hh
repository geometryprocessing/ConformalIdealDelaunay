// Implementation of Discrete Conformal Seamless Similarity Mapping
// along the lines of
// [Campen and Zorin 2017]: "Similarity Maps and Field-Guided T-Splines: a Perfect Couple"
// [Campen and Zorin 2017]: "On Discrete Conformal Seamless Similarity Maps"
//
// Author: Marcel Campen
//
// Version 1.0
// 21 July 2017


#include <Eigen/Sparse>
#include <set>
#include <queue>
#include <vector>
#include <igl/Timer.h>

class Mesh {
public:
  std::vector<int> n; // next halfedge of halfedge
  std::vector<int> to; // to vertex of halfedge
  std::vector<int> f; // face of halfedge
  std::vector<int> h; // one halfedge of face
  std::vector<int> out; // one outgoing halfedge of vertex

  std::vector<double> l; // discrete metric (length per edge)

  int n_halfedges() { return n.size(); }
  int n_edges() { return n_halfedges()/2; }
  int n_faces() { return h.size(); }
  int n_vertices() { return out.size(); }

  int e(int h) { return h/2; }
  int opp(int h) { return (h%2 == 0) ? (h+1) : (h-1); }
  int v0(int h) { return to[opp(h)]; }
  int v1(int h) { return to[h]; }
  int h0(int e) { return e*2; }
  int h1(int e) { return e*2+1; }
  double sign(int h) { return (h%2 == 0) ? 1.0 : -1.0; }
  
  virtual void init() {};

  virtual bool flip_ccw(int _h)
  {
    int ha = _h;
    int hb = opp(_h);
    int f0 = f[ha];
    int f1 = f[hb];
    if(f0 == f1) return false;
    int h2 = n[ha];
    int h3 = n[h2];
    int h4 = n[hb];
    int h5 = n[h4];
    out[to[hb]] = h4;
    out[to[ha]] = h2;
    f[h4] = f0;
    f[h2] = f1;
    h[f0] = h4;
    h[f1] = h2;
    to[ha] = to[h2];
    to[hb] = to[h4];
    n[h5] = h2;
    n[h3] = h4;
    n[h2] = hb;
    n[h4] = ha;
    n[ha] = h3;
    n[hb] = h5;
    return true;
  }
  
  virtual void get_mesh(std::vector<int>& _n, // next halfedge of halfedge
                        std::vector<int>& _to, // to vertex of halfedge
                        std::vector<int>& _f, // face of halfedge
                        std::vector<int>& _h, // one halfedge of face
                        std::vector<int>& _out) // one outgoing halfedge of vertex
  {
    _n = n;
    _to = to;
    _f = f;
    _h = h;
    _out = out;
  }
  
  template<typename T>
  std::vector<T> interpolate(const std::vector<T>& u)
  {
    return u;
  }
  
  bool is_complex()
  {
    int nh = n_halfedges();
    for(int i = 0; i < nh; i++)
    {
      if(to[i] == to[opp(i)]) return true; //contains loop edge
    }
    int nv = n_vertices();
    for(int i = 0; i < nv; i++)
    {
      std::set<int> onering;
      int h = out[i];
      if(h < 0) continue;
      int k = h;
      do {
        int v = to[k];
        if(onering.find(v) != onering.end()) return true; //contains multi-edges
        onering.insert(v);
        k = n[opp(k)];
      } while(k != h);
    }
    return false;
  }
};


class ConformalSeamlessSimilarityMapping {
public:

  Mesh& m;
  
  std::vector<double> Theta_hat; //target cone angles per vertex
  std::vector<double> kappa_hat; //target holonomy angles per gamma loop
  std::vector< std::vector<int> > gamma; //directed dual loops, represented by halfedges (the ones adjacent to the earlier triangles in the dual loop)

  const double cot_infty = 1e10;

  int n_s;
  int n_e;
  int n_h;
  int n_f;
  int n_v;

  std::vector<double> xi;
  std::vector<double> delta_xi;
  std::vector<double> cot_alpha;
  std::vector<double> alpha;

  Eigen::SparseMatrix<double> A;
  Eigen::VectorXd b;

  std::string log_dir;
  std::string name;
  double eps = 1e-12;

  ConformalSeamlessSimilarityMapping(Mesh& _m, const std::vector<double>& _Theta_hat, const std::vector<double>& _kappa_hat, std::vector< std::vector<int> >& _gamma, std::string _log_dir="", std::string _name="", double _eps=1e-12) : m(_m), Theta_hat(_Theta_hat), kappa_hat(_kappa_hat), gamma(_gamma), log_dir(_log_dir), name(_name), eps(_eps)
  {
    n_s = gamma.size();
    n_e = m.n_edges();
    n_h = m.n_halfedges();
    n_f = m.n_faces();
    n_v = m.n_vertices();

    xi.resize(n_h, 0.0);
    delta_xi.resize(n_h, 0.0);
    cot_alpha.resize(n_h);
    alpha.resize(n_h);
  }
  
  void log(const char* c)
  {
    std::cout << c << std::endl;
  }
  
  void compute_angles() // compute alpha and cot_alpha from scaled edge lengths
  {
    #pragma omp parallel for
    for(int f = 0; f < n_f; f++)
    {
      int hi = m.h[f];
      int hj = m.n[hi];
      int hk = m.n[hj];
      // (following "On Discrete Conformal Seamless Similarity Maps")
      double li = m.l[m.e(hi)] * std::exp(1.0/6.0*(xi[hk]-xi[hj]));
      double lj = m.l[m.e(hj)] * std::exp(1.0/6.0*(xi[hi]-xi[hk]));
      double lk = m.l[m.e(hk)] * std::exp(1.0/6.0*(xi[hj]-xi[hi]));
      // (following "A Cotangent Laplacian for Images as Surfaces")
      double s = (li+lj+lk)/2.0;
      double Aijk4 = 4.0*std::sqrt(std::max(0.0, s*(s-li)*(s-lj)*(s-lk)));
      double Ijk = (-li*li+lj*lj+lk*lk);
      double iJk = (li*li-lj*lj+lk*lk);
      double ijK = (li*li+lj*lj-lk*lk);
      cot_alpha[hi] = Aijk4 == 0.0 ? copysign(cot_infty,Ijk) : (Ijk/Aijk4);
      cot_alpha[hj] = Aijk4 == 0.0 ? copysign(cot_infty,iJk) : (iJk/Aijk4);
      cot_alpha[hk] = Aijk4 == 0.0 ? copysign(cot_infty,ijK) : (ijK/Aijk4);
      
      alpha[hi] = std::acos(std::min(1.0, std::max(-1.0, Ijk/(2.0*lj*lk))));
      alpha[hj] = std::acos(std::min(1.0, std::max(-1.0, iJk/(2.0*lk*li))));
      alpha[hk] = std::acos(std::min(1.0, std::max(-1.0, ijK/(2.0*li*lj))));
    }
  }

  void setup_b() // system right-hand sid
  {
    b.resize(n_v-1 + n_s + n_f-1);
    b.fill(0.0);
    
    std::vector<double> Theta(n_v, 0.0);
    std::vector<double> kappa(n_s, 0.0);
    
    for(int h = 0; h < n_h; h++)
    {
      Theta[m.to[m.n[h]]] += alpha[h];
    }
    #pragma omp parallel for
    for(int r = 0; r < n_v-1; r++)
    {
      b[r] = Theta_hat[r] - Theta[r];
    }
    #pragma omp parallel for
    for(int s = 0; s < n_s; s++)
    {
      kappa[s] = 0.0;
      int loop_size = gamma[s].size();
      for(int si = 0; si < loop_size; si++)
      {
        int h = gamma[s][si];
        int hn = m.n[h];
        int hnn = m.n[hn];
        if(m.opp(hn) == gamma[s][(si+1)%loop_size])
          kappa[s] -= alpha[hnn];
        else if(m.opp(hnn) == gamma[s][(si+1)%loop_size])
          kappa[s] += alpha[hn];
        else std::cerr << "ERROR: loop is broken" << std::endl;
      }
      b[n_v-1+s] = kappa_hat[s] - kappa[s];
    }
  }
  
  void setup_A() // system matrix
  {
    A.resize(n_v-1 + n_s + n_f-1, n_e);
    int loop_trips = 0;
    for(int i = 0; i < n_s; i++)
      loop_trips += gamma[i].size();
    
    typedef Eigen::Triplet<double> Trip;
    std::vector<Trip> trips;
    trips.clear();
    trips.resize(n_h*2 + loop_trips + (n_f-1)*3);
    #pragma omp parallel for
    for(int h = 0; h < n_h; h++)
    {
      int v0 = m.v0(h);
      int v1 = m.v1(h);
      if(v0 < n_v-1) trips[h*2] = Trip(v0, m.e(h), m.sign(h)*0.5*cot_alpha[h]);
      if(v1 < n_v-1) trips[h*2+1] = Trip(v1, m.e(h), -m.sign(h)*0.5*cot_alpha[h]);
    }
    
    int base = n_h*2;
    for(int s = 0; s < n_s; s++)
    {
      int loop_size = gamma[s].size();
      #pragma omp parallel for
      for(int si = 0; si < loop_size; si++)
      {
        int h = gamma[s][si];
        trips[base+si] = Trip(n_v-1 + s, m.e(h), m.sign(h)*0.5*(cot_alpha[h]+cot_alpha[m.opp(h)]));
      }
      base += loop_size;
    }
    
    #pragma omp parallel for
    for(int f = 0; f < n_f-1; f++)
    {
      int hi = m.h[f];
      int hj = m.n[hi];
      int hk = m.n[hj];
      trips[base+f*3] = Trip(n_v-1 + n_s + f, m.e(hi), m.sign(hi));
      trips[base+f*3+1] = Trip(n_v-1 + n_s + f, m.e(hj), m.sign(hj));
      trips[base+f*3+2] = Trip(n_v-1 + n_s + f, m.e(hk), m.sign(hk));
    }
    
    A.setFromTriplets(trips.begin(), trips.end());
  }
  
  double I(int i, int j, int k, double lambda = 0.0)
  {
    return m.l[m.e(i)]*std::exp((-xi[j]-delta_xi[j]*lambda)/2) + m.l[m.e(j)]*std::exp((xi[i]+delta_xi[i]*lambda)/2) - m.l[m.e(k)];
  }

  double firstDegeneracy(int& degen, double lambda)
  {
    bool repeat = true;
    while(repeat)
    {
      repeat = false;
      #pragma omp parallel for
      for(int i = 0; i < n_h; i++)
      {
        int j = m.n[i];
        int k = m.n[j];
        double local_lambda = lambda;
        if(I(i,j,k,local_lambda) < 0.0)
        {
          // root finding (from below) by bracketing bisection
          double lo = 0.0;
          double hi = local_lambda;
          for(int r = 0; r < 100; r++)
          {
            double mid = (lo+hi)*0.5;
            if(I(i,j,k,mid) <= 0.0)
              hi = mid;
            else
              lo = mid;
          }
          
          #pragma omp critical
          {
            if(lo < lambda)
            {
              lambda = lo;
              degen = k;
              repeat = true;
            }
          }
        }
      }
    }
    return lambda;
  }
  
  double avg_abs(const Eigen::VectorXd& v)
  {
    double res = 0.0;
    int v_size = v.size();
    for(int i = 0; i < v_size; i++)
      res += std::abs(b[i]);
    return res/v_size;
  }
  
  double max_abs(const Eigen::VectorXd& v)
  {
    double res = 0.0;
    int v_size = v.size();
    for(int i = 0; i < v_size; i++)
      res = std::max(res, std::abs(b[i]));
    return res;
  }
  
  void compute_metric()
  {

    igl::Timer timer;
    timer.start();

    // double eps = 1e-12; //if max curvature error below eps: consider converged
    int max_iter = 25; //max full Newton steps
    bool converged = false;
    
    log("computing angles");
    compute_angles();
    log("setup b");
    setup_b();
    
    std::vector< std::pair<double,double> > errors;
    int n_flips = 0;
    
    log("starting Newton");
    int degen = -1;
    while(!converged && max_iter > 0)
    {
      double error = max_abs(b);
      if(degen < 0) errors.push_back( std::pair<double,double>(avg_abs(b), error) );
      if(error <= eps) { converged = true; break; }
      
      double diff = avg_abs(b);
      
      log("setup A");
      setup_A();
      
      log("factorize A");
      Eigen::SparseLU< Eigen::SparseMatrix<double> > chol(A);
      log("solve Ax=b");
      Eigen::VectorXd result = chol.solve(b);
      if(chol.info() != Eigen::Success) { log("factorization failed"); return; }
      log("solved");
      
      #pragma omp parallel for
      for(int i = 0; i < n_e; i++)
      {
        delta_xi[i*2] = result[i];
        delta_xi[i*2+1] = -result[i];
      }
      
      log("line search");
      double lambda = 1.0;
      
      int max_linesearch = 25;
      degen = -1;
      while(true) // line search
      {
        log("  checking for degeneration events");
        double first_degen = firstDegeneracy(degen, lambda);
        if(first_degen < lambda)
        {
          lambda = first_degen;
          std::cout << "    degeneracy at lambda = " << lambda << std::endl;
        }
        
        log("  checking for improvement");
        std::vector<double> xi_old = xi;
        
        #pragma omp parallel for
        for(int i = 0; i < n_h; i++)
          xi[i] = xi_old[i] + lambda * delta_xi[i];
        
        compute_angles();
        setup_b();
        
        if(lambda == 0.0)
        {
          converged = true;
          break;
        }
        
        double new_diff = avg_abs(b);
        if(new_diff < diff)
        {
          std::cout << "    OK. (" << diff << " -> " << new_diff << ")" << std::endl;
          break;
        }
        
        lambda *= 0.5;
        if(max_linesearch-- == 0) lambda = 0.0;
        std::cout << "    reduced to    lambda = " << lambda << std::endl;
      }
      
      if(degen < 0) max_iter--; //no degeneration event
      
      if(!converged) //flip edge(s) of degeneracy/ies
      {
        std::set<int> degens;
        if(degen >= 0) degens.insert(m.e(degen));
        #pragma omp parallel for
        for(int i = 0; i < n_h; i++) //check for additional (simultaneous) degeneracies
        {
          int j = m.n[i];
          int k = m.n[j];
          if(I(i,j,k) <= 0.0)
          {
            #pragma omp critical
            {
              degens.insert(m.e(k));
            }
          }
        }
        int n_d = degens.size();
        if(n_d == 1) std::cout << "handling a degeneracy by edge flip" << std::endl;
        else if(n_d > 1) std::cout << "handling " << degens.size() << " degeneracies by edge flips" << std::endl;
        for(std::set<int>::iterator it = degens.begin(); it != degens.end(); it++)
        {
          int e = *it;
          int h = m.h0(e);
          int hl = m.n[h];
          int hr = m.n[m.n[m.opp(h)]];
          
          int hlu = m.n[hl];
          int hru = m.n[m.n[hr]];
          int ho = m.opp(h);
          int hu = h;
          
          double angle = alpha[m.n[hl]]+alpha[m.n[m.opp(h)]];
          double a = m.l[m.e(hl)] * std::exp(xi[hl]/2);
          double b = m.l[m.e(hr)] * std::exp(-xi[hr]/2);
          m.l[e] = std::sqrt(a*a + b*b - 2.0*a*b*std::cos(angle)) / std::exp((xi[hl]-xi[hr])/2); //intrinsic flip (law of cosines)
          
          xi[h] = xi[hl]+xi[hr];
          xi[m.opp(h)] = -xi[h];
          
          if(!m.flip_ccw(h)) { std::cerr << "ERROR: edge could not be flipped." << std::endl; converged = true; break; };
          n_flips++;
          
          if(m.l[e] <= 0.0)
          {
            m.l[e] = 1e-20;
            std::cerr << "WARNING: numerical issue: flipped edge had zero length.";
          }
          
          // adjust gamma loops that contain the flipped edge e
          #pragma omp parallel for
          for(int i = 0; i < n_s; ++i)
          {
            std::vector<int>& li = gamma[i];
            int n = li.size();
            for(int j = 0; j < n; ++j)
            {
              int hij = li[j];
              int hij1 = li[(j+1)%n];
              int hij2 = li[(j+2)%n];
              
              bool change = true;
              
              if(hij == hru && hij1 == m.opp(hr)) li.insert(li.begin()+j+1,ho);
              else if(hij == hru && hij1 == hu && hij2 == m.opp(hl)) li[(j+1)%n] = ho;
              else if(hij == hru && hij1 == hu && hij2 == m.opp(hlu)) li.erase(li.begin()+((j+1)%n));
              
              else if(hij == hr && hij1 == m.opp(hru)) li.insert(li.begin()+j+1,hu);
              else if(hij == hr && hij1 == hu && hij2 == m.opp(hlu)) li[(j+1)%n] = hu;
              else if(hij == hr && hij1 == hu && hij2 == m.opp(hl)) li.erase(li.begin()+((j+1)%n));
              
              else if(hij == hl && hij1 == m.opp(hlu)) li.insert(li.begin()+j+1,hu);
              else if(hij == hl && hij1 == ho && hij2 == m.opp(hru)) li[(j+1)%n] = hu;
              else if(hij == hl && hij1 == ho && hij2 == m.opp(hr)) li.erase(li.begin()+((j+1)%n));
              
              else if(hij == hlu && hij1 == m.opp(hl)) li.insert(li.begin()+j+1,ho);
              else if(hij == hlu && hij1 == ho && hij2 == m.opp(hr)) li[(j+1)%n] = ho;
              else if(hij == hlu && hij1 == ho && hij2 == m.opp(hru)) li.erase(li.begin()+((j+1)%n));
              
              else change = false;
              
              if(change) // cleanup "cusps" in loop
              {
                n = li.size();
                int j0 = j;
                int j1 = (j+1)%n;
                int j2 = (j+2)%n;
                if(li[j0] == m.opp(li[j1]))
                {
                  if(j1 < j0) std::swap(j0,j1);
                  li.erase(li.begin()+j1);
                  li.erase(li.begin()+j0);
                }
                else if(li[j1] == m.opp(li[j2]))
                {
                  if(j2 < j1) std::swap(j1,j2);
                  li.erase(li.begin()+j2);
                  li.erase(li.begin()+j1);
                }
              }
            }
          }
        }
        if(n_d > 0) //recompute angles after flipping
        {
          compute_angles();
          setup_b();
          
          //sanity check
          for(int i = 0; i < n_h; i++)
          {
            int j = m.n[i];
            int k = m.n[j];
            double indicator = I(i,j,k);
            if(indicator <= 0.0)
            {
              #pragma omp critical
              {
                if(indicator == 0.0) std::cerr << "WARNING: numerical issue: triangle("<<i<<", "<<j<<", "<<k<<") is degenerate after Newton step." << std::endl;
                if(indicator < 0.0) std::cerr << "ERROR: numerical issue: triangle("<<i<<", "<<j<<", "<<k<<") is violating after Newton step." << std::endl;
                degens.insert(m.e(k));
              }
            }
          }
        }
      }
    }
    
    double error = max_abs(b);
    
    if(error > eps) std::cerr << "WARNING: the final max error is larger than desired ("<<error<<")." << std::endl;
    
    std::cout << "\nSTATISTICS:\n";
    std::cout << "Flips: " << n_flips << std::endl;
    std::cout << "Error Decay: (iter, avg, max)" << std::endl;
    for(size_t i = 0; i < errors.size(); i++)
    {
      std::cout << i << ": " << errors[i].first << "  " << errors[i].second << std::endl;
    }
    std::cout << std::endl;
    if(n_flips > 0)
      std::cout << "HINT: The given mesh m has been modified by edge flips. Get the modified mesh by m.get_mesh(...)" << std::endl;
    if(n_flips > 0 && m.is_complex())
      std::cout << "HINT: The modified mesh is non-simple (e.g. contains a loop edge or multiple edges between a pair of vertices). Beware that many mesh data structures and libraries do not support this appropriately." << std::endl;
    std::cout << std::endl;

    // write n_flip and time spent to log
    auto total_time = timer.getElapsedTime();

    std::ofstream mf;
    mf.open(log_dir+"/summary_similarity.csv", std::ios_base::app);
    std::ifstream nf(log_dir+"/summary_similarity.csv");
    if (nf && nf.peek() == std::ifstream::traits_type::eof() ){
      // if stats file is empty then add column names
      nf.close();
      mf << "name, n_flips, max_error_similarity, Th_hat_preset, time\n" ;
    }

    mf << name << ", " << n_flips << ", " << error <<"," << Theta_hat[0] << ","<< total_time << std::endl;
    mf.close();

  }


  void compute_layout(std::vector<double>& u, std::vector<double>& v) //metric -> parametrization
  {
    std::vector<double> phi(n_h);
    
    u.resize(n_h);
    v.resize(n_h);
    
    //set starting point
    int h = 0;
    phi[h] = 0.0;
    u[h] = 0.0;
    v[h] = 0.0;
    h = m.n[h];
    phi[h] = xi[h];
    u[h] = m.l[m.e(h)]*std::exp(phi[h]/2);
    v[h] = 0.0;
    
    // layout the rest of the mesh by BFS
    std::vector<bool> visited(n_f, false);
    std::queue<int> q;
    q.push(h);
    visited[m.f[h]] = true;
    while(!q.empty())
    {
      h = q.front();
      q.pop();
      
      int hn = m.n[h];
      int hp = m.n[hn];
      
      phi[hn] = phi[h] + xi[hn];
      
      double len = m.l[m.e(hn)] * std::exp((phi[h]+phi[hn])/2);
      
      double ud = u[hp]-u[h];
      double vd = v[hp]-v[h];
      double d = std::sqrt(ud*ud + vd*vd);
      double co = std::cos(alpha[hp]);
      double si = std::sin(alpha[hp]);
      
      u[hn] = u[h] + (co*ud + si*vd)*len/d;
      v[hn] = v[h] + (co*vd - si*ud)*len/d;
      
      int hno = m.opp(hn);
      int hpo = m.opp(hp);
      if(!visited[m.f[hno]])
      {
        visited[m.f[hno]] = true;
        phi[hno] = phi[h];
        phi[m.n[m.n[hno]]] = phi[hn];
        u[hno] = u[h];
        v[hno] = v[h];
        u[m.n[m.n[hno]]] = u[hn];
        v[m.n[m.n[hno]]] = v[hn];
        q.push(hno);
      }
      if(!visited[m.f[hpo]])
      {
        visited[m.f[hpo]] = true;
        phi[hpo] = phi[hn];
        phi[m.n[m.n[hpo]]] = phi[hp];
        u[hpo] = u[hn];
        v[hpo] = v[hn];
        u[m.n[m.n[hpo]]] = u[hp];
        v[m.n[m.n[hpo]]] = v[hp];
        q.push(hpo);
      }
    }
  }

  void compute(std::vector<double>& u, std::vector<double>& v) //main method
  {
    compute_metric();
    compute_layout(u, v);
  }

};
