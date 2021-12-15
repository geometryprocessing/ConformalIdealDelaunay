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

// Authors of OverlayMesh: Nils Affing, Leyi Zhu, Marcel Campen
//
// OverlayProblem::OverlayMesh can be used by the algorithm (that performs edge flips) like a standard halfedge mesh data structure. Internally, however, it keeps track of the (polygonal) overlay of original mesh and current (flip-modified) mesh.
//
// In the end, the polygon mesh (not triangle mesh) that is the overlay, can be read from the halfedge representation (n,to,f,h,out) of OverlayProblem::OverlayMesh.
// It's edges (not halfedges) are tagged in OverlayProblem::OverlayMesh::edge_type as part of the origional, current, or both meshes.
// It's vertices are tagged in OverlayProblem::OverlayMesh::vertex_type as original or segment. Note that the original and the current mesh contain only the "original" vertices, while the overlay additionally contains the "segment" vertices (which are crossings of original and current edges).
//
// Vertices, halfedges, faces that are deleted as the overlay undergoes changes due to flips are kept in the data structure arrays (n, to, f, h, out, etc.) and marked by "-1". At any point (or in the end) one may call OverlayProblem::OverlayMesh::garbage_collection() to remove these entries (freeing space) while reindexing everything to a contiguous-index representation.
//
// (x,y,z)-coordinates of the original mesh's vertices can be interpolated onto all vertices of the overlay mesh by calling OverlayProblem::OverlayMesh::interpolate_along_o_bc() in the end.
// (u,v)-coordinates associated with the current mesh's halfedges (referring to their to-corners) can be interpolated onto all halfedges of the overlay mesh by calling OverlayProblem::OverlayMesh::interpolate_along_c_bc() in the end.

#ifndef OVERLAY_MESH_HH
#define OVERLAY_MESH_HH

#include <vector>
#include <map>
#include <stdlib.h>
#include <algorithm>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>
#include "BarycenterMapping.hh"

#ifdef WITH_MPFR
#include <unsupported/Eigen/MPRealSupport>
#endif

#define DEBUG 0

#define ORIGINAL_VERTEX 0
#define SEGMENT_VERTEX 1

#define ORIGINAL_EDGE 0
#define CURRENT_EDGE 1
#define ORIGINAL_AND_CURRENT_EDGE 2

static const int SAFETY_LIMIT = 9999999;

namespace OverlayProblem{
    template<typename Scalar>
    class Mesh {

    public:
        std::vector<int> n;                 // Next Halfedge of Halfedge, in Counterclockwise Order
        std::vector<int> to;                // To Vertex of Halfedge
        std::vector<int> f;                 // Face of Halfedge
        std::vector<int> h;                 // One Halfedge of Face
        std::vector<int> out;               // One outgoing Halfedge of vertex;
        std::vector<int> opp;               // Opposite halfedge of halfedge
        
        std::vector<Scalar> l;              // discrete metric (length per edge)
  
        std::vector<char> type;             // for symmetric double cover, the halfedges from the original mesh with 1 and the new mesh with 2
        std::vector<char> type_input;       // store the input type for labeling
        std::vector<int> R;                 // reflection map for halfedges
        std::vector<int> v_rep;             // identification map of vertices for tufted cover

        std::vector<Scalar> Th_hat;         

        // vectors to store sampled points
        std::vector<Pt<Scalar>> pts;
        std::vector<std::vector<int>> pt_in_f;

        int n_halfedges() const { return n.size(); }
        int n_edges() const { return n_halfedges() / 2; }
        int n_faces() const { return h.size(); }
        int n_vertices() const { return out.size(); }
        int n_ind_vertices() const { return Th_hat.size(); }
        int e(int h) const { return (h < opp[h]) ? h : opp[h]; }
        int v0(int h) const { return to[opp[h]]; }
        int v1(int h) const { return to[h]; }
        int h0(int e) const { return (e < opp[e]) ? e : opp[e]; }
        int h1(int e) const { return (e < opp[e]) ? opp[e] : e; }
        double sign(int h) const { return (h < opp[h]) ? 1.0 : -1.0; }
        
        Mesh()
        {
            if(type.size() != n.size()) type.resize(n.size(), 0);
        }

        /**
         * Pass sampled points data
         * @param fids points' Face ids
         * @param bcs points' Barycentric Coordinates
         */
        void init_pts(std::vector<int> fids, std::vector<Eigen::Matrix<Scalar, 3, 1>> bcs)
        {
          pt_in_f.resize(h.size());

          for (int i = 0; i < fids.size(); i++)
          {
            Pt<Scalar> pt;
            pt.bc = bcs[i];
            pt.f_id = fids[i];
            pts.push_back(pt);
            pt_in_f[pt.f_id].push_back(i);
          }
        };

        /**
         * Do a Counter-ClockWise edge flip
         * @param _h The Halfedge to fliop
         * @param Ptolemy Whether the flip is a PtolemyFlip
         */
        virtual bool flip_ccw(int _h, bool Ptolemy = true) {
            int ha = _h;
            int hb = opp[_h];

            int va = to[ha];
            int vb = to[hb];

            int f0 = f[ha];
            int f1 = f[hb];
            if (f0 == f1) return false;

            int h2 = n[ha];
            int h3 = n[h2];
            int h4 = n[hb];
            int h5 = n[h4];

            // recompute the bc
            if (pt_in_f.size() > 0)
            {
              if (Ptolemy)
              {
                recompute_bc_hyper(_h, n, h, f, opp, l, pts, pt_in_f);
              }
              else
              {
                recompute_bc_original(_h, n, h, f, opp, l, pts, pt_in_f);
              }
            }

            if (Ptolemy)
            {
              l[h0(_h)] = l[h1(_h)] = (l[e(h2)] * l[e(h4)] + l[e(h3)] * l[e(h5)]) / l[e(_h)];
            }
            else
            {

              Scalar l_ab = l[e(ha)];
              Scalar l_ba = l_ab;
              Scalar l_bc = l[e(h2)];
              Scalar l_ca = l[e(h3)];
              Scalar l_ad = l[e(h4)];
              Scalar l_db = l[e(h5)];

              Scalar tan_alpha_2 = compute_tan_half(l_bc, l_ab, l_ca);
              Scalar tan_delta_2 = compute_tan_half(l_db, l_ba, l_ad);
              Scalar tan_sum_2 = (tan_alpha_2 + tan_delta_2) / (1 - tan_alpha_2 * tan_delta_2);
              Scalar cos_sum = (1 - tan_sum_2 * tan_sum_2) / (1 + tan_sum_2 * tan_sum_2);

              Scalar l_cd = sqrt(l_ad * l_ad + l_ca * l_ca - 2 * l_ad * l_ca * cos_sum);
              l[h0(_h)] = l[h1(_h)] = l_cd;
            }

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

        /**
         * Computes the previous Halfedge of the given Halfedge.
         * @param h The given Halfedge.
         * @return The previous Halfedge.
         */
        int compute_prev(int h) {
            int iterations = 0;

            int current = h;
            while (n[current] != h) {
                current = n[current];

                iterations++;
                if (iterations >= 100) {
                    printf("Could not compute Previous Halfedge of %i!\n", h);
                    return h;
                }
            }

            return current;
        }

        /**
         * Return the Reference of the Current Mesh
         * @return the Reference of the Current Mesh
         */
        virtual Mesh<Scalar>& cmesh(){
          return *this;
        };

        /** 
         * Get the mesh connectivity
         * @param _n next halfedge of halfedge
         * @param _to to vertex of halfedge
         * @param _f face of halfedge
         * @param _out one outgoing halfedge of vertex
         * @param _opp opposite halfedge of halfedge
         */
        virtual void get_mesh(std::vector<int>& _n, // next halfedge of halfedge
                        std::vector<int>& _to, // to vertex of halfedge
                        std::vector<int>& _f, // face of halfedge
                        std::vector<int>& _h, // one halfedge of face
                        std::vector<int>& _out, // one outgoing halfedge of vertex
                        std::vector<int>& _opp) // opposite halfedge of halfedge
        {
          _n = n;
          _to = to;
          _f = f;
          _h = h;
          _out = out;
          _opp = opp;
        }
        
    };

    template<typename Scalar>
    class OverlayMesh: public Mesh<Scalar> {
    public:

        std::vector<int> prev;              // Previous Halfedge of Halfedge, in Clockwise Order

        std::vector<int> first_segment;     // The First Segment of the Halfedge
        std::vector<int> origin;            // The Origin-Halfedges of all Halfedges
        std::vector<int> origin_of_origin;  // The Origin-Halfedges of origin edges for labeling

        std::vector<int> vertex_type;       // The Vertex-Type of the Vertex (0 = Current, 1 = Overlay)
        std::vector<int> edge_type;         // The Type of the O-Edge, O == 0, C == 1, OC == 2

        std::vector< std::vector<Scalar> > seg_bcs; // seg_bcs[h] := bc of to[h] on origin[h]

        Mesh<Scalar> _m;                     // current non-overlay mesh
        bool bypass_overlay = false;         // when this flag is marked it means higher precision is needed for layout & overlay

        /**
         * Return the Reference of the Current Mesh
         * @return The Reference fot the Current Mesh
         */
        Mesh<Scalar>& cmesh()
        {
          return _m;
        }
        
        /**
         *  Init overlaymesh
         * @param m Original Mesh
         */
        OverlayMesh(const Mesh<Scalar>& m): Mesh<Scalar>(), _m(m) {

            this->n = _m.n;
            this->to = _m.to;
            this->f = _m.f;
            this->h = _m.h;
            this->opp = _m.opp;
            this->out = _m.out;

            this->prev = {};
            for (int h = 0; h < _m.n_halfedges(); h++) {
                this->prev.push_back(_m.compute_prev(h));
            }
            
            first_segment = {};
            origin = {};
            for (int h = 0; h < _m.n_halfedges(); h++) {
                first_segment.push_back(h);
                origin.push_back(h);
            }
            origin_of_origin = origin;

            vertex_type = {};
            //0 == O, 1 == 0
            for (int v = 0; v < _m.n_vertices(); v++) {
                vertex_type.push_back(ORIGINAL_VERTEX);
            }

            edge_type = {};
            for (int he = 0; he < _m.n_halfedges(); he++) {
                //O == OC, C == 1, O == 2
                edge_type.push_back(ORIGINAL_AND_CURRENT_EDGE);
            }

            
            seg_bcs = {};
            for (int e = 0; e < _m.n.size(); e++){
              seg_bcs.push_back(std::vector<Scalar>{0.0, 1.0});
            }
        }

        /**
         * Do a Counter-ClockWise edge flip
         * @param _h The Halfedge to fliop
         * @param Ptolemy Whether the flip is a PtolemyFlip
         */
        bool flip_ccw(int _h, bool Ptolemy = true)
        {
          if (this->n.size() > SAFETY_LIMIT)
          {
            garbage_collection();
          }
          if (_m.flip_ccw(_h, Ptolemy) == false)
          {
            return false;
          }
          if(!bypass_overlay)
            return o_flip_ccw(&_m, _h, Ptolemy);
          else
            return true;
        };

        /**
         * Checks the Integrity of all Datastructures located in this Mesh.
         * @param _m The Original Mesh related to this Mesh.
         */
        bool check(Mesh<Scalar>* _m);

        /**
         * Flips the given Current Halfedge in the OverlayMesh.
         * @param _m The Current Mesh.
         * @param _h The Current Halfedge.
         */
        bool o_flip_ccw(Mesh<Scalar>* _m, int _h, bool Ptolemy);
        

        // update seg_bcs during flip_ccw
        void update_bc_bd(int _hjk, int _hki, int _hil, int _hlj, Scalar S0, Scalar S1);
        void update_bc_intersection(Mesh<Scalar>* _m, int _h, bool Ptolemy);

        /** find the end origin vertex of segment halfedge _h
         * @param _h The segment halfedge
         * @return The end origin vertex of segment halfedge
         */

        int find_end_origin(int _h)
        {
          while (vertex_type[this->to[_h]] != ORIGINAL_VERTEX)
          {
            _h = this->n[this->opp[this->n[_h]]];
          }
          return this->to[_h];
        }

        /** distance to the next origin vertex
         * @param _h The segment halfedge
         * @return The distance to the end origin vertex of segment halfedge
         */
        int dist_to_next_origin(int _h)
        {
          int dist = 1;
          while (vertex_type[this->to[_h]] != ORIGINAL_VERTEX)
          {
            _h = this->n[this->opp[this->n[_h]]];
            dist++;
          }
          return dist;
        }

        /** find the next clockwise out halfedge of v0(_h)
         * @param _h The halfedge
         * @return The next clockwise out halfedge of v0(_h)
         */
        int next_out(int _h)
        {
          return this->n[this->opp[_h]];
        }

        /** find the next segment halfedge
         * @param _h The segment halfedge
         * @return the next segment halfedge
         */
        int next_segment(int _h)
        {
          if (valence(_h) <= 2)
          {
            return this->n[_h];
          }
          int cur = this->n[this->opp[this->n[_h]]];
          while (edge_type[this->e(cur)] == ORIGINAL_EDGE)
          {
            cur = this->n[this->opp[this->n[cur]]];
          }
          return cur;
        }
        
        /**
         * Returns the Last-Segment of the given Original Halfedge.
         * @param _h The Original Halfedge.
         * @return Returns the ID of the last Segment/Halfedge in the current Mesh.
         */
        int last_segment(int _h) {
            //Start with the first Segment
            int current = first_segment[_h];

            //Move along the Segments until there is a Original Vertex
            while (vertex_type[this->to[current]] == SEGMENT_VERTEX) {
                int v = this->to[current];

                //Next Segment
                if (this->valence(current) <= 2) {
                    current = this->n[current];
                } else {
                    current = next_segment(current);
                }
            }

            //Return the Segment pointing to the Original Vertex.
            return current;
        }

        /**
         * Returns the Number of Halfedges of a Face.
         * @param f The Id of the Face
         * @return Returns the Number of Halfedges of a Face.
         */
        int face_edge_count(int f) {
            int start = this->h[f];
            if (start == -1) return -1;

            //Print every Halfedge of the Face
            int count = 0;
            int current = start;
            do {
                current = this->n[current];
                count++;

                if (count >= 100) {
                    return 100;
                }
            } while (current != start);
            return count;
        }

        /**
         * Computes the Valence of the given Vertex.
         * @param v The Vertex.
         * @return Returns the Valence of the Vertex.
         */
        int valence(int to_v) {
            if (this->n[to_v] == this->opp[to_v]) return 1;
            int result = 0;

            //Start with one outgoing Edge
            int start = this->n[to_v];

            //Iterate all outgoing Halfedges
            int current = start;
            int iterations = 0;
            do {
                current = this->n[this->opp[current]];
                result++;

                iterations++;
                if (iterations >= 100) {
                    return 100;
                }
            } while (current != start);

            //Return the Sum of all outgoing Halfedges
            return result;
        }

        /**
         * Creates a new Face by extending the related Arrays.
         * @return The ID of the new Face.
         */
        int create_face() {
            int id = this->n_faces();
            this->h.push_back(-1);
            return id;
        }

        /**
         * Creates a new Vertex by extending the related Arrays.
         * @return The ID of the new Vertex.
         */
        int create_vertex() {
            int id = this->n_vertices();
            this->out.push_back(-1);

            //Every new created Vertex has to be a Segment-Vertex
            vertex_type.push_back(SEGMENT_VERTEX);

            return id;
        }

        /**
         * Creates a new Halfedge by extending the related Arrays. The Origin-Halfedge is set to -1 by default.
         * @return The ID of the new Halfedge.
         */
        int create_halfedge(int type, int o0, int o1) {

            int id = this->n_halfedges();
            this->n.push_back(-1);
            this->n.push_back(-1);

            prev.push_back(-1);
            prev.push_back(-1);

            this->to.push_back(-1);
            this->to.push_back(-1);

            this->f.push_back(-1);
            this->f.push_back(-1);

            this->opp.push_back(id + 1);
            this->opp.push_back(id);

            // first_segment.push_back(id);
            // first_segment.push_back(id + 1);

            origin.push_back(o0);
            origin.push_back(o1);

            origin_of_origin.push_back(0);
            origin_of_origin.push_back(0);

            edge_type.push_back(type);
            edge_type.push_back(type);

            // add new bcs
            seg_bcs.push_back(std::vector<Scalar>{0.0, 1.0});
            seg_bcs.push_back(std::vector<Scalar>{0.0, 1.0});

            return id;
        }
        
        /**
         * Remove deleted entities from the element arrays, reindexing everything to continuous representation.
         */
        void garbage_collection()
        {
          int n_v = this->n_vertices();
          int n_h = this->n_halfedges();
          int n_f = this->n_faces();
          std::vector<int> vmap(n_v);
          std::vector<int> hmap(n_h);
          std::vector<int> fmap(n_f);
          
          int vcount = 0;
          for(int i = 0; i < n_v; i++)
          {
            if(this->out[i] >= 0)
            {
              vmap[i] = vcount;
              vcount++;
            }
            else
            {
              vmap[i] = -1;
            }
          }
          
          int hcount = 0;
          for(int i = 0; i < n_h; i++)
          {
            if(this->n[i] >= 0)
            {
              hmap[i] = hcount;
              hcount++;
            }
            else
            {
              hmap[i] = -1;
            }
          }
          
          int fcount = 0;
          for(int i = 0; i < n_f; i++)
          {
            if(this->h[i] >= 0)
            {
              fmap[i] = fcount;
              fcount++;
            }
            else
            {
              fmap[i] = -1;
            }
          }
          
          for(int i = 0; i < n_v; i++)
          {
            if(vmap[i] >= 0)
            {
              this->out[vmap[i]] = hmap[this->out[i]];
              vertex_type[vmap[i]] = vertex_type[i];
            }
          }
          for(int i = 0; i < n_h; i++)
          {
            if(hmap[i] >= 0)
            {
              this->n[hmap[i]] = hmap[this->n[i]];
              this->prev[hmap[i]] = hmap[this->prev[i]];
              this->to[hmap[i]] = vmap[this->to[i]];
              this->f[hmap[i]] = fmap[this->f[i]];
              this->opp[hmap[i]] = hmap[this->opp[i]];
              edge_type[hmap[i]] = edge_type[i];
              origin[hmap[i]] = origin[i];
              origin_of_origin[hmap[i]] = origin_of_origin[i];
              // remap seg_bcs
              seg_bcs[hmap[i]] = seg_bcs[i];
            }
          }
          for(int i = 0; i < n_f; i++)
          {
            if(fmap[i] >= 0)
              this->h[fmap[i]] = hmap[this->h[i]];
          }
          for(size_t i = 0; i < first_segment.size(); i++)
            first_segment[i] = hmap[first_segment[i]];
          this->out.resize(vcount);
          this->n.resize(hcount);
          this->prev.resize(hcount);
          this->to.resize(hcount);
          this->f.resize(hcount);
          this->opp.resize(hcount);
          origin.resize(hcount);
          origin_of_origin.resize(hcount);
          // resize seg_bcs
          seg_bcs.resize(hcount);

          this->h.resize(fcount);
          vertex_type.resize(vcount);
          edge_type.resize(hcount);

        }
        
        /**
         * Given data per vertex, uniformly interpolate it onto crossings along o(and oc)-edges
         */
        template<typename T>
        std::vector<T> interpolate_along_o(const std::vector<T> &x)
        {
          std::vector<T> z = x;
          int n_v = this->n_vertices();
          z.resize(n_v);
          
          for(int v = 0; v < n_v; v++)
          {
            if(this->vertex_type[v] == SEGMENT_VERTEX && this->out[v] >= 0) // crossing vertex, not deleted
            {
              int h = this->out[v];
              if(edge_type[this->e(h)] != ORIGINAL_EDGE) // o edge
                h = this->n[this->opp[h]];
              if(edge_type[this->e(h)] != ORIGINAL_EDGE)
                spdlog::error("not every other edge surrounding a crossing is an original edge");
              int hl = h;
              int hr = this->opp[hl];
              int nl = 1;
              int nr = 0;
              while(this->vertex_type[this->to[hl]] != ORIGINAL_VERTEX)
              {
                hl = this->n[this->opp[this->n[hl]]];
                nl++;
              }
              while(this->vertex_type[this->to[hr]] != ORIGINAL_VERTEX)
              {
                hr = this->n[this->opp[this->n[hr]]];
                nr++;
              }
              int vl = this->to[hl];
              int vr = this->to[hr];
              z[v] = z[vl] * (T(nr)/(nr+nl)) + z[vr] * (T(nl)/(nr+nl));             
            }
          }
          
          return z;
        }
        
        /**
         * Given data per triangle corner (halfedge) of the c-mesh (i.e. the Mesh, not the OverlayMesh), uniformly interpolate it onto all corners along c(and oc)-edges
         */
        template<typename T>
        std::vector<T> interpolate_along_c(const std::vector<T> &x)
        {
          std::vector<bool> is_visited(this->n.size(), false);

          std::vector<T> z(this->n.size());
          
          int n_ch = x.size();
          for(unsigned int i = 0; i < n_ch; i++) //copy from c-mesh to overlay-mesh
          {
            z[this->last_segment(i)] = x[i];
            is_visited[this->last_segment(i)] = true;
          }
          
          int n_h = this->n_halfedges();
          for(int h = 0; h < n_h; h++) // propagate around vertices from c/oc-segments onto o-segments
          {
            if(this->n[h] >= 0 && vertex_type[this->to[h]] == ORIGINAL_VERTEX && edge_type[this->e(h)] != ORIGINAL_EDGE) // c or oc edge before original vertex -> has valid value
            {
              int hc = h;
              while(true)
              {
                hc = this->opp[this->n[hc]];
                if(edge_type[this->e(hc)] != ORIGINAL_EDGE)  // c or oc edge -> end (has own value already)
                  break;
                z[hc] = z[h];
                is_visited[hc] = true;
              }
            }
          }
          
          int n_v = this->n_vertices();
          for(int v = 0; v < n_v; v++)
          {
            if(this->vertex_type[v] == SEGMENT_VERTEX && this->out[v] >= 0) // crossing vertex, not deleted
            {
              int h = this->out[v];
              if(edge_type[this->e(h)] == ORIGINAL_EDGE)
                h = this->n[this->opp[h]];
              int hl = h;
              int hr = this->opp[hl];
              int hopp = hr;
              int nl = 1;
              int nr = 0;
              while(this->vertex_type[this->to[hl]] != ORIGINAL_VERTEX)
              {
                hl = next_segment(hl);
                nl++;
              }
              while(this->vertex_type[this->to[hr]] != ORIGINAL_VERTEX)
              {
                hr = next_segment(hr);
                nr++;
              }
              int h0 = this->prev[h];
              int h1 = this->prev[this->opp[hr]];
              int h2 = this->prev[this->opp[hl]];
              int h3 = this->opp[this->n[hopp]];
              int h4 = this->opp[this->n[h3]];
              z[h0] = z[hl] * (T(nr)/(nr+nl)) + z[h1] * (T(nl)/(nr+nl));
              z[h4] = z[h0];
              z[h3] = z[h2] * (T(nr)/(nr+nl)) + z[hr] * (T(nl)/(nr+nl));
              z[hopp] = z[h3];

              is_visited[h0] = is_visited[h4] = is_visited[h3] = is_visited[hopp] = true;
            }
          }

          
          return z;
        }

        // map seg_bcs form equilateral to scaled lengths
        template<typename T>
        void bc_eq_to_scaled(const std::vector<int> &m_n,
                             const std::vector<int> &m_to, 
                             const std::vector<T> &m_l,
                             const Eigen::Matrix<T, -1, 1> &u)
        {

          for (int h = 0; h < seg_bcs.size(); h++)
          {
            if (edge_type[this->e(h)] == ORIGINAL_EDGE) continue;

            int _hij = origin[h];
            int _hjk = m_n[_hij];
            int _hki = m_n[_hjk];
            T lij = m_l[_hij];
            T ljk = m_l[_hjk];
            T lki = m_l[_hki];
            T ui = u[m_to[_hki]];
            T uj = u[m_to[_hij]];
            T Si = ljk / (lij * lki) * exp(-ui);
            T Sj = lki / (lij * ljk) * exp(-uj);

            seg_bcs[h][0] *= Scalar(Si);
            seg_bcs[h][1] *= Scalar(Sj);
            Scalar sum = seg_bcs[h][0] + seg_bcs[h][1];
            seg_bcs[h][0] /= sum;
            seg_bcs[h][1] /= sum;
          }
        }

        // map seg_bcs form original lengths to equilateral
        template<typename T>
        void bc_original_to_eq(const std::vector<int> &m_n,
                               const std::vector<int> &m_to,
                               const std::vector<T> &m_l)
        {
          for (int h = 0; h< seg_bcs.size(); h++)
          {
            if (edge_type[this->e(h)] == ORIGINAL_EDGE) continue;

            int _hij = origin[h];
            int _hjk = m_n[_hij];
            int _hki = m_n[_hjk];
            T lij = m_l[_hij];
            T ljk = m_l[_hjk];
            T lki = m_l[_hki];
            T Si = (lij * lki) / ljk;
            T Sj = (lij * ljk) / lki;

            seg_bcs[h][0] *= Scalar(Si);
            seg_bcs[h][1] *= Scalar(Sj);
            Scalar sum = seg_bcs[h][0] + seg_bcs[h][1];
            seg_bcs[h][0] /= sum;
            seg_bcs[h][1] /= sum;
          }
        }
        
        /**
         * Given data per vertex, interpolate it onto crossings along o(and oc)-edges using segments' barycentric coordinates
         */
        template<typename T> 
        std::vector<T> interpolate_along_o_bc(const std::vector<int> m_opp, const std::vector<int> m_to, const std::vector<T> &x)
        {
          std::vector<T> z = x;
          int n_v = this->n_vertices();
          z.resize(n_v);
          for (int v = 0; v < n_v; v++)
          {
            if (vertex_type[v] == SEGMENT_VERTEX && this->out[v] >= 0)
            {
              int h_out = this->out[v];
              while (edge_type[h_out] == ORIGINAL_EDGE)
              {
                h_out = this->n[this->opp[h_out]];
              }
              int h_to = this->opp[h_out];
              int _h = origin[h_out];
              int _h_opp = m_opp[_h];

              z[v] = T(seg_bcs[h_to][0]) * x[m_to[_h]] + T(seg_bcs[h_to][1]) * x[m_to[_h_opp]];
            }
          }
          return z;
        }

        /**
         * Given data per triangle corner (halfedge) of the c-mesh (i.e. the Mesh, not the OverlayMesh), interpolate it onto all corners along c(and oc)-edges using segments' barycentric coordinates
         */
        template<typename T>
        std::vector<T> interpolate_along_c_bc(const std::vector<int> m_n, const std::vector<int> m_f, const std::vector<T> &x)
        {
          std::vector<T> z(this->n.size());

          for (int i = 0; i < this->n.size(); i++)
          {
            if (edge_type[this->e(i)] == ORIGINAL_EDGE) continue;
            int h = origin[i];
            int h_prev = m_n[h];
            while (m_n[h_prev] != h)
            {
              h_prev = m_n[h_prev];
            }
            // interpolate
            z[i] = T(seg_bcs[i][0]) * x[h_prev] + T(seg_bcs[i][1]) * x[h];
          }

          // rotate to get values for original edges corners
          for (int i = 0; i < this->n.size(); i++)
          {
            if (edge_type[this->e(i)] != ORIGINAL_EDGE) continue;
            int h = this->prev[this->opp[i]];
            while (edge_type[this->e(h)] == ORIGINAL_EDGE)
            {
              h = this->prev[this->opp[h]];
            }
            z[i] = z[h];
          }
         
          return z;
        }

        /**
         * Interpolate cut_to_singulariy in the OverlayMesh
         */
        std::vector<bool> interpolate_is_cut_h(const std::vector<bool> &x)
        {
          std::vector<bool> z(this->n.size(), false);
          int n_ch = x.size();
          for (unsigned int i = 0; i < n_ch; i++)
          {
            if (!x[i]) continue;
            int h = this->first_segment[i];
            int h_last = this->last_segment(i);
            while (h != h_last)
            {
              z[h] = x[i];

              // next segment
              h = next_segment(h);
            }

            z[h_last] = x[i];
          }
          return z;
        }
        
        /**
         * Check alignment of the bcs of the vertices on the common edge of two triangles
         * @param _m the Current Mesh(pointer)
         * @param _h the Current-halfedge of the common edge to check
         */
        // template <typename Scalar>
        void check_bc_alignment(Mesh<Scalar> *_m, int _h)
        {
          if (first_segment[_h] == last_segment(_h))
            return;
          int _hopp = _m->opp[_h];

          int _f0 = _m->f[_h];
          int _f1 = _m->f[_m->opp[_h]];
          int _hjk = _m->n[_h];
          int _hki = _m->n[_hjk];
          int _hil = _m->n[_m->opp[_h]];
          int _hlj = _m->n[_hil];

          Scalar lil = _m->l[_hil];
          Scalar ljk = _m->l[_hjk];
          Scalar llj = _m->l[_hlj];
          Scalar lki = _m->l[_hki];
          Scalar exp_zij = (llj * lki) / (lil * ljk);
          Scalar z = (exp_zij - 1) / (exp_zij + 1);

          Eigen::Matrix<Scalar, 1, 2> A, B, C, D;
          A << -1, 0;
          B << 1, 0;
          C << z, sqrt(1 - z * z);
          D << -z, -sqrt(1 - z * z);
          Scalar lab = 2;
          Scalar lba = lab;
          Scalar lbc = (B - C).norm();
          Scalar lca = (C - A).norm();
          Scalar lad = (A - D).norm();
          Scalar ldb = (D - B).norm();
          Scalar lcd = (C - D).norm();
          Scalar ldc = lcd;

          // get the vertices on the _h side
          std::vector<Scalar> lambdas;
          int current_seg = first_segment[_h];
          while (vertex_type[this->to[current_seg]] != ORIGINAL_VERTEX)
          {
            std::vector<Scalar> tmp = seg_bcs[current_seg];
            tmp[0] *= Scalar(lbc / (lab * lca)); // A in ABC
            tmp[1] *= Scalar(lca / (lab * lbc)); // B in ABC
            Scalar sum = tmp[0] + tmp[1];
            tmp[0] /= sum;
            tmp[1] /= sum;
            lambdas.push_back(tmp[1]);
            current_seg = next_segment(current_seg);
          }
          // check the vertices on the _hopp side
          
          current_seg = first_segment[_hopp];
          int cnt = lambdas.size() - 1;
          while (vertex_type[this->to[current_seg]] != ORIGINAL_VERTEX)
          {
            std::vector<Scalar> tmp = seg_bcs[current_seg];
            tmp[0] *= Scalar(lad / (lba * ldb)); // B in BAD
            tmp[1] *= Scalar(ldb / (lba * lad)); // A in BAD
            Scalar sum = tmp[0] + tmp[1];
            tmp[0] /= sum;
            tmp[1] /= sum;
            if (abs(lambdas[cnt] - tmp[0]) > 1e-10)
            {
              spdlog::error("alignment problem: {}, {}", lambdas[cnt], tmp[0]);
            }
            cnt--;
            current_seg = next_segment(current_seg);
          }
        }

    private:
        int seek_segment_to(int start_halfedge, int end_halfedge, int _h0, int _h1);

        /**
         * Removes all Segments of the Chain starting with the given Segment.
         * @param start The first Segment of the Chain.
         */
        void remove_all_segments(int _h);

        /**
         * Sets the given Edge-Type to all Segments of the Chain.
         * @param start The first Segment of the Chain.
         * @param type The Edge-Type that should be set.
         */
        void set_segment_edge_type(int start, int type);
         
        /**
         * Construct the flipped Edge of the given Current-Edge. Starting the Boundary Iteration at the given
         * Start- and End-Halfedges.
         * @param _m The Current Mesh
         * @param _h The Current-Halfedge that is flipped.
         * @param start The first Halfedge of the Boundary.
         * @param end The last Halfedge of the Boundary.
         */
        // template<typename Scalar>
        void construct_flipped_current_diagonal(Mesh<Scalar>* _m, int _h, int first_opp, int start, int end, bool Ptolemy);

        /**
         * Splits the given Halfedge creating a new Halfedge.
         * @param h The Halfedge that should be split.
         * @return Returns the ID of the created 2-Valence Vertex.
         */
        int split_halfedge(int h);

        int connect_vertices(int v0_out, int v1_in, int o0, int o1, int type);

        /**
         * Removes the 2-Valence-Vertex the given Halfedge is pointing to.
         * @param v_to_keep The Halfedge pointing to the Vertex. This Halfedge is kept.
         */
        void remove_valence2_vertex(int v_to_keep);

        /**
         * Merge the two Faces incident to the given Halfedge by removing the Halfedge.
         * @param dh The Halfedge that will be removed.
         * @return Returns the ID of the merged Face.
         */
        int merge_faces(int dh);
    };

    
}


/**
 * Get face labels(which part in a double-mesh) for the overlay mesh.
 * @param m_o OverlayMesh structure
 * @return #faces array of labels (1 or 2 for double mesh, -1 otherwise)
 */
template <typename Scalar>
std::vector<int> get_overlay_face_labels(OverlayProblem::OverlayMesh<Scalar> &m_o)
{
  auto m = m_o.cmesh();
  m_o.garbage_collection();
  std::vector<int> f_type(m_o.h.size(), -1);
  for (int i = 0; i < f_type.size(); i++)
  {
    int h0 = m_o.h[i];
    int h = h0;

    do
    {
      int _h_origin = m_o.origin[h];
      if (_h_origin == -1)
        spdlog::error("-1 in origin array");
      if (m_o.edge_type[h] != ORIGINAL_EDGE)
      {
        if (m.type[_h_origin] == 1)
        {
          if (f_type[i] == 2)
            spdlog::error("f type error, face {}", i);
          f_type[i] = 1;
        }
        if (m.type[_h_origin] == 2)
        {
          if (f_type[i] == 1)
            spdlog::error("f type error, face {}", i);
          f_type[i] = 2;
        }
      }
      else
      {
        if (m.type_input[_h_origin] == 1)
        {
          if (f_type[i] == 2)
            spdlog::error("f type error, face {}", i);
          f_type[i] = 1;
        }
        if (m.type_input[_h_origin] == 2)
        {
          if (f_type[i] == 1)
            spdlog::error("f type error, face {}", i);
          f_type[i] = 2;
        }
      }
      h = m_o.n[h];
    } while (h != h0);
  }
  return f_type;
};

#endif