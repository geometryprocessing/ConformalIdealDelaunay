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

#include "OverlayMesh.hh"

template<typename Scalar> 
void compute_S0_S1(Scalar ljk, Scalar lki, Scalar lil, Scalar llj, Scalar &S0, Scalar &S1)
{
    Scalar exp_zij = (llj * lki) / (lil * ljk);
    Scalar z = (exp_zij - 1) / (exp_zij + 1);

    Eigen::Matrix<Scalar, 1, 2> A, B, C, D;
    A << -1, 0;
    B << 1, 0;
    C << z, sqrt(1 - z * z);
    D << -z, -sqrt(1 - z * z);

    Scalar lad = (A - D).norm();
    Scalar ldb = (D - B).norm();

    S0 = (lad * lad * lad * lad) / 16.0;
    S1 = 16.0 / (ldb * ldb * ldb * ldb);
};

template<typename Scalar>
void OverlayProblem::OverlayMesh<Scalar>::update_bc_bd(int _hjk, int _hki, int _hil, int _hlj, Scalar S0, Scalar S1)
{
    int current = first_segment[_hil];
    while (vertex_type[this->to[current]] != ORIGINAL_VERTEX)
    {
        // map from ADC to BAD
        seg_bcs[current][0] *= Scalar(S1);
        Scalar sum = seg_bcs[current][0] + seg_bcs[current][1];
        seg_bcs[current][0] /= sum;
        seg_bcs[current][1] /= sum;
        current = next_segment(current);
    }
    current = first_segment[_hlj];
    while (vertex_type[this->to[current]] != ORIGINAL_VERTEX)
    {
        // map from BCD to BAD
        seg_bcs[current][0] *= Scalar(S0);
        Scalar sum = seg_bcs[current][0] + seg_bcs[current][1];
        seg_bcs[current][0] /= sum;
        seg_bcs[current][1] /= sum;
        current = next_segment(current);
    }
    current = first_segment[_hjk];
    while (vertex_type[this->to[current]] != ORIGINAL_VERTEX)
    {
        // map from BCD to ABC
        seg_bcs[current][0] *= Scalar(S1);
        Scalar sum = seg_bcs[current][0] + seg_bcs[current][1];
        seg_bcs[current][0] /= sum;
        seg_bcs[current][1] /= sum;
        current = next_segment(current);
    }
    current = first_segment[_hki];
    while (vertex_type[this->to[current]] != ORIGINAL_VERTEX)
    {
        // map from ADC to ABC
        seg_bcs[current][0] *= Scalar(S0);
        Scalar sum = seg_bcs[current][0] + seg_bcs[current][1];
        seg_bcs[current][0] /= sum;
        seg_bcs[current][1] /= sum;
        current = next_segment(current);
    }
};

template<typename Scalar>
bool OverlayProblem::OverlayMesh<Scalar>::o_flip_ccw(Mesh<Scalar>* _m, int _h, bool Ptolemy) {
    //Halfedges that should be flipped.
    int _h0 = _h;
    int _h1 = _m->opp[_h0];

    //The Vertices that will be connected by the Flip. V0 == Origin of the new _h. V1 == Destination.
    int v0 = _m->to[_h1];
    int v1 = _m->to[_h0];

    if (vertex_type[v0] == SEGMENT_VERTEX || vertex_type[v1] == SEGMENT_VERTEX) {
        printf("Can only Flip between Original Vertices!\n");
        return false;
    }

    //Gets the Type of the Edge (O == 0, C == 1, OC == 2)
    int type = edge_type[this->e(first_segment[_h0])];
    if (type == ORIGINAL_EDGE) {
        printf("Can not flip O-Edge!\n");
        return false;
    }

    //The Start and End of the Boundary. This Boundary will be iterated and intersecting Edges will be determined.
    int start_seg = first_segment[_m->n[_h1]];
    int end_seg = last_segment(_m->n[_m->n[_h1]]);

    if (type == CURRENT_EDGE) {
        //Check weather there is a diagonal O-Edge...
        int diag_seg = this->seek_segment_to(end_seg, start_seg, _h0, _h1);

        if (diag_seg != -1) {
            // Case: C -> OC
            //First, remove the Current-Edge.
            this->remove_all_segments(_h0);
            //Refreshing Diagonal...
            diag_seg = this->seek_segment_to(end_seg, start_seg, _h0, _h1);
            if (diag_seg == -1) return false;
            this->set_segment_edge_type(diag_seg, ORIGINAL_AND_CURRENT_EDGE); 

            //Update the First-Structure (We know that no Edges will intersect this OC-Edge)
            first_segment[_h0] = this->opp[diag_seg];
            first_segment[_m->opp[_h0]] = diag_seg;
            
            origin_of_origin[diag_seg] = origin[diag_seg];
            origin_of_origin[this->opp[diag_seg]] = origin[this->opp[diag_seg]];
            origin[diag_seg] = _m->opp[_h0];
            origin[this->opp[diag_seg]] = _h0;

            seg_bcs[diag_seg] = std::vector<Scalar>{0.0, 1.0};
            seg_bcs[this->opp[diag_seg]] = std::vector<Scalar>{0.0, 1.0};

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

            Scalar S0, S1;

            if (Ptolemy)
            {
                compute_S0_S1(ljk, lki, lil, llj, S0, S1);
                update_bc_bd(_hjk, _hki, _hil, _hlj, S0, S1);
            }
            

        } 
        else {
            //Case: C -> C

            //Remove Diagonal
            this->remove_all_segments(_h0);

            //Insert the flipped Segments, the Direction of the Segments is equal to the original Halfedge.
            int first_opp_seg = first_segment[_m->n[_h0]];
            this->construct_flipped_current_diagonal(_m, _h0, first_opp_seg, start_seg, end_seg, Ptolemy);
        }
    } else if (type == ORIGINAL_AND_CURRENT_EDGE) {
        //Case: OC -> C
        // the only difference between oc->c and c->c is we need to initiate the oc to o before construct flipped diagonal, in c->c case we need to remove the previous c path
        this->set_segment_edge_type(first_segment[_h0], ORIGINAL_EDGE);
        
        // make sure they have the correct origin
        origin[first_segment[_h0]] = origin_of_origin[first_segment[_h0]];
        origin[this->opp[first_segment[_h0]]] = origin_of_origin[this->opp[first_segment[_h0]]];

        int first_opp_seg = first_segment[_m->n[_h0]];
        this->construct_flipped_current_diagonal(_m, _h0, first_opp_seg, start_seg, end_seg, Ptolemy);
    }

    return true;
}

template<typename Scalar>
int OverlayProblem::OverlayMesh<Scalar>::seek_segment_to(int start_halfedge, int end_halfedge, int _h0, int _h1) {

    //Starting with the first next Halfedge of the Start-Halfedge
    int current = this->n[start_halfedge];
    while (true) {
        //Iterate trough every O-Edge that could be possible a connecting Diagonal...
        int current_type = edge_type[this->e(current)];
        if (current_type != ORIGINAL_EDGE) break;

        //For every O-Edge...
        int seg = current;
        while (true) {
            int v = this->to[seg];
            int vt = vertex_type[v];

            //Forward until we reached an Original Vertex...
            if (vt == SEGMENT_VERTEX) {
                if (this->valence(seg) <= 2) {
                    seg = this->n[seg];
                } else {
                    int h = this->n[seg];
                    if (origin[h] != _h0 && origin[h] != _h1) {
                        break;
                    } else {
                        seg = this->n[this->opp[this->n[seg]]];
                    }
                }
            } else {
                //We are on a Original Vertex! Check if the End-Halfedge is one of the outgoing...
                int start_out = this->n[seg];
                int current_out = start_out;

                int iterations = 0;
                while (true) {
                    if (current_out == end_halfedge) {
                        //Success! We found a Diagonal.
                        return current;
                    }

                    iterations++;
                    if (iterations >= SAFETY_LIMIT) {
                        spdlog::error("infinite loop (A)");
                        break;
                    }

                    current_out = this->n[this->opp[current_out]];
                    if (current_out == start_out) break;
                }
                break;
            }
        }

        current = this->n[this->opp[current]];
    }

    return -1;
}

template<typename Scalar>
void OverlayProblem::OverlayMesh<Scalar>::remove_all_segments(int _h) {
    int start = first_segment[_h];

    int a = 0;
    int current = start;
    if (vertex_type[this->to[current]] == ORIGINAL_VERTEX) {
        //There is no Vertex in the Middle. Just merge this Edge.
        this->merge_faces(current);
        return;
    }

    while (true) {
        //The Vertex the current Segment is pointing to.
        int vt = this->to[current];

        //Compute the next Vertex.
        int next;
        if (this->valence(current) <= 2) {
            next = this->n[current];
        } else {
            next = this->n[this->opp[this->n[current]]];
        }

        this->merge_faces(current);

        if (vertex_type[this->to[next]] == ORIGINAL_VERTEX) {
            this->merge_faces(next);
            break;
        } else {
            current = next;
        }
    }
}

template<typename Scalar>
void OverlayProblem::OverlayMesh<Scalar>::set_segment_edge_type(int start, int type) {

    int current = start;
    while (true) {
        //Set the Type
        edge_type[this->h0(current)] = edge_type[this->h1(current)] = type;

        //Break if we reached the End of the Vertex
        if (vertex_type[this->to[current]] == ORIGINAL_VERTEX) break;

        //Continue to the next Part of the Segment
        if (this->valence(current) <= 2) {

            current = this->n[current];
        } else {
            current = this->n[this->opp[this->n[current]]];
        }
    }
}

template<typename Scalar>
void OverlayProblem::OverlayMesh<Scalar>::construct_flipped_current_diagonal(Mesh<Scalar>* _m, int _h, int first_opp, int start, int end, bool Ptolemy) {
    //////////
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

    if (Ptolemy)
    {
        Scalar S0, S1;
        compute_S0_S1(ljk, lki, lil, llj, S0, S1);
        update_bc_bd(_hjk, _hki, _hil, _hlj, S0, S1);
    }
    
    ///////////////// seperate func
    int vs = this->to[this->opp[start]];
    int ve = this->to[end];

    int o0 = _h;
    int o1 = _m->opp[_h];

    std::vector<int> opp_boundary = {};
    int opp_center = -1;
    int current = first_opp;
    while (true) {
        int vt = this->to[current];
        if (vertex_type[vt] == ORIGINAL_VERTEX) {
            if (opp_center != -1) break;
            opp_center = vt;
        }
        current = this->n[current];
        while (edge_type[this->e(current)] == ORIGINAL_EDGE) {
            current = this->n[this->opp[current]];
        }
        opp_boundary.push_back(current);
    }

    //Last Vertex used in the Chain of Segments. Initially the first Vertex.
    int last_chain_out = start;
    bool first = true;

    current = start;
    do {
        //The current O-Edge of the Vertex "current" is pointing to. Initially the O-Edge next to the current Segment.
        int current_o_edge = this->n[current];
        while (edge_type[this->e(current_o_edge)] == ORIGINAL_EDGE) {
            int h0 = current_o_edge;
            int h1 = this->opp[h0];

            bool i1 = std::find(opp_boundary.begin(), opp_boundary.end(), this->n[h0]) != opp_boundary.end();
            bool i2; // case to[h0] is opp_center, n[to[ho]] may not on the opp_boundary since higher valence
            if (vertex_type[this->to[h0]] == ORIGINAL_VERTEX && this->to[h0] == opp_center) {
                int start = this->n[h0];

                int out = start;
                while (true) {
                    if (std::find(opp_boundary.begin(), opp_boundary.end(), out) != opp_boundary.end()) {
                        i2 = true;
                        break;
                    }
                    if (edge_type[this->e(out)] != ORIGINAL_EDGE) {
                        i2 = false;
                        break;
                    }

                    out = this->n[this->opp[out]];
                    if (out == start) {
                        i2 = false;
                        break;
                    }
                }
            } else {
                i2 = false;
            }

            if (i1 || i2) {
                //Create a new Vertex inside the O-Edge.
                split_halfedge(h0);
                //Connect the last Chain-Vertex with this new created Vertex.
                int to_next = connect_vertices(last_chain_out, h0, o0, o1, CURRENT_EDGE);
                if (first) {
                    //This is the first inserted Segment. Updating the First Array.
                    first_segment[_h] = to_next;
                    first = false;
                }
                last_chain_out = this->opp[h0];
            } else {
                // to make sure v0_out and v1_in are in same face
                if (first && this->to[h0] == vs) {
                    last_chain_out = this->opp[h0];
                }
            }

            //Gets the next O-Edge of the current Vertex
            current_o_edge = this->n[h1];
        }
        current = current_o_edge;
    } while (current != end);

    int to_end = end; // v1_in, last_chain_out = v0_out
    while (this->f[to_end] != this->f[last_chain_out]) { // the same check in connect_vertices
        to_end = this->opp[this->n[to_end]];
    }

    //This is the last inserted Segment. Updating the First Array.
    int to_last = connect_vertices(last_chain_out, to_end, o0, o1, CURRENT_EDGE);
    if (first) { // there are no intersects
        first_segment[_h] = to_last;
    }
    first_segment[_m->opp[_h]] = this->opp[to_last];
    

    update_bc_intersection(_m, _h, Ptolemy);
}

template<typename Scalar>
void OverlayProblem::OverlayMesh<Scalar>::update_bc_intersection(Mesh<Scalar>* _m, int _h, bool Ptolemy)
{
    
    // step0: prepare the two-triangles chart
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
    Scalar lij = _m->l[_h];

    Scalar exp_zij = (llj * lki) / (lil * ljk);
    Scalar z = (exp_zij - 1) / (exp_zij + 1);

    Eigen::Matrix<Scalar, 1, 2> A, B, C, D;
    
    if (Ptolemy)
    {
        A << -1, 0;
        B << 1, 0;
        C << z, sqrt(1 - z * z);
        D << -z, -sqrt(1 - z * z);
    }
    else
    {
        A << 0, 0;
        B << lij, 0;
        Scalar cos_bac = (lij * lij + lki * lki - ljk * ljk) / (2 * lij * lki);
        Scalar sin_bac = sqrt(1 - cos_bac * cos_bac);
        C << lki * cos_bac, lki * sin_bac;
        Scalar cos_bad = (lij * lij + lil * lil - llj * llj) / (2 * lij * lil);
        Scalar sin_bad = sqrt(1 - cos_bad * cos_bad);
        D << lil * cos_bad, -lil * sin_bad;
    }
 
    
    Scalar lab = (A - B).norm();
    Scalar lba = lab;
    Scalar lbc = (B - C).norm();
    Scalar lca = (C - A).norm();
    Scalar lad = (A - D).norm();
    Scalar ldb = (D - B).norm();
    Scalar lcd = (C - D).norm();
    Scalar ldc = lcd;

    // GOAL: go through the new flipped diagonal(from first_segments[_h]), compute the intersection bc for each segment halfedge
    // step1: get the two endpoints v0,v1 on the bnd and bnd_opp
    // step2: remap the bcs of v0,v1 based on the two-triangles chart(from equilateral)
    // step3: use the new bcs(of v0,v1) to compute the intersection
    // step4：map intersection bc to equilateral(from two-triangle chart) and assign it to the halfedge
    int h = first_segment[_h];
    int cnt = 0;
    std::vector<Scalar> lambdas;
    while (vertex_type[this->to[h]] != ORIGINAL_VERTEX)
    {
        /////////////////////
        // step 1
        /////////////////////
        int h_top = this->n[h];
        while (edge_type[this->e(h_top)] == ORIGINAL_EDGE || (origin[h_top] != _hjk && origin[h_top] != _hki))
        {
            h_top = this->prev[this->opp[h_top]];
            if (h_top == this->n[h])
            {
                spdlog::error("error detected (top)");
                break;
            }
        }
        int h_bottom = this->opp[this->prev[this->opp[h]]];
        while (edge_type[this->e(h_bottom)] == ORIGINAL_EDGE || (origin[h_bottom] != _hil && origin[h_bottom] != _hlj))
        {
            h_bottom = this->prev[this->opp[h_bottom]];
            if(h_bottom == this->opp[this->prev[this->opp[h]]])
            {
                spdlog::error("error detected (bottom)");
                break;
            }
        }

        /////////////////////
        // step 2
        /////////////////////

        Eigen::Matrix<Scalar, 1, 2> V_top, V_bottom;

        if (origin[h_top] == _hjk)
        {
            std::vector<Scalar> tmp = seg_bcs[h_top];
            if (Ptolemy)
            {
                tmp[0] *= lca / (lab * lbc); // B in ABC
                tmp[1] *= lab / (lbc * lca); // C in ABC
                Scalar sum = tmp[0] + tmp[1];
                tmp[0] /= sum;
                tmp[1] /= sum;
            }
            V_top = tmp[0] * B + tmp[1] * C;
        }
        else if (origin[h_top] == _hki)
        {
            std::vector<Scalar> tmp = seg_bcs[h_top];
            if (Ptolemy)
            { 
                tmp[0] *= lab / (lbc * lca); // C in ABC
                tmp[1] *= lbc / (lab * lca); // A in ABC
                Scalar sum = tmp[0] + tmp[1];
                tmp[0] /= sum;
                tmp[1] /= sum;
            }
            V_top = tmp[0] * C + tmp[1] * A;
        }

        if (origin[h_bottom] == _hil)
        {
            std::vector<Scalar> tmp = seg_bcs[h_bottom];
            if (Ptolemy)
            {
                tmp[0] *= ldb / (lad * lba); // A in BAD
                tmp[1] *= lba / (ldb * lad); // D in BAD
                Scalar sum = tmp[0] + tmp[1];
                tmp[0] /= sum;
                tmp[1] /= sum;
            }
            V_bottom = tmp[0] * A + tmp[1] * D;
        }
        else if (origin[h_bottom] == _hlj)
        {
            std::vector<Scalar> tmp = seg_bcs[h_bottom];
            if (Ptolemy)
            {
                tmp[0] *= lba / (ldb * lad); // D in BAD
                tmp[1] *= lad / (lba * ldb); // B in BAD
                Scalar sum = tmp[0] + tmp[1];
                tmp[0] /= sum;
                tmp[1] /= sum;
            }
            V_bottom = tmp[0] * D + tmp[1] * B;
        }
        
        /////////////////////
        // step 3
        /////////////////////
        Scalar x_diff = Scalar(V_top(0) - V_bottom(0));
        Scalar y_diff = Scalar(V_top(1) - V_bottom(1));
        Scalar x_intersect;
        if (y_diff == 0)
        {
            x_intersect = V_bottom(0);
        } 
        else
        {
            x_intersect= Scalar(V_bottom(0)) - Scalar(V_bottom(1)) / y_diff * x_diff;
        }
        Scalar lambda = (x_intersect - A(0)) / lab;

        lambdas.push_back(Scalar(lambda));
        h = next_segment(h);
        cnt++;
    }
    
    for (int i = 0 ; i < lambdas.size() - 1 ;i++)
    {
        if (lambdas[i] >= lambdas[i + 1])
        {   
            spdlog::debug("lambda error: {}, {}", lambdas[i], lambdas[i+1]);
            bypass_overlay = true;
        }
    }
    /////////////////////
    // step 4
    /////////////////////
    h = first_segment[_h];
    for (int i = 0; i < cnt; i++)
    {
        std::vector<Scalar> tmp{1 - lambdas[i], lambdas[i]};
        if (Ptolemy)
        {
            tmp[0] *= Scalar(lca * lab / lbc); // A in ABC
            tmp[1] *= Scalar(lab * lbc / lca); // B in ABC
            Scalar sum = tmp[0] + tmp[1];
            tmp[0] /= sum;
            tmp[1] /= sum;
        }
        seg_bcs[h] = tmp;
        h = next_segment(h);
    }
    seg_bcs[h] = std::vector<Scalar>{0.0, 1.0};

    h = first_segment[_m->opp[_h]];
    for (int i = 0; i < cnt; i++)
    {
        std::vector<Scalar> tmp{lambdas[cnt - 1 - i], 1 - lambdas[cnt - 1 - i]};
        if (Ptolemy)
        {
            tmp[0] *= Scalar(lba * ldb / lad); // B in BAD
            tmp[1] *= Scalar(lba * lad / ldb); // A in BAD
            Scalar sum = tmp[0] + tmp[1];
            tmp[0] /= sum;
            tmp[1] /= sum;
        }
        seg_bcs[h] = tmp;
        h = next_segment(h);
    }
    seg_bcs[h] = std::vector<Scalar>{0.0, 1.0};
}

template<typename Scalar>
int OverlayProblem::OverlayMesh<Scalar>::split_halfedge(int h) {
    //Create a mew Vertex
    int v = this->create_vertex(); // return id (add a -1 in out array)

    //The Edge-Type does not change. Same with the Origin of the Edge.
    int h0 = h;
    int h1 = this->opp[h];

    int o0 = origin[h0];
    int o1 = origin[h1];

    int n0 = this->create_halfedge(edge_type[this->e(h)], o0, o1);
    int n1 = n0 + 1;

    int k0 = this->n[h0];
    int k1 = prev[h1];

    /*
     *          h0                             k0
     * xxxxxxxx --------------------> xxxxxxxx -------------------->
     * xxxxxxxx                       xxxxxxxx
     * xxxxxxxx                       xxxxxxxx
     * xxxxxxxx <-------------------- xxxxxxxx <--------------------
     *          h1                             k1
     *
     *          h0                             n0                             k0
     * xxxxxxxx --------------------> xxxxxxxx --------------------> xxxxxxxx -------------------->
     * xxxxxxxx                       xxxxxxxx                       xxxxxxxx
     * xxxxxxxx                       xxxxxxxx                       xxxxxxxx
     * xxxxxxxx <-------------------- xxxxxxxx <-------------------- xxxxxxxx <--------------------
     *          h1                             n1                             k1
     */

    if (first_segment[o1] == h1) {
        first_segment[o1] = n1;
    }

    //Update all Pointers. First Array does not Change.
    this->n[h0] = n0;

    this->n[n0] = k0;
    this->n[k1] = n1;
    this->n[n1] = h1;

    prev[n0] = h0;
    prev[k0] = n0;
    prev[n1] = k1;
    prev[h1] = n1;

    this->to[n0] = this->to[h0];
    this->to[n1] = v;
    this->to[h0] = v;

    this->f[n0] = this->f[h0];
    this->f[n1] = this->f[h1];

    this->out[v] = n0;
    this->out[this->to[n0]] = n1;

    return v;
}

template<typename Scalar>
int OverlayProblem::OverlayMesh<Scalar>::connect_vertices(int v0_out, int v1_in, int o0, int o1, int type) {
    int v0 = this->to[this->opp[v0_out]];
    int v1 = this->to[v1_in];
    int face = this->f[v0_out];

    if (this->f[v0_out] != this->f[v1_in]) {
        return face;
    }

    int h0 = prev[v0_out];
    int n0 = v0_out;
    int h1 = v1_in;
    int n1 = this->n[v1_in];

    if (n1 == h0) {
        return this->opp[h0];
    }

    if (n0 == h1) {
        return n0;
    }

    //Create new Halfedge. All Arrays will be modified/extended.
    int x0 = this->create_halfedge(type, o0, o1);
    int x1 = x0 + 1;

    //Create new Face. All Arrays will be modified/extended.
    int new_face = this->create_face();

    /*       h0 |                       Λ n1
     *          |       New Face        |
     *          |                       |
     *          |                       |
     *          V         x0            |
     * v0 xxxxxxxx --------------------> xxxxxxxx v1
     *    xxxxxxxx                       xxxxxxxx
     *    xxxxxxxx                       xxxxxxxx
     *    xxxxxxxx <-------------------- xxxxxxxx
     *          |         x1            Λ
     *       n0 |                       | h1
     *          |                       |
     *          |                       |
     *          V                       |
     */
    //Test if the Connecting Edge is valid
    if (h0 == -1 || n0 == -1 || h1 == -1 || n1 == -1) {
        printf("Could not find all Halfedges to merge Face! Is the Face corrupted?!\n");
        return -1;
    }

    //Update Links, Out-Array does not change
    this->n[h0] = x0;
    this->n[x0] = n1;
    this->n[h1] = x1;
    this->n[x1] = n0;

    this->to[x1] = v0;
    this->to[x0] = v1;

    prev[n1] = x0;
    prev[x0] = h0;
    prev[n0] = x1;
    prev[x1] = h1;

    this->h[new_face] = x0;
    this->h[face] = x1;

    this->f[x0] = new_face;
    this->f[x1] = face;

    //Propagate the new Face
    int current = x0;
    do {
        this->f[current] = new_face;
        current = this->n[current];
    } while (current != x0);

    return x0;
}

template<typename Scalar>
void OverlayProblem::OverlayMesh<Scalar>::remove_valence2_vertex(int v_to_keep) {
    if (vertex_type[this->to[v_to_keep]] == ORIGINAL_VERTEX) {
        return;
    }

    if (this->valence(v_to_keep) != 2) {
        printf("Can not remove Vertex! It does not have Valence 2!\n");
        return;
    }

    int h0 = v_to_keep;
    int h1 = this->opp[h0];

    int d0 = this->n[h0];
    int d1 = this->opp[d0];

    int n0 = this->n[d0];
    int n1 = prev[d1];

    int v0 = this->to[h1];
    int vd = this->to[h0];
    int v1 = this->to[d0];


    /*  v0                              vd                            v1
     *          h0 / v_keep                    d0 (Delete)                    n0
     * xxxxxxxx --------------------> xxxxxxxx --------------------> xxxxxxxx ----------->
     * xxxxxxxx                       xxxxxxxx                       xxxxxxxx
     * xxxxxxxx                       xxxxxxxx                       xxxxxxxx
     * xxxxxxxx <-------------------- xxxxxxxx <-------------------- xxxxxxxx <-----------
     *          h1                             d1 (Delete)                    n1
     *
     * v0                               v1
     *          h0                             n0
     * xxxxxxxx --------------------> xxxxxxxx --------------------->
     * xxxxxxxx                       xxxxxxxx
     * xxxxxxxx                       xxxxxxxx
     * xxxxxxxx <-------------------- xxxxxxxx <---------------------
     *          h1                             n1
     */

    //Update Pointer, All Arrays are affected.
    if (first_segment[origin[d1]] == d1) {
        first_segment[origin[d1]] = h1;
    }

    if (edge_type[this->e(v_to_keep)] != ORIGINAL_EDGE)
    {
        spdlog::info("remove a valence 2 vertex on a current edge");
    }
    
    this->n[h0] = n0;
    this->n[n1] = h1;
    this->n[d0] = -1;
    this->n[d1] = -1;

    prev[n0] = h0;
    prev[h1] = n1;
    prev[d0] = -1;
    prev[d1] = -1;

    this->to[h0] = v1;
    this->to[d0] = -1;
    this->to[d1] = -1;

    this->f[d0] = -1;
    this->f[d1] = -1;

    this->h[this->f[h0]] = h0;
    this->h[this->f[h1]] = h1;

    this->out[v0] = h0;
    this->out[vd] = -1;
    this->out[v1] = h1;

    origin[d0] = -1;
    origin[d1] = -1;

    vertex_type[vd] = -1;
    edge_type[d0] = -1;
    edge_type[d1] = -1;

    // opp[d0] = -1;
    // opp[d1] = -1;
}

template<typename Scalar>
int OverlayProblem::OverlayMesh<Scalar>::merge_faces(int dh) {

    int h0 = dh;
    int h1 = this->opp[dh];

    int f0 = this->f[h0];
    int f1 = this->f[h1];

    int p0 = this->prev[h0];
    int p1 = this->prev[h1];

    int n0 = this->n[h0];
    int n1 = this->n[h1];

    //Update Pointer
    this->out[this->to[h0]] = n0; // just in case it was h1
    this->out[this->to[h1]] = n1;

    this->n[p0] = n1;
    this->n[p1] = n0;
    this->n[h0] = -1;
    this->n[h1] = -1;

    prev[n1] = p0;
    prev[n0] = p1;
    prev[h0] = -1;
    prev[h1] = -1;

    this->to[h0] = -1;
    this->to[h1] = -1;

    if (first_segment[origin[h0]] == h0) {
        first_segment[origin[h0]] = -1;
    }
    if (first_segment[origin[h1]] == h1) {
        first_segment[origin[h1]] = -1;
    }
    origin[h0] = -1;
    origin[h1] = -1;

    edge_type[h0] = -1;
    edge_type[h1] = -1;

    this->h[f0] = n1;
    this->h[f1] = -1;

    this->f[h0] = -1;
    this->f[h1] = -1;
    
    this->opp[h0] = -1;
    this->opp[h1] = -1;
    /*       p0 |                       Λ n0
     *          |                       |
     *          |                       |
     *          |                       |
     *          V         h0            |
     *    xxxxxxxx --------------------> xxxxxxxx v1
     *    xxxxxxxx                       xxxxxxxx
     *    xxxxxxxx                       xxxxxxxx
     *    xxxxxxxx <-------------------- xxxxxxxx
     *          |         h1            Λ
     *       n1 |                       | p1
     *          |                       |
     *          |                       |
     *          V                       |
     */
    //Update Face
    int start = n1;
    int current = start;
    do {
        this->f[current] = f0;
        current = this->n[current];
    } while (current != start);

    if (valence(p0) == 2) {
        this->remove_valence2_vertex(p0);
    }

    if (valence(p1) == 2) {
        this->remove_valence2_vertex(p1);
    }

    return f0;
}

template<typename Scalar>
bool OverlayProblem::OverlayMesh<Scalar>::check(Mesh<Scalar>* _m) {

    bool good = true;

    //Get the Size of all Arrays
    int nc = this->n.size();
    int n_prev = prev.size();
    int nto = this->to.size();
    int nf = this->f.size();
    int no = this->origin.size();
    int ne = edge_type.size()/2;

    //For every Halfedge in the Overlay-Mesh...
    for (int h = 0; h < this->n_halfedges(); h++) {
        //If One is -1 (Deleted)
        if (this->n[h] == -1 || this->prev[h] == -1 || this->to[h] == -1 || this->f[h] == -1) {
            //All have to be -1
            if (!(this->n[h] == -1 && this->prev[h] == -1 && this->to[h] == -1 && this->f[h] == -1)) {
                //Print the faulty Halfedge
                printf("\tHalfedge is corrupted: N: %i, Prev: %i, To: %i, f: %i\n\n", this->n[h], this->prev[h], this->to[h], this->f[h]);
                good = false;
            }
        }
    }

    //Check the First/Last-Segment Structure
    for (int _h = 0; _h < _m->n_halfedges(); _h++) {
        int _o = _m->opp[_h];

        int fs = first_segment[_h];
        if (fs == -1) {
            printf("\tFirst Segment of Original Halfedge %i is -1?!\n", _h);
            good = false;
            break;
        }
        int ls = last_segment(_h);

        int ofs = first_segment[_o];
        if (ofs == -1) {
            printf("\tFirst Segment of Original Halfedge %i is -1?!\n", _o);
            good = false;
            break;
        }
        int ols = last_segment(_o);

        if (this->opp[fs] != ols || fs != this->opp[ols]) {
            printf("\tOpponent Structure of Segment %i is corrupted!\n", _h);
            printf("\t\tfs = %i, ls = %i, ofs = %i, ols = %i\n", fs, ls, ofs, ols);
            good = false;
        }
    }

    //Check the Segment Structures. Iterate all original Vertices...
    for (int _v = 0; _v < _m->n_vertices(); _v++) {
        int iterations = 0;
        int start_out = this->out[_v];
        int current_out = start_out;

        //Iterate all outgoing Edges...
        while (true) {
            int segment_type = edge_type[this->e(current_out)];
            int current_seg = current_out;

            //Iterate the Segment...
            while (true) {
                if (edge_type[this->e(current_seg)] != segment_type) {
                    printf("Segment %i differs in Type from the other Segments!\n", current_seg);
                    good = false;
                }

                int vt = this->to[current_seg];
                if (vertex_type[vt] == ORIGINAL_VERTEX) break;

                if (this->valence(current_seg) <= 2) {
                    current_seg = this->n[current_seg];
                } else {
                    current_seg = this->n[this->opp[this->n[current_seg]]];
                }
            }

            current_out = this->n[this->opp[current_out]];
            if (current_out == start_out ) break;
            if (iterations++ > SAFETY_LIMIT)
            {
              spdlog::error("ERROR: infinite loop (B)");
              break;
            }
        }
    }

    for (int i = 0; i < this->n_faces(); i++) {
        int edges = this->face_edge_count(i);

        if (edges >= 0 && edges <= 2) {
            printf("Face %i has only %i Edges?\n", i, edges);

            good = false;
            break;
        }
    }

    printf("Counts, N: %i, Prev: %i, To: %i, F: %i, O: %i, NE: %i\n", nc, n_prev, nto, nf, no, ne);

    return good;
}

template bool OverlayProblem::OverlayMesh<double>::o_flip_ccw(Mesh<double>*, int, bool);
template bool OverlayProblem::OverlayMesh<double>::check(Mesh<double>*);
template void OverlayProblem::OverlayMesh<double>::construct_flipped_current_diagonal(Mesh<double>*, int, int, int, int, bool);

#ifdef WITH_MPFR
template bool OverlayProblem::OverlayMesh<mpfr::mpreal>::o_flip_ccw(Mesh<mpfr::mpreal>*, int, bool);
template bool OverlayProblem::OverlayMesh<mpfr::mpreal>::check(Mesh<mpfr::mpreal>*);
template void OverlayProblem::OverlayMesh<mpfr::mpreal>::construct_flipped_current_diagonal(Mesh<mpfr::mpreal>*, int, int, int, int, bool);
#endif
