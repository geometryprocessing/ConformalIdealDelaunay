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
#ifdef PYBIND
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#endif
#ifdef WITH_MPFR
#include <unsupported/Eigen/MPRealSupport>
#endif
#include "ConformalIdealDelaunayMapping.hh"
#include "Halfedge.hh"
#include "ConformalInterface.hh"
#include "Layout.hh"
#include <igl/writeOBJ.h>
#ifdef RENDER_TEXTURE
#include "Sampling.hh"
#endif 

#ifdef PYBIND
// wrap as Python module
PYBIND11_MODULE(conformal_py, m)
{
  m.doc() = "pybind for conformal mapping module";
#ifdef WITH_MPFR
  pybind11::class_<mpfr::mpreal>(m, "mpreal")
    .def(pybind11::init<std::string>())
    .def("__repr__",[](const mpfr::mpreal &a) {return a.toString();});
  pybind11::implicitly_convertible<std::string, mpfr::mpreal>();
  m.def("set_mpf_prec", &mpfr::mpreal::set_default_prec, "set global mpreal prec");
#endif
  pybind11::class_<Connectivity>(m, "Connectivity_cpp")
    .def(pybind11::init<>())
    .def_readwrite("n", &Connectivity::n)
    .def_readwrite("prev", &Connectivity::prev)
    .def_readwrite("to", &Connectivity::to)
    .def_readwrite("f", &Connectivity::f)
    .def_readwrite("h", &Connectivity::h)
    .def_readwrite("out", &Connectivity::out)
    .def_readwrite("opp", &Connectivity::opp);
  pybind11::class_<AlgorithmParameters, std::shared_ptr<AlgorithmParameters>>(m, "AlgorithmParameters")
    .def(pybind11::init<>())
    .def_readwrite("MPFR_PREC", &AlgorithmParameters::MPFR_PREC)
    .def_readwrite("initial_ptolemy", &AlgorithmParameters::initial_ptolemy)
    .def_readwrite("error_eps", &AlgorithmParameters::error_eps)
    .def_readwrite("min_lambda", &AlgorithmParameters::min_lambda)
    .def_readwrite("newton_decr_thres", &AlgorithmParameters::newton_decr_thres)
    .def_readwrite("max_itr", &AlgorithmParameters::max_itr)
    .def_readwrite("bypass_overlay", &AlgorithmParameters::bypass_overlay);
  pybind11::class_<StatsParameters, std::shared_ptr<StatsParameters>>(m, "StatsParameters")
    .def(pybind11::init<>())
    .def_readwrite("flip_count", &StatsParameters::flip_count)
    .def_readwrite("name", &StatsParameters::name)
    .def_readwrite("output_dir", &StatsParameters::output_dir)
    .def_readwrite("error_log", &StatsParameters::error_log)
    .def_readwrite("print_summary", &StatsParameters::print_summary)
    .def_readwrite("log_level", &StatsParameters::log_level);
  pybind11::class_<LineSearchParameters, std::shared_ptr<LineSearchParameters>>(m, "LineSearchParameters")
    .def(pybind11::init<>())
    .def_readwrite("energy_cond", &LineSearchParameters::energy_cond)
    .def_readwrite("energy_samples", &LineSearchParameters::energy_samples)
    .def_readwrite("do_reduction", &LineSearchParameters::do_reduction)
    .def_readwrite("descent_dir_max_variation", &LineSearchParameters::descent_dir_max_variation)
    .def_readwrite("do_grad_norm_decrease", &LineSearchParameters::do_grad_norm_decrease)
    .def_readwrite("bound_norm_thres", &LineSearchParameters::bound_norm_thres)
    .def_readwrite("lambda0", &LineSearchParameters::lambda0)
    .def_readwrite("reset_lambda", &LineSearchParameters::reset_lambda);
  pybind11::class_<OverlayProblem::Mesh<double>>(m, "Mesh_double")
    .def_readwrite("n", &OverlayProblem::Mesh<double>::n)
    .def_readwrite("to", &OverlayProblem::Mesh<double>::to)
    .def_readwrite("f", &OverlayProblem::Mesh<double>::f)
    .def_readwrite("h", &OverlayProblem::Mesh<double>::h)
    .def_readwrite("out", &OverlayProblem::Mesh<double>::out)
    .def_readwrite("opp", &OverlayProblem::Mesh<double>::opp)
    .def_readwrite("R", &OverlayProblem::Mesh<double>::R)
    .def_readwrite("type", &OverlayProblem::Mesh<double>::type)
    .def_readwrite("Th_hat", &OverlayProblem::Mesh<double>::Th_hat)
    .def_readwrite("l", &OverlayProblem::Mesh<double>::l);
#ifdef WITH_MPFR
  pybind11::class_<OverlayProblem::Mesh<mpfr::mpreal>>(m, "Mesh_mpf")
    .def_readwrite("n", &OverlayProblem::Mesh<mpfr::mpreal>::n)
    .def_readwrite("to", &OverlayProblem::Mesh<mpfr::mpreal>::to)
    .def_readwrite("f", &OverlayProblem::Mesh<mpfr::mpreal>::f)
    .def_readwrite("h", &OverlayProblem::Mesh<mpfr::mpreal>::h)
    .def_readwrite("out", &OverlayProblem::Mesh<mpfr::mpreal>::out)
    .def_readwrite("opp", &OverlayProblem::Mesh<mpfr::mpreal>::opp)
    .def_readwrite("R", &OverlayProblem::Mesh<mpfr::mpreal>::R)
    .def_readwrite("type", &OverlayProblem::Mesh<mpfr::mpreal>::type)
    .def_readwrite("Th_hat", &OverlayProblem::Mesh<mpfr::mpreal>::Th_hat)
    .def_readwrite("l", &OverlayProblem::Mesh<mpfr::mpreal>::l);
#endif
  pybind11::class_<OverlayProblem::OverlayMesh<double>>(m, "OverlayMesh_double")
    .def_readwrite("n", &OverlayProblem::OverlayMesh<double>::n)
    .def_readwrite("to", &OverlayProblem::OverlayMesh<double>::to)
    .def_readwrite("f", &OverlayProblem::OverlayMesh<double>::f)
    .def_readwrite("h", &OverlayProblem::OverlayMesh<double>::h)
    .def_readwrite("out", &OverlayProblem::OverlayMesh<double>::out)
    .def_readwrite("opp", &OverlayProblem::OverlayMesh<double>::opp)
    .def_readwrite("R", &OverlayProblem::OverlayMesh<double>::R)
    .def_readwrite("type", &OverlayProblem::OverlayMesh<double>::type)
    .def_readwrite("prev", &OverlayProblem::OverlayMesh<double>::prev)
    .def_readwrite("first_segment", &OverlayProblem::OverlayMesh<double>::first_segment)
    .def_readwrite("origin", &OverlayProblem::OverlayMesh<double>::origin)
    .def_readwrite("origin_of_origin", &OverlayProblem::OverlayMesh<double>::origin_of_origin)
    .def_readwrite("vertex_type", &OverlayProblem::OverlayMesh<double>::vertex_type)
    .def_readwrite("edge_type", &OverlayProblem::OverlayMesh<double>::edge_type)
    .def_readwrite("seg_bcs", &OverlayProblem::OverlayMesh<double>::seg_bcs)
    .def_readwrite("_m", &OverlayProblem::OverlayMesh<double>::_m);
#ifdef WITH_MPFR
  pybind11::class_<OverlayProblem::OverlayMesh<mpfr::mpreal>>(m, "OverlayMesh_mpf")
    .def_readwrite("n", &OverlayProblem::OverlayMesh<mpfr::mpreal>::n)
    .def_readwrite("to", &OverlayProblem::OverlayMesh<mpfr::mpreal>::to)
    .def_readwrite("f", &OverlayProblem::OverlayMesh<mpfr::mpreal>::f)
    .def_readwrite("h", &OverlayProblem::OverlayMesh<mpfr::mpreal>::h)
    .def_readwrite("out", &OverlayProblem::OverlayMesh<mpfr::mpreal>::out)
    .def_readwrite("opp", &OverlayProblem::OverlayMesh<mpfr::mpreal>::opp)
    .def_readwrite("R", &OverlayProblem::OverlayMesh<mpfr::mpreal>::R)
    .def_readwrite("type", &OverlayProblem::OverlayMesh<mpfr::mpreal>::type)
    .def_readwrite("prev", &OverlayProblem::OverlayMesh<mpfr::mpreal>::prev)
    .def_readwrite("first_segment", &OverlayProblem::OverlayMesh<mpfr::mpreal>::first_segment)
    .def_readwrite("origin", &OverlayProblem::OverlayMesh<mpfr::mpreal>::origin)
    .def_readwrite("origin_of_origin", &OverlayProblem::OverlayMesh<mpfr::mpreal>::origin_of_origin)
    .def_readwrite("vertex_type", &OverlayProblem::OverlayMesh<mpfr::mpreal>::vertex_type)
    .def_readwrite("edge_type", &OverlayProblem::OverlayMesh<mpfr::mpreal>::edge_type)
    .def_readwrite("seg_bcs", &OverlayProblem::OverlayMesh<mpfr::mpreal>::seg_bcs)
    .def_readwrite("_m", &OverlayProblem::OverlayMesh<mpfr::mpreal>::_m);
#endif

  m.def("layout_float", &compute_layout<double>, "layout function",
    pybind11::arg("mesh"), pybind11::arg("u"), pybind11::arg("is_cut_h"),
    pybind11::arg("start_h"),
    pybind11::call_guard<pybind11::scoped_ostream_redirect,
    pybind11::scoped_estream_redirect>());
#ifdef WITH_MPFR
  m.def("layout_mpf", &compute_layout<mpfr::mpreal>, "layout function",
    pybind11::arg("mesh"), pybind11::arg("u"), pybind11::arg("is_cut_h"),
    pybind11::arg("start_h"),
    pybind11::call_guard<pybind11::scoped_ostream_redirect,
    pybind11::scoped_estream_redirect>());
  m.def("fv_to_double_mpf", &FV_to_double<mpfr::mpreal>, "convert v, f to mesh data structure in mpreal");
  m.def("newton_decrement_samples",
        &ConformalIdealDelaunay<mpfr::mpreal>::SampleNewtonDecrementStl,
        "sample projected gradient along the newton descent direction",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
        pybind11::scoped_estream_redirect>());
#endif
    
  m.def("fv_to_double_float", &FV_to_double<double>, "convert v, f to mesh data structure in double");
  m.def("conformal_metric_double", &conformal_metric<double>, "Main conformal method in double",
    pybind11::arg("V"), pybind11::arg("F"), pybind11::arg("Theta_hat"), pybind11::arg("pt_fids"), pybind11::arg("pt_bcs"),
    pybind11::arg("alg_params") = nullptr,
    pybind11::arg("ls_params") = nullptr,
    pybind11::arg("stats_params") = nullptr,
    pybind11::call_guard<pybind11::scoped_ostream_redirect,
    pybind11::scoped_estream_redirect>());

#ifdef WITH_MPFR
  m.def("conformal_metric_mpf", &conformal_metric<mpfr::mpreal>, "Main conformal method in multiprecision",
    pybind11::arg("V"), pybind11::arg("F"), pybind11::arg("Theta_hat"), pybind11::arg("pt_fids"), pybind11::arg("pt_bcs"),
    pybind11::arg("alg_params") = nullptr,
    pybind11::arg("ls_params") = nullptr,
    pybind11::arg("stats_params") = nullptr,
    pybind11::call_guard<pybind11::scoped_ostream_redirect,
    pybind11::scoped_estream_redirect>());
#endif
  m.def("get_layout_double", &get_layout<double>, "get layout", 
    pybind11::arg("m"), pybind11::arg("u"), pybind11::arg("bd"), 
    pybind11::arg("singularities"), pybind11::arg("do_trim") = false,
    pybind11::arg("root") = -1,
    pybind11::call_guard<pybind11::scoped_ostream_redirect,
    pybind11::scoped_estream_redirect>());
#ifdef WITH_MPFR
  m.def("get_layout_mpf", &get_layout<mpfr::mpreal>, "get layout",
    pybind11::arg("m"), pybind11::arg("u"), pybind11::arg("bd"), 
    pybind11::arg("singularities"), pybind11::arg("do_trim") = false,
    pybind11::arg("root") = -1,
    pybind11::call_guard<pybind11::scoped_ostream_redirect,
    pybind11::scoped_estream_redirect>());
#endif
  // interface
  m.def("conformal_metric_cl_double", &conformal_metric_CL<double>, "get conformal metric, output: halfedge connectivity(n,opp), l(per halfedge)");
  m.def("conformal_metric_vl_double", &conformal_metric_VL<double>, "get conformal metric, output: (V,F), l(per halfedge)" );
  m.def("conformal_parametrization_cl_double", &conformal_parametrization_CL<double>, "get conformal parametrization, output: connectivity(n,opp), u coordinate(per corner), v coordinate(per corner)");
  m.def("conformal_parametrization_vf_double", &conformal_parametrization_VL<double>, "get conformal parametrization,output: (V, F, u, v)",
  pybind11::arg("V"), pybind11::arg("F"), pybind11::arg("Theta_hat"),
    pybind11::arg("alg_params") = nullptr,
    pybind11::arg("ls_params") = nullptr,
    pybind11::arg("stats_params") = nullptr,pybind11::call_guard<pybind11::scoped_ostream_redirect,
    pybind11::scoped_estream_redirect>());
#ifdef WITH_MPFR
  m.def("conformal_metric_cl_mpf", &conformal_metric_CL<mpfr::mpreal>, "get conformal metric, output: halfedge connectivity(n,opp), l(per halfedge) in multiprecision");
  m.def("conformal_metric_vl_mpf", &conformal_metric_VL<mpfr::mpreal>, "get conformal metric, output: (V,F), l(per halfedge) in multiprecision" );
  m.def("conformal_parametrization_cl_mpf", &conformal_parametrization_CL<mpfr::mpreal>, "get conformal parametrization, output: connectivity(n,opp), u coordinate(per corner), v coordinate(per corner) in multiprecision");
  m.def("conformal_parametrization_vf_mpf", &conformal_parametrization_VL<mpfr::mpreal>, "get conformal parametrization,output: (V, F, u, v) in multiprecision",pybind11::arg("V"), pybind11::arg("F"), pybind11::arg("Theta_hat"),
    pybind11::arg("alg_params") = nullptr,
    pybind11::arg("ls_params") = nullptr,
    pybind11::arg("stats_params") = nullptr,pybind11::call_guard<pybind11::scoped_ostream_redirect,
    pybind11::scoped_estream_redirect>());
#endif

  m.def("write_texture_obj_double", &write_texture_obj<double>, "write obj file with texture coordinates");
#ifdef WITH_MPFR
  m.def("write_texture_obj_mpf", &write_texture_obj<mpfr::mpreal>, "write obj file with texture coordinates in mpfr");
#endif
#ifdef RENDER_TEXTURE
  m.def("cpp_viewer", &cpp_viewer, "viewer mesh in libigl gui");
  m.def("get_pt_mat", &get_pt_mat, "get pt_mat");
  m.def("get_edges" , &get_edges , "get edges mesh");
#endif
}

#endif
