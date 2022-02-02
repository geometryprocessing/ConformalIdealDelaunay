import igl
import numpy as np
import mpmath as mp
import os
import argparse
import matplotlib.pyplot as plt
from conformal_py import *
from overload_math import *
from render import *
from collections import namedtuple
from copy import deepcopy
import meshplot as meshp
import pickle

RenderInfo = namedtuple('RenderInfo', 'pt_fids, pt_bcs, fid_mat, bc_mat, cam, bd_thick, view, proj, H, W')

def render_texture(out, name, v3d, f, m, u, cones, reindex, render_info, build_double):
  fid_mat  = render_info.fid_mat
  pt_fids  = render_info.pt_fids
  pt_bcs   = render_info.pt_bcs
  bc_mat   = render_info.bc_mat
  cam      = render_info.cam
  bd_thick = render_info.bd_thick
  view     = render_info.view
  proj     = render_info.proj
  H        = render_info.H
  W        = render_info.W
  reindex = np.array(reindex)
  # update original cone ids to m
  cones = [idx for idx in range(len(reindex)) if reindex[idx] in cones]

  fid_mat_input = deepcopy(fid_mat)
  bc_mat_input = deepcopy(bc_mat)
  cnt = 0
  for i in trange(H):
      for j in range(W):
          if fid_mat[i][j] > -1:
              fid_mat[i][j] = pt_fids[cnt]
              bc_mat[i][j] = pt_bcs[cnt]
              cnt += 1
              
  is_cut_h = []
  if use_mpf:
    u_cpp, v_cpp, is_cut_h = layout_mpf(m, list(u), is_cut_h, -1)
    u_cpp = [mp.mpf(repr(u_cppi)) for u_cppi in u_cpp]
    v_cpp = [mp.mpf(repr(v_cppi)) for v_cppi in v_cpp]
  else:
    u_cpp, v_cpp, is_cut_h = layout_float(m, list(u), is_cut_h, -1)

  fid_mat = add_cut_to_sin(m.n, m.opp, m.to, cones, m.type, is_cut_h, reindex, v3d, f, bd_thick, fid_mat, cam, H, W, build_double)
  N_bw = 10

  def cprs(x):
      x = max(0,min(1,x))
      return max(0, min(1, 3 * x * x - 2 * x * x * x))

  print("draw grid...")
  if use_mpf:
    u = np.array([mp.mpf(repr(ui)) for ui in u])
    color_rgb_gd = draw_grid_mpf(fid_mat, bc_mat, m.h, m.n, m.to, u_cpp, v_cpp, u, cprs, H, W, N_bw)
  else:
    u = np.array(u)
    color_rgb_gd = draw_grid(fid_mat, bc_mat, m.h, m.n, m.to, u_cpp, v_cpp, u, cprs, H, W, N_bw) # faster but less accurate float alternative: draw_grid 

  plt.imsave(out + "/" + name + "_" + str(N_bw) + "_gd_plain.png", color_rgb_gd)

  print("add shading...")
  add_shading(color_rgb_gd, v3d, f, fid_mat_input, bc_mat_input, view, proj)

  plt.imsave(out + "/" + name + "_" + str(N_bw) + "_gd.png", color_rgb_gd)

def do_conformal(m, dir, out, output_type="param", output_format="obj", use_mpf=False, error_log=False, energy_cond=False, energy_samples=False, suffix=None, flip_count=False, prec=None, no_round_Th_hat=False, print_summary=False, eps=None, no_plot_result=False, bypass_overlay=False, max_itr=500, no_lm_reset=False, do_reduction=False,lambda0=1,bound_norm_thres=1, log_level=2):
    if use_mpf:
      if prec == None:
        mp.prec = 100
      else:
        mp.prec = prec
      if eps == None:
        eps = 0
      float_type = mp.mpf
    else:
      float_type = float
      if eps == None:
        eps = 0

    v3d, f = igl.read_triangle_mesh(dir+'/'+m)
    dot_index = m.rfind(".")
    name = m[:dot_index]

    if suffix != None:
      name = name+"_"+str(suffix)
    else:
      name = name
    
    Th_hat = np.loadtxt(dir+"/"+name+"_Th_hat", dtype=str)
    Th_hat = nparray_from_float64(Th_hat,float_type) 
    
    if use_mpf and not no_round_Th_hat:
      # Round rational multiples of pi to multiprecision accuray
      for i,angle in enumerate(Th_hat):
        n=round(60*angle/mp.pi)
        Th_hat[i] = n*mp.pi/60
    
    # identify the cones - used for visualization
    is_bd = igl.is_border_vertex(v3d, f)
    
    # need to build double mesh when it has boundary
    build_double = (np.sum(is_bd) != 0)
    cones = np.array([id for id in range(len(Th_hat)) if np.abs(Th_hat[id]-2*mpi(float_type)) > 1e-15 and not is_bd[id]], dtype=int)

    W = 500; H = 300 # figure size
    bd_thick = 2; sin_size = 3
    pt_fids = []; pt_bcs=[]
    
    if output_type == "render" and output_format == "png" and not no_plot_result:
      with open("data/cameras/" + name + "_camera.pickle", 'rb') as fp:
        cam = pickle.load(fp)
        vc = pickle.load(fp)
        fc = pickle.load(fp)
        red_size = pickle.load(fp)
        blue_size = pickle.load(fp)
      (view, proj, vp) = cam
      if not build_double:
        fc = fc[:red_size+blue_size,:]
      fid_mat, bc_mat = get_pt_mat(cam, v3d, f, vc, fc, red_size, blue_size, W, H)
      for i in range(H):
          for j in range(W):
              if fid_mat[i][j] > -1:
                  pt_fids.append(fid_mat[i][j])
                  pt_bcs.append(bc_mat[i][j])
    
    # Create algorithm parameter struct
    alg_params = AlgorithmParameters()
    alg_params.MPFR_PREC = mp.prec
    alg_params.initial_ptolemy = False
    alg_params.error_eps = eps
    if use_mpf:
      alg_params.min_lambda = pow(2, -100)
    else:
      alg_params.min_lambda = 1e-16
    alg_params.newton_decr_thres = -0.01 * eps * eps;
    alg_params.max_itr = max_itr
    alg_params.bypass_overlay = bypass_overlay;

    stats_params = StatsParameters()
    stats_params.flip_count = flip_count
    stats_params.output_dir = out
    if use_mpf:
      stats_params.name = name + "_mpf"
    else:
      stats_params.name = name + "_float"
    stats_params.print_summary = print_summary
    stats_params.error_log = error_log
    stats_params.log_level = log_level

    # Create line search parameter struct
    ls_params = LineSearchParameters()
    ls_params.energy_cond = energy_cond
    ls_params.energy_samples = energy_samples
    ls_params.do_reduction = do_reduction
    ls_params.do_grad_norm_decrease = True
    ls_params.bound_norm_thres = bound_norm_thres
    ls_params.lambda0 = lambda0
    ls_params.reset_lambda = not no_lm_reset
    
    if float_type == float:
      if output_type == "he_metric" and output_format == "pickle":
        n, opp, l = conformal_metric_cl_double(v3d, f, Th_hat, alg_params, ls_params, stats_params)
        with open(out + "/" + name + "_out.pickle", 'wb') as pf:
          pickle.dump((n, opp, l), pf)
      elif output_type == "vf_metric" and output_format == "pickle":
        vo, fo, l = conformal_metric_vl_double(v3d, f, Th_hat, alg_params, ls_params, stats_params)
        with open(out + "/" + name + "_out.pickle", 'wb') as pf:
          pickle.dump((vo, fo, l), pf)
      elif output_type == "param" and output_format == "pickle":
        n, opp, u, v = conformal_parametrization_cl_double(v3d, f, Th_hat, alg_params, ls_params, stats_params)
        with open(out + "/" + name + "_out.pickle", 'wb') as pf:
          pickle.dump((n, opp, u, v), pf)
      elif output_type == "param" and output_format == "obj":
        vo, fo, u, v, ft, fn_to_f = conformal_parametrization_vf_double(v3d, f, Th_hat, alg_params, ls_params, stats_params)
        write_texture_obj_double(out + "/" + name + "_out.obj", vo, fo, u, v, ft)
      elif output_type == "render" and output_format == "png": # for texture rendering
        m_o, u, pt_fids, pt_bcs, reindex, _ = conformal_metric_double(v3d, f, Th_hat, pt_fids, pt_bcs, alg_params, ls_params, stats_params);
        m = m_o._m
        if not no_plot_result:
          render_info = RenderInfo(pt_fids, pt_bcs, fid_mat, bc_mat, cam, bd_thick, view, proj, H, W)
          render_texture(out, name, v3d, f, m, u, cones, reindex, render_info, build_double)
      else:
        print("non-supported output-type/output-format")
        print("output_type options:")
        print(" 'render'")
        print(" 'vf_metric'")
        print(" 'he_metric'")
        print(" 'param'")
        print("output format options:")
        print(" 'png' (compatible with 'render' only)")
        print(" 'pickle' (compatible with 'he_metric', 'vf_metric' and 'param')")
        print(" 'obj' (compatible with 'param')")

    else:
      set_mpf_prec(alg_params.MPFR_PREC)
      vnstr = np.vectorize(lambda a:str(repr(a))[5:-2])
      Th_hat = vnstr(Th_hat)
      if output_type == "he_metric" and output_format == "pickle":
        n, opp, l = conformal_metric_cl_mpf(v3d, f, Th_hat, alg_params, ls_params, stats_params)
        l_str = np.array([str(l[idx]) for idx in range(len(l))])
        with open(out + "/" + name + "_out.pickle", 'wb') as pf:
          pickle.dump((n, opp, l_str), pf)
      elif output_type == "vf_metric" and output_format == "pickle":
        vo, fo, l = conformal_metric_vl_mpf(v3d, f, Th_hat, alg_params, ls_params, stats_params)
        vo_str = [[str(vo[i][k]) for k in range(3)] for i in range(len(vo))]
        l_str = np.array([str(l[idx]) for idx in range(len(l))])
        with open(out + "/" + name + "_out.pickle", 'wb') as pf:
          pickle.dump((vo_str, fo, l_str), pf)
      elif output_type == "param" and output_format == "pickle":
        n, opp, u, v = conformal_parametrization_cl_mpf(v3d, f, Th_hat, alg_params, ls_params, stats_params)
        u_str = [str(u[i]) for i in range(len(u))]
        v_str = [str(v[i]) for i in range(len(v))] 
        with open(out + "/" + name + "_out.pickle", 'wb') as pf:
          pickle.dump((n, opp, u_str, v_str), pf)
      elif output_type == "param" and output_format == "obj":
        vo, fo, u, v, ft = conformal_parametrization_vf_mpf(v3d, f, Th_hat, alg_params, ls_params, stats_params)
        vo_fl = [[float(str(vo[i][k])) for k in range(3)] for i in range(len(vo))]
        u_fl = [float(str(u[i])) for i in range(len(u))]
        v_fl = [float(str(v[i])) for i in range(len(v))]
        write_texture_obj_double(out + "/" + name + "_out.obj", vo_fl, fo, u_fl, v_fl, ft)
      elif output_type == "render" and output_format == "png": # default interface - for texture rendering
        m_o, u, pt_fids, pt_bcs, reindex, _ = conformal_metric_mpf(v3d, f, Th_hat, pt_fids, pt_bcs, alg_params, ls_params, stats_params); 
        m = m_o._m
        if not no_plot_result:
          render_info = RenderInfo(pt_fids, pt_bcs, fid_mat, bc_mat, cam, bd_thick, view, proj, H, W)
          render_texture(out, name, v3d, f, m, u, cones, reindex, render_info, build_double)
      else:
        print("non-supported output-type/output-format")
        print("output_type options:")
        print(" 'render'")
        print(" 'vf_metric'")
        print(" 'he_metric'")
        print(" 'param'")
        print("output format options:")
        print(" 'png' (compatible with 'render' only)")
        print(" 'pickle' (compatible with 'he_metric', 'vf_metric' and 'param')")
        print(" 'obj' (compatible with 'param')")
if __name__ == "__main__":

  # Parse arguments for the script
  parser = argparse.ArgumentParser(description='Run the conformal map with options.')

  parser.add_argument("-i", "--input",      help="input folder that stores obj files and Th_hat")
  parser.add_argument("-o", "--output",     help="output folder for stats", default="out")
  parser.add_argument("-f", "--fname",      help="filename of the obj file")
  parser.add_argument("--use_mpf",          action="store_true", help="True for enable multiprecision", default=False)
  parser.add_argument("--do_reduction",     action="store_true", help="do reduction for search direction", default=False)
  parser.add_argument("-p", "--prec",       help="choose the mantissa value of mpf", type=int)
  parser.add_argument("-m", "--max_itr",    help="choose the maximum number of iterations", type=int, default=50)

  parser.add_argument("--energy_cond",      action="store_true", help="True for enable energy computation for line-search")
  parser.add_argument("--energy_samples",   action="store_true", help="True for write out energy sample and newton decrement before linesearch")
  parser.add_argument("--error_log",        action="store_true", help="True for enable writing out the max/ave angle errors per newton iteration")
  parser.add_argument("--flip_count",       action="store_true", help="True for enable collecting flip type stats")
  parser.add_argument("--no_round_Th_hat",  action="store_true", help="True for NOT rounding Th_hat values to multiples of pi/60")
  parser.add_argument("--print_summary",    action="store_true", help="print a summary table contains target angle range and final max curvature error")
  parser.add_argument("--no_plot_result",   action="store_true", help="True for NOT rendering the results, used only for reproducing figures to speedup.")
  parser.add_argument("--bypass_overlay",       action="store_true", help="True for NOT compute overlay, used only for reproducing figures to speedup.")
  parser.add_argument("--no_lm_reset",      action="store_true", help="True for using double the previous lambda for line search.")
  parser.add_argument("--suffix",           help="id assigned to each model for the random test")
  parser.add_argument("--eps",              help="target error threshold")
  parser.add_argument("--lambda0",          help="initial lambda value", type=float, default=1)
  parser.add_argument("--bound_norm_thres", help="threshold to drop the norm bound", type=float, default=1e-10)
  parser.add_argument("--output_type",      action='store', help="output type selection: 'render', 'he_metric', 'vf_metric', 'param'", type=str, default="render")
  parser.add_argument("--output_format",    action='store', help="output file format selection: 'png', 'pickle', 'obj'", type=str, default="png")
  parser.add_argument("--log_level",        help="console logger info level [verbose 0-6]", type=int, default=2)

  args = parser.parse_args()
  output           = args.output
  input            = args.input
  fname            = args.fname
  use_mpf          = args.use_mpf
  do_reduction     = args.do_reduction 
  max_itr          = args.max_itr
  energy_cond      = args.energy_cond
  error_log        = args.error_log
  flip_count       = args.flip_count
  no_round_Th_hat  = args.no_round_Th_hat
  prec             = args.prec
  no_lm_reset      = args.no_lm_reset
  suffix           = args.suffix
  print_summary    = args.print_summary
  no_plot_result   = args.no_plot_result
  bypass_overlay   = args.bypass_overlay
  eps              = args.eps
  lambda0          = args.lambda0
  bound_norm_thres = args.bound_norm_thres
  log_level        = args.log_level
  energy_samples   = args.energy_samples
  output_type      = args.output_type
  output_format    = args.output_format

  if not os.path.isdir(output):
    os.makedirs(output, exist_ok=True)
  
  if eps != None:
    eps = float(eps)

  do_conformal(fname, input, output, output_type, output_format, use_mpf, error_log, energy_cond, energy_samples, suffix, flip_count, prec, no_round_Th_hat, print_summary, eps, no_plot_result, bypass_overlay, max_itr, no_lm_reset, do_reduction, lambda0, bound_norm_thres, log_level)