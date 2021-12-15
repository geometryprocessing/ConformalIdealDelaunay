import igl
import numpy as np
from overload_math import *
from tqdm import trange, tqdm
from conformal_py import *
from matplotlib import cm
import mpmath

# get the number of cut-edges touching to[h0]
def count_valence(n, opp, h0, is_cut):
  hi = opp[n[n[h0]]]
  if is_cut[h0]:
    valence = 1
  else:
    valence = 0
  while hi != h0:
    if is_cut[hi]:
      valence += 1
    hi = opp[n[n[hi]]]
  return valence

# given tree-like cutgraph - try to remove any degree-1 vertices that's not a cone
def trim_cuts(n, opp, to, cones, is_cut_h):

    any_trimmed = True
    while any_trimmed: # repeatedly trim degree-1 cuts
      any_trimmed = False
      for hi in range(len(opp)):
        v0 = to[hi]; v1 = to[opp[hi]]
        valence0 = count_valence(n, opp, opp[hi], is_cut_h)
        valence1 = count_valence(n, opp, hi,      is_cut_h)
        if is_cut_h[hi] and ((valence0 == 1 and v0 not in cones) or (valence1 == 1 and v1 not in cones)):
            is_cut_h[hi] = False
            is_cut_h[opp[hi]] = False
            any_trimmed = True
    

def add_shading(color_rgb, v3d, f, fid_mat_input, bc_mat_input, view, proj, flat_shading=False):
    #compute normals (per face)
    normals = igl.per_face_normals(v3d,f,np.array([1.0,1.0,1.0]))
    pv_normals = igl.per_vertex_normals(v3d, f)
    pv_normals4 = np.zeros((pv_normals.shape[0], 4))
    pv_normals4[:pv_normals.shape[0],:3] = pv_normals
    normals4 = np.zeros((normals.shape[0],4))
    normals4[:normals.shape[0],:3] = normals

    ao = igl.ambient_occlusion(v3d, f, v3d, pv_normals,500)

    # normal transformation matrix
    norm_trans = np.linalg.inv(view).transpose()

    light_eye = np.array([0.0, 0.3, 0.0])
    (H, W, _) = color_rgb.shape
    for i in trange(H):
        for j in range(W):
            fid = fid_mat_input[i][j]
            bc = bc_mat_input[i][j]
            if fid > -1:
                diff = color_rgb[i,j,:]
                amb = 0.2 * diff
                spec = 0.3 + 0.1 * (diff - 0.3)
                ao_factor = ao[f[fid, 0]] * bc[0] + ao[f[fid, 1]] * bc[1] + ao[f[fid, 2]] * bc[2]

                pos = v3d[f[fid, 0]] * bc[0] + v3d[f[fid, 1]] * bc[1] + v3d[f[fid, 2]] * bc[2]
                pos4 = np.ones(4); pos4[0:3] = pos
                pos_eye = np.dot(view, pos4)[0:3]
                if flat_shading:
                    norm4 = normals4[fid]
                else:
                    norm4 = pv_normals4[f[fid, 0]] * bc[0] + pv_normals4[f[fid, 1]] * bc[1] + pv_normals4[f[fid, 2]] * bc[2]
                norm_eye = np.dot(norm_trans, norm4)[0:3]
                norm_eye = norm_eye / np.linalg.norm(norm_eye)

                # diff color
                vec_to_light_eye = light_eye - pos_eye
                dir_to_light_eye = vec_to_light_eye / np.linalg.norm(vec_to_light_eye)
                clamped_dot_prod = max(np.dot(dir_to_light_eye, norm_eye), 0)
                color_diff = clamped_dot_prod * diff

                # spec color
                proj_to_norm = np.dot(-dir_to_light_eye, norm_eye) * norm_eye
                refl_eye = proj_to_norm - (-dir_to_light_eye - proj_to_norm)
                surf_to_view_eye = - pos_eye / np.linalg.norm(pos_eye)
                clamped_dot_prod = max(0, np.dot(refl_eye, surf_to_view_eye))
                spec_factor = pow(clamped_dot_prod, 35)
                color_spec = spec_factor * spec

                color_new = amb + 1.2 * color_diff + color_spec
                for k in range(3):
                    color_new[k] = max(0, min(1, color_new[k]))
                color_rgb[i,j,:] = color_new * (0.5 + (1-ao_factor)*0.5)
    return color_rgb

def draw_grid(fid_mat, bc_mat, h, n, to, u, v, phi, cprs, H, W, N_bw = 15, thick = 0.1):
    u_min = float(np.min(u))
    u_max = float(np.max(u))
    v_min = float(np.min(v))
    v_max = float(np.max(v))
    u_unit = (u_max - u_min) / N_bw
    v_unit = u_unit
    
    r_max = np.ceil(np.max(phi) / np.log(2))
    r_min = np.floor(np.min(phi) / np.log(2))
    print("r_max", r_max)
    print("r_min", r_min)
    
    color_rgb_gd = np.zeros((H, W, 3))
    
    for i in trange(H):
        for j in range(W):
            if fid_mat[i][j] > -1:
                fid = fid_mat[i][j]
                bc = bc_mat[i][j]
                e0 = h[fid]
                e1 = n[e0]
                e2 = n[e1]

                phi0 = float(phi[to[e0]])
                phi1 = float(phi[to[e1]])
                phi2 = float(phi[to[e2]])

                u_pt = float(u[e0]) * bc[1] + float(u[e1]) * bc[2] + float(u[e2]) * bc[0]
                v_pt = float(v[e0]) * bc[1] + float(v[e1]) * bc[2] + float(v[e2]) * bc[0]

                phi_pt = float(phi0 * bc[1] + phi1 * bc[2] + phi2 * bc[0])

                r = r_max - phi_pt / np.log(2)

                r_round = np.round(r)
                r_floor = np.floor(r)
                r_ceil = np.ceil(r)

                u_thick = 0.1 * u_unit / (2**r)
                v_thick = 0.1 * v_unit / (2**r)
                if  u_thick < (u_pt - u_min) % (u_unit / (2**r_floor)) <= u_unit / (2**r_floor) - u_thick and  v_thick < (v_pt - v_min) % (v_unit / (2**r_floor)) <= v_unit / (2**r_floor) - v_thick:
                    color_rgb_gd[i,j,:] = np.array(cm.coolwarm(cprs(float((r_floor) / (r_max - r_min))))[:3]) * (1 + r_floor - r) 
                else:
                    color_rgb_gd[i,j,:] = np.array(cm.coolwarm(cprs(float((r_floor) / (r_max - r_min))))[:3]) * 0.55 * (1 + r_floor - r)

                if  u_thick < (u_pt - u_min) % (u_unit / (2**r_ceil)) <= u_unit / (2**r_ceil) - u_thick and  v_thick < (v_pt - v_min) % (v_unit / (2**r_ceil)) <= v_unit / (2**r_ceil) - v_thick:
                    color_rgb_gd[i,j,:] += np.array(cm.coolwarm(cprs(float((r_ceil) / (r_max - r_min))))[:3]) * (r - r_floor) # mix light colors
                else:
                    color_rgb_gd[i,j,:] += np.array(cm.coolwarm(cprs(float((r_ceil) / (r_max - r_min))))[:3]) * 0.55 * (r - r_floor) # mix dark colors


            elif fid_mat[i][j] == -1:
                color_rgb_gd[i,j,:] = np.array([1.0,1.0,1.0])
            elif fid_mat[i][j] == -2:
                color_rgb_gd[i,j,:] = np.array([0.7,0.1,0.2])
            elif fid_mat[i][j] == -3:
                color_rgb_gd[i,j,:] = np.array([0.5,0.7,0.35])
            elif fid_mat[i][j] == -4:
                color_rgb_gd[i,j,:] = np.array([0,0,0])
            elif fid_mat[i][j] == -5:
                color_rgb_gd[i,j,:] = np.array([1,0.1,0.1])

    return color_rgb_gd
def draw_grid_mpf(fid_mat, bc_mat, h, n, to, u, v, phi, cprs, H, W, N_bw = 15, thick = 0.1, dps = 300):
    mp.dps = dps
    u_min = np.min(u)
    u_max = np.max(u)
    v_min = np.min(v)
    v_max = np.max(v)

    u_unit = (u_max - u_min) / N_bw
    v_unit = u_unit
    
    print("u_unit", u_unit)
    r_max = np.ceil(np.max(phi) / np.log(2))
    r_min = np.floor(np.min(phi) / np.log(2))
    print("r_max", r_max)
    print("r_min", r_min)
    color_rgb_gd = np.zeros((H, W, 3))
    for i in trange(H):
        for j in range(W):
            if fid_mat[i][j] > -1:
                fid = fid_mat[i][j]
                bc = bc_mat[i][j]
                e0 = h[fid]
                e1 = n[e0]
                e2 = n[e1]

                phi0 = float(phi[to[e0]])
                phi1 = float(phi[to[e1]])
                phi2 = float(phi[to[e2]])
                bc_sum = mp.mpf(str(bc[0])) + mp.mpf(str(bc[1])) + mp.mpf(str(bc[2]))
                bc0 = mp.mpf(str(bc[0])) / bc_sum
                bc1 = mp.mpf(str(bc[1])) / bc_sum
                bc2 = mp.mpf(str(bc[2])) / bc_sum
                u_pt = (u[e0]) * bc1 + (u[e1]) * bc2 + (u[e2]) * bc0
                v_pt = (v[e0]) * bc1 + (v[e1]) * bc2 + (v[e2]) * bc0
                phi_pt = float(phi0 * bc1 + phi1 * bc2 + phi2 * bc0)
                r = r_max - phi_pt / np.log(2)

                r_round = np.round(r)
                r_floor = np.floor(r)
                r_ceil = np.ceil(r)

                u_thick = 0.1 * mp.mpf(u_unit) / (mp.mpf(2)**r)
                v_thick = u_thick

                if  u_thick < (u_pt - u_min) % (mp.mpf(u_unit) / (mp.mpf(2)**r_floor)) <= mp.mpf(u_unit) / (mp.mpf(2)**r_floor) - u_thick and  v_thick < (v_pt - v_min) % (mp.mpf(v_unit) / (mp.mpf(2)**r_floor)) <= mp.mpf(v_unit) / (mp.mpf(2)**r_floor) - v_thick:
                    color_rgb_gd[i,j,:] = np.array(cm.coolwarm(cprs(float((r_floor) / (r_max - r_min))))[:3]) * (1 + r_floor - r) 
                else:
                    color_rgb_gd[i,j,:] = np.array(cm.coolwarm(cprs(float((r_floor) / (r_max - r_min))))[:3]) * 0.55 * (1 + r_floor - r)

                if  u_thick < (u_pt - u_min) % (mp.mpf(u_unit) / (mp.mpf(2)**r_ceil)) <= mp.mpf(u_unit) / (mp.mpf(2)**r_ceil) - u_thick and  v_thick < (v_pt - v_min) % ( mp.mpf(v_unit) / (mp.mpf(2)**r_ceil)) <= mp.mpf(v_unit) / (mp.mpf(2)**r_ceil) - v_thick:
                    color_rgb_gd[i,j,:] += np.array(cm.coolwarm(cprs(float((r_ceil) / (r_max - r_min))))[:3]) * (r - r_floor) # mix light colors
                else:
                    color_rgb_gd[i,j,:] += np.array(cm.coolwarm(cprs(float((r_ceil) / (r_max - r_min))))[:3]) * 0.55 * (r - r_floor) # mix dark colors
            elif fid_mat[i][j] == -1:
                color_rgb_gd[i,j,:] = np.array([1.0,1.0,1.0])
            elif fid_mat[i][j] == -2:
                color_rgb_gd[i,j,:] = np.array([0.7,0.1,0.2])
            elif fid_mat[i][j] == -3:
                color_rgb_gd[i,j,:] = np.array([0.5,0.7,0.35])
            elif fid_mat[i][j] == -4:
                color_rgb_gd[i,j,:] = np.array([0,0,0])
            elif fid_mat[i][j] == -5:
                color_rgb_gd[i,j,:] = np.array([1,0.1,0.1])
    return color_rgb_gd

def add_cut_to_sin(n, opp, to, cones, edge_labels, is_cut_h, reindex, v3d, f, bd_thick, fid_mat, cam, H, W, is_mesh_doubled):
    trim_cuts(n, opp, to, cones, is_cut_h)

    cut_to_sin_list = [];
    cnt_cut = 0

    for he in range(len(is_cut_h)):
        if (is_cut_h[he] == True) and (not is_mesh_doubled or (is_mesh_doubled and edge_labels[he] == '\x01')):
            vid_from = to[opp[he]]
            vid_to = to[he]
            cut_to_sin_list.append([reindex[vid_from], reindex[vid_to]])
            cnt_cut += 1
    v_cut_to_sin, f_cut_to_sin = get_edges(v3d, f, cut_to_sin_list, bd_thick)
    fid_mat_sin, bc_mat_sin = get_pt_mat(cam, v3d, f, v_cut_to_sin, f_cut_to_sin, 0, 0, W, H)
    for i in trange(H):
        for j in range(W):
            if fid_mat_sin[i][j] == -4 and fid_mat[i][j] >= 0:
                fid_mat[i][j] = -5
    return fid_mat