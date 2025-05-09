import numpy as np
import torch
import math

from myutils.mi_plugin import *

def sample_env1(wo,
               normals,
               mat,
               use_mesh_normals,
               device,
               envmap_t):
    sample2 = torch.rand(2, len(normals), device=device)
    wi, pdf = sample_envmap(envmap_t, sample2)
    brdf, pdf_brdf = eval_brdf(wi, wo, normals, mat, use_mesh_normals)
    brdf_weight = brdf / (pdf+1e-6)
    # brdf_weight = torch.where(pdf_L > 1e-6, brdf / pdf_L ,torch.tensor(1e-6,device=device))
    brdf_weight = torch.nan_to_num(brdf_weight, nan=0, posinf=0, neginf=0)

    return wi, pdf, brdf_weight


def sample_brdf1(wo, normals, mat, use_mesh_normals, device):
    sample1 = torch.rand(len(normals), device=device)
    sample2 = torch.rand(len(normals), 2, device=device)
    wi, pdf, brdf_weight = sample_brdf(sample1, sample2, wo, normals, mat,
                                       use_mesh_normals)
    return wi, pdf, brdf_weight
def lookup_envmap(envmap, w):
    shape = envmap.shape
    height, width = shape[0], shape[1]
    phi = torch.atan2(w[..., 0], -w[..., 2]) / (2.0 * math.pi)
    u = torch.clamp((phi * width + width) % width, 0, width - 1).int()
    theta = torch.acos(w[..., 1]) / (math.pi)
    v = torch.clamp(theta * height, 0, height - 1).int()
    return envmap[v.long(), u.long()]


def luminance(x):
    return 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]


def build_envmap(envmap):  # envmap is a PyTorch tensor
    # marginal_cdf, conditional_cdf
    shape = envmap.shape
    h, w, _ = envmap.shape

    h01 = torch.tensor([(v + 0.5) / h for v in range(h)], device=envmap.device)
    lum = 0.299 * envmap[:, :, 0] + 0.587 * envmap[:, :, 1] + 0.114 * envmap[:, :, 2]  # (512, 1024)
    sin_theta = torch.sin(torch.pi * h01).reshape((h, -1))
    lum_sin = lum * sin_theta

    conditional_cdf = torch.cumsum(lum_sin, dim=1)
    marginal_cdf = torch.sum(conditional_cdf, dim=1)
    marginal_cdf = torch.cumsum(marginal_cdf, dim=0)
    # breakpoint()

    # Normalize
    conditional_cdf = conditional_cdf / (conditional_cdf[:, -1].reshape((h, -1))+1e-6)
    marginal_cdf = marginal_cdf / (marginal_cdf[-1]+1e-6)
    
    return {
        "envmap": envmap,
        "c_cdf": conditional_cdf,
        "m_cdf": marginal_cdf
    }

def build_envmap_np(envmap):  # envmap is numpy array
    # marginal_cdf, conditional_cdf
    shape = envmap.shape
    h, w, _ = envmap.shape

    h01 = np.array([(v + 0.5) / h for v in range(h)])
    lum = np.apply_along_axis(luminance, 2, envmap)  # (512, 1024)
    sin_theta = np.sin(np.pi * (h01)).reshape((h, -1))
    lum_sin = (np.multiply(lum, sin_theta))

    conditional_cdf = np.cumsum(lum_sin, axis=1)
    marginal_cdf = np.sum(conditional_cdf, axis=1)
    marginal_cdf = np.cumsum(marginal_cdf)

    # Normalize
    conditional_cdf = (conditional_cdf / conditional_cdf[:, -1].reshape(
        (h, -1)))
    marginal_cdf = marginal_cdf / marginal_cdf[-1]
    return {
        "envmap": torch.from_numpy(envmap).cuda().share_memory_(),
        "c_cdf": torch.from_numpy(conditional_cdf).cuda().share_memory_(),
        "m_cdf": torch.from_numpy(marginal_cdf).cuda().share_memory_()
    }

def interp_1d(buf, x, index):
    mask = (index > 0).float()
    index 
    interp_if = (x - buf[index - 1]) / (buf[index] - buf[index - 1])
    interp_else = x / buf[index]
    return mask * interp_if + (1 - mask) * interp_else

def interp_2d(buf, x, index, row=0):
    mask = (index > 0).float()
    # index[index==0] = 1
    index1 = index-1
    index1[index1==-1] = 0
    index[index==32] = 31
    interp_if = (x - buf[row, index1]) / (buf[row, index] - buf[row, index1])
    interp_else = x / buf[row, index]
    return mask * interp_if + (1 - mask) * interp_else

def get_pdf_from_cdf_1d(cdf, idx):
    mask = (idx > 0).float()
    pdf_if = cdf[idx] - cdf[idx - 1]
    pdf_else = cdf[idx]
    return mask * pdf_if + (1 - mask) * pdf_else

def get_pdf_from_cdf_2d(cdf, idx, row_offset=0):
    mask = (idx > 0).float()
    
    idx1 = idx-1
    idx1[idx1==-1] = 0
    idx[idx==32] = 31
    pdf_if = cdf[row_offset, idx] - cdf[row_offset, idx1]
    pdf_else = cdf[row_offset, idx]
    return mask * pdf_if + (1 - mask) * pdf_else



def compute_direction(theta, phi):
    return angle2xyz(theta, phi)


def cdf_search_1d(cdf, x):
    return torch.searchsorted(cdf, x)


def cdf_search_2d(cdf, x, row_offset):
    return torch.searchsorted(cdf[row_offset, :], x)


def importance_sample(envmap_dict, sample2):
    envmap = envmap_dict['envmap']
    marg_cdf = envmap_dict['m_cdf']
    cond_cdf = envmap_dict['c_cdf']

    # Cond (h, w)
    h, w, _ = envmap.shape  # (16, 32)
    x0 = sample2[0, :].reshape(-1, 1)  # 1D 512 * 512 * 64 = n
    x1 = sample2[1, :].reshape(-1, 1)  # 1D 512 * 512 * 64 = n

    # Sample vertical (height) //  Sample row
    v_idx = cdf_search_1d(marg_cdf, x0)  # shape = (n)
    dv = interp_1d(marg_cdf, x0, v_idx)
    pdf_m = get_pdf_from_cdf_1d(marg_cdf, v_idx)
    v = (v_idx + dv)

    # Sample horizontally (width)
    u_idx = cdf_search_2d(cond_cdf, x1, v_idx.flatten())
    du = interp_2d(cond_cdf, x1, u_idx, v_idx)
    pdf_c = get_pdf_from_cdf_2d(cond_cdf, u_idx, v_idx)
    u = (u_idx + du)
    # u = u_idx

    theta = (v) * math.pi / h
    phi = (2.0 * u * math.pi) / w
    two_pi_pi = 2.0 * math.pi * math.pi
    dirs = compute_direction(theta.flatten(), phi.flatten()).float()
    emission = lookup_envmap(envmap, dirs)
    pdf = (h * w) * (pdf_c * pdf_m) / (two_pi_pi * torch.sin(theta))

    return dirs, pdf, emission


def sample_envmap(envmap_dict, sample2):
    envmap = envmap_dict['envmap']
    marg_cdf = envmap_dict['m_cdf']
    cond_cdf = envmap_dict['c_cdf']

    # Cond (h, w)
    h, w, _ = envmap.shape  # (16, 32)
    x0 = sample2[0, :].reshape(-1, 1)  # 1D 512 * 512 * 64 = n
    x1 = sample2[1, :].reshape(-1, 1)  # 1D 512 * 512 * 64 = n

    # Sample vertical (height) //  Sample row
    v_idx = cdf_search_1d(marg_cdf, x0)  # shape = (n)
    dv = interp_1d(marg_cdf, x0, v_idx)
    pdf_m = get_pdf_from_cdf_1d(marg_cdf, v_idx)
    v = (v_idx + dv)

    # Sample horizontally (width)
    u_idx = cdf_search_2d(cond_cdf, x1, v_idx.flatten())
    # du = interp_2d(cond_cdf, x1, u_idx, v_idx)
    pdf_c = get_pdf_from_cdf_2d(cond_cdf, u_idx, v_idx)
    u = (u_idx)
    # u = u_idx

    theta = (v) * math.pi / h
    phi = (2.0 * u * math.pi) / w
    two_pi_pi = 2.0 * math.pi * math.pi
    dirs = compute_direction(theta.flatten(), phi.flatten()).float()
    pdf = (h * w) * (pdf_c * pdf_m) / (two_pi_pi * torch.sin(theta) + 1e-6)

    return dirs, pdf