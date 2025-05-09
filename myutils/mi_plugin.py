import torch.nn as nn
import math
import torch.nn.functional as NF
import torch
import sys
import os
import drjit as dr
import mitsuba as mi
import numpy as np
import open3d as o3d
# from ...fipt.utils.ops import *
import matplotlib.pyplot as plt
import lovely_tensors as lt
import json
from myutils.misc import get_mat
mi.set_variant('cuda_ad_rgb')
lt.monkey_patch()
sys.path.append('../')
import global_config
import warnings
warnings.simplefilter('once', UserWarning)
import math
# dr.set_flag(dr.JitFlag.VCallRecord, False)
# dr.set_flag(dr.JitFlag.LoopRecord, False)

def get_normal_space(normal):
    """ get matrix transform shading space to normal spanned space
    Args:
        normal: Bx3
    Return:
        Bx3x3 transformation matrix
    """
    v1 = torch.zeros_like(normal)
    tangent = v1.clone()
    v1[...,0] = 1.0
    tangent[...,1] = 1.0

    mask = (v1*normal).sum(-1).abs() <= 1e-1
    tangent[mask] = NF.normalize(torch.cross(v1[mask],normal[mask],dim=-1),dim=-1)
    mask = ~mask
    tangent[mask] = NF.normalize(torch.cross(tangent[mask],normal[mask],dim=-1),dim=-1)

    bitangent = torch.cross(normal,tangent,dim=-1)
    return torch.stack([tangent,bitangent,normal],dim=-1)

def angle2xyz(theta,phi):
    """ spherical coordinates to euclidean
    Args:
        theta,phi: B
    Return:
        Bx3 euclidean coordinates
    """
    sin_theta = torch.sin(theta)
    x = sin_theta*torch.cos(phi)
    y = sin_theta*torch.sin(phi)
    z = torch.cos(theta)
    ret = torch.stack([x,y,z],dim=-1)
    return NF.normalize(ret,dim=-1)

def G1_GGX_Schlick(NoV, eta):
    """ G term of schlick GGX
    eta: roughness
    """
    r = eta
    k = (r+1)
    k = k*k/8
    denom = NoV*(1-k)+k + 1e-6
    return 1 /denom

def G_Smith(NoV,NoL,eta):
    """ Smith shadow masking divided by (NoV*NoL)
    eta: roughness
    """
    g1_l = G1_GGX_Schlick(NoL,eta)
    g1_v = G1_GGX_Schlick(NoV,eta)
    return g1_l*g1_v

def fresnelSchlick(VoH,F0):
    """ schlick fresnel """
    x = (1-VoH)**(5)
    return F0 + (1-F0)*x


def fresnelSchlick_sep(VoH):
    """ two terms of schlick fresnel """
    x = (1-VoH)**(5)
    return (1-x),x

def D_GGX(cos_h,eta):
    """GGX normal distribution
    eta: roughness
    """
    alpha = eta*eta
    alpha2 = alpha*alpha
    denom = (cos_h*cos_h*(alpha2-1.0)+1.0) + 1e-6
    denom = math.pi * denom*denom
    return alpha2/denom


def double_sided(V,N):
    """ double sided normal
    Args:
        V: Bx3 viewing direction
        N: Bx3 normal direction
    Return:
        Bx3 flipped normal towards camera direction
    """
    NoV = (N*V).sum(-1)
    flipped = NoV<0
    N[flipped] = -N[flipped]
    return N


def lerp_specular(specular,roughness):
    """ interpolate specular shadings by roughness
    Args:
        specular: Bx6x3 specular shadings
        roughness: Bx1 roughness in [0.02,1.0]
    Return:
        Bx3 interpolated specular shading
    """
    # remap roughness from to [0,1]
    r_min,r_max = 0.02,1.0
    r_num = specular.shape[-2]
    r = (roughness-r_min)/(r_max-r_min)*(r_num-1)


    r1 = r.ceil().long()
    r0 = r.floor().long()
    r_ = (r-r0)
    s0 = torch.gather(specular,1,r0[...,None].expand(r0.shape[0],1,3))[:,0]
    s1 = torch.gather(specular,1,r1[...,None].expand(r1.shape[0],1,3))[:,0]
    s = s0*(1-r_) + s1*r_
    return s

def diffuse_sampler(sample2,normal):
    """ sampling diffuse lobe: wi ~ NoV/math.pi
    Args:
        sample2: Bx2 uniform samples
        normal: Bx3 normal
    Return:
        wi: Bx3 sampled direction in world space
    """
    theta = torch.asin(sample2[...,0].sqrt())
    phi = math.pi*2*sample2[...,1]
    wi = angle2xyz(theta,phi)

    Nmat = get_normal_space(normal)
    wi = (wi[:,None]@Nmat.permute(0,2,1)).squeeze(1)
    return wi

def specular_sampler(sample2,roughness,wo,normal):
    """ sampling ggx lobe: h ~ D/(VoH*4)*NoH
    Args:
        sample2: Bx3 uniform samples
        roughness: Bx1 roughness
        wo: Bx3 viewing direction
        normal: Bx3 normal
    Return:
        wi: Bx3 sampled direction in world space
    """

    roughness = torch.where(roughness <= 0.0, torch.tensor(1.0, device=roughness.device, dtype=roughness.dtype), roughness)
    alpha = (roughness*roughness).squeeze(-1).data

    # sample half vector
    theta = (1-sample2[...,0])/(sample2[...,0]*(alpha*alpha-1)+1)
    theta = torch.acos(theta.sqrt())
    phi = 2*math.pi*sample2[...,1]
    wh = angle2xyz(theta,phi)

    # half vector to wi
    Nmat = get_normal_space(normal)
    wh = (wh[:,None]@Nmat.permute(0,2,1)).squeeze(1)
    wi = 2*(wo*wh).sum(-1,keepdim=True)*wh-wo
    wi = NF.normalize(wi,dim=-1)
    return wi

def transmission_sampler(sample2, roughness, wo, normal, ior):
    """ sampling refraction lobe: wi based on refraction model
    Args:
        sample2: Bx2 uniform samples
        roughness: Bx1 roughness
        wo: Bx3 viewing direction
        normal: Bx3 normal
        ior: Bx1 index of refraction
    Return:
        wi: Bx3 sampled direction in world space
    """
    alpha = (roughness * roughness).squeeze(-1).data

    # Calculate refractive index ratio (n1/n2)
    eta = 1.0 / ior

    # Sample half vector for refraction using roughness and the GGX distribution
    theta = (1 - sample2[..., 0]) / (sample2[..., 0] * (alpha * alpha - 1) + 1)
    theta = torch.acos(theta.sqrt())
    phi = 2 * math.pi * sample2[..., 1]
    wh = angle2xyz(theta, phi)

    # Transform half vector to world space
    Nmat = get_normal_space(normal)
    wh = (wh[:, None] @ Nmat.permute(0, 2, 1)).squeeze(1)

    # Calculate the direction of the refracted ray using Snell's law
    cos_theta_i = (wo * wh).sum(-1, keepdim=True)
    sin_theta_i = (1 - cos_theta_i ** 2).sqrt().clamp(0, 1)

    sin_theta_t = eta * sin_theta_i
    cos_theta_t = (1 - sin_theta_t ** 2).sqrt()

    wi = eta * -wo + (eta * cos_theta_i - cos_theta_t) * wh
    wi = NF.normalize(wi, dim=-1)

    return wi

def mi_specular_sampler(sample2, roughness, wo, normal):
    """ Sampling GGX lobe: h ~ D/(VoH*4)*NoH
    Args:
        sample2: Bx2 uniform samples
        roughness: Bx1 roughness
        wo: Bx3 viewing direction
        normal: Bx3 normal
    Return:
        wi: Bx3 sampled direction in world space
    """
    alpha = roughness * roughness

    # Sample half vector using GGX distribution
    cos_theta = dr.safe_sqrt((1 - sample2[0]) / (sample2[0] * (alpha * alpha - 1) + 1))
    sin_theta = dr.safe_sqrt(dr.maximum(0, 1 - cos_theta * cos_theta))
    # cos_theta = ((1 - sample2[0]) / (sample2[0] * (alpha * alpha - 1) + 1))**0.5
    # sin_theta = (dr.maximum(0, 1 - cos_theta * cos_theta))**0.5
    phi = 2 * math.pi * sample2[1]


    # Convert spherical coordinates to cartesian coordinates
    wh = mi.Vector3f(
        sin_theta * dr.cos(phi),
        sin_theta * dr.sin(phi),
        cos_theta
    )
    # Construct the normal space matrix
    Nmat = mi.Frame3f(normal)

    # Transform wh into the world space using the normal space matrix
    wh = Nmat.to_world(wh)

    # Calculate wi using the reflection equation
    wi = 2 * dr.dot(wo, wh) * wh - wo
    wi = dr.select(dr.isnan(wi),mi.Float(0.0),wi)
    wi = dr.normalize(wi)
    return wi

def mi_diffuse_sampler(sample2, normal):
    """ Sampling diffuse lobe: wi ~ NoV/math.pi
    Args:
        sample2: Bx2 uniform samples
        normal: Bx3 normal
    Return:
        wi: Bx3 sampled direction in world space
    """
    # Convert samples to angles
    theta = dr.asin(dr.safe_sqrt(sample2[0]))
    phi = 2 * math.pi * sample2[1]
    # theta = mi.Float(torch.round(theta.torch(),decimals=5))
    # Convert spherical coordinates to cartesian coordinates
    wi = mi.Vector3f(
        dr.sin(theta) * dr.cos(phi),
        dr.sin(theta) * dr.sin(phi),
        dr.cos(theta)
    )
    # wi = mi.Vector3f(torch.round(wi.torch(),decimals=5))

    # Construct the normal space matrix
    Nmat = mi.Frame3f(normal)

    # Transform wi into the world space using the normal space matrix
    wi = Nmat.to_world(wi)
    wi = dr.select(dr.isnan(wi),mi.Float(0.0),wi)
    return wi



def sample_brdf(sample1, sample2, wo, normal_geo, mat,use_mesh_normal, *args,
                **kwargs):
    """ importance sampling brdf and get brdf/pdf
    for scratch implementation
    Args:
        sample1: B unifrom samples
        sample2: Bx2 uniform samples
        wo: Bx3 viewing direction
        normal: Bx3 normal
        mat: material dict
    Return:
        wi: Bx3 sampled direction
        pdf: Bx1
        brdf_weight: Bx3 brdf/pdf
    """
    B = sample2.shape[0]
    device = sample2.device

    pdf = torch.zeros(B, device=device)
    brdf = torch.zeros(B, 3, device=device)
    wi = torch.zeros(B, 3, device=device)

    albedo, roughness, metallic, normal = mat['albedo'], mat[
        'roughness'], mat['metallic'], mat['normal']

    roughness = roughness.reshape(-1, 1)
    normal = normal.reshape(-1, 3)
    if use_mesh_normal:
        normal = normal_geo
    if sample1 is None:
        sample1 = torch.rand(B, device=device)
    mask = (sample1 > 0.5)
    # sample diffuse
    wi[mask] = diffuse_sampler(sample2[mask], normal[mask])
    mask = ~mask
    # sample specular
    wi[mask] = specular_sampler(sample2[mask], roughness[mask], wo[mask],
                                normal[mask])

    # get brdf,pdf
    brdf, pdf = eval_brdf(wi, wo, normal, mat,use_mesh_normal)

    brdf_weight = torch.where(pdf > 0, brdf / (pdf + 1e-4),
                                torch.tensor(0.0, device=device))

    # brdf_weight[brdf_weight.isnan()] = 0
    brdf_weight = torch.nan_to_num(brdf_weight, nan=0, posinf=0, neginf=0)

    return wi, pdf, brdf_weight

def eval_brdf(wi, wo, normal_geo, mat, use_mesh_normal, *args, **kwargs):
    """ evaluate BRDF and pdf
    for scratch implementation
    Args:
        wi: Bx3 light direction
        wo: Bx3 viewing direction
        normal: Bx3 normal
        mat: surface BRDF dict
    Return:
        brdf: Bx3
        pdf: Bx1
    """
    albedo, roughness, metallic, normal = mat['albedo'], mat[
        'roughness'], mat['metallic'], mat['normal']

    roughness = roughness.reshape(-1, 1)
    albedo = albedo.reshape(-1, 3)
    metallic = metallic.reshape(-1, 1)
    normal = normal.reshape(-1, 3)
    if use_mesh_normal:
        normal = normal_geo
    h = NF.normalize(wi + wo,
                        dim=-1)  # half vector. mid point between wi and wo
    NoL = (wi * normal).sum(-1, keepdim=True).relu()  #o means dot product
    NoV = (wo * normal).sum(-1, keepdim=True).relu()
    VoH = (wo * h).sum(-1, keepdim=True).relu()
    NoH = (normal * h).sum(-1, keepdim=True).relu()

    # get pdf
    D = D_GGX(NoH, roughness)
    D = torch.nan_to_num(D, nan=0, posinf=0, neginf=0)
    pdf_spec = D.data / (4 * VoH.clamp_min(1e-6)) * NoH
    pdf_diff = NoL / math.pi
    pdf = 0.5 * pdf_spec + 0.5 * pdf_diff

    # get brdf
    kd = albedo * (1 - metallic)  # diffuse coefficient
    ks = 0.04 * (1 - metallic) + albedo * metallic  # specular coefficient

    G = G_Smith(NoV, NoL, roughness)
    G = torch.nan_to_num(G, nan=0, posinf=0, neginf=0)
    F = fresnelSchlick(VoH, ks)
    F = torch.nan_to_num(F, nan=0, posinf=0, neginf=0)

    brdf_diff = kd / math.pi
    brdf_spec = D * G * F / 4.0 * NoH

    brdf = 2.0 * (brdf_diff + brdf_spec) * NoL
    brdf = torch.nan_to_num(brdf, nan=0, posinf=0, neginf=0)
    pdf = torch.nan_to_num(pdf, nan=0, posinf=0, neginf=0)

    return brdf, pdf


class BaseBRDF(nn.Module):
    """ Base BRDF class """
    def __init__(self,):
        super(BaseBRDF,self).__init__()
        return

    def forward(self,):
        pass

    def eval_diffuse(self,wi,normal):
        """ evaluate diffuse shading
            and pdf
        """
        pdf = (normal*wi).sum(-1,keepdim=True).relu()/math.pi
        brdf = pdf.expand(len(wi),3)
        return brdf,pdf

    def sample_diffuse(self,sample2,normal):
        """ sample diffuse shading
            and get sampled weight
        """
        # get wi
        wi = diffuse_sampler(sample2,normal)

        # get brdf/pdf, pdf
        brdf_weight = torch.ones(normal.shape,device=normal.device)
        pdf = (normal*wi).sum(-1,keepdim=True).relu()/math.pi
        return wi,pdf,brdf_weight

    def eval_specular(self,wi,wo,normal,roughness):
        """" evaluate specular shadings
            and pdf
        """
        h = NF.normalize(wi+wo,dim=-1)
        NoL = (wi*normal).sum(-1,keepdim=True).relu()
        NoV = (wo*normal).sum(-1,keepdim=True).relu()
        VoH = (wo*h).sum(-1,keepdim=True).relu()
        NoH = (normal*h).sum(-1,keepdim=True).relu()

        D = D_GGX(NoH,roughness)
        pdf = D.data/(4*VoH.clamp_min(1e-4))*NoH

        G = G_Smith(NoV,NoL,roughness)
        F0,F1 = fresnelSchlick_sep(VoH)

        # two term corresponds to two fresnel components
        brdf_spec0 = D*G*F0/4.0*NoL
        brdf_spec1 = D*G*F1/4.0*NoL

        return brdf_spec0,brdf_spec1,pdf

    def sample_specular(self,sample2,wo,normal,roughness):
        """ evaluate specular shadings
            and get sampled weight
        """
        # get wi
        wi = specular_sampler(sample2,roughness,wo,normal)

        # get brdf/pdf, pdf
        h = NF.normalize(wi+wo,dim=-1)
        NoL = (wi*normal).sum(-1,keepdim=True).relu()
        NoV = (wo*normal).sum(-1,keepdim=True).relu()
        VoH = (wo*h).sum(-1,keepdim=True).relu()
        NoH = (normal*h).sum(-1,keepdim=True).relu()

        D = D_GGX(NoH,roughness)
        pdf = D.data/(4*VoH.clamp_min(1e-4))*NoH

        G = G_Smith(NoV,NoL,roughness)
        F0,F1 = fresnelSchlick_sep(VoH)

        fac = G*VoH*NoL/NoH.clamp_min(1e-4)

        brdf_weight0 = F0*fac
        brdf_weight1 = F1*fac
        return wi,pdf,brdf_weight0,brdf_weight1

    def eval_brdf(self,wi,wo,normal_geo,mat,screen_coor,use_mesh_normal,*args,**kwargs):
        """ evaluate BRDF and pdf
        Args:
            wi: Bx3 light direction
            wo: Bx3 viewing direction
            normal: Bx3 normal
            mat: surface BRDF dict
        Return:
            brdf: Bx3
            pdf: Bx1
        """
        albedo,roughness,metallic,normal = mat['albedo'],mat['roughness'],mat['metallic'],mat['normal']
        if 'bg' in mat.keys():
            bg = mat['bg']
        # screen_coor=kwargs['screen_coor']
        x,y = torch.floor(screen_coor[:,0]).long(),torch.floor(screen_coor[:,1]).long()
        roughness = roughness[x,y,:]
        albedo = albedo[x,y,:]
        metallic = metallic[x,y,:]
        normal = normal[x,y,:]
        if use_mesh_normal:
            normal = normal_geo
        if 'bg' in mat.keys():
            bg = bg[x,y,:]
        # albedo = albedo*bg*metallic

        h = NF.normalize(wi+wo,dim=-1) # half vector. mid point between wi and wo
        NoL = (wi*normal).sum(-1,keepdim=True).relu() #o means dot product
        NoV = (wo*normal).sum(-1,keepdim=True).relu()
        VoH = (wo*h).sum(-1,keepdim=True).relu()
        NoH = (normal*h).sum(-1,keepdim=True).relu()
        # breakpoint()
        # get r,a,m by screen coor
        if screen_coor.max() > 512:
            print('screen_coor max value should be lower than image width')
            breakpoint()


        # get pdf
        D = D_GGX(NoH,roughness)
        pdf_spec = D.data/(4*VoH.clamp_min(1e-4))*NoH
        pdf_diff = NoL/math.pi
        pdf = 0.5*pdf_spec + 0.5*pdf_diff

        # get brdf
        kd = albedo*(1-metallic)  # diffuse coefficient
        ks = 0.04*(1-metallic) + albedo*metallic # specular coefficient

        G = G_Smith(NoV,NoL,roughness)
        F = fresnelSchlick(VoH,ks)
        brdf_diff = kd/math.pi*NoL
        brdf_spec = D*G*F/4.0*NoL

        brdf = brdf_diff + brdf_spec

        # breakpoint()
        brdf = torch.nan_to_num(brdf,nan=0,posinf=0,neginf=0)
        pdf = torch.nan_to_num(pdf,nan=0,posinf=0,neginf=0)
        return brdf,pdf

    def sample_brdf(self,sample1,sample2,wo,normal_geo,mat,screen_coor,use_mesh_normal,*args,**kwargs):
        """ imp
        ortance sampling brdf and get brdf/pdf
        Args:
            sample1: B unifrom samples
            sample2: Bx2 uniform samples
            wo: Bx3 viewing direction
            normal: Bx3 normal
            mat: material dict
        Return:
            wi: Bx3 sampled direction
            pdf: Bx1
            brdf_weight: Bx3 brdf/pdf
        """
        B = sample2.shape[0]
        device = sample2.device

        pdf = torch.zeros(B,device=device)
        brdf = torch.zeros(B,3,device=device)
        wi = torch.zeros(B,3,device=device)
        # mask = (sample1 > 0.)
        # screen_coor=kwargs['screen_coor']

        albedo,roughness,metallic,normal = mat['albedo'],mat['roughness'],mat['metallic'],mat['normal']

        x,y = torch.floor(screen_coor[:,0]).long(),torch.floor(screen_coor[:,1]).long()
        roughness = roughness[x,y,:]
        normal = normal[x,y,:]
        if use_mesh_normal:
            normal = normal_geo
        if sample1 is None:
            sample1 = torch.rand(B,device=device)
        mask = (sample1 > 0.5)
        # sample diffuse
        wi[mask] = diffuse_sampler(sample2[mask],normal[mask])
        mask = ~mask
        # sample specular
        wi[mask] = specular_sampler(sample2[mask],roughness[mask],wo[mask],normal[mask])

        # get brdf,pdf
        brdf,pdf = self.eval_brdf(wi,wo,normal,mat,screen_coor=screen_coor,use_mesh_normal=use_mesh_normal)

        brdf_weight = torch.where(pdf>0,brdf/(pdf+1e-4),torch.tensor(0.0,device=device))
        brdf_weight[brdf_weight.isnan()] = 0

        return wi,pdf,brdf_weight

def perspective_projection_matrix_numpy(fov, aspect, near, far):
    """Generate perspective projection matrix
    fov is radians
    """
    f = 1.0 / np.tan(fov / 2.0)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]
    ])

def perspective_projection_matrix(fov, aspect, near, far):
    """Generate perspective projection matrix
    fov is radians
    """
    f = 1.0 / torch.tan(fov / 2.0)
    return torch.tensor([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]
    ], dtype=torch.float32)

def world_to_screen_numpy(world_coords, view_matrix, projection_matrix, screen_width, screen_height):
    """Convert batch world coordinates to screen coordinates"""
    # Expand world coordinates to 4D homogeneous coordinates
    ones = np.ones((world_coords.shape[0], 1))
    world_coords_h = np.hstack((world_coords, ones))

    # Convert world coordinates to camera coordinates
    camera_coords = np.dot(world_coords_h, view_matrix.T)

    # Convert camera coordinates to clip space coordinates
    clip_coords = np.dot(camera_coords, projection_matrix.T)

    # Perform perspective division to obtain normalized device coordinates (NDC)
    ndc_coords = clip_coords[:, :3] / (clip_coords[:, 3][:, np.newaxis])

    # Convert NDC coordinates to screen coordinates
    x_screen = (ndc_coords[:, 0] + 1) * 0.5 * screen_width
    # y_screen = (1 - ndc_coords[:, 1]) * 0.5 * screen_height # Flip y-axis, not sure why don't need to flip
    y_screen = (ndc_coords[:, 1]+1) * 0.5 * screen_height

    # return np.stack((x_screen, y_screen), axis=-1)
    return np.stack((y_screen, x_screen), axis=-1)

def world_to_screen(world_coords, view_matrix, projection_matrix, screen_width, screen_height):
    device = world_coords.device
    if view_matrix.device != device:
        view_matrix = view_matrix.to(device)
        projection_matrix = projection_matrix.to(device)
    """Convert batch world coordinates to screen coordinates"""
    # Expand world coordinates to 4D homogeneous coordinates
    ones = torch.ones((world_coords.shape[0], 1), device=device)
    world_coords_h = torch.cat((world_coords, ones), dim=1)

    # Convert world coordinates to camera coordinates
    camera_coords = torch.matmul(world_coords_h, view_matrix.T)

    # Convert camera coordinates to clip space coordinates
    clip_coords = torch.matmul(camera_coords, projection_matrix.T)

    # Perform perspective division to obtain normalized device coordinates (NDC)
    ndc_coords = clip_coords[:, :3] / (clip_coords[:, 3].unsqueeze(1)+1e-4)

    # Convert NDC coordinates to screen coordinates
    x_screen = (ndc_coords[:, 0] + 1) * 0.5 * screen_width
    y_screen = (ndc_coords[:, 1] + 1) * 0.5 * screen_height

    return torch.stack((y_screen, x_screen), dim=-1)

def mi_world_to_screen(world_coords, view_matrix, projection_matrix, screen_width, screen_height):
    """Convert batch world coordinates to screen coordinates using Mitsuba/Dr.Jit."""

    # Convert world coordinates to 4D homogeneous coordinates
    ones = mi.Float(1.0)
    world_coords_h = mi.Vector4f(world_coords[0], world_coords[1], world_coords[2], ones)

    # Convert world coordinates to camera coordinates
    camera_coords = view_matrix @ world_coords_h

    # Convert camera coordinates to clip space coordinates
    clip_coords = projection_matrix @ camera_coords

    # Perform perspective division to obtain normalized device coordinates (NDC)
    ndc_coords = mi.Vector3f(
        clip_coords[0] / clip_coords[3],
        clip_coords[1] / clip_coords[3],
        clip_coords[2] / clip_coords[3]
    )

    # Convert NDC coordinates to screen coordinates
    x_screen = (ndc_coords[0] + 1) * 0.5 * screen_width
    y_screen = (ndc_coords[1] + 1) * 0.5 * screen_height

    # Stack and return screen coordinates
    # return mi.Vector2f(y_screen, x_screen)
    return mi.Vector2f(x_screen, y_screen)

def comp_refract_dir(wo, normal, ior):
    """ Compute refraction direction using Snell's law
    Args:
        wo: Bx3 incident direction (viewing direction)
        normal: Bx3 surface normal
        ior: Bx1 or scalar, Index of Refraction of the material
    Return:
        wi_refract: Bx3 refracted direction
        valid: Bx1 boolean indicating valid refraction
    """
    cosi = torch.clamp(torch.sum(wo * normal, dim=-1), -1, 1)
    etai = torch.ones_like(cosi)
    etat = ior

    # Check if we're inside the material
    inside = cosi < 0
    etai[inside] = ior[inside] if ior.dim() > 0 else ior
    etat[inside] = 1.0
    cosi[inside] = -cosi[inside]

    eta = etai / etat
    k = 1 - eta ** 2 * (1 - cosi ** 2)

    wi_refract = eta[:, None] * wo + (eta[:, None]  * cosi[:, None] - torch.sqrt(k)[:, None]) * normal
    valid = k >= 0

    return wi_refract, valid

def load_estimated_brdf(root_dir):
    albedo_name = 'albedo.exr'
    roughness_name = 'roughness.exr'
    metallic_name = 'metallic.exr'
    # print('load brdf wo mitsuba optimization')
    normal_name = 'normal.exr'
    albedo = mi.Bitmap(os.path.join(root_dir, albedo_name))
    roughness = mi.Bitmap(os.path.join(root_dir, roughness_name))
    normal = mi.Bitmap(os.path.join(root_dir, normal_name))
    metallic = mi.Bitmap(os.path.join(root_dir, metallic_name))
    roughness = torch.tensor(np.array(roughness)).unsqueeze(-1).float().cuda()
    albedo = torch.tensor(np.array(albedo)).float().cuda()
    metallic = torch.tensor(np.array(metallic)).unsqueeze(-1).float().cuda()
    mat = {
        'albedo': albedo,
        'roughness': (roughness*0.95+0.05),
        'normal': torch.tensor(np.array(normal)).float().cuda(),
        'metallic': metallic,
    }
    if os.path.exists(os.path.join(root_dir, 'bg.png')):
        bg = plt.imread(os.path.join(root_dir, 'bg.png'))
        bg = torch.tensor(np.array(bg)).float().cuda()[...,:3]
        if bg.shape[0] != albedo.shape[0]:
            # breakpoint()
            warnings.warn('background size does not match albedo size, interpolate background')
            bg = NF.interpolate(bg[None].permute(0,3,1,2),size=albedo.shape[:2],mode='bilinear',align_corners=True)[0].permute(1,2,0)
        mat['bg'] = bg
        print('load background for transparency editing')
    if os.path.exists(os.path.join(root_dir, 'mask.png')):
        mask = plt.imread(os.path.join(root_dir, 'mask.png'))
        mask = torch.tensor(np.array(mask)).bool().cuda()[...,0]
        print('load mask for Material editing')
        mat['mask'] = mask
    if os.path.exists(os.path.join(root_dir, 'envmap.hdr')):
        envmap = mi.Bitmap(os.path.join(root_dir, 'envmap.hdr'))
        envmap = torch.tensor(np.array(envmap)).float().cuda()
        mat['envmap'] = envmap
    # breakpoint()
    return mat

# from myutils.misc import get_mat
class MatBSDF(mi.BSDF):
    def __init__(self, props):
        mi.BSDF.__init__(self, props)
        # self.m_flags = mi.BSDFFlags.DeltaReflection | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        self.m_flags = mi.BSDFFlags.SpatiallyVarying|mi.BSDFFlags.DiffuseReflection|mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        self.m_components = [self.m_flags]
        self.base_brdf = BaseBRDF()
        if props.has_property('use_mesh_normal'):
            self.use_mesh_normal = props['use_mesh_normal']
        else:
            self.use_mesh_normal = True

        if props.has_property('mat_dir'):
            try:
                self.mat = load_estimated_brdf(props['mat_dir'])
                print('load estimated brdf successfully')
            # root_dir = 'indoor_test_data/L3D124S8ENDIDSMXOAUI5L7GLUF3P3X4888/L3D124S8ENDIDSMXOAUI5L7GLUF3P3X4888'
            except:
                breakpoint()
                mat_dir,file_idx  = props['mat_dir'].split('|')
                # breakpoint()
                mat_file_name ={'albedo':f'{file_idx}_albedo','normal':f'{file_idx}_normal','material':f'{file_idx}_material','depth':f'{file_idx}_depth','gt':f'{file_idx}_im'}
                self.mat = get_mat(mat_dir,file_name=mat_file_name,resize=True)
        else:
            env = 0
            self.root_dir = os.path.join(global_config.BASE_DIR,'default_cam_env_render')
            albedo = mi.Bitmap(os.path.join(self.root_dir,f'albedo.exr'))
            # albedo = np.array(mi.Bitmap('/home/lewa/inverse_rendering/my_inverse_sss/teapot-full/wood.jpg'))/255
            roughness = mi.Bitmap(os.path.join(self.root_dir,f'roughness.exr'))
            normal = mi.Bitmap(os.path.join(self.root_dir,f'geonormal.exr'))
            depth = mi.Bitmap(os.path.join(self.root_dir,f'depth.exr'))
            metallic = mi.Bitmap(os.path.join(self.root_dir,f'metallic.exr'))
            rgb = mi.Bitmap(os.path.join(self.root_dir,f'rgb_{env}.exr'))
            bg = mi.Bitmap(os.path.join(self.root_dir,f'bg.exr'))

            self.mat = {
                'albedo': torch.tensor(np.array(albedo)).float().cuda(),
                'roughness': torch.tensor(np.array(roughness))[:,:,:1].float().cuda(),
                'normal': torch.tensor(np.array(normal)).float().cuda(),
                'depth': torch.tensor(np.array(depth)).float().cuda(),
                'metallic': torch.tensor(np.array(metallic))[:,:,:1].float().cuda(),
                'rgb': torch.tensor(np.array(rgb)).float().cuda(),
                'bg': torch.tensor(np.array(bg)).float().cuda(),
                'mask': torch.tensor(np.array(metallic)>0.01)[:,:,0].cuda(),
            }

        if props.has_property('cam_meta'):
            self.cam_meta = json.load(open(props['cam_meta']))
        else:
            self.cam_meta = json.load(open(os.path.join(global_config.RESOURCE_DIR, "camera.json")))
        to_world = torch.tensor(self.cam_meta['to_world'])[0]
        self.view_matrix = torch.inverse(to_world)

        self.width, self.height = self.cam_meta['film.size']

        fov = torch.deg2rad(torch.tensor(self.cam_meta['x_fov'][0]))
        self.focal = 0.5*self.width/torch.tan(0.5*fov)
        self.R = to_world[:3,:3]
        nearclip = self.cam_meta['near_clip']
        farclip = self.cam_meta['far_clip']


        self.persp_proj_matx = perspective_projection_matrix(fov, self.width/self.height, nearclip, farclip)

    def sample(self, ctx, si, sample1, sample2, active):
        # cos_theta = mi.Frame3f.cos_theta(si.wi)
        # active &= cos_theta > 0
        wi = si.to_world(si.wi).torch()
        normal = si.n.torch()
        position = si.p.torch()
        # triangle_idx = mi.Int(si.prim_index).torch().long()
        refract = False
        # pos_world_np = si.p.numpy()
        if refract: # refract
            ior = torch.ones(len(normal),device=wi.device)*1.3
            wi_refract,refract_valid = comp_refract_dir(wi,normal,ior)
            # breakpoint()
            if refract_valid.any():
                wi[refract_valid] = wi_refract[refract_valid]

        screen_coor = world_to_screen(position,self.view_matrix,self.persp_proj_matx,self.width,self.height)

        wo,pdf,brdf_weight = self.base_brdf.sample_brdf(
        # wo,pdf,brdf_weight = sample_brdf(
            sample1.torch().reshape(-1),sample2.torch(),
            wi,normal,self.mat,screen_coor=screen_coor,
            use_mesh_normal=self.use_mesh_normal
        )
        

        pdf_mi = mi.Float(pdf.squeeze(-1))
        wo_mi = mi.Vector3f(wo[...,0],wo[...,1],wo[...,2])
        wo_mi = si.to_local(wo_mi)
        value_mi = mi.Vector3f(brdf_weight[...,0],brdf_weight[...,1],brdf_weight[...,2])

        bs = mi.BSDFSample3f()
        bs.pdf = pdf_mi
        bs.sampled_component = mi.UInt32(0)
        bs.sampled_type = mi.UInt32(+self.m_flags)
        bs.wo = wo_mi
        bs.eta = 1.0
        pos_local = si.to_local(si.p).torch()

        return (bs,value_mi)


    def to_string(self):
        return 'MatBSDF'

    def eval(self, ctx, si, wo, active):
        wo = si.to_world(wo).torch()
        wi = si.to_world(si.wi).torch()

        normal = si.n.torch()
        position = si.p.torch()

        screen_coor = world_to_screen(position,self.view_matrix,self.persp_proj_matx,self.width,self.height)
        brdf,_ = self.base_brdf.eval_brdf(wo,wi,normal,self.mat,screen_coor,self.use_mesh_normal)
        brdf = mi.Vector3f(brdf[...,0],brdf[...,1],brdf[...,2])
        return brdf

    def pdf(self, ctx, si, wo,active):
        wo = si.to_world(wo).torch()
        wi = si.to_world(si.wi).torch()

        normal = si.n.torch()
        position = si.p.torch()
        screen_coor = world_to_screen(position,self.view_matrix,self.persp_proj_matx,self.width,self.height)
        _,pdf = self.base_brdf.eval_brdf(wo,wi,normal,self.mat,screen_coor,self.use_mesh_normal)
        pdf = mi.Float(pdf.squeeze(-1))
        return pdf

    def eval_pdf(self, ctx, si, wo, active=True):
        wo = si.to_world(wo).torch()
        wi = si.to_world(si.wi).torch()

        normal = si.n.torch()
        position = si.p.torch()
        screen_coor = world_to_screen(position,self.view_matrix,self.persp_proj_matx,self.width,self.height)
        brdf,pdf = self.base_brdf.eval_brdf(wo,wi,normal,self.mat,screen_coor=screen_coor,use_mesh_normal=self.use_mesh_normal)
        brdf = mi.Vector3f(brdf[...,0],brdf[...,1],brdf[...,2])
        pdf = mi.Float(pdf.squeeze(-1))
        # breakpoint()
        return brdf,pdf

    def traverse(self, callback):
        callback.put_parameter('cam_meta', self.cam_meta, mi.ParamFlags.NonDifferentiable)
        callback.put_parameter('mat', self.mat, mi.ParamFlags.NonDifferentiable)
        
class RefractBaseBRDF(BaseBRDF):
    def __init__(self,ior = 1., specTrans = 0.5, use_mesh_normal = True,keep_albedo_color=False):
        super(RefractBaseBRDF, self).__init__()
        self.ior = ior
        self.specTrans = specTrans # raito of light transmitted
        self.use_mesh_normal = use_mesh_normal
        self.keep_albedo_color = keep_albedo_color

    def sample_brdf(self, sample1, sample2, wo, normal_mesh, mat, screen_coor,refracted_screen_coor , *args, **kwargs):
        """ imp
        ortance sampling brdf and get brdf/pdf
        Args:
            sample1: B unifrom samples
            sample2: Bx2 uniform samples
            wo: Bx3 viewing direction
            normal: Bx3 normal
            mat: material dict
        Return:
            wi: Bx3 sampled direction
            pdf: Bx1
            brdf_weight: Bx3 brdf/pdf
        """
        B = wo.shape[0]
        try:
            device = normal.device
        except:
            device = torch.device('cuda')

        pdf = torch.zeros(B, device=device)
        brdf = torch.zeros(B, 3, device=device)
        wi = torch.zeros(B, 3, device=device)
        # mask = (sample1 > 0.)
        # screen_coor=kwargs['screen_coor']

        albedo, roughness, metallic = mat['albedo'], mat['roughness'], mat['metallic']
        normal = mat['normal']
        mask = mat['mask'].bool()
        roughness[mask] = 0.1

        x, y = torch.floor(screen_coor[:, 0]).long(), torch.floor(screen_coor[:, 1]).long()
        roughness = roughness[x, y, :]
        if self.use_mesh_normal:
            normal = normal_mesh
        else:
            normal = normal[x, y, :]
        if sample1 is None:
            sample1 = torch.rand(B, device=device)

        mask_diffuse = (sample1 > 0.5)
        mask_specular = (sample1 <= 0.5)

        # sample diffuse
        wi[mask_diffuse] = diffuse_sampler(sample2[mask_diffuse], normal[mask_diffuse])

        # sample specular
        wi[mask_specular] = specular_sampler(sample2[mask_specular], roughness[mask_specular], wo[mask_specular], normal[mask_specular])
        bsdf, pdf = self.eval_brdf(wi, wo, normal, mat, screen_coor, refracted_screen_coor)

        bsdf_weight = torch.where(pdf > 0, bsdf / pdf, torch.tensor(0.0, device=device))
        bsdf_weight[bsdf_weight.isnan()] = 0

        return wi, pdf, bsdf_weight

    def eval_brdf(self,wi,wo,normal_mesh,mat,screen_coor,refracted_screen_coor,*args,**kwargs):
        """ evaluate BRDF and pdf
        Args:
            wi: Bx3 light direction
            wo: Bx3 viewing direction
            normal: Bx3 normal
            mat: surface BRDF dict
        Return:
            brdf: Bx3
            pdf: Bx1
        """
        albedo,roughness,metallic,normal = mat['albedo'],mat['roughness'],mat['metallic'],mat['normal']
        if albedo.max() > 1:
            warnings.warn(f'albedo should be in [0,1], but now is {albedo}', stacklevel=2)
            albedo = albedo.clamp(0,1)
        
        bg = mat['bg']
        mask = mat['mask'].bool()
        # breakpoint()
        metallic[mask] = 0
        roughness[mask] = 0.1
        if not self.keep_albedo_color:
            albedo[mask] = 0.9
        x,y = torch.floor(screen_coor[:,0]).long(),torch.floor(screen_coor[:,1]).long()

        mask = mask[x,y]
        refracted_screen_coor[~mask] = screen_coor[~mask]
        x_refract,y_refract = torch.floor(refracted_screen_coor[:,0]).long(),torch.floor(refracted_screen_coor[:,1]).long()

        roughness = roughness[x,y,:]

        albedo = albedo[x,y,:]
        metallic = metallic[x,y,:] * 0
        if self.use_mesh_normal:
            normal = normal_mesh
        else:
            normal = normal[x, y, :]

        bg = bg[x_refract,y_refract,:]


        h = NF.normalize(wi+wo,dim=-1) # half vector. mid point between wi and wo
        NoL = (wi*normal).sum(-1,keepdim=True).relu() #o means dot product， Light, w_in
        NoV = (wo*normal).sum(-1,keepdim=True).relu() # View
        VoH = (wo*h).sum(-1,keepdim=True).relu()
        NoH = (normal*h).sum(-1,keepdim=True).relu()
        LoH = (wi*h).sum(-1,keepdim=True).relu()
        # breakpoint()

        if screen_coor.max() > 512 or refracted_screen_coor.max() > 512:
            print('screen_coor max value should be lower than image width')
            breakpoint()

        # Calculate pdf for reflection and transmission
        D = D_GGX(NoH, roughness)
        pdf_spec = D.data / (4 * VoH.clamp_min(1e-4)) * NoH
        pdf_diff = NoL / math.pi
        pdf_trans = D.data / (VoH.clamp_min(1e-4)) * NoL if self.specTrans > 0 else 0
        pdf = 0.5*pdf_spec + 0.5*pdf_diff

        # original BRDF for scene
        kd = albedo*(1-metallic)  
        ks = 0.04*(1-metallic) + albedo*metallic
        G = G_Smith(NoV,NoL,roughness)
        F = fresnelSchlick(VoH,ks)
        brdf_diff = kd/math.pi*NoL
        brdf_spec = D*G*F/4.0*NoL
        brdf_ori = brdf_diff + brdf_spec

        # Reflective BRDF for material editing
        kd =  albedo * (1 - metallic) * (1 - self.specTrans)   
        baseColor_m = (1-self.specTrans*(1-metallic)) * albedo * metallic  
        baseColor_glass = (1-metallic) * (bg * self.specTrans) * 0.7

        F_m = fresnelSchlick(VoH, baseColor_m)

        brdf_diff = kd / math.pi * NoL
        brdf_metal = D * G * F_m / 4.0 * NoL

        hw_in = 1 / (LoH + 1e-6)
        hw_out = 1 / (VoH + 1e-6)
        nw_in = 1 / (NoL + 1e-6)
        R_s = (hw_in - self.ior * hw_out) / (hw_in + self.ior * hw_out)
        R_p = (self.ior * hw_in - hw_out) / (self.ior * hw_in + hw_out)
        F_glass = 0.5 * (R_s**2 + R_p**2) # ior up, F_glass up
        D_hacking = D_GGX(NoH, roughness*0+1) 
        # try to make the rays not specular reflect when transmit by hacking the roughness
        btdf_glass =  (baseColor_glass)**0.5 * G * D_hacking * (1-F_glass) * (hw_out * hw_in)/(nw_in*(hw_in+self.ior*hw_out)**2)
        brdf_spec_edit = baseColor_glass * D * G / (4 * nw_in)

        bsdf_edit = brdf_diff + brdf_metal + btdf_glass + brdf_spec_edit

        bsdf_edit[~mask] = 0
        brdf_ori[mask] = 0
        bsdf = bsdf_edit + brdf_ori

        bsdf = torch.nan_to_num(bsdf, nan=0, posinf=0, neginf=0)
        pdf = torch.nan_to_num(pdf, nan=0, posinf=0, neginf=0)
        return bsdf, pdf

class MatrefractBSDF(MatBSDF):
    def __init__(self, props):
        MatBSDF.__init__(self, props)
        if props.has_property('ior'):
            self.ior = props['ior']
        else:
            self.ior = 1.3
        if props.has_property('keep_albedo_color'):
            self.keep_albedo_color = props['keep_albedo_color']
            self.refract_distance = 1.0 * 100 # scale factor for real scene
        else:
            self.keep_albedo_color = False
            self.refract_distance = 1.0 

        self.specTrans = 0.8
        self.base_brdf = RefractBaseBRDF(ior = self.ior, use_mesh_normal = self.use_mesh_normal,specTrans=self.specTrans,keep_albedo_color=self.keep_albedo_color)


    def calculate_refraction(self, wi, normal, ior_ratio):
        # Compute refraction direction using Snell's law
        cos_theta_i = (wi * normal).sum(-1, keepdim=True)
        sin2_theta_i = torch.max(torch.tensor(0.0, device=wi.device), 1.0 - cos_theta_i**2)
        sin2_theta_t = ior_ratio**2 * sin2_theta_i
        cos_theta_t = torch.sqrt(1.0 - sin2_theta_t)

        # refracted_dir = ior_ratio * wi + (ior_ratio * cos_theta_i - cos_theta_t) * normal
        refracted_dir = ior_ratio * (normal * cos_theta_i - wi) - normal * cos_theta_t
        return NF.normalize(refracted_dir, dim=-1)

    def calculate_refracted_screen_coor(self, wi, normal, ior_ratio, position, screen_coor):
        ior_ratio = 1.0 / self.ior
        #1st refract
        refracted_dir1 = self.calculate_refraction(wi, normal, ior_ratio)
        x, y = torch.floor(screen_coor[:, 0]).long(), torch.floor(screen_coor[:, 1]).long()
        # refract_distance1 = self.path_length[x, y, 2:3]
        refracted_position1 = position + 0.3 * self.refract_distance * refracted_dir1
        refracted_screen_coor1 = world_to_screen(refracted_position1, self.view_matrix, self.persp_proj_matx, self.width, self.height)
        refracted_screen_coor1 = torch.clamp(refracted_screen_coor1,0,self.width-1)
        #2nd refract
        # x,y = torch.floor(refracted_screen_coor1[:, 0]).long(), torch.floor(refracted_screen_coor1[:, 1]).long()
        # refract_distance2 = self.path_length[x, y, 1:2]
        # normal_mirrored = normal.clone()
        # normal_mirrored[...,1] = -normal[..., 1]

        # assume the normal is the same as the first refracted normal
        refracted_dir2 = self.calculate_refraction(-refracted_dir1, normal, 1.0/ior_ratio)
        refracted_position2 = refracted_position1 + self.refract_distance * refracted_dir2
        refracted_screen_coor2 = world_to_screen(refracted_position2, self.view_matrix, self.persp_proj_matx, self.width, self.height)
        refracted_screen_coor2 = torch.clamp(refracted_screen_coor2,0,self.width-1)
        refracted_screen_coor2 = torch.nan_to_num(refracted_screen_coor2,nan=0,posinf=0,neginf=0)
        return refracted_screen_coor2

    def sample(self, ctx, si, sample1, sample2, active):
        wi = si.to_world(si.wi).torch()
        normal_mesh = si.n.torch()
        position = si.p.torch()

        screen_coor = world_to_screen(position, self.view_matrix, self.persp_proj_matx, self.width, self.height)
        if self.use_mesh_normal:
            normal = normal_mesh
        else:
            normal = self.mat['normal']
            x, y = torch.floor(screen_coor[:, 0]).long(), torch.floor(screen_coor[:, 1]).long()
            normal = normal[x, y, :]
        # Calculate refraction direction
        refracted_screen_coor = self.calculate_refracted_screen_coor(wi, normal, 1.0 / self.ior, position, screen_coor)
        # breakpoint()
        wo, pdf, brdf_weight = self.base_brdf.sample_brdf(
            sample1.torch().reshape(-1), sample2.torch(),
            wi, normal, self.mat, screen_coor=screen_coor,refracted_screen_coor=refracted_screen_coor,
        )

        pdf_mi = mi.Float(pdf.squeeze(-1))
        wo_mi = mi.Vector3f(wo[..., 0], wo[..., 1], wo[..., 2])
        # wo_mi = si.to_local(wo_mi)
        value_mi = mi.Vector3f(brdf_weight[..., 0], brdf_weight[..., 1], brdf_weight[..., 2])

        bs = mi.BSDFSample3f()
        bs.pdf = pdf_mi
        bs.sampled_component = mi.UInt32(0)
        bs.sampled_type = mi.UInt32(+self.m_flags)
        bs.wo = wo_mi
        bs.eta = self.ior

        return (bs, value_mi)

    def eval(self, ctx, si, wo, active):
        wo = si.to_world(wo).torch()
        wi = si.to_world(si.wi).torch()
        normal_mesh = si.n.torch()
        position = si.p.torch()

        # Calculate the position after refraction
        refracted_screen_coor = self.calculate_refracted_screen_coor(wi, normal, 1.0 / self.ior, position, screen_coor)
        screen_coor = world_to_screen(position, self.view_matrix, self.persp_proj_matx, self.width, self.height)
        if self.use_mesh_normal:
            normal = normal_mesh
        else:
            normal = self.mat['normal']
            x, y = torch.floor(screen_coor[:, 0]).long(), torch.floor(screen_coor[:, 1]).long()
            normal = normal[x, y, :]
        brdf, _ = self.base_brdf.eval_brdf(wo, wi, normal, self.mat, screen_coor,refracted_screen_coor)
        brdf = mi.Vector3f(brdf[..., 0], brdf[..., 1], brdf[..., 2])

        return brdf

    def pdf(self, ctx, si, wo, active):
        wo = si.to_world(wo).torch()
        wi = si.to_world(si.wi).torch()
        normal_mesh = si.n.torch()
        position = si.p.torch()

        screen_coor = world_to_screen(position, self.view_matrix, self.persp_proj_matx, self.width, self.height)

        if self.use_mesh_normal:
            normal = normal_mesh
        else:
            normal = self.mat['normal']
            x, y = torch.floor(screen_coor[:, 0]).long(), torch.floor(screen_coor[:, 1]).long()
            normal = normal[x, y, :]

        refracted_screen_coor = self.calculate_refracted_screen_coor(wi, normal, 1.0 / self.ior, position, screen_coor)

        _, pdf = self.base_brdf.eval_brdf(wo,wi,normal,self.mat,screen_coor=screen_coor,refracted_screen_coor=refracted_screen_coor)
        pdf = mi.Float(pdf.squeeze(-1))

        return pdf

    def eval_pdf(self, ctx, si, wo, active=True):
        wo = si.to_world(wo).torch()
        wi = si.to_world(si.wi).torch()
        # triangle_idx = mi.Int(si.prim_index).torch().long()

        normal_mesh = si.n.torch()
        position = si.p.torch()

        screen_coor = world_to_screen(position, self.view_matrix, self.persp_proj_matx, self.width, self.height)

        if self.use_mesh_normal:
            normal = normal_mesh
        else:
            normal = self.mat['normal']
            x, y = torch.floor(screen_coor[:, 0]).long(), torch.floor(screen_coor[:, 1]).long()
            normal = normal[x, y, :]
        refracted_screen_coor = self.calculate_refracted_screen_coor(wi, normal, 1.0 / self.ior, position, screen_coor)

        # mat = self.material_net(position)
        brdf,pdf = self.base_brdf.eval_brdf(wo,wi,normal,self.mat,screen_coor=screen_coor,refracted_screen_coor=refracted_screen_coor)
        # brdf[self.is_emitter[triangle_idx]] = 0
        brdf = mi.Vector3f(brdf[...,0],brdf[...,1],brdf[...,2])
        pdf = mi.Float(pdf.squeeze(-1))
        # breakpoint()
        return brdf,pdf


def to_mitsuba(tensor):
    """Automatically convert a PyTorch tensor to the appropriate Mitsuba type."""
    if tensor.dtype == torch.float32:
        if tensor.ndim == 1 and tensor.shape[0] == 3:
            # Convert 1D tensor with 3 elements to mi.Vector3f
            return mi.Vector3f(tensor[0].item(), tensor[1].item(), tensor[2].item())
        elif tensor.ndim == 2 and tensor.shape == (4, 4):
            # Convert 2D tensor with shape 4x4 to mi.Matrix4f
            return mi.Matrix4f(tensor.numpy().astype(np.float32))
        else:
            # Default conversion for float32 tensors to mi.Float
            return mi.TensorXf(tensor.numpy().astype(np.float32))
    elif tensor.dtype == torch.int32:
        return mi.Int(tensor.numpy().astype(np.int32))
    elif tensor.dtype == torch.bool:
        return mi.Bool(tensor.numpy().astype(np.bool_))
    else:
        raise TypeError(f"Unsupported tensor dtype: {tensor.dtype}")


class MatDiffBSDF(mi.BSDF):
    def __init__(self, props):
        mi.BSDF.__init__(self, props)
        self.m_flags = mi.BSDFFlags.SpatiallyVarying|mi.BSDFFlags.DiffuseReflection|mi.BSDFFlags.FrontSide #| mi.BSDFFlags.BackSide
        self.m_components = [self.m_flags]
        if props.has_property('use_mesh_normal'):
            self.use_mesh_normal = props['use_mesh_normal']
        else:
            self.use_mesh_normal = True
        self.a = mi.TensorXf(0.5,shape=(512,512,3))
        self.r = mi.TensorXf(0.5,shape=(512,512,1))
        self.m = mi.TensorXf(0.5,shape=(512,512,1))
        self.n = mi.TensorXf(0.5,shape=(512,512,3))

        if props.has_property('mat_dir'):
            self.mat = load_estimated_brdf(props['mat_dir'])
            # pred_mat_dir = props['pred_mat_dir']
            # self.mat = self.load_pred_brdf(pred_mat_dir)
            keys = self.mat.keys()
            for key in keys:
                if key != 'mask':
                    self.mat[key] = mi.TensorXf(self.mat[key])
            self.a = mi.TensorXf(self.mat['albedo'])
            self.r = mi.TensorXf(self.mat['roughness'])
            self.m = mi.TensorXf(self.mat['metallic'])
            print('Load optimized BRDF from:',props['mat_dir'])
        else:
            self.mat = {}
            print('No BRDF loaded, need to update BRDF params by mi.traverse')

        if props.has_property('cam_meta'):
            self.cam_meta = json.load(open(props['cam_meta']))
        else:
            self.cam_meta = json.load(open(os.path.join(global_config.RESOURCE_DIR, "camera.json")))
        to_world = torch.tensor(self.cam_meta['to_world'])[0]
        self.view_matrix = torch.inverse(to_world)

        self.width, self.height = self.cam_meta['film.size']

        fov = torch.deg2rad(torch.tensor(self.cam_meta['x_fov'][0]))
        self.focal = 0.5*self.width/torch.tan(0.5*fov)
        self.R = to_world[:3,:3]
        nearclip = self.cam_meta['near_clip']
        farclip = self.cam_meta['far_clip']
        self.persp_proj_matx = perspective_projection_matrix(fov, self.width/self.height, nearclip, farclip)
        self.persp_proj_matx = mi.Matrix4f(self.persp_proj_matx.numpy())
        self.view_matrix = mi.Matrix4f(self.view_matrix.numpy())

    def load_pred_brdf(self,pred_mat_dir):

        albedo = plt.imread(os.path.join(pred_mat_dir,'albedoPred.png'))
        normal = np.array(mi.Bitmap(os.path.join(pred_mat_dir,'normalPred.exr')))
        roughness = plt.imread(os.path.join(pred_mat_dir,'roughnessPred.png'))
        metallic = plt.imread(os.path.join(pred_mat_dir,'metallicPred.png'))
        depth = np.array(mi.Bitmap(os.path.join(pred_mat_dir,'depthPred.exr')))
        gt_image = np.array(mi.Bitmap(os.path.join(pred_mat_dir,'gt_image.exr')))

        mat = {
            'albedo': torch.from_numpy(albedo).float().cuda(),
            'normal': torch.from_numpy(normal).float().cuda(),
            'roughness': torch.from_numpy(roughness).float().cuda(),
            'metallic': torch.from_numpy(metallic).float().cuda(),
            'depth': torch.from_numpy(depth).float().cuda(),
            'gt_image': torch.from_numpy(gt_image).float().cuda()
        }

        return mat
    def sample_brdf(self, sample1, sample2, wo, normal_mesh, mat, screen_coor):
        """ Importance sampling BRDF and get BRDF/PDF """
        # B = sample2.shape[0]
        B = len(sample1)

        # pdf = dr.zeros(B)
        brdf = mi.Vector3f(0.0)
        wi = mi.Vector3f(0.0)

        roughness = self.r

        x = mi.Int(dr.floor(screen_coor[0]))
        y = mi.Int(dr.floor(screen_coor[1]))

        flat_index = x + y * roughness.shape[0]
        roughness = dr.gather(mi.Float, roughness.array, flat_index)
        if self.use_mesh_normal:
            normal = normal_mesh
        else:
            normal = dr.gather(mi.Normal3f, self.n.array, flat_index)

        normal_offset = False
        if normal_offset:
            max_angle = 0.3  # Maximum allowed angular deviation in radians
            dot_nw = dr.dot(normal, wo)
            metallic = dr.gather(mi.Float, self.m.array, flat_index)
            normal = dr.select((dot_nw < dr.cos(max_angle))&(roughness<0.3)&(metallic>0.7), dr.normalize(0.5 * normal + 0.5 * wo), normal)
            # normal = dr.select(roughness <= 0.1, dr.normalize(normal*0.5+0.5*wo), normal)
        
        if sample1 is None:
            # sample1 = dr.random<Float>(B)
            breakpoint()

        mask = (sample1 > 0.5)
        wi[mask] = mi_diffuse_sampler(sample2[mask], normal[mask])
        mask = ~mask
        wi[mask] = mi_specular_sampler(sample2[mask], roughness[mask], wo[mask], normal[mask])

        # get brdf,pdf
        brdf, pdf = self.eval_brdf(wi, wo, normal, mat, screen_coor)

        # brdf_weight = dr.select(pdf > 1e-6, brdf / pdf, mi.Vector3f(0.0))
        brdf_weight = brdf / (pdf+1e-6)
        brdf_weight = dr.select(pdf > 1e-6, brdf_weight, mi.Vector3f(0.0))
        pdf = dr.select(pdf > 0, pdf, mi.Float(0.0))
        return wi, pdf, brdf_weight

    def index_to_2d(self,index, width):
        """Convert 1D index back to 2D (x, y) coordinates."""
        x = index % width
        y = index // width
        return x, y
    
    def eval(self, ctx, si, wo, active):
        wo = si.to_world(wo)
        wi = si.to_world(si.wi)
        normal = si.n
        position = si.p
        screen_coor = mi_world_to_screen(position,self.view_matrix,self.persp_proj_matx,self.width,self.height)
        brdf,_ = self.eval_brdf(wi, wo, normal, self.mat, screen_coor)

        return brdf

    def pdf(self, ctx, si, wo,active):
        wo = si.to_world(wo)
        wi = si.to_world(si.wi)

        normal = si.n
        position = si.p

        screen_coor = mi_world_to_screen(position,self.view_matrix,self.persp_proj_matx,self.width,self.height)

        _,pdf = self.eval_brdf(wi, wo, normal, self.mat, screen_coor)

        return pdf

    def eval_brdf(self, wi, wo, normal_mesh, mat, screen_coor):
        """ Evaluate BRDF and PDF """
        albedo = self.a
        roughness = self.r
        metallic = self.m
        # get r,a,m by screen coor
        x = mi.Int(dr.floor(screen_coor[0]))
        y = mi.Int(dr.floor(screen_coor[1]))

        flat_index = x + y * roughness.shape[0]

        albedo = dr.gather(mi.Vector3f, albedo.array, flat_index)
        roughness = dr.gather(mi.Float, roughness.array, flat_index)
        metallic = dr.gather(mi.Float, metallic.array, flat_index)
        if self.use_mesh_normal:
            normal = normal_mesh
        else:
            normal = dr.gather(mi.Normal3f, self.n.array, flat_index)

        h = dr.normalize(wi + wo)
        NoL = dr.maximum(dr.dot(normal, wi), 0)
        NoV = dr.maximum(dr.dot(normal, wo), 0)
        VoH = dr.maximum(dr.dot(wo, h), 0)
        NoH = dr.maximum(dr.dot(normal, h), 0)

        # get pdf
        D = D_GGX(NoH, roughness)
        pdf_spec = D / (4 * dr.maximum(VoH, 1e-6)) * NoH
        pdf_diff = NoL / math.pi
        pdf = 0.5 * pdf_spec + 0.5 * pdf_diff

        disney_brdf = True
        if disney_brdf:
            baseColor_d = albedo * (1 - metallic)
            F_D90 = 0.5 + 2 * VoH ** 2 * roughness
            F_D_w_out = 1 + (F_D90 - 1) * (1 - NoV) ** 5
            F_D_w_in = 1 + (F_D90 - 1) * (1 - NoL) ** 5
            brdf_diff = baseColor_d / math.pi * F_D_w_out * F_D_w_in * NoL

            G = G_Smith(NoV, NoL, roughness)
            C_0 = (1 - metallic) * 0.04 + metallic * albedo
            F_m = C_0 + (1 - C_0) * (1 - VoH) ** 5
            brdf_metal = D * G  * F_m / 4 * NoL
            brdf = brdf_diff + brdf_metal
        else:
            # get brdf
            kd = albedo * (1 - metallic) 
            # kd = albedo * metallic
            ks = 0.04 * (1 - metallic) + albedo * metallic 

            G = G_Smith(NoV, NoL, roughness)
            F = fresnelSchlick(VoH, ks)
            brdf_diff = kd / math.pi * NoL
            brdf_spec = D * G * F / 4.0 * NoL
            brdf = brdf_diff + brdf_spec
        return brdf, pdf

    def sample(self, ctx, si, sample1, sample2, active):

        wi = si.to_world(si.wi)
        normal = si.n
        position = si.p

        screen_coor = mi_world_to_screen(position,self.view_matrix,self.persp_proj_matx,self.width,self.height)

        wo,pdf,brdf_weight = self.sample_brdf(sample1,sample2,wi,normal,self.mat,screen_coor)
        pdf_mi = mi.Float(pdf)
        value_mi = mi.Vector3f(brdf_weight)
        bs = mi.BSDFSample3f()
        bs.pdf = pdf_mi
        bs.sampled_component = mi.UInt32(0)
        bs.sampled_type = mi.UInt32(+self.m_flags)
        bs.wo = wo
        bs.eta = 1.0
        return (bs,value_mi)


    def eval_pdf(self, ctx, si, wo, active=True):
        wo = si.to_world(wo)
        wi = si.to_world(si.wi)

        normal = si.n
        position = si.p

        screen_coor = mi_world_to_screen(position,self.view_matrix,self.persp_proj_matx,self.width,self.height)

        brdf,pdf = self.eval_brdf(wo,wi,normal,self.mat,screen_coor)

        return brdf,pdf

    def to_string(self):
        return 'MatDiffBSDF'
    def traverse(self, callback):
        callback.put_parameter('a', self.a, mi.ParamFlags.Differentiable)
        callback.put_parameter('r', self.r, mi.ParamFlags.Differentiable)
        callback.put_parameter('m', self.m, mi.ParamFlags.Differentiable)
        callback.put_parameter('n', self.n, mi.ParamFlags.Differentiable)
        callback.put_parameter('use_mesh_normal', self.use_mesh_normal, mi.ParamFlags.NonDifferentiable)

    def parameters_changed(self, keys):
        # print(self.mat_nn)
        # print(dr.grad_enabled(self.mat_nn))
        # breakpoint()
        pass

class TransBSDF(MatDiffBSDF):
    def __init__(self, props):
        MatDiffBSDF.__init__(self, props)
        if props.has_property('ior'):
            self.ior = props['ior']
        else:
            self.ior = 1.3
        if props.has_property('keep_albedo_color'):
            self.keep_albedo_color = props['keep_albedo_color']
            self.refract_distance = 1.0 * 100 # scale factor for real scene
        else:
            self.keep_albedo_color = False
            self.refract_distance = 1.0 
        self.specTrans = mi.Float(0.8)
        self.bg = mi.TensorXf(0.5,shape=(512,512,3))
        self.mask = mi.TensorXf(0,shape=(512,512)) > 0

    def calculate_refraction(self, wi, normal, ior_ratio):
        cos_theta_i = dr.dot(wi, normal)
        sin2_theta_i = dr.maximum(0.0, 1.0 - cos_theta_i**2)
        sin2_theta_t = ior_ratio**2 * sin2_theta_i
        cos_theta_t = dr.safe_sqrt(1.0 - sin2_theta_t)
        refracted_dir = ior_ratio * (normal * cos_theta_i - wi) - normal * cos_theta_t
        refracted_dir = dr.normalize(refracted_dir)
        return refracted_dir
    
    def calculate_refracted_screen_coor(self, wi, normal, ior_ratio, position, screen_coor):
        ior_ratio = 1.0 / ior_ratio
        # breakpoint()
        #1st refract
        refracted_dir1 = self.calculate_refraction(wi, normal, ior_ratio)
        refracted_position1 = position + 0.3 * self.refract_distance * refracted_dir1
        refracted_screen_coor1 = mi_world_to_screen(refracted_position1, self.view_matrix, self.persp_proj_matx, self.width, self.height)
        refracted_screen_coor1 = dr.clamp(refracted_screen_coor1,0,self.width-1)
        #2nd refract

        # assume the normal is the same as the first refracted normal
        refracted_dir2 = self.calculate_refraction(-refracted_dir1, normal, 1.0/ior_ratio)
        refracted_position2 = refracted_position1 + self.refract_distance * refracted_dir2
        refracted_screen_coor2 = mi_world_to_screen(refracted_position2, self.view_matrix, self.persp_proj_matx, self.width, self.height)
        refracted_screen_coor2 = dr.clamp(refracted_screen_coor2,0,self.width-1)
        refracted_screen_coor2 = dr.select(refracted_screen_coor2>0,refracted_screen_coor2,0)
        return refracted_screen_coor2
    
    def sample(self, ctx, si, sample1, sample2, active):
        wi = si.to_world(si.wi)
        normal = si.n
        position = si.p

        screen_coor = mi_world_to_screen(position, self.view_matrix, self.persp_proj_matx, self.width, self.height)
        # Calculate refraction direction
        refracted_screen_coor = self.calculate_refracted_screen_coor(wi, normal, 1.0 / self.ior, position, screen_coor)
        # breakpoint()
        wo, pdf, brdf_weight = self.sample_brdf(
            sample1, sample2,
            wi, normal, self.mat, screen_coor=screen_coor,refracted_screen_coor=refracted_screen_coor,
        )

        pdf_mi = mi.Float(pdf)
        value_mi = mi.Vector3f(brdf_weight)
        bs = mi.BSDFSample3f()
        bs.pdf = pdf_mi
        bs.sampled_component = mi.UInt32(0)
        bs.sampled_type = mi.UInt32(+self.m_flags)
        bs.wo = wo
        bs.eta = self.ior

        return (bs, value_mi)

    def eval(self, ctx, si, wo, active):
        wo = si.to_world(wo)
        wi = si.to_world(si.wi)
        normal_mesh = si.n
        position = si.p

        # Calculate the position after refraction
        refracted_screen_coor = self.calculate_refracted_screen_coor(wi, normal, 1.0 / self.ior, position, screen_coor)
        screen_coor = world_to_screen(position, self.view_matrix, self.persp_proj_matx, self.width, self.height)
        if self.use_mesh_normal:
            normal = normal_mesh
        else:
            normal = self.mat['normal']
            x, y = torch.floor(screen_coor[:, 0]).long(), torch.floor(screen_coor[:, 1]).long()
            normal = normal[x, y, :]
        brdf, _ = self.eval_brdf(wo, wi, normal, self.mat, screen_coor,refracted_screen_coor)
        # brdf = mi.Vector3f(brdf[..., 0], brdf[..., 1], brdf[..., 2])

        return brdf
    

    def sample_brdf(self, sample1, sample2, wo, normal_mesh, mat, screen_coor,refracted_screen_coor):
                # B = sample2.shape[0]
        B = len(sample1)

        # pdf = dr.zeros(B)
        brdf = mi.Vector3f(0.0)
        wi = mi.Vector3f(0.0)

        roughness = self.r
        

        x = mi.Int(dr.floor(screen_coor[0]))
        y = mi.Int(dr.floor(screen_coor[1]))

        flat_index = x + y * roughness.shape[0]
        roughness = dr.gather(mi.Float, roughness.array, flat_index)
        # mask = dr.gather(mi.Bool, self.mask.array, flat_index)
        # roughness[mask] = 0.1
        if self.use_mesh_normal:
            normal = normal_mesh
        else:
            normal = dr.gather(mi.Normal3f, self.n.array, flat_index)

        normal_offset = False
        if normal_offset:
            max_angle = 0.3  # Maximum allowed angular deviation in radians
            dot_nw = dr.dot(normal, wo)
            normal = dr.select(dot_nw < dr.cos(max_angle), dr.normalize(0.5 * normal + 0.5 * wo), normal)
            # normal = dr.select(roughness <= 0.1, dr.normalize(normal*0.5+0.5*wo), normal)


        if sample1 is None:
            # sample1 = dr.random<Float>(B)
            breakpoint()

        mask = (sample1 > 0.5)
        # sample diffuse
        wi[mask] = mi_diffuse_sampler(sample2[mask], normal[mask])
        mask = ~mask
        # sample specular
        wi[mask] = mi_specular_sampler(sample2[mask], roughness[mask], wo[mask], normal[mask])

        # get brdf,pdf
        brdf, pdf = self.eval_brdf(wi, wo, normal, mat, screen_coor,refracted_screen_coor)

        # brdf_weight = dr.select(pdf > 0, brdf / (pdf+1e-4), mi.Vector3f(0.0))
        brdf_weight = brdf / (pdf+1e-4)
        brdf_weight = dr.select(pdf > 0, brdf_weight, mi.Vector3f(0.0))
        pdf = dr.select(pdf > 0, pdf, mi.Float(0.0))
        return wi, pdf, brdf_weight

    def eval_brdf(self, wi, wo, normal_mesh, mat, screen_coor,refracted_screen_coor):
        albedo = self.a
        roughness = self.r
        metallic = self.m
        bg = self.bg
        mask = self.mask

        # get r,a,m by screen coor
        x = mi.Int(dr.floor(screen_coor[0]))
        y = mi.Int(dr.floor(screen_coor[1]))
        flat_index = x + y * roughness.shape[0]
        mask = dr.gather(mi.Bool, mask.array, flat_index)

        refracted_screen_coor[~mask] = screen_coor[~mask]
        x_refract = mi.Int(dr.floor(refracted_screen_coor[0]))
        y_refract = mi.Int(dr.floor(refracted_screen_coor[1]))
        flat_index_refract = x_refract + y_refract * roughness.shape[0]

        albedo = dr.gather(mi.Vector3f, albedo.array, flat_index)
        roughness = dr.gather(mi.Float, roughness.array, flat_index)
        # roughness[mask] = 0.1
        metallic = dr.gather(mi.Float, metallic.array, flat_index)
        bg = dr.gather(mi.Vector3f, bg.array, flat_index_refract)
        
        # normal = dr.gather(mi.Normal3f, normal.array, flat_index)
        # if not self.keep_albedo_color:
        #     albedo[mask] = 0.9
        if self.use_mesh_normal:
            normal = normal_mesh
        else:
            normal = dr.gather(mi.Normal3f, self.n.array, flat_index)

        h = dr.normalize(wi + wo)
        NoL = dr.maximum(dr.dot(normal, wi), 0)
        NoV = dr.maximum(dr.dot(normal, wo), 0)
        VoH = dr.maximum(dr.dot(wo, h), 0)
        NoH = dr.maximum(dr.dot(normal, h), 0)

        # get pdf
        D = D_GGX(NoH, roughness)
        pdf_spec = D / (4 * dr.maximum(VoH, 1e-4)) * NoH
        pdf_diff = NoL / math.pi
        pdf = 0.5 * pdf_spec + 0.5 * pdf_diff

        ##############################
        # original BRDF for scene
        disney_brdf = True
        if disney_brdf:
            baseColor_d = albedo * (1 - metallic)
            F_D90 = 0.5 + 2 * VoH ** 2 * roughness
            F_D_w_out = 1 + (F_D90 - 1) * (1 - NoV) ** 5
            F_D_w_in = 1 + (F_D90 - 1) * (1 - NoL) ** 5
            brdf_diff = baseColor_d / math.pi * F_D_w_out * F_D_w_in * NoL
            
            G = G_Smith(NoV, NoL, roughness)
            C_0 = (1 - metallic) * 0.04 + metallic * albedo
            F_m = C_0 + (1 - C_0) * (1 - VoH) ** 5
            brdf_metal = D * G  * F_m / 4 * NoL
            brdf_ori = brdf_diff + brdf_metal
            
        else:
            # get brdf
            kd = albedo * (1 - metallic) 
            # kd = albedo * metallic
            ks = 0.04 * (1 - metallic) + albedo * metallic 
            G = G_Smith(NoV, NoL, roughness)
            F = fresnelSchlick(VoH, ks)
            brdf_diff = kd / math.pi * NoL
            brdf_spec = D * G * F / 4.0 * NoL
            brdf_ori = brdf_diff + brdf_spec 

        # Reflective BRDF for material editing
        kd =  albedo * (1 - metallic) * (1 - self.specTrans)   
        baseColor_m = (1-self.specTrans*(1-metallic)) * albedo * metallic  
        baseColor_glass = (1-metallic) * (bg * self.specTrans) 
        # F_m = fresnelSchlick(VoH, baseColor_m)
        C_0 = (1 - metallic) * 0.04 + metallic * albedo
        F_m = C_0 + (1 - C_0) * (1 - VoH) ** 5

        sign = NoL * NoV
        glass_mask = sign > 0

        brdf_diff = kd / math.pi * NoL
        brdf_metal = D * G * F_m / 4.0 * NoL
        LoH = dr.maximum(dr.dot(wi, h), 0)
        hw_in = 1 / (LoH + 1e-6)
        hw_out = 1 / (VoH + 1e-6)
        nw_in = 1 / (NoL + 1e-6)
        nw_out = 1 / (NoV + 1e-6)
        R_s = (hw_in - self.ior * hw_out) / (hw_in + self.ior * hw_out)
        R_p = (self.ior * hw_in - hw_out) / (self.ior * hw_in + hw_out)
        F_glass = 0.5 * (R_s**2 + R_p**2) # ior up, F_glass up
        D_hacking = D_GGX(NoH, roughness*0+1) 
        # try to make the rays not specular reflect when transmit by hacking the roughness
        btdf_glass = (baseColor_glass)**0.5 * G * D_hacking * (1 - F_glass) * (self.ior**2 * hw_in * hw_out) / (nw_in * nw_out * (self.ior * hw_in + hw_out)**2)
        brdf_spec_edit = baseColor_glass * D * G / (4 * nw_in) 
        # brdf_spec_edit = 0
        f_glass = dr.select(glass_mask, brdf_spec_edit, btdf_glass)
        bsdf_edit = brdf_diff + brdf_metal + f_glass

        bsdf_edit[~mask] = 0
        brdf_ori[mask] = 0
        bsdf = bsdf_edit + brdf_ori

        bsdf = dr.select(bsdf>0,bsdf,mi.Vector3f(0.0))
        pdf = dr.select(pdf>0,pdf,mi.Float(0.0))
        return bsdf, pdf

    def pdf(self, ctx, si, wo, active):
        wo = si.to_world(wo).torch()
        wi = si.to_world(si.wi).torch()
        normal_mesh = si.n.torch()
        position = si.p.torch()

        screen_coor = world_to_screen(position, self.view_matrix, self.persp_proj_matx, self.width, self.height)

        if self.use_mesh_normal:
            normal = normal_mesh
        else:
            normal = self.mat['normal']
            x, y = torch.floor(screen_coor[:, 0]).long(), torch.floor(screen_coor[:, 1]).long()
            normal = normal[x, y, :]

        refracted_screen_coor = self.calculate_refracted_screen_coor(wi, normal, 1.0 / self.ior, position, screen_coor)

        _, pdf = self.base_brdf.eval_brdf(wo,wi,normal,self.mat,screen_coor=screen_coor,refracted_screen_coor=refracted_screen_coor)
        pdf = mi.Float(pdf.squeeze(-1))

        return pdf

    def eval_pdf(self, ctx, si, wo, active=True):
        wo = si.to_world(wo)
        wi = si.to_world(si.wi)

        normal = si.n
        position = si.p

        screen_coor = mi_world_to_screen(position,self.view_matrix,self.persp_proj_matx,self.width,self.height)

        refracted_screen_coor = self.calculate_refracted_screen_coor(wi, normal, 1.0 / self.ior, position, screen_coor)

        brdf,pdf = self.eval_brdf(wo,wi,normal,self.mat,screen_coor=screen_coor,refracted_screen_coor=refracted_screen_coor)

        return brdf,pdf
    
    def traverse(self, callback):
        callback.put_parameter('a', self.a, mi.ParamFlags.Differentiable)
        callback.put_parameter('r', self.r, mi.ParamFlags.Differentiable)
        callback.put_parameter('m', self.m, mi.ParamFlags.Differentiable)
        callback.put_parameter('bg', self.bg, mi.ParamFlags.Differentiable)
        callback.put_parameter('mask', self.mask, mi.ParamFlags.Differentiable)
        callback.put_parameter('specTrans', self.specTrans, mi.ParamFlags.Differentiable)
        callback.put_parameter('ior', self.ior, mi.ParamFlags.Differentiable)


class BRDF4scratch(BaseBRDF):

    def __init__(self,
                 in_dims,
                 out_dims,
                 dims,
                 skip_connection=(),
                 weight_norm=True,
                 multires_view=0,
                 output_type='envmap',color_ch = 5):
        super().__init__()
        self.init_range = np.sqrt(3 / dims[0])

        dims = [in_dims] + dims + [out_dims]
        first_omega = 1
        hidden_omega = 1
        self.output_type = output_type

        self.embedview_fn = lambda x: x

        if multires_view > 0:

            embedview_fn, input_ch = get_embedder(multires_view, in_dims)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - in_dims) + color_ch
        # breakpoint()
        self.num_layers = len(dims)
        self.skip_connection = skip_connection

        for l in range(0, self.num_layers - 1):

            if l + 1 in self.skip_connection:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            is_first = (l == 0) and (multires_view == 0)
            is_last = (l == (self.num_layers - 2))

            if not is_last:
                omega_0 = first_omega if is_first else hidden_omega
                lin = SineLayer(dims[l], out_dim, True, is_first, omega_0,
                                weight_norm)
            else:
                lin = nn.Linear(dims[l], out_dim)
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
                if weight_norm:
                    lin = nn.utils.weight_norm(lin)
                    if torch.isnan(lin.weight).any():
                        raise ValueError(f'nan value in lin{l}.weight')

            setattr(self, "lin" + str(l), lin)

            # self.last_active_fun = nn.Tanh()
            # self.last_active_fun = nn.Identity()
            self.last_active_fun = nn.Softplus()
            # self.last_active_fun = nn.ReLU()
        pass

    def forward(self, points):
        init_x = self.embedview_fn(points)
        x = init_x

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if hasattr(lin, 'linear') and torch.isnan(lin.linear.weight).any():
                raise ValueError(f'nan value in lin{l}.weight')
            elif hasattr(lin, 'weight') and torch.isnan(lin.weight).any():
                raise ValueError(f'nan value in lin{l}.weight')

            if l in self.skip_connection:
                x = torch.cat([x, init_x], -1)

            x = lin(x)

            if torch.isnan(x).any():
                raise ValueError(f'nan value in x in lin{l}')
        if self.output_type == 'envmap':
            x = nn.Softplus()(x) # make sure positive
        elif self.output_type == 'mat':
            x = nn.Tanh()(x)
            x = x * 0.5 + 0.5  # scale to [0,1]
        else:
            raise ValueError('output_type should be envmap or mat')
        return x
