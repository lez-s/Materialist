from myutils.mi_plugin import load_estimated_brdf
import json
import global_config
from render_final import load_estimated_mesh_w_env
import os
import drjit as dr
import numpy as np
import mitsuba as mi
from tqdm import tqdm
import torch
from torchvision.utils import save_image
from myutils.misc import linear_to_srgb
dr.set_flag(dr.JitFlag.VCallRecord, False)
dr.set_flag(dr.JitFlag.LoopRecord, False)

def transprancy_edit(mesh_path,env_path,save_name,ior,keep_albedo_color,specTrans,n_iter=10):
    mat_dir = f'output_imgs/{save_name}/best_results'
    scene = load_estimated_mesh_w_env(mesh_path,env_path,mat_dir,bsdf={'name':'TransBSDF','ior':ior,'keep_albedo_color':keep_albedo_color})
    env_id = env_path.split('/')[-1][:-4]
    mat = load_estimated_brdf(mat_dir)
    mask = mat['mask']
    albedo = mat['albedo']
    roughness = mat['roughness']
    metallic = mat['metallic']
    if not keep_albedo_color:
        albedo[mask] = 0.7
    roughness[mask] = roughness[mask] * 0 + 0.3
    metallic[mask] = metallic[mask] * 0.
    mi_params = mi.traverse(scene)
    mi_params['shape.bsdf.a'] = albedo
    mi_params['shape.bsdf.r'] = roughness 
    mi_params['shape.bsdf.m'] = metallic
    mi_params['emitter.data'] = mat['envmap']
    mi_params['shape.bsdf.bg'] = mat['bg']
    mi_params['shape.bsdf.mask'] = mi.TensorXf(mat['mask'].float()) >= 1 
    mi_params['shape.bsdf.specTrans'] = specTrans
    mi_params['shape.bsdf.ior'] = ior
    mi_params.update()
    empty_img = mi.TensorXf(0.0,shape=(512,512,3))
    for i in tqdm(range(n_iter)):
        img = mi.render(scene,spp=64,seed=i)
        empty_img += img
    img = empty_img/n_iter

    albedo_flag = 'wA' if keep_albedo_color else 'woA'
    filename = f'mi_trans_{ior}_{albedo_flag}_{specTrans}_{save_name}_{env_id}'
    mi.util.write_bitmap(os.path.join(global_config.OUT_DIR,save_name,f'{filename}.exr'),img)
    mi.util.write_bitmap(os.path.join(global_config.OUT_DIR,save_name,f'{filename}.png'),img)
    print("Wrote file to ", os.path.join(global_config.OUT_DIR,save_name,f'{filename}.png'))

def render_trans_scene(save_name,ior,keep_albedo_color,specTrans,env_path=None):
    mesh_path = os.path.join(global_config.OUT_DIR,save_name,f'{save_name}.ply')
    if env_path == None:
        env_path = os.path.join(global_config.OUT_DIR,save_name,'best_results')
        if os.path.exists(os.path.join(env_path,'envmap.hdr')):
            print("Using original envmap")
            env_path = os.path.join(env_path,'envmap.hdr')
        else:
            raise ValueError("No envmap found")
    transprancy_edit(mesh_path,env_path,save_name,ior,keep_albedo_color,specTrans,n_iter=10)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Render a scene with transparency editing')
    parser.add_argument('--save_name', type=str, required=True, help='Name of the save directory')
    parser.add_argument('--ior', type=float, default=1.2, help='Index of refraction')
    parser.add_argument('--keep_albedo_color', action='store_true', help='Keep the albedo color')
    parser.add_argument('--specTrans', type=float, default=0.4, help='Specular transmission')
    parser.add_argument('--env_path', type=str, default=None, help='Path to the environment map')
    return parser.parse_args()

if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")
    args = parse_args()
    save_name = args.save_name
    ior = args.ior
    keep_albedo_color = args.keep_albedo_color
    specTrans = args.specTrans
    env_path = args.env_path
    render_trans_scene(save_name,ior,keep_albedo_color,specTrans,env_path)