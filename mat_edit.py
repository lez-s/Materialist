from myutils.mi_plugin import RefractBaseBRDF
import json
import global_config
from render_final_old import load_estimated_mesh_w_env
import os
import drjit as dr
import numpy as np
import mitsuba as mi
from tqdm import tqdm
import torch
from torchvision.utils import save_image
from myutils.misc import linear_to_srgb
# dr.set_flag(dr.JitFlag.VCallRecord, False)
# dr.set_flag(dr.JitFlag.LoopRecord, False)

def transprancy_edit(mesh_path,env_path,save_name,ior,keep_albedo_color,n_iter=10):
    root_dir = f'output_imgs/{save_name}/best_results'
    scene = load_estimated_mesh_w_env(mesh_path,env_path,root_dir,bsdf={'name':'lezrefractBSDF','ior':ior,'keep_albedo_color':keep_albedo_color})
    env_id = env_path.split('/')[-1][:-4]
    # img = mi.render(scene,spp=64)
    empty_img = np.zeros((512,512,3),dtype=np.float32)
    for i in tqdm(range(n_iter)):
        img = mi.render(scene,spp=64,seed=i)
        empty_img += img.numpy()
    img = empty_img/n_iter 
    albedo_flag = 'wA' if keep_albedo_color else 'woA'
    mi.util.write_bitmap(os.path.join(global_config.OUT_DIR,save_name,f'mi_trans_{ior}_{albedo_flag}_{save_name}_{env_id}.exr'),img)
    
    img = linear_to_srgb(img)
    img_torch = torch.from_numpy(img).permute(2,0,1)
    save_image(img_torch,os.path.join(global_config.OUT_DIR,save_name,f'mi_trans_{ior}_{albedo_flag}_{save_name}_{env_id}.png'))
    print("Wrote file to ", os.path.join(global_config.OUT_DIR,save_name,f'mi_trans_{ior}_{albedo_flag}_{save_name}_{env_id}.exr'))

def render_trans_scene(save_name,ior,keep_albedo_color,env_path=None):
    mesh_path = os.path.join(global_config.OUT_DIR,save_name,f'{save_name}.ply')
    if env_path == None:
        env_path = os.path.join(global_config.OUT_DIR,save_name,'best_results')
        if os.path.exists(os.path.join(env_path,'envmap_opt.hdr')):
            print("Using mitsuba optimized envmap")
            env_path = os.path.join(env_path,'envmap_opt.hdr')
        elif os.path.exists(os.path.join(env_path,'envmap.hdr')):
            print("Using original envmap")
            env_path = os.path.join(env_path,'envmap.hdr')
        else:
            raise ValueError("No envmap found")
    transprancy_edit(mesh_path,env_path,save_name,ior,keep_albedo_color,n_iter=10)

if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")
    # mi.register_bsdf("lezrefractBSDF", lambda props: RefractBaseBRDF(props))
    save_name = 'vase_rm'
    keep_albedo_color = False
    ior = 1.0

    render_trans_scene(save_name,ior,keep_albedo_color,env_path=None)