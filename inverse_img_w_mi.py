import mitsuba as mi
import drjit as dr
from myutils.mi_plugin import MatDiffBSDF
from huggingface_hub import hf_hub_download
from Material_net.dpt import MaterialNet
mi.register_bsdf('MatDiffBSDF', lambda props: MatDiffBSDF(props))
import global_config
from torchvision.utils import save_image,make_grid
from tqdm import tqdm
import gc
from myutils.misc import *
from mymodels.mlps import PosMLP
from myutils.envmap_utils import lookup_envmap, importance_sample, build_envmap,sample_brdf1,sample_env1
from myutils.mesh_recon import depth_file_to_mesh,rotate_mesh_around_x
import open3d as o3d
import argparse
import torch.nn.functional as NF
import json
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR,StepLR
import matplotlib.pyplot as plt
import warnings
import sys
import time
import imageio
import os

# from render_final import load_estimated_mesh_w_env

def load_estimated_mesh(mesh_path,use_mesh_normal,max_path=4):
    camera_cfg = {"type": "perspective",
                "fov":35,
                "to_world": mi.ScalarTransform4f.look_at(
                        origin=[0, 0, 0], target=[0, 0, -1], up=[0, 1, 0]),
                "film": {
                "type": "hdrfilm",
                "width": 512,  # Viewport width in pixels
                "height": 512,  # Viewport height in pixels
                }}
    camera = mi.load_dict(camera_cfg)
    cam_cfg_path = os.path.join(global_config.BASE_DIR,"myutils", "default_cam.json")
    scene = mi.load_dict({
            'type': 'scene',
            'shape':{
                'type': 'ply',
                'filename': mesh_path,
                'bsdf': {'type': 'MatDiffBSDF','cam_meta':cam_cfg_path,'use_mesh_normal':use_mesh_normal},
            },
            'integrator': {
                'type': 'path',
                'max_depth': max_path,
            },
            'sensor': camera,
            'emitter': {'type': 'envmap','filename':'envmaps/0.hdr'}
        })
    return scene


@dr.wrap_ad(source='torch', target='drjit')
def render_envmap(scene,envmap,spp=64):
    params = mi.traverse(scene)
    random_seed = np.random.randint(0,1000)
    params['emitter.data']=envmap 
    params.update()
    rendered_img=mi.render(scene, params, spp=spp, seed=random_seed)

    return rendered_img

@dr.wrap_ad(source='torch', target='drjit')
def render_w_brdf(scene,albedo,roughness, metallic,normal=None,spp=64):
    random_seed = np.random.randint(0,1000)
    params = mi.traverse(scene)
    params['shape.bsdf.a'] = albedo
    params['shape.bsdf.r'] = roughness
    params['shape.bsdf.m'] = metallic
    if normal is not None:
        params['shape.bsdf.n'] = normal
    params.update()
    rendered_img=mi.render(scene, params, spp=spp,seed=random_seed)
    return rendered_img

def get_output_dir(save_name, save_path=None):
    """Determine the output directory based on save_name and save_path.
    
    Args:
        save_name: Name of the save directory
        save_path: Optional path where results should be saved
        
    Returns:
        Full path to the output directory
    """
    if save_path:
        # If save_path is provided, use it directly
        if os.path.isabs(save_path):
            return os.path.join(save_path, save_name)
        else:
            # If save_path is relative, treat it relative to OUT_DIR
            return os.path.join(global_config.OUT_DIR, save_path, save_name)
    
    # If no save_path, use the previous logic for backwards compatibility
    if os.path.isabs(save_name):
        return save_name
    else:
        return os.path.join(global_config.OUT_DIR, save_name)

def optimize_envmap_ARMN(scene,cam_cfg,mat,save_folder,use_mesh_normal,
                    output_type,optimize_order,spp=64,use_gt_scene = False,
                    model_name='pos_mlp',opt_env_from=0,
                    opt_src='arm',use_mask=False,scale_delta=0.1, save_path=None):
    '''
    mat: dict, albedo:H,W,C, roughness:H,W,1, metallic:H,W,1, normal:H,W,3, depth:H,W,1, gt_image:H,W,3
    '''
    device = torch.device('cuda')
    depth = 4
    width = 256
    weight_norm = False
    envmap_net = PosMLP(in_dims=5,
                            out_dims=3,
                            dims=[width] * depth,
                            skip_connection=[1,3],
                            weight_norm=weight_norm,
                            multires_view = 2,
                            output_type='envmap',color_ch=3)
    envmap_net = envmap_net.cuda()
    # opt_env = torch.optim.Adam(envmap_net.parameters(), lr=1e-4)
    
    # Get output directory
    output_dir = get_output_dir(save_folder, save_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, 'best_results'))
    
    # Create directories for intermediate results
    env_frames_dir = os.path.join(output_dir, 'env_frames')
    mat_frames_dir = os.path.join(output_dir, 'mat_frames')
    os.makedirs(env_frames_dir, exist_ok=True)
    os.makedirs(mat_frames_dir, exist_ok=True)
    
    # Lists to collect frames for videos
    env_frames = []
    mat_frames = []

    opt_normal = False
    if opt_normal:
        normal_net = PosMLP(in_dims=5,
                            out_dims=3,
                            dims=[width] * depth,
                            skip_connection=[1,3],
                            weight_norm=weight_norm,
                            output_type='normal')
        normal_net = normal_net.cuda()
        opt_normal_net = torch.optim.Adam(normal_net.parameters(), lr=1e-12)

    # brdf net

    if model_name == 'unet':
        raise ValueError('Do not use unet for this task')
    elif model_name == 'pos_mlp':
        if output_type == 'arm':
            multires_view = 2 # for pos embedding
            color_ch = 5
            brdf_net = PosMLP(in_dims=7,out_dims=color_ch,dims=[width] * depth,skip_connection=[1,3],weight_norm=weight_norm,multires_view=multires_view,output_type=output_type,color_ch = color_ch)
            brdf_net = brdf_net.cuda()
            # opt_brdf = torch.optim.Adam(brdf_net.parameters(), lr=1e-4)
            
        elif output_type == 'armn':
            multires_view = 0
            color_ch = 8
            brdf_net = PosMLP(in_dims=10,out_dims=color_ch,dims=[width] * depth,skip_connection=[1,3],weight_norm=weight_norm,multires_view=multires_view,output_type=output_type,color_ch = color_ch)
            brdf_net = brdf_net.cuda()
            # opt_brdf = torch.optim.Adam(brdf_net.parameters(), lr=1e-4)


    gt_image = mat['gt_image']
    if use_gt_scene:
        gt_envmap = mat['gt_envmap']

    env_h, env_w = 16, 32
    start_envmap = torch.ones(env_h, env_w, 3, device=device)
    start_envmap = start_envmap.reshape(-1, 3)
    
    roughness_shift = 0.7
    metallic_shift = 0.05
    if not 'r' in opt_src:
        mat['roughness'] = mat['roughness'] * 0 + roughness_shift
    if not 'm' in opt_src:
        mat['metallic'] = mat['metallic'] * 0 + metallic_shift

    albedo_ori = mat['albedo']
    roughness_ori = mat['roughness']
    metallic_ori = mat['metallic']
    normal_ori = mat['normal']
    normal_ori = NF.normalize(normal_ori, p=2, dim=-1)

    start_normal = normal_ori.reshape(-1,3)
    
    if not 'r' in opt_src and opt_src != 'skip':
        roughness_ori = roughness_ori * 0 + roughness_shift
    if not 'm' in opt_src and opt_src != 'skip':
        metallic_ori = metallic_ori * 0 + metallic_shift
    if output_type == 'armn':
        start_arm = torch.cat([albedo_ori.reshape(-1,3), roughness_ori.reshape(-1,1), metallic_ori.reshape(-1,1),normal_ori.reshape(-1,3)], dim=-1)
    elif output_type == 'arm':
        start_arm = torch.cat([albedo_ori.reshape(-1,3), roughness_ori.reshape(-1,1), metallic_ori.reshape(-1,1)], dim=-1).clamp(0,1)
    else:
        raise ValueError('output_type should be arm or armn')
    # start_arm = gt_image.reshape(1,-1,3)
    start_arm_unet = torch.cat([albedo_ori.permute(2,0,1), roughness_ori.permute(2,0,1), metallic_ori.permute(2,0,1)], dim=0).unsqueeze(0)

    num_epochs = 5000
    epoch = 0
    saver = SaveBest()
    loop_num = 0

    mi_params = mi.traverse(scene)
    mi_params['shape.bsdf.a'] = mat['albedo']
    mi_params['shape.bsdf.r'] = mat['roughness']
    mi_params['shape.bsdf.m'] = mat['metallic']
    mi_params.update()
    
    early_stopping_all = EarlyStopping(patience=2, min_delta=0.025)
    while loop_num <= 10:
        loop_num +=1
        if loop_num == 1:
            opt_env = torch.optim.Adam(envmap_net.parameters(), lr=1e-3)
            scheduler_env = StepLR(opt_env, step_size=100, gamma=0.8)
        else:
            opt_env = torch.optim.Adam(envmap_net.parameters(), lr=1e-4)
        # optimize envmap
        if opt_src == 'skip' and opt_src == 'skip':
            patience_env = 500
        else:
            patience_env = 100
        early_stopping = EarlyStopping(patience=patience_env, min_delta=0.01)
        with tqdm(total=num_epochs,desc='Opt envmap', unit='epochs',file=sys.stdout) as pbar:
            for epoch in range(num_epochs):
                envmap_pred = envmap_net(start_envmap)
                envmap_pred = envmap_pred.squeeze().reshape(env_h, env_w, 3)
                pred_image = render_envmap(scene,envmap_pred,spp)
                pred_image_srgb = linear_to_srgb(pred_image)
                gt_image_srgb = linear_to_srgb(gt_image)
                loss_mse = NF.mse_loss(pred_image_srgb, gt_image_srgb)
                loss_l1 = NF.l1_loss(pred_image_srgb, gt_image_srgb)
                loss = loss_mse + loss_l1
                
                saver.update(loss_mse.item(), mat['albedo'], mat['roughness'], mat['metallic'],mat['normal'],envmap_pred,pred_image)
                loss.backward()

                early_stopping(loss_mse.item())
                opt_env.step()
                opt_env.zero_grad()
                if loop_num == 1:
                    scheduler_env.step()
                pbar.set_postfix(loss=loss.item(),loss_mse=loss_mse.item())
                pbar.update(1)
                if epoch % 10 ==0 or early_stopping.early_stop:
                    all_env = torch.concat([envmap_pred], dim=1)
                    env_display = torch.zeros_like(gt_image_srgb)  # Blank canvas same size as gt_image
                    h, w = envmap_pred.shape[:2]
                    display_h = min(h * 3, env_display.shape[0] // 2)  
                    display_w = int(display_h * (w / h))
                    
                    # Position the envmap in the center of the blank image
                    start_h = (env_display.shape[0] - display_h) // 2
                    start_w = (env_display.shape[1] - display_w) // 2
                    
                    # Resize and place the envmap
                    envmap_display = NF.interpolate(
                        envmap_pred.permute(2, 0, 1).unsqueeze(0),
                        size=(display_h, display_w),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0)
                    
                    env_display[start_h:start_h+display_h, start_w:start_w+display_w] = envmap_display
                    all_image = torch.concat([gt_image_srgb, pred_image_srgb,env_display], dim=1)
                    
                    save_image(all_env.permute(2, 0, 1).unsqueeze(0), os.path.join(output_dir, f'env.png'))
                    
                    # Also save to output directory for easy viewing of latest frame
                    env_frame_path = os.path.join(env_frames_dir, f'opt_env_frame_{loop_num}_{epoch:04d}.png')
                    save_image(all_image.permute(2, 0, 1).unsqueeze(0),env_frame_path)
                    env_frames.append(env_frame_path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                if loop_num < opt_env_from:
                    print(f'loop_num {loop_num} < opt_env_from {opt_env_from}, break')
                    break
                if 'rm' not in opt_src and loop_num == 1:
                    if opt_src != 'skip':
                        print(f'rm not in opt_src and loop_num == 1, break')
                        break

        final_envmap = saver.best_envmap
        mi.util.write_bitmap(os.path.join(output_dir, 'final_envmap.hdr'), final_envmap)
        save_image(all_image.permute(2, 0, 1).unsqueeze(0),os.path.join(output_dir, f'opt_env_img.png'))
        torch.cuda.empty_cache()
        gc.collect()

        if loop_num >= opt_env_from :
            saver.save_results(os.path.join(output_dir, f'best_results'))
        early_stopping_all(loss_mse.item())
        if early_stopping_all.early_stop:
            print("Early stopping")
            loop_num = 11
            break
        if loop_num >=3:
            break
        if opt_src == 'skip' or opt_src == 'skip':
            break
        
        ############################
        # optimize brdf
        ###########################
        if loop_num < opt_env_from and loop_num==1:
            if 'gt_envmap' in mat.keys():
                envmap4render = mat['gt_envmap']
                print('use gt envmap for brdf optimization')
            else:
                envmap4render = torch.ones(16,32,3).cuda()
                print('Use envmap = 1 for brdf optimization')
        else:
            envmap4render = final_envmap
            print('Use Optimized envmap for brdf optimization')
        
        if loop_num <=1:
            if not 'r' in opt_src:
                mat['roughness'] = mat['roughness'] * 0 + roughness_shift
            if not 'm' in opt_src:
                mat['metallic'] = mat['metallic'] * 0 + metallic_shift

        mi_params['emitter.data'] = envmap4render
        if use_mesh_normal:
            mi_params['shape.bsdf.use_mesh_normal'] = True
            print('Use Mesh Normal')
        else:
            mi_params['shape.bsdf.use_mesh_normal'] = False
            print('Use Predicted Normal')
        # mi_params['integrator.max_depth'] = 2
        mi_params.update()
        for optimize_part in optimize_order:
            if optimize_part == 'a' and loop_num <= 1: #skip the first loop for albedo
                continue
                
            if model_name == 'none':
                print(f"Directly optimizing {optimize_part} without neural network")
                optimizable_params = {}
                if 'a' in optimize_part:
                    optimizable_params['albedo'] = torch.nn.Parameter(mat['albedo'].clone())
                if 'r' in optimize_part:
                    optimizable_params['roughness'] = torch.nn.Parameter(mat['roughness'].clone())
                if 'm' in optimize_part:
                    optimizable_params['metallic'] = torch.nn.Parameter(mat['metallic'].clone())
                if 'n' in optimize_part and not use_mesh_normal:
                    optimizable_params['normal'] = torch.nn.Parameter(mat['normal'].clone())

                opt_params = torch.optim.Adam(optimizable_params.values(), lr=3e-4)
                scheduler_params = StepLR(opt_params, step_size=100, gamma=0.8)
                
                if 'a' in optimize_part:
                    early_stopping = EarlyStopping(patience=200//loop_num, min_delta=0.005)
                else:
                    early_stopping = EarlyStopping(patience=200//loop_num, min_delta=0.001)
                    
                with tqdm(total=num_epochs, desc=f'Opt {optimize_part} directly', unit='e', file=sys.stdout) as pbar:
                    for epoch in range(num_epochs):

                        if 'a' in optimize_part:
                            mat['albedo'] = optimizable_params['albedo'].clamp(0, 1)
                        if 'r' in optimize_part:
                            mat['roughness'] = optimizable_params['roughness'].clamp(0.07, 1)
                        if 'm' in optimize_part:
                            mat['metallic'] = optimizable_params['metallic'].clamp(0, 1)
                        if 'n' in optimize_part and not use_mesh_normal:
                            mat['normal'] = NF.normalize(optimizable_params['normal'], p=2, dim=-1)
                            
                        if use_mask:
                            mat['roughness'][mat['mask']] = mat['roughness'][mat['mask']].mean()
                            mat['metallic'][mat['mask']] = mat['metallic'][mat['mask']].mean()
                            
                        if use_mesh_normal:
                            pred_image = render_w_brdf(scene, mat['albedo'], mat['roughness'], mat['metallic'], None, spp)
                        else:
                            pred_image = render_w_brdf(scene, mat['albedo'], mat['roughness'], mat['metallic'], mat['normal'], spp)
                            
                        ratio = gt_image.mean()/pred_image.detach().mean()
                        pred_image = pred_image * ratio 
                        
                        pred_image_srgb = linear_to_srgb(pred_image)
                        gt_image_srgb = linear_to_srgb(gt_image)

                        loss_mse = NF.mse_loss(pred_image_srgb, gt_image_srgb)
                        loss_l1 = NF.l1_loss(pred_image_srgb, gt_image_srgb)
                        

                        if 'a' in optimize_part:
                            loss_a = NF.l1_loss(mat['albedo'], albedo_ori)
                        else:
                            loss_a = 0
                        if 'r' in optimize_part:
                            loss_r = NF.l1_loss(mat['roughness'], roughness_ori)
                        else:
                            loss_r = 0
                        if 'm' in optimize_part:
                            loss_m = NF.l1_loss(mat['metallic'], metallic_ori)
                        else:
                            loss_m = 0
                        if 'n' in optimize_part and not use_mesh_normal:
                            loss_n = NF.l1_loss(mat['normal'], normal_ori)
                        else:
                            loss_n = 0
                            
                        scale_raito = loss_l1.detach()/loss_mse.detach()
                        aux_loss = loss_a + (loss_r + loss_m + loss_n)
                        render_loss = 3 * scale_raito * loss_mse + loss_l1
                        loss = render_loss + aux_loss * scale_delta

                        loss.backward()
                        
                        saver.update(loss_mse.item(), mat['albedo'], mat['roughness'], mat['metallic'], mat['normal'], 
                                     envmap4render, pred_image_srgb, None)
                        
                        for param_group in opt_params.param_groups:
                            current_lr = param_group['lr']

                        early_stopping(loss_mse.item())
                        opt_params.step()
                        opt_params.zero_grad()
                        if current_lr > 1.5e-4:
                            scheduler_params.step()

                        pbar.set_postfix(loss=loss.item(), render=render_loss.item(), aux=aux_loss.item(), 
                                         lr=f"{current_lr:.2e}")
                        pbar.update(1)
                        
                        if epoch % 10 == 0 or early_stopping.early_stop:
                            all_image = torch.stack([gt_image_srgb, pred_image_srgb, mat['albedo'], 
                                                    mat['roughness'].repeat(1, 1, 3), 
                                                    mat['metallic'].repeat(1, 1, 3), mat['normal']], dim=0)
                            all_image = make_grid(all_image.permute(0, 3, 1, 2), nrow=3)
                            
                            mat_frame_path = os.path.join(mat_frames_dir, f'mat_frame_{loop_num}_{optimize_part}_{epoch:04d}.png')
                            save_image(all_image, mat_frame_path)
                            mat_frames.append(mat_frame_path)
                            
                        if early_stopping.early_stop:
                            print("Early stopping")
                            if 'a' in optimize_part:
                                mat['albedo'] = saver.best_albedo
                            if 'r' in optimize_part:
                                mat['roughness'] = saver.best_roughness
                            if 'm' in optimize_part:
                                mat['metallic'] = saver.best_metallic
                            if 'n' in optimize_part:
                                mat['normal'] = saver.best_normal
                            break
                            
                mat['albedo'] = saver.best_albedo
                mat['roughness'] = saver.best_roughness
                mat['metallic'] = saver.best_metallic
                mat['normal'] = saver.best_normal
                
                saver.save_results(os.path.join(output_dir, f'best_results'))
                
                torch.cuda.empty_cache()
                gc.collect()
            else:

                opt_brdf = torch.optim.AdamW(brdf_net.parameters(), lr=3e-4)
                scheduler_brdf = StepLR(opt_brdf, step_size=100, gamma=0.8)
                # scheduler_brdf = OneCycleLR(opt_brdf, max_lr=5e-4, total_steps=num_epochs//50//loop_num)
                if 'a' in optimize_part:
                    early_stopping = EarlyStopping(patience=200//loop_num, min_delta=0.005)
                else:
                    early_stopping = EarlyStopping(patience=200//loop_num, min_delta=0.001)
                with tqdm(total=num_epochs,desc=f'Opt {optimize_part}', unit='e',file=sys.stdout) as pbar:
                    for epoch in range(num_epochs):
                        if model_name == 'unet':
                            albedo_pred, roughness_pred,metallic_pred = brdf_net(start_arm_unet)
                            if 'a' in optimize_part:
                                albedo = albedo_pred.squeeze(0).permute(1,2,0)
                                mat['albedo'] = albedo
                            if 'r' in optimize_part:
                                roughness = roughness_pred.squeeze(0).permute(1,2,0)
                                mat['roughness'] = roughness
                            if 'm' in optimize_part:
                                metallic = metallic_pred.squeeze(0).permute(1,2,0)
                                mat['metallic'] = metallic

                        elif model_name == 'mlp' or model_name == 'pos_mlp':
                            arm_pred = brdf_net(start_arm)
                            albedo = (arm_pred[...,0:3]).clamp(0,1)
                            roughness = (arm_pred[...,3:4] * 0.93 + 0.07).clamp(0,1)
                            metallic = (arm_pred[...,4:5]).clamp(0,1)
                            if output_type == 'armn':
                                normal = NF.normalize(arm_pred[...,5:8],p=2, dim=1)
                            if 'a' in optimize_part:
                                mat['albedo'] = albedo.reshape(512,512,3)
                            if 'r' in optimize_part:
                                mat['roughness'] = roughness.reshape(512,512,1)
                            if 'm' in optimize_part:
                                mat['metallic'] = metallic.reshape(512,512,1)
                            if 'n' in optimize_part:
                                mat['normal'] = normal.reshape(512,512,3)
                        else:
                            raise ValueError('model_name should be unet or mlp or pos_mlp')
                        if use_mask:
                            mat['roughness'][mat['mask']] = mat['roughness'][mat['mask']].mean()
                            mat['metallic'][mat['mask']] = mat['metallic'][mat['mask']].mean()
                        if use_mesh_normal:
                            pred_image = render_w_brdf(scene,mat['albedo'],mat['roughness'],mat['metallic'],None,spp)
                        else:
                            pred_image = render_w_brdf(scene,mat['albedo'],mat['roughness'],mat['metallic'],mat['normal'],spp)
                        ratio = gt_image.mean()/pred_image.detach().mean()
                        pred_image = pred_image * ratio 
                        pred_image_srgb = linear_to_srgb(pred_image)
                        gt_image_srgb = linear_to_srgb(gt_image)

                        loss_mse = NF.mse_loss(pred_image_srgb, gt_image_srgb)
                        loss_l1 = NF.l1_loss(pred_image_srgb, gt_image_srgb)
                        if 'a' in optimize_part:
                            loss_a = NF.l1_loss(albedo.reshape(512,512,3), albedo_ori)
                        else:
                            loss_a = 0
                        if 'r' in optimize_part:
                            loss_r = NF.l1_loss(roughness.reshape(512,512,1), roughness_ori)
                        else:
                            loss_r = 0
                        if 'm' in optimize_part:
                            loss_m = NF.l1_loss(metallic.reshape(512,512,1), metallic_ori)
                        else:
                            loss_m = 0
                        if 'n' in optimize_part:
                            loss_n = NF.l1_loss(normal.reshape(512,512,3), normal_ori)
                        else:
                            loss_n = 0
                        scale_raito = loss_l1.detach()/loss_mse.detach()
                        aux_loss = loss_a + (loss_r + loss_m + loss_n)
                        render_loss = 3 * scale_raito * loss_mse + loss_l1
                        loss = render_loss + aux_loss * scale_delta

                        loss.backward()
                        current_weights = brdf_net.state_dict()
                        
                        saver.update(loss_mse.item(), mat['albedo'],mat['roughness'], mat['metallic'],mat['normal'], envmap4render,pred_image_srgb,current_weights)
                        for param_group in opt_brdf.param_groups:
                            current_lr = param_group['lr']

                        early_stopping(loss_mse.item())
                        opt_brdf.step()
                        opt_brdf.zero_grad()
                        if current_lr > 1.5e-4:
                            scheduler_brdf.step()

                        pbar.set_postfix(loss=loss.item(),render=render_loss.item(),aux=aux_loss.item(),lr=f"{current_lr:.2e}")
                        pbar.update(1)
                        if epoch % 10 == 0 or early_stopping.early_stop:
                            all_image = torch.stack([gt_image_srgb,pred_image_srgb,mat['albedo'], mat['roughness'].repeat(1, 1, 3), mat['metallic'].repeat(1, 1, 3),mat['normal']], dim=0)
                            all_image = make_grid(all_image.permute(0,3,1,2), nrow=3)
                            
                            # Save to frames directory with frame number in filename
                            mat_frame_path = os.path.join(mat_frames_dir, f'mat_frame_{loop_num}_{optimize_part}_{epoch:04d}.png')
                            save_image(all_image, mat_frame_path)
                            mat_frames.append(mat_frame_path)
                            

                        if early_stopping.early_stop:
                            print("Early stopping")
                            if 'a' in optimize_part:
                                mat['albedo'] = saver.best_albedo
                            if 'r' in optimize_part:
                                mat['roughness'] = saver.best_roughness
                            if 'm' in optimize_part:
                                mat['metallic'] = saver.best_metallic
                            if 'n' in optimize_part:
                                mat['normal'] = saver.best_normal

                            break
                    torch.cuda.empty_cache()
                    gc.collect()
                mat['albedo'] = saver.best_albedo
                mat['roughness'] = saver.best_roughness
                mat['metallic'] = saver.best_metallic
                mat['normal'] = saver.best_normal
                if saver.best_brdfnet_weight is not None:
                    brdf_net.load_state_dict(saver.best_brdfnet_weight)

                saver.save_results(os.path.join(output_dir, f'best_results'))
        # optimize_order = ['arm']

        
    # After all optimization is done, create videos from collected frames
    if env_frames:
        create_video_from_frames(env_frames, os.path.join(output_dir, 'env_optimization.mp4'), fps=10)
    
    if mat_frames:
        create_video_from_frames(mat_frames, os.path.join(output_dir, 'mat_optimization.mp4'), fps=10)

# Add this helper function to create videos from frames
def create_video_from_frames(frame_paths, output_path, fps=10):
    if not frame_paths:
        print(f"No frames found to create video: {output_path}")
        return
    try:
        print(f"Creating video from {len(frame_paths)} frames: {output_path}")
        frames = [imageio.imread(path) for path in tqdm(frame_paths, desc="Loading frames")]
        imageio.mimwrite(output_path, frames, format='ffmpeg', fps=fps, quality=8)
        print(f"Video saved to: {output_path}")
    except Exception as e:
        print(f"Error creating video: {str(e)}")

def countdown(seconds):
    while seconds:
        mins, secs = divmod(seconds, 60)
        timeformat = '{:02d}:{:02d}'.format(mins, secs)
        print(timeformat, end='\r')
        time.sleep(1)
        seconds -= 1
    print('00:00')

def inverse_image(img_inverse_path, save_name, opt_src, opt_order, use_mask, opt_env_from, save_path=None):
    print(f'Inverse image {img_inverse_path}')
    spp=64
    use_sh = False
    camera = json.load(
            open(os.path.join(global_config.BASE_DIR, 'myutils','default_cam.json')))
    c2w = torch.FloatTensor(camera['to_world'])[0][:3, :]
    cam_cfg = {'to_world': c2w, 'fov': camera['x_fov'][0]}
    model_name = 'pos_mlp'
    
    # Get output directory
    output_dir = get_output_dir(save_name, save_path)
    
    # Ensure output directories exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, 'best_results'))
    
    img_inverse = mi.Bitmap(img_inverse_path)
    img_inverse = center_crop_and_resize(np.array(img_inverse), (512, 512), return_tensor=False)
    if not img_inverse_path.endswith('.exr'):
        warnings.warn('The input image is in PNG/JPG format, assume it is sRGB, will convert to linear', UserWarning)
        img_inverse = srgb_to_linear(img_inverse)
    
    if opt_src != 'skip' or opt_order != ['skip']:
        model_path = hf_hub_download(
            repo_id="Lez/MatNet",
            filename="matnet_weights.pth",
            repo_type="model"
        )
        matnet = MaterialNet(encoder='vitb', features=128, out_channels=[96, 192, 384, 768], use_bn=False, use_clstoken=False)
        matnet.load_state_dict(torch.load(model_path, weights_only=True))
        matnet = matnet.cuda()
        pred_mat = matnet.infer_image(img_inverse)
        albedo = pred_mat['albedo']
        normal = pred_mat['normal']
        roughness = pred_mat['roughness'] 
        metallic = pred_mat['metallic'] 
        depth = pred_mat['depth']

        mat = {}
        
        mat['gt_image'] = torch.from_numpy(img_inverse).cuda()
        mat['albedo'] = torch.from_numpy(albedo).cuda().clamp(0,1)
        mat['normal'] = torch.from_numpy(normal).cuda()
        mat['roughness'] = torch.from_numpy(roughness).unsqueeze(-1).cuda().clamp(0.07,1)
        mat['metallic'] = torch.from_numpy(metallic).unsqueeze(-1).cuda().clamp(0,1)
        mat['depth'] = torch.from_numpy(depth).unsqueeze(-1).cuda()

        mi.util.write_bitmap(os.path.join(output_dir,'albedoPred.exr'), albedo)
        mi.util.write_bitmap(os.path.join(output_dir,'normalPred.exr'), normal)
        mi.util.write_bitmap(os.path.join(output_dir,'roughnessPred.png'), roughness)
        mi.util.write_bitmap(os.path.join(output_dir,'metallicPred.png'), metallic)
        mi.util.write_bitmap(os.path.join(output_dir,'depthPred.exr'), depth)
        mi.util.write_bitmap(os.path.join(output_dir,'gt_image.exr'), img_inverse)
        mi.util.write_bitmap(os.path.join(output_dir,'gt_image.png'), img_inverse)
        config = {
            'img_path': img_inverse_path,
            'save_name': save_name,
            'opt_src': opt_src,
            'opt_order': opt_order,
            'use_mask': use_mask,
            'opt_env_from': opt_env_from,
            'model_name': model_name,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        # Save additional parameters about the scene and optimization
        config['image_size'] = img_inverse.shape[:2]
        config['spp'] = spp
        config['output_type'] = 'armn' if 'n' in str(opt_order) else 'arm'
        config['use_mesh_normal'] = not ('n' in str(opt_order))

        # Write configuration to JSON file
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)

        print(f"Configuration saved to {os.path.join(output_dir, 'config.json')}")

        if use_mask:
            mask_root_dir = os.path.join(output_dir, 'best_results')
            if os.path.exists(os.path.join(mask_root_dir, 'mask.png')):
                mask = plt.imread(os.path.join(mask_root_dir, 'mask.png'))
                mask = torch.tensor(np.array(mask)).bool().cuda()[...,0]
                mat['mask'] = mask
            else:
                warnings.warn('No mask found, Do you want to continue without mask?', UserWarning)
                countdown(20)
                use_mask = False
        
        mesh_path = os.path.join(output_dir,f'{save_name}.ply')
        mesh_mask_path = os.path.join(output_dir,'mesh_mask.png')
        if os.path.exists(mesh_mask_path):
            mesh_mask = plt.imread(mesh_mask_path)
            mesh_mask = np.array(mesh_mask, dtype=np.bool_)
            if mesh_mask.ndim > 2:  # If it's an RGB image, use only the first channel
                mesh_mask = mesh_mask[..., 0]
        if not os.path.exists(mesh_path):
            depth = 2 * depth.max() - depth
            if os.path.exists(mesh_mask_path):
                depth[mesh_mask] = 0
                print(f"Applied mask from {mesh_mask_path} to depth map")
            mesh, b_points  = depth_file_to_mesh(depth,cameraMatrix=None, minAngle=6, sun3d=False, depthScale=1.0)
            mesh = rotate_mesh_around_x(mesh, 180)
            o3d.io.write_triangle_mesh(mesh_path, mesh)

        if opt_env_from > 1:
            opt_envmap_path = os.path.join(output_dir,'best_results','envmap.hdr')
            if os.path.exists(opt_envmap_path):
                print(f'Load envmap from {opt_envmap_path}')
                mat['gt_envmap'] = torch.from_numpy(np.array(mi.Bitmap(opt_envmap_path))).cuda()
            else:
                print(f'No envmap found in {opt_envmap_path}, will use envmap=1 instead')

    else:
        print('Load Pre Opted Brdf')
        mesh_path = os.path.join(output_dir,f'{save_name}.ply')
        opted_albedo = np.array(mi.Bitmap(os.path.join(output_dir,'best_results','albedo.exr')),dtype=np.float32)
        opted_roughness = np.array(mi.Bitmap(os.path.join(output_dir,'best_results','roughness.exr')),dtype=np.float32)
        opted_metallic = np.array(mi.Bitmap(os.path.join(output_dir,'best_results','metallic.exr')),dtype=np.float32)
        opted_normal = np.array(mi.Bitmap(os.path.join(output_dir,'best_results','normal.exr')),dtype=np.float32)
        mat = {}
        mat['albedo'] = torch.from_numpy(opted_albedo).cuda().clamp(0,1)
        mat['roughness'] = torch.from_numpy(opted_roughness).unsqueeze(-1).cuda().clamp(0.07,1)
        mat['metallic'] = torch.from_numpy(opted_metallic).unsqueeze(-1).cuda().clamp(0,1)
        mat['normal'] = torch.from_numpy(opted_normal).cuda()
        mat['gt_image'] = torch.from_numpy(img_inverse).cuda()

    if 'n' in str(opt_order):
        use_mesh_normal = False
        output_type = 'armn'
        print('Use normal map')
    else:
        use_mesh_normal = True
        output_type = 'arm'
        print('Use mesh normal')
    scene = load_estimated_mesh(mesh_path,use_mesh_normal)
    optimize_envmap_ARMN(cam_cfg=cam_cfg,scene=scene,
                    save_folder=save_name,
                    mat=mat,use_mesh_normal=use_mesh_normal,
                    output_type=output_type,optimize_order=opt_order,
                    use_gt_scene=False,
                    model_name=model_name,
                    spp=spp,
                    opt_env_from=opt_env_from,
                    opt_src=opt_src,
                    use_mask=use_mask,
                    save_path=save_path)
def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=('inverse a given image'))
    parser.add_argument('--img_inverse_path', required=True,type=str, help='img_inverse_path')
    parser.add_argument('--save_name',required=True,type=str,help='save_name')
    parser.add_argument('--opt_src',required=True,type=str,default='arm',help='if use predicted albedo,roughness,metallic to optimize')
    parser.add_argument('--opt_order',required=False, nargs='+',default=['arm'],help='optimize order')
    parser.add_argument('--use_mask',required=False,action='store_true',help='use mask')
    parser.add_argument('--opt_env_from', required=False,default=0,type=int,help='start env opt from n-th round')
    parser.add_argument('--save_path', required=False, default=None, type=str, help='path to save results')
    parser.add_argument('--model_name', required=False, default='pos_mlp', type=str, choices=['pos_mlp', 'none'], 
                        help='model to use for optimization (pos_mlp or none)')
    return parser.parse_args()



def inverse_real(args):
    img_path = args.img_inverse_path
    save_name = args.save_name
    opt_src = args.opt_src
    opt_order = args.opt_order
    save_path = args.save_path
    inverse_image(img_path, save_name, opt_src, opt_order, 
                 use_mask=args.use_mask, 
                 opt_env_from=args.opt_env_from,
                 save_path=save_path)


if __name__ == '__main__':
    args = parse_args()
    inverse_real(args)