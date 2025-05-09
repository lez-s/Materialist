from myutils.mi_plugin import MatrefractBSDF,MatDiffBSDF,load_estimated_brdf,TransBSDF
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")
import os
import global_config
from tqdm import tqdm
import argparse
import numpy as np
import torch
from torchvision.utils import save_image
from myutils.misc import *
import drjit as dr
from skimage.color import rgb2hsv, hsv2rgb
dr.set_flag(dr.JitFlag.VCallRecord, False)
dr.set_flag(dr.JitFlag.LoopRecord, False)

def load_estimated_mesh_w_env(mesh_path,env_path,use_mesh_normal=True,bsdf = {'name':'matbsdf'}):
    '''
    mat_dir: directory of estimated material
    '''
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
    # breakpoint()
    if bsdf['name'] == 'matbsdf' or bsdf['name'] == 'matDiffBSDF':
        bsdf_name = bsdf['name']
        if bsdf_name == 'matDiffBSDF':
            mi.register_bsdf(bsdf_name, lambda props: MatDiffBSDF(props))
        
        scene = mi.load_dict({
            'type': 'scene',
            'shape':{
                'type': 'ply',
                'filename': mesh_path,
                'bsdf': {'type': bsdf_name,'cam_meta':cam_cfg_path,'use_mesh_normal':use_mesh_normal},
            },
            'integrator': {
                'type': 'path',
                'max_depth': 4,
            },
            'sensor': camera,
            'emitter': {'type': 'envmap','filename': env_path}
        })
    elif bsdf['name'] == 'matrefractBSDF':
        mi.register_bsdf("matrefractBSDF", lambda props: MatrefractBSDF(props))
        bsdf_name = bsdf['name']
        ior = bsdf['ior']
        keep_albedo_color = bsdf['keep_albedo_color']
        mat_dir = bsdf['mat_dir']
        scene = mi.load_dict({
            'type': 'scene',
            'shape':{
                'type': 'ply',
                'filename': mesh_path,
                'bsdf': {'type': bsdf_name,'cam_meta':cam_cfg_path,'mat_dir':mat_dir,
                         'use_mesh_normal':use_mesh_normal,'ior':ior,'keep_albedo_color':keep_albedo_color},
            },
            'integrator': {
                'type': 'path',
                'max_depth': 4,
            },
            'sensor': camera,
            'emitter': {'type': 'envmap','filename': env_path}
        })
    elif bsdf['name'] == 'TransBSDF':
        mi.register_bsdf("TransBSDF", lambda props: TransBSDF(props))
        bsdf_name = bsdf['name']
        ior = bsdf['ior']
        keep_albedo_color = bsdf['keep_albedo_color']
        scene = mi.load_dict({
            'type': 'scene',
            'shape':{
                'type': 'ply',
                'filename': mesh_path,
                'bsdf': {'type': bsdf_name,'cam_meta':cam_cfg_path,
                         'use_mesh_normal':use_mesh_normal,'ior':ior,'keep_albedo_color':keep_albedo_color},
            },
            'integrator': {
                'type': 'path',
                'max_depth': 4,
            },
            'sensor': camera,
            'emitter': {'type': 'envmap','filename': env_path}
        })
    else:
        raise ValueError("Invalid bsdf, must be matbsdf or matrefractBSDF")
    return scene


def load_estimated_mesh_w_env_insert(mesh_path,env_path,mat_dir,use_mesh_normal=True):
    '''
    mat_dir: directory of estimated material
    '''
    mi.register_bsdf("matbsdf", lambda props: matDiffBSDF(props))
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
                'bsdf': {'type': 'matbsdf','cam_meta':cam_cfg_path,'use_mesh_normal':use_mesh_normal},
            },
            'insert_ply':{
                'type': 'ply',
                'filename': os.path.join(os.path.dirname(mat_dir), "oi.ply"),
                'bsdf':{'type':'dielectric','int_ior': 'acrylic glass','ext_ior': 'air'},
            },
            'insert_ply2':{
                'type': 'ply',
                'filename': os.path.join(os.path.dirname(mat_dir), "oi2.ply"),
                'bsdf':{'type':'diffuse','reflectance': {'type': 'rgb','value':[0.8, 0.8, 0.8]}},
            },
            'integrator': {
                'type': 'path',
                'max_depth': 16,
            },
            'sensor': camera,
            'emitter': {'type': 'envmap','filename': env_path}
            })

    return scene

def adj_albedo(albedo,hue_shift):
    hsv_img = rgb2hsv(albedo.cpu().numpy())
    hsv_img = (hsv_img + hue_shift).clip(0,1)
    return torch.from_numpy(hsv2rgb(hsv_img)).float().cuda()

def render_w_mi(mesh_path, env_path, save_name, n_iter=10, edit={'albedo':None,'roughness':None,'metallic':None}, input_path=None, save_path=None):
    if input_path is not None:
        mat_dir = os.path.join(input_path, save_name, 'best_results')
    else:
        mat_dir = os.path.join('output_imgs', save_name, 'best_results')
    
    if 'mn' in save_name:
        use_mesh_normal = False
        print('Use Optimized Normal')
    else:
        use_mesh_normal = True
        print('Use Mesh Normal')
    scene = load_estimated_mesh_w_env(mesh_path,env_path,use_mesh_normal=use_mesh_normal,bsdf={'name':'matDiffBSDF'})
    env_id = env_path.split('/')[-1][:-4]   
    empty_img = np.zeros((512,512,3),dtype=np.float32)
    denoiser = mi.OptixDenoiser(input_size=empty_img.shape[:2], albedo=False, normals=False, temporal=False)
    mat = load_estimated_brdf(mat_dir)
    edit_flag = ''
    for key in edit.keys():
        if edit[key] is not None:
            if 'mask' in mat.keys():
                mask = mat['mask']
                # mat[key][mask] = (mat[key][mask] * edit[key]).clamp(0,1)
                if key == 'albedo':
                    mat[key][mask] = adj_albedo(mat[key][mask],edit[key])
                    edit_flag+=f'_{key[:1]}_{edit[key].tolist()[0,0]}'
                else:
                    mat[key][mask] = edit[key]
                    # mat[key][mask] = mat[key][mask] - edit[key]
                    # mat[key] = mat[key]*0
                    edit_flag+=f'_{key[:1]}_{edit[key]}'
            else:
                raise FileNotFoundError('Unable to edit img, no mask found')
    
    mi_params = mi.traverse(scene)
    mi_params['shape.bsdf.a'] = mat['albedo']
    mi_params['shape.bsdf.r'] = mat['roughness'] 
    mi_params['shape.bsdf.m'] = mat['metallic'] 
    # mi_params['shape.bsdf.n'] = mat['normal']
    if env_path == None:
        mi_params['emitter.data'] = mat['envmap']
    use_mesh_normal = True
    if not use_mesh_normal:
        mi_params['shape.bsdf.n'] = mat['normal']
    mi_params.update()
    for i in tqdm(range(n_iter)):
        img = mi.render(scene,spp=64,seed=i)
        img = denoiser(img)
        empty_img += img.numpy()
    img = empty_img/n_iter
    output_dir = os.path.join(save_path if save_path else global_config.OUT_DIR, save_name)
    mi.util.write_bitmap(os.path.join(output_dir, f'mi_{save_name}_{env_id}_{edit_flag}.exr'), img)
    img = linear_to_srgb(img)
    img_torch = torch.from_numpy(img).permute(2,0,1)
    save_image(img_torch, os.path.join(output_dir, f'mi_{save_name}_{env_id}_{edit_flag}.png'))
    print("Wrote file to ", os.path.join(output_dir, f'mi_{save_name}_{env_id}_{edit_flag}.png'))



def render_w_mi_insert(mesh_path, env_path, save_name, n_iter=10, input_path=None, save_path=None):
    if input_path is not None:
        mat_dir = os.path.join(input_path, save_name, 'best_results')
    else:
        mat_dir = os.path.join('output_imgs', save_name, 'best_results')
    
    scene = load_estimated_mesh_w_env_insert(mesh_path, env_path, mat_dir)
    env_id = env_path.split('/')[-1][:-4]
    empty_img = np.zeros((512,512,3),dtype=np.float32)
    denoiser = mi.OptixDenoiser(input_size=empty_img.shape[:2], albedo=False, normals=False, temporal=False)
    mat = load_estimated_brdf(mat_dir)
    mi_params = mi.traverse(scene)
    mi_params['shape.bsdf.a'] = mat['albedo']
    mi_params['shape.bsdf.r'] = mat['roughness']
    mi_params['shape.bsdf.m'] = mat['metallic']
    if env_path == None:
        mi_params['emitter.data'] = mat['envmap']
    mi_params.update()

    for i in tqdm(range(n_iter)):
        img = mi.render(scene,spp=32,seed=i)
        img = denoiser(img)
        empty_img += img.numpy()

    img = empty_img/n_iter
    output_dir = os.path.join(save_path if save_path else global_config.OUT_DIR, save_name)
    mi.util.write_bitmap(os.path.join(output_dir, f'mi_oi_{save_name}_{env_id}.exr'), img)
    img = linear_to_srgb(img)
    img_torch = torch.from_numpy(img).permute(2,0,1)
    save_image(img_torch, os.path.join(output_dir, f'mi_oi_{save_name}_{env_id}.png'))
    print("Wrote file to ", os.path.join(output_dir, f'mi_oi_{save_name}_{env_id}.png'))



def render_real(save_name, env_path=None, edit={'albedo':None,'roughness':None,'metallic':None}, n_iter=1, input_path=None, save_path=None):
    mesh_path = os.path.join(input_path if input_path is not None else global_config.OUT_DIR, save_name, f'{save_name}.ply')
    if env_path == None:
        if input_path is not None:
            input_best_results = os.path.join(input_path, save_name, 'best_results')
            if os.path.exists(os.path.join(input_best_results, 'envmap.hdr')):
                env_path = os.path.join(input_best_results, 'envmap.hdr')
            else:
                default_best_results = os.path.join(global_config.OUT_DIR, save_name, 'best_results')
                if os.path.exists(os.path.join(default_best_results, 'envmap.hdr')):
                    env_path = os.path.join(default_best_results, 'envmap.hdr')
                else:
                    raise ValueError("No envmap found")
        else:
            best_results = os.path.join(global_config.OUT_DIR, save_name, 'best_results')
            if os.path.exists(os.path.join(best_results, 'envmap.hdr')):
                env_path = os.path.join(best_results, 'envmap.hdr')
            else:
                raise ValueError("No envmap found")
    render_w_mi(mesh_path, env_path, save_name, edit=edit, n_iter=n_iter, input_path=input_path, save_path=save_path)


def render_io(save_name, env_path, input_path=None, save_path=None):
    mesh_path = os.path.join(global_config.OUT_DIR, save_name, f'{save_name}.ply')
    if env_path == None:
        if input_path is not None:
            input_best_results = os.path.join(input_path, save_name, 'best_results')
            if os.path.exists(os.path.join(input_best_results, 'envmap_opt.hdr')):
                env_path = os.path.join(input_best_results, 'envmap_opt.hdr')
            elif os.path.exists(os.path.join(input_best_results, 'envmap.hdr')):
                env_path = os.path.join(input_best_results, 'envmap.hdr')
            else:
                default_best_results = os.path.join(global_config.OUT_DIR, save_name, 'best_results')
                if os.path.exists(os.path.join(default_best_results, 'envmap_opt.hdr')):
                    env_path = os.path.join(default_best_results, 'envmap_opt.hdr')
                elif os.path.exists(os.path.join(default_best_results, 'envmap.hdr')):
                    env_path = os.path.join(default_best_results, 'envmap.hdr')
                else:
                    raise ValueError("No envmap found")
        else:
            best_results = os.path.join(global_config.OUT_DIR, save_name, 'best_results')
            if os.path.exists(os.path.join(best_results, 'envmap_opt.hdr')):
                env_path = os.path.join(best_results, 'envmap_opt.hdr')
            elif os.path.exists(os.path.join(best_results, 'envmap.hdr')):
                env_path = os.path.join(best_results, 'envmap.hdr')
            else:
                raise ValueError("No envmap found")
    render_w_mi_insert(mesh_path, env_path, save_name, input_path=input_path, save_path=save_path)

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=('inverse a given image'))
    parser.add_argument('--env_path', required=False,default=None,type=str, help='env_path')
    parser.add_argument('--save_name',required=True,type=str,help='save_name')
    parser.add_argument('--mode',required=True,type=str,help='mode, real or io')
    parser.add_argument('--input_path', required=False, default=None, type=str, help='custom path for material loading')
    parser.add_argument('--save_path', required=False, default=None, type=str, help='custom path for saving rendered images')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    save_name = args.save_name
    env_path = args.env_path
    mode = args.mode
    input_path = args.input_path
    save_path = args.save_path
    # albedo_edit = np.array([[0,-0.8,0]])
    edit = {'albedo':None,'roughness':None,'metallic':None}
    n_iter = 10
    if mode == 'real':
        render_real(save_name, env_path, edit=edit, n_iter=n_iter, input_path=input_path, save_path=save_path)
    elif mode == 'oi':
        render_io(save_name, env_path, input_path=input_path, save_path=save_path)
    else:
        raise ValueError("Invalid mode")
