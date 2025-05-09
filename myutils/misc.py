import copy
import torch
import mitsuba as mi
import os
import numpy as np
import global_config
import torch.nn.functional as F


def center_crop_and_resize(exr_array, target_size=(518, 518),return_tensor=False):
    h, w, c = exr_array.shape

    min_dim = min(h, w)
    start_h = (h - min_dim) // 2
    start_w = (w - min_dim) // 2

    cropped_array = exr_array[start_h:start_h + min_dim, start_w:start_w + min_dim, :3]

    if exr_array.dtype == np.uint8:
        cropped_array = cropped_array.astype(np.float32) / 255.0
    elif exr_array.dtype == np.float32 or exr_array.dtype == np.float16:
        cropped_array = cropped_array.astype(np.float32)
    else:
        raise ValueError('Unsupported data type, only uint8 and float16/32 are supported.')

    cropped_tensor = torch.from_numpy(cropped_array).permute(2, 0, 1).unsqueeze(0).to(torch.float32)  # (1, C, H, W)

    resized_tensor = F.interpolate(cropped_tensor, size=target_size, mode='bilinear', align_corners=True)

    if return_tensor:
        return resized_tensor
    else:
        resized_array = resized_tensor.squeeze(0).permute(1, 2, 0).numpy()
        return resized_array
    

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                               This should be a percentage (e.g., 0.01 for 1%).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss * (1 - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class SaveBest:
    def __init__(self,save_type=['envmap','albedo','roughness','metallic','normal','rendered_img','brdfnet_weight']):
        self.best_loss = float('inf')
        self.best_albedo = None
        self.best_roughness = None
        self.best_metallic = None
        self.best_envmap = None
        self.rendered_img = None
        self.best_normal = None
        self.save_type = save_type
        self.best_brdfnet_weight = None

    def update(self, loss, albedo, roughness, metallic,normal, envmap,rendered_img,brdfnet_weights=None):
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_albedo = self._detach_and_clone(albedo)
            self.best_roughness = self._detach_and_clone(roughness)
            self.best_metallic = self._detach_and_clone(metallic)
            self.best_envmap = self._detach_and_clone(envmap)
            self.rendered_img = self._detach_and_clone(rendered_img)
            self.best_normal = self._detach_and_clone(normal)
            if brdfnet_weights is not None:
                self.best_brdfnet_weight = copy.deepcopy(brdfnet_weights)
            # print(f"Updated best loss: {loss}")

    def _detach_and_clone(self, tensor):

        if isinstance(tensor, torch.Tensor):
            return tensor.detach().clone()  # Remove gradients, transfer to CPU, deep copy
        else:
            # If it is not a tensor (such as a pure python object), directly copy in depth
            return copy.deepcopy(tensor)

    def get_best(self):
        return {'envmap': self.best_envmap, 'albedo': self.best_albedo, 'roughness': self.best_roughness,
                'metallic': self.best_metallic,'normal':self.best_normal,'rendered_img':self.rendered_img}

    def save_results(self, path):
        if 'envmap' in self.save_type:
            mi.util.write_bitmap(f'{path}/envmap.hdr', self.best_envmap.cpu().numpy())
        if 'albedo' in self.save_type:
            mi.util.write_bitmap(f'{path}/albedo.exr', self.best_albedo.cpu().numpy())
        if 'roughness' in self.save_type:
            mi.util.write_bitmap(f'{path}/roughness.exr', self.best_roughness.cpu().numpy())
        if 'metallic' in self.save_type:
            mi.util.write_bitmap(f'{path}/metallic.exr', self.best_metallic.cpu().numpy())
        if 'rendered_img' in self.save_type:
            mi.util.write_bitmap(f'{path}/rendered_img.exr', self.rendered_img.cpu().numpy())
        if 'normal' in self.save_type:
            mi.util.write_bitmap(f'{path}/normal.exr', self.best_normal.cpu().numpy())


def get_mat(root_dir,file_name,resize=True):
    if 'albedo' in file_name.keys():
        albedo = mi.Bitmap(os.path.join(root_dir, f'{file_name["albedo"]}.exr'))
    if 'roughness' in file_name.keys():
        roughness = mi.Bitmap(os.path.join(root_dir, f'{file_name["roughness"]}.exr'))
    if 'normal' in file_name.keys():
        normal = mi.Bitmap(os.path.join(root_dir, f'{file_name["normal"]}.exr'))
    if 'depth' in file_name.keys():
        depth = mi.Bitmap(os.path.join(root_dir, f'{file_name["depth"]}.exr'))
    if 'metallic' in file_name.keys():
        metallic = mi.Bitmap(os.path.join(root_dir, f'{file_name["metallic"]}.exr'))
    if 'material' in file_name.keys():
        material = mi.Bitmap(os.path.join(root_dir, f'{file_name["material"]}.exr'))
        roughness = np.array(material)[...,:1]
        metallic = np.array(material)[...,1:2]
    if 'gt' in file_name.keys():
        gt_image = mi.Bitmap(os.path.join(root_dir, f'{file_name["gt"]}.exr'))
    if resize:
        albedo = center_crop_and_resize(np.array(albedo),(512,512))
        roughness = center_crop_and_resize(np.array(roughness),(512,512))
        normal = center_crop_and_resize(np.array(normal),(512,512))
        depth = center_crop_and_resize(np.array(depth),(512,512))
        metallic = center_crop_and_resize(np.array(metallic),(512,512))
        gt_image = center_crop_and_resize(np.array(gt_image),(512,512))

    albedo = torch.from_numpy(np.array(albedo)).float().cuda()
    roughness = torch.from_numpy(np.array(roughness)).float().cuda()[...,:1]
    normal = torch.from_numpy(np.array(normal)).float().cuda()
    depth_ori = torch.from_numpy(np.array(depth)).float().cuda()
    metallic = torch.from_numpy(np.array(metallic)).float().cuda()[...,:1]
    gt_image = torch.from_numpy(np.array(gt_image)).float().cuda()
    if 'envmap_id' in file_name.keys():
        gt_envmap = mi.Bitmap(os.path.join(global_config.BASE_DIR,'envmap', f'{file_name["envmap_id"]}.hdr'))
        gt_envmap = torch.from_numpy(np.array(gt_envmap)).float().cuda()
    else:
        gt_envmap = None

    mat = {
        'albedo': albedo,
        'roughness': roughness,
        'normal': normal,
        'depth': depth_ori,
        'metallic': metallic,
        'gt_envmap': gt_envmap,
        'gt_image': gt_image
    }
    return mat


def srgb_to_linear(image):
    linear_image = image ** 2.2
    return linear_image

def linear_to_srgb(image):

    srgb_image = image ** (1.0 / 2.2)
    return srgb_image