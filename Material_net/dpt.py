import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from .dinov2 import DINOv2
from .util.blocks import FeatureFusionBlock, _make_scratch
from .util.transform import Resize, NormalizeImage, PrepareForNet
import warnings

def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.conv_block(x)


class DPTHead(nn.Module):
    def __init__(
        self, 
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        use_clstoken=False,
        output_type='depth',
    ):
        super(DPTHead, self).__init__()
        
        self.use_clstoken = use_clstoken
        self.output_type = output_type
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        head_features_1 = features
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        
        if output_type == 'depth':
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
                nn.ReLU(True),
                nn.Identity(),
            )
        elif output_type == 'material': # Albedo, Roughness, Metallic, Normal
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_2, 8, kernel_size=1, stride=1, padding=0),
            )
        else:
            raise NotImplementedError

    
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        if self.output_type == 'depth':
            out = self.scratch.output_conv2(out)
        elif self.output_type == 'material':
            out = self.scratch.output_conv2(out)
            arm = out[:, :5]
            arm = nn.ReLU()(arm)
            normal = out[:, 5:8]
            normal = nn.Tanh()(normal)
            normal = F.normalize(normal,p=2, dim=1, eps=1e-6)
            out = torch.cat((arm, normal), 1)
            
        return out


class MaterialNet(nn.Module):
    def __init__(
        self, 
        encoder='vitb', 
        features=128, 
        out_channels=[96, 192, 384, 768], 
        use_bn=False, 
        use_clstoken=False
    ):
        super().__init__()
        
        self.intermediate_layer_idx = {
            'vitb': [2, 5, 8, 11], 
        }
        
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)
        
        self.depth_head = DPTHead(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken,output_type='depth')
        self.material_head = DPTHead(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken, output_type='material')
        # breakpoint()

    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        
        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.relu(depth)

        armn = self.material_head(features, patch_h, patch_w)
        albedo = armn[:, :3]
        roughness = armn[:, 3:4]
        metallic = armn[:, 4:5]
        normal = armn[:, 5:8]

        out = {
            'depth': depth,
            'albedo': albedo,
            'roughness': roughness,
            'metallic': metallic,
            'normal': normal
        }
        return out
    
    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518):
        '''
        input: raw_image (np.array), shape: (H, W, 3)
        output: dict, keys: ['depth', 'albedo', 'roughness', 'metallic', 'normal'], values: np.array
        '''
        image, (h, w) = self.image2tensor(raw_image, input_size)
        if image.mean() >= 10:
            warnings.warn('Pixel intensity is too high, input dtype may be wrong. Dividing by 255 to avoid Error.', UserWarning)
            image = image / 255.0
        mat_pred = self.forward(image)
        depth = mat_pred['depth']
        albedo = mat_pred['albedo']
        roughness = mat_pred['roughness']
        metallic = mat_pred['metallic']
        normal = mat_pred['normal']
        
        depth = F.interpolate(depth, (h, w), mode="bilinear", align_corners=True)[0, 0]
        albedo = F.interpolate(albedo, (h, w), mode="bilinear", align_corners=True)[0].permute(1, 2, 0)
        roughness = F.interpolate(roughness, (h, w), mode="bilinear", align_corners=True)[0,0]
        metallic = F.interpolate(metallic, (h, w), mode="bilinear", align_corners=True)[0,0]
        normal = F.interpolate(normal, (h, w), mode="bilinear", align_corners=True)[0].permute(1, 2, 0)
        return {'depth': depth.cpu().numpy(), 'albedo': albedo.cpu().numpy(), 'roughness': roughness.cpu().numpy(), 'metallic': metallic.cpu().numpy(), 'normal': normal.cpu().numpy()}
    
    def image2tensor(self, raw_image, input_size=518):        
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            PrepareForNet(),
        ])
        
        h, w = raw_image.shape[:2]

        if raw_image.dtype == 'uint8':
            image = raw_image / 255.0
        else:
            image = raw_image

        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)
        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        image = image.to(DEVICE)
        return image, (h, w)

