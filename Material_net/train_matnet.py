import torch
import numpy as np
import torch.optim as optim
import argparse
import random
import os
import importlib
import torchvision.utils as vutils
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
import time
import datetime
from tqdm import tqdm
import yaml
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from Material_net.dpt import MaterialNet
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from pytorch_lightning import seed_everything
from .mydataset import MGDataset
from pytorch_lightning.loggers import WandbLogger
import wandb
import lovely_tensors as lt
import ipdb
from torchvision.utils import save_image,make_grid
lt.monkey_patch()



class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target,valid_mask):
        epsilon = 1e-4
        pred = torch.clamp(pred, min=epsilon)
        target = torch.clamp(target, min=epsilon)
        valid_mask = valid_mask.detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))
        if torch.isnan(loss):
            raise ValueError('nan loss in depth loss')
        return loss
    


class MatNetTrainer(pl.LightningModule):
    def __init__(self, log_per={'train':4,'val':4}):
        super().__init__()
        self.matnet = MaterialNet(encoder='vitb', features=128, out_channels=[96, 192, 384, 768], use_bn=False, use_clstoken=False)
        self.criterion_depth = SiLogLoss(lambd=0.5)
        self.criterion_normal = nn.CosineSimilarity(dim=1, eps=1e-4)
        self.criterion_l1 = nn.L1Loss()
        self.criterion_l2 = nn.MSELoss()
        self.criterion_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex') 
        self.train_step = 0
        self.val_step = 0
        self.min_depth = 0.01
        self.max_depth = 20
        self.log_per = log_per
        self.freeze_layers()

    def pred_mat(self, batch):
        im = batch['im']
        mat_pred = self.matnet(im)
        return mat_pred

    def training_step(self, batch, batch_idx):
        mat_pred = self.pred_mat(batch)

        loss = self.calc_loss(batch,mat_pred)
        loss_total = loss['total']
        loss_depth = loss['depth']
        loss_normal = loss['normal']
        loss_albedo = loss['albedo']
        loss_roughness_l1 = loss['roughness']
        loss_metallic_l1 = loss['metallic']

        wandb.log({'train_loss_total':loss_total,'train_loss_depth': loss_depth, 'train_loss_normal': loss_normal, 
                   'train_loss_albedo': loss_albedo, 'train_loss_roughness': loss_roughness_l1, 'train_loss_metallic': loss_metallic_l1})
        self.log('train_loss_total',loss_total,prog_bar=True)
        if self.train_step%self.log_per['train'] == 0:
            self.log_images(batch,mat_pred,_type='train')
        self.train_step += 1
        return loss_total
    
    def calc_loss(self,batch,pred):
        albedo = batch['albedo']
        roughness = batch['roughness']
        metallic = batch['metallic']
        normal = batch['normal']
        depth = batch['depth']
        depth_pred = pred['depth']
        albedo_pred = pred['albedo']
        roughness_pred = pred['roughness']
        metallic_pred = pred['metallic']
        normal_pred = pred['normal']
        depth_valid_mask = (depth >= self.min_depth) & (depth <= self.max_depth)

        loss_depth = self.criterion_depth(depth_pred, depth,depth_valid_mask)
        loss_normal = 1 - self.criterion_normal(normal_pred, normal).mean() + self.criterion_l1(normal_pred, normal)
        loss_albedo_lpips = self.criterion_lpips(albedo_pred, albedo)
        loss_albedo_l1 = self.criterion_l1(albedo_pred, albedo)
        loss_albedo = loss_albedo_lpips + loss_albedo_l1
        loss_roughness_l1 = self.criterion_l1(roughness_pred, roughness)
        loss_metallic_l1 = self.criterion_l1(metallic_pred, metallic)
        loss_total = loss_depth + loss_normal + loss_albedo + loss_roughness_l1 + loss_metallic_l1

        return {'total':loss_total,'depth':loss_depth,'normal':loss_normal,'albedo':loss_albedo,'roughness':loss_roughness_l1,'metallic':loss_metallic_l1}
    
    def validation_step(self, batch, batch_idx):
        mat_pred = self.pred_mat(batch)
        loss = self.calc_loss(batch,mat_pred)
        loss_total = loss['total']
        loss_depth = loss['depth']
        loss_normal = loss['normal']
        loss_albedo = loss['albedo']
        loss_roughness_l1 = loss['roughness']
        loss_metallic_l1 = loss['metallic']

        wandb.log({'val_loss_total':loss_total,'val_loss_depth': loss_depth, 'val_loss_normal': loss_normal,
                     'val_loss_albedo': loss_albedo, 'val_loss_roughness': loss_roughness_l1, 'val_loss_metallic': loss_metallic_l1})
        
        self.log('val_loss_total',loss_total,prog_bar=True)
        if self.val_step%self.log_per['val'] == 0:
            self.log_images(batch,mat_pred,_type='val')
        
        self.val_step += 1
        return loss_total
    def test_step(self, batch, batch_idx):
        mat_pred = self.pred_mat(batch)
        loss = self.calc_loss(batch,mat_pred)
        loss_total = loss['total']
        loss_depth = loss['depth']
        loss_normal = loss['normal']
        loss_albedo = loss['albedo']
        loss_roughness_l1 = loss['roughness']
        loss_metallic_l1 = loss['metallic']

        wandb.log({'test_loss_total':loss_total,'test_loss_depth': loss_depth, 'test_loss_normal': loss_normal,
                     'test_loss_albedo': loss_albedo, 'test_loss_roughness': loss_roughness_l1, 'test_loss_metallic': loss_metallic_l1})
        
        if batch_idx == 0:
            self.log_images(batch,mat_pred,_type='test')
        return loss_total
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        mat_pred = self.pred_mat(batch)
        loss = self.calc_loss(batch,mat_pred)
        loss_total = loss['total']
        loss_depth = loss['depth']
        loss_normal = loss['normal']
        loss_albedo = loss['albedo']
        loss_roughness_l1 = loss['roughness']
        loss_metallic_l1 = loss['metallic']


    def freeze_layers(self):
        for param in self.parameters():
            param.requires_grad = False

        for param in self.matnet.depth_head.scratch.output_conv2.parameters():
            param.requires_grad = True
        
        for param in self.matnet.material_head.scratch.parameters():
            param.requires_grad = True

    def log_images(self,batch, mat_pred,_type):
        albedo = batch['albedo'][1]
        roughness = batch['roughness'][1].repeat(3,1,1)
        metallic = batch['metallic'][1].repeat(3,1,1)
        normal = batch['normal'][1]
        depth = batch['depth'][1].repeat(3,1,1)
        im = batch['im'][1]
        depth_pred = mat_pred['depth'][1].repeat(3,1,1)
        albedo_pred = mat_pred['albedo'][1]
        roughness_pred = mat_pred['roughness'][1].repeat(3,1,1)
        metallic_pred = mat_pred['metallic'][1].repeat(3,1,1)
        normal_pred = mat_pred['normal'][1]

        all_image = make_grid([albedo,albedo_pred,roughness,roughness_pred,metallic,metallic_pred,normal,normal_pred,depth,depth_pred,im],nrow=4)
        wandb.log({f'{_type}_images': [wandb.Image(all_image)]})

    def configure_optimizers(self):
        params_depth = list(self.matnet.depth_head.scratch.output_conv2.parameters())
        params_material = list(self.matnet.material_head.scratch.parameters())
        params = params_depth + params_material
        optimizer = torch.optim.AdamW(params,lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01)
        return optimizer
    
def get_setting(args,save_per):
    if args.debug:
        fast_dev_run = True
        checkpoint_callback = [ModelCheckpoint(save_top_k=0)]
        args.log_name = 'nolog'
    else:
        fast_dev_run = False
        checkpoint_callback1 = ModelCheckpoint(dirpath=args.model_save_path,
                                              monitor="val_loss_total",filename='%s-{epoch}-{val_loss:.3e}'%args.log_name)
        checkpoint_callback2 = ModelCheckpoint(dirpath=args.model_save_path,every_n_train_steps=save_per,
                                              filename='%s-latest'%args.log_name)
        checkpoint_callback = [checkpoint_callback1,checkpoint_callback2]
        print('Save every %i'%save_per)

    if args.log_name != 'nolog':
        if args.resume is not None:
            wandb_logger = WandbLogger(project="mat_pred",name=args.log_name, 
                                       entity="lez",config=args,resume='allow',id=args.resumeid)
            print('W&B is resume online')
        else:
            id = wandb.util.generate_id()
            wandb.init(project="mat_pred",name=args.log_name,id=id, resume="allow")
            wandb_logger = WandbLogger(project="mat_pred",name=args.log_name, entity="lez",config=args)
            print('W&B is logging online')
    else:
        wandb.init(mode="disabled")
        wandb_logger = WandbLogger(project="mat_pred",name=args.log_name, entity="lez",config=args,offline=True)
        print('W&B is logging offline')

    return fast_dev_run,checkpoint_callback,wandb_logger

class MatNetTrainer2(MatNetTrainer):
    def __init__(self, log_per={'train':4,'val':4}):
        super().__init__(log_per)
        self.unfreezed_layers = [self.matnet.depth_head.scratch.parameters(),
                                 self.matnet.material_head.parameters()]
        

    def freeze_layers(self):
        for param in self.parameters():
            param.requires_grad = False
        # for lays in self.unfreezed_layers:
        #     for param in lays:
        #         param.requires_grad = True
        for param in self.matnet.depth_head.scratch.parameters():
            param.requires_grad = True
        for param in self.matnet.material_head.parameters():
            param.requires_grad = True

    def configure_optimizers(self):
        params_depth = list(self.matnet.depth_head.scratch.parameters())
        params_material = list(self.matnet.material_head.parameters())
        params = params_depth + params_material
        # params = []
        # for lays in self.unfreezed_layers:
        #     params.append(list(lays))
        optimizer = torch.optim.AdamW(params,lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01)
        return optimizer

def get_model(model_name):
    try:
        module = importlib.import_module('train_matnet')
        return getattr(module, model_name)
    except ImportError:
        raise ValueError('model not found')
