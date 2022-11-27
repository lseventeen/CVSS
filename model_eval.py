import argparse
from loguru import logger
from data import build_train_loader
from trainer import Trainer
from utils.helpers import seed_torch
from losses import *
from datetime import datetime
import wandb
from configs.config import get_config
from models import build_model
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
import os
import torch.backends.cudnn as cudnn
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import models

model_2d = {
"UNet",
"FR_UNet",
"Att_UNet",
"CSNet",
"UNet_Nested",
"MAA_Net",
"Res_UNet"

}

model_3d = {
"UNet_3D",
"FR_UNet_3D",
"CSNet3D",
"Att_UNet_3D",
"Res_UNet_3D",
"UNet_Nested_3D",
# "PHTrans"
"IPN",
"PSC"

}

def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))

def model_build(model_name):
    if model_name in model_2d:
        return getattr(models, model_name)(
        num_classes = 2,
        num_channels = 8
        )
    elif model_name in model_3d:
        return getattr(models, model_name)(
        num_classes =  2,
        num_channels = 1
        )
    else:
        raise NotImplementedError(f"Unkown model: {model_name}")



  


if __name__ == '__main__':
  
    print_model_parm_nums(model_build("PSC"))
