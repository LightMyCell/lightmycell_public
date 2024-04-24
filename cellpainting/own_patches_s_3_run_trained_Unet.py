#import sys
#sys.path.append('/home/local-admin/lightmycells/Label_free_cell_painting')

import os
from Networks.own_all_organelles_patches_unet2d import UNet
import torch
import torch.optim 
import matplotlib
matplotlib.use('Agg')
import cv2
import numpy as np
from PIL import Image
import sys, glob
import re
from aicsimageio import AICSImage
from torch.utils.data import DataLoader
import tables
from Utils.trainUtils import LoadingDatasetTest
from scipy import ndimage

# Load the saved model weights e.g. 28th epoch of trained WGAN model
dataname = "patches_s_3"                            # naming of save states
epoch_num = 44                                       # what epoch to load
pytable_name = "study_patches_d"                    # name of pytable with test data (without train/valid/test phase in name)
checkpoint = torch.load(f'./checkpoints/Unet/{dataname}_epoch_{epoch_num}_Unet.pth') 
output_path = "./result/Unet"

# Network/Training Parameters (copied from training)
ignore_index = 0 
gpuid=0
n_classes= 4
in_channels= 1
padding= True
depth= 6
wf= 5 
up_mode= 'upconv' 
batch_norm = False
batch_size=1
patch_size=256
edge_weight = 1.1 
phases = ["train","val"] 
validation_phases= ["val"] 

# Specify if we should use a GPU (cuda) or only the CPU
if(torch.cuda.is_available()):
    print(torch.cuda.get_device_properties(gpuid))
    torch.cuda.set_device(gpuid)
    device = torch.device(f'cuda:{gpuid}')
else:
    device = torch.device(f'cpu')

# Define the network
Gen = UNet(n_classes=n_classes, in_channels=in_channels, padding=padding,depth=depth,wf=wf, up_mode=up_mode, batch_norm=batch_norm).to(device)
print(f"total params: \t{sum([np.prod(p.size()) for p in Gen.parameters()])}")
Gen.load_state_dict(checkpoint['model_dict'])
Gen.eval()

# Define empty arrays
checkfull = {}
    
batch_size=1
dataset={} 
dataLoader={}

dataset["test"]=LoadingDatasetTest(f"pytables/{pytable_name}_test.pytable")                    # test !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
dataLoader["test"]=DataLoader(dataset["test"], batch_size=batch_size, 
                                    shuffle=False, num_workers=0, pin_memory=False)

for ii, (X, y) in enumerate(dataLoader["test"]): #for each of the batches
    x_in = X
    x_in = x_in.to(device)
    prediction1 = Gen(x_in)
    for channel in range(n_classes): 
        checkfull = prediction1[0,channel,:,:]
        checkfull_cpu = checkfull.cpu()
        prediction = checkfull_cpu.detach().numpy()
        gt = y[0][channel].numpy()

        gt_height, gt_width = gt.shape[-2:]                          # max output size is 1183x1183, if the gt is larger, resize pred to same size
        if (prediction.shape[-1] < gt_width) or (prediction.shape[-2] < gt_height):
            pred = Image.fromarray(prediction)
            prediction = np.array(pred.resize((gt_width, gt_height)))
        
        directory_im = output_path
        os.makedirs(directory_im, exist_ok=True)
        img = Image.fromarray(prediction)
        img.save(f"{directory_im}/Unet_epoch_{epoch_num}_channel_{channel}_img_{ii}_{dataname}.tif")

       
        directory_gt = output_path
        os.makedirs(directory_gt, exist_ok=True)
        gtim = Image.fromarray(gt)
        gtim.save(f"{directory_gt}/Unet_gt_epoch_{epoch_num}_channel_{channel}_img_{ii}_{dataname}.tif")

