#import sys
#sys.path.append('/home/local-admin/lightmycells/Label_free_cell_painting')

import torch
import torch.optim 
from torch.utils.data import DataLoader
import h5py
from Networks.own_mit_nuc_PC_unet2d import UNet
from Networks.own_mit_nuc_PC_patchdiscrim2d import Discriminator
from Utils.GANLoss import GenLoss
import matplotlib
matplotlib.use('Agg')
import numpy as np
from numpy import save
from torch.utils.tensorboard import SummaryWriter
import time
import math
import tables
from torch.autograd import Variable
import torch.autograd as autograd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import sys
from Utils.trainUtils import Dataset

fresh_train = True
epoch_num = 283
dataname="PC_mito_nuc" 
state_save_path = './checkpoints/GAN/'
pytable_name = "study_PC_mito_nuc_6a"
epoch_path_gen = f"{state_save_path}{dataname}_epoch_{epoch_num}_GEN.pth"
epoch_path_disc  = f"{state_save_path}{dataname}_epoch_{epoch_num}_DISC.pth"

ignore_index = 0 # his value won't be included in the loss calculation (output image value)- e.g. 0 is good for this data.
gpuid=0

#Unet params
n_classes= 2    #output channels (fluorescent)
in_channels= 6  #input channels (brightfield)
padding= True   #should levels be padded
depth= 6     #depth of the network 
wf= 5           #wf (int): number of filters in the first layer is 2**wf, was 6
up_mode= 'upconv' #upsample or interpolation 
batch_norm = False #sbatch normalization between the layers

#Training params
batch_size=2
patch_size=256
num_epochs = 500
edge_weight = 1.1 
phases = ["train","valid","test"] 
validation_phases= ["valid"] 


#specify if we should use a GPU (cuda) or only the CPU
if(torch.cuda.is_available()):
    print(torch.cuda.get_device_properties(gpuid))
    torch.cuda.set_device(gpuid)
    device = torch.device(f'cuda:{gpuid}')
else:
    device = torch.device(f'cpu')

Gen = UNet(n_classes=n_classes, in_channels=in_channels, padding=padding,depth=depth,wf=wf, up_mode=up_mode, batch_norm=batch_norm).to(device)
#print(f"total params: \t{sum([np.prod(p.size()) for p in Gen.parameters()])}")

Disc = Discriminator().to(device)
gen_criterion = GenLoss()

tables.file._open_files.close_all()

dataset={} 
#dataset2={}
dataLoader={}
#dataLoader2={}
for phase in phases: #now for each of the phases, we're creating the dataloader
                      #interestingly, given the batch size, i've not seen any improvements from using a num_workers>0
     #f = h5py.File("./Data/study_5_pytable_file.pytable")
     #f.close()
     dataset[phase]=Dataset(f"{pytable_name}_{phase}.pytable")
     dataLoader[phase]=DataLoader(dataset[phase], batch_size=batch_size, 
                                 shuffle=True, num_workers=0, pin_memory=False)
     tables.file._open_files.close_all() 

optimizerG = torch.optim.Adam(Gen.parameters(),lr=.0002, betas =(0.,0.9))
optimizerD = torch.optim.Adam(Disc.parameters(),lr=0.0002, betas=(0.,0.9), weight_decay=0.001)

nclasses = dataset["train"].numpixels.shape[1]

#Generator loss function
gen_criterion = GenLoss()

writer=SummaryWriter() 
best_loss_on_test = np.Infinity
edge_weight=torch.tensor(edge_weight).to(device)
start_time = time.time()

#Resume training from saved checkpoints - load below

#e.g. resume from epoch 30

if fresh_train:
    start_epoch = 0
    print('Start Epoch: ', start_epoch)
else:
    checkpoint = torch.load(epoch_path_gen)
    checkpoint2 = torch.load(epoch_path_disc)
    Gen.load_state_dict(checkpoint['model_dict'])
    Disc.load_state_dict(checkpoint2['model_dict'])
    start_epoch = checkpoint['epoch']
    print('Start Epoch: ', start_epoch)

#Save some variables e.g. MAE, SSIM etc
#The blank arrays are defined in file called 'make_variable_table.py'

SSIM_train, DICE_train, MAE_train, MSE_train, PSNR_train, LOSS_train = {}, {}, {}, {}, {}, {}
SSIM_val, DICE_val, MAE_val, MSE_val, PSNR_val, LOSS_val = {}, {}, {}, {}, {}, {}

def PSNR(im1, im2):
    im1 = im1.astype(np.float64) / 255
    im2 = im2.astype(float) / 255
    mse = np.mean((im1 - im2)**2)
    return 10*math.log10(1. / mse)


def calculate_gradient_penalty(real_images, fake_images):
        eta = torch.FloatTensor(batch_size,1,1,1).uniform_(0,1)
        eta = eta.expand(batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
        eta = eta.to(device)
        interpolated = eta * real_images + ((1 - eta) * fake_images)
        
        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)
        interpolated = interpolated.to(device)

        # calculate probability of interpolated examples
        prob_interpolated = Disc(interpolated)
        prob_interpolated = prob_interpolated.to(device)
        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                               create_graph=True, retain_graph=True)[0]
        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return grad_penalty

loss_values, loss_values_val = [], []
running_loss, running_loss_val = 0.0, 0.0
for epoch in range(start_epoch, num_epochs):
    #zero out epoch based performance variables 
    all_acc = {key: 0 for key in phases} 
    all_loss = {key: torch.zeros(0).to(device) for key in phases}
    cmatrix = {key: np.zeros((2,2)) for key in phases}
    for ii , (X, y) in enumerate(dataLoader["train"]): #for each of the batches
        optimizerD.zero_grad()
        y = y.to(device)
        X = X.to(device) 
        real_imgs = y
        fake_imgs = Gen(X)            
        real_concat_with_input = torch.cat((real_imgs, X),1)
        fake_concat_with_input = torch.cat((fake_imgs, X),1)
        
        # Update Discriminator                 
        real_out = Disc(real_concat_with_input).mean() # real images in disc
        fake_out = Disc(fake_concat_with_input).mean() # fake images in disc
        gradient_penalty = calculate_gradient_penalty(real_concat_with_input, fake_concat_with_input)
        
        # Compute W-div gradient penalty
        print('gp: ',gradient_penalty)
        was_loss = (fake_out - real_out) + 10*gradient_penalty
        was_loss.create_graph = True
        print('was loss: ',was_loss)
        was_loss.backward(retain_graph=True)
        optimizerD.step()
        
        # Update Generator
        optimizerG.zero_grad()
        
        if ii % 5 == 0:
            fake_imgs = Gen(X)
            fake_concat_with_input = torch.cat((fake_imgs, X),1)
            fake_out = Disc(fake_concat_with_input).mean()
            g_loss = gen_criterion(fake_out, fake_imgs, real_imgs, epoch)
            g_loss.backward()
            optimizerG.step()
        
        if ii % 10 == 0:
            # save metrics to their arrays every 10 updates
            # save model every 10 updates             
            
            state = {'epoch' : epoch +1,
            'model_dict': Gen.state_dict(),
            'optim_dict': optimizerG.state_dict(),
            'best_loss_on_test': all_loss,
            'n_classes': n_classes,
            'in_channels': in_channels,
            'padding': padding,
            'depth': depth,
            'wf': wf,
            'up_mode': up_mode, 'batch_norm': batch_norm}
            
            state2 = {'epoch': epoch + 1,
            'model_dict': Disc.state_dict(),
            'optim_dict': optimizerD.state_dict(),
            'best_loss_on_test': all_loss,
            'n_classes': n_classes,
            'in_channels': in_channels,
            'padding': padding,
            'depth': depth,
            'wf': wf,
            'up_mode': up_mode, 'batch_norm': batch_norm}
            
            for channel in range(n_classes):
            
                hold_pred = fake_imgs[0,channel,:,:]
                prediction = hold_pred.cpu().detach().numpy()
                hold_gt = y[0,channel,:,:]
                ground_truth = hold_gt.cpu().detach().numpy()
                ssim_train = ssim(prediction, ground_truth, data_range=1)           # Change these to the competition's
                mse_train = mean_squared_error(prediction, ground_truth)
                mae_train = mean_absolute_error(prediction, ground_truth)
                psnr_train = PSNR(prediction, ground_truth)
                print('TRAINING: SSIM: ',ssim_train, ' MSE: ',mse_train, ' MAE: ',mae_train, ' PSNR: ', psnr_train)
#                SSIM_train[channel] = np.append(SSIM_train[channel], ssim_train)
#                save(f'SSIM_train_{channel}.npy', SSIM_train[channel])
#                MSE_train[channel] = np.append(MSE_train[channel], mse_train)
#                save(f'MSE_train_{channel}.npy', MSE_train[channel])
#                MAE_train[channel] = np.append(MAE_train[channel], mae_train)
#                save(f'MAE_train_{channel}.npy', MAE_train[channel])
#                PSNR_train[channel] = np.append(PSNR_train[channel], psnr_train)
#                save(f'PSNR_train_{channel}.npy', PSNR_train[channel])
    
    if epoch % 100 == 0:
        torch.save(state2, f"{state_save_path}{dataname}_epoch_{epoch}_DISC.pth")
        torch.save(state, f"{state_save_path}{dataname}_epoch_{epoch}_GEN.pth")
    
    for jj , (X, y) in enumerate(dataLoader["valid"]):
        Gen.eval()
        Disc.eval()
        y = y.to(device)
        X = X.to(device)
        prediction1 = Gen(X)
        fake_concat_with_input1 = torch.cat((prediction1, X),1)
        fake_out1 = Disc(fake_concat_with_input1).mean()
        g_loss_val = gen_criterion(fake_out1, prediction1, y, epoch)
        loss_val = g_loss_val.cpu().detach().numpy()
        LOSS_val = np.append(LOSS_val, loss_val)
        save(f'{state_save_path}LOSS_val.npy', LOSS_val)
                
        for channel in range(n_classes):
            hold_pred = prediction1[0,channel,:,:]
            prediction = hold_pred.cpu().detach().numpy()
            hold_gt = y[0,channel,:,:]
            ground_truth = hold_gt.cpu().detach().numpy() 
            ssim_val = ssim(prediction, ground_truth, data_range=1)                         # Change these to the competition's
            mse_val = mean_squared_error(prediction, ground_truth)
            mae_val = mean_absolute_error(prediction, ground_truth)
            psnr_val = PSNR(prediction, ground_truth)   
            print('VALIDATION: SSIM: ',ssim_val, ' MSE: ',mse_val, ' MAE: ',mae_val, ' PSNR: ', psnr_val)
#            SSIM_val[channel] = np.append(SSIM_val[channel], ssim_val)
#            save(f'SSIM_val_{channel}.npy', SSIM_val[channel])
#            MSE_val[channel] = np.append(MSE_val[channel], mse_val)
#            save(f'MSE_val_{channel}.npy', MSE_val[channel])
#            MAE_val[channel] = np.append(MAE_val[channel], mae_val)
#            save(f'MAE_val_{channel}.npy', MAE_val[channel])
#            PSNR_val[channel] = np.append(PSNR_val[channel], psnr_val)
#            save(f'PSNR_val_{channel}.npy', PSNR_val[channel])
