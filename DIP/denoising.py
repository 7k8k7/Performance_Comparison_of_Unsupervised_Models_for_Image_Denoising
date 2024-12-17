from __future__ import print_function
import matplotlib.pyplot as plt

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
from models import *

import imageio
import glob
import torch
import torch.optim

from skimage.measure import compare_psnr
from utils.denoising_utils import *


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
psnr_ns=[]
psnr_gs=[]
n=0
imsize =-1
PLOT = True
sigma = 15
sigma_ = sigma/255.
folder_path = "data/BSDS300"
output_folder = "data/denoised_BSDS300"
 # Define image extensions to look for
image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif", "*.tiff", "*.webp")
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(folder_path, ext)))
print(f"{len(image_files)} images are ready for denoising")
for image_file in image_files:
    print(f"Denoising image: {image_file}")
    img_pil = crop_image(get_image(image_file, imsize)[0], d=32)
    img_np = pil_to_np(img_pil)
    
    img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)
    noisy_img_to_save = np.clip(img_noisy_np, 0, 1)
    if noisy_img_to_save.ndim == 2:  # Grayscale
        noisy_img_uint8 = (noisy_img_to_save * 255).astype(np.uint8)
    elif noisy_img_to_save.ndim == 3 and noisy_img_to_save.shape[0] in [1, 3]:
        # CHW to HWC
        noisy_img_to_save = np.moveaxis(noisy_img_to_save, 0, -1)
        noisy_img_uint8 = (noisy_img_to_save * 255).astype(np.uint8)
    else:
        # Already HWC format
        noisy_img_uint8 = (noisy_img_to_save * 255).astype(np.uint8)

    # Create a folder for noisy images if it doesn't exist
    os.makedirs(os.path.join(output_folder, 'noisy'), exist_ok=True)

    # Save the noisy image
    noisy_image_path = os.path.join(output_folder, 'noisy', os.path.basename(image_file))
    imageio.imwrite(noisy_image_path, noisy_img_uint8)
    #setup
    INPUT = 'noise' # 'meshgrid'
    pad = 'reflection'
    OPT_OVER = 'net' # 'net,input'

    reg_noise_std = 1./34.  # For noise with sigma = 15, set to 1./34. ;set to 1./30. for sigma=25 ;set to 1./20. for sigma=50. 
    LR = 0.01

    OPTIMIZER='adam' # 'LBFGS'
    show_every = 1000
    exp_weight=0.99
    num_iter = 3000
    input_depth = 32 
    figsize = 4 
    
    
    net = get_net(input_depth, 'skip', pad,
                  skip_n33d=128, 
                  skip_n33u=128, 
                  skip_n11=4, 
                  num_scales=5,
                  upsample_mode='bilinear').type(dtype)

    net = net.type(dtype)
    net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()

    # Compute number of parameters
    s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 

    # Loss
    mse = torch.nn.MSELoss().type(dtype)

    img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)

    #optimize
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    out_avg = None
    last_net = None
    psrn_noisy_last = 0
    #noisy_image=net(net_input)
    i = 0
    def closure():
    
        global i, out_avg, psrn_noisy_last, last_net, net_input
    
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)
    
        out = net(net_input)
        # Smoothing
        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)
            
        total_loss = mse(out, img_noisy_torch)
        total_loss.backward()
        
    
        psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0]) 
        psrn_gt    = compare_psnr(img_np, out.detach().cpu().numpy()[0]) 
        psrn_gt_sm = compare_psnr(img_np, out_avg.detach().cpu().numpy()[0]) 
    
        # Note that if we do not have GT, 'PSRN_gt', 'PSNR_gt_sm' make no sense
        print ('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm), '\r', end='')
        if  PLOT and i % show_every == 0:
            out_np = torch_to_np(out)
            #plot_image_grid([np.clip(out_np, 0, 1), 
                            #np.clip(torch_to_np(out_avg), 0, 1)], factor=figsize, nrow=1)
        
        
    
        # Backtracking
        if i % show_every:
            if psrn_noisy - psrn_noisy_last < -5: 
                print('Falling back to previous checkpoint.')

                for new_param, net_param in zip(last_net, net.parameters()):
                    net_param.data.copy_(new_param.cuda())

                return total_loss*0
            else:
                last_net = [x.detach().cpu() for x in net.parameters()]
                psrn_noisy_last = psrn_noisy
        if i==2999:
            psnr_noisy_float = float(psrn_noisy)
            psnr_gt_float = float(psrn_gt)
            psnr_ns.append(psnr_noisy_float)
            psnr_gs.append(psnr_gt_float)
        i += 1

        return total_loss
    p = get_params(OPT_OVER, net, net_input)
    optimize(OPTIMIZER, p, closure, LR, num_iter)
    out_np = torch_to_np(net(net_input))
    #q = plot_image_grid([np.clip(out_np, 0, 1), img_np], factor=13)
    os.makedirs(os.path.join(output_folder, 'denoised'), exist_ok=True)

    left_image = np.clip(out_np, 0, 1)  # Ensure pixel values are between [0, 1]

    # Handle single-channel images (grayscale) or add an RGB dimension if missing
    if left_image.ndim == 2:  # Grayscale
        left_image_uint8 = (left_image * 255).astype(np.uint8)
    elif left_image.ndim == 3 and left_image.shape[0] in [1, 3]:  # Handle CHW format
        left_image = np.moveaxis(left_image, 0, -1)  # Convert to HWC (Height, Width, Channels)
        left_image_uint8 = (left_image * 255).astype(np.uint8)
    else:  # Already HWC format
        left_image_uint8 = (left_image * 255).astype(np.uint8)
    name_path=output_folder+'/denoised/'+os.path.basename(image_file)
    imageio.imwrite(name_path, left_image_uint8)
    print(f"psnr_noise we have are {psnr_ns}")
    print(f"psnr_gt we have are {psnr_gs}")
    #print(f"Denoised image: {os.path.basename(image_file)} with psnr {psnr_gs[n]}")
    n=n+1
    print(f"{n} images deniosed. Let's do the next one")
     
print(f"expect {len(image_files)} images to be denoised, {n} images deniosed.")
gt_average = sum(psnr_gs) / len(psnr_gs)
noise_average = sum(psnr_ns) / len(psnr_ns)
print(f"As the noise level is sigma {sigma}")
print("The average psnr_noise is", noise_average)
print("The average psnr_gt is", gt_average)




        
    