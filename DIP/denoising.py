from __future__ import print_function
import matplotlib.pyplot as plt

import os
import numpy as np
from models import *

import imageio
import glob
import torch
import torch.optim

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from utils.denoising_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

psnr_ns = []
psnr_gs = []
n = 0
imsize = -1

sigma = 25
sigma_ = sigma/255.

# Choose noise type: 'gaussian' or 'poisson'
noise_type = 'poisson'  
noise_intensity = 0.01  # Adjust this value to control Poisson noise intensity

folder_path = "C:/Users/shixi/Documents/NYUWork/Image and Video Processing/Project/DIP/deep-image-prior/data/BSDS300"
output_folder = "C:/Users/shixi/Documents/NYUWork/Image and Video Processing/Project/DIP/deep-image-prior/data/denoised_BSDS300"

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

    # Add noise according to chosen noise_type
    if noise_type == 'gaussian':
        # Gaussian noise
        img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)
    elif noise_type == 'poisson':
        # Poisson noise
        lam = (img_np * 255.0 * noise_intensity)
        lam[lam < 0] = 0
        noisy_counts = np.random.poisson(lam)
        # Scale back to [0,1]
        img_noisy_np = noisy_counts / (255.0 * noise_intensity)
        img_noisy_pil = np_to_pil(img_noisy_np)
    else:
        raise ValueError("noise_type must be 'gaussian' or 'poisson'.")

    # Save the noisy image
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

    os.makedirs(os.path.join(output_folder, 'noisy'), exist_ok=True)

    # Save the noisy image
    noisy_image_path = os.path.join(output_folder, 'noisy', os.path.basename(image_file))
    imageio.imwrite(noisy_image_path, noisy_img_uint8)

    # Setup for DIP
    INPUT = 'noise' # 'meshgrid'
    pad = 'reflection'
    OPT_OVER = 'net' # 'net,input'

    # For noise with sigma = 15, set to 1./34. ;set to 1./30. for sigma=25 ;set to 1./20. for sigma=50. For poisson noise and other noise, you may want to experiment with this parameter
    reg_noise_std = 1./5.  
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

    # Optimize
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    out_avg = None
    last_net = None
    psrn_noisy_last = 0
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
        
        psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0], data_range=1.0)
        psrn_gt = compare_psnr(img_np, out.detach().cpu().numpy()[0], data_range=1.0)
        psrn_gt_sm = compare_psnr(img_np, out_avg.detach().cpu().numpy()[0], data_range=1.0)

        print ('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % 
               (i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm), '\r', end='')

        if  i % show_every == 0:
            out_np = torch_to_np(out)

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

        if i == 2999:
            psnr_noisy_float = float(psrn_noisy)
            psnr_gt_float = float(psrn_gt)
            psnr_ns.append(psnr_noisy_float)
            psnr_gs.append(psnr_gt_float)

        i += 1
        return total_loss

    p = get_params(OPT_OVER, net, net_input)
    optimize(OPTIMIZER, p, closure, LR, num_iter)

    out_np = torch_to_np(net(net_input))
    os.makedirs(os.path.join(output_folder, 'denoised'), exist_ok=True)

    left_image = np.clip(out_np, 0, 1)  # Ensure pixel values are between [0, 1]

    # Handle single-channel or CHW images
    if left_image.ndim == 2:  # Grayscale
        left_image_uint8 = (left_image * 255).astype(np.uint8)
    elif left_image.ndim == 3 and left_image.shape[0] in [1, 3]:  # CHW format
        left_image = np.moveaxis(left_image, 0, -1)  # Convert to HWC
        left_image_uint8 = (left_image * 255).astype(np.uint8)
    else:  # Already HWC
        left_image_uint8 = (left_image * 255).astype(np.uint8)

    name_path = os.path.join(output_folder, 'denoised', os.path.basename(image_file))
    imageio.imwrite(name_path, left_image_uint8)
    print(f"psnr_noise we have are {psnr_ns}")
    print(f"psnr_gt we have are {psnr_gs}")
    n = n+1
    if(n==20):
        break
    print(f"{n} images denoised. Let's do the next one")

print(f"Expect {len(image_files)} images to be denoised, {n} images denoised.")
gt_average = sum(psnr_gs) / len(psnr_gs) if len(psnr_gs) > 0 else 0
noise_average = sum(psnr_ns) / len(psnr_ns) if len(psnr_ns) > 0 else 0
if noise_type == 'gaussian':
        # Gaussian noise
    print(f"As the noise level is sigma {sigma}")
elif noise_type == 'poisson':
    print(f"As the noise level is poisson scale {noise_intensity}")
else:
    raise ValueError("noise_type must be 'gaussian' or 'poisson'.")  
print("The average psnr_noise is", noise_average)
print("The average psnr_gt is", gt_average)



        
    