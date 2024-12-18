#concept
The original idea comes form https://dmitryulyanov.github.io/deep_image_prior
Our team works on DIP by writing a program using its idea and features and tests its performance on denoising BSDS300 image test sets.
The average psnr_gt is 29.60 when denoising noisy images added gaussion noise with segma 15
The average psnr_gt is 27.01 when denoising noisy images added gaussion noise with segma 25
The average psnr_gt is 22.96 when denoising noisy images added gaussion noise with segma 50
The average psnr_gt is 29.85 when denoising noisy images added poisson noise with intensity scale 1
The average psnr_gt is 26.48 when denoising noisy images added poisson noise with intensity scale 0.1
The average psnr_gt is 20.43 when denoising noisy images added poisson noise with intensity scale 0.001


# Install

Here is the list of libraries you need to install to execute the code:
- python = 3.6
- [pytorch](http://pytorch.org/) = 0.4
- numpy
- scipy
- matplotlib
- scikit-image

#excecute

1.Change the input and output folder before using the program.
2.Change the noise_type, using the one you want. Change sigma or noise_intensity according to the noise type you want.
3.Change the reg_noise_std. For guassion noise with sigma = 15, set to 1./34. ; for sigma=25, set to 1./30. ;for sigma=50, set to 1./20. ;For poisson noise and other noise, you may want to experiment with this parameter.
4.Then you can excecute it by  python denoising.py (on windows terminal)
