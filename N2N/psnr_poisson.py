import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from pathlib import Path
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as psnr

# ================== Simple Denoising Model ==================
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ================== Add Poisson Noise ==================
def add_poisson_noise(image_tensor, scale=1.0):
    """
    Add Poisson noise to an image tensor.
    Parameters:
        image_tensor (torch.Tensor): Input image tensor in [0, 1].
        scale (float): Scaling factor for Poisson noise.
    Returns:
        torch.Tensor: Image tensor with Poisson noise, clipped to [0, 1].
    """
    scaled_image = image_tensor * scale
    noisy_image = torch.poisson(scaled_image * 255.0) / 255.0 / scale
    return noisy_image.clamp(0, 1)

# ================== Add Noise and Denoise ==================
def calculate_snr(clean_image, noisy_image):
    """
    Calculate the Signal-to-Noise Ratio (SNR) between clean and noisy images.
    Parameters:
        clean_image (torch.Tensor): Clean image tensor in [0, 1].
        noisy_image (torch.Tensor): Noisy image tensor in [0, 1].
    Returns:
        float: SNR value in dB.
    """
    noise = noisy_image - clean_image
    signal_power = torch.mean(clean_image ** 2).item()
    noise_power = torch.mean(noise ** 2).item()
    if noise_power == 0:
        return float('inf')  # Perfect reconstruction
    return 10 * np.log10(signal_power / noise_power)

def process_images(model_path, clean_dir, output_dir, transform, scales):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = DenoisingAutoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    for scale in scales:
        scale_prefix = f"p_scale{str(scale).replace('.', '')}_"
        scale_output_dir = os.path.join(output_dir, scale_prefix)
        os.makedirs(scale_output_dir, exist_ok=True)

        # Get all clean images
        clean_files = sorted(os.listdir(clean_dir))
        psnr_values = []
        snr_values = []

        for clean_file in clean_files:
            clean_path = os.path.join(clean_dir, clean_file)
            clean_image = Image.open(clean_path).convert("RGB")

            # Apply transform and add Poisson noise
            clean_tensor = transform(clean_image).unsqueeze(0).to(device)
            noisy_tensor = add_poisson_noise(clean_tensor, scale=scale)

            # Save noisy image
            noisy_image = noisy_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            noisy_image = (noisy_image * 255).clip(0, 255).astype(np.uint8)  # Scale back to [0, 255]
            noisy_image_pil = Image.fromarray(noisy_image)
            noisy_image_path = os.path.join(scale_output_dir, f"{scale_prefix}{clean_file}")
            noisy_image_pil.save(noisy_image_path)

            # Denoise using the model
            with torch.no_grad():
                denoised_tensor = model(noisy_tensor)

            # Save denoised image
            denoised_image = denoised_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            denoised_image = (denoised_image * 255).clip(0, 255).astype(np.uint8)  # Scale back to [0, 255]
            denoised_image_pil = Image.fromarray(denoised_image)
            denoised_image_path = os.path.join(scale_output_dir, f"denoise_{scale_prefix}{clean_file}")
            denoised_image_pil.save(denoised_image_path)

            # Calculate PSNR between clean and denoised image
            clean_np = clean_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            denoised_np = denoised_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            psnr_value = psnr(clean_np, denoised_np, data_range=1.0)
            psnr_values.append(psnr_value)

            # Calculate SNR between clean and noisy image
            snr_value = calculate_snr(clean_tensor.squeeze(0), noisy_tensor.squeeze(0))
            snr_values.append(snr_value)

        # Calculate average PSNR and SNR for this scale
        average_psnr = np.mean(psnr_values)
        average_snr = np.mean(snr_values)
        print(f"Scale {scale}: Average PSNR: {average_psnr:.2f} dB, Average SNR: {average_snr:.2f} dB")

# ================== Run Processing ==================
if __name__ == "__main__":
    # Get current script directory
    current_dir = Path(__file__).parent
    clean_images_dir = current_dir / "BSDS300/images/test"  # Replace with actual test image path
    output_images_dir = current_dir / "p-Poisson"  # Output image directory
    model_path = current_dir / "denoising_autoencoder_n2n_3.pth"

    transform = transforms.Compose([
        transforms.ToTensor()  # Convert to tensor
    ])

    scales = [1.0, 0.1, 0.01]

    process_images(model_path, clean_images_dir, output_images_dir, transform, scales)
