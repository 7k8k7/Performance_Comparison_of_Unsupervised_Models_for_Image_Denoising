import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
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

# ================== Add Noise and Denoise ==================
def process_images(model_path, clean_dir, output_dir, transform, noise_std):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = DenoisingAutoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all clean images
    clean_files = sorted(os.listdir(clean_dir))
    psnr_values = []

    for clean_file in clean_files:
        clean_path = os.path.join(clean_dir, clean_file)
        clean_image = Image.open(clean_path).convert("RGB")

        # Apply transform and add Gaussian noise
        clean_tensor = transform(clean_image).unsqueeze(0).to(device)
        noise = torch.randn_like(clean_tensor) * noise_std / 255.0
        noisy_tensor = clean_tensor + noise

        # Save noisy image
        noisy_image = noisy_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        noisy_image = (noisy_image * 255).clip(0, 255).astype(np.uint8)  # Scale back to [0, 255]
        noisy_image_pil = Image.fromarray(noisy_image)
        noisy_image_path = os.path.join(output_dir, f"noise50_{clean_file}")
        noisy_image_pil.save(noisy_image_path)

        # Denoise using the model
        with torch.no_grad():
            denoised_tensor = model(noisy_tensor)

        # Save denoised image
        denoised_image = denoised_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        denoised_image = (denoised_image * 255).clip(0, 255).astype(np.uint8)  # Scale back to [0, 255]
        denoised_image_pil = Image.fromarray(denoised_image)
        denoised_image_path = os.path.join(output_dir, f"denoise50_{clean_file}")
        denoised_image_pil.save(denoised_image_path)

        # Calculate PSNR between clean and denoised image
        clean_np = clean_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        denoised_np = denoised_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        psnr_value = psnr(clean_np, denoised_np, data_range=1.0)
        psnr_values.append(psnr_value)

    # Calculate average PSNR
    average_psnr = np.mean(psnr_values)
    print(f"Average PSNR: {average_psnr:.2f} dB")

# ================== Run Processing ==================
if __name__ == "__main__":
    # 获取当前脚本所在目录
    current_dir = Path(__file__).parent
    clean_images_dir = current_dir / "BSDS300/images/test"  # 替换为实际测试图片路径
    output_images_dir = current_dir / "p-Gaussian/noise50"  # 输出图像的目录
    model_path = current_dir / "denoising_autoencoder_poisson_n2n.pth"

    transform = transforms.Compose([
        transforms.ToTensor()  # 转换为张量，但不改变尺寸
    ])

    process_images(model_path, clean_images_dir, output_images_dir, transform, noise_std=50)
