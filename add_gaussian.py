import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from pathlib import Path

def add_gaussian_noise(image_tensor, std=15):
    """
    Add Gaussian noise to an image tensor.
    Parameters:
        image_tensor (torch.Tensor): Input image tensor in [0, 1].
        std (float): Standard deviation of Gaussian noise.
    Returns:
        torch.Tensor: Image tensor with Gaussian noise, clipped to [0, 1].
    """
    noise = torch.randn_like(image_tensor) * (std / 255.0)  # Normalize std to [0, 1]
    noisy_image = image_tensor + noise
    return noisy_image.clamp(0, 1)

def augment_and_save_images(input_dir, output_dir, std=15):
    """
    Add Gaussian noise to original images and save the results.
    Parameters:
        input_dir (str): Path to the directory containing clean images.
        output_dir (str): Path to save augmented images.
        std (float): Standard deviation for Gaussian noise.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define transforms to convert between PIL Image and Tensor
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    # Get all image files from input directory
    input_images = sorted(os.listdir(input_dir))

    for img_file in input_images:
        img_path = os.path.join(input_dir, img_file)
        image = Image.open(img_path).convert("RGB")  # Ensure RGB format

        # Convert image to Tensor
        image_tensor = to_tensor(image)

        # Add Gaussian noise
        noisy_image = add_gaussian_noise(image_tensor, std=std)
        noisy_pil = to_pil(noisy_image)  # Convert back to PIL Image

        # Save the noisy image with modified filename
        noisy_filename = f"gaussian_{std}_{img_file}"
        noisy_pil.save(os.path.join(output_dir, noisy_filename))

    print(f"Gaussian noise (std={std}) added. Total images saved: {len(input_images)}")

if __name__ == "__main__":
    # Define input and output directories
    current_dir = Path(__file__).parent
    clean_images_dir = current_dir / "BSDS300/images/train"  # Input image directory

    # Standard deviations and output directories
    stds_and_dirs = {
        15: current_dir / "gaussian_noise15",
        25: current_dir / "gaussian_noise25",
        50: current_dir / "gaussian_noise50",
    }

    # Add Gaussian noise for each std and save to corresponding directory
    for std, output_dir in stds_and_dirs.items():
        print(f"Processing Gaussian noise with std={std}...")
        augment_and_save_images(clean_images_dir, output_dir, std=std)