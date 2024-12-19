import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from pathlib import Path

def add_poisson_noise(image_tensor, scale=1.0):
    """
    Add Poisson noise to an image tensor.
    Parameters:
        image_tensor (torch.Tensor): Input image tensor in [0, 1].
        scale (float): Scaling factor for Poisson noise.
    Returns:
        torch.Tensor: Image tensor with Poisson noise, clipped to [0, 1].
    """
    # Scale the image tensor to increase noise level
    scaled_image = image_tensor * scale
    
    # Generate Poisson noise (values follow Poisson distribution)
    noisy_image = torch.poisson(scaled_image * 255.0) / 255.0 / scale
    
    # Clamp pixel values to [0, 1]
    return noisy_image.clamp(0, 1)

def augment_and_save_images(input_dir, output_dir, scale=1.0):
    """
    Add Poisson noise to original images and save the results.
    Parameters:
        input_dir (str): Path to the directory containing clean images.
        output_dir (str): Path to save augmented images.
        scale (float): Scaling factor for Poisson noise.
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

        # Add Poisson noise
        noisy_image = add_poisson_noise(image_tensor, scale=scale)
        noisy_pil = to_pil(noisy_image)  # Convert back to PIL Image

        # Save the noisy image with modified filename
        noisy_filename = f"poisson_{img_file}"
        noisy_pil.save(os.path.join(output_dir, noisy_filename))

    print(f"Poisson noise added. Total images saved: {len(input_images)}")

if __name__ == "__main__":
    # Define input and output directories
    current_dir = Path(__file__).parent
    clean_images_dir = current_dir / "BSDS300/images/test"  # Input image directory

    # Scales and output directories
    scales_and_dirs = {
        1.0: current_dir / "poisson_noise1_test",
        0.1: current_dir / "poisson_noise01_test",
        0.01: current_dir / "poisson_noise001_test",
    }

    # Add Poisson noise for each scale and save to corresponding directory
    for scale, output_dir in scales_and_dirs.items():
        print(f"Processing scale={scale}...")
        augment_and_save_images(clean_images_dir, output_dir, scale=scale)
