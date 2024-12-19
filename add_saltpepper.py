import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from pathlib import Path

def add_salt_and_pepper_noise(image_tensor, prob=0.01):
    """
    Add salt-and-pepper noise to an image tensor.
    Parameters:
        image_tensor (torch.Tensor): Input image tensor in [0, 1].
        prob (float): Probability of salt-and-pepper noise.
    Returns:
        torch.Tensor: Image tensor with salt-and-pepper noise, clipped to [0, 1].
    """
    noisy_image = image_tensor.clone()
    salt = torch.rand_like(image_tensor) < prob / 2  # Salt noise
    pepper = torch.rand_like(image_tensor) > 1 - prob / 2  # Pepper noise

    noisy_image[salt] = 1.0
    noisy_image[pepper] = 0.0
    return noisy_image.clamp(0, 1)

def augment_and_save_images(input_dir, output_dir, prob=0.01):
    """
    Add salt-and-pepper noise to original images and save the results.
    Parameters:
        input_dir (str): Path to the directory containing clean images.
        output_dir (str): Path to save augmented images.
        prob (float): Probability for salt-and-pepper noise.
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

        # Add salt-and-pepper noise
        noisy_image = add_salt_and_pepper_noise(image_tensor, prob=prob)
        noisy_pil = to_pil(noisy_image)  # Convert back to PIL Image

        # Save the noisy image with modified filename
        noisy_filename = f"saltpepper_{int(prob*100)}_{img_file}"
        noisy_pil.save(os.path.join(output_dir, noisy_filename))

    print(f"Salt-and-pepper noise (prob={prob}) added. Total images saved: {len(input_images)}")

if __name__ == "__main__":
    # Define input and output directories
    current_dir = Path(__file__).parent
    clean_images_dir = current_dir / "BSDS300/images/train"  # Input image directory

    # Noise probabilities and output directories
    probs_and_dirs = {
        0.01: current_dir / "saltpepper_noise001",
        0.05: current_dir / "saltpepper_noise005",
        0.1: current_dir / "saltpepper_noise01",
    }

    # Add salt-and-pepper noise for each probability and save to corresponding directory
    for prob, output_dir in probs_and_dirs.items():
        print(f"Processing salt-and-pepper noise with prob={prob}...")
        augment_and_save_images(clean_images_dir, output_dir, prob=prob)