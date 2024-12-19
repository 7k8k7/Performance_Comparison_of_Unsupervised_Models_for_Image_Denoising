import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import load_model
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image


def load_images(folder_path, image_size=(128, 128)):

    images = []
    file_paths = []

    for file_name in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file_name)
        if file_name.lower().endswith(('png', 'jpg', 'jpeg')):
            file_paths.append(file_path)
            img = Image.open(file_path).convert("RGB")
            img = img.resize(image_size)
            img = np.array(img) / 255.0
            images.append(img)

    images = np.array(images)
    return images, file_paths


def test_model_with_clean_images(model, noisy_images, clean_images, file_paths, save_results_folder=None):

    psnr_scores = []

    if save_results_folder:
        os.makedirs(save_results_folder, exist_ok=True)


    for idx, (noisy_image, clean_image) in enumerate(zip(noisy_images, clean_images)):
        noisy_image_input = np.expand_dims(noisy_image, axis=0)
        denoised_image = model.predict(noisy_image_input)[0]
        denoised_image = np.clip(denoised_image, 0, 1)


        score = psnr(clean_image, denoised_image, data_range=1.0)
        psnr_scores.append(score)

        print(f"Image: {file_paths[idx]} - PSNR: {score:.2f}")


        if save_results_folder:
            save_path = os.path.join(save_results_folder, os.path.basename(file_paths[idx]))
            denoised_image = (denoised_image * 255).astype(np.uint8)  # 转为 [0, 255]
            denoised_img = Image.fromarray(denoised_image)
            denoised_img.save(save_path)

    return psnr_scores


if __name__ == "__main__":

    noisy_folder = "noised_images/noised_images/saltpepper_noise005_test"
    clean_folder = "noised_images/noised_images/clean_images/test"
    model_path = "denoising_model_all_nosie.h5"
    save_results_folder = "defusion_results_s005"


    image_size = (64, 64)


    print("Loading noisy images...")
    noisy_images, noisy_file_paths = load_images(noisy_folder, image_size=image_size)

    print("Loading clean images...")
    clean_images, _ = load_images(clean_folder, image_size=image_size)

    assert noisy_images.shape == clean_images.shape, "Not Match!"

    print("Loading model...")
    model = load_model(model_path)


    print("Testing model...")
    psnr_scores = test_model_with_clean_images(model, noisy_images, clean_images, noisy_file_paths, save_results_folder)

    avg_psnr = np.mean(psnr_scores)
    print(f"\nAverage PSNR: {avg_psnr:.2f}")
