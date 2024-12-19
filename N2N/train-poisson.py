import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from pathlib import Path

# ================== Dataset Class ==================
class DynamicNoisyDataset(Dataset):
    def __init__(self, clean_dir, transform=None, noise_intensity_range=(1, 50)):
        self.clean_dir = clean_dir
        self.transform = transform
        self.noise_intensity_range = noise_intensity_range
        self.clean_files = sorted(os.listdir(clean_dir))

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean_path = os.path.join(self.clean_dir, self.clean_files[idx])
        clean_image = Image.open(clean_path).convert("RGB")

        if self.transform:
            clean_tensor = self.transform(clean_image)  # Normalized tensor in [0, 1]

        # Generate two independent noisy versions using Poisson noise
        noise_intensity_1 = np.random.uniform(*self.noise_intensity_range)
        noise_intensity_2 = np.random.uniform(*self.noise_intensity_range)

        noisy_tensor_1 = torch.poisson(clean_tensor * noise_intensity_1) / noise_intensity_1
        noisy_tensor_2 = torch.poisson(clean_tensor * noise_intensity_2) / noise_intensity_2

        return noisy_tensor_1, noisy_tensor_2

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

# ================== Training Code ==================
def train_model(clean_dir, output_model, epochs=10, batch_size=8, lr=1e-3, noise_intensity_range=(1, 50)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整图像大小
        transforms.ToTensor()
    ])

    # Datasets and DataLoader
    train_dataset = DynamicNoisyDataset(clean_dir, transform=transform, noise_intensity_range=noise_intensity_range)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model, Loss, Optimizer
    model = DenoisingAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning Rate Scheduler
    def compute_ramped_down_lrate(epoch, total_epochs, learning_rate):
        ramp_down_start = total_epochs * 0.8  # Ramp down starts at 80% of training
        if epoch >= ramp_down_start:
            t = (epoch - ramp_down_start) / (total_epochs - ramp_down_start)
            smooth = (0.5 + 0.5 * np.cos(t * np.pi)) ** 2
            return learning_rate * smooth
        return learning_rate

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: compute_ramped_down_lrate(epoch, epochs, lr))

    # Training Loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for noisy_imgs_1, noisy_imgs_2 in train_loader:
            noisy_imgs_1, noisy_imgs_2 = noisy_imgs_1.to(device), noisy_imgs_2.to(device)

            # Forward pass
            outputs = model(noisy_imgs_1)
            loss = criterion(outputs, noisy_imgs_2)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()  # Update learning rate
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

    # Save the trained model
    torch.save(model.state_dict(), output_model)
    print(f"Model saved to {output_model}")



# ================== Run Training ==================
if __name__ == "__main__":

    current_dir = Path(__file__).parent
    clean_images_dir = current_dir / "BSDS300/images/train" 
    output_model_path = current_dir / "denoising_autoencoder_poisson_n2n.pth"

    train_model(
        clean_images_dir, 
        output_model_path, 
        epochs=300, 
        batch_size=8, 
        lr=1e-2, 
        noise_intensity_range=(1, 50)  # 替换为 Poisson 的 noise_intensity_range
    )
