from scipy.io import loadmat
from torch.utils.data import Dataset
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

class IMUImageDataset(Dataset):
    def __init__(self, mat_path, image_folder, img_size=(128, 128)):
        data = loadmat(mat_path)

        imu_window = data['imu_window']  # (6, window_size, N)
        self.imu_window = np.transpose(imu_window, (2, 1, 0))  # (N, window_size, 6)

        x = data['x']  # (12, N)
        self.positions = x[0:3, :].T
        self.orientations = x[6:10, :].T
        self.velocities = x[3:6, :].T

        self.image_folder = image_folder
        self.img_size = img_size

        self.N = self.positions.shape[0]
        assert self.imu_window.shape[0] == self.N

        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Assuming grayscale, adjust for RGB
        ])

    def __len__(self):
        return self.N - 1

    def __getitem__(self, idx):
        imu_seq = self.imu_window[idx]  # (window_size, 6)

        pos_t = self.positions[idx]
        vel_t = self.velocities[idx]

        pos_t1 = self.positions[idx + 1]
        vel_t1 = self.velocities[idx + 1]
        quat_t1 = self.orientations[idx + 1]

        pos_vel = np.concatenate([pos_t, vel_t])
        window_size = imu_seq.shape[0]
        pos_vel_repeated = np.tile(pos_vel, (window_size, 1))
        imu_augmented = np.concatenate([imu_seq, pos_vel_repeated], axis=1)  # (window_size, 12)

        imu_tensor = torch.tensor(imu_augmented, dtype=torch.float32)

        gt_pos = torch.tensor(pos_t1, dtype=torch.float32)
        gt_vel = torch.tensor(vel_t1, dtype=torch.float32)
        gt_quat = torch.tensor(quat_t1, dtype=torch.float32)
        gt = torch.cat([gt_pos, gt_quat, gt_vel], dim=0)

        # ---- Load corresponding image ----
        img_path = os.path.join(self.image_folder, f"image_{idx:05d}.png")  # Or .jpg
        image = Image.open(img_path).convert('RGB')  # Use 'L' if grayscale
        img_tensor = self.transform(image)  # Shape: (3, H, W)

        return imu_tensor, img_tensor, gt