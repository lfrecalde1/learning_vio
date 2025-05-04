import torchvision.transforms as transforms
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset
import numpy as np
import torch
import os

class IMUImagePairDataset(Dataset):
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
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # RGB normalization
        ])

    def __len__(self):
        return self.N - 1  # Because we use img_k and img_k+1

    def __getitem__(self, idx):
        imu_seq = self.imu_window[idx]  # (window_size, 6)

        pos_t1 = self.positions[idx + 1]
        vel_t1 = self.velocities[idx + 1]
        quat_t1 = self.orientations[idx + 1]

        pos_vel = np.concatenate([self.positions[idx], self.velocities[idx]])
        window_size = imu_seq.shape[0]
        pos_vel_repeated = np.tile(pos_vel, (window_size, 1))
        imu_augmented = np.concatenate([imu_seq, pos_vel_repeated], axis=1)  # (window_size, 12)

        imu_tensor = torch.tensor(imu_augmented, dtype=torch.float32)
        gt_pos = torch.tensor(pos_t1, dtype=torch.float32)
        gt_vel = torch.tensor(vel_t1, dtype=torch.float32)
        gt_quat = torch.tensor(quat_t1, dtype=torch.float32)
        gt = torch.cat([gt_pos, gt_quat, gt_vel], dim=0)  # (10,)

        # --- Load image pair ---
        img_path1 = os.path.join(self.image_folder, f"image_{idx:05d}.png")
        img_path2 = os.path.join(self.image_folder, f"image_{idx + 1:05d}.png")

        img1 = Image.open(img_path1).convert('RGB')
        img2 = Image.open(img_path2).convert('RGB')

        img_tensor1 = self.transform(img1)  # (3, H, W)
        img_tensor2 = self.transform(img2)  # (3, H, W)

        img_pair = torch.cat([img_tensor1, img_tensor2], dim=0)  # (6, H, W)

        return imu_tensor, img_pair, gt