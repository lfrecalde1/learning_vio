from scipy.io import loadmat
from torch.utils.data import Dataset
import numpy as np
import torch

def quaternion_conjugate(q):
    q_conj = np.copy(q)
    q_conj[1:4] *= -1
    return q_conj

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

class IMUDatasetFromMat(Dataset):
    def __init__(self, mat_path):
        data = loadmat(mat_path)

        imu_window = data['imu_window']  # shape (6, window_size, N)
        self.imu_window = np.transpose(imu_window, (2, 1, 0))  # (N, window_size, 6)
        print(self.imu_window.shape)

        x = data['x']  # (12, N)
        self.positions = x[0:3, :].T       # (N, 3)
        self.orientations = x[6:10, :].T   # (N, 4)
        self.velocities = x[3:6, :].T      # (N, 3)

        self.N = self.positions.shape[0]
        assert self.imu_window.shape[0] == self.N

    def __len__(self):
        return self.N - 1  # Because we predict t+1

    def __getitem__(self, idx):
        imu_seq = self.imu_window[idx]   # (window_size, 6)

        pos_t = self.positions[idx]
        vel_t = self.velocities[idx]

        pos_t1 = self.positions[idx + 1]
        vel_t1 = self.velocities[idx + 1]
        quat_t1 = self.orientations[idx + 1]

        # Build input: concatenate pos_t, vel_t to each time step of imu
        pos_vel = np.concatenate([pos_t, vel_t])  # (6,)
        window_size = imu_seq.shape[0]
        pos_vel_repeated = np.tile(pos_vel, (window_size, 1))  # (window_size, 6)

        imu_augmented = np.concatenate([imu_seq, pos_vel_repeated], axis=1)  # (window_size, 12)

        imu_tensor = torch.tensor(imu_augmented, dtype=torch.float32)
        gt_pos = torch.tensor(pos_t1, dtype=torch.float32)
        gt_vel = torch.tensor(vel_t1, dtype=torch.float32)
        gt_quat = torch.tensor(quat_t1, dtype=torch.float32)

        gt = torch.cat([gt_pos, gt_quat, gt_vel], dim=0)  # (7,)

        return imu_tensor, gt