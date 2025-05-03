import os
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as R
import torch
from scipy.io import loadmat

from scipy.io import loadmat
from torch.utils.data import Dataset
import numpy as np
import torch

def quaternion_conjugate(q):
    """Conjugate of quaternion [w, x, y, z]"""
    q_conj = np.copy(q)
    q_conj[1:4] *= -1
    return q_conj

def quaternion_multiply(q1, q2):
    """Quaternion multiplication: q1 ⊗ q2 (both [w, x, y, z])"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def quat_to_rot(q):
    """Rotation matrix from quaternion [w, x, y, z]"""
    q0, q1, q2, q3 = q
    R = np.array([
        [q0**2 + q1**2 - q2**2 - q3**2, 2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3),             q0**2 + q2**2 - q1**2 - q3**2, 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2),             2*(q2*q3 + q0*q1),     q0**2 + q3**2 - q1**2 - q2**2]
    ])
    return R

class IMUDatasetFromMat(Dataset):
    def __init__(self, mat_path):
        data = loadmat(mat_path)

        imu_window = data['imu_window']  # (6, 10, N)
        self.imu_window = np.transpose(imu_window, (2, 1, 0))  # (N, 10, 6)

        x = data['x']  # (12, N)
        self.positions = x[3:6, :].T    # (N, 3)
        self.orientations = x[6:10, :].T  # (N, 4) already in [w, x, y, z]

        assert self.imu_window.shape[0] == self.positions.shape[0], "Mismatch in sample count."
        self.N = self.positions.shape[0]

        # Normalize quaternions to unit norm
        #norms = np.linalg.norm(self.orientations, axis=1, keepdims=True)
        self.orientations = self.orientations

    def __len__(self):
        return self.N - 1  # we need t and t-1

    def __getitem__(self, idx):
        imu_seq = self.imu_window[idx]    # (10, 6)

        pos_t0 = self.positions[idx]
        pos_t1 = self.positions[idx + 1]

        quat_t0 = self.orientations[idx]      # [w, x, y, z]
        quat_t1 = self.orientations[idx + 1]

        # --- Delta position ---
        delta_pos_global = pos_t0
        R_t0 = quat_to_rot(quat_t0)
        delta_pos = delta_pos_global  # express in t-1 frame

        # --- Relative quaternion ---
        quat_t0_conj = quaternion_conjugate(quat_t0)
        delta_quat = quat_t0
        #delta_quat /= np.linalg.norm(delta_quat)

        imu_tensor = torch.tensor(imu_seq, dtype=torch.float32)
        delta_pos_tensor = torch.tensor(delta_pos, dtype=torch.float32)
        delta_quat_tensor = torch.tensor(delta_quat, dtype=torch.float32)

        gt = torch.cat([delta_pos_tensor, delta_quat_tensor], dim=0)  # (7,)

        return imu_tensor, gt