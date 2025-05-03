import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Models import *
from utils import *
from tqdm import tqdm
from scipy.io import loadmat

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def quaternion_conjugate_torch(q):
    q_conj = q.clone()
    q_conj[..., 1:4] *= -1
    return q_conj

def quaternion_multiply_torch(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=-1)

def quat_to_rot(q):
    """Rotation matrix from quaternion [w, x, y, z]"""
    q0, q1, q2, q3 = q
    R = np.array([
        [q0**2 + q1**2 - q2**2 - q3**2, 2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3),             q0**2 + q2**2 - q1**2 - q3**2, 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2),             2*(q2*q3 + q0*q1),     q0**2 + q3**2 - q1**2 - q2**2]
    ])
    return R

# ------------ LOAD DATASET AND MODEL ----------------------

val_dataset = IMUDatasetFromMat("trajectories_3.mat")
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

model = PoseNet(
    input_dim=6,
    output_dim=7,
    hidden_dim=256,
    dropout=0.2,
    num_layers=2
).to(DEVICE)

# Load the best trained weights
ckpt_path = '/home/fer/Lectures/computer_vision/Learning_vio/ckpts/best_model.pth'
model.load_state_dict(torch.load(ckpt_path))
model.eval()

# Load Data Validation
data = loadmat("trajectories_3.mat")
imu_window = data['imu_window']  # (6, 10, N)
x = data['x']  # (12, N)

print(imu_window.shape)
print(x.shape)

# ------------ RUN VALIDATION ----------------------

all_pos_pred = []
all_pos_gt = []
all_quat_pred = []
all_quat_gt = []


estimation_positions = np.zeros((3, x.shape[1]))
estimation_positions[:, 0] = x[3:6, 0]

gt_positions = x[3:6, :]
k = 0

dt = 0.01667  # time step

def position_derivative(position, delta, R):
    return R @ delta
with torch.no_grad():
    for imu_tensor, gt in tqdm(val_loader, desc="Running Validation"):
        imu_tensor = imu_tensor.to(DEVICE)
        gt = gt.to(DEVICE)

        pred = model(imu_tensor)

        pos_pred, quat_pred = pred[:, :3], pred[:, 3:]
        pos_gt, quat_gt = gt[:, :3], gt[:, 3:]

        quat_pred = quat_pred / quat_pred.norm(dim=1, keepdim=True)

        all_pos_pred.append(pos_pred.cpu().numpy())
        all_pos_gt.append(pos_gt.cpu().numpy())
        all_quat_pred.append(quat_pred.cpu().numpy())
        all_quat_gt.append(quat_gt.cpu().numpy())

        delta_pos = pos_pred.cpu().numpy().reshape((3,))
        q_pred = quat_pred.cpu().numpy().reshape((4,))
        q_gt = quat_gt.cpu().numpy().reshape((4,))
        R = quat_to_rot(q_gt)

        p0 = estimation_positions[:, k]  # current position

        # RK4 integration
        k1 = position_derivative(p0, delta_pos, R)
        k2 = position_derivative(p0 + 0.5 * dt * k1, delta_pos, R)
        k3 = position_derivative(p0 + 0.5 * dt * k2, delta_pos, R)
        k4 = position_derivative(p0 + dt * k3, delta_pos, R)

        dp = (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        estimation_positions[:, k+1] = p0 + dp

        k += 1

# Stack everything to shape (N, 3) or (N, 4)
all_pos_pred = np.vstack(all_pos_pred)
all_pos_gt = np.vstack(all_pos_gt)
all_quat_pred = np.vstack(all_quat_pred)
all_quat_gt = np.vstack(all_quat_gt)

# ------------ PLOTTING ----------------------

# Position plot
plt.figure(figsize=(12, 8))
for i, label in enumerate(['X', 'Y', 'Z']):
    plt.plot(all_pos_gt[:, i], label=f'GT {label}')
    plt.plot(all_pos_pred[:, i], '--', label=f'Pred {label}')
plt.title('Position Comparison')
plt.xlabel('Sample index')
plt.ylabel('Position (m)')
plt.legend()
plt.grid()
plt.show()

# Quaternion plot
plt.figure(figsize=(12, 8))
for i, label in enumerate(['w', 'x', 'y', 'z']):
    plt.plot(all_quat_gt[:, i], label=f'GT {label}')
    plt.plot(all_quat_pred[:, i], '--', label=f'Pred {label}')
plt.title('Quaternion Comparison')
plt.xlabel('Sample index')
plt.ylabel('Quaternion component')
plt.legend()
plt.grid()
plt.show()

# ------------ PLOTTING TRAJECTORIES ----------------------

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)

# Ground truth trajectory
ax.plot(gt_positions[0, :], gt_positions[1, :], label='Ground Truth', color='blue')

# Estimated trajectory
ax.plot(estimation_positions[0, :], estimation_positions[1, :], 
        label='Estimated', color='red', linestyle='--')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('3D Trajectory Comparison')
ax.legend()
plt.grid(True)
plt.tight_layout()
plt.show()