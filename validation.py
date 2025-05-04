import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from Models import *  # Your new PoseNet with cross-attention
from utils import *
from scipy.io import loadmat
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def quat_to_rot(q):
    q0, q1, q2, q3 = q
    R = np.array([
        [q0**2 + q1**2 - q2**2 - q3**2, 2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3),             q0**2 + q2**2 - q1**2 - q3**2, 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2),             2*(q2*q3 + q0*q1),     q0**2 + q3**2 - q1**2 - q2**2]
    ])
    return R

# ------------ LOAD DATASET ----------------------

data = loadmat("liss.mat")
imu_window = data['imu_window']  # (6, window_length, N)
x = data['x']  # (13, N)

print("imu_window shape:", imu_window.shape)
print("x shape:", x.shape)

image_folder = "./liss/saved_images"  # folder with images 000000.png, 000001.png, etc.
img_size = (128, 128)   # Same size used in training

transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ------------ LOAD MODEL ----------------------

model = PoseNet(
    input_dim=12,
    output_dim=10,
    hidden_dim=256,
    dropout=0.2,
    num_layers=2
).to(DEVICE)

ckpt_path = '/home/fer/Lectures/computer_vision/Learning_vio/ckpts/best_model.pth'
model.load_state_dict(torch.load(ckpt_path))
model.eval()

# ------------ INITIALIZE ----------------------

N = imu_window.shape[2]
window_length = imu_window.shape[1]

pos_predictions = np.zeros((3, N))
vel_predictions = np.zeros((3, N))
quat_predictions = np.zeros((4, N))

# Initial values from GT for first step
pos_predictions[:, 0] = x[0:3, 0]
vel_predictions[:, 0] = x[3:6, 0]
quat_predictions[:, 0] = x[6:10, 0]

# ------------ PREDICTION LOOP ----------------------

for k in tqdm(range(N - 1), desc="Prediction"):

    imu_seq_k = imu_window[:, :, k].T  # (window_length, 6)

    pos_vel = np.hstack([x[0:3, k], x[3:6, k]])  # use GT pos & vel as in training
    pos_vel_repeated = np.tile(pos_vel, (window_length, 1))
    net_input = np.concatenate([imu_seq_k, pos_vel_repeated], axis=1)  # (window_length, 12)

    net_input_torch = torch.tensor(net_input, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # --- Load image pair ---
    img_path1 = os.path.join(image_folder, f"image_{k:05d}.png")
    img_path2 = os.path.join(image_folder, f"image_{k + 1:05d}.png")

    img1 = Image.open(img_path1).convert('RGB')
    img2 = Image.open(img_path2).convert('RGB')

    img_tensor1 = transform(img1)
    img_tensor2 = transform(img2)
    img_pair = torch.cat([img_tensor1, img_tensor2], dim=0).unsqueeze(0).to(DEVICE)  # (1, 6, H, W)

    with torch.no_grad():
        pred = model(net_input_torch, img_pair)

    pos_pred = pred[0, :3].cpu().numpy()
    quat_pred = pred[0, 3:7].cpu().numpy()
    vel_pred = pred[0, 7:].cpu().numpy()

    quat_pred /= np.linalg.norm(quat_pred)

    pos_predictions[:, k + 1] = pos_pred
    vel_predictions[:, k + 1] = vel_pred
    quat_predictions[:, k + 1] = quat_pred

# ------------ PLOTTING ----------------------

gt_positions = x[0:3, :]
gt_quaternions = x[6:10, :]

# ----- Position Plot -----
plt.figure(figsize=(12, 8))
for i, label in enumerate(['X', 'Y', 'Z']):
    plt.plot(gt_positions[i, :], label=f'GT {label}')
    plt.plot(pos_predictions[i, :], '--', label=f'Pred {label}')
plt.title('Position Comparison')
plt.xlabel('Sample index')
plt.ylabel('Position (m)')
plt.legend()
plt.grid()
plt.show()

# ----- Quaternion Plot -----
#plt.figure(figsize=(12, 8))
#for i, label in ['w', 'x', 'y', 'z']:
#    plt.plot(gt_quaternions[i, :], label=f'GT {label}')
#    plt.plot(quat_predictions[i, :], '--', label=f'Pred {label}')
#plt.title('Quaternion Comparison')
#plt.xlabel('Sample index')
#plt.ylabel('Quaternion component')
#plt.legend()
#plt.grid()
#plt.show()

# ----- Trajectory Plot -----
plt.figure(figsize=(10, 8))
plt.plot(gt_positions[0, :], gt_positions[1, :], label='Ground Truth', color='blue')
plt.plot(pos_predictions[0, :], pos_predictions[1, :], '--', label='Predicted', color='red')

plt.plot(gt_positions[0, 0], gt_positions[1, 0], 'o', color='blue', markersize=10, label='GT Start')
plt.plot(pos_predictions[0, 0], pos_predictions[1, 0], 'o', color='red', markersize=10, label='Pred Start')

plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Trajectory Comparison')
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.legend()
plt.grid(True)
plt.show()