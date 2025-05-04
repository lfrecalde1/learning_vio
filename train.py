import os
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
from Models import *
from utils import *
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------- Quaternion Utils ---------------------------
def quaternion_conjugate_torch(q):
    q_conj = q.clone()
    q_conj[..., 1:4] *= -1
    return q_conj

def quaternion_product(q1, q2):
    w1, x1, y1, z1 = q1[:, 0:1], q1[:, 1:2], q1[:, 2:3], q1[:, 3:]
    w2, x2, y2, z2 = q2[:, 0:1], q2[:, 1:2], q2[:, 2:3], q2[:, 3:]

    q_prod = torch.cat((
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ), dim=1)

    return q_prod

def quaternion_log(q):
    q = q / q.norm(dim=1, keepdim=True)  # Safety normalization
    q_v = q[:, 1:]  # Vector part
    q_v_norm = torch.norm(q_v, dim=1, keepdim=True)

    eps = 1e-8
    q_v_norm = torch.clamp(q_v_norm, min=eps)

    theta = 2 * torch.atan2(q_v_norm, q[:, 0:1])
    q_log = theta * q_v / q_v_norm
    return q_log

def compute_loss(pred, gt):
    pos_pred, quat_pred, vel_pred = pred[:, :3], pred[:, 3:7], pred[:, 7:10]
    pos_gt, quat_gt, vel_gt = gt[:, :3], gt[:, 3:7], gt[:, 7:10]

    # Normalize predicted quaternion
    quat_pred = quat_pred / quat_pred.norm(dim=1, keepdim=True)

    # Position loss (MSE)
    pos_loss = F.mse_loss(pos_pred, pos_gt)
    vel_loss = F.mse_loss(vel_pred, vel_gt)

    # Quaternion loss (log map in tangent space)

    # Inverse of predicted quaternion
    quat_pred_inv = torch.cat((quat_pred[:, 0:1], -quat_pred[:, 1:]), dim=1)

    # Quaternion error: q_err = q_gt * q_pred^{-1}
    q_err = quaternion_product(quat_gt, quat_pred_inv)

    # Log map to tangent space
    q_log = quaternion_log(q_err)

    # Squared norm (theta^2), averaged over the batch
    quat_loss = q_log.pow(2).sum(dim=1).mean()

    total_loss = pos_loss + quat_loss + vel_loss

    return total_loss, pos_loss, quat_loss

# --------------------- Training ---------------------------

def train_one_epoch(model, loader, optimizer, quat_weight):
    model.train()
    total_loss = 0.0
    pos_total_loss = 0.0
    quat_total_loss = 0.0

    for imu_tensor, gt in tqdm(loader, desc="Training"):
        imu_tensor = imu_tensor.to(DEVICE)
        gt = gt.to(DEVICE)

        optimizer.zero_grad()
        pred = model(imu_tensor)
        loss, pos_loss, quat_loss = compute_loss(pred, gt)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pos_total_loss += pos_loss.item()
        quat_total_loss += quat_loss.item()

    N = len(loader)
    return total_loss / N, pos_total_loss / N, quat_total_loss / N

def validate(model, loader, quat_weight):
    model.eval()
    total_loss = 0.0
    pos_total_loss = 0.0
    quat_total_loss = 0.0

    with torch.no_grad():
        for imu_tensor, gt in tqdm(loader, desc="Validation"):
            imu_tensor = imu_tensor.to(DEVICE)
            gt = gt.to(DEVICE)

            pred = model(imu_tensor)
            loss, pos_loss, quat_loss = compute_loss(pred, gt)

            total_loss += loss.item()
            pos_total_loss += pos_loss.item()
            quat_total_loss += quat_loss.item()

    N = len(loader)
    return total_loss / N, pos_total_loss / N, quat_total_loss / N

# --------------------- Main ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="IO/", help="Log Directory")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--quat_weight", type=float, default=1.0)  # Weight for quaternion loss
    args = parser.parse_args()

    log_dir = os.path.join('/home/fer/Lectures/computer_vision/Learning_vio/', args.log_dir)
    ckpt_path = '/home/fer/Lectures/computer_vision/Learning_vio/ckpts'

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(log_dir, "runs"))

    # ----- Data -----
    train_dataset = IMUDatasetFromMat("liss.mat")
    val_dataset = IMUDatasetFromMat("liss.mat")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # ----- Model -----
    model = PoseNet(
        input_dim=12,
        output_dim=9,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_layers=args.num_layers
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")

    # ----- Training Loop -----
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")

        train_loss, train_pos_loss, train_quat_loss = train_one_epoch(model, train_loader, optimizer, args.quat_weight)
        val_loss, val_pos_loss, val_quat_loss = validate(model, val_loader, args.quat_weight)

        print(f"Train Pos Loss: {train_pos_loss:.4f} | Train Quat Loss: {train_quat_loss:.4f}")
        print(f"Val   Pos Loss: {val_pos_loss:.4f} | Val   Quat Loss: {val_quat_loss:.4f}")

        # ---- TensorBoard logging ----
        writer.add_scalar("Loss/Train_Total", train_loss, epoch)
        writer.add_scalar("Loss/Train_Position", train_pos_loss, epoch)
        writer.add_scalar("Loss/Train_Quaternion", train_quat_loss, epoch)
        writer.add_scalar("Loss/Validation_Total", val_loss, epoch)
        writer.add_scalar("Loss/Validation_Position", val_pos_loss, epoch)
        writer.add_scalar("Loss/Validation_Quaternion", val_quat_loss, epoch)

        # ---- Save best model ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(ckpt_path, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with val loss {best_val_loss:.4f}")

    writer.close()

if __name__ == "__main__":
    main()