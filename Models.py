import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import pdb

# ------------ Fourier Feature Lifting ------------------

class FourierLifting(nn.Module):
    def __init__(self, input_dim, num_frequencies=20):
        super(FourierLifting, self).__init__()
        self.num_frequencies = num_frequencies
        self.register_buffer('freq_bands', 2.0 ** torch.arange(0, num_frequencies) * torch.pi)

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        """
        batch_size, seq_len, input_dim = x.shape
        x_expanded = []

        for f in self.freq_bands:
            x_expanded.append(torch.sin(f * x))
            x_expanded.append(torch.cos(f * x))

        x_expanded = torch.cat(x_expanded, dim=-1)
        return torch.cat([x, x_expanded], dim=-1)

# ------------ Stereographic Projection ------------------

def stereographic_projection_to_quaternion(r):
    """
    r: (batch_size, 3) Euclidean vector → quaternion
    Returns:
        q: (batch_size, 4) quaternion [w, x, y, z]
    """
    norm_sq = torch.sum(r ** 2, dim=1, keepdim=True)
    denom = 1 + norm_sq

    q0 = (1 - norm_sq) / denom
    q_xyz = 2 * r / denom

    q = torch.cat([q0, q_xyz], dim=1)
    q = q / q.norm(dim=1, keepdim=True)  # Ensure numerical unit norm
    return q

# ------------ IMUNet ------------------

class IMUNet(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=256, output_dim=6, num_layers=2, dropout_rate=0.2, num_frequencies=20):
        super(IMUNet, self).__init__()

        self.lifting = FourierLifting(input_dim=input_dim, num_frequencies=num_frequencies)
        lifted_dim = input_dim + input_dim * 2 * num_frequencies

        self.lstm1 = nn.LSTM(
            input_size=lifted_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_rate,
            bidirectional=True,
            batch_first=True
        )
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out_layer = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.lifting(x)
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.out_layer(out)
        return out

# ------------ PoseNet ------------------

class PoseNet(nn.Module):
    def __init__(self, input_dim=6, output_dim=7, hidden_dim=256, dropout=0.2, num_layers=2, num_frequencies=20):
        super(PoseNet, self).__init__()
        self.net = IMUNet(
            input_dim=input_dim,
            output_dim=6,  # 3 pos + 3 stereographic vector
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout,
            num_frequencies=num_frequencies
        )

    def forward(self, x):
        x = self.net(x)

        # Scale velocity appropriately
        velocity_scale = 10.0  # Choose based on your dataset
        pos = velocity_scale * torch.tanh(x[:, :3])

        stereo_vec = x[:, 3:]
        q = stereographic_projection_to_quaternion(stereo_vec)

        return torch.cat([pos, q], dim=1)