import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class FourierLifting(nn.Module):
    def __init__(self, input_dim, num_frequencies=20):
        super(FourierLifting, self).__init__()
        self.num_frequencies = num_frequencies
        self.register_buffer('freq_bands', 2.0 ** torch.arange(0, num_frequencies) * torch.pi)

    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        x_expanded = []
        for f in self.freq_bands:
            x_expanded.append(torch.sin(f * x))
            x_expanded.append(torch.cos(f * x))
        x_expanded = torch.cat(x_expanded, dim=-1)
        return torch.cat([x, x_expanded], dim=-1)

def stereographic_projection_to_quaternion(r):
    norm_sq = torch.sum(r ** 2, dim=1, keepdim=True)
    denom = 1 + norm_sq
    q0 = (1 - norm_sq) / denom
    q_xyz = 2 * r / denom
    q = torch.cat([q0, q_xyz], dim=1)
    q = q / q.norm(dim=1, keepdim=True)
    return q
# ------------ IMUNet ------------------
class IMUNet(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=256, output_dim=9, num_layers=2, dropout_rate=0.2, num_frequencies=20):
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
        out = out[:, -1, :]  # Last timestep
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.out_layer(out)
        return out
import torchvision.models as models

class PoseNet(nn.Module):
    def __init__(self, input_dim=12, output_dim=9, hidden_dim=256, dropout=0.2, num_layers=2, num_frequencies=50):
        super(PoseNet, self).__init__()

        # --- IMU network ---
        self.net = IMUNet(
            input_dim=input_dim,
            output_dim=hidden_dim * 2,  # Instead of final prediction, output feature vector
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout,
            num_frequencies=num_frequencies
        )

        # --- Image encoder ---
        resnet = models.resnet18(pretrained=True)
        resnet.fc = nn.Identity()  # Remove classification layer
        self.image_encoder = resnet  # Output: 512-dim feature vector

        # --- Fusion ---
        self.fc_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)  # output_dim = 9: pos(3), stereo_vec(3), vel(3)
        )

    def forward(self, imu_input, img_input):
        """
        imu_input: (batch, window, 12)
        img_input: (batch, 3, H, W)
        """
        # ---- IMU feature ----
        lifted = self.net.lifting(imu_input)
        lstm_out, _ = self.net.lstm1(lifted)
        lstm_out2, _ = self.net.lstm2(lstm_out)
        imu_feat = lstm_out2[:, -1, :]  # Last time step output

        # ---- Image feature ----
        img_feat = self.image_encoder(img_input)  # Shape: (batch, 512)

        # ---- Fuse ----
        fused = torch.cat([imu_feat, img_feat], dim=1)
        out = self.fc_fuse(fused)

        # ---- Parse outputs ----
        pos = 10.0 * torch.tanh(out[:, :3])  # Position (bounded)
        stereo_vec = out[:, 3:6]
        vel = 10.0 * torch.tanh(out[:, 6:9])  # Velocity (bounded)

        q = stereographic_projection_to_quaternion(stereo_vec)

        return torch.cat([pos, q, vel], dim=1)