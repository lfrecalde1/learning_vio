import torchvision.models as models
import torch.nn as nn
import torch
import torch.nn.functional as F

class FourierLifting(nn.Module):
    def __init__(self, input_dim, num_frequencies=10):
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

class CrossAttention(nn.Module):
    def __init__(self, dim):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, imu_feat, img_feat):
        Q = self.query(imu_feat).unsqueeze(1)
        K = self.key(img_feat).unsqueeze(1)
        V = self.value(img_feat).unsqueeze(1)

        attn_scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attended = torch.bmm(attn_weights, V).squeeze(1)
        return attended

class IMUNet(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=256, num_layers=2, dropout_rate=0.2, num_frequencies=10):
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

    def forward(self, x):
        x = self.lifting(x)
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        return out[:, -1, :]  # Last timestep

class PoseNet(nn.Module):
    def __init__(self, input_dim=12, output_dim=9, hidden_dim=256, dropout=0.2, num_layers=2, num_frequencies=10):
        super(PoseNet, self).__init__()

        self.imu_net = IMUNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout,
            num_frequencies=num_frequencies
        )

        resnet = models.resnet18(pretrained=True)
        resnet.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Accept 6-channel input
        resnet.fc = nn.Identity()
        self.image_encoder = resnet

        D_common = 256
        self.imu_proj = nn.Linear(hidden_dim * 2, D_common)
        self.img_proj = nn.Linear(512, D_common)
        self.cross_attention = CrossAttention(D_common)

        self.fc_fuse = nn.Sequential(
            nn.Linear(D_common, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, imu_input, img_input):
        imu_feat = self.imu_net(imu_input)
        img_feat = self.image_encoder(img_input)

        imu_proj = self.imu_proj(imu_feat)
        img_proj = self.img_proj(img_feat)

        fused = self.cross_attention(imu_proj, img_proj)
        out = self.fc_fuse(fused)

        pos = 10.0 * torch.tanh(out[:, :3])
        stereo_vec = out[:, 3:6]
        vel = 10.0 * torch.tanh(out[:, 6:9])

        q = stereographic_projection_to_quaternion(stereo_vec)

        return torch.cat([pos, q, vel], dim=1)