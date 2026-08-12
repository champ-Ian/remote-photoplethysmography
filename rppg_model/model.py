import torch
import torch.nn as nn


class RPPGNet(nn.Module):
    def __init__(self, in_channels=3, input_length=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        # x: B x T x C  -> convert to B x C x T
        x = x.permute(0, 2, 1)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out.squeeze(1)
