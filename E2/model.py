"""Controlled-capacity multimodal forecasters for E1/E2."""
import torch
from torch import nn
from torch.nn import functional as F


class CausalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.dilation = dilation
        self.projection = nn.Linear(3 * width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, sequence):
        dilation, length = self.dilation, sequence.shape[1]
        padded = F.pad(sequence, (0, 0, 2 * dilation, 0))
        taps = torch.cat((padded[:, :length], padded[:, dilation:dilation + length],
                          padded[:, 2 * dilation:]), dim=-1)
        return sequence + F.silu(self.norm(self.projection(taps)))


class TemporalEncoder(nn.Module):
    def __init__(self, channels, kind):
        super().__init__()
        self.kind = kind
        if kind == "tcn":
            self.network = nn.Sequential(nn.Linear(channels, 32),
                                         *(CausalBlock(32, d) for d in (1, 2, 4, 8)))
        elif kind == "gru":
            self.network = nn.GRU(channels, 32, batch_first=True)
        else:
            raise ValueError(kind)

    def forward(self, sequence):
        if self.kind == "tcn":
            return self.network(sequence)[:, -1]
        _, hidden = self.network(sequence)
        return hidden[-1]


class FrameEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.GroupNorm(6, 48), nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(192, 32), nn.SiLU())

    def forward(self, images):
        return self.network(images)


class VisualEncoder(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        if kind == "cnn_gru":
            self.frame = FrameEncoder()
            self.temporal = nn.GRU(32, 32, batch_first=True)
        elif kind == "3dcnn":
            self.network = nn.Sequential(
                nn.Conv3d(3, 16, (3, 5, 5), padding=(1, 2, 2), stride=(1, 2, 2)),
                nn.GroupNorm(4, 16), nn.SiLU(),
                nn.Conv3d(16, 32, 3, padding=1, stride=(2, 2, 2)),
                nn.GroupNorm(4, 32), nn.SiLU(),
                nn.Conv3d(32, 48, 3, padding=1, stride=(2, 2, 2)),
                nn.GroupNorm(6, 48), nn.SiLU(),
                nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(48, 32), nn.SiLU())
        else:
            raise ValueError(kind)

    def forward(self, images):
        if self.kind == "cnn_gru":
            batch, steps = images.shape[:2]
            encoded = self.frame(images.flatten(0, 1)).reshape(batch, steps, 32)
            _, hidden = self.temporal(encoded)
            return hidden[-1]
        return self.network(images.transpose(1, 2))


class Forecaster(nn.Module):
    def __init__(self, channels, leads, temporal="tcn", visual="cnn_gru", fusion="concat"):
        super().__init__()
        self.temporal = TemporalEncoder(channels, temporal)
        self.visual = VisualEncoder(visual)
        self.fusion = fusion
        self.gate = nn.Sequential(nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 2))
        self.head = nn.Sequential(nn.Linear(64 + leads, 64), nn.SiLU(), nn.Dropout(.1),
                                  nn.Linear(64, leads))

    def forward(self, sequence, images, future_clear, use_images):
        temporal = self.temporal(sequence)
        visual = self.visual(images) if use_images else torch.zeros_like(temporal)
        if self.fusion == "gate":
            weights = self.gate(torch.cat((temporal, visual), dim=1)).sigmoid()
            fused = torch.cat((weights[:, :1] * temporal, weights[:, 1:] * visual), dim=1)
        else:
            fused = torch.cat((temporal, visual), dim=1)
        return self.head(torch.cat((fused, future_clear), dim=1))

