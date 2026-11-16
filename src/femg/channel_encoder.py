import math

import torch
from torch import nn


class TemporalStem(nn.Module):
    """(N, T) raw EMG of single channels -> (N, T // patch, D)."""

    def __init__(self, dim: int, patch: int):
        super().__init__()
        self.patch = patch
        self.layers = nn.Sequential(
            nn.Conv1d(1, dim, kernel_size=patch, stride=patch),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x.unsqueeze(1)).transpose(1, 2)


class ElectrodeEmbedding(nn.Module):
    """(..., 3) positions (angle deg, z, bipolar) -> (..., D).

    Angle: harmonics 1..n_angle of the angle (periodic, ~360 / (2 n_angle) deg resolution). z: sin/cos at
    n_z octaves. NaN angle or z: the learned `unknown` vector. Bipolar or monopolar: a learned vector added to both."""

    def __init__(self, dim: int, n_angle: int = 6, n_z: int = 4):
        super().__init__()
        self.register_buffer("harmonics", torch.arange(1, n_angle + 1, dtype=torch.float32))
        self.register_buffer("octaves", math.pi * 2.0 ** torch.arange(n_z, dtype=torch.float32))
        self.mlp = nn.Sequential(nn.Linear(2 * n_angle + 2 * n_z, dim), nn.GELU(), nn.Linear(dim, dim))
        self.unknown = nn.Parameter(0.02 * torch.randn(dim))
        self.kind = nn.Embedding(2, dim)

    def features(self, positions: torch.Tensor) -> torch.Tensor:
        angle = torch.deg2rad(positions[..., :1].nan_to_num()) * self.harmonics
        z = positions[..., 1:2].nan_to_num() * self.octaves
        return torch.cat([angle.sin(), angle.cos(), z.sin(), z.cos()], dim=-1)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        known = positions[..., :2].isfinite().all(dim=-1, keepdim=True)
        where = torch.where(known, self.mlp(self.features(positions)), self.unknown)
        return where + self.kind(positions[..., 2].nan_to_num().long())


class ChannelEncoder(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        n_latents: int = 8,
        n_heads: int = 4,
        patch: int = 5,  # 500 Hz EMG -> 100 Hz tokens, the label rate
        n_angle: int = 6,
        n_z: int = 4,
    ):
        super().__init__()
        self.stem = TemporalStem(dim, patch)
        self.embed = ElectrodeEmbedding(dim, n_angle, n_z)
        self.mix = nn.TransformerEncoderLayer(dim, n_heads, 2 * dim, dropout=0.0, batch_first=True, norm_first=True)
        self.queries = nn.Parameter(0.02 * torch.randn(n_latents, dim))
        self.pool = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(
        self,
        emg: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor | None = None,
        calibration: "ElectrodeCalibration | None" = None,
    ) -> torch.Tensor:
        """emg (B, C, T) normalized per channel, positions (B, C, 3), mask (B, C) True = real channel.
        Returns latents (B, T // patch, K, D)."""
        latents, _ = self.pool_channels(emg, positions, mask, calibration)
        return latents

    def attention(self, emg, positions, mask=None, calibration=None) -> torch.Tensor:
        """(B, T', K, C): how much each latent reads each channel."""
        return self.pool_channels(emg, positions, mask, calibration)[1]

    def pool_channels(self, emg, positions, mask, calibration) -> tuple[torch.Tensor, torch.Tensor]:
        if calibration is not None:
            emg, positions = calibration(emg, positions)
        b, c, t = emg.shape
        mask = torch.ones(b, c, dtype=torch.bool, device=emg.device) if mask is None else mask

        tokens = self.stem(emg.reshape(b * c, t))  # (B C, T', D)
        steps, dim = tokens.shape[1:]
        tokens = tokens.reshape(b, c, steps, dim) + self.embed(positions).unsqueeze(2)
        tokens = tokens.transpose(1, 2).reshape(b * steps, c, dim)  # one set of C tokens per time step
        ignore = (~mask).repeat_interleave(steps, dim=0)

        tokens = self.mix(tokens, src_key_padding_mask=ignore)
        queries = self.queries.expand(b * steps, -1, -1)
        latents, weights = self.pool(queries, tokens, tokens, key_padding_mask=ignore)
        latents = latents + self.ffn(self.norm(latents))
        return latents.reshape(b, steps, -1, dim), weights.reshape(b, steps, -1, c)

    def freeze(self) -> "ChannelEncoder":
        for p in self.parameters():
            p.requires_grad_(False)
        return self


class ElectrodeCalibration(nn.Module):
    """Per user: where the band really sits (rotation deg, slide along the forearm) and one gain per channel.
    Starts as the identity; fitted through the frozen encoder on a short calibration recording."""

    def __init__(self, n_channels: int):
        super().__init__()
        self.rotation = nn.Parameter(torch.zeros(()))
        self.slide = nn.Parameter(torch.zeros(()))
        self.log_gain = nn.Parameter(torch.zeros(n_channels))

    def forward(self, emg: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shift = torch.stack([self.rotation, self.slide, torch.zeros_like(self.slide)])
        return emg * self.log_gain.exp()[:, None], positions + shift
