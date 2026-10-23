"""
Reusable reduced-input neural architectures: TFT-inspired encoder, BiLSTM-Transformer and CNN-GRU.

Revision v2 calls these classes from src/revision/experiment.py with nine
15-minute history boundaries and either four channels (CGM, HR, time sine/cosine)
or three channels (CGM and time sine/cosine). Historical feature builders and
WINDOW below describe the legacy runner, not the revision-v2 experiment.
These variants do not reproduce or verify unavailable original six-modal weights.
"""

import numpy as np
import torch
import torch.nn as nn

WINDOW = 288  # 24h at 5-min sampling, per manuscript Sec 5.1 (TFT lookback)
QUANTILES = (0.1, 0.5, 0.9)


def pinball_loss(pred_q, target, quantiles=QUANTILES):
    """pred_q: [B, H, Q], target: [B, H]. Asymmetric quantile (pinball) loss, averaged."""
    target = target.unsqueeze(-1)  # [B, H, 1]
    losses = []
    for i, q in enumerate(quantiles):
        err = target[..., 0] - pred_q[..., i]
        losses.append(torch.maximum(q * err, (q - 1) * err))
    return torch.stack(losses, dim=-1).mean()


class GatedResidualNetwork(nn.Module):
    """GRN: the core building block of TFT (Lim et al. 2021) -- a gated skip connection
    around a two-layer MLP, with GLU-style gating deciding how much of the nonlinear
    branch to apply."""

    def __init__(self, d_in, d_hidden, d_out, dropout=0.1):
        super().__init__()
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(d_out, d_out * 2)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x):
        residual = self.skip(x)
        h = self.fc2(self.elu(self.fc1(x)))
        h = self.dropout(h)
        gated, gate_logits = self.gate(h).chunk(2, dim=-1)
        h = gated * torch.sigmoid(gate_logits)
        return self.norm(residual + h)


class VariableSelectionNetwork(nn.Module):
    """Per-timestep soft feature selection: a GRN over the flattened features produces
    per-feature weights (softmax), and each feature is independently transformed by its
    own single-feature GRN before the weighted sum. Reduces sensitivity to uninformative
    engineered features (of which the 32-D DDATSHR set has several near-collinear ones,
    e.g. multiple CGM rolling windows)."""

    def __init__(self, num_features, d_hidden, dropout=0.1):
        super().__init__()
        self.num_features = num_features
        self.flatten_grn = GatedResidualNetwork(num_features * d_hidden, d_hidden, num_features, dropout)
        self.feature_grns = nn.ModuleList(
            [GatedResidualNetwork(d_hidden, d_hidden, d_hidden, dropout) for _ in range(num_features)]
        )
        self.feature_proj = nn.Linear(1, d_hidden)

    def forward(self, x):
        # x: [B, T, num_features] -> per-feature scalar projected to d_hidden first
        B, T, F = x.shape
        per_feat = self.feature_proj(x.unsqueeze(-1))  # [B, T, F, d_hidden]
        flat = per_feat.reshape(B, T, F * per_feat.shape[-1])
        weights = torch.softmax(self.flatten_grn(flat), dim=-1)  # [B, T, F]
        transformed = torch.stack(
            [grn(per_feat[:, :, i, :]) for i, grn in enumerate(self.feature_grns)], dim=2
        )  # [B, T, F, d_hidden]
        selected = (transformed * weights.unsqueeze(-1)).sum(dim=2)  # [B, T, d_hidden]
        return selected, weights


class TFT(nn.Module):
    """Simplified encoder-only Temporal Fusion Transformer: variable selection -> LSTM
    encoder -> multi-head self-attention -> GRN -> per-horizon quantile heads.

    The original TFT also models known-future covariates via an LSTM decoder; DDATSHR's
    feature set has no known-future covariates (no scheduled meals/insulin in this
    dataset), so this is an encoder-only variant over the historical window -- a
    legitimate TFT configuration for pure autoregressive forecasting, not a shortcut
    around the architecture.
    """

    def __init__(self, num_features, hidden=64, n_heads=4, d_model=64, dropout=0.1, n_horizons=3):
        super().__init__()
        self.vsn = VariableSelectionNetwork(num_features, hidden, dropout)
        self.encoder_lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.post_attn_grn = GatedResidualNetwork(d_model, hidden, d_model, dropout)
        self.pre_output_grn = GatedResidualNetwork(d_model, hidden, d_model, dropout)
        self.head = nn.Linear(d_model, n_horizons * len(QUANTILES))
        self.n_horizons = n_horizons

    def forward(self, x):
        selected, _ = self.vsn(x)  # [B, T, hidden]
        enc, _ = self.encoder_lstm(selected)  # [B, T, hidden]
        attn_out, _ = self.attn(enc, enc, enc)
        h = self.post_attn_grn(attn_out + enc)
        h = self.pre_output_grn(h)
        last = h[:, -1, :]  # representation at the current (last observed) timestep
        out = self.head(last)  # [B, n_horizons * Q]
        return out.view(-1, self.n_horizons, len(QUANTILES))


class BiLSTMTransformer(nn.Module):
    """2-layer BiLSTM + 2-layer Transformer encoder. Original architecture applies
    cross-modal attention over ECG/HRV and SpO2 at each LSTM timestep; DDATSHR has
    neither signal, so that term has no inputs here and is omitted (self-attention only,
    over the BiLSTM's own outputs) -- see module docstring point 1."""

    def __init__(self, num_features, hidden=64, ffn=256, n_heads=4, dropout=0.1, n_horizons=3):
        super().__init__()
        self.input_proj = nn.Linear(num_features, hidden * 2)
        self.bilstm = nn.LSTM(hidden * 2, hidden, num_layers=2, batch_first=True,
                               bidirectional=True, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden * 2, nhead=n_heads, dim_feedforward=ffn,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Linear(hidden * 2, n_horizons)

    def forward(self, x):
        h = self.input_proj(x)
        h, _ = self.bilstm(h)
        h = self.transformer(h)
        return self.head(h[:, -1, :])  # [B, n_horizons]


class CNNGRU(nn.Module):
    """Two 1D-conv branches over the two modalities DDATSHR actually provides (CGM-
    derived features vs. HR-derived features), concatenated before a 2-layer GRU --
    see module docstring point 2 for why the branch split differs from the original."""

    def __init__(self, cgm_idx, hr_idx, static_idx, hidden=64, dropout=0.1, n_horizons=3):
        super().__init__()
        self.cgm_idx = cgm_idx
        self.hr_idx = hr_idx
        self.static_idx = static_idx
        n_cgm, n_hr, n_static = len(cgm_idx), len(hr_idx), len(static_idx)

        def conv_branch(n_in):
            return nn.Sequential(
                nn.Conv1d(n_in, 32, kernel_size=5, padding=2), nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
                nn.Dropout(dropout),
            )

        self.cgm_conv = conv_branch(max(n_cgm, 1))
        self.hr_conv = conv_branch(max(n_hr, 1))
        gru_input = 64 + 64 + n_static
        self.gru = nn.GRU(gru_input, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, n_horizons)

    def forward(self, x):
        # x: [B, T, F] -> conv branches expect [B, C, T]
        cgm = x[:, :, self.cgm_idx].transpose(1, 2)
        hr = x[:, :, self.hr_idx].transpose(1, 2)
        cgm_h = self.cgm_conv(cgm).transpose(1, 2)  # [B, T, 64]
        hr_h = self.hr_conv(hr).transpose(1, 2)  # [B, T, 64]
        parts = [cgm_h, hr_h]
        if self.static_idx:
            parts.append(x[:, :, self.static_idx])
        merged = torch.cat(parts, dim=-1)
        out, _ = self.gru(merged)
        return self.head(out[:, -1, :])  # [B, n_horizons]


def split_feature_groups(feature_cols):
    """Partitions the 32-D DDATSHR feature vector into CGM-derived, HR-derived, and
    static-demographic index groups, for CNN-GRU's per-modality conv branches."""
    cgm_idx, hr_idx, static_idx = [], [], []
    for i, c in enumerate(feature_cols):
        if c.startswith("cgm_") or c.startswith("glucose") or c in ("tod_sin", "tod_cos"):
            cgm_idx.append(i)
        elif c.startswith("hr_"):
            hr_idx.append(i)
        else:
            static_idx.append(i)
    return cgm_idx, hr_idx, static_idx
