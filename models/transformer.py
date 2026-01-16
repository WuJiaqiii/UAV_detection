import torch
import torch.nn as nn
import torch.nn.functional as F


class RelPosBias2D(nn.Module):
    """
    2D relative position -> attention bias (per head).
    Given token centers (t, f), build bias_{i,j} = MLP([t_i - t_j, f_i - f_j]) for each head.

    Output shape will be (B*nhead, S, S) which can be fed into nn.MultiheadAttention(attn_mask=...).
    """
    def __init__(self, nhead: int, hidden: int = 64, clamp: float = 5.0):
        super().__init__()
        self.nhead = nhead
        self.clamp = float(clamp) if clamp is not None else None
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, nhead),
        )

    @staticmethod
    def _masked_standardize(x: torch.Tensor, key_padding_mask: torch.Tensor | None, eps: float = 1e-6):
        """
        x: (B,S)
        key_padding_mask: (B,S), True for padding (ignored)
        Return standardized x with masked mean/std (padding does not affect statistics).
        """
        if key_padding_mask is None:
            mean = x.mean(dim=1, keepdim=True)
            var = (x - mean).pow(2).mean(dim=1, keepdim=True)
            std = torch.sqrt(var + eps)
            return (x - mean) / std

        valid = (~key_padding_mask).to(dtype=x.dtype)  # (B,S), 1 for valid tokens
        cnt = valid.sum(dim=1, keepdim=True).clamp_min(1.0)

        mean = (x * valid).sum(dim=1, keepdim=True) / cnt
        var = ((x - mean).pow(2) * valid).sum(dim=1, keepdim=True) / cnt
        std = torch.sqrt(var + eps)
        return (x - mean) / std

    def forward(self, t: torch.Tensor, f: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        """
        t,f: (B,S) float tensors
        key_padding_mask: (B,S) bool, True for padding
        Returns:
          attn_bias: (B*nhead, S, S) float tensor (additive bias to attention logits)
        """
        # 标准化到“相对尺度”，避免 Hz/秒的量纲过大导致 MLP 难学
        t = self._masked_standardize(t, key_padding_mask)
        f = self._masked_standardize(f, key_padding_mask)

        # dt/df: (B,S,S)
        dt = t[:, :, None] - t[:, None, :]
        df = f[:, :, None] - f[:, None, :]

        rel = torch.stack([dt, df], dim=-1)  # (B,S,S,2)
        bias = self.mlp(rel)                 # (B,S,S,nhead)

        if self.clamp is not None:
            bias = bias.clamp(min=-self.clamp, max=self.clamp)

        # (B,nhead,S,S) -> (B*nhead,S,S) 供 MultiheadAttention 使用（3D attn_mask）
        bias = bias.permute(0, 3, 1, 2).contiguous()
        B, H, S, _ = bias.shape
        return bias.view(B * H, S, S)


class TransformerEncoderLayer(nn.Module):
    """A single Transformer encoder layer (Self-Attention + Feed-Forward)."""
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.nhead = nhead
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout_final = nn.Dropout(dropout)

    def forward(self, src, src_key_padding_mask=None, attn_bias=None):
        """
        src: (B,S,d_model)
        src_key_padding_mask: (B,S) bool, True where padded
        attn_bias: (B*nhead,S,S) float additive bias (optional)
        """
        # MultiheadAttention supports:
        #   attn_mask: (S,S) or (B*nhead,S,S) for batch_first=True (PyTorch 1.12+ / 2.x stable)
        attn_output, _ = self.self_attn(
            src, src, src,
            key_padding_mask=src_key_padding_mask,
            attn_mask=attn_bias
        )
        src = self.norm1(src + self.dropout_attn(attn_output))

        ff_output = self.linear2(self.dropout_ff(F.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout_final(ff_output))
        return src


class SignalTransformerClassifier(nn.Module):
    def __init__(
        self,
        feature_dim, d_model, nhead, num_layers, num_classes,
        dim_feedforward=None, dropout=0.1, pooling='mean',
        # ===== new: 2D relative position bias config =====
        use_rel_pos_bias: bool = True,
        rel_pos_hidden: int = 64,
        rel_pos_clamp: float = 5.0,
        pos_t_index: int = 0,
        pos_f_index: int = 1,
    ):
        """
        Transformer-based classifier for signal token sequences.

        New:
          - use_rel_pos_bias: enable 2D relative position (Δt, Δf) attention bias.
          - pos_t_index/pos_f_index: indices of (time_center, freq_center) in token features.
        """
        super(SignalTransformerClassifier, self).__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.d_model = d_model
        self.nhead = nhead

        # Input feature -> d_model embedding
        self.feature_embed = nn.Linear(feature_dim, d_model)

        # 2D relative position bias
        self.use_rel_pos_bias = bool(use_rel_pos_bias)
        self.pos_t_index = int(pos_t_index)
        self.pos_f_index = int(pos_f_index)
        if self.use_rel_pos_bias:
            self.relpos = RelPosBias2D(nhead=nhead, hidden=rel_pos_hidden, clamp=rel_pos_clamp)
        else:
            self.relpos = None

        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Pooling method
        self.pooling = pooling
        if pooling == 'attn':
            self.query_vector = nn.Parameter(torch.randn(d_model))

        # Classification head
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, x, src_key_padding_mask=None):
        """
        x: (B,S,feature_dim) raw tokens (包含时间/频率等特征)
        src_key_padding_mask: (B,S) bool, True where padded
        """
        # 计算 attention bias（基于原始 tokens 的 t,f），然后再做 embedding
        attn_bias = None
        if self.use_rel_pos_bias and self.relpos is not None:
            # 取出 t/f（要求 token 里包含这两维）
            t = x[..., self.pos_t_index].to(dtype=torch.float32)
            f = x[..., self.pos_f_index].to(dtype=torch.float32)
            attn_bias = self.relpos(t, f, key_padding_mask=src_key_padding_mask)
            # attn_bias: (B*nhead,S,S)

        # Project input features to model dimension
        x = self.feature_embed(x)  # (B,S,d_model)

        # Transformer encoder with bias
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask, attn_bias=attn_bias)

        # Pooling
        if self.pooling == 'mean':
            if src_key_padding_mask is not None:
                mask_inv = (~src_key_padding_mask)  # True for real data
                mask_expand = mask_inv.unsqueeze(-1).type_as(x)
                x_masked = x * mask_expand
                sum_vec = x_masked.sum(dim=1)
                count = mask_expand.sum(dim=1)
                count[count == 0] = 1
                rep = sum_vec / count
            else:
                rep = x.mean(dim=1)

        elif self.pooling == 'attn':
            scores = (x * self.query_vector).sum(dim=-1)
            if src_key_padding_mask is not None:
                scores = scores.masked_fill(src_key_padding_mask, float('-inf'))
            attn_weights = F.softmax(scores, dim=1)
            rep = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)

        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        logits = self.fc_out(rep)
        return logits
