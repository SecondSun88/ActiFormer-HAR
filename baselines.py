# baselines.py
import torch
import torch.nn as nn
import math


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# EXISTING BASELINES (CNN, Transformer, Performer)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class CNN_Baseline(nn.Module):
    """CNN baseline architecture as shown in Figure 3."""

    def __init__(self, input_shape, num_classes):
        super(CNN_Baseline, self).__init__()
        in_chans = input_shape[1]

        self.cnn_block = nn.Sequential(
            nn.Conv1d(in_chans, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Global Average Pooling (GAP)
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.cnn_block(x)
        return self.head(x)


class Transformer_Baseline(nn.Module):
    """Standard Transformer baseline architecture as shown in Figure 3."""

    def __init__(self, input_shape, num_classes):
        super(Transformer_Baseline, self).__init__()
        in_chans, time_steps = input_shape[1], input_shape[2]
        embed_dim = 192
        num_heads = 4

        self.patch_embed = nn.Conv1d(in_chans, embed_dim, kernel_size=3, padding=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, time_steps, embed_dim))

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.Linear(embed_dim * 2, embed_dim))
        self.norm2 = nn.LayerNorm(embed_dim)

        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(embed_dim, num_classes))

    def forward(self, x):
        x = self.patch_embed(x).permute(0, 2, 1)  # (B, N, C)
        x = x + self.pos_embed
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        x = x.permute(0, 2, 1)  # (B, C, N) for pooling
        return self.head(x)


class Performer_Baseline(nn.Module):
    """
    Performer (Linear Attention) baseline. It replaces the softmax attention in the
    standard Transformer with a linear attention using ReLU feature maps.
    """

    def __init__(self, input_shape, num_classes):
        super(Performer_Baseline, self).__init__()
        in_chans, time_steps = input_shape[1], input_shape[2]
        embed_dim = 192
        num_heads = 4
        self.head_dim = embed_dim // num_heads

        self.patch_embed = nn.Conv1d(in_chans, embed_dim, kernel_size=3, padding=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, time_steps, embed_dim))

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.Linear(embed_dim * 2, embed_dim))
        self.norm2 = nn.LayerNorm(embed_dim)

        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(embed_dim, num_classes))

    def forward(self, x):
        B, C, N = x.shape
        x = self.patch_embed(x).permute(0, 2, 1)  # (B, N, C)
        x = x + self.pos_embed

        # Linear Attention Block
        x_norm = self.norm1(x)
        q, k, v = self.qkv(x_norm).chunk(3, dim=-1)

        # ReLU feature map for linear attention
        q, k = torch.relu(q), torch.relu(k)

        kv = torch.einsum('bnd,bne->bde', k, v)
        qkv = torch.einsum('bnd,bde->bne', q, kv)
        denom = torch.einsum('bnd->bn', q).sum(dim=-1, keepdim=True).unsqueeze(-1)
        attn_output = qkv / (denom + 1e-6)
        x = x + self.proj(attn_output)

        x = x + self.mlp(self.norm2(x))
        x = x.permute(0, 2, 1)  # (B, C, N) for pooling
        return self.head(x)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# NEW BASELINES (Informer, Lite Transformer)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class ProbSparseAttention(nn.Module):
    """
    ProbSparse Attention mechanism used in Informer. It selects the top-k queries
    [cite_start]to reduce complexity from O(N^2) to O(N log N)[cite: 407, 408].
    """

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

    def forward(self, q, k, v):
        B, N, _ = q.shape
        # Project and reshape for multi-head
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate scores and select top-k
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Sparsity: select top-k queries (simplified for clarity)
        top_k = int(math.log(N) * 2)
        top_scores, _ = scores.topk(top_k, dim=-1, largest=True)

        # Mask out non-top-k scores
        mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top_scores.long(), True)
        scores.masked_fill_(~mask, -float('inf'))

        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).reshape(B, N, -1)
        return context


class Informer_Baseline(nn.Module):
    """Informer baseline architecture, using ProbSparseAttention."""

    def __init__(self, input_shape, num_classes):
        super(Informer_Baseline, self).__init__()
        in_chans, time_steps = input_shape[1], input_shape[2]
        embed_dim = 192
        num_heads = 4

        self.patch_embed = nn.Conv1d(in_chans, embed_dim, kernel_size=3, padding=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, time_steps, embed_dim))

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn = ProbSparseAttention(embed_dim, num_heads)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.Linear(embed_dim * 2, embed_dim))
        self.norm2 = nn.LayerNorm(embed_dim)

        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(embed_dim, num_classes))

    def forward(self, x):
        x = self.patch_embed(x).permute(0, 2, 1)
        x = x + self.pos_embed

        x_norm = self.norm1(x)
        q, k, v = self.qkv(x_norm).chunk(3, dim=-1)
        attn_output = self.attn(q, k, v)
        x = x + self.proj(attn_output)

        x = x + self.mlp(self.norm2(x))
        x = x.permute(0, 2, 1)
        return self.head(x)


class DecoupledAttention(nn.Module):
    """
    Decoupled Attention for Lite Transformer. This improves efficiency by decoupling
    [cite_start]self-attention across the channel and spatial dimensions[cite: 409].
    """

    def __init__(self, dim, kernel_size=3):
        super().__init__()
        # Depth-wise conv for spatial/temporal interaction
        self.depth_wise_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size,
                                         padding=(kernel_size - 1) // 2, groups=dim)
        # Point-wise conv for channel interaction
        self.point_wise_conv = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        # Input shape for Conv1d: (B, C, N)
        x = x.permute(0, 2, 1)
        x = self.depth_wise_conv(x)
        x = self.point_wise_conv(x)
        # Output shape back to (B, N, C)
        return x.permute(0, 2, 1)


class LiteTransformer_Baseline(nn.Module):
    """Lite Transformer baseline architecture, using Decoupled Attention."""

    def __init__(self, input_shape, num_classes):
        super(LiteTransformer_Baseline, self).__init__()
        in_chans, time_steps = input_shape[1], input_shape[2]
        embed_dim = 192

        self.patch_embed = nn.Conv1d(in_chans, embed_dim, kernel_size=3, padding=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, time_steps, embed_dim))

        self.attn = DecoupledAttention(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.Linear(embed_dim * 2, embed_dim))
        self.norm2 = nn.LayerNorm(embed_dim)

        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(embed_dim, num_classes))

    def forward(self, x):
        x = self.patch_embed(x).permute(0, 2, 1)
        x = x + self.pos_embed

        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        x = x.permute(0, 2, 1)
        return self.head(x)