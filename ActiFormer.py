# ActiFormer.py
import torch
import torch.nn as nn
from thop import profile


class SignAwareAttention(nn.Module):
    """
    Implementation of the Sign-Aware Attention mechanism as described in the ActiFormer paper.
    This module decouples query-key interactions into synergistic (same-sign) and
    cross-sign streams, preserving bidirectional signal dynamics.
    It also includes the learnable Entropy-Scaling Function.
    """

    def __init__(self, dim, num_patches, num_heads=2, qkv_bias=True, attn_drop=0.0, proj_drop=0.0, alpha=3.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_patches = num_patches

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        # Value vectors are split for synergic (v_s) and cross-sign (v_c) components [cite: 230]
        self.v_s = nn.Linear(dim, dim // 2, bias=qkv_bias)
        self.v_c = nn.Linear(dim, dim // 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Learnable exponent for the Entropy-Scaling function, as in Eq. (7) [cite: 233]
        self.power = nn.Parameter(torch.ones(1, num_heads, 1, self.head_dim) * 1.0)
        self.alpha = alpha

        # Learnable sign-weighting matrices W^s and W^c [cite: 100, 246]
        self.W_s = nn.Parameter(torch.ones(num_patches, self.head_dim // 2))
        self.W_c = nn.Parameter(torch.ones(num_patches, self.head_dim // 2))
        nn.init.uniform_(self.W_s, 0, 1)  # Initialize with Uniform(0,1) as per paper's findings [cite: 770]
        nn.init.uniform_(self.W_c, 0, 1)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_s = self.v_s(x).reshape(B, N, self.num_heads, self.head_dim // 2).permute(0, 2, 1, 3)
        v_c = self.v_c(x).reshape(B, N, self.num_heads, self.head_dim // 2).permute(0, 2, 1, 3)

        # Decompose queries and keys into positive and negative parts [cite: 213]
        q_pos, q_neg = torch.relu(q), torch.relu(-q)
        k_pos, k_neg = torch.relu(k), torch.relu(-k)

        # Apply Entropy-Scaling Function [cite: 251]
        power_val = 1 + self.alpha * torch.sigmoid(self.power)
        q_pos, q_neg = q_pos ** power_val, q_neg ** power_val
        k_pos, k_neg = k_pos ** power_val, k_neg ** power_val

        # Compute context for synergic and cross-sign streams, as in Eqs. (4) and (5) [cite: 243, 244]
        kv_s = torch.einsum('bhnj,bhjd->bhnd', k_pos, v_s) + torch.einsum('bhnj,bhjd->bhnd', k_neg, v_s)
        q_s_context = torch.einsum('bhni,bhid->bhnd', q_pos, kv_s) + torch.einsum('bhni,bhid->bhnd', q_neg, kv_s)

        kv_c = torch.einsum('bhnj,bhjd->bhnd', k_neg, v_c) + torch.einsum('bhnj,bhjd->bhnd', k_pos, v_c)
        q_c_context = torch.einsum('bhni,bhid->bhnd', q_pos, kv_c) + torch.einsum('bhni,bhid->bhnd', q_neg, kv_c)

        # Normalization
        denom_s = torch.einsum('bhni,bhi->bhn', q_pos, k_pos.sum(dim=-1)) + torch.einsum('bhni,bhi->bhn', q_neg,
                                                                                         k_neg.sum(dim=-1))
        denom_c = torch.einsum('bhni,bhi->bhn', q_pos, k_neg.sum(dim=-1)) + torch.einsum('bhni,bhi->bhn', q_neg,
                                                                                         k_pos.sum(dim=-1))

        attn_s = q_s_context / (denom_s.unsqueeze(-1) + 1e-6)
        attn_c = q_c_context / (denom_c.unsqueeze(-1) + 1e-6)

        # Apply learnable sign-weighting matrices
        attn_s = attn_s * self.W_s
        attn_c = attn_c * self.W_c

        # Concatenate and project
        x = torch.cat([attn_s, attn_c], dim=-1).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ActiFormer(nn.Module):
    """
    The main ActiFormer model architecture, based on the structure in Figure 3 of the paper.
    It uses a convolutional patch embedding followed by a Transformer block containing
    the SignAwareAttention mechanism and a feed-forward network (MLP).
    """

    def __init__(self, input_shape, num_classes):
        super(ActiFormer, self).__init__()
        in_chans, time_steps = input_shape[1], input_shape[2]
        embed_dim = 192  # Typical dimension for 'tiny' models
        num_heads = 4

        # Convolutional Patch Embedding
        self.patch_embed = nn.Conv1d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, time_steps, embed_dim))

        # Main Attention Block
        self.attn_block = nn.Sequential(
            nn.LayerNorm(embed_dim),
            SignAwareAttention(embed_dim, time_steps, num_heads=num_heads),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),  # FFN
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        # Classifier Head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv1d, nn.LayerNorm)):
                nn.init.normal_(m.weight, mean=1.0, std=0.02)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.patch_embed(x).permute(0, 2, 1)
        x = x + self.pos_embed
        x = x + self.attn_block(x)
        x = x.permute(0, 2, 1)  # (B, C, N) for pooling
        return self.head(x)


if __name__ == "__main__":
    input_shape = (64, 6, 128)  # Example for UCI-HAR (Batch, Channels, Timesteps)
    model = ActiFormer(input_shape, num_classes=6).cuda()
    input_tensor = torch.randn(1, *input_shape[1:]).cuda()
    flops, params = profile(model, inputs=(input_tensor,))
    print(f"ActiFormer Params: {params / 1e6:.2f}M, FLOPs: {flops / 1e9:.2f}G")