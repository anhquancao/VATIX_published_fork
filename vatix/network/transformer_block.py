import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, dim, h_dim, multiple_of=256, bias=False, dropout=0.):
        super().__init__()
        self.dropout = dropout
        # swinGLU
        h_dim = int(2 * h_dim / 3)
        # make sure it is a power of 256
        h_dim = multiple_of * ((h_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, h_dim, bias=bias)
        self.w2 = nn.Linear(h_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, h_dim, bias=bias)

    def forward(self, x):
        # SwiGLU activation
        x = F.silu(self.w1(x)) * self.w3(x)
        if self.dropout > 0. and self.training:
            x = F.dropout(x, self.dropout)

        return self.w2(x)

class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim, linear=False, bias=False)
        self.key_norm = RMSNorm(dim, linear=False, bias=False)

    def forward(self, q, k, v):
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=False):
        super().__init__()
        self.n_local_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.wq = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.wk = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.wv = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.wo = nn.Linear(num_heads * self.head_dim, embed_dim, bias=bias)

        self.qk_norm = QKNorm(num_heads * self.head_dim)

        # will be KVCache object managed by inference context manager
        self.cache = None

    def forward(self, x, mask=None, return_attn=False):
        b, h_w, _ = x.shape
        # calculate query, key, value and split out heads
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # normalize queries and keys
        xq, xk = self.qk_norm(xq, xk, xv)
        xq = xq.view(b, h_w, self.n_local_heads, self.head_dim)
        xk = xk.view(b, h_w, self.n_local_heads, self.head_dim)
        xv = xv.view(b, h_w, self.n_local_heads, self.head_dim)

        # make heads be a batch dim
        xq, xk, xv = (x.transpose(1, 2) for x in (xq, xk, xv))
        # attention

        if mask is not None:
            # mask: (L, L) or (B, L, L)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)          # (1, L, L)
            mask = mask.unsqueeze(1)              # (B, 1, L, L)
        output = F.scaled_dot_product_attention(xq, xk, xv, mask, dropout_p=self.dropout if self.training else 0.)
        
        # concatenate all the heads
        output = output.transpose(1, 2).contiguous().view(b, h_w, -1)
        # output projection
        proj = self.wo(output)
        if self.dropout > 0. and self.training:
            proj = F.dropout(proj, self.dropout)

        return proj

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, linear=True, bias=True):
        super().__init__()
        self.eps = eps
        self.linear = linear
        self.add_bias = bias
        if self.linear:
            self.weight = nn.Parameter(torch.ones(dim))
        if self.add_bias:
            self.bias = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.linear:
            output = self.weight * output
        if self.add_bias:
            output = output + self.bias
        return output

class TimestepEmbedder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim * 4),
            nn.SiLU(),
            nn.Linear(in_dim * 4, out_dim)
        )

    def forward(self, t):
        half = self.mlp[0].in_features // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        freqs = freqs[None, None, :]  # (1, 1, half)
        t_ = t[:, :, None]            # (B, T, 1)
        freqs = t_ * freqs            # (B, T, half)
        emb = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)
        return self.mlp(emb)
    
