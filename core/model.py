import math
import torch
import torch.nn as nn



class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int = 32):
        super().__init__()
        self.emb = nn.Embedding(num_positions, d_model)

    def forward(self, x):
        # x: [B, T, d_model]
        T = x.size(1)
        positions = torch.arange(T, device=x.device)
        pos_emb = self.emb(positions).unsqueeze(0)  # [1, T, d_model]
        return x + pos_emb


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, d_internal: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        assert d_internal % num_heads == 0, "d_internal must be divisible by num_heads"

        self.d_model = d_model
        self.d_internal = d_internal
        self.num_heads = num_heads
        self.head_dim = d_internal // num_heads

        # Project from model dim -> full internal dim
        self.W_q = nn.Linear(d_model, d_internal)
        self.W_k = nn.Linear(d_model, d_internal)
        self.W_v = nn.Linear(d_model, d_internal)

        # Project concatenated heads back to model dim
        self.out_proj = nn.Linear(d_internal, d_model)

        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, mask=None):
        # x: [B, T, d_model]
        B, T, _ = x.shape

        Q = self.W_q(x)  # [B, T, d_internal]
        K = self.W_k(x)  # [B, T, d_internal]
        V = self.W_v(x)  # [B, T, d_internal]

        # Split into heads
        # [B, T, d_internal] -> [B, T, num_heads, head_dim] -> [B, num_heads, T, head_dim]
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores: [B, num_heads, T, T]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            # mask should be broadcastable to [B, num_heads, T, T]
            # common shapes:
            # [B, 1, 1, T] for padding mask
            # [1, 1, T, T] for causal mask
            # or combined [B, 1, T, T]
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # [B, num_heads, T, T] @ [B, num_heads, T, head_dim]
        # -> [B, num_heads, T, head_dim]
        attn_out = torch.matmul(attn, V)

        # Recombine heads:
        # [B, num_heads, T, head_dim] -> [B, T, num_heads, head_dim] -> [B, T, d_internal]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_internal)

        # Back to model dim
        out = self.out_proj(attn_out)  # [B, T, d_model]
        return out


class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, d_internal: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        self.self_attn = MultiHeadSelfAttention(
            d_model=d_model,
            d_internal=d_internal,
            num_heads=num_heads,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: [B, T, d_model]
        attn_out = self.self_attn(x, mask=mask)      # [B, T, d_model]
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)                        # [B, T, d_model]
        x = self.norm2(x + self.dropout(ffn_out))

        return x


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        num_positions: int = 32,
        d_model: int = 128,
        d_internal: int = 256,
        num_heads: int = 8,
        num_classes: int = 3,
        dropout: float = 0.1,
        layers: int = 1
    ):
        super().__init__()

        assert d_internal % num_heads == 0, "d_internal must be divisible by num_heads"

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(d_model, num_positions)

        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model=d_model,
                d_internal=d_internal,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(layers)
        ])

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attention_mask=None):
        # input_ids: [B, T]
        x = self.embedding(input_ids)                # [B, T, d_model]
        x = self.positional_encoding(x)              # [B, T, d_model]

        B, T = input_ids.shape
        attn_mask = None

        if attention_mask is not None:
            # padding mask: [B, T] -> [B, 1, 1, T]
            # this masks which key positions can be attended to
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        for layer in self.layers:
            x = layer(x, mask=attn_mask)             # [B, T, d_model]

        cls_repr = x[:, 0, :]                        # [B, d_model]
        logits = self.classifier(cls_repr)           # [B, num_classes]
        return logits