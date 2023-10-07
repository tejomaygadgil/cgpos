"""
Define transformers model.
"""
# Author: Tejomay Gadgil <tejomaygadgil@gmail.com>
import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """
    Self-attention decoder (unidirectional).
    """

    def __init__(self, head_size, n_emb, block_size, dropout):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        # Compute K, Q, V
        k = self.key(x)  # What am I advertising?
        q = self.query(x)  # What am I interested in?
        v = self.value(x)  # What will I actually give you if we match?
        # Compute attention score ("affinities")
        wei = q @ k.transpose(-2, -1) * (self.head_size**-0.5)  # Normalized attention
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # Mask
        wei = F.softmax(wei, dim=-1)  # Normalize
        wei = self.dropout(wei)
        # Aggregate values by attention scores
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """
    Compute multiple self-attention heads in parallel.
    """

    def __init__(self, num_heads, head_size, n_emb, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_emb, block_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(head_size * num_heads, n_emb)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """
    Simple MLP.
    """

    def __init__(self, n_emb, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.ReLU(),
            nn.Linear(4 * n_emb, n_emb),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    Transformer block: communication -> computation.
    """

    def __init__(self, n_emb, n_head, block_size, dropout):
        super().__init__()
        head_size = n_emb // n_head
        self.ln1 = nn.LayerNorm(n_emb)  # Prenorm formulation
        self.sa = MultiHeadAttention(n_head, head_size, n_emb, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_emb)  # Prenorm formulation
        self.ffwd = FeedForward(n_emb, dropout)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # Skip connections
        x = x + self.ffwd(self.ln2(x))
        return x


class Transformer(nn.Module):
    """
    Predicts next character using only input character embeddings.
    """

    def __init__(self, vocab_size, block_size, n_layer, n_head, n_emb, dropout):
        super().__init__()
        self.tok_emb_table = nn.Embedding(vocab_size, n_emb)
        self.pos_emb_table = nn.Embedding(block_size, n_emb)
        self.blocks = nn.Sequential(
            *[Block(n_emb, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_emb)
        self.lm_head = nn.Linear(n_emb, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.tok_emb_table(idx)  # (B, T, C)
        pos_emb = self.pos_emb_table(torch.arange(T))  # T, C
        x = tok_emb + pos_emb  # Broadcasting will update pos_emb to (B, T, C)
        x = self.blocks(x)  # Attention block
        x = self.ln_f(x)  # Final layer norm
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # Negative log likelihood loss
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_tokens, block_size):
        # idx is (B, T) array of current context (input) indices
        for _ in range(max_tokens):
            # Crop idx
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)  # Get prediction
            logits = logits[:, -1, :]  # Last time step (B, C)
            probs = F.softmax(logits, dim=-1)  # Convert to probs
            idx_next = torch.multinomial(probs, num_samples=1)  # Sample (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # Concatenate (B, T + 1)

        return idx
