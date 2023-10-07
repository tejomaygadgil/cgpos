"""
Pre-train transformers model on cleaned Greek data.
"""
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
from greek_accentuation.syllabify import syllabify
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import config

# from model import Transformer
from util import read_pkl

# Author: Tejomay Gadgil <tejomaygadgil@gmail.com>
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=log_fmt)

# Hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = max_iters // 20
learning_rate = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
generate_len = 32
n_head = 8
n_emb = 64 * n_head
n_layer = 6
dropout = 0.3

# Read, tokenize, and flatten text into one big list
tokens = read_pkl(config.ft_syl)
vocab = sorted(set(tokens))
vocab_size = len(vocab)
logging.info(f"Vocab size: {vocab_size}")
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for ch, i in stoi.items()}


def encode(text):
    return [stoi[c] for c in text]


def decode(tokens):
    return "".join([itos[i] for i in tokens])


# Train and test
data = torch.tensor(encode(tokens), dtype=torch.long)
n = int(len(data) * 0.98)
train_data = data[:n]
val_data = data[n:]


# Data loading
def get_batch(split):
    # Generate a batch of data of inputs X and targets Y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)

    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """
    Self-attention decoder (unidirectional).
    """

    def __init__(self, head_size):
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

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_emb, n_emb)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """
    Simple MLP.
    """

    def __init__(self, n_emb):
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

    def __init__(self, n_emb, n_head):
        super().__init__()
        head_size = n_emb // n_head
        self.ln1 = nn.LayerNorm(n_emb)  # Prenorm formulation
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ln2 = nn.LayerNorm(n_emb)  # Prenorm formulation
        self.ffwd = FeedForward(n_emb)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # Skip connections
        x = x + self.ffwd(self.ln2(x))
        return x


class Transformer(nn.Module):
    """
    Predicts next character using only input character embeddings.
    """

    def __init__(self, vocab_size, block_size, n_layer, n_head, n_emb):
        super().__init__()
        self.tok_emb_table = nn.Embedding(vocab_size, n_emb)
        self.pos_emb_table = nn.Embedding(block_size, n_emb)
        self.blocks = nn.Sequential(*[Block(n_emb, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_emb)
        self.lm_head = nn.Linear(n_emb, vocab_size)

    def forward(self, idx, targets):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.tok_emb_table(idx)  # (B, T, C)
        pos_emb = self.pos_emb_table(torch.arange(T, device=device))  # T, C
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

    def generate(self, idx, max_tokens):
        # idx is (B, T) array of current context (input) indices
        for _ in range(max_tokens):
            # Crop idx
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond, None)  # Get prediction
            logits = logits[:, -1, :]  # Last time step (B, C)
            probs = F.softmax(logits, dim=-1)  # Convert to probs
            idx_next = torch.multinomial(probs, num_samples=1)  # Sample (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # Concatenate (B, T + 1)

        return idx


def generate(length=100):
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    return decode(model.generate(context, length)[0].tolist())


# Train
model = Transformer(vocab_size, block_size, n_layer, n_head, n_emb)
m = model.to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
for step in tqdm(range(max_iters + 1)):
    # Evaluate training and val loss every eval_interval
    if step % eval_interval == 0:
        losses = estimate_loss()
        with logging_redirect_tqdm():
            logging.info(
                f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            logging.info(generate(generate_len))

    # Sample batch
    xb, yb = get_batch("train")
    _, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
