"""
Pre-train transformers model on cleaned Greek data.
"""
# Author: Tejomay Gadgil <tejomaygadgil@gmail.com>
import logging
import random

import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import config as cfg
from model import Transformer
from util import read_pkl

# Author: Tejomay Gadgil <tejomaygadgil@gmail.com>
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=log_fmt)

torch.manual_seed(20)
# Train params
train_size = 0.98
n_chunks = 500
random_seed = 40
random.seed(random_seed)
# Device params
device = "cuda" if torch.cuda.is_available() else "cpu"
# Model hyperparameters
batch_size = 64
block_size = 256
n_head = 8
n_emb = 64 * n_head
n_layer = 6
dropout = 0.6
# Training hyperparameters
max_iters = 5000
eval_interval = max_iters // 20
learning_rate = 3e-4
# Monitor settings
eval_iters = 200
generate_len = 32

# Read data
# tokens = read_pkl(cfg.pt_syl)
tokens = read_pkl("/content/drive/MyDrive/Colab Notebooks/pt_syl.pkl")
vocab = sorted(set(tokens))
vocab_size = len(vocab)

# Build tokenizer
tok2int = {ch: i for i, ch in enumerate(vocab)}
int2tok = {i: ch for ch, i in tok2int.items()}
encode = lambda text: [tok2int[c] for c in text]
decode = lambda tokens: "".join([int2tok[i] for i in tokens])

# Train and test split
data = torch.tensor(encode(tokens), dtype=torch.long)
chunks = torch.split(data, len(data) // (n_chunks - 1))
l = [1] * int(n_chunks * train_size) + [0] * int(n_chunks * (1 - train_size))
random.shuffle(l)
train_data = torch.cat([chunks[i] for i, v in enumerate(l) if v])
val_data = torch.cat([chunks[i] for i, v in enumerate(l) if not v])

logging.info(f"vocab_size: {vocab_size:,}")
logging.info(f"train_size: {train_size}")
logging.info(f"n_chunks: {n_chunks}")
logging.info(f"Train set: {len(train_data):,} obs")
logging.info(f"Val set: {len(val_data):,} obs")


# Data loading
def get_batch(split):
    # Generate a small batch of data of inputs x and targets y
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


def generate(length, model):
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    return decode(model.generate(context, length)[0].tolist())


# Train
model = Transformer(vocab_size, block_size, n_layer, n_head, n_emb, dropout)
m = model.to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
for step in tqdm(range(max_iters)):
    # Evaluate training and val loss every eval_interval
    if step % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        with logging_redirect_tqdm():
            logging.info(
                f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            logging.info(generate(generate_len, model))

    # Sample batch
    xb, yb = get_batch("train")
    _, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
