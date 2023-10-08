"""
Pre-train transformers model on cleaned Greek data.
"""
# Author: Tejomay Gadgil <tejomaygadgil@gmail.com>

from datetime import datetime
import logging
import random
from sys import argv

import torch
import wandb
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import config as cfg
from model import Transformer
from util import read_pkl, display_bar, write_pkl, get_batch, encode, decode

log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=log_fmt)

n_head = 8
max_iters = 10000
n_eval = 20

wandb.init(
    project="ncgpos",
    config={
        "train_size": 0.98,  # Train params
        "n_chunks": 500,
        "unc_rate": 0.01,
        "device": "cuda" if torch.cuda.is_available() else "cpu",  # Device params
        "batch_size": 64,  # Model hyperparameters
        "block_size": 256,
        "n_head": n_head,
        "emb_size": 64 * n_head,
        "n_layer": 6,
        "dropout": 0.6,  # Training hyperparameters
        "max_iters": max_iters,
        "eval_interval": max_iters // n_eval,
        "learning_rate": 3e-4,
        "eval_iters": 200,  # Monitor settings
        "generate_len": 32,
        "torch_seed": 20,  # Seeds
        "random_seed": 40,
    },
)

# Seeds
torch.manual_seed(wandb.config.torch_seed)
random.seed(wandb.config.random_seed)
# Train params
train_size = wandb.config.train_size
n_chunks = wandb.config.n_chunks
random_seed = wandb.config.random_seed
unc_rate = wandb.config.unc_rate
# Device params
device = wandb.config.device
# Model hyperparameters
batch_size = wandb.config.batch_size
block_size = wandb.config.block_size
n_head = wandb.config.n_head
emb_size = wandb.config.emb_size
n_layer = wandb.config.n_layer
dropout = wandb.config.dropout
# Training hyperparameters
max_iters = wandb.config.max_iters
eval_interval = wandb.config.eval_interval
learning_rate = wandb.config.learning_rate
# Monitor settings
eval_iters = wandb.config.eval_iters
generate_len = wandb.config.generate_len

# Read data
match argv[1]:
    case "local_pt":
        data = read_pkl(cfg.pt_syl)
    case "local_ft":
        data = read_pkl(cfg.ft_syl)
    case "cloud_pt":
        data = read_pkl("/content/drive/MyDrive/Colab Notebooks/pt_syl.pkl")
    case _:
        raise ValueError("Specify a read location.")
vocab = ["<UNK>"] + sorted(set(data))
data = [d if random.random() > unc_rate else "<UNK>" for d in data]
vocab_size = len(vocab)

# Build tokenizer
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for ch, i in stoi.items()}
tokens = torch.tensor(encode(stoi, data), dtype=torch.long)
write_pkl(stoi, cfg.pt_stoi)
write_pkl(itos, cfg.pt_itos)

# Train and test split
chunks = torch.split(tokens, len(data) // (n_chunks - 1))
l = [1] * int(n_chunks * train_size) + [0] * int(n_chunks * (1 - train_size))
random.shuffle(l)
train_data = torch.cat([chunks[i] for i, v in enumerate(l) if v])
val_data = torch.cat([chunks[i] for i, v in enumerate(l) if not v])

logging.info(f"vocab_size: {vocab_size:,}")
logging.info(f"train_size: {train_size}")
logging.info(f"n_chunks: {n_chunks}")
logging.info(f"Train set: {len(train_data):,} obs")
logging.info(f"Val set: {len(val_data):,} obs")
display_bar(l)


@torch.no_grad()
def estimate_loss(eval_iters, device, *batch_args):
    out = []
    model.eval()
    for data in [train_data, val_data]:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(data, *batch_args)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out.append(losses.mean())
    model.train()
    return out


def generate(length, block_size, model, device):
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    return decode(itos, model.generate(context, length, block_size)[0].tolist())


# Train
model = Transformer(vocab_size, block_size, emb_size, n_layer, n_head, dropout, device)
m = model.to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
for step in tqdm(range(max_iters)):
    # Evaluate training and val loss every eval_interval
    if (step % eval_interval == 0) or (iter == max_iters - 1):
        train_loss, val_loss = estimate_loss(
            eval_iters, device, block_size, batch_size, device
        )
        wandb.log({"train_loss": train_loss, "val_loss": val_loss})
        with logging_redirect_tqdm():
            logging.info(
                f"step {step}: train loss {train_loss:.4f}, val loss {val_loss:.4f}"
            )
            logging.info(generate(generate_len, block_size, model, device))

    # Sample batch
    xb, yb = get_batch(train_data, block_size, batch_size, device)
    _, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

wandb.finish()


# Save model
save_dir = f"/content/drive/MyDrive/Colab Notebooks/wts_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth"
torch.save(model.state_dict(), save_dir)
