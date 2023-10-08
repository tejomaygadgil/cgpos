"""
Pre-train transformers model on cleaned Greek data.
"""
# Author: Tejomay Gadgil <tejomaygadgil@gmail.com>

from collections import defaultdict
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


def setup(read_loc):
    logger = logging.getLogger(__name__)
    logger.info("Pre-training setup:")

    n_head = 8
    max_iters = 10000
    n_eval = 20

    params = {
        "train_size": 0.98,  # Train params
        "n_chunks": 500,
        "unc_rate": 0.005,
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
    }
    for param, value in params.items():
        globals()[param] = value

    # Seeds
    torch.manual_seed(torch_seed)
    random.seed(random_seed)

    # Read data
    breakpoint()
    data = read_pkl(read_loc)
    vocab = ["<UNK>"] + sorted(set(data))
    data = [d if random.random() > unc_rate else "<UNK>" for d in data]
    vocab_size = len(vocab)

    # Build tokenizer
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    tokens = torch.tensor(encode(stoi, data), dtype=torch.long)

    # Train and test split
    chunks = torch.split(tokens, len(data) // (n_chunks - 1))
    l = [1] * int(n_chunks * train_size) + [0] * int(n_chunks * (1 - train_size))
    random.shuffle(l)
    train_data = torch.cat([chunks[i] for i, v in enumerate(l) if v])
    val_data = torch.cat([chunks[i] for i, v in enumerate(l) if not v])

    # Write
    write_pkl(params, cfg.pt_params)
    write_pkl(stoi, cfg.pt_stoi)
    write_pkl(itos, cfg.pt_itos)
    write_pkl(train_data, cfg.pt_train)
    write_pkl(val_data, cfg.pt_val)

    # Log
    logger.info(f"vocab_size: {vocab_size:,}")
    logger.info(f"train_size: {train_size}")
    logger.info(f"n_chunks: {n_chunks}")
    logger.info(f"Train set: {len(train_data):,} obs")
    logger.info(f"Val set: {len(val_data):,} obs")
    display_bar(l)


def train():
    logger = logging.getLogger(__name__)
    logger.info("Pre-training!")

    # Load data
    itos = read_pkl(cfg.pt_itos)
    train_data = read_pkl(cfg.pt_train)
    val_data = read_pkl(cfg.pt_val)

    # Load params
    params = read_pkl(cfg.pt_params)
    wandb.init(project="ncgpos", config=params)
    for param, value in params.items():
        globals()[param] = value
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Seeds
    torch.manual_seed(torch_seed)
    random.seed(random_seed)

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
    model = Transformer(
        vocab_size, block_size, emb_size, n_layer, n_head, dropout, device
    )
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
                logger.info(
                    f"step {step}: train loss {train_loss:.4f}, val loss {val_loss:.4f}"
                )
                logger.info(generate(generate_len, block_size, model, device))

        # Sample batch
        xb, yb = get_batch(train_data, block_size, batch_size, device)
        _, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    wandb.finish()

    # Save weights
    torch.save(model.state_dict(), cfg.pt_wts)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    match argv[1]:
        case "setup":
            match argv[2]:
                case "pt_local":
                    read_loc = cfg.pt_syl
                case "ft_local":
                    read_loc = cfg.ft_syl
                case "pt_cloud":
                    read_loc = cfg.pt_syl_cloud
                case _:
                    raise ValueError("Specify a read location.")
            setup(read_loc)
        case "train":
            setup()
            train()
