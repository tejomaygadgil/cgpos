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
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import config as cfg
from model import Transformer
from util import read_pkl, display_bar, write_pkl, get_batch, encode, generate, rate

# Set params
params = {
    "trunc_len": 500,
    "batch_size": 32,  # Model hyperparameters
    "block_size": 256,
    "n_layer": 6,
    "n_head": 8,
    "emb_size": 64 * 8,
    "dropout": 0.3,  # Training hyperparameters
    "max_iters": 5000,
    "eval_interval": 250,
    "base_lr": 1e-1,
    "eval_iters": 200,  # Monitor settings
    "torch_seed": 20,  # Seeds
    "random_seed": 40,
}


def zero_loss_check():
    logger = logging.getLogger(__name__)
    logger.info("Checking zero loss:")

    for param, value in params.items():
        globals()[param] = value

    # Seeds
    torch.manual_seed(torch_seed)
    random.seed(random_seed)

    # Read data

    data = read_pkl(cfg.pt_syl_cloud)
    data = data[:trunc_len]  # Truncate data
    vocab = sorted(set(data))
    vocab_size = len(vocab)
    params["vocab_size"] = vocab_size

    # Build tokenizer
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    data = torch.tensor(encode(stoi, data), dtype=torch.long)

    xb, yb = get_batch(data, block_size, batch_size, device)

    logger.info(f"vocab_size: {vocab_size}")
    logger.info(f"data len : {data.shape}")
    logger.info(f"data: {data}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model, optimizer, and scheduler
    model = Transformer(
        vocab_size=vocab_size,
        block_size=block_size,
        emb_size=emb_size,
        n_layer=n_layer,
        n_head=n_head,
        dropout=dropout,
        device=device,
    )
    optimizer = AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(step, model_size=emb_size, factor=1.0, warmup=3000),
    )

    # Send to device
    model.to(device)

    # Train
    for step in tqdm(range(max_iters)):
        model.train()

        _, loss = model(xb, yb)
        with logging_redirect_tqdm():
            logger.info(f"Step {step} - loss: train {loss:.3f}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    match argv[1]:
        case "zero_loss_check":
            zero_loss_check()
