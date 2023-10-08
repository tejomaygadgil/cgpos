"""
Pre-train transformers model on cleaned Greek data.
"""
# Author: Tejomay Gadgil <tejomaygadgil@gmail.com>

from collections import defaultdict
from datetime import datetime
import logging
import random
from sys import argv

import wandb
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import config as cfg
from model import Transformer
from util import read_pkl, display_bar, write_pkl, get_batch, encode, decode, generate


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
    data = read_pkl(read_loc)
    vocab = ["<UNK>"] + sorted(set(data))
    data = [d if random.random() > unc_rate else "<UNK>" for d in data]
    vocab_size = len(vocab)
    params["vocab_size"] = vocab_size

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


def pre_train():
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
    def estimate_loss():
        out = []
        model.eval()
        for data in [train_data, val_data]:
            losses = torch.zeros(eval_iters, device=device)
            for k in range(eval_iters):
                X, Y = get_batch(data, block_size, batch_size, device)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            out.append(losses.mean())
        model.train()
        return out

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
                logger.info(generate(generate_len, block_size, itos, model, device))

        # Sample batch
        xb, yb = get_batch(train_data, block_size, batch_size, device)
        _, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    wandb.finish()

    # Save weights
    torch.save(model.state_dict(), cfg.pt_wts)


def fine_tune():
    logger = logging.getLogger(__name__)
    logger.info("Fine-tuning!")

    # Set params
    params = read_pkl(cfg.pt_params)
    params["learning_rate"] = 1e-4
    params["max_iters"] = 10000
    params["eval_interval"] = 250
    params["dropout"] = 0.8
    wandb.init(project="ncgpos_ft", config=params)
    for param, value in params.items():
        globals()[param] = value
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # Read encodings
    stoi = read_pkl(cfg.pt_stoi)
    itos = read_pkl(cfg.pt_itos)

    # Read data
    ft_syl = read_pkl(cfg.ft_syl)
    ft_targets = read_pkl(cfg.ft_targets)
    ft_targets_map = read_pkl(cfg.ft_targets_map)
    assert len(ft_syl) == len(ft_targets)

    # Process data
    default_stoi = defaultdict(lambda: 0, stoi)  # Set to "<UNK>" if OOV
    tokens = []
    labels = []
    for i, word in enumerate(ft_syl):
        for token in encode(default_stoi, word):
            tokens.append(token)
            labels.append(ft_targets[i][0])

    tokens = torch.tensor(tokens, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    assert len(tokens) == len(labels)
    print(list(zip(ft_syl, ft_targets))[:5])
    print(list(zip(tokens.tolist(), labels.tolist()))[:10])

    # Train val split
    tokens_chunks = torch.split(tokens, len(tokens) // (n_chunks - 1))
    labels_chunks = torch.split(labels, len(labels) // (n_chunks - 1))
    l = [1] * int(n_chunks * train_size) + [0] * int(n_chunks * (1 - train_size))
    random.shuffle(l)
    train_tokens = torch.cat([tokens_chunks[i] for i, v in enumerate(l) if v])
    train_labels = torch.cat([labels_chunks[i] for i, v in enumerate(l) if v])
    val_tokens = torch.cat([tokens_chunks[i] for i, v in enumerate(l) if not v])
    val_labels = torch.cat([labels_chunks[i] for i, v in enumerate(l) if not v])

    print(f"train_size: {train_size}")
    print(f"n_chunks: {n_chunks}")
    print(f"Train size: {len(train_tokens):,} obs")
    print(f"Val size: {len(val_tokens):,} obs")
    display_bar(l)

    model = Transformer(
        vocab_size=vocab_size,
        block_size=block_size,
        emb_size=emb_size,
        n_layer=n_layer,
        n_head=n_head,
        dropout=dropout,
        device=device,
    )
    model.load_state_dict(torch.load(cfg.pt_wts, map_location=torch.device(device)))
    model.lm_head = nn.Linear(512, len(ft_targets_map[1][0]))  # Modify output head
    m = model.to(device)

    @torch.no_grad()
    def estimate_loss():
        out = []
        model.eval()
        for tokens, labels in [(train_tokens, train_labels), (val_tokens, val_labels)]:
            losses = torch.zeros(eval_iters, device=device)
            accs = torch.zeros(eval_iters, device=device)
            for k in range(eval_iters):
                X, Y = get_batch(tokens, block_size, batch_size, device, y=labels)
                logits, loss = model(X, Y)
                preds = F.softmax(logits, dim=-1).argmax(dim=-1)
                acc = torch.sum(preds == Y.view(-1)) / Y.view(-1).size(0)
                losses[k] = loss.item()
                accs[k] = acc
            out.append([losses.mean(), accs.mean()])
        model.train()
        return out

    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    for step in tqdm(range(max_iters)):
        if (step % eval_interval == 0) or (iter == max_iters - 1):
            [train_loss, train_acc], [val_loss, val_acc] = estimate_loss()
            with logging_redirect_tqdm():
                logger.info(f"step {step}:")
                logger.info(f"train loss {train_loss:.4f}")
                logger.info(f"val loss {val_loss:.4f}")
                logger.info(f"train acc {train_acc:.4f}")
                logger.info(f"val loss {val_acc:.4f}")
                logger.info(generate(generate_len, block_size, itos, model, device))
            wandb.log(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                }
            )
        xb, yb = get_batch(train_tokens, block_size, batch_size, device, y=train_labels)
        _, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    wandb.finish()

    # Save weights
    torch.save(model.state_dict(), cfg.ft_wts)
    write_pkl(params, cfg.ft_params)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    def get_read_loc(arg):
        match arg:
            case "pt_local":
                return cfg.pt_syl
            case "ft_local":
                return cfg.ft_syl
            case "pt_cloud":
                return cfg.pt_syl_cloud
            case "":
                raise ValueError("Specify a read location.")
            case _:
                raise ValueError("Not a valid read location.")

    match argv[1]:
        case "setup":
            setup(get_read_loc(argv[2]))
        case "pre_train":
            setup(get_read_loc(argv[2]))
            pre_train()
        case "fine_tune":
            setup(get_read_loc(argv[2]))
            fine_tune()
