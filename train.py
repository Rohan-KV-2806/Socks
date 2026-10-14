import argparse
import json
import math
import random
import time
from array import array
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from model import SocksConfig, SocksModel

PRETRAIN_PATH = Path("data/raw/fineweb_edu.txt")
CHAT_PATH = Path("data/raw/smol_smoltalk.jsonl")
TOKENIZER_PATH = Path("artifacts/socks_tokenizer.json")
CHECKPOINT_ROOT = Path("checkpoints")

BATCH_SIZE = 8
BLOCK_SIZE = 512
PRETRAIN_ITERS = 10_000
CHAT_ITERS = 5_000
EVAL_INTERVAL = 250
EVAL_ITERS = 20
LEARNING_RATE = 3e-4
MIN_LR_RATIO = 0.1
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
VAL_FRACTION = 0.02
SEED = 42


def iter_documents(path):
    parts = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                parts.append(line.rstrip())
            elif parts:
                yield "\n".join(parts)
                parts = []
        if parts:
            yield "\n".join(parts)


def load_pretrain_buffers(tokenizer):
    docs = list(iter_documents(PRETRAIN_PATH))
    random.Random(SEED).shuffle(docs)
    split = max(1, int(len(docs) * (1 - VAL_FRACTION)))
    train_docs, val_docs = docs[:split], docs[split:]
    del docs

    newline_ids = tokenizer.encode("\n\n").ids
    eos_id = tokenizer.token_to_id("[EOS]")

    def encode_docs(items):
        buf = array("H")
        for doc in items:
            buf.extend(tokenizer.encode(doc).ids)
            buf.extend(newline_ids)
            buf.append(eos_id)
        return torch.from_numpy(np.frombuffer(buf, dtype=np.uint16).copy())

    print("Encoding pretraining corpus...")
    train_ids = encode_docs(train_docs)
    val_ids = encode_docs(val_docs)
    del train_docs, val_docs
    return train_ids, val_ids, None, None


def load_chat_buffers(tokenizer):
    records = []
    with CHAT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            messages = record.get("messages", [])
            if (
                len(messages) >= 2
                and any(m["role"] == "user" for m in messages)
                and messages[-1]["role"] == "assistant"
            ):
                records.append(messages)

    random.Random(SEED).shuffle(records)
    split = max(1, int(len(records) * (1 - VAL_FRACTION)))
    train_records, val_records = records[:split], records[split:]
    del records

    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    newline_ids = tokenizer.encode("\n").ids

    def encode_records(items):
        ids = array("H")
        mask = bytearray()

        for messages in items:
            ids.append(bos_id)
            mask.append(0)

            for message in messages:
                role = message.get("role", "")
                content = message.get("content", "").strip()
                if role not in {"system", "user", "assistant"} or not content:
                    continue

                prefix_ids = tokenizer.encode(f"{role.capitalize()}: ").ids
                ids.extend(prefix_ids)
                mask.extend([0] * len(prefix_ids))

                content_ids = tokenizer.encode(content).ids
                ids.extend(content_ids)
                mask.extend([1 if role == "assistant" else 0] * len(content_ids))

                if role == "assistant":
                    ids.append(eos_id)
                    mask.append(1)

                ids.extend(newline_ids)
                mask.extend([0] * len(newline_ids))

        id_tensor = torch.from_numpy(np.frombuffer(ids, dtype=np.uint16).copy())
        mask_tensor = torch.from_numpy(np.frombuffer(mask, dtype=np.uint8).copy()).bool()
        return id_tensor, mask_tensor

    print("Encoding conversational corpus...")
    train_ids, train_mask = encode_records(train_records)
    val_ids, val_mask = encode_records(val_records)
    del train_records, val_records
    return train_ids, val_ids, train_mask, val_mask


def make_valid_starts(ids, mask, block_size, assistant_only):
    # Each batch predicts ids[start+1 : start+block_size+1].
    n = len(ids) - block_size - 1
    if n <= 0:
        raise ValueError("Token corpus is too small for the selected context length.")

    starts = torch.arange(n, dtype=torch.long)
    if not assistant_only:
        return starts

    target_mask = mask[1:]
    prefix = torch.cat([
        torch.zeros(1, dtype=torch.long),
        target_mask.to(torch.long).cumsum(0),
    ])
    counts = prefix[starts + block_size] - prefix[starts]
    valid = starts[counts > 0]
    if len(valid) == 0:
        raise ValueError("No assistant response targets found in this dataset.")
    return valid


def get_batch(ids, mask, valid_starts, device, block_size, batch_size):
    chosen = torch.randint(len(valid_starts), (batch_size,))
    starts = valid_starts[chosen]
    offsets = torch.arange(block_size)
    positions = starts[:, None] + offsets[None, :]

    x = ids[positions].long().to(device, non_blocking=True)
    y = ids[positions + 1].long().to(device, non_blocking=True)

    if mask is not None:
        target_mask = mask[positions + 1].to(device, non_blocking=True)
        y[~target_mask] = -100

    return x, y


@torch.no_grad()
def estimate_loss(model, datasets, device, use_amp):
    model.eval()
    results = {}

    for name, data in datasets.items():
        ids, mask, valid_starts = data
        values = []

        for _ in range(EVAL_ITERS):
            x, y = get_batch(
                ids, mask, valid_starts, device, BLOCK_SIZE, BATCH_SIZE
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                _, loss = model(x, y)
            values.append(float(loss.float().item()))

        results[name] = sum(values) / len(values)

    model.train()
    return results


def save_checkpoint(path, model, optimizer, scaler, config, iteration, best_val, stage):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "stage": stage,
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": {
            "vocab_size": config.vocab_size,
            "block_size": config.block_size,
            "n_embd": config.n_embd,
            "n_heads": config.n_heads,
            "n_layers": config.n_layers,
            "dropout": config.dropout,
        },
        "best_val_loss": best_val,
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pretrain", "chat"], required=True)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError("Tokenizer missing. Run `python train_tokenizer.py` first.")
    source_path = PRETRAIN_PATH if args.stage == "pretrain" else CHAT_PATH
    if not source_path.exists():
        raise FileNotFoundError(f"Dataset missing: {source_path}. Run `python download_data.py`.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if use_amp:
        torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    config = SocksConfig(
        vocab_size=tokenizer.get_vocab_size(),
        block_size=BLOCK_SIZE,
    )

    if args.stage == "pretrain":
        train_ids, val_ids, train_mask, val_mask = load_pretrain_buffers(tokenizer)
        assistant_only = False
        max_iters = args.max_iters or PRETRAIN_ITERS
        checkpoint_dir = CHECKPOINT_ROOT / "pretrain"
    else:
        train_ids, val_ids, train_mask, val_mask = load_chat_buffers(tokenizer)
        assistant_only = True
        max_iters = args.max_iters or CHAT_ITERS
        checkpoint_dir = CHECKPOINT_ROOT / "chat"

    train_starts = make_valid_starts(train_ids, train_mask, BLOCK_SIZE, assistant_only)
    val_starts = make_valid_starts(val_ids, val_mask, BLOCK_SIZE, assistant_only)

    datasets = {
        "train": (train_ids, train_mask, train_starts),
        "val": (val_ids, val_mask, val_starts),
    }

    model = SocksModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    latest_path = checkpoint_dir / "latest.pt"
    best_path = checkpoint_dir / "best.pt"
    start_iteration = 0
    best_val = float("inf")

    if args.resume:
        if not latest_path.exists():
            raise FileNotFoundError(f"No checkpoint to resume: {latest_path}")
        checkpoint = torch.load(latest_path, map_location=device, weights_only=True)
        if checkpoint.get("stage") != args.stage:
            raise ValueError("Checkpoint stage does not match requested stage.")
        if checkpoint["config"] != {
            "vocab_size": config.vocab_size,
            "block_size": config.block_size,
            "n_embd": config.n_embd,
            "n_heads": config.n_heads,
            "n_layers": config.n_layers,
            "dropout": config.dropout,
        }:
            raise ValueError("Checkpoint model config does not match current config.")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_iteration = int(checkpoint["iteration"])
        best_val = float(checkpoint["best_val_loss"])
        print(f"Resuming {args.stage} from step {start_iteration}.")
    elif args.stage == "chat":
        pretrain_path = CHECKPOINT_ROOT / "pretrain" / "best.pt"
        if not pretrain_path.exists():
            raise FileNotFoundError(
                "Pretraining checkpoint missing. Run `python train.py --stage pretrain` first."
            )
        checkpoint = torch.load(pretrain_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Initialized conversational training from pretrained Socks weights.")

    print("=" * 60)
    print(f"SOCKS {args.stage.upper()} TRAINING")
    print("=" * 60)
    print("Device:", device)
    if use_amp:
        print("GPU:", torch.cuda.get_device_name(0))
        print("Mixed precision: FP16")
    print("Parameters:", f"{sum(p.numel() for p in model.parameters()):,}")
    print("Train tokens:", f"{len(train_ids):,}")
    print("Validation tokens:", f"{len(val_ids):,}")
    print("Batch size:", BATCH_SIZE, "| Context:", BLOCK_SIZE)
    print("Iterations:", max_iters)

    start_time = time.time()
    model.train()
    iteration = start_iteration

    try:
        for iteration in range(start_iteration + 1, max_iters + 1):
            # Cosine decay from LR to 10% of LR.
            progress = min(1.0, iteration / max_iters)
            lr = LEARNING_RATE * (
                MIN_LR_RATIO
                + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * progress))
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            x, y = get_batch(
                train_ids, train_mask, train_starts,
                device, BLOCK_SIZE, BATCH_SIZE
            )
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                _, loss = model(x, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            if iteration % EVAL_INTERVAL == 0 or iteration == start_iteration + 1:
                losses = estimate_loss(model, datasets, device, use_amp)
                elapsed = (time.time() - start_time) / 60
                print(
                    f"Step {iteration:6d}/{max_iters} | "
                    f"Train loss: {losses['train']:.4f} | "
                    f"Val loss: {losses['val']:.4f} | "
                    f"LR: {lr:.2e} | Time: {elapsed:.1f} min"
                )

                if losses["val"] < best_val:
                    best_val = losses["val"]
                    save_checkpoint(
                        best_path, model, optimizer, scaler, config,
                        iteration, best_val, args.stage
                    )
                    print("  Saved new best checkpoint.")

                save_checkpoint(
                    latest_path, model, optimizer, scaler, config,
                    iteration, best_val, args.stage
                )

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving latest checkpoint...")
        save_checkpoint(
            latest_path, model, optimizer, scaler, config,
            iteration, best_val, args.stage
        )

    print("\nTraining run ended.")
    print("Best validation loss:", best_val)
    print("Checkpoint directory:", checkpoint_dir)


if __name__ == "__main__":
    main()
