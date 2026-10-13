
import os
import random
import time
from dataclasses import asdict

import torch
from tokenizers import Tokenizer

from model import SocksConfig, SocksModel


# ============================================================
# 1. Configuration
# ============================================================

DATA_PATH = "socks_training.txt"
TOKENIZER_PATH = "socks_tokenizer.json"
CHECKPOINT_DIR = "checkpoints"

BATCH_SIZE = 32
BLOCK_SIZE = 256
MAX_ITERS = 3000
EVAL_INTERVAL = 250
EVAL_ITERS = 50

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
VALIDATION_FRACTION = 0.05
SEED = 42

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"

print("=" * 60)
print("SOCKS - FROM-SCRATCH LANGUAGE MODEL TRAINING")
print("=" * 60)
print("Device:", device)

if use_amp:
    print("GPU:", torch.cuda.get_device_name(0))
    print("Mixed precision: FP16")

# ============================================================
# 2. Load and split the corpus
# ============================================================

print("\nLoading corpus...")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_text = f.read()

stories = [
    story.strip()
    for story in raw_text.split("\n\n")
    if story.strip()
]

random.shuffle(stories)

split_index = int(len(stories) * (1 - VALIDATION_FRACTION))

train_stories = stories[:split_index]
val_stories = stories[split_index:]

print("Total stories:", len(stories))
print("Training stories:", len(train_stories))
print("Validation stories:", len(val_stories))

# ============================================================
# 3. Encode stories using Socks' tokenizer
# ============================================================

print("\nLoading tokenizer...")

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

config = SocksConfig()
config.block_size = BLOCK_SIZE

config.vocab_size = tokenizer.get_vocab_size()

eos_id = tokenizer.token_to_id("[EOS]")

if eos_id is None:
    raise ValueError("Tokenizer is missing the [EOS] token.")

print("Vocabulary size:", config.vocab_size)
print("End-of-story token ID:", eos_id)


def encode_stories(story_list):
    token_ids = []

    for story in story_list:
        ids = tokenizer.encode(story).ids
        token_ids.extend(ids)
        token_ids.append(eos_id)

    return torch.tensor(token_ids, dtype=torch.long)


train_ids = encode_stories(train_stories)
val_ids = encode_stories(val_stories)

del stories, train_stories, val_stories, raw_text

if len(train_ids) <= BLOCK_SIZE + 1:
    raise ValueError("Training corpus is too small.")

if len(val_ids) <= BLOCK_SIZE + 1:
    raise ValueError("Validation corpus is too small.")

print("\nTraining tokens:", f"{len(train_ids):,}")
print("Validation tokens:", f"{len(val_ids):,}")

# Move encoded data to the GPU when available.
train_ids = train_ids.to(device)
val_ids = val_ids.to(device)

# ============================================================
# 4. Initialize Socks from random weights
# ============================================================

print("\nInitializing Transformer...")

model = SocksModel(config).to(device)

total_params = sum(p.numel() for p in model.parameters())

print("Parameters:", f"{total_params:,}")
print("Context length:", config.block_size)
print("Transformer layers:", config.n_layers)
print("Attention heads:", config.n_heads)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

# ============================================================
# 5. Prepare random next-token training batches
# ============================================================

def get_batch(split):
    data = train_ids if split == "train" else val_ids

    starts = torch.randint(
        0,
        len(data) - BLOCK_SIZE - 1,
        (BATCH_SIZE,),
        device=device,
    )

    offsets = torch.arange(BLOCK_SIZE, device=device)

    indices = starts[:, None] + offsets[None, :]

    x = data[indices]
    y = data[indices + 1]

    return x, y


# ============================================================
# 6. Estimate training and validation loss
# ============================================================

@torch.no_grad()
def estimate_loss():
    model.eval()

    results = {}

    for split in ("train", "val"):
        losses = torch.zeros(EVAL_ITERS)

        for i in range(EVAL_ITERS):
            x, y = get_batch(split)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                _, loss = model(x, y)

            losses[i] = loss.detach().float().cpu()

        results[split] = losses.mean().item()

    model.train()
    return results


# ============================================================
# 7. Checkpoint saving
# ============================================================

def save_checkpoint(path, iteration, best_val_loss):
    torch.save(
        {
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "best_val_loss": best_val_loss,
            "tokenizer_path": TOKENIZER_PATH,
        },
        path,
    )


# ============================================================
# 8. Train
# ============================================================

print("\nStarting training...")
print("Maximum iterations:", MAX_ITERS)
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LEARNING_RATE)
print("Checkpoints:", CHECKPOINT_DIR)

best_val_loss = float("inf")
start_time = time.time()

model.train()

try:
    for iteration in range(1, MAX_ITERS + 1):

        x, y = get_batch("train")

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            _, loss = model(x, y)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        scaler.step(optimizer)
        scaler.update()

        if iteration % EVAL_INTERVAL == 0 or iteration == 1:
            losses = estimate_loss()

            elapsed = time.time() - start_time

            print(
                f"Step {iteration:5d}/{MAX_ITERS} | "
                f"Train loss: {losses['train']:.4f} | "
                f"Val loss: {losses['val']:.4f} | "
                f"Time: {elapsed / 60:.1f} min"
            )

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]

                save_checkpoint(
                    os.path.join(CHECKPOINT_DIR, "best.pt"),
                    iteration,
                    best_val_loss,
                )

                print("  Saved new best checkpoint.")

            save_checkpoint(
                os.path.join(CHECKPOINT_DIR, "latest.pt"),
                iteration,
                best_val_loss,
            )

            if use_amp:
                torch.cuda.empty_cache()

except KeyboardInterrupt:
    print("\nTraining interrupted by user.")

finally:
    save_checkpoint(
        os.path.join(CHECKPOINT_DIR, "latest.pt"),
        iteration if "iteration" in locals() else 0,
        best_val_loss,
    )

print("\nTraining finished.")
print("Best validation loss:", best_val_loss)
print("Elapsed minutes:", round((time.time() - start_time) / 60, 2))
print("Best model:", os.path.join(CHECKPOINT_DIR, "best.pt"))
print("Latest model:", os.path.join(CHECKPOINT_DIR, "latest.pt"))
