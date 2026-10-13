
import os
import json
import random
import time

import numpy as np
import torch
from tokenizers import Tokenizer

from model import SocksConfig, SocksModel


# ============================================================
# 1. Configuration
# ============================================================

DATA_PATH = "socks_conversations.jsonl"
TOKENIZER_PATH = "socks_tokenizer.json"
CHECKPOINT_DIR = "checkpoints_chat"

BLOCK_SIZE = 256
BATCH_SIZE = 32
MAX_ITERS = 5000
EVAL_INTERVAL = 250
EVAL_ITERS = 50

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
VAL_FRACTION = 0.05
SEED = 42

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
use_amp = device.type == "cuda"

print("=" * 60)
print("SOCKS - CONVERSATIONAL TRAINING")
print("=" * 60)
print("Device:", device)

if use_amp:
    print("GPU:", torch.cuda.get_device_name(0))
    print("Mixed precision: FP16")


# ============================================================
# 2. Load conversations and split into training/validation
# ============================================================

print("\nLoading conversations...")

conversations = []

with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        record = json.loads(line)
        messages = record["messages"]

        # Only retain complete, alternating conversations.
        if len(messages) < 2:
            continue

        if messages[0]["role"] != "user":
            continue

        if messages[-1]["role"] != "assistant":
            continue

        if any(
            messages[i]["role"] == messages[i - 1]["role"]
            for i in range(1, len(messages))
        ):
            continue

        if any(
            message["role"] not in ("user", "assistant")
            or not message["content"].strip()
            for message in messages
        ):
            continue

        conversations.append(messages)

random.shuffle(conversations)

split = int(len(conversations) * (1 - VAL_FRACTION))

train_conversations = conversations[:split]
val_conversations = conversations[split:]

print("Total conversations:", len(conversations))
print("Training conversations:", len(train_conversations))
print("Validation conversations:", len(val_conversations))


# ============================================================
# 3. Load tokenizer and identify special tokens
# ============================================================

print("\nLoading tokenizer...")

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

bos_id = tokenizer.token_to_id("[BOS]")
eos_id = tokenizer.token_to_id("[EOS]")

if bos_id is None or eos_id is None:
    raise ValueError("Tokenizer must contain [BOS] and [EOS].")

config = SocksConfig()
config.vocab_size = tokenizer.get_vocab_size()
config.block_size = BLOCK_SIZE

user_prefix = tokenizer.encode("User: ").ids
assistant_prefix = tokenizer.encode("Assistant: ").ids
newline = tokenizer.encode("\n").ids

print("Vocabulary size:", config.vocab_size)
print("Context length:", BLOCK_SIZE)


# ============================================================
# 4. Encode conversations with assistant-only loss
# ============================================================

def encode_conversations(items):
    all_ids = []
    all_supervised = []

    for messages in items:
        ids = [bos_id]
        supervised = [False]

        for message in messages:
            role = message["role"]
            content = message["content"].strip()

            if role == "user":
                prefix = user_prefix
                train_response = False
            else:
                prefix = assistant_prefix
                train_response = True

            # Role prefix supplies context but is not a target.
            ids.extend(prefix)
            supervised.extend([False] * len(prefix))

            # Only assistant response tokens receive loss.
            content_ids = tokenizer.encode(content).ids

            ids.extend(content_ids)
            supervised.extend(
                [train_response] * len(content_ids)
            )

            if role == "assistant":
                # Teach the model when its response ends.
                ids.append(eos_id)
                supervised.append(True)

            # Separator between messages.
            ids.extend(newline)
            supervised.extend([False] * len(newline))

        all_ids.extend(ids)
        all_supervised.extend(supervised)

    return (
        np.asarray(all_ids, dtype=np.int32),
        np.asarray(all_supervised, dtype=np.bool_),
    )


print("\nEncoding training conversations...")

train_array, train_mask = encode_conversations(
    train_conversations
)

print("Encoding validation conversations...")

val_array, val_mask = encode_conversations(
    val_conversations
)

del conversations, train_conversations, val_conversations

print("Training tokens:", f"{len(train_array):,}")
print("Validation tokens:", f"{len(val_array):,}")


# ============================================================
# 5. Prepare efficient batches
# ============================================================

def prepare_data(token_array, supervised_array):
    ids = torch.from_numpy(token_array.astype(np.int64))
    mask = torch.from_numpy(supervised_array)

    if len(ids) <= BLOCK_SIZE:
        raise ValueError("Dataset is too small for BLOCK_SIZE.")

    # Prefix sums allow us to find chunks containing at least
    # one assistant target. This prevents all-masked batches.
    target_mask = mask[1:].to(torch.int64)
    prefix = torch.cat([
        torch.zeros(1, dtype=torch.int64),
        target_mask.cumsum(dim=0),
    ])

    number_of_starts = len(ids) - BLOCK_SIZE

    starts = torch.arange(number_of_starts)

    counts = (
        prefix[starts + BLOCK_SIZE] - prefix[starts]
    )

    valid_starts = starts[counts > 0]

    if len(valid_starts) == 0:
        raise ValueError("No assistant training targets found.")

    return ids.to(device), mask.to(device), valid_starts


train_ids, train_mask, train_starts = prepare_data(
    train_array, train_mask
)

val_ids, val_mask, val_starts = prepare_data(
    val_array, val_mask
)

del train_array, val_array

offsets = torch.arange(BLOCK_SIZE, device=device)


def get_batch(split_name):
    if split_name == "train":
        ids, mask, valid_starts = (
            train_ids, train_mask, train_starts
        )
    else:
        ids, mask, valid_starts = (
            val_ids, val_mask, val_starts
        )

    selected = torch.randint(
        len(valid_starts), (BATCH_SIZE,)
    )

    starts = valid_starts[selected].to(device)

    positions = starts[:, None] + offsets[None, :]

    x = ids[positions]

    target_positions = positions + 1
    y = ids[target_positions].clone()

    # Ignore user messages, role prefixes and separators.
    target_mask = mask[target_positions]
    y[~target_mask] = -100

    return x, y


# ============================================================
# 6. Initialize a fresh model
# ============================================================

print("\nInitializing a new Transformer...")

model = SocksModel(config).to(device)

parameters = sum(p.numel() for p in model.parameters())

print("Parameters:", f"{parameters:,}")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp,
)


# ============================================================
# 7. Evaluation
# ============================================================

@torch.no_grad()
def estimate_loss():
    model.eval()

    results = {}

    for split_name in ("train", "val"):
        losses = []

        for _ in range(EVAL_ITERS):
            x, y = get_batch(split_name)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                _, loss = model(x, y)

            losses.append(loss.float().item())

        results[split_name] = sum(losses) / len(losses)

    model.train()
    return results


# ============================================================
# 8. Save checkpoints
# ============================================================

def save_checkpoint(path, iteration, best_val_loss):
    torch.save(
        {
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                key: getattr(config, key)
                for key in (
                    "vocab_size",
                    "block_size",
                    "n_embd",
                    "n_heads",
                    "n_layers",
                    "dropout",
                )
            },
            "best_val_loss": best_val_loss,
            "tokenizer_path": TOKENIZER_PATH,
        },
        path,
    )


# ============================================================
# 9. Training loop
# ============================================================

print("\nStarting training...")
print("Iterations:", MAX_ITERS)
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LEARNING_RATE)
print("Checkpoint directory:", CHECKPOINT_DIR)

best_val_loss = float("inf")
start_time = time.time()
iteration = 0

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

        if (
            iteration % EVAL_INTERVAL == 0
            or iteration == 1
        ):
            losses = estimate_loss()
            elapsed = (time.time() - start_time) / 60

            print(
                f"Step {iteration:5d}/{MAX_ITERS} | "
                f"Train loss: {losses['train']:.4f} | "
                f"Val loss: {losses['val']:.4f} | "
                f"Time: {elapsed:.2f} min"
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

except KeyboardInterrupt:
    print("\nTraining interrupted by user.")

finally:
    save_checkpoint(
        os.path.join(CHECKPOINT_DIR, "latest.pt"),
        iteration,
        best_val_loss,
    )

print("\nTraining finished.")
print("Best validation loss:", best_val_loss)
print("Elapsed minutes:", round(
    (time.time() - start_time) / 60, 2
))
print("Best checkpoint:", os.path.join(
    CHECKPOINT_DIR, "best.pt"
))
print("Latest checkpoint:", os.path.join(
    CHECKPOINT_DIR, "latest.pt"
))
