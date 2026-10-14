import json
from pathlib import Path
from datasets import load_dataset

# Adjust these sample sizes if desired.
PRETRAIN_DOCS = 100_000
CHAT_EXAMPLES = 100_000
SEED = 42

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

PRETRAIN_PATH = RAW_DIR / "fineweb_edu.txt"
CHAT_PATH = RAW_DIR / "smol_smoltalk.jsonl"


def download_fineweb():
    print("Streaming FineWeb-Edu sample...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    ).shuffle(seed=SEED, buffer_size=10_000)

    count = 0
    chars = 0

    with PRETRAIN_PATH.open("w", encoding="utf-8") as out:
        for row in dataset:
            text = row.get("text", "").strip()
            if len(text) < 200:
                continue

            out.write(text.replace("\x00", "") + "\n\n")
            count += 1
            chars += len(text)

            if count % 10_000 == 0:
                print(f"FineWeb-Edu documents: {count:,}/{PRETRAIN_DOCS:,}")
            if count >= PRETRAIN_DOCS:
                break

    if count == 0:
        raise RuntimeError("No FineWeb-Edu documents were downloaded.")

    print(f"Saved {count:,} documents ({chars:,} characters) to {PRETRAIN_PATH}")


def normalize_messages(row):
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None

    cleaned = []
    for message in messages:
        if not isinstance(message, dict):
            continue

        role = message.get("role", "")
        content = message.get("content", "")
        if not isinstance(content, str):
            continue
        content = content.strip()

        if role not in {"system", "user", "assistant"} or not content:
            continue

        cleaned.append({"role": role, "content": content})

    roles = {m["role"] for m in cleaned}
    if not cleaned or "user" not in roles or "assistant" not in roles:
        return None
    if cleaned[-1]["role"] != "assistant":
        return None
    return cleaned


def download_smoltalk():
    print("Streaming Smol-SmolTalk...")
    dataset = load_dataset(
        "HuggingFaceTB/smol-smoltalk",
        split="train",
        streaming=True,
    ).shuffle(seed=SEED, buffer_size=10_000)

    count = 0
    chars = 0

    with CHAT_PATH.open("w", encoding="utf-8") as out:
        for row in dataset:
            messages = normalize_messages(row)
            if messages is None:
                continue

            out.write(json.dumps(
                {"messages": messages},
                ensure_ascii=False,
            ) + "\n")

            count += 1
            chars += sum(len(m["content"]) for m in messages)

            if count % 10_000 == 0:
                print(f"Smol-SmolTalk examples: {count:,}/{CHAT_EXAMPLES:,}")
            if count >= CHAT_EXAMPLES:
                break

    if count == 0:
        raise RuntimeError(
            "No Smol-SmolTalk examples were downloaded. "
            "Check the dataset schema/name and Hugging Face access."
        )

    print(f"Saved {count:,} conversations ({chars:,} characters) to {CHAT_PATH}")


if __name__ == "__main__":
    download_fineweb()
    download_smoltalk()
    print("\nDataset download complete.")
