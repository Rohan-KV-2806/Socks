import json
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

PRETRAIN_PATH = Path("data/raw/fineweb_edu.txt")
CHAT_PATH = Path("data/raw/smol_smoltalk.jsonl")
OUTPUT_PATH = Path("artifacts/socks_tokenizer.json")
VOCAB_SIZE = 8000


def iter_documents(path):
    """Yield blank-line-separated documents without loading the full file."""
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


def training_texts():
    for text in iter_documents(PRETRAIN_PATH):
        yield text

    with CHAT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            for message in record["messages"]:
                yield f"{message['role'].capitalize()}: {message['content']}"


def main():
    if not PRETRAIN_PATH.exists() or not CHAT_PATH.exists():
        raise FileNotFoundError(
            "Dataset files are missing. Run `python download_data.py` first."
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    print("Training byte-level BPE tokenizer on FineWeb-Edu + Smol-SmolTalk...")
    tokenizer.train_from_iterator(training_texts(), trainer=trainer)
    tokenizer.save(str(OUTPUT_PATH))

    tests = [
        "Hello! How are you?",
        "User: What is Python?",
        "Assistant: Python is a programming language.",
        "Once upon a time, there was a little girl.",
    ]

    print("\nVocabulary size:", tokenizer.get_vocab_size())
    for text in tests:
        decoded = tokenizer.decode(tokenizer.encode(text).ids)
        assert decoded == text, f"Round-trip failed: {text!r} -> {decoded!r}"
    for token in special_tokens:
        assert tokenizer.token_to_id(token) is not None, f"Missing token: {token}"

    print("Round-trip and special-token checks passed.")
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
