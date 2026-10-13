
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

STORIES_PATH = "socks_training.txt"
CONVERSATIONS_PATH = "socks_conversations.txt"
OUTPUT_PATH = "socks_tokenizer.json"
VOCAB_SIZE = 8000

# Create a completely new tokenizer.
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
    add_prefix_space=False
)
tokenizer.decoder = decoders.ByteLevel()

special_tokens = [
    "[UNK]",
    "[PAD]",
    "[BOS]",
    "[EOS]",
]

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=special_tokens,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)

def training_texts():
    # General language and story data.
    with open(STORIES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield line

    # Conversation data, including User/Assistant labels.
    with open(CONVERSATIONS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield line

print("Training tokenizer on TinyStories + OASST1...")
tokenizer.train_from_iterator(
    training_texts(),
    trainer=trainer,
)

tokenizer.save(OUTPUT_PATH)

# Verify encoding and decoding.
tests = [
    "Hello! How are you?",
    "User: What is Python?",
    "Assistant: Python is a programming language.",
    "Once upon a time, there was a little girl.",
]

print("\nTokenizer training complete.")
print("Vocabulary size:", tokenizer.get_vocab_size())

for text in tests:
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded.ids)

    print("\nOriginal:", text)
    print("Token count:", len(encoded.ids))
    print("Decoded:", decoded)

    assert decoded == text, "Round-trip test failed!"

print("\nAll round-trip tests passed.")
print("Saved to:", OUTPUT_PATH)
