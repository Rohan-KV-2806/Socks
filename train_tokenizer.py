
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

DATA_PATH = "socks_training.txt"
OUTPUT_PATH = "socks_tokenizer.json"
VOCAB_SIZE = 8000

# Create a new tokenizer from scratch
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

print(f"Training tokenizer on {DATA_PATH}...")

with open(DATA_PATH, "r", encoding="utf-8") as file:
    tokenizer.train_from_iterator(
        file,
        trainer=trainer,
        length=10_000_000,
    )

tokenizer.save(OUTPUT_PATH)

# Verify the tokenizer
test_text = "Do you know who I am? I'm the biggest bitch in the word who the fuck are you bitch!."
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded.ids)

print("\nTokenizer training complete.")
print("Vocabulary size:", tokenizer.get_vocab_size())
print("Test text:", test_text)
print("Token IDs:", encoded.ids)
print("Decoded text:", decoded)
print("Saved to:", OUTPUT_PATH)

assert decoded == test_text, "Tokenizer round-trip test failed!"
print("Round-trip test passed!")
