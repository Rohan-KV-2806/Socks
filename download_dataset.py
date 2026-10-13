
from datasets import load_dataset

print("Downloading TinyStories...")

dataset = load_dataset(
    "roneneldan/TinyStories",
    split="train[:10000]"
)

print("Number of stories:", len(dataset))
print("\nFirst example:")
print(dataset[0]["text"])

# Save the downloaded sample as a plain text file.
with open("socks_training.txt", "w", encoding="utf-8") as f:
    for example in dataset:
        text = example["text"].strip()
        if text:
            f.write(text + "\n\n")

print("\nSaved to socks_training.txt")
