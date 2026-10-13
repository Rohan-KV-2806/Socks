
from datasets import load_dataset

print("Downloading OpenAssistant conversations...")

dataset = load_dataset(
    "OpenAssistant/oasst1",
    split="train"
)

with open("socks_conversations.txt", "w", encoding="utf-8") as f:
    for item in dataset:
        if item["role"] in ("user", "assistant"):
            text = item["text"].strip()

            if text:
                role = item["role"].upper()
                f.write(f"{role}: {text}\n\n")

print(f"Saved {len(dataset)} messages to socks_conversations.txt")
