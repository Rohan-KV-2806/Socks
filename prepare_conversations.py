
import json
from collections import defaultdict
from datasets import load_dataset

DATASET_NAME = "OpenAssistant/oasst1"

OUTPUT_TEXT = "socks_conversations.txt"
OUTPUT_JSONL = "socks_conversations.jsonl"

print("Loading original OASST1 dataset...")

dataset = load_dataset(DATASET_NAME, split="train")

print("Loaded rows:", len(dataset))
print("Available columns:", dataset.column_names)

# Keep English messages and normalize their roles.
messages = {}

for row in dataset:
    if row.get("lang") != "en":
        continue

    role = row.get("role")

    if role == "prompter":
        role = "user"
    elif role == "user":
        role = "user"
    elif role == "assistant":
        role = "assistant"
    else:
        continue

    text = row.get("text", "").strip()

    if not text:
        continue

    # Keep each message on one physical line.
    text = " ".join(text.split())

    message_id = row["message_id"]

    messages[message_id] = {
        "id": message_id,
        "parent_id": row.get("parent_id"),
        "role": role,
        "text": text,
        "created_date": row.get("created_date", ""),
    }

print("English messages retained:", len(messages))

# Build the conversation tree.
children = defaultdict(list)
roots = []

for message_id, message in messages.items():
    parent_id = message["parent_id"]

    if parent_id is None or parent_id not in messages:
        roots.append(message_id)
    else:
        children[parent_id].append(message_id)

# Sort children for deterministic output.
for parent_id in children:
    children[parent_id].sort(
        key=lambda child_id: (
            messages[child_id]["created_date"] or "",
            child_id,
        )
    )

roots.sort(
    key=lambda message_id: (
        messages[message_id]["created_date"] or "",
        message_id,
    )
)

conversation_count = 0
total_messages = 0
total_characters = 0

print("Reconstructing conversation paths...")

with (
    open(OUTPUT_TEXT, "w", encoding="utf-8") as text_file,
    open(OUTPUT_JSONL, "w", encoding="utf-8") as jsonl_file,
):

    def save_conversation(path):
        global conversation_count, total_messages, total_characters

        # Require both sides of a conversation.
        roles = {message["role"] for message in path}

        # Require complete conversations ending with an assistant reply.
        if len(path) < 2:
            return

        if path[-1]["role"] != "assistant":
            return

        # Require strictly alternating user and assistant messages.
        for i in range(1, len(path)):
            if path[i]["role"] == path[i - 1]["role"]:
                return

        if path[0]["role"] != "user":
            return

        formatted_messages = [
            {
                "role": message["role"],
                "content": message["text"],
            }
            for message in path
        ]

        # Human-readable training corpus.
        lines = []

        for message in formatted_messages:
            label = (
                "User"
                if message["role"] == "user"
                else "Assistant"
            )
            lines.append(f"{label}: {message['content']}")

        conversation_text = "\n".join(lines)

        text_file.write(conversation_text + "\n\n")

        # Structured format for future training pipelines.
        jsonl_file.write(
            json.dumps(
                {"messages": formatted_messages},
                ensure_ascii=False,
            )
            + "\n"
        )

        conversation_count += 1
        total_messages += len(path)
        total_characters += len(conversation_text)

    def walk(message_id, path):
        message = messages[message_id]
        new_path = path + [message]

        child_ids = children.get(message_id, [])

        if not child_ids:
            save_conversation(new_path)
            return

        for child_id in child_ids:
            walk(child_id, new_path)

    for root_id in roots:
        walk(root_id, [])

print("\nPreparation complete.")
print("Conversation paths:", conversation_count)
print("Messages across saved paths:", total_messages)
print("Characters:", f"{total_characters:,}")
print("Text output:", OUTPUT_TEXT)
print("JSONL output:", OUTPUT_JSONL)

# Display a sample for inspection.
with open(OUTPUT_TEXT, "r", encoding="utf-8") as f:
    sample = f.read(2000)

print("\n========== SAMPLE ==========\n")
print(sample or "No conversations were generated.")
print("============================")
