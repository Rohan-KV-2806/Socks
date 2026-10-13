
import torch
from tokenizers import Tokenizer

from model import SocksConfig, SocksModel


# Configuration
CHECKPOINT_PATH = "checkpoints/best.pt"
TOKENIZER_PATH = "socks_tokenizer.json"

MAX_NEW_TOKENS = 150
TEMPERATURE = 0.8
TOP_K = 40


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Loading Socks on {device}...")

    # Load the trained checkpoint
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    # Reconstruct the exact model configuration used in training
    config = SocksConfig(**checkpoint["config"])

    model = SocksModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    eos_id = tokenizer.token_to_id("[EOS]")

    print("Socks loaded successfully!")
    print("Training step:", checkpoint["iteration"])
    print("Best validation loss:", checkpoint["best_val_loss"])
    print("\nType 'exit' to quit.\n")

    while True:
        prompt = input("You: ").strip()

        if prompt.lower() in {"exit", "quit"}:
            print("Socks: Goodbye!")
            break

        if not prompt:
            continue

        # Convert the prompt into token IDs
        encoded = tokenizer.encode(prompt)

        input_ids = torch.tensor(
            [encoded.ids],
            dtype=torch.long,
            device=device,
        )

        # Generate text
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_k=TOP_K,
            )

        # Decode only the newly generated tokens
        generated_ids = output_ids[0, len(encoded.ids):].tolist()

        # Stop at the first end-of-story token
        if eos_id in generated_ids:
            generated_ids = generated_ids[:generated_ids.index(eos_id)]

        response = tokenizer.decode(generated_ids)

        print(f"Socks: {response.strip()}\n")


if __name__ == "__main__":
    main()
