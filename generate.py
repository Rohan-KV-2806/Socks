
import torch
from tokenizers import Tokenizer

from model import SocksConfig, SocksModel


CHECKPOINT_PATH = "checkpoints_chat/best.pt"
TOKENIZER_PATH = "socks_tokenizer.json"

MAX_NEW_TOKENS = 150
TEMPERATURE = 0.8
TOP_K = 40


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Loading Socks on {device}...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    config = SocksConfig(**checkpoint["config"])
    model = SocksModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")

    if bos_id is None or eos_id is None:
        raise ValueError("Tokenizer is missing BOS or EOS tokens.")

    print("Socks loaded successfully!")
    print("Training step:", checkpoint["iteration"])
    print("Best validation loss:", checkpoint["best_val_loss"])
    print("\nType 'exit' to quit.")
    print("Commands: /reset, /temp VALUE, /tokens VALUE\n")

    history = []
    temperature = TEMPERATURE
    max_new_tokens = MAX_NEW_TOKENS

    while True:
        try:
            user_text = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if user_text.lower() in {"exit", "quit"}:
            print("Socks: Goodbye!")
            break

        if not user_text:
            continue

        if user_text == "/reset":
            history.clear()
            print("Conversation history cleared.\n")
            continue

        if user_text.startswith("/temp "):
            try:
                value = float(user_text.split(maxsplit=1)[1])
                if value <= 0:
                    raise ValueError

                temperature = value
                print(f"Temperature set to {temperature}.\n")
            except ValueError:
                print("Usage: /temp 0.8 (must be positive)\n")
            continue

        if user_text.startswith("/tokens "):
            try:
                value = int(user_text.split(maxsplit=1)[1])
                if not 1 <= value <= 512:
                    raise ValueError

                max_new_tokens = value
                print(f"Maximum new tokens set to {value}.\n")
            except ValueError:
                print("Usage: /tokens 150 (range: 1-512)\n")
            continue

        # Keep the current turn and previous turns in context.
        candidate_history = history + [
            ("User", user_text)
        ]

        while True:
            prompt = "".join(
                f"{role}: {text}\n"
                for role, text in candidate_history
            ) + "Assistant: "

            encoded = tokenizer.encode(prompt)

            # Reserve room for generation and stay within context.
            max_context = config.block_size - max_new_tokens
            if max_context < 1:
                max_context = 1

            if len(encoded.ids) <= max_context:
                break

            # Remove the oldest complete turn where possible.
            if len(candidate_history) > 1:
                candidate_history.pop(0)
            else:
                # Long single-message prompts must be truncated.
                encoded.ids = encoded.ids[-max_context:]
                break

        input_ids = torch.tensor(
            [[bos_id] + encoded.ids],
            dtype=torch.long,
            device=device,
        )

        generation_limit = min(
            max_new_tokens,
            config.block_size - input_ids.shape[1],
        )

        if generation_limit <= 0:
            print("Socks: The prompt is too long. Try a shorter message.\n")
            continue

        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=generation_limit,
                    temperature=temperature,
                    top_k=TOP_K,
                )

            generated_ids = output_ids[0, input_ids.shape[1]:].tolist()

            # Stop at the first EOS or unexpected special token.
            special_ids = {
                token_id
                for token_id in (
                    tokenizer.token_to_id("[EOS]"),
                    tokenizer.token_to_id("[BOS]"),
                    tokenizer.token_to_id("[PAD]"),
                    tokenizer.token_to_id("[UNK]"),
                )
                if token_id is not None
            }

            for i, token_id in enumerate(generated_ids):
                if token_id in special_ids:
                    generated_ids = generated_ids[:i]
                    break

            response = tokenizer.decode(generated_ids).strip()

            # Avoid treating an empty response as a successful answer.
            if not response:
                print("Socks: [No response generated]\n")
                continue

            # Add the actual generated turn to conversation history.
            history = candidate_history + [
                ("Assistant", response)
            ]

            print(f"Socks: {response}\n")

        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                print("Generation ran out of GPU memory.")
                print("Try a shorter prompt or fewer output tokens.\n")
            else:
                raise


if __name__ == "__main__":
    main()
