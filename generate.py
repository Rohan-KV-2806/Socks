import torch
from tokenizers import Tokenizer
from model import SocksConfig, SocksModel

CHECKPOINT_PATH = "checkpoints/chat/best.pt"
TOKENIZER_PATH = "artifacts/socks_tokenizer.json"
MAX_NEW_TOKENS = 160
TEMPERATURE = 0.8
TOP_K = 40


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    special_ids = {
        tokenizer.token_to_id(token)
        for token in ("[EOS]", "[BOS]", "[PAD]", "[UNK]")
    }
    special_ids.discard(None)

    history = []
    temperature = TEMPERATURE
    max_new_tokens = MAX_NEW_TOKENS

    print(f"Socks loaded on {device}.")
    print("Checkpoint step:", checkpoint["iteration"])
    print("Validation loss:", checkpoint["best_val_loss"])
    print("Commands: /reset, /temp 0.8, /tokens 100, exit\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_text.lower() in {"exit", "quit"}:
            print("Socks: Goodbye!")
            break
        if not user_text:
            continue
        if user_text == "/reset":
            history.clear()
            print("Conversation reset.\n")
            continue
        if user_text.startswith("/temp "):
            try:
                value = float(user_text.split(maxsplit=1)[1])
                if value <= 0:
                    raise ValueError
                temperature = value
                print(f"Temperature: {temperature}\n")
            except ValueError:
                print("Use /temp 0.8 with a positive number.\n")
            continue
        if user_text.startswith("/tokens "):
            try:
                value = int(user_text.split(maxsplit=1)[1])
                if not 1 <= value <= config.block_size - 2:
                    raise ValueError
                max_new_tokens = value
                print(f"Maximum new tokens: {value}\n")
            except ValueError:
                print(f"Choose between 1 and {config.block_size - 2}.\n")
            continue

        candidate = history + [("user", user_text)]

        # Reserve context space for the expected reply.
        while True:
            prompt = "".join(
                f"{role.capitalize()}: {content}\n"
                for role, content in candidate
            ) + "Assistant: "
            prompt_ids = tokenizer.encode(prompt).ids
            available_prompt = config.block_size - max_new_tokens - 1
            if len(prompt_ids) <= available_prompt:
                break
            if len(candidate) > 1:
                candidate.pop(0)
            else:
                prompt_ids = prompt_ids[-available_prompt:]
                break

        input_ids = torch.tensor(
            [[bos_id] + prompt_ids], dtype=torch.long, device=device
        )
        new_tokens = min(max_new_tokens, config.block_size - input_ids.shape[1])
        if new_tokens <= 0:
            print("Socks: Please use a shorter prompt.\n")
            continue

        with torch.inference_mode():
            output = model.generate(
                input_ids,
                max_new_tokens=new_tokens,
                temperature=temperature,
                top_k=TOP_K,
            )

        generated = output[0, input_ids.shape[1]:].tolist()
        for i, token_id in enumerate(generated):
            if token_id in special_ids:
                generated = generated[:i]
                break

        response = tokenizer.decode(generated).strip()
        if not response:
            print("Socks: [No response generated]\n")
            continue

        history = candidate + [("assistant", response)]
        print(f"Socks: {response}\n")


if __name__ == "__main__":
    main()
