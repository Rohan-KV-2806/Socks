
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


@dataclass
class SocksConfig:
    vocab_size: int = 8000
    block_size: int = 256
    n_embd: int = 256
    n_heads: int = 4
    n_layers: int = 6
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_heads == 0

        self.n_heads = config.n_heads
        self.head_dim = config.n_embd // config.n_heads

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch_size, seq_len, channels = x.shape

        q, k, v = self.qkv(x).split(channels, dim=2)

        q = q.view(
            batch_size, seq_len, self.n_heads, self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch_size, seq_len, self.n_heads, self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size, seq_len, self.n_heads, self.head_dim
        ).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(
            batch_size, seq_len, channels
        )

        return self.resid_dropout(self.proj(y))


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class SocksModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()

        self.config = config or SocksConfig()

        self.token_embedding = nn.Embedding(
            self.config.vocab_size, self.config.n_embd
        )
        self.position_embedding = nn.Embedding(
            self.config.block_size, self.config.n_embd
        )

        self.dropout = nn.Dropout(self.config.dropout)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(self.config)
                for _ in range(self.config.n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(self.config.n_embd)
        self.lm_head = nn.Linear(
            self.config.n_embd, self.config.vocab_size, bias=False
        )

        # Share token embeddings and output projection weights.
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        batch_size, seq_len = input_ids.shape

        if seq_len > self.config.block_size:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"block size {self.config.block_size}"
            )

        positions = torch.arange(
            seq_len, device=input_ids.device
        )

        x = self.token_embedding(input_ids)
        x = x + self.position_embedding(positions)
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.final_norm(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=100,
        temperature=0.8,
        top_k=40,
    ):
        self.eval()

        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.block_size:]

            logits, _ = self(context)
            logits = logits[:, -1, :]

            logits = logits / max(temperature, 1e-5)

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, k)
                logits[logits < values[:, [-1]]] = float("-inf")

            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)

            input_ids = torch.cat((input_ids, next_token), dim=1)

        return input_ids


if __name__ == "__main__":
    config = SocksConfig()
    model = SocksModel(config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print("Socks Transformer initialized successfully.")
    print(f"Vocabulary size: {config.vocab_size:,}")
    print(f"Layers: {config.n_layers}")
    print(f"Attention heads: {config.n_heads}")
    print(f"Embedding dimension: {config.n_embd}")
    print(f"Context length: {config.block_size}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    dummy_input = torch.randint(
        0, config.vocab_size, (2, 64)
    )

    logits, loss = model(dummy_input, dummy_input)

    print("Output shape:", tuple(logits.shape))
    print("Test loss:", round(loss.item(), 4))
    print("Model test passed.")
