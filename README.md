# Socks — from-scratch language model

Socks is a small decoder-only Transformer trained from randomly initialized weights. It is designed as a learning project, not as a pretrained-model wrapper.

## Project layout

```text
Socks/
├── .gitignore
├── README.md
├── requirements.txt
├── check_gpu.py
├── download_data.py
├── train_tokenizer.py
├── model.py
├── train.py
├── generate.py
├── data/
│   └── raw/                 # downloaded datasets (created automatically)
├── artifacts/               # tokenizer (created automatically)
└── checkpoints/
    ├── pretrain/             # general-text checkpoints
    └── chat/                 # conversational checkpoints
```

Dataset files, tokenizer files, and checkpoints are generated locally and are not included in this ZIP. Your existing `.venv` is also not included.

## 1. Activate your existing virtual environment

From your project folder on Windows:

```powershell
.venv\Scripts\activate
python -m pip install -r requirements.txt
python check_gpu.py
```

Use a CUDA-enabled PyTorch build for your RTX 3060. If `check_gpu.py` reports `CUDA available: False`, install the appropriate CUDA wheel from the official PyTorch selector before continuing.

## 2. Download datasets

```powershell
python download_data.py
```

The script streams a configurable sample from FineWeb-Edu and Smol-SmolTalk into `data/raw/`. The defaults are 100,000 general-text documents and 100,000 conversation examples. You can adjust these constants at the top of `download_data.py`.

Streaming avoids downloading the entire dataset, though the Hugging Face client may still cache data shards.

## 3. Train the tokenizer

```powershell
python train_tokenizer.py
```

This trains a new byte-level BPE tokenizer from both corpora and saves it to `artifacts/socks_tokenizer.json`.

**Do not change the tokenizer after model training starts.** Changing token IDs makes existing model checkpoints incompatible.

## 4. Pretrain Socks from random weights

```powershell
python train.py --stage pretrain
```

This trains next-token prediction on FineWeb-Edu text. The default is 10,000 iterations. You can override it, for example:

```powershell
python train.py --stage pretrain --max-iters 10000
```

Checkpoints are written to `checkpoints/pretrain/`.

## 5. Train conversational behavior

After pretraining completes:

```powershell
python train.py --stage chat
```

This initializes from the best pretraining checkpoint and continues training on Smol-SmolTalk. Loss is applied to assistant content and assistant end-of-turn tokens, not user or system content. Checkpoints are written to `checkpoints/chat/`.

To resume an interrupted run of the same stage, pass `--resume`:

```powershell
python train.py --stage chat --resume
```

Use `--resume` only for a stage that already has a `latest.pt` checkpoint. Do not use it to start a new stage.

## 6. Chat with Socks

```powershell
python generate.py
```

This loads `checkpoints/chat/best.pt`. It expects both training stages to have completed.

## Model configuration

The default model is approximately 55 million parameters, with 10 layers, 10 attention heads, 640-dimensional embeddings, an 8,000-token vocabulary, and a 512-token context window. Batch size defaults to 8 and CUDA mixed precision is enabled. Memory use varies by PyTorch/CUDA versions and data; lower `BATCH_SIZE` or `BLOCK_SIZE` in `train.py` if you encounter CUDA out-of-memory errors.

## Important limitations

- This project trains from random weights; no pretrained model is used.
- The initial dataset samples and iteration counts are practical starting settings, not a guarantee of assistant quality.
- FineWeb-Edu is educational web text, not a complete or perfectly clean knowledge corpus.
- Smol-SmolTalk examples can vary in quality and task type.
- Keep the tokenizer fixed after training begins.
