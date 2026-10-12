Step 1 — Create the Socks project's Environment 
```
python -m venv .venv
```

Activate the environment on Windows:
```
.venv\Scripts\activate
```

Install PyTorch with CUDA support:
```
pip install -r requirements. txt
```

Save that as check_gpu.py and run:
```
python check_gpu.py
```

If the pip installed the CPU version unistal it and install the CUDA version if you have a GPU
```
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```