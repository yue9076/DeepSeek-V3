# CLAUDE.md — DeepSeek-V3 Codebase Guide

This file provides context for AI assistants working in this repository.

## Overview

DeepSeek-V3 is a 671B-parameter Mixture-of-Experts (MoE) language model that activates only ~37B parameters per forward pass. This repository contains:

- **Custom inference code** with FP8 quantization and distributed tensor parallelism
- **Triton GPU kernels** for quantized matrix operations
- **Utility scripts** for converting and casting model checkpoints
- **A Flask chat application** wrapping the OpenAI-compatible API

The core architectural innovations are:
- **MLA (Multi-Head Latent Attention)**: Compresses KV cache via low-rank projection
- **MoE (Mixture-of-Experts)**: Routes tokens to 8 of 256 experts per token
- **FP8 mixed precision**: Block-wise 128×128 quantization for memory efficiency

---

## Repository Structure

```
DeepSeek-V3/
├── inference/                  # Core inference code
│   ├── model.py                # Transformer + MLA + MoE architecture
│   ├── generate.py             # Token generation loop and CLI entry point
│   ├── kernel.py               # Triton FP8 quantization kernels
│   ├── convert.py              # HuggingFace → model-parallel checkpoint converter
│   ├── fp8_cast_bf16.py        # FP8 → BF16 weight conversion utility
│   ├── requirements.txt        # Inference dependencies
│   └── configs/
│       ├── config_16B.json     # 16B param config (6B activated)
│       ├── config_236B.json    # 236B param config (21B activated)
│       ├── config_671B.json    # 671B param config (37B activated) — production default
│       └── config_v3.1.json    # Extended v3.1 variant
├── app.py                      # Flask REST API chatbot (OpenAI-compatible)
├── requirements.txt            # Root dependencies (Streamlit, requests, python-dotenv)
├── README.md                   # User-facing documentation and benchmark results
├── README_WEIGHTS.md           # Weight file structure and naming conventions
├── figures/                    # Benchmark images
├── LICENSE-CODE                # MIT License (code)
└── LICENSE-MODEL               # DeepSeek Model License (weights)
```

---

## Key Files and Their Roles

### `inference/model.py`
The full model architecture. Key classes (in order of composition):

| Class | Role |
|---|---|
| `ModelArgs` | Dataclass holding all 27 model hyperparameters; loaded from JSON configs |
| `ParallelEmbedding` | Token embedding layer sharded across tensor-parallel ranks |
| `Linear` | Unified linear layer supporting BF16 and FP8 (calls `fp8_gemm` when enabled) |
| `ColumnParallelLinear` | Splits output features across ranks; used for Q, K, V, gate projections |
| `RowParallelLinear` | Splits input features across ranks; followed by AllReduce |
| `RMSNorm` | Root Mean Square normalization (no bias) |
| `MLA` | Multi-Head Latent Attention — absorbs KV projection into QK for efficiency |
| `MLP` | Standard SwiGLU FFN used in dense layers |
| `Gate` | Sigmoid/softmax router for MoE with expert grouping and bias correction |
| `Expert` | Single SwiGLU expert (w1/w2/w3) |
| `MoE` | Sparse MoE block: routes to `n_activated_experts` of `n_routed_experts` |
| `Block` | One transformer layer: MLA + (MLP or MoE) with pre-norm |
| `Transformer` | Full model: embedding → N blocks → RMSNorm → output projection |

Global module-level state in `model.py` (set before loading any model):
```python
world_size = 1          # Number of tensor-parallel ranks
rank = 0                # This process's rank
block_size = 128        # FP8 quantization block size
gemm_impl = "bf16"      # "bf16" or "fp8"
attn_impl = "absorb"    # "naive" or "absorb"
```

### `inference/generate.py`
CLI and generation logic:
- `sample(logits, temperature)` — Gumbel-max sampling; greedy when `temperature=0`
- `generate(model, prompt_tokens, max_new_tokens, eos_id, temperature)` — batched autoregressive loop with KV caching via the model's internal buffer
- `main(...)` — sets up distributed env vars (`WORLD_SIZE`, `RANK`, `LOCAL_RANK`), loads config + model + tokenizer, then runs interactive or batch mode

### `inference/kernel.py`
Triton GPU kernels for FP8 operations:
- `act_quant(x, block_size=128, scale_fmt=None)` — quantizes activations to `float8_e4m3fn`, returns `(quantized, scales)`; supports `scale_fmt="ue8m0"` for power-of-two scales
- `weight_dequant(x, s, block_size=128)` — dequantizes FP8 weights back to default dtype using per-block scales
- `fp8_gemm(a, a_s, b, b_s)` — auto-tuned FP8 matrix multiply with block scaling

### `inference/convert.py`
One-shot HuggingFace → internal format converter:
- Remaps layer names via `mapping` dict (e.g., `"q_proj"` → `"wq"`)
- Skips MTP layers (layer 61 in 671B)
- Shards tensors across `mp` ranks (column-parallel on dim 0, row-parallel on dim 1)
- Outputs `model{rank}-mp{world_size}.safetensors` files

### `inference/fp8_cast_bf16.py`
Converts FP8 HuggingFace checkpoints to BF16 in-place:
- Streams files to stay within GPU memory (keeps ≤2 files cached)
- Detects FP8 tensors by `element_size() == 1`, looks up paired `_scale_inv` tensor
- Updates `model.safetensors.index.json` to remove scale references

### `app.py`
Flask REST API with two endpoints:
- `POST /chat` — accepts `{"message": "..."}`, calls `openai.ChatCompletion.create` with `gpt-3.5-turbo`, returns `{"response": "..."}`
- `POST /upload` — accepts multipart file, saves to `uploads/` directory
- Reads `OPENAI_API_KEY` from environment

---

## Development Workflows

### Running Inference (single GPU)

```bash
cd inference
pip install -r requirements.txt

python generate.py \
  --ckpt-path /path/to/model \
  --config configs/config_671B.json \
  --interactive \
  --max-new-tokens 200 \
  --temperature 0.2
```

Interactive commands: `/exit` to quit, `/clear` to reset conversation history.

### Running Inference (multi-GPU, tensor parallelism)

```bash
torchrun --nproc-per-node 8 inference/generate.py \
  --ckpt-path /path/to/model \
  --config configs/config_671B.json \
  --interactive
```

The model shards embedding, attention projections, and expert weights across `WORLD_SIZE` ranks automatically. Only rank 0 handles I/O.

### Converting HuggingFace Checkpoints

```bash
# Convert HF format → internal model-parallel format
python inference/convert.py \
  --hf-ckpt-path /path/to/hf_model \
  --save-path /path/to/converted \
  --n-experts 256 \
  --model-parallel 8
```

`n-experts` must be divisible by `model-parallel`.

### Casting FP8 Weights to BF16

```bash
# Cast FP8 HF checkpoint to BF16
python inference/fp8_cast_bf16.py \
  --input-fp8-hf-path /path/to/fp8_model \
  --output-bf16-hf-path /path/to/bf16_model
```

### Running the Flask App

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
python app.py  # Starts on http://0.0.0.0:5000
```

---

## Code Conventions

### Naming
- **Classes**: PascalCase (`ParallelEmbedding`, `MLA`, `Transformer`)
- **Functions/methods**: snake_case (`precompute_freqs_cis`, `apply_rotary_emb`)
- **Constants/globals**: lowercase snake_case for module-level config (`world_size`, `gemm_impl`)
- **Triton constants**: UPPER_SNAKE_CASE constexpr args (`BLOCK_SIZE`, `BLOCK_SIZE_M`)

### Type Hints and Docstrings
All public functions and classes use:
- Full type hints on all parameters and return values
- Google-style docstrings with `Args:`, `Returns:`, `Raises:` sections

```python
def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (M//block_size, N//block_size).
        block_size (int, optional): The block size. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor.
    """
```

### Inference-Mode Decorators
All generation functions use `@torch.inference_mode()`. Do not use `torch.no_grad()` in this codebase.

### Distributed Patterns
```python
# Conditional all-reduce (only when using multiple GPUs)
if world_size > 1:
    dist.all_reduce(x)

# Broadcast input in multi-GPU interactive mode
objects = [prompt]
dist.broadcast_object_list(objects, src=0)
```

### Import Order
1. Standard library (`os`, `json`, `math`, `dataclasses`, `typing`)
2. PyTorch (`torch`, `torch.nn`, `torch.nn.functional`, `torch.distributed`)
3. Third-party (`transformers`, `safetensors`, `triton`)
4. Local (`from model import ...`, `from kernel import ...`)

---

## Model Configuration Reference

All configs live in `inference/configs/`. Parameters map directly to `ModelArgs` fields.

| Parameter | 671B | 236B | 16B | Description |
|---|---|---|---|---|
| `vocab_size` | 129280 | 102400 | 102400 | Vocabulary size |
| `dim` | 7168 | 5120 | 2048 | Hidden dimension |
| `n_layers` | 61 | 60 | 27 | Total transformer layers |
| `n_dense_layers` | 3 | 1 | 1 | Initial layers using dense MLP (not MoE) |
| `n_heads` | 128 | 128 | 16 | Number of attention heads |
| `n_routed_experts` | 256 | 160 | 64 | Total MoE experts |
| `n_activated_experts` | 8 | 6 | 6 | Experts activated per token |
| `dtype` | `fp8` | `bf16` | `bf16` | Default weight dtype |
| `q_lora_rank` | 1536 | 1536 | 0 | MLA query LoRA rank (0 = no LoRA) |
| `kv_lora_rank` | 512 | 512 | 512 | MLA KV LoRA rank |

The `max_batch_size` and `max_seq_len` fields in `ModelArgs` control KV cache pre-allocation and are set at runtime (defaults: 8 and 16384 respectively).

---

## Architecture Notes

### Multi-Head Latent Attention (MLA)
MLA reduces the KV cache by projecting keys and values into a low-dimensional latent space:
- Input → `wkv_a` → `kv_lora_rank`-dim latent → `wkv_b` → full K/V
- In `absorb` mode (default), `wkv_b` is absorbed into `wq_b` and `wo` to avoid materializing full K/V during decoding
- KV cache stores only the compressed `kv_lora_rank + qk_rope_head_dim` per head

### MoE Routing
The `Gate` module implements:
1. Linear gate projection → scores per expert
2. Group-level gating: limits to `n_limited_groups` expert groups
3. Top-`n_activated_experts` selection with sigmoid/softmax normalization
4. Optional bias correction (`e_score_correction_bias`) for load balancing

### FP8 Quantization
Block-wise quantization with 128-element blocks:
- Activations: quantized dynamically per inference step
- Weights: pre-quantized (loaded as `float8_e4m3fn` with paired `_scale_inv` tensors)
- Scale format: standard float32, or `ue8m0` (power-of-two exponent only)

### Tensor Parallelism
- `ColumnParallelLinear`: each rank holds `out_features // world_size` rows
- `RowParallelLinear`: each rank holds `in_features // world_size` columns, followed by AllReduce
- `ParallelEmbedding`: each rank holds a slice of the vocab
- Experts: distributed round-robin, each rank processes `n_routed_experts // world_size` experts

---

## CI/CD

`.github/workflows/stale.yml` — marks issues stale after 30 days of inactivity, closes after 14 more days. No build or test automation exists in the repository.

---

## Dependencies

**Inference** (`inference/requirements.txt`):
```
torch==2.4.1
triton==3.0.0
transformers==4.46.3
safetensors==0.4.5
```

**App** (`requirements.txt`):
```
streamlit
requests
python-dotenv
```

CUDA is required for inference. Triton kernels compile JIT on first run (expect a delay).

---

## Common Gotchas

1. **KV cache pre-allocation**: `Transformer.__init__` allocates KV cache tensors based on `max_batch_size` and `max_seq_len`. Set these to match your actual workload, not the maximum possible.

2. **Model loading order**: In `generate.py`, the model is initialized on CUDA before weights are loaded. The dummy generation call (`tokenizer.decode(generate(model, ..., 2, -1, 1.))`) warms up Triton compilation before the real checkpoint is loaded.

3. **Rank-0-only printing**: In multi-GPU mode, `print` is monkey-patched to a no-op on non-zero ranks. Do not use `logging` or direct writes to bypass this.

4. **convert.py skips layer 61**: The 671B model has MTP (Multi-Token Prediction) modules at layer index 61 that are excluded from the standard checkpoint conversion.

5. **FP8 requires contiguous tensors**: `act_quant`, `weight_dequant`, and `fp8_gemm` all assert input contiguity. Call `.contiguous()` before passing sliced tensors.

6. **`scale_fmt="ue8m0"`**: An alternative scale format where the quantization scale is rounded to the nearest power of two. Required for certain hardware backends. Passed through from config → `ModelArgs.scale_fmt` → `Linear.forward`.
