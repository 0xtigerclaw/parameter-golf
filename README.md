# Parameter Golf — Experiments by @0xtigerclaw

Personal experimentation branch for the [OpenAI Parameter Golf Challenge](https://github.com/openai/parameter-golf) 

Training a language model to minimize validation bits-per-byte (val_bpb) within a 16MB package and 10-minute training budget on 8×H100s.

**Current best: `1.4680 val_bpb`**
**Official baseline to beat: `1.2244 val_bpb`**

---

## What this challenge is

Train a language model from scratch on the FineWeb dataset (1024-token vocabulary).  

Lower bits-per-byte = the model is less surprised by the text prediction = it has built a better model of language.

Two hard constraints:
- Submission package must be **< 16MB**
- Training must complete in **< 10 minutes** on 8×H100s

This makes it a constrained optimization across three dimensions simultaneously: model intelligence, parameter efficiency, and training speed.

---

## Architecture

The base architecture is a **U-Net style encoder-decoder transformer** with skip connections — not a standard transformer. The first half of layers encode the input and store activations; the second half decode while receiving those stored activations in reverse order (like a residual highway from early to late layers).

**Best config so far:**
```
VOCAB_SIZE   = 1024
NUM_LAYERS   = 9  (4 encoder + 5 decoder)
MODEL_DIM    = 256
NUM_HEADS    = 8
NUM_KV_HEADS = 4   (Grouped Query Attention)
MLP_MULT     = 3   (FFN expands to 3×MODEL_DIM)
MATRIX_LR    = 0.03
WARMDOWN_ITERS = 400
```

**Parameter count:** ~483K parameters, ~1.9MB at float32, ~6.4MB after INT8+zlib

**Key architectural features:**
- **Grouped Query Attention (GQA):** 8 query heads share 4 KV heads. Diversity lives in the queries — keys and values can be shared without much loss.
- **Muon optimizer:** Active by default for all transformer matrix parameters. Orthogonalizes gradient updates for faster convergence than Adam — critical when you only get ~800 training steps.
- **RMSNorm + residual mixing:** Each block has learned scale parameters and a residual mix vector that blends current and initial token representations.
- **Tied embeddings:** Input and output embedding matrices are shared, saving VOCAB_SIZE × MODEL_DIM parameters.
- **INT8 + zlib quantization:** Automatic at export. Weights quantized to INT8 and compressed with zlib for the submission package.

---

## Experiment log

All experiments run on a single NVIDIA A40 (48GB VRAM) via RunPod. The competition target hardware is 8×H100 — step times will be significantly faster there.

| Exp | NUM_LAYERS | MODEL_DIM | MLP_MULT | MATRIX_LR | WARMDOWN | Steps | val_bpb | Size | Notes |
|-----|-----------|-----------|----------|-----------|----------|-------|---------|------|-------|
| E1  | 9         | 256       | 2        | 0.04      | 1200     | 859   | 1.4912  | 5.3MB | Baseline small model |
| E2  | 9         | 512       | 2        | 0.04      | 1200     | 410   | 1.5297  | 9.4MB | Wider lost — 2× slower steps |
| E3  | 18        | 256       | 2        | 0.04      | 1200     | ~200  | 1.8125* | —    | Deeper lost — no tying |
| E4  | 9         | 256       | 3        | 0.04      | 400      | 802   | 1.4702  | 6.4MB | MLP_MULT=3 + warmdown fix |
| E5  | 9         | 320       | 3        | 0.04      | 350      | 539   | 1.4885  | 7.5MB | Wider lost again |
| E6  | 9         | 256       | 3        | 0.04      | 400      | 773   | 1.4741  | 6.4MB | 63 shards vs 1 — no difference |
| E7  | 18        | 256       | 3        | 0.04      | 400      | —     | —       | —    | OOM with layer tying on Mac |
| E8  | 12        | 256       | 3        | 0.04      | 400      | 635   | 1.6001  | —    | Layer tying lost on A40 |
| E9  | 9         | 256       | 3        | 0.03      | 400      | 808   | **1.4680** | 6.4MB | **Current best** |

*killed early

---

## Key learnings

### On hardware and time budgets
- **Step speed matters more than model capacity** on a time budget. A model that runs 800 steps beats a larger model that runs 400 steps — every time.
- `MODEL_DIM` scales **squared** in attention and FFN (both sides of the matrix). Halving MODEL_DIM saves ~75% of those parameters, not 50%.
- The A40 penalizes deep graphs heavily. Layer tying (18 layers, 3 unique) doubled step time and lost despite the parameter savings. This is an H100-specific optimization.
- MLX (Apple Silicon) traces computational graphs dynamically in Python — completely different behavior from CUDA. Never use Mac results to validate architecture decisions for CUDA hardware.

### On architecture
- **Thin + fast beats wide + slow** at this compute budget. MODEL_DIM=256 at ~750ms/step outperforms MODEL_DIM=512 at ~1466ms/step.
- **MLP_MULT=3 beats default MLP_MULT=2** — slightly wider FFN gives the model more "thinking room" per token.
- **WARMDOWN_ITERS must fit within your actual step budget.** Default is 1200 but you only get ~800 steps — LR never decays properly without this fix.
- The architecture is a **U-Net encoder-decoder**, not a plain transformer. Skip connections let decoder layers access early representations directly.
- **Layer tying** (sharing weights cyclically across layers) saves parameters but not compute. Depth without tying just means slower steps.

### On the competition
- **val_bpb = compression quality.** A model that predicts the next byte confidently needs fewer bits to represent it — lower bpb = smarter model.
- The 16MB limit constrains the serialized INT8+zlib checkpoint, not raw parameter count. Current best uses only 6.4MB — significant headroom remains.
- Muon optimizer is active by default for matrix parameters. All experiments benefit from this — faster convergence per step than Adam.
- Data diversity (63 shards vs 1) makes no difference at ~800 steps — you see such a tiny fraction of even 1 shard that more data doesn't help yet.

---

## What's next

### Immediate (no code changes needed)
- [ ] `MATRIX_LR=0.02` — keep probing the learning rate curve
- [ ] `MLP_MULT=4` — test wider FFN now that we know 3 beats 2

### Code changes with real potential
- [ ] **INT6 Quantization-Aware Training (QAT) with STE** — train with fake INT6 noise so model adapts. INT6 gives ~25% more parameters in same 16MB vs INT8.
- [ ] **Sliding window attention** — restrict attention to last N tokens. Faster attention computation = faster steps = more steps in 10 minutes.
- [ ] **Test-Time Training (TTT)** — adapt model weights on validation data at inference time. Current leaderboard meta, used by all top-10 entries.

### Leaderboard context
The current SOTA is **1.0865 bpb** using LoRA TTT + INT6 QAT. The top entries all combine:
1. Aggressive quantization (INT5/INT6) to maximize parameters in 16MB
2. Test-Time Training to adapt at inference
3. Muon/Turbo-Muon optimizer variants for faster convergence

---

## How to run

```bash
# Clone and setup
git clone https://github.com/0xtigerclaw/parameter-golf.git
cd parameter-golf
bash setup.sh

# Best known config
RUN_ID=my_experiment \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=9 \
MODEL_DIM=256 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=3 \
WARMDOWN_ITERS=400 \
MATRIX_LR=0.03 \
VAL_LOSS_EVERY=200 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

---

## Infrastructure

- **Local dev:** Apple M-series (MLX) — useful for code editing, not for benchmarking
- **Cloud GPU:** RunPod A40 (48GB VRAM, 50GB RAM) — ~$0.20/hr spot
- **Target hardware:** 8×H100 (competition cluster, provided by OpenAI)
- **Pod setup:** `bash setup.sh` — handles git config, data download, log path fix
