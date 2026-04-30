# Open Attention Residuals

An open-source implementation of [Attention Residuals](https://arxiv.org/abs/2603.15031) (Kimi Team, 2025) — replacing standard additive residual connections with learned softmax attention over previous sublayer outputs.

<p align="center">
  <img src="figures/training_loss_0.6b.png" width="700">
</p>

## Key Results

### 0.6B Model (d=1024, L=28, same as Qwen3-0.6B, 20k steps)

| Model | Train Loss | WikiText-2 PPL | LAMBADA Acc | HellaSwag Acc |
|-------|-----------|----------------|-------------|---------------|
| Baseline (Standard Residual) | 3.303 | 60.21 | 0.082 | 0.325 |
| **Attention Residuals** | **3.350** | **55.69** | **0.114** | **0.340** |

For reference, the pretrained Qwen3-0.6B (15T tokens) achieves PPL 20.97, LAMBADA 0.364, HellaSwag 0.410.

## How It Works

Standard transformers use additive residual connections:
```
h_l = h_{l-1} + f_{l-1}(h_{l-1})
```

**Attention Residuals** replace this with learned depth-wise attention over previous representations:

```
h_l = Σ α_{i→l} · s_i
```

where `s_i` are source representations (block-level sums or cumulative states), and `α_{i→l}` are softmax attention weights computed with a per-layer learned query vector.

```python
def block_attn_res(blocks, partial_block, proj, norm, recency_bias):
    """Attend over block representations + current partial block."""
    V = torch.stack(blocks + [partial_block])       # (N+1, B, T, D)
    K = norm(V)                                      # RMSNorm keys
    query = proj.weight.view(-1)                     # learned query (D,)
    logits = einsum("d, n b t d -> n b t", query, K)
    logits[-1] += recency_bias                       # boost current block
    weights = softmax(logits, dim=0)                 # (N+1, B, T)
    return einsum("n b t, n b t d -> b t d", weights, V)
```

Each layer selectively attends over previous block representations — "which block's information should I re-use?"

### Modes

- **Block AttnRes** (default): Groups layers into N blocks and sums sublayer outputs within each block before applying cross-block attention. This reduces memory from O(L×d) to O(N×d).
- **Full AttnRes**: Attends over all cumulative hidden states (one per sublayer), providing the finest-grained routing at the cost of O(L²d) compute.

### Layer Dependency Visualization

<p align="center">
  <b>AttnRes (0.6B trained from scratch)</b><br>
  <img src="figures/attnres_deps.png" width="600">
</p>

The visualization shows each sublayer's attention weights over previous sublayer outputs. The model learns genuine cross-layer routing patterns — selectively attending to specific earlier layers, not just the most recent one.

## Quick Start

### Install
```bash
pip install -r requirements.txt
```

### Train from Scratch
```bash
# Baseline
torchrun --nproc_per_node=8 train.py --mode baseline

# Block AttnRes (recommended)
torchrun --nproc_per_node=8 train.py --mode block --num_blocks 4

# Full AttnRes
torchrun --nproc_per_node=8 train.py --mode full
```

### Evaluate
```bash
python eval.py --model_path output/scratch-block-d512-L12-20k/final --mode block
```

### Interactive Visualization
```bash
python app.py --model_path output/scratch-block-d512-L12-20k/final --mode block
```

## Model Architecture

```
100M: d=512, L=12, heads=8, kv_heads=4, ff=1536
0.6B: d=1024, L=28, heads=16, kv_heads=8, ff=3072 (same as Qwen3-0.6B)
```

AttnRes adds per layer:
- 2× projection vectors (`res_proj`, d-dimensional, zero-initialized)
- 2× RMSNorm layers (`res_norm`)

Total overhead: **0.03% parameters**, **<2% latency**.

## Pretrained Weights

| Model | Mode | Link |
|-------|------|------|
| 100M Baseline | — | [wdlctc/open-attnres-baseline](https://huggingface.co/wdlctc/open-attnres-baseline) |
| 100M Block AttnRes | 4 blocks | [wdlctc/open-attnres-block](https://huggingface.co/wdlctc/open-attnres-block) |
| 0.6B Baseline | — | [wdlctc/open-attnres-0.6b-baseline](https://huggingface.co/wdlctc/open-attnres-0.6b-baseline) |
| 0.6B Block AttnRes | 8 blocks | [wdlctc/open-attnres-0.6b-block](https://huggingface.co/wdlctc/open-attnres-0.6b-block) |

## Findings

1. **Block AttnRes achieves the best training loss.** Block-level sums are distinctive (cos sim ~0.69), giving the softmax clean gradients.

2. **Full AttnRes wins on downstream evals despite higher training loss.** At 0.6B scale, Full AttnRes achieves the best LAMBADA (0.114) and HellaSwag (0.340).

3. **Train from scratch for maximum benefit.** Fine-tuning pretrained models yields small gains (~0.02 loss) because pretrained weights are committed to standard residual flow.

4. **Zero-init queries work best.** Default initialization (all projection weights = 0 → uniform softmax) outperforms all alternatives we tried.

## Acknowledgments

- [Attention Residuals](https://arxiv.org/abs/2603.15031) — Kimi Team (original concept)
- [Qwen3](https://arxiv.org/abs/2505.09388) — Qwen Team (base architecture)
