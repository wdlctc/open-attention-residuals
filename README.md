# Open Attention Residuals

An open-source implementation of [Attention Residuals](https://arxiv.org/abs/2603.15031) (Kimi Team, 2025) — replacing standard additive residual connections with learned softmax attention over previous layer representations.

<p align="center">
  <img src="figures/training_loss.png" width="700">
</p>

## Key Results

Training a ~100M parameter model from scratch on FineWeb-Edu (20k steps):

| Model | Train Loss | WikiText-2 PPL | LAMBADA Acc | HellaSwag Acc |
|-------|-----------|----------------|-------------|---------------|
| Baseline (Standard Residual) | 3.523 | 76.76 | 0.076 | 0.315 |
| **Block AttnRes (4 blocks)** | **3.489** | **70.82** | 0.084 | **0.340** |
| Full AttnRes (per-sublayer) | 3.502 | 72.70 | **0.102** | 0.305 |

**Block Attention Residuals reduce perplexity by 7.7%** with only 0.03% additional parameters.

## How It Works

Standard transformers use additive residual connections — each layer adds its output to a running sum:
```
h = h + Attention(Norm(h))
h = h + MLP(Norm(h))
```

**Attention Residuals** replace this with learned depth-wise attention. Layers are grouped into blocks, and before each sublayer, the model attends over all previous block representations:

```python
def block_attn_res(blocks, partial_block, proj, norm):
    V = torch.stack(blocks + [partial_block])     # (N+1, B, T, D)
    K = norm(V)
    query = proj.weight.view(-1)                   # (D,)
    logits = einsum("d, n b t d -> n b t", query, K)
    weights = softmax(logits, dim=0)               # (N+1, B, T)
    return einsum("n b t, n b t d -> b t d", weights, V)
```

This allows layers to selectively retrieve information from any earlier block — not just the cumulative sum.

### Layer Dependency Visualizations

<p align="center">
  <b>Block AttnRes (4 blocks)</b><br>
  <img src="figures/attnres_block_deps.png" width="400">
</p>

<p align="center">
  <b>Full AttnRes (per-sublayer)</b><br>
  <img src="figures/attnres_layer_deps.png" width="600">
</p>

## Two Modes

| Mode | Sources | Memory | Best for |
|------|---------|--------|----------|
| **Block** | N blocks (~4-8) | O(N×d) | Training from scratch, smaller models |
| **Full** | All sublayer outputs | O(2L×d) | Larger models, when fine-grained routing matters |

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

This launches a Gradio app where you can:
- Enter any text and see how each layer's AttnRes routes information
- Switch between the layer dependency heatmap and per-token weight views
- Explore individual layers and sublayers

## Model Architecture

```
Config: d=512, L=12, heads=8, kv_heads=4, ff=1536
Total params: ~119M (baseline) / ~119M + 25K (AttnRes)
```

AttnRes adds per layer:
- 2× projection vectors (`res_proj`, d-dimensional, zero-initialized)
- 2× RMSNorm layers (`res_norm`)

Total AttnRes overhead: **0.03% parameters**, **<2% latency**.

## Pretrained Weights

| Model | Mode | Link |
|-------|------|------|
| 100M Baseline | — | [wdlctc/open-attnres-baseline](https://huggingface.co/wdlctc/open-attnres-baseline) |
| 100M Block AttnRes | 4 blocks | [wdlctc/open-attnres-block](https://huggingface.co/wdlctc/open-attnres-block) |
| 100M Full AttnRes | per-sublayer | [wdlctc/open-attnres-full](https://huggingface.co/wdlctc/open-attnres-full) |

## Lessons Learned

1. **Train from scratch** for maximum benefit. Fine-tuning pretrained models with AttnRes yields small gains (~0.02 loss) because the pretrained weights are committed to standard residual flow.

2. **Block mode > Full mode** at small scale. 4 blocks create distinctive representations that are easier to route. Full mode's 25 near-identical sublayer outputs overwhelm the softmax.

3. **Zero-init queries** (the paper's default) work best. We tried recency bias, sigmoid gates, LoRA co-adaptation — none beat the simple zero-init approach.

4. **AttnRes learns genuine cross-layer patterns** when trained from scratch. The visualization shows layers selectively attending to earlier blocks, not just the most recent layer.

## Citation

```bibtex
@software{luo2025openattnres,
  title={Open Attention Residuals},
  author={Cheng Luo and Zefan Cai},
  url={https://github.com/wdlctc/open-attention-residuals},
  year={2025}
}

@article{kimi2025attention,
  title={Attention Residuals},
  author={Kimi Team},
  journal={arXiv preprint arXiv:2603.15031},
  year={2025}
}
```

## Acknowledgments

- [Attention Residuals](https://arxiv.org/abs/2603.15031) — Kimi Team (original paper)
- [Qwen3](https://arxiv.org/abs/2505.09388) — Qwen Team (base architecture)
- [qibin0506/Cortex](https://github.com/qibin0506/Cortex) — Independent AttnRes implementation
