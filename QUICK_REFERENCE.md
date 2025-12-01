# LAPDOG-GRPO Quick Reference

## One-Command Training

### Basic GRPO Training (Recommended)
```bash
python -m torch.distributed.launch --nproc_per_node=2 train.py \
  --shuffle --per_gpu_batch_size=2 --accumulation_steps=2 \
  --total_steps=12000 --eval_freq=1000 --save_freq=1000 \
  --name=my_grpo_run --checkpoint_dir=ckpt/my_grpo_run/ \
  --gold_score_mode=f1rougebleudist --reader_rl_learning \
  --use_grpo --grpo_group_size=4 --grpo_train_retriever \
  --grpo_retriever_group_size=4 --grpo_kl_coeff=0.01 \
  --reader_model_type=t5-base --lr=1e-5 --warmup_steps=500 \
  --train_data="data/convai2/train.jsonl" \
  --eval_data="data/convai2/valid.jsonl" \
  --passages="data/corpora/story/story.jsonl" \
  --model_path="ckpt/stage1/checkpoint/step-12000" \
  --generation_length_penalty=1.0 --use_wandb
```

## Key Parameters

### GRPO Configuration
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `--use_grpo` | flag | Enable GRPO algorithm |
| `--grpo_group_size` | 4 | Responses per group |
| `--grpo_kl_coeff` | 0.01 | KL penalty weight |
| `--grpo_train_retriever` | flag | Enable joint training |
| `--grpo_retriever_group_size` | 4 | Docs per retriever group |

### Learning
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `--lr` | 1e-5 | Learning rate (GRPO) |
| `--warmup_steps` | 500 | LR warmup steps |
| `--total_steps` | 12000 | Training steps |

### Memory Optimization
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `--per_gpu_batch_size` | 2 | Batch per GPU |
| `--accumulation_steps` | 2 | Gradient accumulation |
| `--per_gpu_embedder_batch_size` | 512 | Index building batch |

### Generation Control
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `--generation_length_penalty` | 1.0 | Control verbosity |
| `--gold_score_mode` | f1rougebleudist | Reward function |

## Common Commands

### Check GPU Usage
```bash
nvidia-smi
watch -n 1 nvidia-smi
```

### Monitor Training
```bash
# Terminal logs
tail -f training_*.log

# TensorBoard
tensorboard --logdir=ckpt/

# WandB
# Visit: https://wandb.ai/your-project
```

### Evaluate Model
```bash
python evaluate.py \
  --model_path=ckpt/my_grpo_run/checkpoint/step-12000 \
  --eval_data="data/convai2/valid.jsonl" \
  --passages="data/corpora/story/story.jsonl" \
  --write_results
```

### Compute Metrics
```bash
python compute_metrics.py \
  --predictions=ckpt/my_grpo_run/valid-step-12000.jsonl \
  --references=data/convai2/valid.jsonl
```

## Typical Metrics

### Good Performance
- F1: 24-26
- BLEU: 3.0-4.0
- ROUGE-L: 14-16
- Length Ratio: 0.9-1.1

### Training Behavior
- Loss: Decreases from ~8 to ~2
- Rewards: Increase steadily
- Steps/sec: ~0.12 (2x A100 80GB)
- VRAM: ~32GB per GPU

## Troubleshooting Quick Fixes

### OOM Error
```bash
--per_gpu_batch_size=1 --accumulation_steps=4
```

### Slow Training
```bash
--per_gpu_batch_size=2 --accumulation_steps=2  # Better than 4x1!
```

### Index Building Error
```bash
--per_gpu_embedder_batch_size=512  # Or lower
```

### NaN Loss
```bash
--lr=5e-6 --clip=1.0 --precision=fp32
```

## File Locations

### Checkpoints
```
ckpt/
└── my_grpo_run/
    ├── checkpoint/
    │   ├── step-1000/
    │   ├── step-2000/
    │   └── ...
    ├── valid-step-1000.jsonl
    └── events.out.tfevents.*
```

### Logs
```
training_my_grpo_run.log  # Terminal output
wandb/                     # WandB logs
```

## Architecture Overview

```
Input Query
    ↓
[Retriever] → Top-k Documents
    ↓
[Generator] + Documents → Response
    ↓
[Rewards: F1, ROUGE, BLEU]
    ↓
[GRPO Loss] → Update Both Models
```

## Workflow Summary

1. **Stage 1**: Supervised pre-training (12k steps, ~6 hours)
2. **Stage 2**: Joint GRPO fine-tuning (12k steps, ~28 hours)
3. **Evaluation**: Generate predictions + compute metrics
4. **Analysis**: Compare with baselines

## Expected Timeline

| Stage | Time | Hardware |
|-------|------|----------|
| Setup | 30min | Any |
| Stage 1 | 6h | 2x A100 |
| Stage 2 | 28h | 2x A100 |
| Eval | 1h | 2x A100 |
| **Total** | **~36h** | |

## Documentation Links

- `README.md` - Overview and features
- `SETUP_GUIDE.md` - Detailed setup
- `GRPO_IMPLEMENTATION_SUMMARY.md` - Technical details
- `GRPO_QUICKSTART.md` - Quick start
- `JOINT_GRPO_IMPLEMENTATION.md` - Joint training

## Support

- GitHub Issues: https://github.com/yourusername/LAPDOG-GRPO/issues
- Original LAPDOG: https://github.com/hqsiswiliam/LAPDOG
- Paper: https://aclanthology.org/2023.emnlp-main.154/
