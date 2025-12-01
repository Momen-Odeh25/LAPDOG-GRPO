# LAPDOG with GRPO Training

Extended implementation of LAPDOG (Learning Retrieval Augmentation for Personalized Dialogue Generation) with **Group Relative Policy Optimization (GRPO)** for enhanced reinforcement learning.

## 📋 About

This repository extends the original [LAPDOG](https://github.com/hqsiswiliam/LAPDOG) model with GRPO training capabilities. The original LAPDOG paper: [Learning Retrieval Augmentation for Personalized Dialogue Generation](https://aclanthology.org/2023.emnlp-main.154/).

### Original LAPDOG
LAPDOG addresses personalized dialogue generation by leveraging external knowledge retrieval. The model consists of:
- **Story Retriever**: Retrieves relevant information from story documents using persona profiles
- **Dialogue Generator**: Generates personalized responses using dialogue history and augmented persona profiles
- **Joint Training**: Collaboratively optimizes both components toward dialogue quality metrics

### Our Contributions

We extend LAPDOG with:

1. **GRPO Training Algorithm**: Group Relative Policy Optimization for more stable reinforcement learning
2. **Joint GRPO Training**: Simultaneous GRPO optimization for both retriever and generator
3. **Length Penalty Control**: Dynamic generation length penalty to control response verbosity
4. **Enhanced Monitoring**: WandB integration for comprehensive training visualization

## 🏗️ Architecture

![LAPDOG Architecture](figures/framework.png)

The model combines:
- **T5-based Generator**: Produces personalized dialogue responses
- **Contriever-based Retriever**: Retrieves relevant context from external knowledge
- **GRPO Optimization**: Group-relative advantage estimation for stable policy gradients

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/LAPDOG-GRPO.git
cd LAPDOG-GRPO

# Create environment
conda env create -f env.yml
conda activate lapdog
```

### Data Preparation

The repository expects data in the following structure:
```
data/
├── convai2/
│   ├── train.jsonl
│   └── valid.jsonl
└── corpora/
    └── story/
        └── story.jsonl
```

### Training

#### Stage 1: Supervised Fine-tuning (Generator Only)

```bash
python -m torch.distributed.launch --nproc_per_node=2 train.py \
  --closed_book \
  --shuffle \
  --per_gpu_batch_size=64 \
  --total_steps=12000 \
  --eval_freq=2000 \
  --save_freq=2000 \
  --name=stage1_t5base \
  --checkpoint_dir=ckpt/stage1/ \
  --reader_model_type=t5-base \
  --lr=5e-5 \
  --train_data="data/convai2/train.jsonl" \
  --eval_data="data/convai2/valid.jsonl"
```

#### Stage 2: Joint GRPO Training (Generator + Retriever)

```bash
python -m torch.distributed.launch --nproc_per_node=2 train.py \
  --shuffle \
  --per_gpu_batch_size=2 \
  --accumulation_steps=2 \
  --total_steps=12000 \
  --eval_freq=1000 \
  --save_freq=1000 \
  --name=joint_grpo \
  --checkpoint_dir=ckpt/joint_grpo/ \
  --gold_score_mode=f1rougebleudist \
  --reader_rl_learning \
  --use_grpo \
  --grpo_group_size=4 \
  --grpo_train_retriever \
  --grpo_retriever_group_size=4 \
  --grpo_kl_coeff=0.01 \
  --grpo_clip_epsilon=0.2 \
  --reader_model_type=t5-base \
  --lr=1e-5 \
  --warmup_steps=500 \
  --train_data="data/convai2/train.jsonl" \
  --eval_data="data/convai2/valid.jsonl" \
  --passages="data/corpora/story/story.jsonl" \
  --model_path="ckpt/stage1/checkpoint/step-12000" \
  --generation_length_penalty=1.0 \
  --use_wandb \
  --wandb_project=lapdog-grpo
```

### Evaluation

```bash
python evaluate.py \
  --model_path=ckpt/joint_grpo/checkpoint/step-12000 \
  --eval_data="data/convai2/valid.jsonl" \
  --passages="data/corpora/story/story.jsonl" \
  --write_results
```

## 📊 Key Features

### GRPO (Group Relative Policy Optimization)

Standard REINFORCE uses absolute rewards for policy gradients, leading to high variance. GRPO improves this by:

1. **Group Generation**: Generate k responses per prompt (default k=4)
2. **Relative Advantages**: Normalize rewards within each group: `A_i = (r_i - μ_r) / σ_r`
3. **KL Regularization**: Prevent policy divergence from reference model
4. **Stable Learning**: Reduced gradient variance and more consistent improvements

**Algorithm:**
```
For each training batch:
  1. Generate k responses per input
  2. Compute rewards (F1 + ROUGE + BLEU) for all responses
  3. Calculate group-relative advantages
  4. Compute KL divergence with reference model
  5. Update policy: loss = advantage * (-log_prob) + kl_penalty
```

### Joint GRPO Training

Unlike the original LAPDOG which trains retriever and generator separately, we enable **simultaneous GRPO optimization** for both:

- **Generator GRPO**: Optimizes response quality (F1, ROUGE, BLEU)
- **Retriever GRPO**: Optimizes document selection to maximize downstream generation quality
- **Coordinated Learning**: Both components learn to work together more effectively

### Generation Length Control

We implement a **symmetric length penalty** to control response verbosity:

```python
# Penalize deviations from target length ratio (default: 1.0)
deviation = |1.0 - length_ratio|
penalty = exp(-strength * deviation)
# Double penalty for too-short responses
strength = 2.0 if length_ratio < 1.0 else 1.0
```

This helps maintain BLEU scores by preventing overly verbose or brief responses.

## 🔧 Configuration

### Key Training Arguments

| Argument | Description | Default | Recommended |
|----------|-------------|---------|-------------|
| `--use_grpo` | Enable GRPO algorithm | False | True for RL training |
| `--grpo_group_size` | Responses per group | 4 | 4-8 |
| `--grpo_kl_coeff` | KL divergence weight | 0.1 | 0.01-0.1 |
| `--grpo_train_retriever` | Enable retriever GRPO | False | True for joint training |
| `--grpo_retriever_group_size` | Documents per retriever group | 5 | 4-8 |
| `--generation_length_penalty` | Length control strength | 0.0 | 1.0 |
| `--per_gpu_batch_size` | Batch size per GPU | 1 | 2-4 |
| `--accumulation_steps` | Gradient accumulation | 1 | 1-2 |
| `--lr` | Learning rate | 1e-4 | 1e-5 for GRPO |
| `--warmup_steps` | LR warmup steps | 100 | 500 for GRPO |

### Metric Scale Factors

Control the relative importance of different metrics:

```bash
--metric_scale_factors "{'bleu':0.01,'f1':0.01,'rouge':0.01}" \
--scale_rouge  # Scale ROUGE by 100x
```

## 📈 Monitoring with WandB

Track training progress with Weights & Biases:

```bash
--use_wandb \
--wandb_project=your-project-name \
--wandb_run_name=experiment-name
```

Logged metrics include:
- Loss (reader, retriever, KL divergence)
- Rewards (F1, ROUGE, BLEU, length ratio)
- Learning rates, gradient norms
- Memory usage and training speed

## 🎯 Evaluation Metrics

The model is evaluated on:

- **BLEU**: N-gram precision (emphasizes exact matches)
- **F1**: Token-level precision-recall balance
- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap  
- **ROUGE-L**: Longest common subsequence
- **Length Ratio**: Generated length / Reference length

## 📝 Citation

If you use this code, please cite both the original LAPDOG paper and acknowledge this extension:

```bibtex
@inproceedings{huang2023learning,
  title={Learning Retrieval Augmentation for Personalized Dialogue Generation},
  author={Huang, Qiushi and Yamada, Tome and Berant, Jonathan and Ju, Zhengying},
  booktitle={Proceedings of EMNLP},
  year={2023}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📜 License

This project maintains the same license as the original LAPDOG repository.

## 🙏 Acknowledgments

- Original LAPDOG implementation: [hqsiswiliam/LAPDOG](https://github.com/hqsiswiliam/LAPDOG)
- GRPO algorithm based on recent advances in group-relative policy optimization
- Built with PyTorch, Transformers, and Fairscale

## 📚 Additional Documentation

- `GRPO_IMPLEMENTATION_SUMMARY.md` - Detailed technical implementation notes
- `GRPO_QUICKSTART.md` - Quick start guide for GRPO training
- `JOINT_GRPO_IMPLEMENTATION.md` - Joint retriever-generator GRPO details
- `compute_metrics.py` - Standalone metric computation script

## 🐛 Troubleshooting

### OOM (Out of Memory) Errors

Reduce batch size or enable gradient checkpointing:
```bash
--per_gpu_batch_size=1 \
--accumulation_steps=4 \
--use_gradient_checkpoint_reader
```

### Slow Training

Use gradient accumulation with smaller batches for better GPU utilization:
```bash
--per_gpu_batch_size=2 \
--accumulation_steps=2
```

### VRAM Optimization

For index building, reduce embedder batch size:
```bash
--per_gpu_embedder_batch_size=512
```

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the maintainers.
