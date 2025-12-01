# GRPO Implementation - Quick Start Guide

##  Implementation Complete!

All GRPO modifications have been successfully implemented and backed up.

##  Files Modified

1. **src/options.py** - Added 5 new GRPO parameters
2. **src/lapdog.py** - Added 5 new methods + modified 2 existing methods
3. **Backups created in**: `src/backup/`

##  How to Use GRPO

### Basic GRPO Training Command

```bash
python train.py \
    --reader_rl_learning \
    --use_grpo \
    --grpo_group_size 4 \
    --grpo_kl_coeff 0.1 \
    --gold_score_mode f1rougebleudist \
    --name grpo_experiment \
    --checkpoint_dir ./ckpt/grpo/ \
    --train_data data/convai2/train.jsonl \
    --eval_data data/convai2/valid.jsonl \
    --per_gpu_batch_size 2
```

### Using Original REINFORCE (for comparison)

```bash
python train.py \
    --reader_rl_learning \
    --gold_score_mode f1rougebleudist \
    --name reinforce_experiment \
    --checkpoint_dir ./ckpt/reinforce/ \
    --train_data data/convai2/train.jsonl \
    --eval_data data/convai2/valid.jsonl \
    --per_gpu_batch_size 2
```

##  GRPO Parameters Explained

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use_grpo` | False | Enable GRPO algorithm (otherwise uses REINFORCE) |
| `--grpo_group_size` | 4 | Number of responses to generate per prompt |
| `--grpo_kl_coeff` | 0.1 | KL divergence coefficient (controls how much model can deviate from reference) |
| `--grpo_top_k_ratio` | 0.5 | Ratio of top responses for advantage computation (reserved for future use) |
| `--reference_model_path` | None | Path to reference model checkpoint (if None, uses copy of initial model) |

##  What GRPO Does

**Standard REINFORCE:**
- Generates 1 response per input
- Uses absolute reward for gradient
- No KL regularization

**GRPO (Group Relative Policy Optimization):**
- Generates k responses per input (default k=4)
- Computes group-relative advantages: `A_i = (r_i - _r) / _r`
- Adds KL divergence penalty with reference model
- More stable training with variance reduction

##  Key Differences

### Algorithm Flow

**REINFORCE:**
```
Input  Generate 1 response  Compute reward  Policy gradient
```

**GRPO:**
```
Input  Generate k responses  Compute all rewards  
Group advantages  Policy gradient with KL penalty
```

### Loss Functions

**REINFORCE:**
```python
loss = -log_prob * reward
```

**GRPO:**
```python
advantage = (reward - mean(rewards)) / std(rewards)
kl_penalty = kl_coeff * (current_logprob - reference_logprob)
loss = -log_prob * advantage + kl_penalty
```

##  Hyperparameter Tuning Tips

### Group Size (`--grpo_group_size`)
- **Small (2-4)**: Faster training, less stable
- **Medium (4-8)**: Good balance (recommended)
- **Large (8-16)**: More stable but slower and memory-intensive

### KL Coefficient (`--grpo_kl_coeff`)
- **Low (0.01-0.05)**: More exploration, risk of instability
- **Medium (0.1-0.2)**: Balanced (recommended)
- **High (0.5-1.0)**: Conservative, slower learning

##  Example Training Scripts

### Experiment 1: GRPO with Conservative KL
```bash
python train.py \
    --reader_rl_learning --use_grpo \
    --grpo_group_size 4 --grpo_kl_coeff 0.2 \
    --gold_score_mode f1rougebleudist \
    --name grpo_conservative \
    --per_gpu_batch_size 2
```

### Experiment 2: GRPO with More Exploration
```bash
python train.py \
    --reader_rl_learning --use_grpo \
    --grpo_group_size 8 --grpo_kl_coeff 0.05 \
    --gold_score_mode f1rougebleudist \
    --name grpo_exploratory \
    --per_gpu_batch_size 1
```

### Experiment 3: Using Pre-trained Reference Model
```bash
python train.py \
    --reader_rl_learning --use_grpo \
    --grpo_group_size 4 --grpo_kl_coeff 0.1 \
    --reference_model_path ./ckpt/pretrained/checkpoint \
    --gold_score_mode f1rougebleudist \
    --name grpo_with_reference \
    --per_gpu_batch_size 2
```

##  Important Notes

1. **Memory Usage**: GRPO uses `group_size` times more memory during generation
   - Reduce `per_gpu_batch_size` if you run out of memory
   - Example: If batch_size=4 works for REINFORCE, use batch_size=1 for GRPO with group_size=4

2. **Gold Score Mode**: GRPO only works with `--gold_score_mode f1rougebleudist`
   - This combines F1, ROUGE, and BLEU metrics
   - Required for proper reward computation

3. **Reference Model**: 
   - If `--reference_model_path` is not specified, uses copy of initial model
   - Reference model is automatically frozen (no gradient updates)
   - Loaded onto GPU for KL computation

4. **Training Time**:
   - GRPO is approximately `group_size` times slower per step than REINFORCE
   - But converges faster due to variance reduction
   - Overall training time may be similar or better


##  Summary of Implementation

### New Methods in `lapdog.py`:
1. `_initialize_reference_model()` - Sets up frozen reference model
2. `generate_with_sampling()` - Generates diverse responses
3. `f1rougebleu_score_grpo()` - Computes rewards for multiple responses
4. `grpo_learning()` - Core GRPO algorithm
5. `reinforce_learning()` - Original REINFORCE (extracted for clarity)

### Modified Methods:
1. `__init__()` - Initializes reference model when GRPO is enabled
2. `reinforcement_learning()` - Routes to GRPO or REINFORCE

### New Options in `options.py`:
1. `--use_grpo` - Enable GRPO
2. `--grpo_group_size` - Group size
3. `--grpo_kl_coeff` - KL coefficient
4. `--grpo_top_k_ratio` - Reserved for future use
5. `--reference_model_path` - Reference model path

---

