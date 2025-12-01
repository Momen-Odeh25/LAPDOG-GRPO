# GRPO Implementation Summary

## Overview
This document summarizes the implementation of GRPO (Group Relative Policy Optimization) for fine-tuning the generator in the LAPDOG model.

## Files Modified

### 1. src/options.py
Added GRPO-specific command-line arguments:

```python
--use_grpo                  # Enable GRPO instead of standard REINFORCE
--grpo_group_size          # Number of responses per group (default: 4)
--grpo_kl_coeff            # KL divergence coefficient (default: 0.1)
--grpo_top_k_ratio         # Ratio of top responses for advantage (default: 0.5)
--reference_model_path     # Path to reference model for KL computation
```

### 2. src/lapdog.py

#### New Methods Added:

1. **`_initialize_reference_model()`**
   - Initializes frozen reference model for KL divergence computation
   - Can load from path or create copy of current reader
   - Automatically freezes all parameters

2. **`generate_with_sampling(tokens, query, temperature=1.0)`**
   - Generates responses with sampling for diversity
   - Uses temperature and top_p for stochastic generation
   - Essential for generating multiple diverse responses per input

3. **`f1rougebleu_score_grpo(reader_ids, reader_mask, decoder_input_ids, labels, cfg, bsz, group_size)`**
   - Extended version of `f1rougebleu_score` for GRPO
   - Generates `k` responses per input (where k = group_size)
   - Computes rewards (F1, ROUGE, BLEU) for all responses
   - Returns grouped scores with shape [group_size, bsz, n_context]

4. **`grpo_learning(reader_ids, reader_mask, decoder_input_ids, labels, cfg, bsz)`**
   - Core GRPO implementation
   - Computes group-relative advantages: `A_i = (r_i - μ_r) / σ_r`
   - Calculates KL divergence with reference model
   - Implements GRPO loss: `advantage * (-log_prob) + KL_penalty`

#### Modified Methods:

1. **`__init__()`**
   - Added reference model initialization when `use_grpo=True`
   - Calls `_initialize_reference_model()` during setup

2. **`reinforcement_learning()`** (ORIGINAL METHOD - NOW ACTS AS ROUTER)
   - **Original behavior**: Implemented standard REINFORCE algorithm
   - **Modified behavior**: Now routes to appropriate RL algorithm:
     - If `--use_grpo` is set  calls `grpo_learning()` (new GRPO implementation)
     - If not  calls original REINFORCE logic
   - Maintains backward compatibility with existing code

## GRPO Algorithm Implementation

The implementation follows the GRPO algorithm:

### Step 1: Group Generation
For each input prompt, generate `k` responses (default k=4):
```python
grouped_scores, all_generated, all_input_tokens = self.f1rougebleu_score_grpo(
    reader_ids, reader_mask, decoder_input_ids, labels, cfg, bsz, group_size
)
```

### Step 2: Reward Computation
Calculate rewards for all responses using F1 + ROUGE + BLEU metrics:
```python
group_score = pt_f1 + pt_rouge + pt_bleu_normed
```

### Step 3: Group-Relative Advantage
Compute normalized advantages within each group:
```python
group_mean = grouped_scores.mean(dim=0, keepdim=True)
group_std = grouped_scores.std(dim=0, keepdim=True) + 1e-8
advantages = (grouped_scores - group_mean) / group_std
```

### Step 4: KL Divergence
Compute KL penalty with frozen reference model:
```python
kl_penalty = grpo_kl_coeff * (current_logprobs - ref_logprobs)
```

### Step 5: Policy Loss
Combine advantage and KL penalty:
```python
policy_loss = advantage * (-current_logprobs) + kl_penalty
```

## Usage

### Training with GRPO

```bash
python train.py \
    --reader_rl_learning \
    --use_grpo \
    --grpo_group_size 4 \
    --grpo_kl_coeff 0.1 \
    --gold_score_mode f1rougebleudist \
    [other training arguments]
```

### Training with Original REINFORCE

```bash
python train.py \
    --reader_rl_learning \
    --gold_score_mode f1rougebleudist \
    [other training arguments]
```

## Key Features

1. **Backward Compatibility**: Original REINFORCE still works when `--use_grpo` is not specified
2. **Flexible Reference Model**: Can use existing checkpoint or auto-create from initial model
3. **Group-Based Rewards**: Leverages existing F1/ROUGE/BLEU metrics
4. **KL Regularization**: Prevents policy from diverging too far from reference
5. **Sampling-Based Generation**: Uses temperature and top_p for diverse responses

## Configuration Recommendations

- **Group Size**: Start with 4, increase to 8 for more stable gradients
- **KL Coefficient**: 0.1 is a good starting point, decrease if learning is too conservative
- **Temperature**: Fixed at 1.0 for diversity (can be made configurable if needed)
- **Top-p**: Fixed at 0.9 for nucleus sampling

## Summary of Changes

### What Already Existed in Original LAPDOG:
- `reinforcement_learning()` method with standard REINFORCE
- `f1rougebleu_score()` for reward computation
- RL training infrastructure

### What We Added:
- GRPO algorithm implementation (`grpo_learning()` method)
- Group-based generation (`f1rougebleu_score_grpo()` method)
- Reference model support for KL divergence
- Stochastic sampling (`generate_with_sampling()` method)
- Router logic in existing `reinforcement_learning()` method
- Command-line arguments for GRPO configuration

## Next Steps

To use this implementation:

1. Ensure you have a trained retriever-reader model
2. Enable GRPO with `--use_grpo` flag
3. Set `--gold_score_mode f1rougebleudist` (required)
4. Use `--reader_rl_learning` to activate RL training
5. Monitor KL divergence and reward progression during training

