# Complete List of Modifications to Original LAPDOG

This document provides a detailed, accurate list of all changes made to the original LAPDOG codebase to add GRPO functionality.

## Files Modified

### 1. src/options.py

**Added 9 new command-line arguments:**

```python
# GRPO-specific parameters
--use_grpo                      # Enable GRPO algorithm (default: False)
--grpo_group_size              # Number of responses per group (default: 4)
--grpo_kl_coeff                # KL divergence coefficient (default: 0.1)
--grpo_clip_epsilon            # PPO-style clipping epsilon (default: 0.2)
--grpo_top_k_ratio             # Ratio of top responses for advantage (default: 0.5)
--reference_model_path         # Path to reference model for KL computation
--grpo_train_retriever         # Enable joint GRPO for retriever (default: False)
--grpo_retriever_group_size    # Documents per retriever group (default: 5)
--metric_scale_factors         # Scale factors for metrics (JSON string)
--generation_length_penalty    # Length penalty strength (default: 0.0)
```

### 2. src/lapdog.py

#### A. New Methods Added (4 methods):

1. **`_initialize_reference_model()`** (lines ~100-120)
   - Purpose: Initialize frozen reference model for KL divergence
   - What it does:
     - Loads reference model from checkpoint OR creates copy of current reader
     - Freezes all parameters (no gradient updates)
     - Moves to same device as main model

2. **`generate_with_sampling(tokens, query, temperature=1.0)`** (lines ~300-350)
   - Purpose: Generate diverse responses using sampling
   - What it does:
     - Uses temperature and top_p nucleus sampling
     - Returns generated token IDs
     - Different from original deterministic generation

3. **`f1rougebleu_score_grpo(reader_ids, reader_mask, decoder_input_ids, labels, cfg, bsz, group_size)`** (lines ~700-850)
   - Purpose: Extended reward computation for GRPO
   - What it does:
     - Generates `k` responses per input (k = group_size)
     - Computes F1, ROUGE, BLEU for all responses
     - Applies length penalty if enabled
     - Returns grouped scores [group_size, bsz, n_context]

4. **`grpo_learning(reader_ids, reader_mask, decoder_input_ids, labels, cfg, bsz)`** (lines ~1100-1250)
   - Purpose: Core GRPO algorithm implementation
   - What it does:
     - Calls `f1rougebleu_score_grpo()` to generate group responses
     - Computes group-relative advantages: `(reward - mean) / std`
     - Calculates KL divergence with reference model
     - Combines advantage and KL into policy loss
     - Supports joint retriever GRPO if enabled

#### B. Modified Methods (2 methods):

1. **`__init__()`** (constructor)
   - **Added lines:**
     ```python
     self.opt = opt  # Store options
     if opt.use_grpo:
         self._initialize_reference_model()  # Initialize reference model
     ```
   - **Purpose:** Set up GRPO infrastructure if enabled

2. **`reinforcement_learning()`** (EXISTED IN ORIGINAL - NOW MODIFIED)
   - **Original behavior:**
     - Implemented standard REINFORCE algorithm
     - Used `f1rougebleu_score()` for rewards
     - Applied policy gradient directly

   - **New behavior (added routing logic):**
     ```python
     def reinforcement_learning(self, ...):
         if self.opt.use_grpo:
             return self.grpo_learning(...)  # NEW: Route to GRPO
         else:
             # ORIGINAL: Keep original REINFORCE logic
             gold_score, generated, input_tokens = self.f1rougebleu_score(...)
             # ... original REINFORCE code ...
     ```
   - **Key point:** Method already existed - we just added branching logic

#### C. Enhanced Existing Method:

**`f1rougebleu_score()`** (EXISTED - ENHANCED)
- **Added:** Length penalty computation
  ```python
  if self.opt.generation_length_penalty > 0:
      # Symmetric penalty for deviation from ratio=1.0
      deviation = torch.abs(1.0 - length_ratio)
      strength = torch.ones_like(length_ratio) * self.opt.generation_length_penalty
      strength[length_ratio < 1.0] *= 2.0  # Double penalty for short outputs
      length_penalty = torch.exp(-strength * deviation)
      group_score = group_score * length_penalty
  ```

### 3. train.py

**Added WandB integration (lines ~50-70, ~235-260):**

```python
# At top of file
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# In main()
if opt.use_wandb and opt.is_main:
    wandb.init(
        project=opt.wandb_project,
        entity=opt.wandb_entity,
        name=opt.wandb_run_name,
        config={...}
    )

# In training loop
if use_wandb:
    wandb.log(metrics, step=step)
```

**Added CUDA cache clearing (line ~148):**
```python
if step % 50 == 0:
    torch.cuda.empty_cache()  # Prevent memory accumulation
```

## Summary of Changes

### What Was Already in Original LAPDOG:
 `reinforcement_learning()` method (REINFORCE algorithm)  
 `f1rougebleu_score()` method (reward computation)  
 RL training infrastructure  
 Retriever and generator joint training  
 Command-line argument system  

### What We Added:
âž **4 new methods** in `src/lapdog.py`:
   - `_initialize_reference_model()`
   - `generate_with_sampling()`
   - `f1rougebleu_score_grpo()`
   - `grpo_learning()`

âž **Modified 2 existing methods**:
   - `__init__()` - Added GRPO initialization
   - `reinforcement_learning()` - Added routing to GRPO

âž **Enhanced 1 existing method**:
   - `f1rougebleu_score()` - Added length penalty

âž **10 new command-line arguments** in `src/options.py`

âž **WandB logging integration** in `train.py`

âž **Memory optimization** in `train.py` (CUDA cache clearing)

### What We Did NOT Add:
 New training script (used existing `train.py`)  
 New evaluation logic (used existing `evaluate.py`)  
 New data processing (used existing task modules)  
 New model architectures (used existing T5 + Contriever)  

## Code Statistics

**Lines of code added:**
- `src/lapdog.py`: ~350 lines
- `src/options.py`: ~80 lines
- `train.py`: ~50 lines
- **Total: ~480 lines**

**Methods added:** 4 new methods  
**Methods modified:** 3 existing methods  
**Arguments added:** 10 new CLI arguments  

## Backward Compatibility

 **100% backward compatible:**
- Original REINFORCE still works with `--reader_rl_learning` (without `--use_grpo`)
- All original training modes still functional
- No breaking changes to existing code

## Testing GRPO vs Original

**To use GRPO:**
```bash
python train.py --reader_rl_learning --use_grpo --grpo_group_size 4
```

**To use original REINFORCE:**
```bash
python train.py --reader_rl_learning  # No --use_grpo flag
```

## Key Implementation Details

### Length Penalty
- **Location:** `src/lapdog.py::f1rougebleu_score()`
- **Formula:** `exp(-strength * |1 - ratio|)`
- **Special case:** 2x penalty for short outputs (ratio < 1.0)

### GRPO Advantage
- **Location:** `src/lapdog.py::grpo_learning()`
- **Formula:** `(reward - mean(group_rewards)) / std(group_rewards)`
- **Applied per-group:** Each input has its own group statistics

### KL Divergence
- **Location:** `src/lapdog.py::grpo_learning()`
- **Computed against:** Frozen reference model (copy of initial model)
- **Formula:** `kl_coeff * (current_logprob - ref_logprob)`

### Joint Retriever GRPO
- **Location:** `src/lapdog.py::grpo_learning()`
- **Enabled with:** `--grpo_train_retriever --grpo_retriever_group_size k`
- **Method:** Sample k documents, compute group advantages for retriever

## Files Not Modified

The following files remain unchanged from original LAPDOG:
- `src/evaluation.py`
- `src/model_io.py`
- `src/index_io.py`
- `src/index.py`
- `src/retrievers.py`
- `src/modeling_t5.py`
- `src/modeling_bert.py`
- `src/dist_utils.py`
- `src/slurm.py`
- `src/util.py`
- `src/tasks/*.py`
- `evaluate.py`

## Conclusion

This implementation **extends** the original LAPDOG with GRPO capabilities while maintaining full backward compatibility. The core contribution is the GRPO algorithm implementation (`grpo_learning()` and `f1rougebleu_score_grpo()` methods) plus supporting infrastructure for reference models, stochastic sampling, and length control.
