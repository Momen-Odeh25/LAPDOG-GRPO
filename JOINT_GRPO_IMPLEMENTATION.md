# Joint GRPO Training Implementation

## Overview

This implementation adds **joint GRPO training** for both the retriever and generator in LAPDOG. You can now train:
1. **Generator-only GRPO** (existing, default behavior)
2. **Joint GRPO** (new, trains both retriever and generator)

## Architecture

### Outer Loop: Retriever Policy
- For each query $q$, the retriever policy $\pi_\phi$ samples $k$ documents from the corpus
- Each document is an "action" for the retriever
- Documents are sampled with temperature to ensure diversity

### Inner Loop: Generator Policy  
- For each of the $k$ documents, the generator policy $\pi_\theta$ samples $G$ responses
- Total of $k \times G$ responses generated per training example
- Each response is scored with your reward metrics (F1, ROUGE, BLEU)

### Loss Computation

**Generator Loss (per document):**
$$\mathcal{L}_{Gen}^{(i)} = \frac{1}{G}\sum_{j=1}^{G}\frac{1}{|o_{i,j}|}\sum_{t=1}^{|o_{i,j}|}\min(r_t(\theta)\hat{A}_{i,j,t}, \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_{i,j,t}) - \beta_{KL} D_{KL}(\pi_\theta||\pi_{ref})$$

Where:
- $r_t = \exp(\log \pi_{old} - \log \pi_\theta)$ is the probability ratio per token
- $\hat{A}_{i,j,t}$ is the group-relative advantage (normalized within the $G$ responses)
- PPO clipping prevents catastrophic updates

**Retriever Loss:**
$$\mathcal{L}_{Ret} = \frac{1}{k}\sum_{i=1}^{k}\min(r_i(\phi)\hat{A}_i, \text{clip}(r_i, 1-\epsilon, 1+\epsilon)\hat{A}_i) - \beta_{KL} D_{KL}(\pi_\phi||\pi_{ref})$$

Where:
- Retriever reward: $R_{Ret}^{(i)} = \frac{1}{G}\sum_{j=1}^{G} R_{i,j}$ (average generator reward for document $i$)
- $\hat{A}_i$ is the group-relative advantage (normalized within the $k$ documents)

**Total Loss:**
$$\mathcal{L}_{Total} = \sum_{i=1}^{k}\mathcal{L}_{Gen}^{(i)} + \mathcal{L}_{Ret}$$

## New Command Line Arguments

```bash
# Enable joint GRPO training (default: False, generator-only)
--grpo_train_retriever

# Number of documents to sample for retriever GRPO (default: 5)
--grpo_retriever_group_size 5

# Existing generator group size (default: 4)
--grpo_group_size 4

# PPO clipping epsilon (applies to both retriever and generator, default: 0.2)
--grpo_clip_epsilon 0.2

# KL divergence coefficient (applies to both, default: 0.1)
--grpo_kl_coeff 0.01
```

## Usage Examples

### Generator-Only GRPO (Existing Behavior)
```bash
python train.py \
  --use_grpo \
  --grpo_group_size 4 \
  --grpo_kl_coeff 0.01 \
  --grpo_clip_epsilon 0.2 \
  --reader_rl_learning \
  [... other args ...]
```

### Joint GRPO Training (New)
```bash
python train.py \
  --use_grpo \
  --grpo_train_retriever \
  --grpo_group_size 4 \
  --grpo_retriever_group_size 5 \
  --grpo_kl_coeff 0.01 \
  --grpo_clip_epsilon 0.2 \
  --reader_rl_learning \
  [... other args ...]
```

## Implementation Details

### Key Functions

1. **`sample_documents_for_grpo()`** (lines ~700-760)
   - Samples $k$ diverse documents from index using temperature
   - Stores old policy log-probabilities for PPO ratio computation
   - Uses softmax over retriever scores with temperature scaling

2. **`grpo_retriever_learning()`** (lines ~1283-1383)
   - Computes retriever GRPO loss with PPO clipping
   - Group-relative advantages across $k$ documents
   - KL divergence against frozen reference retriever
   - Returns single retriever loss scalar

3. **`joint_grpo_learning()`** (lines ~1385-1570)
   - Main orchestration function for joint training
   - Outer loop: samples $k$ documents
   - Inner loop: for each doc, samples $G$ responses and computes generator loss
   - Aggregates rewards for retriever (average across $G$ responses per doc)
   - Combines generator and retriever losses

4. **`reinforcement_learning()`** (lines ~1103-1119)
   - Routing function: decides between REINFORCE, generator-only GRPO, or joint GRPO
   - Checks `grpo_train_retriever` flag to enable joint training

### Reference Models

Both retriever and generator have frozen reference models for KL divergence:
- **Reference Generator**: Copy of reader (FiD model) frozen at training start
- **Reference Retriever**: Copy of retriever (Contriever) frozen at training start

Initialized in `__init__`:
```python
if opt.use_grpo:
    self._initialize_reference_model()  # Generator reference
    if opt.grpo_train_retriever:
        self._initialize_reference_retriever()  # Retriever reference
```

## Computational Cost

Joint GRPO is more expensive than generator-only:
- **Generator-only**: $G$ forward passes per training example
- **Joint GRPO**: $k \times G$ forward passes per training example

Example with default settings:
- Generator-only: 4 forward passes
- Joint GRPO: 5 × 4 = 20 forward passes

**Recommendation**: Start with smaller batch sizes and consider:
- Reducing `grpo_retriever_group_size` (e.g., 3 instead of 5)
- Using gradient accumulation to handle memory constraints
- Training generator-only first, then enabling joint training

## Debugging

If you encounter issues:

1. **Memory errors**: Reduce `grpo_retriever_group_size` or batch size
2. **NaN losses**: Check KL coefficients (reduce if too high), verify temperature scaling
3. **No improvement**: 
   - Ensure `--reader_rl_learning` is enabled
   - Check that retriever is not frozen (`train_retriever=True` in forward pass)
   - Monitor retriever advantages (should not all be near zero)

## Differences from Generator-Only GRPO

| Aspect | Generator-Only | Joint GRPO |
|--------|---------------|------------|
| Documents used | Fixed (top-k from retriever) | Sampled with temperature |
| Retriever updated | No | Yes (with GRPO) |
| Forward passes | $G$ per example | $k \times G$ per example |
| Loss components | Generator only | Generator + Retriever |
| Reference models | Reader only | Reader + Retriever |
| Computational cost | Baseline | $k \times$ baseline |

## Mathematical Notation

- $q$: Query
- $k$: Number of documents sampled (retriever group size)
- $G$: Number of responses per document (generator group size)
- $d_i$: $i$-th sampled document
- $\hat{y}_{i,j}$: $j$-th response generated using document $d_i$
- $R_{i,j}$: Reward for response $\hat{y}_{i,j}$
- $R_{Ret}^{(i)}$: Retriever reward for document $d_i$ (average of $\{R_{i,1}, ..., R_{i,G}\}$)
- $\pi_\theta$: Generator policy (reader)
- $\pi_\phi$: Retriever policy
- $\hat{A}_{i,j}$: Generator advantage (group-relative within $G$ responses)
- $\hat{A}_i$: Retriever advantage (group-relative within $k$ documents)
- $\epsilon$: PPO clipping epsilon (default 0.2)
- $\beta_{KL}$: KL divergence coefficient (default 0.01)

## Code Structure

```
lapdog.py
├── __init__
│   ├── _initialize_reference_model() [Generator reference]
│   └── _initialize_reference_retriever() [Retriever reference]
├── sample_documents_for_grpo() [New]
├── grpo_retriever_learning() [New]
├── joint_grpo_learning() [New]
├── reinforcement_learning() [Modified - routing]
├── grpo_learning() [Existing - generator-only]
└── forward()
    └── calls reinforcement_learning() [Modified - pass extra params]
```

## Hyperparameter Tuning

Start with these values and tune based on results:

```bash
--grpo_retriever_group_size 3      # Start small due to computational cost
--grpo_group_size 4                # Standard generator group size
--grpo_clip_epsilon 0.2            # Standard PPO clipping
--grpo_kl_coeff 0.01               # Moderate KL penalty
--metric_scale_factors "{'bleu':0.01,'f1':1.0,'rouge':0.01}"
```

Increase `grpo_retriever_group_size` to 5-7 if:
- Memory allows
- Training speed is acceptable
- You want more diverse document exploration

Increase `grpo_kl_coeff` to 0.05-0.1 if:
- Policies diverge too quickly from reference
- Training becomes unstable
- Want more conservative updates

