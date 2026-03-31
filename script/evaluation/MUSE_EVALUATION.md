# MUSE Evaluation Scripts

This directory contains three specialized scripts for evaluating machine unlearning using the MUSE (Machine Unlearning Six-way Evaluation) framework. The scripts are separated by computational cost to allow flexible evaluation workflows.

## Scripts Overview

| Script | Metrics | Computational Cost | Typical Runtime* |
|--------|---------|-------------------|------------------|
| `run_muse_core.py` | 1-4 | Low | 5-15 minutes |
| `run_muse_scalability.py` | 5 | Medium | 15-60 minutes |
| `run_muse_sustainability.py` | 6 | High | 30-120 minutes |

*Runtime depends on model size, dataset size, and hardware

---

## 1. Core Metrics Evaluation (`run_muse_core.py`)

Evaluates the essential MUSE metrics for unlearning effectiveness and utility preservation.

### Metrics Included

1. **Verbatim Memorization**: Tests if model can reproduce exact forgotten text
2. **Knowledge Memorization**: Tests semantic knowledge retention via paraphrased questions
3. **Privacy Leakage**: Membership Inference Attack to detect training data
4. **Utility Preservation**: Performance on retain set

### Usage

```bash
# Basic usage
python script/evaluation/run_muse_core.py \
    --model_path /path/to/unlearned_model \
    --base_model meta-llama/Llama-3.2-1B

# With custom datasets
python script/evaluation/run_muse_core.py \
    --model_path /path/to/unlearned_model \
    --base_model meta-llama/Llama-3.2-1B \
    --forget_data data/tofu/forget10.json \
    --retain_data data/tofu/retain90.json \
    --paraphrased_data data/tofu/paraphrased_forget10.json \
    --non_training_data data/tofu/non_training.json

# With sampling for faster evaluation
python script/evaluation/run_muse_core.py \
    --model_path /path/to/unlearned_model \
    --base_model meta-llama/Llama-3.2-1B \
    --sample_num 50 \
    --output_path results/core_metrics.json
```

### Arguments

- `--model_path`: Path to the trained/unlearned model (required)
- `--base_model`: Base model name for tokenizer (default: meta-llama/Llama-3.2-1B)
- `--config`: Path to MUSE config file (default: config/muse_eval.yaml)
- `--output_path`: Output JSON file (default: model_path/muse_core_results.json)
- `--forget_data`: Path to forget dataset
- `--retain_data`: Path to retain dataset
- `--paraphrased_data`: Path to paraphrased forget dataset (for metric 2)
- `--non_training_data`: Path to non-training dataset (for metric 3)
- `--sample_num`: Number of examples to sample per dataset
- `--device`: Device for evaluation (cuda/cpu)

### Output Example

```json
{
  "verbatim": {
    "verbatim_rouge_mean": 0.15,
    "verbatim_rouge_std": 0.08
  },
  "knowledge": {
    "knowledge_rouge_mean": 0.22,
    "knowledge_rouge_std": 0.11
  },
  "privacy": {
    "mia_accuracy": 0.52,
    "forget_confidence_mean": -2.34,
    "non_training_confidence_mean": -2.41
  },
  "utility": {
    "utility_rouge_mean": 0.73,
    "utility_rouge_std": 0.09
  }
}
```

---

## 2. Scalability Evaluation (`run_muse_scalability.py`)

Tests whether unlearning effectiveness remains consistent across different forget set sizes.

### Metric Included

5. **Scalability**: Evaluates verbatim memorization on progressively larger forget subsets

### Usage

```bash
# Basic usage (default subset sizes: 10, 50, 100, 200, 500)
python script/evaluation/run_muse_scalability.py \
    --model_path /path/to/unlearned_model \
    --base_model meta-llama/Llama-3.2-1B

# Custom subset sizes
python script/evaluation/run_muse_scalability.py \
    --model_path /path/to/unlearned_model \
    --base_model meta-llama/Llama-3.2-1B \
    --subset_sizes 10 25 50 100 250 500 1000

# With custom forget dataset
python script/evaluation/run_muse_scalability.py \
    --model_path /path/to/unlearned_model \
    --base_model meta-llama/Llama-3.2-1B \
    --forget_data data/tofu/forget_full.json \
    --subset_sizes 50 100 200 500
```

### Arguments

- `--model_path`: Path to the trained/unlearned model (required)
- `--base_model`: Base model name for tokenizer (default: meta-llama/Llama-3.2-1B)
- `--config`: Path to MUSE config file (default: config/muse_eval.yaml)
- `--output_path`: Output JSON file (default: model_path/muse_scalability_results.json)
- `--forget_data`: Path to forget dataset
- `--subset_sizes`: List of subset sizes to evaluate (e.g., 10 50 100 200)
- `--device`: Device for evaluation (cuda/cpu)

### Output Example

```json
{
  "scalability": {
    "size_10": {
      "verbatim_rouge_mean": 0.14,
      "subset_size": 10
    },
    "size_50": {
      "verbatim_rouge_mean": 0.16,
      "subset_size": 50
    },
    "size_100": {
      "verbatim_rouge_mean": 0.15,
      "subset_size": 100
    },
    "size_200": {
      "verbatim_rouge_mean": 0.17,
      "subset_size": 200
    }
  }
}
```

### Interpretation

- **Range < 0.1**: Highly consistent - excellent scalability
- **Range 0.1-0.2**: Moderately consistent - good scalability
- **Range > 0.2**: Variable performance - scalability concerns

---

## 3. Sustainability Evaluation (`run_muse_sustainability.py`)

Tests whether unlearning persists after continued training on retain data.

### Metric Included

6. **Sustainability**: Evaluates unlearning robustness through retraining

### Process

1. Evaluate baseline (pre-retraining)
2. Fine-tune model on retain set for N steps
3. Re-evaluate (post-retraining)
4. Compare results to detect knowledge leakage

### Usage

```bash
# Full sustainability evaluation with 100 retraining steps
python script/evaluation/run_muse_sustainability.py \
    --model_path /path/to/unlearned_model \
    --base_model meta-llama/Llama-3.2-1B \
    --num_steps 100

# With custom retraining parameters
python script/evaluation/run_muse_sustainability.py \
    --model_path /path/to/unlearned_model \
    --base_model meta-llama/Llama-3.2-1B \
    --num_steps 200 \
    --learning_rate 1e-5 \
    --batch_size 8 \
    --save_retrained_model

# Placeholder evaluation (skip actual retraining)
python script/evaluation/run_muse_sustainability.py \
    --model_path /path/to/unlearned_model \
    --base_model meta-llama/Llama-3.2-1B \
    --skip_retraining
```

### Arguments

- `--model_path`: Path to the trained/unlearned model (required)
- `--base_model`: Base model name for tokenizer (default: meta-llama/Llama-3.2-1B)
- `--config`: Path to MUSE config file (default: config/muse_eval.yaml)
- `--output_path`: Output JSON file (default: model_path/muse_sustainability_results.json)
- `--retrained_model_path`: Path to save retrained model (default: model_path/retrained_for_sustainability)
- `--forget_data`: Path to forget dataset
- `--retain_data`: Path to retain dataset
- `--num_steps`: Number of retraining steps (default: 100)
- `--learning_rate`: Learning rate for retraining (default: 5e-5)
- `--batch_size`: Batch size for retraining (default: 4)
- `--device`: Device for evaluation (cuda/cpu)
- `--skip_retraining`: Skip retraining and use placeholder (fast testing)
- `--save_retrained_model`: Save the retrained model checkpoint

### Output Example

```json
{
  "sustainability": {
    "pre_retraining_rouge_mean": 0.15,
    "post_retraining_rouge_mean": 0.18,
    "rouge_delta": 0.03,
    "rouge_delta_percentage": 20.0,
    "retraining_info": {
      "total_steps": 100,
      "training_time_seconds": 245.3,
      "average_loss": 1.42,
      "learning_rate": 5e-5,
      "batch_size": 4
    }
  }
}
```

### Interpretation

- **|Δ| < 0.05**: Excellent sustainability - minimal knowledge leakage
- **|Δ| < 0.15**: Good sustainability - acceptable robustness
- **Δ > 0.15**: Concern - significant knowledge leakage after retraining
- **Δ < 0**: Improved - unlearning strengthened by continued training

---

## Recommended Workflow

### Quick Evaluation (5-15 minutes)
```bash
# Run core metrics only
python script/evaluation/run_muse_core.py \
    --model_path /path/to/model \
    --sample_num 50
```

### Standard Evaluation (20-45 minutes)
```bash
# Run core metrics + scalability
python script/evaluation/run_muse_core.py \
    --model_path /path/to/model \
    --sample_num 100

python script/evaluation/run_muse_scalability.py \
    --model_path /path/to/model \
    --subset_sizes 10 50 100 200
```

### Comprehensive Evaluation (1-2 hours)
```bash
# Run all metrics including sustainability
python script/evaluation/run_muse_core.py \
    --model_path /path/to/model

python script/evaluation/run_muse_scalability.py \
    --model_path /path/to/model

python script/evaluation/run_muse_sustainability.py \
    --model_path /path/to/model \
    --num_steps 200 \
    --save_retrained_model
```

---

## Configuration File

All scripts use `config/muse_eval.yaml` by default. Key settings:

```yaml
# Evaluation sampling
evaluation:
  sample_num: 100  # null for all examples

# Dataset paths
datasets:
  forget_set: "data/tofu/forget10.json"
  retain_set: "data/tofu/retain90.json"
  paraphrased_set: null
  non_training_set: null

# Metric configurations
metrics:
  scalability:
    subset_sizes: [10, 50, 100, 200]

  sustainability:
    additional_training_steps: 100
```

---

## Output Files

Each script generates a JSON file with detailed results:

- **run_muse_core.py** → `muse_core_results.json`
- **run_muse_scalability.py** → `muse_scalability_results.json`
- **run_muse_sustainability.py** → `muse_sustainability_results.json`

All results include metadata about model, datasets, and evaluation parameters.

---

## Comparison with Original Script

The original `run_muse_eval.py` runs all metrics together. The new scripts provide:

✓ **Modularity**: Run only needed metrics
✓ **Efficiency**: Avoid expensive computations when not needed
✓ **Flexibility**: Different parameters per metric
✓ **Clarity**: Focused evaluation per script
✓ **Better logging**: Detailed progress per metric type

You can still use `run_muse_eval.py` for all-in-one evaluation by enabling/disabling metrics in the config file.

---

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- ROUGE Score
- PyYAML
- tqdm
- numpy

Install via:
```bash
pip install -r requirements.txt
```

---

## Troubleshooting

### Out of Memory Errors
- Reduce `--sample_num`
- Use smaller `--batch_size` for sustainability evaluation
- Use `--device cpu` (slower but uses less VRAM)

### Long Evaluation Times
- Use `--sample_num` to subsample datasets
- For scalability: reduce `--subset_sizes`
- For sustainability: reduce `--num_steps` or use `--skip_retraining`

### Missing Dependencies
```bash
pip install rouge-score transformers torch pyyaml tqdm numpy
```

---

## Citation

If you use these evaluation scripts, please cite the MUSE paper and RAQUEL:

```bibtex
@article{raquel2024,
  title={RAQUEL: Robust Validation of Machine Unlearning through Query Executions},
  author={...},
  year={2024}
}
```
