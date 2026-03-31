# RAQUEL: Robust Validation of Machine Unlearning through Query Executions over an Aligned Database

**RAQUEL** is a framework that evaluates and enforces machine unlearning by aligning model knowledge with a structured database. It ensures robust forgetting by validating unlearning across semantically equivalent natural language and SQL queries, addressing paraphrase leakage and incompleteness in conventional benchmarks.

---

[![Docker Image Version (tag)](https://img.shields.io/docker/v/hyukkyukang/raquel/latest)](https://img.shields.io/docker/v/hyukkyukang/raquel/latest)
[![license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/hyukkykuang/raquel/blob/main/LICENSE)

## 🚀 Getting Started

### 0. Docker Environment

We provide pre-built Docker images for various environments on [Docker Hub](https://hub.docker.com/repository/docker/hyukkyukang/raquel/tags):

- `py3.13-cuda-12.8.1-cudnn-devel-ubuntu24.04`

### 1. Launch Docker Container

To quickly start a container using Docker Compose:

```bash
# Set environment variables to match the host user
export UID=$(id -u)
export GID=$(id -g)

# Launch container
docker compose up -d
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

```bash
export HF_TOKEN=hf_...
export OPENAI_API_KEY=...

export HF_HOME=/home/user/RAQUEL/data/huggingface
# LiteLLM cache directory (DSPy 2.x uses litellm for LLM calls)
# Cache is stored in ~/.litellm/cache by default, or configure in config/llm/default.yaml
```

The default LLM profile now uses OpenAI across the pipeline. The configured
LiteLLM model is defined in `config/llm/default.yaml`. The legacy
`llm=gemini_flash_lite` profile name is kept only as a compatibility alias and
resolves to the same default model.

# Creating the Dataset

**Note**: All scripts now use config-based file paths defined in `config/paths/default.yaml`. This ensures consistency across the workflow and makes it easy to modify file locations if needed.

## 1. Construct Aligned Database
```bash
python script/stages/construct_aligned_db.py model.aligned_db.sample_num=20
```

Default aligned-build text handling is intentionally scoped:
- Stage 1-3 discovery/schema generation use canonical QA text.
- Stage 4 extraction uses deterministic normalized QA text.
- LLM-backed QA naturalization stays disabled by default because it is still experimental and can reduce structural recall.

## 2. Create a dump of the aligned database
```bash
PGPASSWORD=postgres pg_dump -h localhost -p 5432 -U postgres -d tofu_data --data-only --inserts > /home/user/RAQUEL/data/aligned_db/raw_tofu_data_inserts.sql
python script/stages/clean_insert_statements.py
```

## 3. Copy the aligned database to the null database
```bash
# Export PGPASSWORD so psql doesn’t prompt
export PGPASSWORD=postgres

# Drop and create a new database
PGPASSWORD=postgres psql -h localhost -p 5433 -U postgres -d postgres -c "DROP DATABASE IF EXISTS tofu_data_null;"
PGPASSWORD=postgres psql -h localhost -p 5433 -U postgres -d postgres -c "CREATE DATABASE tofu_data_null;"

# Apply schema and insert data
PGPASSWORD=postgres psql -h localhost -p 5433 -U postgres -d tofu_data_null -f /home/user/RAQUEL/data/aligned_db/schema.sql
PGPASSWORD=postgres psql -h localhost -p 5433 -U postgres -d tofu_data_null -f /home/user/RAQUEL/data/aligned_db/tofu_data_inserts.sql
```


## 4. Generate and apply nullify upsert statements (from the forget QA pairs)
```bash
python script/stages/update_null.py
```

## 5. Synthesize SQL Queries
```bash
python script/stages/synthesize_query.py
```

## 6. Translate SQL Queries to Natural Language
```bash
python script/stages/translate_query.py
```

## 7. Execute SQL Queries
```bash
python script/stages/execute_query.py
```

## 8. Paraphrase the data
```bash
python script/stages/paraphrase_data.py
```

## Fine-tuning and Unlearning
`model.device_map=auto` is safe for Lightning-managed single-GPU runs in this repo: it is normalized to `cuda:0` so Hugging Face does not shard the model across all visible GPUs. Multi-GPU runs still keep Hugging Face auto-placement.
External experiment tracking now uses MLflow. By default, runs are logged to a local SQLite-backed MLflow store (`./mlflow.db`) with artifacts under `./mlartifacts`; set `MLFLOW_TRACKING_URI` or `mlflow.tracking_uri=...` to use a remote tracking server, or disable external tracking with `tracking.enabled=false`.

### 1. Fine-tuning the base model
```bash
python script/train/finetune_retain.py --config-name finetune/retain_llama_3b
```

### 2. Full fine-tuning
```bash
python script/train/finetune_full.py --config-name finetune/full_llama_3b
```

### 3. Unlearning
```bash
python script/train/unlearn.py --config-name unlearn/ga model.path=/path/to/finetuned/model
```

### 4. Public TinyLlama QLoRA validation path
Recommended end-to-end local validation:
```bash
python script/validation/run_tinyllama_qlora_validation.py
```

This writes bounded dataset slices, fine-tune and unlearn artifacts, RAQUEL and MUSE outputs, a local SQLite-backed MLflow store, and a compact `validation_summary.json` under `results/local_validation/...`.

Manual equivalent:
```bash
python script/train/finetune_retain.py \
  --config-name finetune/retain_tinyllama_qlora \
  tracking.enabled=false

python script/train/unlearn.py \
  --config-name unlearn/ga_tinyllama_qlora \
  tracking.enabled=false

python script/evaluation/run_raquel_eval.py \
  --model_path model/finetune_retain-tinyllama-qlora/finetune/TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --affected_file data/aligned_db/affected_synthesized_queries_results.json \
  --unaffected_file data/aligned_db/unaffected_synthesized_queries_results.json \
  --quantize_4bit

python script/evaluation/run_muse_eval.py \
  --model_path model/unlearn_ga-tinyllama-qlora/unlearn/TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --quantize_4bit
```

Legacy compatibility note:
`script/evaluation/eval.py`, `src.generator.SchemaGenerator`, and
`src.aligned_db.nullify` are kept only for backwards compatibility. New code
should use the dedicated evaluation runners, `DynamicSchemaGenerator`, and
`NullifiedDBBuilder`.
