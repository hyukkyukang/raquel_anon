Legacy compatibility paths kept in this repo:

- `src.generator.SchemaGenerator`
  Replacement: `src.generator.DynamicSchemaGenerator`
- `src.aligned_db.nullify.UpdateNullExecutor`
  Replacement: `src.aligned_db.nullified_db.NullifiedDBBuilder`
- `src.dataset.providers` and `src.dataset.unlearning_utils`
  Replacement: `src.training.data` and `src.training.data.pl_module`
- `script/evaluation/eval.py`
  Replacement: `script/evaluation/run_raquel_eval.py`, `script/evaluation/run_muse_eval.py`, and `src.evaluation`

These paths are preserved only for backwards compatibility. New code should not
depend on them.
