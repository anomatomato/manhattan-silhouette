# evaluation/research_questions/2026-05-04_runtime_manhattan_silhouette/

## Responsibility

Benchmark pipeline for runtime/scaling comparison of Manhattan silhouette
implementations (custom fast kernels vs scikit-learn baseline) over `n`, `k`,
and `d` research-question grids.

## Design

- Config-centric setup in `_conf.py` (`RQ1_N_SCALING`, `RQ2_K_SCALING`,
  `RQ3_D_SCALING`, paths).
- Script pipeline: `_00_generate_instances.py` → `_01_run_benchmark.py` →
  `_03_extract_data.py`, orchestrated by `run_all.py`.
- Separate artifact cleanup in `clean.py`.
- Script-style sibling imports (`_conf`, `_utils`) because dated folder is not
  importable package name.

## Flow

1. `_00_generate_instances.py` builds `InstanceName` combinations for 3 RQs,
   shuffles deterministically by seed, materializes data via
   `_utils.make_instance`, stores `.npz` records in `PUBLIC_DATA/instance_db`.
2. `_01_run_benchmark.py` loads instances, warms numba kernels once, runs each
   strategy from `_utils.STRATEGIES`, and stores per-run results in AlgBench.
3. `_03_extract_data.py` filters valid strategy entries, deduplicates by
   `(instance_name, strategy)`, writes `simplified_results.json.zip`.
4. `run_all.py` provides CLI toggles for phase skipping, compression, and
   notebook execution.

## Integration

- Uses `manhattan_silhouette.silhouette_score_manhattan` and
  `sklearn.metrics.silhouette_score` via strategy registry.
- Uses `algbench.Benchmark`/`read_as_pandas` for run persistence and extraction.
- Uses `slurminade` for optional distributed execution.
- Utilities documented in [`_utils/codemap.md`](_utils/codemap.md).
