# evaluation/research_questions/2026-05-04_runtime_manhattan_silhouette/\_utils/

## Responsibility

Utility layer for Manhattan-runtime benchmark: instance schema/naming, synthetic
instance generation, file-based instance persistence, and silhouette strategy
registry.

## Design

- Facade export in `__init__.py` (`InstanceDB`, `InstanceName`, strategy map,
  factory functions).
- Pydantic+numpydantic schema models in `schema.py` with string-serializable
  `InstanceName` and `ResultKey` enums.
- Storage abstraction `InstanceDB` in `instance_db.py` writing compressed `.npz`
  files, with cached reads.
- Strategy registry in `strategies.py` mapping names to callables for fair
  benchmark loops.

## Flow

1. Generation scripts request `InstanceName` objects and call `make_instance()`.
2. `instance.py` dispatches by `InstanceKind` to blob or uniform builders,
   validates input, and normalizes sklearn RNG seed handling.
3. `InstanceDB` persists and reloads `{name}.npz` as
   `ManhattanSilhouetteInstance`.
4. Benchmark scripts iterate `STRATEGIES` and store returned ASW values keyed by
   `ResultKey` fields.

## Integration

- Used by top-level scripts `_00_generate_instances.py`, `_01_run_benchmark.py`,
  `_03_extract_data.py`.
- Integrates with `manhattan_silhouette` and scikit-learn silhouette APIs.
- Integrates with extraction pipeline through shared `ResultKey` names.
