# src/manhattan_silhouette/src/manhattan_silhouette/

## Responsibility

Runtime package implementing Manhattan silhouette sample/mean APIs with
validation, specialized 1D acceleration, and Numba-backed kernels.

## Design

- `__init__.py` exports public API from `manhattan.py`.
- `prepare.py` defines validated `PreparedInput` and contiguous label mapping.
- `_axis.py` handles stable axis sorting and strict-disjoint interval checks.
- `_compute_samples.py` provides cluster-wise and axis-wise orchestration.
- `_kernels.py` implements numeric kernels for accumulation and silhouette
  synthesis, including disjoint 1D specialized path.
- `types.py` centralizes ndarray/type aliases; `testing.py` provides test
  assertion helpers.

## Flow

1. API call enters `manhattan.py`.
2. Input validation/remapping runs in `prepare_input`.
3. Optional 1D strict-disjoint optimization executes when valid.
4. General path dispatches to selected orchestration strategy.
5. Kernels produce per-sample silhouettes; score API returns mean.

## Integration

- Runtime contract uses NumPy arrays through all layers.
- Performance-critical loops JIT-compiled with Numba.
- Package tests call runtime modules directly and use `testing.py` helpers.
