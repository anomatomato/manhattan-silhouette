import numpy as np
from sklearn.datasets import make_blobs

from .schema import InstanceKind, ManhattanSilhouetteInstance


def _validate_input(n: int, k: int, d: int):
    if n < 3:
        raise ValueError(f"n must be >= 3, got {n}")
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    if d <= 0:
        raise ValueError(f"d must be positive, got {d}")
    if k > n:
        raise ValueError(
            f"k must be <= n so every cluster can appear, got k={k}, n={n}"
        )


def _sklearn_seed_from_128(seed: int) -> int:
    if not 0 <= seed < 2**128:
        raise ValueError(f"seed must be a 128-bit unsigned integer, got {seed}")

    return int(np.random.SeedSequence(seed).generate_state(1, dtype=np.uint32)[0])


def make_instance(
    *, kind: InstanceKind, n: int, k: int, d: int, seed: int
) -> ManhattanSilhouetteInstance:
    _validate_input(n, k, d)

    match kind:
        case InstanceKind.BLOBS:
            instance = make_blobs_instance(n=n, k=k, d=d, seed=seed)
        case InstanceKind.UNIFORM:
            instance = make_uniform_instance(n=n, k=k, d=d, seed=seed)
        case _:
            raise ValueError(f"Unsupported instance kind: {kind}")

    return instance


def make_blobs_instance(
    *, n: int, k: int, d: int, seed: int
) -> ManhattanSilhouetteInstance:
    legacy_seed = _sklearn_seed_from_128(seed)

    blobs = make_blobs(
        n_samples=n,
        n_features=d,
        centers=k,
        cluster_std=1.0,
        center_box=(-10.0, 10.0),
        random_state=legacy_seed,
    )
    data, labels = blobs[:2]
    return ManhattanSilhouetteInstance(data=data, labels=labels)


def make_uniform_instance(
    *, n: int, k: int, d: int, seed: int
) -> ManhattanSilhouetteInstance:

    rng = np.random.default_rng(seed)
    data = rng.uniform(low=-10, high=10, size=(n, d))

    # Ensure every label 0...k-1 appears at least once
    labels = rng.integers(low=0, high=k - 1, size=n)
    # Ensure every label 0...k-1 appears at least once
    labels[:k] = np.arange(k)
    rng.shuffle(labels)

    return ManhattanSilhouetteInstance(data=data, labels=labels)
