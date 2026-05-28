"""
First, we need to generate the instances, if we do not have a ready set of instances.
"""

import logging
import random
import secrets
from collections.abc import Iterator
from typing import Iterable

from _conf import INSTANCE_DB, RQ1_N_SCALING, RQ2_K_SCALING, RQ3_D_SCALING
from _utils import InstanceDB, InstanceName, ManhattanSilhouetteInstance, make_instance

NamedInstance = tuple[str, ManhattanSilhouetteInstance]


def _store_instance_names(
    instance_db: InstanceDB, instance_names: Iterable[InstanceName], seed: int
):
    existing_names = set(instance_db)
    stored = 0
    for name in instance_names:
        instance_name = str(name)
        if instance_name in existing_names:
            continue

        logging.info(f"Creating '{instance_name}'")
        instance = make_instance(
            kind=name.kind, n=name.n, k=name.k, d=name.dim, seed=seed
        )

        logging.info(f"Storing '{instance_name}'")
        instance_db[instance_name] = instance
        stored += 1
    return stored


def _select_instance_names(
    instance_names: Iterable[InstanceName], max_instances: int | None, seed: int
) -> list[InstanceName]:
    selected = sorted(instance_names, key=str)
    rng = random.Random(seed)
    rng.shuffle(selected)

    if max_instances is None:
        return selected

    return selected[:max_instances]


def generate_rq1_instance_names(seed: int) -> Iterator[InstanceName]:
    for d in RQ1_N_SCALING["d_values"]:
        for n in RQ1_N_SCALING["n_values"]:
            for kind in RQ1_N_SCALING["kind_values"]:
                yield InstanceName(
                    rq=RQ1_N_SCALING["rq"],
                    dim=d,
                    n=n,
                    k=RQ1_N_SCALING["k"],
                    kind=kind,
                    seed=seed,
                )


def generate_rq2_instance_names(seed: int) -> Iterator[InstanceName]:
    for k in RQ2_K_SCALING["k_values"]:
        for d in RQ2_K_SCALING["d_values"]:
            for kind in RQ2_K_SCALING["kind_values"]:
                yield InstanceName(
                    rq=RQ2_K_SCALING["rq"],
                    dim=d,
                    n=RQ2_K_SCALING["n"],
                    k=k,
                    kind=kind,
                    seed=seed,
                )


def generate_rq3_instance_names(seed: int) -> Iterator[InstanceName]:
    for d in RQ3_D_SCALING["d_values"]:
        for kind in RQ3_D_SCALING["kind_values"]:
            yield InstanceName(
                rq=RQ3_D_SCALING["rq"],
                dim=d,
                n=RQ3_D_SCALING["n"],
                k=RQ3_D_SCALING["k"],
                kind=kind,
                seed=seed,
            )


def generate_instance_names(seed: int) -> Iterator[InstanceName]:
    yield from generate_rq1_instance_names(seed)
    yield from generate_rq2_instance_names(seed)
    yield from generate_rq3_instance_names(seed)


def generate_instances(
    max_instances: int | None = None,
    seed: int | None = None,
) -> int:
    if seed is None:
        seed = secrets.randbits(128)
        logging.info(f"Generated seed={seed}")

    instance_db = InstanceDB(INSTANCE_DB)
    instance_names = _select_instance_names(
        generate_instance_names(seed), max_instances, seed
    )

    return _store_instance_names(instance_db, instance_names, seed)


if __name__ == "__main__":
    generate_instances()
