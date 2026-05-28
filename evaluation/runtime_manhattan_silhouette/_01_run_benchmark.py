"""
This file runs the actual benchmark on the instances.

Slurminade: This script uses the slurminade package to distribute the benchmark on a cluster. If you do not have a slurm-cluster, it will run the benchmark locally.
AlgBench: This script uses the AlgBench package to capture and manage the results
"""

import random

import numpy as np
import slurminade  # pip install slurminade
from _conf import (
    EXPERIMENT_DATA,
    INSTANCE_DB,
)
from _utils import (
    STRATEGIES,
    InstanceDB,
    InstanceName,
    ManhattanSilhouetteInstance,
    ResultKey,
)

# for saving the results easily
from algbench import Benchmark  # pip install algbench

from manhattan_silhouette import silhouette_score_manhattan

instancedb = InstanceDB(INSTANCE_DB)
benchmark = Benchmark(EXPERIMENT_DATA)

# Arguments for simple slurm. For a complete list of sbatch arguments see https://slurm.schedmd.com/sbatch.html
slurminade.update_default_configuration(
    partition="alg",
    # constraint="alggen03",  # algry[01-04], 120.000 MEM, 16 CPUs
    constraint="alggen05",  # algra[01-06], 80.000 MEM, 24 CPUs
    exclusive=True,  # To use all cores on a node exclusively
    cpus_per_task=24,
    mail_user="hoang@ibr.cs.tu-bs.de",
    mail_type="FAIL",  # NONE, BEGIN, END, FAIL, ALL
)


@slurminade.slurmify()  # makes the function distributable on a slurm cluster
def load_instance_and_compute_silhouette(instance_name: str):
    instance = instancedb[instance_name]

    # logger_name = "Evaluation"
    # logger = logging.getLogger(logger_name)
    # benchmark.capture_logger(logger_name, logging.INFO)

    # Function creating an entry in the database
    def compute_silhouette(
        # Arguments without _ in front used for identifying entries, should be JSON-compatible
        instance_name: str,
        strategy: str,
        # Arguments starting with _ are not saved in the experiment data.
        _instance: ManhattanSilhouetteInstance,
    ):

        silhouette_alg = STRATEGIES[strategy]
        asw = silhouette_alg(_instance)

        parsed_name = InstanceName.parse(instance_name=instance_name)

        return {  # the returned values are saved to the database
            ResultKey.RQ: parsed_name.rq,
            ResultKey.N: parsed_name.n,
            ResultKey.DIM: parsed_name.dim,
            ResultKey.K: parsed_name.k,
            ResultKey.KIND: parsed_name.kind.value,
            ResultKey.ASW: asw,
        }

    # Will only run if the last instance is not already solved
    for strategy in STRATEGIES:
        benchmark.add(
            compute_silhouette,
            instance_name,
            strategy,
            instance,
        )


# --------------------------
# Compression is not thread-safe so we make it a separate function
# if you only notify about failures, you may want to do
# ``@slurminade.slurmify(mail_type="ALL)`` to be notified after completion.
@slurminade.slurmify()
def compress():
    benchmark.compress()


def _select_instance_names(
    instance_names: list[str],
    seed: int | None,
    max_instances: int | None,
) -> list[str]:
    selected = sorted(
        instance_name
        for instance_name in instance_names
        # if _is_selected_instance_name(instance_name, seed)
        # if "rq2" in instance_name
    )

    rng = random.Random(seed)
    rng.shuffle(selected)

    if max_instances is None:
        return selected

    if seed is None:
        msg = "seed must be provided when max_instances is set."
        raise ValueError(msg)

    return selected[:max_instances]


def warm_numba() -> None:
    """Compile numba kernels, so benchmark doesn't slow down in the beginning"""
    data = np.array([[0.0], [1.0], [10.0], [11.0]], dtype=np.float64)
    labels = np.array([0, 0, 1, 1], dtype=np.intp)
    silhouette_score_manhattan(data, labels, compute_by_cluster=True)
    silhouette_score_manhattan(data, labels, compute_by_cluster=False)


# --------------------------
# Run the benchmark on all instances.
def run_benchmark(
    seed: int | None = None,
    max_instances: int | None = None,
):
    instance_names = _select_instance_names(
        list(instancedb),
        seed=seed,
        max_instances=max_instances,
    )

    # Compile numba kernels
    warm_numba()

    # Distribute the benchmark on a cluster
    with slurminade.JobBundling(
        max_size=8
    ) as _:  # automatically bundles up to 20 tasks
        for instance_name in instance_names:
            load_instance_and_compute_silhouette.distribute(instance_name)
    slurminade.join()  # make sure to wait until every job has finished
    # Compress the results at the end
    compress.distribute()
    slurminade.join()  # make sure to wait until compression finished


if __name__ == "__main__":
    run_benchmark()
