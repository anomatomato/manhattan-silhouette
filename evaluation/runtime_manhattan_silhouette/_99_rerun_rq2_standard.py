import slurminade
from algbench import Benchmark
from _conf import EXPERIMENT_DATA, INSTANCE_DB
from _03_extract_data import extract_data
from _01_run_benchmark import compress, warm_numba
from _utils import (
    STRATEGIES,
    InstanceDB,
    InstanceName,
    ResultKey,
    ManhattanSilhouetteInstance,
)

instancedb = InstanceDB(INSTANCE_DB)
benchmark = Benchmark(EXPERIMENT_DATA)

TARGET_K = {60_000, 70_000}
TARGET_RQ = "rq2_k_scaling"
TARGET_STRATEGY = "sklearn_manhattan"


def compute_silhouette(
    instance_name: str,
    strategy: str,
    _instance: ManhattanSilhouetteInstance,
):
    silhouette_alg = STRATEGIES[strategy]
    asw = silhouette_alg(_instance)
    parsed_name = InstanceName.parse(instance_name=instance_name)
    return {
        ResultKey.RQ: parsed_name.rq,
        ResultKey.N: parsed_name.n,
        ResultKey.DIM: parsed_name.dim,
        ResultKey.K: parsed_name.k,
        ResultKey.KIND: parsed_name.kind.value,
        ResultKey.ASW: asw,
    }


def is_target(instance_name: str) -> bool:
    parsed = InstanceName.parse(instance_name)
    return parsed.rq == TARGET_RQ and parsed.k in TARGET_K


@slurminade.slurmify()
def rerun_one(instance_name: str) -> None:
    instance = instancedb[instance_name]
    # Important: run(), not add(), because add() skips existing entries.
    benchmark.run(
        compute_silhouette,
        instance_name,
        TARGET_STRATEGY,
        instance,
    )


def main() -> None:
    instance_names = sorted(name for name in instancedb if is_target(name))
    print(f"Rerunning {len(instance_names)} instances:")
    for name in instance_names:
        print(name)
    warm_numba()
    with slurminade.JobBundling(max_size=8):
        for instance_name in instance_names:
            rerun_one.distribute(instance_name)
    slurminade.join()
    compress.distribute()
    slurminade.join()
    extract_data.distribute()
    slurminade.join()


if __name__ == "__main__":
    main()
