import argparse

from _conf import EXPERIMENT_DATA, INSTANCE_DB, SIMPLIFIED_RESULTS
from algbench import Benchmark

benchmark = Benchmark(EXPERIMENT_DATA)


def main():
    # Flags
    parser = argparse.ArgumentParser(description="Clean up benchmark artifacts")
    parser.add_argument(
        "--instances",
        action="store_true",
        help="Remove InstanceDB",
    )
    args = parser.parse_args()

    print("Clear benchmarks...")
    benchmark.clear()

    print("Remove simplified results...")
    SIMPLIFIED_RESULTS.unlink(missing_ok=True)

    if args.instances:
        print("Remove InstanceDB...")
        INSTANCE_DB.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
