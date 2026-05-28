import argparse
import logging
import secrets
from pathlib import Path

import papermill as pm
import slurminade
from _00_generate_instances import generate_instances
from _01_run_benchmark import compress, run_benchmark
from _03_extract_data import extract_data

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    # format="%(asctime)s|%(levelname)s: %(message)s",
    # datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger(__name__)


@slurminade.slurmify()
def run_analyze_notebook():
    notebook_path = Path(__file__).parent / "_04_analyze.ipynb"
    pm.execute_notebook(notebook_path, notebook_path)


def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description="Run benchmark on gurobi solver.")
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Limit instance generation and benchmark runs to first MAX_INSTANCES instances.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Root seed for deterministic instance generation. If omitted, one is generated.",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip generating instances",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip benchmark execution",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Compress database. Useful when not being able to compress in benchmark run phase.",
    )
    parser.add_argument(
        "--skip-extract", action="store_true", help="Skip data extraction"
    )
    parser.add_argument(
        "--analyze", action="store_true", help="Plot data inside notebook"
    )
    args = parser.parse_args()

    if args.skip_generate and not args.skip_benchmark and args.seed is None:
        msg = "--seed required when benchmarking without instance generation."
        raise ValueError(msg)

    seed = args.seed
    if seed is None:
        seed = secrets.randbits(128)
        logging.info(f"Generating root_seed={seed}")
    else:
        logging.info(f"Using root_seed={seed}")

    # Generate the instances
    if not args.skip_generate:
        generate_instances(
            max_instances=args.max_instances,
            seed=seed,
        )
    # Run the benchmark
    if not args.skip_benchmark:
        run_benchmark(
            max_instances=args.max_instances,
            seed=seed,
        )
    if args.compress:
        compress.distribute()
        slurminade.join()
    # Extract the data into a nice table
    if not args.skip_extract:
        extract_data.distribute()
        slurminade.join()
    # Plot the data by running the notebook
    # jupyter nbconvert --to notebook --execute 04_analyze.ipynb
    if args.analyze:
        run_analyze_notebook.distribute()


if __name__ == "__main__":
    main()
