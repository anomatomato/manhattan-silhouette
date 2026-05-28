"""
Having a separate file for the configuration of paths and constants allows
you to, e.g., quickly change the database without having to overwrite the
old data.
"""

from pathlib import Path

from _utils import InstanceKind

# Data that is meant to be shared to verify the results.
PUBLIC_DATA = Path(__file__).parent / "PUBLIC_DATA"
# Data meant for debugging and investigation, not to be shared because of its size.
PRIVATE_DATA = Path(__file__).parent / "PRIVATE_DATA"

# Saving the instances to repeat the experiment on exactly the same data.
INSTANCE_DB = PUBLIC_DATA / "instance_db"
# Saving the full experiment data for potential debugging.
EXPERIMENT_DATA = PRIVATE_DATA / "full_experiment_data"
# Saving the simplified experiment data for analysis.
SIMPLIFIED_RESULTS = PUBLIC_DATA / "simplified_results.json.zip"


N_SCALE = [200_000, 400_000, 600_000, 800_000, 1_000_000]
K_SCALE = [
    2,
    16,
    2000,
    5000,
    10_000,
    20_000,
    30_000,
    40_000,
    50_000,
    60_000,
    70_000,
    80_000,
]
# K_SCALE = [30_000, 40_000, 50_000, 60_000, 70_000, 80_000]
D_SCALE = [1, 2, 50, 100, 150, 200, 250, 300, 350, 400]

# Different research questions
RQ1_N_SCALING = {
    "rq": "rq1_n_scaling",
    "kind_values": [InstanceKind.BLOBS, InstanceKind.UNIFORM],
    "n_values": N_SCALE,
    "k": 5,
    "d_values": [1, 2, 3],
}
RQ2_K_SCALING = {
    "rq": "rq2_k_scaling",
    "kind_values": [InstanceKind.BLOBS, InstanceKind.UNIFORM],
    "n": 100_000,
    "k_values": K_SCALE,
    "d_values": [1, 2, 3],
}
RQ3_D_SCALING = {
    "rq": "rq3_D_scaling",
    "kind_values": [InstanceKind.BLOBS, InstanceKind.UNIFORM],
    "n": 100_000,
    "k": 5,
    "d_values": D_SCALE,
}

# If you write a paper about your study, you want to use
# a uniform width for your plots. Find out, what the optimal
# width is and save it here to be shared for all notebooks.
PLOT_DOC_FULL_WIDTH = 10  # full width plot
PLOT_DOC_HALF_WIDTH = 4.5  # two plots in a row

# For seeds best use a 128-bit random number, e.g. with secrets.randbits(128).
# See more at https://blog.scientific-python.org/numpy/numpy-rng/#random-number-generation-with-numpy
# DEFAULT_NUM_SEEDS = 5

# TIME_LIMIT = 120
