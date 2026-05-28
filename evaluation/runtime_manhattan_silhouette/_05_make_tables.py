"""Create LaTeX tables from the simplified benchmark results."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

from _conf import PUBLIC_DATA, SIMPLIFIED_RESULTS
from _utils.schema import ResultKey

TABLE_DIR = PUBLIC_DATA / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY_STANDARD = "sklearn_manhattan"
STRATEGY_CLUSTER_FIRST = "fast_by_cluster_score"


def _load_results() -> pd.DataFrame:
    with zipfile.ZipFile(SIMPLIFIED_RESULTS) as archive:
        with archive.open("simplified_results.json") as file:
            return pd.DataFrame(json.load(file))


def _format_seconds(value: float) -> str:
    if value < 1:
        return f"{value:.3f}"
    if value < 10:
        return f"{value:.2f}"
    return f"{value:.1f}"


def make_rq1_million_point_speedup_table(results: pd.DataFrame) -> pd.DataFrame:
    rq_col = ResultKey.RQ.value
    n_col = ResultKey.N.value
    dim_col = ResultKey.DIM.value
    runtime_col = "runtime"
    strategy_col = "strategy"

    selected = results[
        (results[rq_col] == "rq1_n_scaling")
        & (results[n_col] == 1_000_000)
        & (results[strategy_col].isin([STRATEGY_STANDARD, STRATEGY_CLUSTER_FIRST]))
    ]

    table = (
        selected.groupby([dim_col, strategy_col], as_index=False)[runtime_col]
        .mean()
        .pivot(index=dim_col, columns=strategy_col, values=runtime_col)
        .rename(
            columns={
                STRATEGY_STANDARD: "standard_s",
                STRATEGY_CLUSTER_FIRST: "cluster_first_s",
            }
        )
        .sort_index()
    )
    table["speedup"] = table["standard_s"] / table["cluster_first_s"]
    return table.reset_index()


def _latex_tabular(table: pd.DataFrame) -> str:
    dim_col = ResultKey.DIM.value
    rows = [
        r"\begin{tabular}{rrrr}",
        r"\hline",
        r"\(\D\) & Standard (s) & \fmsClusterFirst{} (s) & Speedup \\",
        r"\hline",
    ]
    for _, row in table.iterrows():
        rows.append(
            f"{int(row[dim_col])} & {_format_seconds(row['standard_s'])} & "
            f"{_format_seconds(row['cluster_first_s'])} & {row['speedup']:,.0f}x \\\\"
        )
    rows.extend([r"\hline", r"\end{tabular}", ""])
    return "\n".join(rows)


def write_rq1_million_point_speedup_table(table: pd.DataFrame) -> Path:
    path = TABLE_DIR / "rq1_million_point_speedup.tex"
    path.write_text(_latex_tabular(table))
    return path


def main() -> None:
    results = _load_results()
    table = make_rq1_million_point_speedup_table(results)
    path = write_rq1_million_point_speedup_table(table)
    print(table.to_string(index=False))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
