import pandas as pd
import slurminade
from _conf import EXPERIMENT_DATA, SIMPLIFIED_RESULTS
from _utils import STRATEGIES, ResultKey
from algbench import read_as_pandas


def is_valid(entry: dict):
    return entry["parameters"]["args"]["strategy"] in STRATEGIES


def row_creator(entry: dict):
    if not is_valid(entry):  # Skip invalid entries
        return None
    result = entry["result"]
    args = entry["parameters"]["args"]
    return {
        "instance_name": args["instance_name"],
        ResultKey.RQ: result[ResultKey.RQ],
        ResultKey.N: result[ResultKey.N],
        ResultKey.K: result[ResultKey.K],
        ResultKey.DIM: result[ResultKey.DIM],
        ResultKey.KIND: result[ResultKey.KIND],
        "runtime": entry["runtime"],  # automatically saved
        "timestamp": entry["timestamp"],
        "strategy": args["strategy"],
        ResultKey.ASW: result[ResultKey.ASW],
    }


@slurminade.slurmify()
def extract_data():
    """
    Extract simple pandas table from benchmark data
    """
    t = read_as_pandas(EXPERIMENT_DATA, row_creator)

    dedup_keys = ["instance_name", "strategy"]

    # Drop old versions
    t["timestamp"] = pd.to_datetime(t["timestamp"])
    t.sort_values("timestamp", inplace=True)
    # print(t[t.duplicated(subset=dedup_keys, keep=False)])
    t.drop_duplicates(subset=dedup_keys, keep="last", inplace=True)

    t.to_json(SIMPLIFIED_RESULTS)


if __name__ == "__main__":
    extract_data()
