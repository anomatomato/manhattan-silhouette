from pathlib import Path

import numpy as np
from _conf import InstanceKind
from _utils import InstanceDB
from _utils.instance import make_instance


def test_instance_db_roundtrip_npz(tmp_path: Path):
    db = InstanceDB(tmp_path / "instance_db")
    example_file = "example"

    instance = make_instance(kind=InstanceKind.UNIFORM, n=10, k=3, d=2, seed=123)

    db[example_file] = instance

    loaded = db[example_file]
    np.testing.assert_allclose(loaded.data, instance.data, strict=True)
    np.testing.assert_allclose(loaded.labels, instance.labels, strict=True)
    assert list(db) == [example_file]
