import functools
from pathlib import Path

import numpy as np

from .schema import ManhattanSilhouetteInstance


class InstanceDB:
    """
    Simple helper to store and load the instances.

    Store each instance numpy arrays as compressed .npz file
    """

    FILE_SUFFIX = ".npz"

    def __init__(self, path: Path):
        self.path = path

    @functools.lru_cache(10)
    def __getitem__(self, name: str):
        instance_path = self._instance_path(name)
        with np.load(instance_path) as loaded:
            data = loaded["data"]
            labels = loaded["labels"]
        return ManhattanSilhouetteInstance(data=data, labels=labels)

    def __setitem__(self, name: str, instance: ManhattanSilhouetteInstance):
        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)
        self.__getitem__.cache_clear()

        target = self._instance_path(name)
        tmp = target.with_suffix(".tmp.npz")

        np.savez_compressed(tmp, data=instance.data, labels=instance.labels)
        tmp.replace(target)

    def __iter__(self):
        if not self.path.exists():
            return

        for path in self.path.glob(f"*{self.FILE_SUFFIX}"):
            yield path.stem

    def _instance_path(self, name: str) -> Path:
        instance_path = self.path / f"{name}"
        return instance_path.with_suffix(self.FILE_SUFFIX)
