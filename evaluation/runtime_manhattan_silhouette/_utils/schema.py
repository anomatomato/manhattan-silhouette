from enum import StrEnum, auto
from typing import Annotated as A
from typing import ClassVar, Self

import numpy as np
from numpydantic import NDArraySchema, Shape
from numpydantic.vendor.nptyping import Float, Int
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# Result dictionary keys for saved results from running benchmark
class ResultKey(StrEnum):
    RQ = auto()
    N = auto()
    K = auto()
    DIM = auto()
    KIND = auto()
    ASW = auto()


class InstanceKind(StrEnum):
    BLOBS = auto()
    UNIFORM = auto()


class ManhattanSilhouetteInstance(BaseModel):
    model_config = ConfigDict(
        extra="forbid", arbitrary_types_allowed=True, strict=True, frozen=True
    )

    data: A[
        np.ndarray,
        NDArraySchema(Shape("3-*, *"), Float),
    ] = Field(..., description="The points to cluster. Minimum of 3 points.")
    labels: A[np.ndarray, NDArraySchema(Shape("3-*")), Int] = Field(
        ..., description="List of assigned cluster labels for every data point."
    )

    @field_validator("data", mode="before")
    @classmethod
    def data_to_ndarray(cls, v: np.ndarray):
        return np.asarray(v, dtype=np.float64)

    @model_validator(mode="after")
    def validate_data_and_labels_len_match(self) -> Self:
        if len(self.data) != len(self.labels):
            msg = "Data and labels require same length."
            raise ValueError(msg)
        return self


class InstanceName(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rq: str = Field(..., description="Research question")
    dim: int = Field(..., description="Dimension of data")
    n: int = Field(..., description="Number of data points")
    k: int = Field(..., description="Number of clusters")
    kind: InstanceKind = Field(..., description="Instance variant")
    seed: int | None = Field(default=None, description="Root seed used")
    # replicate: int | None = Field(default=None, description="Replicated index")

    PART_DELIMITER: ClassVar[str] = "__"
    KEY_VALUE_DELIMITER: ClassVar[str] = "="

    def __str__(self) -> str:
        data = self.model_dump(exclude_none=True)
        return self.PART_DELIMITER.join(
            f"{key}{self.KEY_VALUE_DELIMITER}{value}" for key, value in data.items()
        )

    @classmethod
    def parse(cls, instance_name: str) -> Self:
        raw: dict[str, str] = {}
        for part in instance_name.split(cls.PART_DELIMITER):
            key, value = part.split(cls.KEY_VALUE_DELIMITER, maxsplit=1)
            raw[key] = value
        return cls.model_validate(raw)
