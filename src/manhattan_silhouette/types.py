from typing import Any, Literal, TypeAliasType, TypeVar

import numpy as np

Strategy = Literal["auto", "general"]

_SCT = TypeVar("_SCT", bound=np.generic, default=Any)
Array1D = TypeAliasType(
    "Array1D",
    np.ndarray[tuple[int], np.dtype[_SCT]],
    type_params=(_SCT,),
)
Array2D = TypeAliasType(
    "Array2D",
    np.ndarray[tuple[int, int], np.dtype[_SCT]],
    type_params=(_SCT,),
)
DataArray = TypeAliasType("DataArray", Array2D[np.float64])
SilhouetteSamples = TypeAliasType("SilhouetteSamples", Array1D[np.float64])
ClusterLabels = TypeAliasType("ClusterLabels", Array1D[np.intp])
ClusterSizes = TypeAliasType("ClusterSizes", Array1D[np.intp])
