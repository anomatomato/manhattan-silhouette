"""One-dimensional Manhattan distance-sum helpers."""

from dataclasses import dataclass

import numpy as np

from ._kernels import _accumulate_sorted_axis_distance_sums
from .types import Array1D, Array2D, ClusterLabels, ClusterSizes


@dataclass(frozen=True)
class SortedAxis:
    values: Array1D[np.float64]
    labels: Array1D[np.intp]
    indices: Array1D[np.intp]


def sort_axis(axis_values: Array1D[np.float64], labels: ClusterLabels) -> SortedAxis:
    n_points = axis_values.shape[0]
    is_sorted = bool(np.all(axis_values[:-1] <= axis_values[1:]))
    if is_sorted:
        order = np.arange(n_points, dtype=np.intp)
    else:
        order = np.argsort(axis_values, kind="stable")

    sorted_values = np.ascontiguousarray(axis_values[order], dtype=np.float64)
    sorted_labels = np.ascontiguousarray(labels[order], dtype=np.intp)
    sorted_indices = np.ascontiguousarray(order, dtype=np.intp)

    return SortedAxis(sorted_values, sorted_labels, sorted_indices)


def has_disjoint_intervals_1d(sorted_axis: SortedAxis) -> bool:
    values = sorted_axis.values
    labels = sorted_axis.labels
    n_points = values.shape[0]
    if n_points <= 1:
        return True

    max_label = int(np.max(labels))
    seen = np.zeros(max_label + 1, dtype=np.bool_)
    prev_label = int(labels[0])
    seen[prev_label] = True

    for i in range(1, n_points):
        current_label = int(labels[i])
        if current_label == prev_label:
            continue

        left_value = values[i - 1]
        right_value = values[i]
        if left_value >= right_value:
            return False

        if seen[current_label]:
            return False

        seen[current_label] = True
        prev_label = current_label

    return True


def accumulate_axis_distance_sums(
    axis_values: Array1D[np.float64],
    labels: ClusterLabels,
    cluster_sizes: ClusterSizes,
    out: Array2D[np.float64],
):
    """Accumulate one-axis point-to-cluster distance sums into `out`"""
    sorted_axis = sort_axis(axis_values, labels)

    _accumulate_sorted_axis_distance_sums(
        sorted_axis.values,
        sorted_axis.labels,
        sorted_axis.indices,
        cluster_sizes,
        out,
    )
