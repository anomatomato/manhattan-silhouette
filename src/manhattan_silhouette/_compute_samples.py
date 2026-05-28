import numpy as np

from ._axis import accumulate_axis_distance_sums, sort_axis
from ._kernels import (
    _accumulate_sorted_axis_distance_sums_to_cluster,
    _silhouette_from_a_b,
    _silhouette_from_distance_sums,
    _update_a_b_from_distance_sums_to_cluster,
)
from .prepare import PreparedInput
from .types import SilhouetteSamples


def _compute_samples_by_cluster(prepared: PreparedInput) -> SilhouetteSamples:
    data = prepared.data
    labels = prepared.labels
    cluster_sizes = prepared.cluster_sizes

    n_points, n_features = data.shape
    n_clusters = cluster_sizes.shape[0]

    a = np.zeros(n_points, dtype=np.float64)
    b = np.full(n_points, np.inf, dtype=np.float64)
    sorted_axes = [sort_axis(data[:, axis], labels) for axis in range(n_features)]

    dist_sums = np.empty(n_points, dtype=np.float64)
    for k in range(n_clusters):
        dist_sums.fill(0.0)
        for axis in range(n_features):
            sorted_axis = sorted_axes[axis]
            _accumulate_sorted_axis_distance_sums_to_cluster(
                sorted_axis.values,
                sorted_axis.labels,
                sorted_axis.indices,
                cluster_sizes,
                k,
                dist_sums,
            )

        _update_a_b_from_distance_sums_to_cluster(
            dist_sums, labels, k, cluster_sizes[k], a, b
        )

    return _silhouette_from_a_b(a, b, labels, cluster_sizes)


def _compute_samples_by_axis(prepared: PreparedInput) -> SilhouetteSamples:
    data = prepared.data
    labels = prepared.labels
    cluster_sizes = prepared.cluster_sizes

    n_points, n_features = data.shape
    n_clusters = cluster_sizes.shape[0]

    distance_sums = np.zeros((n_points, n_clusters), dtype=np.float64)
    for axis in range(n_features):
        axis_array = data[:, axis]
        accumulate_axis_distance_sums(axis_array, labels, cluster_sizes, distance_sums)

    return _silhouette_from_distance_sums(distance_sums, labels, cluster_sizes)
