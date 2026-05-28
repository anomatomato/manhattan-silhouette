"""
Preparation helpers for Manhattan silhouette computations.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .types import ClusterLabels, ClusterSizes, DataArray


@dataclass(frozen=True)
class PreparedInput:
    """Validated and normalized inputs for Manhattan silhouette kernels."""

    data: DataArray
    labels: ClusterLabels
    cluster_sizes: ClusterSizes


def _get_mapped_labels(labels: np.ndarray) -> ClusterLabels:
    # Indices of sorted unique array to reconstruct input array
    _, inverse = np.unique(labels, return_inverse=True)
    mapped_labels = np.ascontiguousarray(inverse, dtype=np.intp)
    return mapped_labels


def _get_num_clusters(mapped_labels: ClusterLabels) -> int:
    n_clusters = int(max(mapped_labels) + 1) if mapped_labels.size > 0 else 0
    return n_clusters


def _get_cluster_sizes(mapped_labels: ClusterLabels) -> ClusterSizes:
    cluster_sizes = np.ascontiguousarray(np.bincount(mapped_labels))
    return cluster_sizes


def prepare_input(data: ArrayLike, labels: ArrayLike) -> PreparedInput:
    """
    Validate inputs and normalize labels for internal kernels.

    Parameters
    ----------
    data : array-like of shape (n_points, n_features)
        Input feature matrix.
    labels : array-like of shape (n_points,)
        Cluster labels for each point.

    Returns
    -------
    PreparedInput
        Data as ``float64``, labels remapped to ``0..K-1`` as ``intp``, and
        cluster sizes.

    Raises
    ------
    ValueError
        If dimensions are invalid, lengths differ, fewer than 3 points are
        provided, or fewer than 2 clusters are present.
    """
    data_array = np.asarray(data, dtype=np.float64)
    if data_array.ndim != 2:
        msg = "data must be a 2D array of shape (n_points, n_features)."
        raise ValueError(msg)

    n_points = data_array.shape[0]
    if n_points < 3:
        msg = "data must have at least 3 points."
        raise ValueError(msg)

    labels_array = np.asarray(labels)
    if labels_array.ndim != 1:
        msg = "labels must be a 1D array of shape (n_points,)."
        raise ValueError(msg)
    if labels_array.shape[0] != n_points:
        msg = "data and labels must have the same length."
        raise ValueError(msg)

    mapped_labels = _get_mapped_labels(labels_array)

    n_clusters = _get_num_clusters(mapped_labels)
    if n_clusters < 2:
        msg = "silhouette is only defined for at least two clusters."
        raise ValueError(msg)

    cluster_sizes = _get_cluster_sizes(mapped_labels)
    if np.any(cluster_sizes == 0):
        msg = "Every cluster must contain at least one point."
        raise ValueError(msg)

    prepared = PreparedInput(
        data=data_array, labels=mapped_labels, cluster_sizes=cluster_sizes
    )
    return prepared
