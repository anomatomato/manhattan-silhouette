import numpy as np

from ._axis import has_disjoint_intervals_1d, sort_axis
from ._compute_samples import _compute_samples_by_axis, _compute_samples_by_cluster
from ._kernels import (
    _silhouette_disjoint_1d_sorted,
)
from .prepare import prepare_input
from .types import SilhouetteSamples

# from sklearn.metrics import silhouette_score, silhouette_samples


def silhouette_samples_manhattan(
    X: np.ArrayLike,
    labels: np.ArrayLike,
    check_1d_disjoint: bool = False,
    compute_by_cluster: bool = True,
) -> SilhouetteSamples:
    """
    Compute per-sample silhouette values using Manhattan (L1) distances.

    Parameters
    ----------

    X : array-like of shape (n_samples, n_features)
        Input feature matrix.

    labels : array-like of shape (n_samples,)
        Cluster labels. Labels are remapped internally to contiguous integer
        IDs while preserving membership.

    check_1d_disjoint : bool, default=False
        If ``True`` and ``X`` has one feature, use a specialized kernel when
        cluster intervals are disjoint on that axis.

    compute_by_cluster : bool, default=True
        If ``True``, compute using cluster-oriented kernels. Otherwise use
        axis-oriented kernels.

    Returns
    -------

    SilhouetteSamples
        Per-sample silhouette values in ``[-1, 1]``.
    """
    prepared = prepare_input(X, labels)
    n_points, n_features = prepared.data.shape

    if check_1d_disjoint:
        if n_features == 1:
            sorted_axis = sort_axis(prepared.data[:, 0], prepared.labels)
            if has_disjoint_intervals_1d(sorted_axis):
                return _silhouette_disjoint_1d_sorted(
                    sorted_axis.values,
                    sorted_axis.labels,
                    sorted_axis.indices,
                    prepared.cluster_sizes,
                )

    if compute_by_cluster:
        return _compute_samples_by_cluster(prepared)
    else:
        return _compute_samples_by_axis(prepared)


def silhouette_score_manhattan(
    X: np.ArrayLike,
    labels: np.ArrayLike,
    check_1d_disjoint: bool = False,
    compute_by_cluster: bool = True,
) -> float:
    """
    Compute mean silhouette score using Manhattan (L1) distances.

    Parameters
    ----------

    X : array-like of shape (n_samples, n_features)
        Input feature matrix.

    labels : array-like of shape (n_samples,)
        Cluster labels.

    check_1d_disjoint : bool, default=False
        Forwarded to :func:`silhouette_samples_manhattan`.

    compute_by_cluster : bool, default=True
        Forwarded to :func:`silhouette_samples_manhattan`.

    Returns
    -------

    float
        Mean of per-sample silhouette values.
    """
    return float(
        np.mean(
            silhouette_samples_manhattan(
                X,
                labels,
                check_1d_disjoint=check_1d_disjoint,
                compute_by_cluster=compute_by_cluster,
            )
        )
    )
