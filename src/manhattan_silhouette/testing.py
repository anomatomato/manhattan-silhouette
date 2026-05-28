"""
This module contains validation functions for manhattan silhouette score functions.
"""

import numpy as np

from .types import ClusterLabels, DataArray, SilhouetteSamples


def assert_silhouette_score_matches_sklearn(
    X: DataArray,
    labels: ClusterLabels,
    score: float,
    rtol: float = 1e-8,
    atol: float = 1e-10,
):
    """
    Assert that the silhouette score is consistent with the standard sklearn implementation.
    """
    from sklearn.metrics import silhouette_score

    expected = silhouette_score(X, labels, metric="manhattan")
    if not np.isclose(score, expected, rtol=rtol, atol=atol):
        msg = f"Silhouette score mismatch: expected {expected}, got {score}."
        raise AssertionError(msg)


def assert_silhouette_samples_match_sklearn(
    X: DataArray,
    labels: ClusterLabels,
    samples: SilhouetteSamples,
    rtol: float = 1e-8,
    atol: float = 1e-10,
):
    """
    Assert that the silhouette samples are consistent with the standard sklearn implementation.
    """
    from sklearn.metrics import silhouette_samples

    actual = samples
    expected = np.asarray(
        silhouette_samples(X, labels, metric="manhattan"), dtype=np.float64
    )
    if actual.shape != expected.shape:
        msg = f"Expected {expected.shape[0]} silhouette samples, got {actual.shape[0]}."
        raise AssertionError(msg)

    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        max_diff = np.max(np.abs(actual - expected))
        msg = (
            f"Silhouette samples mismatch; max abs diff = {max_diff}."
            f"Expected {expected.tolist()}, "
            f"got {actual.tolist()}"
        )
        raise AssertionError(msg)


def assert_label_count_matches_data(
    X: DataArray,
    labels: ClusterLabels,
):
    """
    Assert that every data point is assigned to a cluster in the solution.
    """
    expected = X.shape[0]
    actual = labels.shape[0]
    if actual != expected:
        msg = f"Expected {expected} cluster labels, got {actual}."
        raise AssertionError(msg)


def assert_silhouette_score_in_range(score: float, atol: float = 1e-10):
    if not -1 - atol <= score <= 1 + atol:
        msg = f"Expected score to in range [-1, 1], got {score}."
        raise AssertionError(msg)


def assert_silhouette_samples_in_range(samples: SilhouetteSamples, atol: float = 1e-10):
    if np.any(samples > 1 + atol) or np.any(samples < -1 - atol):
        msg = f"Expected every sample to in range [-1, 1], got {samples.tolist()}."
        raise AssertionError(msg)


def assert_silhouette_score_is_valid(
    X: DataArray,
    labels: ClusterLabels,
    score: float,
    # check_objective: bool = True,
    # check_samples: bool = True,
    rtol: float = 1e-8,
    atol: float = 1e-10,
):
    """
    Run all standard feasibility checks for the computed silhouette score.
    """
    assert_label_count_matches_data(X, labels)
    assert_silhouette_score_in_range(score)
    assert_silhouette_score_matches_sklearn(X, labels, score, rtol=rtol, atol=atol)


def assert_silhouette_samples_are_valid(
    X: DataArray,
    labels: ClusterLabels,
    samples: SilhouetteSamples,
    rtol: float = 1e-8,
    atol: float = 1e-10,
):
    """
    Assert that the silhouette samples are consistent with the standard sklearn implementation.
    """
    assert_label_count_matches_data(X, labels)
    assert_silhouette_samples_in_range(samples)
    assert_silhouette_samples_match_sklearn(X, labels, samples, rtol=rtol, atol=atol)
