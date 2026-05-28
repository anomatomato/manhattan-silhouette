import numpy as np
import pytest
from manhattan_silhouette.prepare import PreparedInput, prepare_input
from manhattan_silhouette.types import ClusterLabels, DataArray


@pytest.fixture(scope="module")
def prepared_4pts_2clusters() -> PreparedInput:
    """Tiny instance verification."""
    data = np.array([[0], [1], [2], [3]], dtype=np.float64)
    labels = np.array([10, 20, 10, 20], dtype=np.intp)

    prepared = prepare_input(data, labels)
    return prepared


def test_prepare_input_maps_non_contiguous_labels(
    prepared_4pts_2clusters: PreparedInput,
) -> None:
    actual = prepared_4pts_2clusters.labels
    desired = np.array([0, 1, 0, 1], dtype=np.intp)

    np.testing.assert_array_equal(actual, desired, strict=True)


def test_prepare_input_calculates_cluster_sizes(
    prepared_4pts_2clusters: PreparedInput,
) -> None:
    """Prepared input calculates correct cluster sizes."""
    actual = prepared_4pts_2clusters.cluster_sizes
    desired = np.array([2, 2], dtype=np.intp)

    np.testing.assert_array_equal(actual, desired, strict=True)


@pytest.mark.parametrize(
    ("data", "labels", "match"),
    [
        (np.zeros((3)), np.array([0, 1, 1]), "2D"),  # 2D data
        (np.zeros((2, 1, 1)), np.array([0, 1, 1]), "2D"),  # 2D data
        (np.zeros((2, 1)), np.array([0, 1]), "at least 3 points"),
        (np.zeros((3, 1)), np.array([0, 1, 1, 2]), "same length"),  # not same length
        (
            np.zeros((3, 1)),
            np.array([0, 0, 0]),
            "at least two clusters",
        ),
        # (np.zeros((3, 1)), np.array([0, 1, 2]), "at most n_points - 1"),
    ],
)
def test_prepare_input_rejects_invalid_inputs(
    data: DataArray, labels: ClusterLabels, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        prepare_input(data, labels)
