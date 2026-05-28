import numpy as np
from manhattan_silhouette._axis import (
    accumulate_axis_distance_sums,
    has_disjoint_intervals_1d,
    sort_axis,
)

RTOL = 1e-12


class TestAxisSweep:
    def test_matches_manual_distance_sums(self) -> None:
        values = np.array([0.0, 2.0, 1.0, 3.0], dtype=np.float64)
        labels = np.array([0, 1, 0, 1], dtype=np.intp)
        cluster_sizes = np.array([2, 2], dtype=np.intp)
        actual = np.zeros((4, 2), dtype=np.float64)

        accumulate_axis_distance_sums(values, labels, cluster_sizes, actual)

        expected = np.array(
            [
                [1.0, 5.0],
                [3.0, 1.0],
                [1.0, 3.0],
                [5.0, 1.0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(actual, expected, rtol=RTOL, strict=True)


class TestSortAxisAndDisjointIntervals:
    def test_sort_axis_returns_original_indices(self) -> None:
        values = np.array([2.0, 0.0, 1.0], dtype=np.float64)
        labels = np.array([1, 0, 1], dtype=np.intp)

        sorted_axis = sort_axis(values, labels)

        np.testing.assert_allclose(
            sorted_axis.values, np.array([0.0, 1.0, 2.0]), rtol=RTOL, strict=True
        )
        np.testing.assert_array_equal(sorted_axis.labels, np.array([0, 1, 1]))
        np.testing.assert_array_equal(sorted_axis.indices, np.array([1, 2, 0]))

    def test_accepts_strict_contiguous_intervals(self) -> None:
        values = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        labels = np.array([0, 0, 1, 1], dtype=np.intp)

        sorted_axis = sort_axis(values, labels)

        assert has_disjoint_intervals_1d(sorted_axis)

    def test_rejects_interleaved_labels(self) -> None:
        values = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        labels = np.array([0, 1, 0, 1], dtype=np.intp)

        sorted_axis = sort_axis(values, labels)

        assert not has_disjoint_intervals_1d(sorted_axis)

    def test_rejects_equal_coordinate_boundary_across_labels(self) -> None:
        values = np.array([0.0, 1.0, 1.0, 2.0], dtype=np.float64)
        labels = np.array([0, 0, 1, 1], dtype=np.intp)

        sorted_axis = sort_axis(values, labels)

        assert not has_disjoint_intervals_1d(sorted_axis)

    def test_handles_already_sorted_values(self) -> None:
        values = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        labels = np.array([0, 0, 1, 1], dtype=np.intp)
        cluster_sizes = np.array([2, 2], dtype=np.intp)
        actual = np.zeros((4, 2), dtype=np.float64)

        accumulate_axis_distance_sums(values, labels, cluster_sizes, actual)

        expected = np.array(
            [
                [1.0, 5.0],
                [1.0, 3.0],
                [3.0, 1.0],
                [5.0, 1.0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(actual, expected, rtol=RTOL, strict=True)
