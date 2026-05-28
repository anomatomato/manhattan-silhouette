import numpy as np
from manhattan_silhouette._kernels import (
    _accumulate_sorted_axis_distance_sums,
    _silhouette_from_distance_sums,
)

RTOL = 1e-12


class TestComputeDistanceSums:
    """Tests that verify correct distance sum computation"""

    def test_accumulates_two_clusters_sorted_axis(self) -> None:
        sorted_values = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        sorted_labels = np.array([0, 0, 1, 1], dtype=np.intp)
        sorted_indices = np.array([0, 1, 2, 3], dtype=np.intp)
        cluster_sizes = np.array([2, 2], dtype=np.intp)
        out = np.zeros((4, 2), dtype=np.float64)
        _accumulate_sorted_axis_distance_sums(
            sorted_values,
            sorted_labels,
            sorted_indices,
            cluster_sizes,
            out,
        )
        expected = np.array(
            [
                [1.0, 5.0],
                [1.0, 3.0],
                [3.0, 1.0],
                [5.0, 1.0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(out, expected, rtol=RTOL, strict=True)

    def test_writes_back_to_original_indices(self) -> None:
        sorted_values = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        sorted_labels = np.array([0, 0, 1, 1], dtype=np.intp)
        sorted_indices = np.array([2, 0, 3, 1], dtype=np.intp)
        cluster_sizes = np.array([2, 2], dtype=np.intp)
        out = np.zeros((4, 2), dtype=np.float64)
        _accumulate_sorted_axis_distance_sums(
            sorted_values,
            sorted_labels,
            sorted_indices,
            cluster_sizes,
            out,
        )
        # same rows as sorted test, but placed at original indices [2,0,3,1]
        expected = np.zeros((4, 2), dtype=np.float64)
        expected[2] = [1.0, 5.0]
        expected[0] = [1.0, 3.0]
        expected[3] = [3.0, 1.0]
        expected[1] = [5.0, 1.0]
        np.testing.assert_allclose(out, expected, rtol=RTOL, strict=True)


class TestSilhouetteFromDistanceSum:
    """Tests that verify correct silhouette computation from distance sums"""

    def test_handles_singleton_cluster(self) -> None:
        """Singleton sample has silhouette 0."""
        distance_sums = np.array(
            [
                [0.0, 10.0],
                [10.0, 0.0],
                [20.0, 5.0],
            ],
            dtype=np.float64,
        )
        labels = np.array([0, 1, 1], dtype=np.intp)
        cluster_sizes = np.array([1, 2], dtype=np.intp)

        actual = _silhouette_from_distance_sums(distance_sums, labels, cluster_sizes)

        assert actual[0] == 0.0

    def test_handles_zero_denominator(self) -> None:
        """If a(i) and b(i) are both 0.0, silhouette sample should be 0."""
        distance_sums = np.zeros((3, 2), dtype=np.float64)
        labels = np.array([0, 0, 1], dtype=np.intp)
        cluster_sizes = np.array([2, 1], dtype=np.intp)

        actual = _silhouette_from_distance_sums(distance_sums, labels, cluster_sizes)
        expected = np.zeros(3, dtype=np.float64)

        np.testing.assert_array_equal(actual, expected, strict=True)

    def test_computes_positive_silhouette(self) -> None:
        """b > a gives positive silhouette."""
        distance_sums = np.array(
            [
                [2.0, 10.0],  # a = 2/(2-1)=2, b=10/2=5 => 0.6
                [2.0, 8.0],  # a = 2, b=4 => 0.5
                [10.0, 2.0],
                [8.0, 2.0],
            ],
            dtype=np.float64,
        )
        labels = np.array([0, 0, 1, 1], dtype=np.intp)
        cluster_sizes = np.array([2, 2], dtype=np.intp)
        actual = _silhouette_from_distance_sums(distance_sums, labels, cluster_sizes)
        expected = np.array([0.6, 0.5, 0.6, 0.5], dtype=np.float64)
        np.testing.assert_allclose(actual, expected, rtol=RTOL, strict=True)

    def test_computes_negative_silhouette(self) -> None:
        """b < a gives negative silhouette."""
        distance_sums = np.array(
            [
                [10.0, 2.0],  # a=10, b=1 => -0.9
                [10.0, 4.0],  # a=10, b=2 => -0.8
                [2.0, 10.0],
                [4.0, 10.0],
            ],
            dtype=np.float64,
        )
        labels = np.array([0, 0, 1, 1], dtype=np.intp)
        cluster_sizes = np.array([2, 2], dtype=np.intp)
        actual = _silhouette_from_distance_sums(distance_sums, labels, cluster_sizes)
        expected = np.array([-0.9, -0.8, -0.9, -0.8], dtype=np.float64)
        np.testing.assert_allclose(actual, expected, rtol=RTOL, strict=True)
