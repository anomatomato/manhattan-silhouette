import numpy as np
from manhattan_silhouette.testing import (
    assert_silhouette_samples_match_sklearn,
    assert_silhouette_score_matches_sklearn,
)

from manhattan_silhouette import (
    silhouette_samples_manhattan,
    silhouette_score_manhattan,
)


def assert_public_api_matches_sklearn(
    data: np.ndarray,
    labels: np.ndarray,
    check_1d_disjoint: bool = False,
    compute_by_cluster: bool = True,
) -> None:
    samples = silhouette_samples_manhattan(
        data,
        labels,
        check_1d_disjoint=check_1d_disjoint,
        compute_by_cluster=compute_by_cluster,
    )
    score = silhouette_score_manhattan(
        data,
        labels,
        check_1d_disjoint=check_1d_disjoint,
        compute_by_cluster=compute_by_cluster,
    )

    assert_silhouette_samples_match_sklearn(data, labels, samples)
    assert_silhouette_score_matches_sklearn(data, labels, score)


class TestSilhouetteSamplesManhattan:
    def test_interleaved_1d_clusters(self) -> None:
        data = np.array([[0.0], [10.0], [1.0], [11.0]])
        labels = np.array([0, 1, 0, 1])

        assert_public_api_matches_sklearn(data, labels)

    def test_multidimensional_data(self) -> None:
        data = np.array(
            [
                [0.0, 0.0],
                [1.0, -1.0],
                [10.0, 10.0],
                [11.0, 9.0],
                [20.0, 20.0],
            ]
        )
        labels = np.array([0, 0, 1, 1, 2])

        assert_public_api_matches_sklearn(data, labels)

    def test_equal_coordinates_across_clusters(self) -> None:
        data = np.array([[0.0], [0.0], [1.0], [2.0]])
        labels = np.array([0, 1, 0, 1])

        assert_public_api_matches_sklearn(data, labels)


class TestComputationOptions:
    def test_compute_modes_match_sklearn_with_large_coordinate_offset(self) -> None:
        data = np.array(
            [[1e16], [1e16 + 2.0], [1e16 + 100.0], [1e16 + 104.0]],
            dtype=np.float64,
        )
        labels = np.array([0, 0, 1, 1], dtype=np.intp)

        by_cluster = silhouette_samples_manhattan(data, labels, compute_by_cluster=True)
        by_axis = silhouette_samples_manhattan(data, labels, compute_by_cluster=False)

        assert_silhouette_samples_match_sklearn(data, labels, by_cluster)
        assert_silhouette_samples_match_sklearn(data, labels, by_axis)

    def test_compute_by_axis_works_for_interleaved_1d(self) -> None:
        data = np.array([[0.0], [10.0], [1.0], [11.0]])
        labels = np.array([0, 1, 0, 1])

        samples = silhouette_samples_manhattan(data, labels, compute_by_cluster=False)
        score = silhouette_score_manhattan(data, labels, compute_by_cluster=False)

        assert_silhouette_samples_match_sklearn(data, labels, samples)
        assert_silhouette_score_matches_sklearn(data, labels, score)

    def test_check_1d_disjoint_true_uses_disjoint_1d_fast_lane(self) -> None:
        data = np.array([[0.0], [1.0], [10.0], [11.0], [20.0], [21.0]])
        labels = np.array([0, 0, 1, 1, 2, 2])

        assert_public_api_matches_sklearn(data, labels, check_1d_disjoint=True)

    def test_check_1d_disjoint_true_falls_back_for_non_disjoint_1d(self) -> None:
        data = np.array([[0.0], [10.0], [1.0], [11.0]])
        labels = np.array([0, 1, 0, 1])

        assert_public_api_matches_sklearn(data, labels, check_1d_disjoint=True)

    def test_check_1d_disjoint_true_falls_back_for_equal_boundary_1d(self) -> None:
        data = np.array([[0.0], [1.0], [1.0], [2.0]])
        labels = np.array([0, 0, 1, 1])

        assert_public_api_matches_sklearn(data, labels, check_1d_disjoint=True)

    def test_check_1d_disjoint_true_falls_back_for_multi_dimensional_data(self) -> None:
        data = np.array([[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]])
        labels = np.array([0, 0, 1, 1])

        assert_public_api_matches_sklearn(data, labels, check_1d_disjoint=True)

    def test_score_and_samples_forward_computation_options(self) -> None:
        data = np.array([[0.0], [2.0], [10.0], [11.0], [12.0], [20.0]])
        labels = np.array([0, 0, 1, 1, 1, 2])

        assert_public_api_matches_sklearn(
            data,
            labels,
            check_1d_disjoint=True,
            compute_by_cluster=False,
        )

    def test_check_1d_disjoint_true_disjoint_unequal_cluster_sizes(self) -> None:
        data = np.array([[0.0], [2.0], [10.0], [11.0], [12.0], [20.0]])
        labels = np.array([0, 0, 1, 1, 1, 2])

        assert_public_api_matches_sklearn(data, labels, check_1d_disjoint=True)
