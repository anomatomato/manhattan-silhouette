import numpy as np
from hypothesis import example, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from manhattan_silhouette.testing import (
    assert_silhouette_samples_in_range,
    assert_silhouette_samples_match_sklearn,
)

from manhattan_silhouette import silhouette_samples_manhattan

RTOL = 1e-8


@st.composite
def valid_clustered_arrays(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray]:
    n_samples = draw(st.integers(min_value=3, max_value=10))
    n_features = draw(st.integers(min_value=1, max_value=4))
    n_clusters = draw(st.integers(min_value=2, max_value=n_samples - 1))

    data = draw(
        hnp.arrays(
            dtype=np.float64,
            shape=(n_samples, n_features),
            elements=st.floats(
                min_value=-20, max_value=20, allow_infinity=False, allow_nan=False
            ),
        )
    )

    labels = np.array(
        list(range(n_clusters))
        + draw(
            st.lists(
                st.integers(min_value=0, max_value=n_clusters - 1),
                min_size=n_samples - n_clusters,
                max_size=n_samples - n_clusters,
            )
        ),
        dtype=np.intp,
    )
    permutations = np.array(draw(st.permutations(range(n_samples))), dtype=np.intp)

    return data, labels[permutations]


class TestProperties:
    @example(
        case=(
            np.array(
                [
                    [1.7e01],
                    [1.0e-12],
                    [1.7e01],
                    [1.7e01],
                    [1.7e01],
                    [1.7e01],
                    [1.7e01],
                    [1.7e01],
                ]
            ),
            np.array([0, 1, 2, 0, 0, 0, 0, 0]),
        )
    ).via("regression test")
    @example(
        case=(
            np.array(
                [
                    [-1.0],
                    [0.0],
                    [0.0],
                    [3.44891914e-178],
                    [3.44891914e-178],
                ]
            ),
            np.array([1, 0, 2, 0, 0]),
        )
    ).via("regression, discovered silhouette samples with 1.5")
    @given(valid_clustered_arrays())
    @settings(deadline=None, max_examples=100)
    def test_samples_match_sklearn_for_random_small_input(
        self, case: tuple[np.ndarray, np.ndarray]
    ):
        X, labels = case

        silhouette_samples = silhouette_samples_manhattan(X, labels)

        assert_silhouette_samples_match_sklearn(
            X, labels, silhouette_samples, rtol=RTOL
        )

    @given(valid_clustered_arrays())
    @settings(deadline=None, max_examples=100)
    def test_check_1d_disjoint_option_matches_default(
        self, case: tuple[np.ndarray, np.ndarray]
    ):
        X, labels = case

        with_disjoint_check = silhouette_samples_manhattan(
            X, labels, check_1d_disjoint=True
        )
        default_samples = silhouette_samples_manhattan(
            X, labels, check_1d_disjoint=False
        )

        np.testing.assert_allclose(with_disjoint_check, default_samples, rtol=RTOL)

    @example(
        case=(np.array([[-1.0], [0.001], [0.001], [0.0]]), np.array([0, 1, 0, 0]))
    ).via("discovered third sample is below -1.0")
    @given(valid_clustered_arrays())
    @settings(deadline=None, max_examples=100)
    def test_samples_are_bounded(self, case: tuple[np.ndarray, np.ndarray]):
        X, labels = case

        samples = silhouette_samples_manhattan(
            X,
            labels,
        )

        assert_silhouette_samples_in_range(samples)
