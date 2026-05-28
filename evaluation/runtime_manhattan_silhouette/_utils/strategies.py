from typing import Callable

from sklearn.metrics import silhouette_score

from manhattan_silhouette import silhouette_score_manhattan

from .schema import ManhattanSilhouetteInstance

ScoreFunc = Callable[[ManhattanSilhouetteInstance], float]


def fast_by_cluster_score(instance: ManhattanSilhouetteInstance) -> float:
    return silhouette_score_manhattan(
        instance.data, instance.labels, compute_by_cluster=True
    )


def fast_by_axis_score(instance: ManhattanSilhouetteInstance) -> float:
    return silhouette_score_manhattan(
        instance.data, instance.labels, compute_by_cluster=False
    )


def sklearn_score(instance: ManhattanSilhouetteInstance) -> float:
    return float(silhouette_score(instance.data, instance.labels, metric="manhattan"))


STRATEGIES: dict[str, ScoreFunc] = {
    fast_by_cluster_score.__name__: fast_by_cluster_score,
    fast_by_axis_score.__name__: fast_by_axis_score,
    "sklearn_manhattan": sklearn_score,
}
