"""
Private numba kernels
"""

import numpy as np
from numba import njit

from .types import Array1D, Array2D, ClusterLabels, ClusterSizes, SilhouetteSamples


@njit(cache=True)
def _accumulate_sorted_axis_distance_sums_to_cluster(
    sorted_values: Array1D[np.float64],
    sorted_labels: ClusterLabels,
    sorted_indices: Array1D[np.intp],
    cluster_sizes: ClusterSizes,
    target_cluster: int,
    out: Array1D[np.float64],
) -> None:
    X = sorted_values
    L = sorted_labels
    n_points = X.shape[0]
    cluster_size = cluster_sizes[target_cluster]

    # 1. Cluster values: every point in cluster target_cluster sorted
    cluster_values = np.empty(cluster_size, dtype=np.float64)
    count = 0
    for i in range(n_points):
        if L[i] == target_cluster:
            cluster_values[count] = X[i]
            count += 1

    # 2. Prefix_offsets: Diff of every point x in target_cluster to most left point in target_cluster: x - first_point
    first_cluster_point = cluster_values[0]

    prefix_offsets = np.empty(cluster_size, dtype=np.float64)
    current_offset_sum = 0.0
    for k in range(cluster_size):
        offset = cluster_values[k] - first_cluster_point
        current_offset_sum += offset
        prefix_offsets[k] = current_offset_sum

    left_count = 0

    # Loop O(n), calc distances of every point to cluster k
    # distance_sum = left_sum + right_sum
    # left_sum = (left_count * x_offset) - left_offset_sum
    # left_offset_sum = prefix_offsets[left_count - 1] if left_count > 0
    # right_sum = right_offset_sum - (right_count * x_offset)
    # right_offset_sum = total_offset_sum - left_offset_sum
    for i in range(n_points):
        x = X[i]
        x_offset = x - first_cluster_point
        right_count = cluster_size - left_count

        left_offset_sum = 0 if left_count == 0 else prefix_offsets[left_count - 1]
        total_offset_sum = prefix_offsets[cluster_size - 1]
        right_offset_sum = total_offset_sum - left_offset_sum

        left_sum = (left_count * x_offset) - left_offset_sum
        right_sum = right_offset_sum - (right_count * x_offset)
        dist_sum = left_sum + right_sum

        original_idx = sorted_indices[i]
        out[original_idx] += dist_sum

        # Increase left counter, if current point in target_cluster
        if L[i] == target_cluster:
            left_count += 1


@njit(cache=True)
def _update_a_b_from_distance_sums_to_cluster(
    distance_sums: Array1D[np.float64],
    labels: ClusterLabels,
    target_cluster: int,
    target_cluster_size: int,
    a: Array1D[np.float64],
    b: Array1D[np.float64],
) -> None:
    """Compute silhouette samples from point-to-cluster distance sums."""
    n_points = distance_sums.shape[0]

    for i in range(n_points):
        label = labels[i]

        if label == target_cluster and target_cluster_size > 1:
            a[i] = distance_sums[i] / (target_cluster_size - 1)
        elif label != target_cluster:
            candidate = distance_sums[i] / target_cluster_size
            if candidate < b[i]:
                b[i] = candidate


@njit(cache=True)
def _silhouette_from_a_b(
    a: Array1D[np.float64],
    b: Array1D[np.float64],
    labels: ClusterLabels,
    cluster_sizes: ClusterSizes,
) -> SilhouetteSamples:
    """Compute silhouette samples from point-to-cluster distance sums."""
    n_points = a.shape[0]
    samples = np.empty(n_points, dtype=np.float64)

    for i in range(n_points):
        label = labels[i]
        cluster_size = cluster_sizes[label]

        if cluster_size == 1:
            # Singleton cluster
            samples[i] = 0.0
            continue

        max_a_b = max(a[i], b[i])
        if max_a_b == 0.0:
            # a(i) and b(i) are both zero
            samples[i] = 0.0
            continue

        # Standard silhouette formula: s_i = (b_i - a_i) / max{a_i, b_i}
        samples[i] = (b[i] - a[i]) / max_a_b

    return samples


@njit(cache=True)
def _accumulate_sorted_axis_distance_sums(
    sorted_values: Array1D[np.float64],
    sorted_labels: ClusterLabels,
    sorted_indices: Array1D[np.intp],
    cluster_sizes: ClusterSizes,
    out: Array2D[np.float64],
) -> None:
    """Accumulate 1D point-to-cluster distance sums for one sorted axis.
    For each point x_i and each cluster C_k, this computes
            sum_{y in C_k} |x_i - y|
        and adds it to ``out[original_index_i, k]``.

    Parameters
    ---
    out : ndarray
        Output array in which to place the result.
        Is mutated in place because each feature axis contributes additively to the final Manhattan distance sums.
    """
    X = sorted_values
    L = sorted_labels
    n_points = X.shape[0]
    n_clusters = cluster_sizes.shape[0]

    # ------------------------------------------------------------------
    # 1. Group sorted values by cluster.
    #
    # sorted_values is already sorted globally. If we scan it from left to
    # right and append each value to its cluster, then every cluster row is
    # also sorted.
    #
    # Example:
    # sorted_values:  -1, 0, 0, tiny, tiny
    # sorted_labels:   1, 0, 2,    0,    0
    #
    # cluster_values[0, :3] = 0, tiny, tiny
    # cluster_values[1, :1] = -1
    # cluster_values[2, :1] = 0
    #
    # Packed storage:
    # cluster_values: [0, tiny, tiny, -1, 0]
    # cluster_starts: [0, 3, 4, 5]
    #
    # Meaning:
    # cluster k lives in:
    # cluster_values[cluster_starts[k] : cluster_starts[k + 1]]
    # ------------------------------------------------------------------

    cluster_starts = np.empty(n_clusters + 1, dtype=np.intp)

    cluster_starts[0] = 0
    for k in range(n_clusters):
        cluster_starts[k + 1] = cluster_starts[k] + cluster_sizes[k]

    cluster_values = np.empty(n_points, dtype=np.float64)
    write_pos = cluster_starts.copy()

    for i in range(n_points):
        label = L[i]
        pos = write_pos[label]
        cluster_values[pos] = X[i]
        write_pos[label] += 1

    # ------------------------------------------------------------------
    # 2. Build stable prefix sums per cluster.
    #
    # For each cluster k:
    #
    # values:  v0, v1, v2, ...
    # min_value:     v0
    # offsets: 0,  v1-v0, v2-v0, ...
    #
    # prefix_offsets[k, j] stores sum of first j offsets.
    #
    # Why offsets?
    # If values are 1e16, 1e16 + small, raw sums lose "small".
    # Offsets keep distances near the cluster scale.
    # ------------------------------------------------------------------
    # cluster_min_value = np.empty(n_clusters, dtype=np.float64)
    prefix_offsets = np.empty(n_points, dtype=np.float64)

    for k in range(n_clusters):
        start = cluster_starts[k]
        size = cluster_sizes[k]
        min_value = cluster_values[start]

        current_offset_sum = 0.0
        for j in range(size):
            offset = cluster_values[start + j] - min_value
            current_offset_sum += offset
            prefix_offsets[start + j] = current_offset_sum

    # ------------------------------------------------------------------
    # 3. Sweep target points from left to right.
    #
    # For fixed cluster k and target x, split cluster values into:
    #
    # left side:  y <= x
    # right side: y > x
    #
    # Then:
    #
    # sum |x - y|
    # = sum_left  (x - y)
    # + sum_right (y - x)
    #
    # With offsets from cluster_min_value:
    #
    # x_offset = x - min_value(k)
    # y_offset = y - min_value(k)
    #
    # left distance sum:
    #   sum_left (x - y)
    # = x_offset * left_count - left_offset_sum
    #
    # right distance sum:
    #   sum_right (y - x)
    # = right_offset_sum - x_offset * right_count
    #
    # Total:
    #   dist_sum =
    #       x_offset * left_count
    #       - left_offset_sum
    #       + right_offset_sum
    #       - x_offset * right_count
    #
    # left_counts[k] is a moving pointer. Since target x only increases,
    # each pointer only moves forward.
    # ------------------------------------------------------------------

    # Number of points of cluster k already passed by the sweep.
    left_counts = np.zeros(n_clusters, dtype=np.intp)

    for i in range(n_points):
        x = X[i]
        original_idx = sorted_indices[i]
        for k in range(n_clusters):
            start = cluster_starts[k]
            min_value = cluster_values[start]
            size = cluster_sizes[k]
            left_count = left_counts[k]
            right_count = size - left_count

            x_offset = x - min_value

            left_offset_sum = (
                0 if left_count == 0 else prefix_offsets[start + left_count - 1]
            )

            total_offset_sum = prefix_offsets[start + size - 1]
            right_offset_sum = total_offset_sum - left_offset_sum

            # Distance sum to points in cluster k, that are left to x
            left_dist_sum = x_offset * left_count - left_offset_sum
            # Distance sum to points in cluster k, that are right to x
            right_dist_sum = right_offset_sum - x_offset * right_count

            dist_sum = left_dist_sum + right_dist_sum
            out[original_idx, k] += dist_sum

        current_label = L[i]
        left_counts[current_label] += 1


@njit(cache=True)
def _silhouette_from_distance_sums(
    distance_sums: Array2D[np.float64],
    labels: ClusterLabels,
    cluster_sizes: ClusterSizes,
) -> SilhouetteSamples:
    """Compute silhouette samples from point-to-cluster distance sums."""
    n_points, n_clusters = distance_sums.shape
    samples = np.empty(n_points, dtype=np.float64)

    for i in range(n_points):
        current_label = labels[i]
        current_cluster_size = cluster_sizes[current_label]

        if current_cluster_size == 1:
            # Singleton cluster
            samples[i] = 0.0
            continue

        a_i = distance_sums[i, current_label] / (current_cluster_size - 1)
        b_i = np.inf
        for k in range(n_clusters):
            if current_label != k:
                candidate = distance_sums[i, k] / cluster_sizes[k]
                if candidate < b_i:
                    b_i = candidate

        max_a_b = max(a_i, b_i)
        if max_a_b == 0.0:
            # a(i) and b(i) are both zero
            samples[i] = 0.0
            continue

        # Standard silhouette formula: s_i = (b_i - a_i) / max{a_i, b_i}
        s_i = (b_i - a_i) / max_a_b
        samples[i] = s_i

    return samples


@njit(cache=True)
def _silhouette_disjoint_1d_sorted(
    sorted_values: Array1D[np.float64],
    sorted_labels: ClusterLabels,
    sorted_indices: Array1D[np.intp],
    cluster_sizes: ClusterSizes,
) -> SilhouetteSamples:
    n_points = sorted_values.shape[0]
    n_clusters = cluster_sizes.shape[0]
    samples = np.empty(n_points, dtype=np.float64)

    # Strict-disjoint precondition means every cluster label appears as exactly
    # one contiguous block in sorted order, for example:
    # values:  0  1  2    10 11    20 21
    # labels:  0  0  0     1  1     2  2
    # block_label_order stores the physical left-to-right block order.
    block_start = np.empty(n_clusters, dtype=np.intp)
    block_end = np.empty(n_clusters, dtype=np.intp)
    block_ref = np.zeros(n_clusters, dtype=np.float64)
    block_offset_sum = np.zeros(n_clusters, dtype=np.float64)
    block_label_order = np.empty(n_clusters, dtype=np.intp)

    for k in range(n_clusters):
        block_start[k] = -1
        block_end[k] = -1

    prev_label = sorted_labels[0]
    block_count = 1
    block_label_order[0] = prev_label
    block_start[prev_label] = 0
    block_ref[prev_label] = sorted_values[0]

    for i in range(1, n_points):
        current_label = sorted_labels[i]
        if current_label != prev_label:
            block_end[prev_label] = i - 1
            block_label_order[block_count] = current_label
            block_count += 1
            block_start[current_label] = i
            block_ref[current_label] = sorted_values[i]
            prev_label = current_label
        # Offset from the first coordinate in this block. The first point adds
        # zero. These stable offset sums let us compute neighbor means without
        # subtracting nearly equal raw coordinate sums.
        block_offset_sum[current_label] += sorted_values[i] - block_ref[current_label]
    block_end[prev_label] = n_points - 1

    for block_idx in range(block_count):
        label = block_label_order[block_idx]
        size = cluster_sizes[label]
        start = block_start[label]
        end = block_end[label]

        left_label = -1
        right_label = -1
        if block_idx > 0:
            left_label = block_label_order[block_idx - 1]
        if block_idx + 1 < block_count:
            right_label = block_label_order[block_idx + 1]

        left_mean_offset = 0.0
        left_ref = 0.0
        if left_label != -1:
            left_ref = block_ref[left_label]
            left_mean_offset = block_offset_sum[left_label] / cluster_sizes[left_label]

        right_mean_offset = 0.0
        right_ref = 0.0
        if right_label != -1:
            right_ref = block_ref[right_label]
            right_mean_offset = (
                block_offset_sum[right_label] / cluster_sizes[right_label]
            )

        # Initial own-cluster distance sum for the first point in the block.
        # All other own-cluster points are to its right, so this is exactly the
        # sum of offsets from the block reference.
        intra_sum = block_offset_sum[label]

        for i in range(start, end + 1):
            x = sorted_values[i]
            original_idx = sorted_indices[i]

            if size == 1:
                samples[original_idx] = 0.0
                continue

            # Recurrence for a(i). Move from previous point to current point.
            # Points left of current point become farther by delta; current and
            # right-side points become closer by delta. Thus:
            # A_i = A_{i-1} + delta * (left_count - right_count).
            if i > start:
                delta = x - sorted_values[i - 1]
                position_in_block = i - start
                left_size = position_in_block
                right_size = size - position_in_block
                intra_sum += delta * (left_size - right_size)

            a_i = intra_sum / (size - 1)

            # For strict disjoint intervals, the nearest foreign cluster must
            # be an adjacent block. Since all left-neighbor points are <= x,
            # mean distance to left neighbor is x - mean(left). Since all
            # right-neighbor points are >= x, it is mean(right) - x. Means are
            # represented as ref + mean_offset for numerical stability.
            b_i = np.inf
            if left_label != -1:
                left_candidate = (x - left_ref) - left_mean_offset
                if left_candidate < b_i:
                    b_i = left_candidate
            if right_label != -1:
                right_candidate = (right_ref - x) + right_mean_offset
                if right_candidate < b_i:
                    b_i = right_candidate

            max_a_b = max(a_i, b_i)
            if max_a_b == 0.0:
                samples[original_idx] = 0.0
            else:
                samples[original_idx] = (b_i - a_i) / max_a_b

    return samples
