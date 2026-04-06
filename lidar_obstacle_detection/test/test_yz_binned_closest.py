"""Unit tests for YZ-binned forward depth / closest surface helper."""

from __future__ import annotations

import numpy as np

from lidar_obstacle_detection.surface_obstacle_segmentation import yz_binned_closest_and_smooth_x


def test_closest_ignores_isolated_outlier_when_bin_underfilled() -> None:
    """Single-point bin does not set robust X; closest comes from dense wall bins."""
    rng = np.random.default_rng(42)
    n = 80
    wall = np.zeros((n, 3), dtype=np.float64)
    wall[:, 0] = 5.0 + rng.normal(0.0, 0.02, size=n)
    wall[:, 1] = rng.uniform(0.0, 1.0, size=n)
    wall[:, 2] = rng.uniform(0.0, 1.0, size=n)
    outlier = np.array([[0.15, 5.0, 5.0]], dtype=np.float64)
    cluster = np.vstack([wall, outlier])

    closest, smooth = yz_binned_closest_and_smooth_x(
        cluster,
        bin_size_m=0.2,
        min_points=3,
        percentile=10.0,
    )
    assert np.all(np.isfinite(closest))
    assert closest[0] > 3.0, 'closest X should reflect the wall, not the lone outlier bin'
    assert smooth.shape[0] == cluster.shape[0]


def test_binning_disabled_uses_raw_minimum_x() -> None:
    pts = np.array(
        [
            [2.0, 0.0, 0.0],
            [5.0, 0.1, 0.0],
            [5.0, 0.2, 0.0],
        ],
        dtype=np.float64,
    )
    closest, smooth = yz_binned_closest_and_smooth_x(
        pts,
        bin_size_m=0.0,
        min_points=3,
        percentile=8.0,
    )
    assert float(closest[0]) == 2.0
    np.testing.assert_array_almost_equal(smooth, pts[:, 0])
