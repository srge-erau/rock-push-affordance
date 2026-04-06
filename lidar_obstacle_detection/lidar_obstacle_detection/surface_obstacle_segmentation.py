"""
Library: surface normal estimation, dominant-plane cosine split, DBSCAN clusters.

Implements surface-normal cosine split and DBSCAN clustering without ROS coupling:
normals are oriented to point along ``up_axis`` (e.g. +Z in ``base_link``), ground
vs obstacle is decided by |cos(n, n_ref)| with ``n_ref`` from the median of oriented
normals or a fixed ``reference_normal_xyz``. Obstacle points are clustered with
Open3D DBSCAN.

Also computes, per cluster, a **robust surface normal** from **ground** points whose
(X,Y) lies inside the cluster AABB footprint. The box can be scaled about its center
(``obstacle_bbox_scale``: 1.0 = tight cluster AABB, 1.5 = 50% larger). Ground normals
in the footprint are **binned** on the sphere (angular quantization), each bin is
summarized by a unit **mean** direction, then the **component-wise median** of those
bin representatives is taken and renormalized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import open3d as o3d
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header

# Point labels: 0 = dominant surface (ground), -1 = obstacle noise (no cluster), >=1 = cluster id
LABEL_GROUND = 0
LABEL_OBSTACLE_NOISE = -1


@dataclass
class SurfaceObstacleParams:
    """Tunable perception hyperparameters (mirror YAML / node declarations)."""

    cosine_threshold: float = 0.88
    normal_search_radius: float = 0.5
    max_nn: int = 30
    dbscan_eps: float = 0.15
    dbscan_min_points: int = 15
    obstacle_bbox_scale: float = 1.0
    surface_normal_bin_deg: float = 10.0
    up_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    reference_normal_xyz: tuple[float, float, float] | None = None
    yz_depth_bin_size_m: float = 0.15
    yz_depth_min_points: int = 3
    yz_depth_percentile: float = 8.0
    segmented_cloud_depth_shading: bool = False


@dataclass
class ObstacleDetection:
    """One DBSCAN obstacle cluster plus derived geometry and surface normal."""

    cluster_id: int
    min_bound: np.ndarray
    max_bound: np.ndarray
    center: np.ndarray
    median_surface_normal: np.ndarray
    normal_anchor: np.ndarray
    closest_surface_point: np.ndarray


@dataclass
class SegmentationResult:
    """Outputs of :func:`segment_surface_obstacles`."""

    points: np.ndarray
    normals: np.ndarray
    labels: np.ndarray
    obstacles: list[ObstacleDetection]
    forward_x_smoothed: np.ndarray | None = None


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return (v / n).astype(np.float64)


def orient_normals_upward(
    normals: np.ndarray,
    up: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """
    Flip normals so each has non-negative dot product with ``up`` (e.g. +Z).

    Parameters
    ----------
    normals
        (N, 3) array.
    up
        3-vector; need not be unit length.

    Returns
    -------
    numpy.ndarray
        Same shape as ``normals``, dtype float64.

    """
    up_v = _unit(np.asarray(up, dtype=np.float64).reshape(3))
    out = np.asarray(normals, dtype=np.float64).copy()
    dots = out @ up_v
    out[dots < 0.0] *= -1.0
    return out


def yz_binned_closest_and_smooth_x(
    cluster_xyz: np.ndarray,
    *,
    bin_size_m: float,
    min_points: int,
    percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    YZ-bin robust forward X per point and a representative closest-surface point.

    In ``base_link``-style frames (+X forward), smaller X is closer. Each non-empty
    (Y, Z) column uses a low percentile of X when the bin has at least
    ``min_points`` samples; otherwise that point keeps its raw X. The returned
    closest point is chosen from the bin with the smallest robust X among qualifying
    bins (point whose X is nearest that robust value). If binning is disabled
    (``bin_size_m <= 0``) or no bin qualifies, falls back to the raw minimum-X point.

    Returns
    -------
    closest_point : ndarray
        Shape (3,), float64. NaN if ``cluster_xyz`` is empty.
    smooth_x_per_point : ndarray
        Shape (M,), float64, same row order as ``cluster_xyz``.

    """
    xyz = np.asarray(cluster_xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError('cluster_xyz must be (M, 3)')
    m = xyz.shape[0]
    nan3 = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    if m == 0:
        return nan3, np.zeros((0,), dtype=np.float64)

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    pct = float(np.clip(percentile, 0.0, 100.0))
    mp = int(max(1, min_points))

    if float(bin_size_m) <= 0.0:
        imin = int(np.argmin(x))
        closest = xyz[imin].copy()
        return closest, x.copy()

    y_min, y_max = float(np.min(y)), float(np.max(y))
    z_min, z_max = float(np.min(z)), float(np.max(z))
    bs = float(bin_size_m)
    eps = 1e-12
    span_y = y_max - y_min
    span_z = z_max - z_min
    ny = max(1, int(np.ceil((span_y + eps) / bs)))
    nz = max(1, int(np.ceil((span_z + eps) / bs)))

    if span_y < eps:
        iy = np.zeros(m, dtype=np.int64)
    else:
        iy = np.floor((y - y_min) / bs).astype(np.int64)
        iy = np.clip(iy, 0, ny - 1)

    if span_z < eps:
        iz = np.zeros(m, dtype=np.int64)
    else:
        iz = np.floor((z - z_min) / bs).astype(np.int64)
        iz = np.clip(iz, 0, nz - 1)

    bin_key = iy * nz + iz
    unique_keys, inverse, counts = np.unique(bin_key, return_inverse=True, return_counts=True)
    n_bins = int(unique_keys.shape[0])
    smooth_x = x.copy()
    robust_fill = np.full(n_bins, np.inf, dtype=np.float64)

    for i in range(n_bins):
        mask_i = inverse == i
        ni = int(np.count_nonzero(mask_i))
        if ni >= mp:
            rx = float(np.percentile(x[mask_i], pct))
            smooth_x[mask_i] = rx
            robust_fill[i] = rx

    if not np.any(np.isfinite(robust_fill) & (robust_fill < np.inf)):
        imin = int(np.argmin(x))
        return xyz[imin].copy(), smooth_x

    best_i = int(np.argmin(robust_fill))
    best_rx = float(robust_fill[best_i])
    mask_b = inverse == best_i
    sub_x = x[mask_b]
    sub_xyz = xyz[mask_b]
    j_loc = int(np.argmin(np.abs(sub_x - best_rx)))
    closest = sub_xyz[j_loc].copy()
    return closest, smooth_x


def _label_colors(
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """RGB + alpha per point from integer labels."""
    n = labels.shape[0]
    r = np.zeros(n, dtype=np.uint32)
    g = np.zeros(n, dtype=np.uint32)
    b = np.zeros(n, dtype=np.uint32)
    a = np.full(n, 255, dtype=np.uint32)

    is_ground = labels == LABEL_GROUND
    r[is_ground] = 80
    g[is_ground] = 80
    b[is_ground] = 255

    is_noise = labels == LABEL_OBSTACLE_NOISE
    r[is_noise] = 120
    g[is_noise] = 120
    b[is_noise] = 120

    pal = np.array(
        [
            (0, 255, 0),
            (0, 128, 255),
            (128, 255, 0),
            (255, 255, 0),
            (0, 255, 255),
            (0, 0, 255),
            (255, 0, 255),
            (255, 128, 0),
            (128, 0, 255),
        ],
        dtype=np.uint32,
    )
    obj_mask = (labels >= 1) & ~is_ground
    if np.any(obj_mask):
        idx = (labels[obj_mask].astype(np.int64) - 1) % pal.shape[0]
        r[obj_mask] = pal[idx, 0]
        g[obj_mask] = pal[idx, 1]
        b[obj_mask] = pal[idx, 2]

    return r, g, b, a


def _apply_forward_x_shading(
    labels: np.ndarray,
    forward_x_smoothed: np.ndarray,
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    *,
    shade_min: float = 0.35,
    shade_max: float = 1.0,
) -> None:
    """
    Modulate obstacle cluster RGB by per-point smoothed forward X (in-place).

    Per cluster, normalized depth uses finite smoothed X only; smaller X (closer)
    maps to higher brightness. Ground and noise labels are unchanged.
    """
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    xs = np.asarray(forward_x_smoothed, dtype=np.float64).reshape(-1)
    if xs.shape[0] != labels.shape[0]:
        raise ValueError('forward_x_smoothed length must match labels')
    eps = 1e-6
    smin = float(shade_min)
    smax = float(shade_max)
    for cid in np.unique(labels):
        if cid < 1:
            continue
        m = labels == cid
        if not np.any(m):
            continue
        xv = xs[m]
        finite = np.isfinite(xv)
        if not np.any(finite):
            continue
        x_lo = float(np.min(xv[finite]))
        x_hi = float(np.max(xv[finite]))
        denom = x_hi - x_lo + eps
        t = (xv - x_lo) / denom
        t = np.clip(t, 0.0, 1.0)
        mult = smax - (smax - smin) * t
        mult[~finite] = 1.0
        r[m] = np.clip(r[m].astype(np.float64) * mult, 0.0, 255.0).astype(np.uint32)
        g[m] = np.clip(g[m].astype(np.float64) * mult, 0.0, 255.0).astype(np.uint32)
        b[m] = np.clip(b[m].astype(np.float64) * mult, 0.0, 255.0).astype(np.uint32)


def build_xyz_rgba_pointcloud2(
    header: Header,
    xyz: np.ndarray,
    labels: np.ndarray,
    *,
    forward_x_smoothed: np.ndarray | None = None,
    depth_shading: bool = False,
) -> PointCloud2:
    """Build ``PointCloud2`` with float32 xyz + uint32 rgba (cluster colors)."""
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError('xyz must be (N, 3)')
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    if labels.shape[0] != xyz.shape[0]:
        raise ValueError('labels length must match number of points')

    rr, gg, bb, aa = _label_colors(labels)
    if depth_shading:
        if forward_x_smoothed is None:
            raise ValueError('forward_x_smoothed is required when depth_shading is True')
        _apply_forward_x_shading(labels, forward_x_smoothed, rr, gg, bb)
    rgba = rr | (gg << 8) | (bb << 16) | (aa << 24)
    rgba = rgba.astype(np.uint32)

    dt = np.dtype(
        {
            'names': ['x', 'y', 'z', 'rgba'],
            'formats': [np.float32, np.float32, np.float32, np.uint32],
            'offsets': [0, 4, 8, 12],
            'itemsize': 16,
        },
    )
    structured = np.zeros(xyz.shape[0], dtype=dt)
    structured['x'] = xyz[:, 0]
    structured['y'] = xyz[:, 1]
    structured['z'] = xyz[:, 2]
    structured['rgba'] = rgba

    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgba', offset=12, datatype=PointField.UINT32, count=1),
    ]
    return pc2.create_cloud(header, fields, structured)


def _footprint_xy_from_aabb(
    bbox: o3d.geometry.AxisAlignedBoundingBox,
) -> tuple[float, float, float, float]:
    corners = np.asarray(bbox.get_box_points())
    min_x = float(np.min(corners[:, 0]))
    max_x = float(np.max(corners[:, 0]))
    min_y = float(np.min(corners[:, 1]))
    max_y = float(np.max(corners[:, 1]))
    return min_x, max_x, min_y, max_y


def _binned_median_unit_normals(
    unit_normals: np.ndarray,
    bin_deg: float,
) -> np.ndarray:
    """
    Quantize similar unit normals into spherical bins, mean per bin, then median.

    Bin keys use rounded inclination from +Z and azimuth (``atan2``), both in radians
    quantized by ``bin_deg``. Each non-empty bin contributes one unit **mean**
    direction; the final vector is **component-wise median** of those representatives,
    then L2-normalized. If ``bin_deg`` <= 0, skips binning and uses the median of
    raw normals (per axis, then normalize).
    """
    sel = np.asarray(unit_normals, dtype=np.float64)
    if sel.shape[0] == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    # renormalize rows
    norms = np.linalg.norm(sel, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    sel = sel / norms

    if float(bin_deg) <= 0.0:
        med = np.median(sel, axis=0)
        nm = float(np.linalg.norm(med))
        if nm < 1e-12:
            return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        return med / nm

    d = np.deg2rad(max(float(bin_deg), 1e-6))
    z = np.clip(sel[:, 2], -1.0, 1.0)
    theta = np.arccos(z)
    phi = np.arctan2(sel[:, 1], sel[:, 0])
    tb = np.round(theta / d).astype(np.int64)
    pb = np.round(phi / d).astype(np.int64)
    keys = np.stack([tb, pb], axis=1)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    n_bin = int(inv.max()) + 1 if inv.size else 0
    reps: list[np.ndarray] = []
    for b in range(n_bin):
        m = inv == b
        if not np.any(m):
            continue
        chunk = sel[m]
        v = np.mean(chunk, axis=0)
        nv = float(np.linalg.norm(v))
        if nv < 1e-12:
            continue
        reps.append(v / nv)
    if not reps:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    stack = np.stack(reps, axis=0)
    med = np.median(stack, axis=0)
    nm = float(np.linalg.norm(med))
    if nm < 1e-12:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    return med / nm


def _median_ground_normal_in_footprint(
    xyz: np.ndarray,
    normals: np.ndarray,
    ground_mask: np.ndarray,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    surface_normal_bin_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Binned-median oriented normal and mean position of ground points in XY rectangle.

    Returns (normal_unit, anchor_xyz). If no points inside, normal is NaN and anchor
    is the rectangle center at z = median ground z (fallback).
    """
    gx = xyz[ground_mask, 0]
    gy = xyz[ground_mask, 1]
    gz = xyz[ground_mask, 2]
    gn = normals[ground_mask]
    inside = (gx >= min_x) & (gx <= max_x) & (gy >= min_y) & (gy <= max_y)
    if np.count_nonzero(inside) < 1:
        nan = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        z0 = float(np.median(xyz[ground_mask, 2])) if np.any(ground_mask) else 0.0
        anchor = np.array([(min_x + max_x) * 0.5, (min_y + max_y) * 0.5, z0], dtype=np.float64)
        return nan, anchor

    sel = gn[inside]
    pos = np.stack([gx[inside], gy[inside], gz[inside]], axis=1)
    med_n = _binned_median_unit_normals(sel, surface_normal_bin_deg)
    anchor = np.mean(pos, axis=0)
    return med_n, anchor


def segment_surface_obstacles(
    xyz: np.ndarray,
    params: SurfaceObstacleParams,
) -> SegmentationResult:
    """
    Run normal estimation → orient → cosine split → DBSCAN on obstacle points.

    Parameters
    ----------
    xyz
        (N, 3) point coordinates in meters. Frame should match ``params.up_axis``
        (e.g. ``base_link`` with +Z up).
    params
        Search radii, cosine threshold, DBSCAN settings, optional fixed reference
        normal, AABB scale for footprint sampling, and spherical bin size (deg) for
        ground-normal aggregation.

    Returns
    -------
    SegmentationResult
        Per-point normals and labels plus one :class:`ObstacleDetection` per
        DBSCAN cluster (excluding noise). Each obstacle includes
        ``closest_surface_point`` (YZ-binned robust minimum-X sample). When
        ``params.segmented_cloud_depth_shading`` is true, ``forward_x_smoothed``
        is an (N,) array (NaN on ground/noise) for debug cloud shading; otherwise
        it is None.

    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError('xyz must be of shape (N, 3)')
    n_pts = xyz.shape[0]
    if n_pts == 0:
        return SegmentationResult(
            points=xyz,
            normals=np.zeros((0, 3), dtype=np.float64),
            labels=np.zeros((0,), dtype=np.int32),
            obstacles=[],
            forward_x_smoothed=None,
        )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(params.normal_search_radius),
            max_nn=int(params.max_nn),
        ),
    )
    normals = np.asarray(pcd.normals, dtype=np.float64)
    normals = orient_normals_upward(normals, params.up_axis)

    if params.reference_normal_xyz is None:
        ref = np.median(normals, axis=0)
    else:
        ref = np.asarray(params.reference_normal_xyz, dtype=np.float64).reshape(3)
    ref = _unit(ref)

    cos_sim = np.abs(normals @ ref)
    ground_mask = cos_sim > float(params.cosine_threshold)
    obstacle_mask = ~ground_mask

    labels = np.full(n_pts, LABEL_OBSTACLE_NOISE, dtype=np.int32)
    labels[ground_mask] = LABEL_GROUND

    obstacles: list[ObstacleDetection] = []
    forward_full: np.ndarray | None = (
        np.full(n_pts, np.nan, dtype=np.float64)
        if bool(params.segmented_cloud_depth_shading)
        else None
    )
    if np.count_nonzero(obstacle_mask) == 0:
        return SegmentationResult(
            points=xyz,
            normals=normals,
            labels=labels,
            obstacles=[],
            forward_x_smoothed=forward_full,
        )

    obj_idx = np.flatnonzero(obstacle_mask)
    obj_xyz = xyz[obstacle_mask]
    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(obj_xyz)
    db_labels = np.array(
        obj_pcd.cluster_dbscan(
            eps=float(params.dbscan_eps),
            min_points=int(params.dbscan_min_points),
            print_progress=False,
        ),
        dtype=np.int32,
    )

    next_id = 1
    unique = np.unique(db_labels)
    for lab in unique:
        if lab < 0:
            continue
        local = np.flatnonzero(db_labels == lab)
        global_idx = obj_idx[local]
        cid = next_id
        next_id += 1
        labels[global_idx] = cid

        cluster_xyz = xyz[global_idx]
        cpcd = o3d.geometry.PointCloud()
        cpcd.points = o3d.utility.Vector3dVector(cluster_xyz)
        bbox = cpcd.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        scale = float(params.obstacle_bbox_scale)
        if scale <= 0.0:
            scale = 1.0
        if scale != 1.0:
            bbox = bbox.scale(scale, center)

        min_b = np.asarray(bbox.min_bound, dtype=np.float64)
        max_b = np.asarray(bbox.max_bound, dtype=np.float64)
        min_x, max_x, min_y, max_y = _footprint_xy_from_aabb(bbox)
        med_surf_n, anchor = _median_ground_normal_in_footprint(
            xyz,
            normals,
            ground_mask,
            min_x,
            max_x,
            min_y,
            max_y,
            float(params.surface_normal_bin_deg),
        )

        closest_pt, smooth_local = yz_binned_closest_and_smooth_x(
            cluster_xyz,
            bin_size_m=float(params.yz_depth_bin_size_m),
            min_points=int(params.yz_depth_min_points),
            percentile=float(params.yz_depth_percentile),
        )
        if forward_full is not None:
            forward_full[global_idx] = smooth_local

        obstacles.append(
            ObstacleDetection(
                cluster_id=int(cid),
                min_bound=min_b,
                max_bound=max_b,
                center=np.asarray(center, dtype=np.float64),
                median_surface_normal=med_surf_n,
                normal_anchor=anchor,
                closest_surface_point=np.asarray(closest_pt, dtype=np.float64),
            ),
        )

    return SegmentationResult(
        points=xyz,
        normals=normals,
        labels=labels,
        obstacles=obstacles,
        forward_x_smoothed=forward_full,
    )
