"""CRS-aware empirical semivariogram estimation and parametric variogram fitting.

Implements optional geostatistical support for field-scale spatial carbon analysis
without introducing GIS rendering dependencies (PDF Sections 24 & 34, ADR-001).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist

from st_ds.domain.errors import SchemaError

VariogramModelName = Literal["spherical", "exponential", "gaussian"]
CoordOrder = Literal["lon_lat", "xy_m"]

SUPPORTED_VARIOGRAM_MODELS: tuple[VariogramModelName, ...] = (
    "spherical",
    "exponential",
    "gaussian",
)
SUPPORTED_COORD_ORDERS: tuple[CoordOrder, ...] = ("lon_lat", "xy_m")
EARTH_RADIUS_M = 6_371_000.0


def validate_coord_array(
    coords: np.ndarray | Sequence[Sequence[float]],
    *,
    name: str = "coords",
) -> np.ndarray:
    """Validate a finite (N, 2) array of 2D spatial coordinates."""
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise SchemaError(
            f"'{name}' must be a 2D array with shape (N, 2); got {arr.shape}.",
            details={"field": name, "shape": list(arr.shape)},
        )
    if arr.shape[0] == 0:
        raise SchemaError(f"'{name}' must contain at least one coordinate pair.")
    if not np.all(np.isfinite(arr)):
        raise SchemaError(f"'{name}' contains non-finite (NaN or Inf) coordinates.")
    return arr


@dataclass(frozen=True)
class SpatialCoordinateFrame:
    """Coordinate reference frame and local tangent-plane projection to meters."""

    crs_epsg: str
    coord_order: CoordOrder
    origin_x: float
    origin_y: float

    @classmethod
    def from_coords(
        cls,
        coords: np.ndarray | Sequence[Sequence[float]],
        *,
        crs_epsg: str = "EPSG:4326",
        coord_order: CoordOrder = "lon_lat",
    ) -> SpatialCoordinateFrame:
        arr = validate_coord_array(coords, name="sample_coords")
        norm_crs = crs_epsg.strip().upper() if isinstance(crs_epsg, str) else ""
        if not norm_crs:
            raise SchemaError("crs_epsg must be a non-empty CRS identifier string.")
        if coord_order not in SUPPORTED_COORD_ORDERS:
            raise SchemaError(
                f"Unsupported coord_order '{coord_order}'; expected one of {SUPPORTED_COORD_ORDERS}.",
                details={"coord_order": coord_order},
            )
        if norm_crs == "EPSG:4326" and coord_order != "lon_lat":
            raise SchemaError(
                "EPSG:4326 requires coord_order='lon_lat' to avoid lat/lon axis swaps.",
                details={"crs_epsg": crs_epsg, "coord_order": coord_order},
            )
        frame = cls(
            crs_epsg=norm_crs,
            coord_order=coord_order,
            origin_x=float(np.mean(arr[:, 0])),
            origin_y=float(np.mean(arr[:, 1])),
        )
        frame.validate_coords(arr)
        return frame

    def validate_coords(
        self,
        coords: np.ndarray | Sequence[Sequence[float]],
        *,
        name: str = "coords",
    ) -> np.ndarray:
        arr = validate_coord_array(coords, name=name)
        if self.crs_epsg == "EPSG:4326" or self.coord_order == "lon_lat":
            lons = arr[:, 0]
            lats = arr[:, 1]
            if np.any((lons < -180.0) | (lons > 180.0)):
                raise SchemaError(
                    f"Longitude values in '{name}' must lie within [-180, 180] degrees.",
                    details={"min_lon": float(np.min(lons)), "max_lon": float(np.max(lons))},
                )
            if np.any((lats < -90.0) | (lats > 90.0)):
                raise SchemaError(
                    f"Latitude values in '{name}' must lie within [-90, 90] degrees.",
                    details={"min_lat": float(np.min(lats)), "max_lat": float(np.max(lats))},
                )
        return arr

    def to_local_meters(
        self,
        coords: np.ndarray | Sequence[Sequence[float]],
        *,
        name: str = "coords",
    ) -> np.ndarray:
        """Transform coordinates to local metric (x_m, y_m) relative to frame origin."""
        arr = self.validate_coords(coords, name=name)
        if self.crs_epsg == "EPSG:4326":
            lat0_rad = math.radians(self.origin_y)
            cos_lat0 = max(1e-6, math.cos(lat0_rad))
            dx = EARTH_RADIUS_M * cos_lat0 * np.radians(arr[:, 0] - self.origin_x)
            dy = EARTH_RADIUS_M * np.radians(arr[:, 1] - self.origin_y)
            return np.column_stack((dx, dy))
        return np.column_stack((arr[:, 0] - self.origin_x, arr[:, 1] - self.origin_y))

    def to_dict(self) -> dict[str, Any]:
        return {
            "crs_epsg": self.crs_epsg,
            "coord_order": self.coord_order,
            "origin_x": self.origin_x,
            "origin_y": self.origin_y,
        }


@dataclass(frozen=True)
class EmpiricalVariogram:
    """Empirical Matheron semivariogram lag summary."""

    lag_centers_m: np.ndarray
    semivariances: np.ndarray
    pair_counts: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "lag_centers_m": self.lag_centers_m.tolist(),
            "semivariances": self.semivariances.tolist(),
            "pair_counts": self.pair_counts.tolist(),
        }


@dataclass(frozen=True)
class VariogramModel:
    """Fitted isotropic parametric semivariogram model."""

    model_name: VariogramModelName
    nugget: float
    partial_sill: float
    range_m: float
    sill: float
    fit_rmse: float
    n_samples: int
    coordinate_frame: SpatialCoordinateFrame
    empirical: EmpiricalVariogram
    sample_bounds_m: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "nugget": self.nugget,
            "partial_sill": self.partial_sill,
            "range_m": self.range_m,
            "sill": self.sill,
            "fit_rmse": self.fit_rmse,
            "n_samples": self.n_samples,
            "coordinate_frame": self.coordinate_frame.to_dict(),
            "empirical": self.empirical.to_dict(),
            "sample_bounds_m": list(self.sample_bounds_m),
        }


def evaluate_variogram_params(
    h: np.ndarray | float,
    *,
    model_name: VariogramModelName,
    nugget: float,
    partial_sill: float,
    range_m: float,
) -> np.ndarray:
    """Evaluate gamma(h) with gamma(0) = 0 and practical range convention."""
    dist = np.asarray(h, dtype=np.float64)
    scalar_input = dist.ndim == 0
    dist_arr = np.atleast_1d(dist)
    out = np.zeros_like(dist_arr, dtype=np.float64)
    pos = dist_arr > 0.0
    if np.any(pos):
        u = dist_arr[pos] / range_m
        if model_name == "spherical":
            u_clip = np.minimum(u, 1.0)
            structure = 1.5 * u_clip - 0.5 * (u_clip ** 3)
        elif model_name == "exponential":
            structure = 1.0 - np.exp(-3.0 * u)
        elif model_name == "gaussian":
            structure = 1.0 - np.exp(-3.0 * (u ** 2))
        else:
            raise SchemaError(
                f"Unsupported variogram model '{model_name}'.",
                details={"model_name": model_name},
            )
        out[pos] = nugget + partial_sill * structure
    return out[0] if scalar_input else out


def evaluate_variogram(h: np.ndarray | float, model: VariogramModel) -> np.ndarray:
    """Evaluate a fitted VariogramModel at lag distance(s) h (meters)."""
    return evaluate_variogram_params(
        h,
        model_name=model.model_name,
        nugget=model.nugget,
        partial_sill=model.partial_sill,
        range_m=model.range_m,
    )


def _compute_empirical_variogram(
    distances_m: np.ndarray,
    half_sq_diffs: np.ndarray,
    n_lags: int,
) -> EmpiricalVariogram:
    max_dist = float(np.max(distances_m))
    edges = np.linspace(0.0, max_dist * (1.0 + 1e-9), n_lags + 1)

    centers: list[float] = []
    semivars: list[float] = []
    counts: list[int] = []

    for i in range(n_lags):
        mask = (distances_m >= edges[i]) & (distances_m < edges[i + 1])
        cnt = int(np.sum(mask))
        if cnt > 0:
            centers.append(float(np.mean(distances_m[mask])))
            semivars.append(float(np.mean(half_sq_diffs[mask])))
            counts.append(cnt)

    if len(centers) < 2 and len(distances_m) >= 2:
        q_edges = np.quantile(distances_m, np.linspace(0.0, 1.0, min(n_lags, len(distances_m)) + 1))
        q_edges = np.unique(q_edges)
        if len(q_edges) >= 3:
            q_edges[-1] = q_edges[-1] * (1.0 + 1e-9)
            centers, semivars, counts = [], [], []
            for i in range(len(q_edges) - 1):
                mask = (distances_m >= q_edges[i]) & (distances_m < q_edges[i + 1])
                cnt = int(np.sum(mask))
                if cnt > 0:
                    centers.append(float(np.mean(distances_m[mask])))
                    semivars.append(float(np.mean(half_sq_diffs[mask])))
                    counts.append(cnt)

    if not centers:
        raise SchemaError("Unable to construct non-empty empirical variogram lag bins.")

    return EmpiricalVariogram(
        lag_centers_m=np.asarray(centers, dtype=np.float64),
        semivariances=np.asarray(semivars, dtype=np.float64),
        pair_counts=np.asarray(counts, dtype=np.int64),
    )


def fit_variogram(
    coords: np.ndarray | Sequence[Sequence[float]],
    values: np.ndarray | Sequence[float],
    *,
    crs_epsg: str = "EPSG:4326",
    coord_order: CoordOrder = "lon_lat",
    model: VariogramModelName = "spherical",
    n_lags: int = 8,
    min_points: int = 8,
) -> VariogramModel:
    """Fit an empirical + parametric isotropic semivariogram to spatial observations."""
    if model not in SUPPORTED_VARIOGRAM_MODELS:
        raise SchemaError(
            f"Unsupported variogram model '{model}'; choose from {SUPPORTED_VARIOGRAM_MODELS}.",
            details={"model": model},
        )
    if n_lags < 3:
        raise SchemaError("n_lags must be at least 3.", details={"n_lags": n_lags})
    if min_points < 4:
        raise SchemaError("min_points must be at least 4.", details={"min_points": min_points})

    coords_arr = validate_coord_array(coords, name="sample_coords")
    vals = np.asarray(values, dtype=np.float64)
    if vals.ndim != 1 or vals.shape[0] != coords_arr.shape[0]:
        raise SchemaError(
            "sample_values must be a 1D array matching sample_coords length.",
            details={"n_coords": coords_arr.shape[0], "values_shape": list(vals.shape)},
        )
    if not np.all(np.isfinite(vals)):
        raise SchemaError("sample_values contains non-finite (NaN or Inf) entries.")

    n_samples = int(coords_arr.shape[0])
    if n_samples < min_points:
        raise SchemaError(
            f"Insufficient spatial sample count ({n_samples} < min_points={min_points}) to fit variogram.",
            details={"n_samples": n_samples, "min_points": min_points},
        )

    frame = SpatialCoordinateFrame.from_coords(
        coords_arr,
        crs_epsg=crs_epsg,
        coord_order=coord_order,
    )
    local_m = frame.to_local_meters(coords_arr, name="sample_coords")

    d_vec = pdist(local_m, metric="euclidean")
    max_dist = float(np.max(d_vec))
    if max_dist <= 1e-6:
        raise SchemaError(
            "All sample coordinates collapse to a single spatial location (zero spatial spread); "
            "variogram fitting is ill-posed.",
            details={"max_pairwise_distance_m": max_dist},
        )

    idx_i, idx_j = np.triu_indices(n_samples, k=1)
    half_sq_diffs = 0.5 * np.square(vals[idx_i] - vals[idx_j])
    empirical = _compute_empirical_variogram(d_vec, half_sq_diffs, n_lags=n_lags)

    lags = empirical.lag_centers_m
    semivars = empirical.semivariances
    weights = np.sqrt(empirical.pair_counts.astype(np.float64))

    sample_var = float(np.var(vals))
    scale_sill = max(1e-6, sample_var, float(np.max(semivars)))
    min_pos_lag = max(1e-3, float(np.min(lags[lags > 0.0])) if np.any(lags > 0.0) else 1.0)

    p0 = np.array(
        [
            max(0.0, min(0.05 * scale_sill, float(semivars[0]) * 0.25)),
            max(1e-5, scale_sill),
            max(min_pos_lag * 1.5, 0.5 * max_dist),
        ],
        dtype=np.float64,
    )
    lower = np.array([0.0, 1e-8, min_pos_lag * 0.25], dtype=np.float64)
    upper = np.array([scale_sill * 3.0, scale_sill * 10.0, max_dist * 5.0], dtype=np.float64)
    p0 = np.clip(p0, lower + 1e-9, upper - 1e-9)

    def _residuals(params: np.ndarray) -> np.ndarray:
        pred = evaluate_variogram_params(
            lags,
            model_name=model,
            nugget=float(params[0]),
            partial_sill=float(params[1]),
            range_m=float(params[2]),
        )
        return weights * (pred - semivars)

    opt = least_squares(_residuals, x0=p0, bounds=(lower, upper), max_nfev=500)
    nugget, partial_sill, range_m = (float(opt.x[0]), float(opt.x[1]), float(opt.x[2]))

    fitted_curve = evaluate_variogram_params(
        lags,
        model_name=model,
        nugget=nugget,
        partial_sill=partial_sill,
        range_m=range_m,
    )
    fit_rmse = float(np.sqrt(np.mean(np.square(fitted_curve - semivars))))

    bounds_m = (
        float(np.min(local_m[:, 0])),
        float(np.max(local_m[:, 0])),
        float(np.min(local_m[:, 1])),
        float(np.max(local_m[:, 1])),
    )

    return VariogramModel(
        model_name=model,
        nugget=nugget,
        partial_sill=partial_sill,
        range_m=range_m,
        sill=float(nugget + partial_sill),
        fit_rmse=fit_rmse,
        n_samples=n_samples,
        coordinate_frame=frame,
        empirical=empirical,
        sample_bounds_m=bounds_m,
    )
