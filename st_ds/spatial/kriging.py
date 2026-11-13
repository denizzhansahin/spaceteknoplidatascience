"""Ordinary Kriging interpolation, uncertainty variance, and GeoJSON export.

Solves the (N+1) x (N+1) Ordinary Kriging system in covariance form:
    [ C   1 ] [ lambda ] = [ c(s0) ]
    [ 1^T 0 ] [   mu   ]   [   1   ]
where C(h) = sill - gamma(h) and C(0) = sill = nugget + partial_sill.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy.spatial import Delaunay, QhullError
from scipy.spatial.distance import cdist

from st_ds.domain.errors import SchemaError
from st_ds.domain.quality import QualityFlag, QualityStatus
from st_ds.spatial.variogram import (
    CoordOrder,
    VariogramModel,
    VariogramModelName,
    evaluate_variogram,
    fit_variogram,
    validate_coord_array,
)


@dataclass(frozen=True)
class KrigingPrediction:
    """Output container for Ordinary Kriging predictions and diagnostics."""

    query_coords: np.ndarray
    query_coords_m: np.ndarray
    predicted_values: np.ndarray
    kriging_variance: np.ndarray
    kriging_std: np.ndarray
    weights_sum_max_error: float
    is_extrapolated_mask: np.ndarray
    quality_status: QualityStatus
    quality_flags: list[QualityFlag] = field(default_factory=list)
    variogram: VariogramModel | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_coords": self.query_coords.tolist(),
            "query_coords_m": self.query_coords_m.tolist(),
            "predicted_values": self.predicted_values.tolist(),
            "kriging_variance": self.kriging_variance.tolist(),
            "kriging_std": self.kriging_std.tolist(),
            "weights_sum_max_error": self.weights_sum_max_error,
            "is_extrapolated_mask": [bool(x) for x in self.is_extrapolated_mask.tolist()],
            "quality_status": self.quality_status.value,
            "quality_flags": [flag.value for flag in self.quality_flags],
            "variogram": self.variogram.to_dict() if self.variogram is not None else None,
            "diagnostics": dict(self.diagnostics),
        }


def _compute_extrapolation_mask(
    sample_m: np.ndarray,
    query_m: np.ndarray,
) -> np.ndarray:
    """Flag query locations lying outside the convex support of sample points."""
    try:
        hull = Delaunay(sample_m)
        return hull.find_simplex(query_m, tol=1e-9) < 0
    except QhullError:
        min_x, max_x = float(np.min(sample_m[:, 0])), float(np.max(sample_m[:, 0]))
        min_y, max_y = float(np.min(sample_m[:, 1])), float(np.max(sample_m[:, 1]))
        eps = 1e-9
        return (
            (query_m[:, 0] < min_x - eps)
            | (query_m[:, 0] > max_x + eps)
            | (query_m[:, 1] < min_y - eps)
            | (query_m[:, 1] > max_y + eps)
        )


def ordinary_kriging(
    sample_coords: np.ndarray | Sequence[Sequence[float]],
    sample_values: np.ndarray | Sequence[float],
    query_coords: np.ndarray | Sequence[Sequence[float]],
    *,
    variogram: VariogramModel | None = None,
    crs_epsg: str = "EPSG:4326",
    coord_order: CoordOrder = "lon_lat",
    model: VariogramModelName = "spherical",
    n_lags: int = 8,
    min_points: int = 8,
    ridge: float = 1e-10,
) -> KrigingPrediction:
    """Perform 2D Ordinary Kriging interpolation with unbiasedness & variance checks."""
    s_coords = validate_coord_array(sample_coords, name="sample_coords")
    q_coords = validate_coord_array(query_coords, name="query_coords")
    s_vals = np.asarray(sample_values, dtype=np.float64)
    if s_vals.ndim != 1 or s_vals.shape[0] != s_coords.shape[0]:
        raise SchemaError(
            "sample_values must be a 1D array matching sample_coords length.",
            details={"n_coords": s_coords.shape[0], "values_shape": list(s_vals.shape)},
        )
    if not np.all(np.isfinite(s_vals)):
        raise SchemaError("sample_values contains non-finite (NaN or Inf) entries.")

    if variogram is None:
        v_model = fit_variogram(
            s_coords,
            s_vals,
            crs_epsg=crs_epsg,
            coord_order=coord_order,
            model=model,
            n_lags=n_lags,
            min_points=min_points,
        )
    else:
        v_model = variogram
        if s_coords.shape[0] < min_points:
            raise SchemaError(
                f"Insufficient spatial sample count ({s_coords.shape[0]} < min_points={min_points}).",
                details={"n_samples": int(s_coords.shape[0]), "min_points": min_points},
            )
        norm_crs = crs_epsg.strip().upper()
        if v_model.coordinate_frame.crs_epsg != norm_crs or v_model.coordinate_frame.coord_order != coord_order:
            raise SchemaError(
                "Provided variogram coordinate frame does not match crs_epsg/coord_order.",
                details={
                    "variogram_frame": v_model.coordinate_frame.to_dict(),
                    "requested_crs": norm_crs,
                    "requested_coord_order": coord_order,
                },
            )

    frame = v_model.coordinate_frame
    sample_m = frame.to_local_meters(s_coords, name="sample_coords")
    query_m = frame.to_local_meters(q_coords, name="query_coords")

    n_samples = sample_m.shape[0]
    n_queries = query_m.shape[0]

    dist_ss = cdist(sample_m, sample_m, metric="euclidean")
    off_diag = dist_ss[np.triu_indices(n_samples, k=1)]
    if off_diag.size == 0 or float(np.min(off_diag)) <= 1e-6:
        raise SchemaError(
            "Duplicate or nearly coincident sample coordinates detected; "
            "Ordinary Kriging matrix is singular.",
            details={"min_pairwise_distance_m": float(np.min(off_diag)) if off_diag.size else 0.0},
        )

    c0 = float(v_model.sill)
    cov_ss = c0 - evaluate_variogram(dist_ss, v_model)
    # Tiny diagonal stabilization preserves exact covariance conditioning without distorting weights.
    cov_ss = cov_ss + ridge * max(c0, 1.0) * np.eye(n_samples, dtype=np.float64)

    ok_mat = np.zeros((n_samples + 1, n_samples + 1), dtype=np.float64)
    ok_mat[:n_samples, :n_samples] = cov_ss
    ok_mat[:n_samples, n_samples] = 1.0
    ok_mat[n_samples, :n_samples] = 1.0
    ok_mat[n_samples, n_samples] = 0.0

    dist_sq = cdist(sample_m, query_m, metric="euclidean")
    cov_sq = c0 - evaluate_variogram(dist_sq, v_model)

    rhs = np.ones((n_samples + 1, n_queries), dtype=np.float64)
    rhs[:n_samples, :] = cov_sq

    try:
        sol = np.linalg.solve(ok_mat, rhs)
    except np.linalg.LinAlgError as exc:
        raise SchemaError(
            f"Ordinary Kriging linear system solve failed: {exc}",
            details={"n_samples": n_samples},
        ) from exc

    weights = sol[:n_samples, :]  # shape (N, M)
    lagrange_mu = sol[n_samples, :]  # shape (M,)

    # Use np.dot instead of '@' to avoid Apple Silicon NumPy 2.x BLAS warnings.
    preds = np.dot(s_vals, weights)
    expected_dot = np.sum(weights * cov_sq, axis=0)
    raw_var = c0 - expected_dot - lagrange_mu
    raw_var = np.where((raw_var < 0.0) & (raw_var > -1e-9 * max(c0, 1.0)), 0.0, raw_var)
    kriging_var = np.maximum(0.0, raw_var)
    kriging_std = np.sqrt(kriging_var)

    weight_sums = np.sum(weights, axis=0)
    weights_sum_max_error = float(np.max(np.abs(weight_sums - 1.0)))

    extrap_mask = _compute_extrapolation_mask(sample_m, query_m)
    flags: list[QualityFlag] = []
    status = QualityStatus.OK
    if np.any(extrap_mask):
        flags.append(QualityFlag.OUT_OF_CALIBRATION_DOMAIN)
        status = QualityStatus.WARNING

    return KrigingPrediction(
        query_coords=q_coords,
        query_coords_m=query_m,
        predicted_values=preds,
        kriging_variance=kriging_var,
        kriging_std=kriging_std,
        weights_sum_max_error=weights_sum_max_error,
        is_extrapolated_mask=extrap_mask,
        quality_status=status,
        quality_flags=flags,
        variogram=v_model,
        diagnostics={
            "n_samples": n_samples,
            "n_queries": n_queries,
            "n_extrapolated": int(np.sum(extrap_mask)),
            "min_sample_spacing_m": float(np.min(off_diag)),
            "max_sample_spacing_m": float(np.max(off_diag)),
            "sill": c0,
            "nugget": float(v_model.nugget),
            "range_m": float(v_model.range_m),
        },
    )


def kriging_to_geojson(
    prediction: KrigingPrediction,
    *,
    value_property: str = "predicted_value",
    variance_property: str = "kriging_variance",
    std_property: str = "kriging_std",
) -> dict[str, Any]:
    """Serialize KrigingPrediction points into a pure JSON-compatible GeoJSON FeatureCollection."""
    if not isinstance(prediction, KrigingPrediction):
        raise SchemaError("prediction must be a KrigingPrediction instance.")

    features: list[dict[str, Any]] = []
    n_points = prediction.query_coords.shape[0]
    for idx in range(n_points):
        x_val = float(prediction.query_coords[idx, 0])
        y_val = float(prediction.query_coords[idx, 1])
        pred_val = float(prediction.predicted_values[idx])
        var_val = float(prediction.kriging_variance[idx])
        std_val = float(prediction.kriging_std[idx])
        if not (
            math.isfinite(x_val)
            and math.isfinite(y_val)
            and math.isfinite(pred_val)
            and math.isfinite(var_val)
            and math.isfinite(std_val)
        ):
            raise SchemaError(
                f"Non-finite value at GeoJSON feature index {idx}.",
                details={"index": idx},
            )
        features.append(
            {
                "type": "Feature",
                "id": idx,
                "geometry": {
                    "type": "Point",
                    "coordinates": [x_val, y_val],
                },
                "properties": {
                    value_property: pred_val,
                    variance_property: var_val,
                    std_property: std_val,
                    "is_extrapolated": bool(prediction.is_extrapolated_mask[idx]),
                },
            }
        )

    crs_meta = (
        prediction.variogram.coordinate_frame.to_dict()
        if prediction.variogram is not None
        else None
    )
    return {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "quality_status": prediction.quality_status.value,
            "quality_flags": [flag.value for flag in prediction.quality_flags],
            "weights_sum_max_error": prediction.weights_sum_max_error,
            "coordinate_frame": crs_meta,
        },
    }
