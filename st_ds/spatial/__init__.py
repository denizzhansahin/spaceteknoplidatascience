"""Optional CRS-aware geostatistical variogram and Ordinary Kriging support."""

from st_ds.spatial.kriging import (
    KrigingPrediction,
    kriging_to_geojson,
    ordinary_kriging,
)
from st_ds.spatial.variogram import (
    EARTH_RADIUS_M,
    SUPPORTED_COORD_ORDERS,
    SUPPORTED_VARIOGRAM_MODELS,
    CoordOrder,
    EmpiricalVariogram,
    SpatialCoordinateFrame,
    VariogramModel,
    VariogramModelName,
    evaluate_variogram,
    fit_variogram,
    validate_coord_array,
)

__all__ = [
    "EARTH_RADIUS_M",
    "SUPPORTED_COORD_ORDERS",
    "SUPPORTED_VARIOGRAM_MODELS",
    "CoordOrder",
    "EmpiricalVariogram",
    "SpatialCoordinateFrame",
    "VariogramModel",
    "VariogramModelName",
    "validate_coord_array",
    "evaluate_variogram",
    "fit_variogram",
    "KrigingPrediction",
    "ordinary_kriging",
    "kriging_to_geojson",
]
