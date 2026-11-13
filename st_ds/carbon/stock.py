"""Soil carbon stock calculation and first-order covariance uncertainty propagation.

Canonical equation (PDF Section 23, DEF-005):
    S = C_pct * rho_b * d_cm * (1 - f_coarse)   [Mg C ha^-1]

where:
    C_pct     in % (pct dry matter, canonical)
    rho_b     in g/cm^3 (canonical)
    d_cm      in cm (canonical layer thickness = depth_bottom_cm - depth_top_cm)
    f_coarse  in [0, 0.99] volume/mass fraction (canonical)

Strict fail-closed policy (ADR-004 / DEF-005):
    Missing bulk density NEVER falls back to an implicit default such as 1.35 g/cm^3.
    Instead, ``stock_mg_ha`` is returned as ``None`` with ``QualityFlag.MISSING_BULK_DENSITY``
    and ``QualityStatus.INSUFFICIENT_DATA``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from st_ds.domain.errors import SchemaError
from st_ds.domain.quality import QualityFlag, QualityStatus
from st_ds.domain.schemas import DepthInterval
from st_ds.domain.units import UnitRegistry

STOCK_PARAM_NAMES: tuple[str, str, str, str] = (
    "carbon_pct",
    "bulk_density",
    "depth_cm",
    "coarse_fragments",
)


@dataclass(frozen=True)
class NormalizedStockInputs:
    """Canonical physical inputs for single-layer soil carbon stock calculation."""

    carbon_pct: float
    bulk_density_g_cm3: float | None
    depth_top_cm: float
    depth_bottom_cm: float
    thickness_cm: float
    coarse_fraction: float
    original_units: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "carbon_pct": self.carbon_pct,
            "bulk_density_g_cm3": self.bulk_density_g_cm3,
            "depth_top_cm": self.depth_top_cm,
            "depth_bottom_cm": self.depth_bottom_cm,
            "thickness_cm": self.thickness_cm,
            "coarse_fraction": self.coarse_fraction,
            "original_units": dict(self.original_units),
        }


@dataclass(frozen=True)
class LayerStockInput:
    """Declarative input specification for one depth horizon in a soil profile."""

    carbon_value: float
    carbon_unit: str = "pct"
    bulk_density: float | None = None
    bulk_density_unit: str = "g/cm3"
    depth_top: float = 0.0
    depth_bottom: float = 30.0
    depth_unit: str = "cm"
    coarse_fraction: float = 0.0
    coarse_unit: str = "fraction"
    carbon_std: float | None = None
    bulk_density_std: float | None = None
    depth_std: float | None = None
    coarse_std: float | None = None
    cov_matrix: np.ndarray | list[list[float]] | None = None
    label: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> LayerStockInput:
        allowed = set(cls.__dataclass_fields__.keys())
        unknown = set(data.keys()) - allowed
        if unknown:
            raise SchemaError(
                f"Unknown LayerStockInput fields: {sorted(unknown)}.",
                details={"unknown_fields": sorted(unknown)},
            )
        return cls(**dict(data))


@dataclass(frozen=True)
class StockUncertaintyResult:
    """First-order covariance uncertainty propagation output for carbon stock."""

    stock_std_mg_ha: float
    stock_variance_mg_ha2: float
    gradient: dict[str, float]
    variance_contributions: dict[str, float]
    covariance_interaction_mg_ha2: float
    dominant_uncertainty_source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stock_std_mg_ha": self.stock_std_mg_ha,
            "stock_variance_mg_ha2": self.stock_variance_mg_ha2,
            "gradient": dict(self.gradient),
            "variance_contributions": dict(self.variance_contributions),
            "covariance_interaction_mg_ha2": self.covariance_interaction_mg_ha2,
            "dominant_uncertainty_source": self.dominant_uncertainty_source,
        }


@dataclass(frozen=True)
class CarbonStockResult:
    """Carbon stock estimation and quality diagnostics for a single soil layer."""

    stock_mg_ha: float | None
    stock_std_mg_ha: float | None
    normalized_inputs: NormalizedStockInputs
    uncertainty: StockUncertaintyResult | None
    quality_status: QualityStatus
    quality_flags: list[QualityFlag] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "stock_mg_ha": self.stock_mg_ha,
            "stock_std_mg_ha": self.stock_std_mg_ha,
            "normalized_inputs": self.normalized_inputs.to_dict(),
            "uncertainty": self.uncertainty.to_dict() if self.uncertainty is not None else None,
            "quality_status": self.quality_status.value,
            "quality_flags": [flag.value for flag in self.quality_flags],
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ProfileStockResult:
    """Aggregated multi-horizon soil profile carbon stock result."""

    total_stock_mg_ha: float | None
    total_stock_std_mg_ha: float | None
    total_depth_cm: float
    layer_results: list[CarbonStockResult]
    dominant_uncertainty_source: str | None
    quality_status: QualityStatus
    quality_flags: list[QualityFlag] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_stock_mg_ha": self.total_stock_mg_ha,
            "total_stock_std_mg_ha": self.total_stock_std_mg_ha,
            "total_depth_cm": self.total_depth_cm,
            "layer_results": [layer.to_dict() for layer in self.layer_results],
            "dominant_uncertainty_source": self.dominant_uncertainty_source,
            "quality_status": self.quality_status.value,
            "quality_flags": [flag.value for flag in self.quality_flags],
            "notes": list(self.notes),
        }


def _convert_nonnegative_std(
    value: float | None,
    quantity_type: str,
    unit: str,
    field_name: str,
) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(
            f"Uncertainty '{field_name}' must be a finite real number.",
            details={"field": field_name, "value": value},
        )
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise SchemaError(
            f"Uncertainty '{field_name}' must be finite and non-negative.",
            details={"field": field_name, "value": value},
        )
    converted = UnitRegistry.to_canonical(
        quantity_type,
        numeric,
        unit,
        validate_bounds=False,
    )
    if converted < 0.0 or not math.isfinite(converted):
        raise SchemaError(
            f"Converted uncertainty '{field_name}' must be finite and non-negative.",
            details={"field": field_name, "converted_value": converted},
        )
    return float(converted)


def validate_stock_units(
    *,
    carbon_value: float,
    carbon_unit: str,
    bulk_density: float | None = None,
    bulk_density_unit: str = "g/cm3",
    depth_top: float = 0.0,
    depth_bottom: float = 30.0,
    depth_unit: str = "cm",
    coarse_fraction: float = 0.0,
    coarse_unit: str = "fraction",
) -> NormalizedStockInputs:
    """Validate units and physical bounds and convert all stock inputs to canonical units."""
    carbon_pct = UnitRegistry.to_canonical(
        "carbon_concentration",
        carbon_value,
        carbon_unit,
        validate_bounds=True,
    )

    rho_b: float | None = None
    if bulk_density is not None:
        rho_b = UnitRegistry.to_canonical(
            "bulk_density",
            bulk_density,
            bulk_density_unit,
            validate_bounds=True,
        )

    top_cm = UnitRegistry.to_canonical(
        "depth",
        depth_top,
        depth_unit,
        validate_bounds=True,
    )
    bottom_cm = UnitRegistry.to_canonical(
        "depth",
        depth_bottom,
        depth_unit,
        validate_bounds=True,
    )
    interval = DepthInterval(top_cm=top_cm, bottom_cm=bottom_cm)

    f_coarse = UnitRegistry.to_canonical(
        "coarse_fragments",
        coarse_fraction,
        coarse_unit,
        validate_bounds=True,
    )

    return NormalizedStockInputs(
        carbon_pct=carbon_pct,
        bulk_density_g_cm3=rho_b,
        depth_top_cm=interval.top_cm,
        depth_bottom_cm=interval.bottom_cm,
        thickness_cm=interval.thickness_cm,
        coarse_fraction=f_coarse,
        original_units={
            "carbon": UnitRegistry.normalize_unit_key(carbon_unit),
            "bulk_density": UnitRegistry.normalize_unit_key(bulk_density_unit)
            if bulk_density is not None
            else "missing",
            "depth": UnitRegistry.normalize_unit_key(depth_unit),
            "coarse_fragments": UnitRegistry.normalize_unit_key(coarse_unit),
        },
    )


def _validate_covariance_matrix(cov_matrix: np.ndarray | list[list[float]]) -> np.ndarray:
    cov = np.asarray(cov_matrix, dtype=np.float64)
    if cov.shape != (4, 4):
        raise SchemaError(
            f"cov_matrix must have shape (4, 4) corresponding to {STOCK_PARAM_NAMES}; got {cov.shape}.",
            details={"shape": list(cov.shape)},
        )
    if not np.all(np.isfinite(cov)):
        raise SchemaError("cov_matrix contains non-finite entries.")
    if not np.allclose(cov, cov.T, rtol=1e-7, atol=1e-10):
        raise SchemaError("cov_matrix must be symmetric.")
    cov = 0.5 * (cov + cov.T)
    diag = np.diag(cov)
    if np.any(diag < -1e-12):
        raise SchemaError(
            "cov_matrix diagonal variances cannot be negative.",
            details={"diagonal": diag.tolist()},
        )
    eigvals = np.linalg.eigvalsh(cov)
    if float(np.min(eigvals)) < -1e-9:
        raise SchemaError(
            "cov_matrix must be positive semi-definite.",
            details={"min_eigenvalue": float(np.min(eigvals))},
        )
    return cov


def propagate_stock_uncertainty(
    *,
    carbon_pct: float,
    bulk_density_g_cm3: float,
    depth_cm: float,
    coarse_fraction: float = 0.0,
    carbon_std_pct: float = 0.0,
    bulk_density_std_g_cm3: float = 0.0,
    depth_std_cm: float = 0.0,
    coarse_std_fraction: float = 0.0,
    cov_matrix: np.ndarray | list[list[float]] | None = None,
) -> StockUncertaintyResult:
    """Propagate parameter uncertainties through S = C * rho_b * d * (1 - f_c).

    Computes the analytical first-order variance:
        sigma_S^2 = grad(S)^T * Cov(theta) * grad(S)
    where theta = (C_pct, rho_b, d_cm, f_coarse) in canonical units.
    """
    UnitRegistry.validate_physical_range("carbon_concentration", float(carbon_pct))
    UnitRegistry.validate_physical_range("bulk_density", float(bulk_density_g_cm3))
    UnitRegistry.validate_physical_range("depth", float(depth_cm))
    if depth_cm <= 0.0:
        raise SchemaError("Layer thickness depth_cm must be strictly positive.", details={"depth_cm": depth_cm})
    UnitRegistry.validate_physical_range("coarse_fragments", float(coarse_fraction))

    c_val = float(carbon_pct)
    rho_val = float(bulk_density_g_cm3)
    d_val = float(depth_cm)
    fc_val = float(coarse_fraction)

    g_c = rho_val * d_val * (1.0 - fc_val)
    g_rho = c_val * d_val * (1.0 - fc_val)
    g_d = c_val * rho_val * (1.0 - fc_val)
    g_fc = -c_val * rho_val * d_val
    grad = np.array([g_c, g_rho, g_d, g_fc], dtype=np.float64)

    if cov_matrix is not None:
        cov = _validate_covariance_matrix(cov_matrix)
    else:
        stds = [
            _convert_nonnegative_std(carbon_std_pct, "carbon_concentration", "pct", "carbon_std_pct"),
            _convert_nonnegative_std(bulk_density_std_g_cm3, "bulk_density", "g/cm3", "bulk_density_std_g_cm3"),
            _convert_nonnegative_std(depth_std_cm, "depth", "cm", "depth_std_cm"),
            _convert_nonnegative_std(coarse_std_fraction, "coarse_fragments", "fraction", "coarse_std_fraction"),
        ]
        cov = np.diag(np.square(np.asarray(stds, dtype=np.float64)))

    # Use np.dot instead of '@' to avoid Apple Silicon NumPy 2.x BLAS warnings.
    cov_grad = np.dot(cov, grad)
    total_var = float(np.dot(grad, cov_grad))
    if total_var < 0.0 and total_var > -1e-11:
        total_var = 0.0
    elif total_var < 0.0:
        raise SchemaError(
            f"Propagated stock variance is negative ({total_var}); verify covariance matrix.",
            details={"variance": total_var},
        )

    diag_contribs = {
        name: float(max(0.0, (grad[idx] ** 2) * cov[idx, idx]))
        for idx, name in enumerate(STOCK_PARAM_NAMES)
    }
    diag_sum = float(sum(diag_contribs.values()))
    interaction = float(total_var - diag_sum)

    if diag_sum <= 0.0:
        dominant = "none"
    else:
        dominant = max(diag_contribs, key=lambda k: diag_contribs[k])

    return StockUncertaintyResult(
        stock_std_mg_ha=float(math.sqrt(total_var)),
        stock_variance_mg_ha2=total_var,
        gradient={name: float(grad[idx]) for idx, name in enumerate(STOCK_PARAM_NAMES)},
        variance_contributions=diag_contribs,
        covariance_interaction_mg_ha2=interaction,
        dominant_uncertainty_source=dominant,
    )


def compute_layer_stock(
    *,
    carbon_value: float,
    carbon_unit: str = "pct",
    bulk_density: float | None = None,
    bulk_density_unit: str = "g/cm3",
    depth_top: float = 0.0,
    depth_bottom: float = 30.0,
    depth_unit: str = "cm",
    coarse_fraction: float = 0.0,
    coarse_unit: str = "fraction",
    carbon_std: float | None = None,
    bulk_density_std: float | None = None,
    depth_std: float | None = None,
    coarse_std: float | None = None,
    cov_matrix: np.ndarray | list[list[float]] | None = None,
    label: str | None = None,
) -> CarbonStockResult:
    """Compute soil carbon stock (Mg C ha^-1) for a single depth interval.

    If ``bulk_density`` is ``None``, no stock is computed and no implicit
    fallback density is substituted (DEF-005 / ADR-004).
    """
    norm = validate_stock_units(
        carbon_value=carbon_value,
        carbon_unit=carbon_unit,
        bulk_density=bulk_density,
        bulk_density_unit=bulk_density_unit,
        depth_top=depth_top,
        depth_bottom=depth_bottom,
        depth_unit=depth_unit,
        coarse_fraction=coarse_fraction,
        coarse_unit=coarse_unit,
    )

    if norm.bulk_density_g_cm3 is None:
        return CarbonStockResult(
            stock_mg_ha=None,
            stock_std_mg_ha=None,
            normalized_inputs=norm,
            uncertainty=None,
            quality_status=QualityStatus.INSUFFICIENT_DATA,
            quality_flags=[QualityFlag.MISSING_BULK_DENSITY],
            notes=[
                "Bulk density is missing; carbon stock cannot be computed (no implicit default per ADR-004/DEF-005)."
            ],
            label=label,
        )

    stock_mg_ha = float(
        norm.carbon_pct
        * norm.bulk_density_g_cm3
        * norm.thickness_cm
        * (1.0 - norm.coarse_fraction)
    )
    UnitRegistry.validate_physical_range("carbon_stock", stock_mg_ha)

    has_uncertainty = (
        cov_matrix is not None
        or carbon_std is not None
        or bulk_density_std is not None
        or depth_std is not None
        or coarse_std is not None
    )

    uncertainty_res: StockUncertaintyResult | None = None
    if has_uncertainty:
        c_std_pct = _convert_nonnegative_std(
            carbon_std, "carbon_concentration", carbon_unit, "carbon_std"
        )
        rho_std = _convert_nonnegative_std(
            bulk_density_std, "bulk_density", bulk_density_unit, "bulk_density_std"
        )
        d_std_cm = _convert_nonnegative_std(
            depth_std, "depth", depth_unit, "depth_std"
        )
        fc_std = _convert_nonnegative_std(
            coarse_std, "coarse_fragments", coarse_unit, "coarse_std"
        )
        uncertainty_res = propagate_stock_uncertainty(
            carbon_pct=norm.carbon_pct,
            bulk_density_g_cm3=norm.bulk_density_g_cm3,
            depth_cm=norm.thickness_cm,
            coarse_fraction=norm.coarse_fraction,
            carbon_std_pct=c_std_pct,
            bulk_density_std_g_cm3=rho_std,
            depth_std_cm=d_std_cm,
            coarse_std_fraction=fc_std,
            cov_matrix=cov_matrix,
        )

    return CarbonStockResult(
        stock_mg_ha=stock_mg_ha,
        stock_std_mg_ha=uncertainty_res.stock_std_mg_ha if uncertainty_res is not None else None,
        normalized_inputs=norm,
        uncertainty=uncertainty_res,
        quality_status=QualityStatus.OK,
        quality_flags=[],
        notes=[],
        label=label,
    )


def compute_profile_stock(
    layers: Sequence[LayerStockInput | Mapping[str, Any]],
    *,
    require_contiguous: bool = False,
) -> ProfileStockResult:
    """Aggregate carbon stocks across non-overlapping depth horizons in a soil profile."""
    if not layers:
        raise SchemaError("At least one depth layer is required to compute profile stock.")

    parsed_layers = [
        layer if isinstance(layer, LayerStockInput) else LayerStockInput.from_mapping(layer)
        for layer in layers
    ]

    layer_results = [
        compute_layer_stock(
            carbon_value=layer.carbon_value,
            carbon_unit=layer.carbon_unit,
            bulk_density=layer.bulk_density,
            bulk_density_unit=layer.bulk_density_unit,
            depth_top=layer.depth_top,
            depth_bottom=layer.depth_bottom,
            depth_unit=layer.depth_unit,
            coarse_fraction=layer.coarse_fraction,
            coarse_unit=layer.coarse_unit,
            carbon_std=layer.carbon_std,
            bulk_density_std=layer.bulk_density_std,
            depth_std=layer.depth_std,
            coarse_std=layer.coarse_std,
            cov_matrix=layer.cov_matrix,
            label=layer.label,
        )
        for layer in parsed_layers
    ]

    ordered = sorted(
        layer_results,
        key=lambda res: (
            res.normalized_inputs.depth_top_cm,
            res.normalized_inputs.depth_bottom_cm,
        ),
    )

    for prev, curr in zip(ordered, ordered[1:]):
        prev_bottom = prev.normalized_inputs.depth_bottom_cm
        curr_top = curr.normalized_inputs.depth_top_cm
        if curr_top < prev_bottom - 1e-9:
            raise SchemaError(
                "Overlapping depth intervals are forbidden in profile carbon stock aggregation.",
                details={
                    "previous_interval_cm": [prev.normalized_inputs.depth_top_cm, prev_bottom],
                    "current_interval_cm": [curr_top, curr.normalized_inputs.depth_bottom_cm],
                },
            )
        if require_contiguous and abs(curr_top - prev_bottom) > 1e-6:
            raise SchemaError(
                "Non-contiguous depth intervals encountered while require_contiguous=True.",
                details={
                    "previous_bottom_cm": prev_bottom,
                    "current_top_cm": curr_top,
                },
            )

    total_depth_cm = float(sum(res.normalized_inputs.thickness_cm for res in ordered))
    collected_flags: list[QualityFlag] = []
    notes: list[str] = []
    for res in ordered:
        for flag in res.quality_flags:
            if flag not in collected_flags:
                collected_flags.append(flag)
        notes.extend(res.notes)

    if any(res.stock_mg_ha is None for res in ordered):
        return ProfileStockResult(
            total_stock_mg_ha=None,
            total_stock_std_mg_ha=None,
            total_depth_cm=total_depth_cm,
            layer_results=ordered,
            dominant_uncertainty_source=None,
            quality_status=QualityStatus.INSUFFICIENT_DATA,
            quality_flags=collected_flags,
            notes=notes,
        )

    total_stock = float(sum(res.stock_mg_ha for res in ordered if res.stock_mg_ha is not None))
    UnitRegistry.validate_physical_range("carbon_stock", total_stock)

    layers_with_unc = [res for res in ordered if res.uncertainty is not None]
    total_std: float | None = None
    dominant_source: str | None = None
    if layers_with_unc:
        total_var = float(sum(res.uncertainty.stock_variance_mg_ha2 for res in layers_with_unc))
        total_std = float(math.sqrt(max(0.0, total_var)))
        agg_contribs = {name: 0.0 for name in STOCK_PARAM_NAMES}
        for res in layers_with_unc:
            for name in STOCK_PARAM_NAMES:
                agg_contribs[name] += res.uncertainty.variance_contributions.get(name, 0.0)
        if sum(agg_contribs.values()) <= 0.0:
            dominant_source = "none"
        else:
            dominant_source = max(agg_contribs, key=lambda k: agg_contribs[k])

    return ProfileStockResult(
        total_stock_mg_ha=total_stock,
        total_stock_std_mg_ha=total_std,
        total_depth_cm=total_depth_cm,
        layer_results=ordered,
        dominant_uncertainty_source=dominant_source,
        quality_status=QualityStatus.WARNING if collected_flags else QualityStatus.OK,
        quality_flags=collected_flags,
        notes=notes,
    )
