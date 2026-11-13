"""Canonical unit registry, unit conversion, and physical range validation (ADR-002, DEF-005).

Every physical quantity entering STDS is converted into a single canonical internal unit.
Ambiguous, missing, or unknown units raise UnitError (Fail-Closed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from st_ds.domain.errors import SchemaError, UnitError


@dataclass(frozen=True)
class QuantitySpec:
    """Specification of a physical quantity, its canonical unit, and valid physical range."""

    quantity_type: str
    canonical_unit: str
    min_value: float
    max_value: float
    description: str


class UnitRegistry:
    """Registry for converting physical quantities to canonical units and checking bounds."""

    SPECS: dict[str, QuantitySpec] = {
        "carbon_concentration": QuantitySpec(
            quantity_type="carbon_concentration",
            canonical_unit="pct",
            min_value=0.0,
            max_value=60.0,
            description="Dry-matter soil carbon concentration (% w/w, 0 to 60%)",
        ),
        "moisture_volumetric": QuantitySpec(
            quantity_type="moisture_volumetric",
            canonical_unit="pct",
            min_value=0.0,
            max_value=100.0,
            description="Volumetric water content theta_v (% v/v, 0 to 100%)",
        ),
        "moisture_gravimetric": QuantitySpec(
            quantity_type="moisture_gravimetric",
            canonical_unit="pct",
            min_value=0.0,
            max_value=200.0,
            description="Gravimetric water content (% g water / g dry soil)",
        ),
        "ph": QuantitySpec(
            quantity_type="ph",
            canonical_unit="dimensionless",
            min_value=2.0,
            max_value=12.0,
            description="Soil pH (dimensionless, 2.0 to 12.0)",
        ),
        "temperature": QuantitySpec(
            quantity_type="temperature",
            canonical_unit="degC",
            min_value=-40.0,
            max_value=85.0,
            description="Soil or sensor temperature in degrees Celsius (-40 to 85 C)",
        ),
        "ec": QuantitySpec(
            quantity_type="ec",
            canonical_unit="dS/m",
            min_value=0.0,
            max_value=200.0,
            description="Electrical conductivity in dS/m",
        ),
        "impedance": QuantitySpec(
            quantity_type="impedance",
            canonical_unit="ohm",
            min_value=-1e8,
            max_value=1e8,
            description="Complex impedance component (real or imaginary) in Ohms",
        ),
        "frequency": QuantitySpec(
            quantity_type="frequency",
            canonical_unit="Hz",
            min_value=0.01,
            max_value=1e9,
            description="EIS excitation frequency in Hz",
        ),
        "bulk_density": QuantitySpec(
            quantity_type="bulk_density",
            canonical_unit="g/cm3",
            min_value=0.2,
            max_value=2.65,
            description="Dry soil bulk density rho_b in g/cm3 (0.2 to 2.65 g/cm3)",
        ),
        "depth": QuantitySpec(
            quantity_type="depth",
            canonical_unit="cm",
            min_value=0.0,
            max_value=500.0,
            description="Soil sampling depth in cm (0 to 500 cm)",
        ),
        "coarse_fragments": QuantitySpec(
            quantity_type="coarse_fragments",
            canonical_unit="fraction",
            min_value=0.0,
            max_value=0.99,
            description="Coarse fragments (>2 mm gravel/stone) volume/mass fraction in [0, 0.99]",
        ),
        "wavelength": QuantitySpec(
            quantity_type="wavelength",
            canonical_unit="nm",
            min_value=150.0,
            max_value=25000.0,
            description="Spectral wavelength in nanometers (nm)",
        ),
        "carbon_stock": QuantitySpec(
            quantity_type="carbon_stock",
            canonical_unit="Mg_C_ha",
            min_value=0.0,
            max_value=5000.0,
            description="Soil carbon stock per unit area in Mg C ha^-1 (tonnes C / ha)",
        ),
    }

    _CONVERTERS: dict[str, dict[str, Callable[[float], float]]] = {
        "carbon_concentration": {
            "pct": lambda v: v,
            "%": lambda v: v,
            "percent": lambda v: v,
            "pct_dry": lambda v: v,
            "g/kg": lambda v: v * 0.1,
            "g_kg": lambda v: v * 0.1,
            "mg/kg": lambda v: v * 1e-4,
            "ppm": lambda v: v * 1e-4,
            "g/g": lambda v: v * 100.0,
            "fraction": lambda v: v * 100.0,
        },
        "moisture_volumetric": {
            "pct": lambda v: v,
            "%": lambda v: v,
            "m3/m3": lambda v: v * 100.0,
            "cm3/cm3": lambda v: v * 100.0,
            "v/v": lambda v: v * 100.0,
            "fraction": lambda v: v * 100.0,
        },
        "moisture_gravimetric": {
            "pct": lambda v: v,
            "%": lambda v: v,
            "g/g": lambda v: v * 100.0,
            "kg/kg": lambda v: v * 100.0,
            "fraction": lambda v: v * 100.0,
        },
        "ph": {
            "dimensionless": lambda v: v,
            "ph": lambda v: v,
            "unitless": lambda v: v,
        },
        "temperature": {
            "degc": lambda v: v,
            "c": lambda v: v,
            "celsius": lambda v: v,
            "k": lambda v: v - 273.15,
            "kelvin": lambda v: v - 273.15,
            "degf": lambda v: (v - 32.0) * (5.0 / 9.0),
            "f": lambda v: (v - 32.0) * (5.0 / 9.0),
            "fahrenheit": lambda v: (v - 32.0) * (5.0 / 9.0),
        },
        "ec": {
            "ds/m": lambda v: v,
            "ms/cm": lambda v: v,
            "s/m": lambda v: v * 10.0,
            "us/cm": lambda v: v * 1e-3,
            "ms/m": lambda v: v * 0.01,
        },
        "impedance": {
            "ohm": lambda v: v,
            "ohms": lambda v: v,
            "ω": lambda v: v,
            "kohm": lambda v: v * 1000.0,
            "mohm": lambda v: v * 1e6,
        },
        "frequency": {
            "hz": lambda v: v,
            "khz": lambda v: v * 1e3,
            "mhz": lambda v: v * 1e6,
        },
        "bulk_density": {
            "g/cm3": lambda v: v,
            "g_cm3": lambda v: v,
            "mg/m3": lambda v: v,
            "t/m3": lambda v: v,
            "kg/m3": lambda v: v * 1e-3,
            "kg_m3": lambda v: v * 1e-3,
        },
        "depth": {
            "cm": lambda v: v,
            "m": lambda v: v * 100.0,
            "mm": lambda v: v * 0.1,
        },
        "coarse_fragments": {
            "fraction": lambda v: v,
            "g/g": lambda v: v,
            "v/v": lambda v: v,
            "pct": lambda v: v * 0.01,
            "%": lambda v: v * 0.01,
        },
        "wavelength": {
            "nm": lambda v: v,
            "nanometer": lambda v: v,
            "um": lambda v: v * 1000.0,
            "µm": lambda v: v * 1000.0,
            "micrometer": lambda v: v * 1000.0,
        },
        "carbon_stock": {
            "mg_c_ha": lambda v: v,
            "mg/ha": lambda v: v,
            "t/ha": lambda v: v,
            "t_c_ha": lambda v: v,
            "kg/m2": lambda v: v * 10.0,
            "kg_c_m2": lambda v: v * 10.0,
            "g/m2": lambda v: v * 0.01,
        },
    }

    @classmethod
    def normalize_unit_key(cls, unit: str | None) -> str:
        if unit is None or not isinstance(unit, str) or not unit.strip():
            raise UnitError(
                "Physical unit is missing or empty; implicit units are forbidden (ADR-002).",
                details={"received_unit": unit},
            )
        return unit.strip().lower()

    @classmethod
    def to_canonical(
        cls,
        quantity_type: str,
        value: float,
        unit: str | None,
        *,
        validate_bounds: bool = True,
    ) -> float:
        """Convert `value` from `unit` into the canonical unit for `quantity_type`."""
        if quantity_type not in cls.SPECS:
            raise UnitError(
                f"Unknown quantity_type '{quantity_type}'.",
                details={"quantity_type": quantity_type, "supported": list(cls.SPECS.keys())},
            )

        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SchemaError(
                f"Value for '{quantity_type}' must be a finite real number.",
                details={"quantity_type": quantity_type, "value": value},
            )

        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise SchemaError(
                f"Non-finite value (NaN/Inf) encountered for '{quantity_type}'.",
                details={"quantity_type": quantity_type, "value": value},
            )

        unit_key = cls.normalize_unit_key(unit)
        converters = cls._CONVERTERS[quantity_type]
        if unit_key not in converters:
            raise UnitError(
                f"Unsupported unit '{unit}' for quantity_type '{quantity_type}'.",
                details={
                    "quantity_type": quantity_type,
                    "received_unit": unit,
                    "supported_units": list(converters.keys()),
                    "canonical_unit": cls.SPECS[quantity_type].canonical_unit,
                },
            )

        converted = float(converters[unit_key](numeric_value))
        if validate_bounds:
            cls.validate_physical_range(quantity_type, converted)
        return converted

    @classmethod
    def validate_physical_range(cls, quantity_type: str, canonical_value: float) -> None:
        """Verify that a canonical value lies within physically plausible bounds."""
        spec = cls.SPECS[quantity_type]
        if not math.isfinite(canonical_value):
            raise SchemaError(
                f"Converted value for '{quantity_type}' is not finite.",
                details={"quantity_type": quantity_type, "value": canonical_value},
            )
        if canonical_value < spec.min_value or canonical_value > spec.max_value:
            raise SchemaError(
                f"Value {canonical_value:.6g} {spec.canonical_unit} for '{quantity_type}' "
                f"is outside physical bounds [{spec.min_value}, {spec.max_value}] {spec.canonical_unit}.",
                details={
                    "quantity_type": quantity_type,
                    "canonical_value": canonical_value,
                    "canonical_unit": spec.canonical_unit,
                    "min_value": spec.min_value,
                    "max_value": spec.max_value,
                },
            )

    @classmethod
    def canonical_unit_for(cls, quantity_type: str) -> str:
        if quantity_type not in cls.SPECS:
            raise UnitError(f"Unknown quantity_type '{quantity_type}'.")
        return cls.SPECS[quantity_type].canonical_unit
