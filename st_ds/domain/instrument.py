"""Spectral instrument profiles and wavelength grid contracts (Section 9, ADR-002, DEF-003).

Replaces hardcoded 24-channel assumptions with explicit, wavelength-aware instrument profiles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from st_ds.domain.errors import SchemaError
from st_ds.domain.units import UnitRegistry


class SpectralModality(str, Enum):
    """Physical sensing modality of the spectral instrument."""

    LED_DISCRETE = "led_discrete"
    MULTISPECTRAL_ARRAY = "multispectral_array"
    VIS_NIR_SWIR = "vis_nir_swir"
    FT_NIR = "ft_nir"
    LIBS = "libs"


@dataclass(frozen=True)
class InstrumentProfile:
    """Immutable specification of a spectral sensor's wavelength grid and capabilities."""

    profile_id: str
    sensor_model: str
    modality: SpectralModality
    wavelength_grid_nm: tuple[float, ...]
    channel_ids: tuple[str, ...]
    fwhm_nm: tuple[float, ...] | None = None
    calibration_file_id: str = "default_cal_v1"
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.profile_id or not self.profile_id.strip():
            raise SchemaError("InstrumentProfile.profile_id cannot be empty.")

        wl = np.asarray(self.wavelength_grid_nm, dtype=float)
        if wl.ndim != 1 or wl.size == 0:
            raise SchemaError(
                "InstrumentProfile.wavelength_grid_nm must be a non-empty 1D sequence.",
                details={"profile_id": self.profile_id},
            )
        if not np.all(np.isfinite(wl)):
            raise SchemaError(
                "InstrumentProfile.wavelength_grid_nm contains NaN or Inf values.",
                details={"profile_id": self.profile_id},
            )
        for val in wl:
            UnitRegistry.validate_physical_range("wavelength", float(val))

        if wl.size > 1 and not np.all(np.diff(wl) > 0):
            raise SchemaError(
                "InstrumentProfile.wavelength_grid_nm must be strictly increasing.",
                details={"profile_id": self.profile_id, "wavelength_grid_nm": list(self.wavelength_grid_nm)},
            )

        if len(self.channel_ids) != wl.size:
            raise SchemaError(
                "Length of channel_ids must match wavelength_grid_nm.",
                details={
                    "profile_id": self.profile_id,
                    "n_channels": len(self.channel_ids),
                    "n_wavelengths": int(wl.size),
                },
            )
        if len(set(self.channel_ids)) != len(self.channel_ids):
            raise SchemaError(
                "InstrumentProfile.channel_ids must be unique.",
                details={"profile_id": self.profile_id},
            )

        if self.fwhm_nm is not None:
            if len(self.fwhm_nm) != wl.size:
                raise SchemaError(
                    "Length of fwhm_nm must match wavelength_grid_nm.",
                    details={"profile_id": self.profile_id},
                )
            if any(f <= 0 or not np.isfinite(f) for f in self.fwhm_nm):
                raise SchemaError(
                    "All FWHM values must be positive finite numbers in nm.",
                    details={"profile_id": self.profile_id},
                )

    @property
    def n_channels(self) -> int:
        return len(self.wavelength_grid_nm)

    @property
    def spectral_range_nm(self) -> tuple[float, float]:
        return (float(self.wavelength_grid_nm[0]), float(self.wavelength_grid_nm[-1]))

    @property
    def is_uniform_grid(self) -> bool:
        if len(self.wavelength_grid_nm) <= 2:
            return True
        diffs = np.diff(np.asarray(self.wavelength_grid_nm, dtype=float))
        return bool(np.allclose(diffs, diffs[0], rtol=1e-4, atol=1e-4))

    def bands_in_window(self, min_nm: float, max_nm: float) -> list[int]:
        """Return indices of channels falling inside [min_nm, max_nm]."""
        return [
            idx
            for idx, wl in enumerate(self.wavelength_grid_nm)
            if min_nm <= wl <= max_nm
        ]

    @property
    def has_water_bands(self) -> bool:
        """Check coverage of primary O-H water absorption bands (1410-1450 nm or 1900-1950 nm)."""
        return bool(
            self.bands_in_window(1400.0, 1460.0) or self.bands_in_window(1890.0, 1960.0)
        )

    @property
    def has_clay_band(self) -> bool:
        """Check coverage of clay mineral Al-OH band (~2185-2225 nm)."""
        return bool(self.bands_in_window(2185.0, 2225.0))

    @property
    def has_organic_band(self) -> bool:
        """Check coverage of organic C-H / N-H functional group band (2130-2207 nm)."""
        return bool(self.bands_in_window(2130.0, 2207.0))

    @property
    def has_carbonate_bands(self) -> bool:
        """Check coverage of carbonate CO3 overtone bands (2310-2345 nm or 2500-2550 nm) or LIBS C I."""
        if self.modality == SpectralModality.LIBS:
            return bool(
                self.bands_in_window(192.5, 193.6)
                or self.bands_in_window(247.3, 248.3)
            )
        return bool(
            self.bands_in_window(2305.0, 2350.0) or self.bands_in_window(2495.0, 2555.0)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "sensor_model": self.sensor_model,
            "modality": self.modality.value,
            "n_channels": self.n_channels,
            "wavelength_grid_nm": list(self.wavelength_grid_nm),
            "channel_ids": list(self.channel_ids),
            "fwhm_nm": list(self.fwhm_nm) if self.fwhm_nm is not None else None,
            "spectral_range_nm": list(self.spectral_range_nm),
            "is_uniform_grid": self.is_uniform_grid,
            "has_water_bands": self.has_water_bands,
            "has_organic_band": self.has_organic_band,
            "has_carbonate_bands": self.has_carbonate_bands,
            "calibration_file_id": self.calibration_file_id,
            "description": self.description,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> InstrumentProfile:
        """Reconstruct an `InstrumentProfile` from a serialized dictionary."""
        try:
            fwhm_raw = data.get("fwhm_nm")
            return cls(
                profile_id=str(data["profile_id"]),
                sensor_model=str(data.get("sensor_model", data["profile_id"])),
                modality=SpectralModality(str(data.get("modality", "vis_nir_swir"))),
                wavelength_grid_nm=tuple(float(w) for w in data["wavelength_grid_nm"]),
                channel_ids=tuple(str(c) for c in data["channel_ids"]),
                fwhm_nm=tuple(float(f) for f in fwhm_raw) if fwhm_raw is not None else None,
                calibration_file_id=str(data.get("calibration_file_id", "default_cal_v1")),
                description=str(data.get("description", "")),
                metadata=dict(data.get("metadata") or {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SchemaError(
                f"Malformed InstrumentProfile payload: {exc}",
                details={"profile_id": data.get("profile_id") if isinstance(data, Mapping) else None},
            ) from exc


def _build_builtin_profiles() -> dict[str, InstrumentProfile]:
    # 1. 5-band discrete LED optical head (low-cost field probe)
    led_wl = (460.0, 660.0, 850.0, 940.0, 1450.0)
    led_profile = InstrumentProfile(
        profile_id="led_5band_v1",
        sensor_model="STDS-LED5-OpticalProbe",
        modality=SpectralModality.LED_DISCRETE,
        wavelength_grid_nm=led_wl,
        channel_ids=tuple(f"LED_{int(w)}nm" for w in led_wl),
        fwhm_nm=(25.0, 25.0, 35.0, 40.0, 60.0),
        description="5-band discrete LED reflectance probe (Vis-NIR + 1450nm moisture band)",
    )

    # 2. 18-channel multispectral sensor (410-940 nm)
    ms18_wl = (
        410.0, 435.0, 460.0, 485.0, 510.0, 535.0,
        560.0, 585.0, 610.0, 645.0, 680.0, 705.0,
        730.0, 760.0, 810.0, 860.0, 900.0, 940.0,
    )
    ms18_profile = InstrumentProfile(
        profile_id="multispectral_18ch_v1",
        sensor_model="AS7265x-18Ch-VisNIR",
        modality=SpectralModality.MULTISPECTRAL_ARRAY,
        wavelength_grid_nm=ms18_wl,
        channel_ids=tuple(f"MS18_{int(w)}nm" for w in ms18_wl),
        fwhm_nm=tuple(20.0 for _ in ms18_wl),
        description="18-channel Vis-NIR multispectral triad (410-940 nm)",
    )

    # 3. 24-channel Vis-SWIR array spanning key water, organic, clay and carbonate windows
    vis_swir_24_wl = (
        450.0, 550.0, 650.0, 750.0, 850.0, 950.0,
        1050.0, 1200.0, 1350.0, 1415.0, 1445.0, 1600.0,
        1720.0, 1850.0, 1915.0, 1945.0, 2050.0, 2145.0,
        2175.0, 2205.0, 2260.0, 2320.0, 2340.0, 2520.0,
    )
    vis_swir_24_profile = InstrumentProfile(
        profile_id="visnir_24ch_v1",
        sensor_model="STDS-InGaAs-24Ch",
        modality=SpectralModality.VIS_NIR_SWIR,
        wavelength_grid_nm=vis_swir_24_wl,
        channel_ids=tuple(f"CH_{idx + 1:02d}_{int(w)}nm" for idx, w in enumerate(vis_swir_24_wl)),
        fwhm_nm=tuple(15.0 for _ in vis_swir_24_wl),
        description="24-channel targeted Vis-SWIR array covering O-H, C-H, Al-OH, and CO3 bands",
    )

    # 4. 1300-2550 nm continuous MEMS FT-NIR (126 uniform channels at 10 nm spacing)
    ftnir_wl = tuple(float(w) for w in np.linspace(1300.0, 2550.0, 126))
    ftnir_profile = InstrumentProfile(
        profile_id="ftnir_1300_2550nm_v1",
        sensor_model="MEMS-FTNIR-1300-2550",
        modality=SpectralModality.FT_NIR,
        wavelength_grid_nm=ftnir_wl,
        channel_ids=tuple(f"FTNIR_{int(round(w))}nm" for w in ftnir_wl),
        fwhm_nm=tuple(8.0 for _ in ftnir_wl),
        description="126-channel uniform MEMS FT-NIR (1300-2550 nm, 10 nm step)",
    )

    # 5. LIBS spectral windows around C I (193.1, 229.7, 247.8 nm) and H I (656.3 nm)
    w_c1 = np.linspace(192.1, 194.1, 21)
    w_c2 = np.linspace(228.7, 230.7, 21)
    w_c3 = np.linspace(246.8, 248.8, 21)
    w_h1 = np.linspace(655.3, 657.3, 21)
    libs_wl = tuple(round(float(w), 3) for w in np.concatenate([w_c1, w_c2, w_c3, w_h1]))
    libs_profile = InstrumentProfile(
        profile_id="libs_ci_hi_v1",
        sensor_model="STDS-LIBS-UVVis",
        modality=SpectralModality.LIBS,
        wavelength_grid_nm=libs_wl,
        channel_ids=tuple(f"LIBS_{w:.2f}nm" for w in libs_wl),
        fwhm_nm=tuple(0.1 for _ in libs_wl),
        description="84-channel high-resolution LIBS windows for C I (193.1, 229.7, 247.8 nm) and H I (656.3 nm)",
    )

    return {
        p.profile_id: p
        for p in (led_profile, ms18_profile, vis_swir_24_profile, ftnir_profile, libs_profile)
    }


BUILTIN_PROFILES: dict[str, InstrumentProfile] = _build_builtin_profiles()


def get_builtin_profile(profile_id: str) -> InstrumentProfile:
    """Retrieve a built-in InstrumentProfile by ID or raise SchemaError."""
    if profile_id not in BUILTIN_PROFILES:
        raise SchemaError(
            f"Unknown built-in instrument profile '{profile_id}'.",
            details={"profile_id": profile_id, "available_profiles": list(BUILTIN_PROFILES.keys())},
        )
    return BUILTIN_PROFILES[profile_id]


def resolve_instrument_profile(profile_id_or_path: str | Path) -> InstrumentProfile:
    """Resolve either a built-in profile ID or a JSON file defining an `InstrumentProfile`."""
    raw = str(profile_id_or_path).strip()
    if raw in BUILTIN_PROFILES:
        return BUILTIN_PROFILES[raw]
    candidate = Path(raw)
    if candidate.is_file():
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        return InstrumentProfile.from_dict(payload)
    raise SchemaError(
        f"Instrument profile '{raw}' is neither a built-in profile ID nor an existing JSON file.",
        details={"profile_id_or_path": raw, "available_profiles": list(BUILTIN_PROFILES.keys())},
    )
