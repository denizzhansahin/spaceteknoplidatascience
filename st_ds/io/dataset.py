"""Dataset management, lineage validation, source isolation, and CSV/JSON adapters (Section 31, Page 33).

Enforces:
- Strict source separation: 'synthetic' records can never be silently mixed with 'real_lab' records.
- Sample/replicate integrity and deterministic SHA-256 dataset hashing.
- Complete lineage checking before training admission.
- Round-trip CSV and JSON serialization with unit and wavelength preservation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from st_ds import SCHEMA_VERSION
from st_ds.domain.errors import SchemaError
from st_ds.domain.instrument import BUILTIN_PROFILES, InstrumentProfile
from st_ds.domain.quality import QualityFlag, QualityMetadata, QualityStatus
from st_ds.domain.schemas import (
    CovariateSet,
    DatasetSourceType,
    DepthInterval,
    EISMeasurement,
    Measurement,
    SoilReference,
    SpatialCoordinates,
    SpectralSignalMode,
    Spectrum,
)


@dataclass(frozen=True)
class DatasetManifest:
    """Immutable metadata and provenance manifest for a SoilDataset."""

    dataset_id: str
    dataset_source: DatasetSourceType
    instrument_profile_id: str
    n_records: int
    n_unique_samples: int
    n_unique_sites: int
    schema_version: str
    sha256_hash: str
    created_utc: str = "2026-11-12T00:00:00Z"
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_source": self.dataset_source,
            "instrument_profile_id": self.instrument_profile_id,
            "n_records": self.n_records,
            "n_unique_samples": self.n_unique_samples,
            "n_unique_sites": self.n_unique_sites,
            "schema_version": self.schema_version,
            "sha256_hash": self.sha256_hash,
            "created_utc": self.created_utc,
            "description": self.description,
        }


@dataclass
class SoilDataset:
    """Validated collection of canonical Measurement objects with source isolation guard."""

    dataset_id: str
    records: list[Measurement]
    dataset_source: DatasetSourceType = "real_lab"
    description: str = ""
    allow_mixed_profiles: bool = False

    def __post_init__(self) -> None:
        if not self.dataset_id or not self.dataset_id.strip():
            raise SchemaError("SoilDataset.dataset_id cannot be empty.")
        if len(self.records) == 0:
            raise SchemaError("SoilDataset.records cannot be empty.")
        if self.dataset_source not in ("real_lab", "synthetic"):
            raise SchemaError(
                f"Invalid dataset_source '{self.dataset_source}'. Must be 'real_lab' or 'synthetic'."
            )

        # Enforce strict source isolation (Section 31: prevent mixing synthetic with real_lab)
        record_sources = {r.dataset_source for r in self.records}
        if len(record_sources) > 1 or next(iter(record_sources)) != self.dataset_source:
            raise SchemaError(
                f"Dataset source mismatch or contamination detected! Dataset declared as "
                f"'{self.dataset_source}', but records contain sources {sorted(record_sources)}. "
                "Mixing synthetic and real laboratory data is strictly forbidden.",
                details={
                    "declared_source": self.dataset_source,
                    "found_sources": sorted(record_sources),
                },
            )

        # Check instrument profile consistency unless explicitly multi-profile
        profiles = {r.instrument_profile_id for r in self.records}
        if not self.allow_mixed_profiles and len(profiles) > 1:
            raise SchemaError(
                f"Multiple instrument profiles found in dataset: {sorted(profiles)}. "
                "Set allow_mixed_profiles=True only when using validated wavelength harmonization.",
                details={"profiles": sorted(profiles)},
            )

        # Check sample_id + replicate_id + moisture_scenario uniqueness
        seen_keys: set[tuple[str, str, str]] = set()
        for r in self.records:
            key = (r.sample_id, r.spectrum.replicate_id, r.moisture_scenario)
            if key in seen_keys:
                raise SchemaError(
                    f"Duplicate record key (sample_id='{r.sample_id}', "
                    f"replicate_id='{r.spectrum.replicate_id}', "
                    f"moisture_scenario='{r.moisture_scenario}') in dataset.",
                    details={"duplicate_key": key},
                )
            seen_keys.add(key)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def instrument_profile_id(self) -> str:
        return self.records[0].instrument_profile_id

    @property
    def unique_sample_ids(self) -> list[str]:
        return sorted({r.sample_id for r in self.records})

    @property
    def unique_site_ids(self) -> list[str]:
        return sorted({r.site_id for r in self.records})

    def compute_sha256(self) -> str:
        """Compute a deterministic SHA-256 digest of all records in the dataset."""
        hasher = hashlib.sha256()
        for r in self.records:
            payload = json.dumps(r.to_dict(), sort_keys=True, ensure_ascii=True).encode("utf-8")
            hasher.update(payload)
        return hasher.hexdigest()

    def build_manifest(self) -> DatasetManifest:
        return DatasetManifest(
            dataset_id=self.dataset_id,
            dataset_source=self.dataset_source,
            instrument_profile_id=self.instrument_profile_id,
            n_records=len(self.records),
            n_unique_samples=len(self.unique_sample_ids),
            n_unique_sites=len(self.unique_site_ids),
            schema_version=SCHEMA_VERSION,
            sha256_hash=self.compute_sha256(),
            description=self.description,
        )

    def validate_for_training(self, *, require_sic: bool = True) -> None:
        """Verify that every record has complete lineage and reference labels required for training."""
        for idx, r in enumerate(self.records):
            if r.site_id == "unknown_site" or not r.site_id.strip():
                raise SchemaError(
                    f"Record {idx} (sample_id='{r.sample_id}') lacks valid site_id lineage required for grouped splitting.",
                    details={"sample_id": r.sample_id},
                )
            if r.reference is None or r.reference.soc_pct is None:
                raise SchemaError(
                    f"Record {idx} (sample_id='{r.sample_id}') lacks reference SOC label.",
                    details={"sample_id": r.sample_id},
                )
            if require_sic and r.reference.sic_pct is None:
                raise SchemaError(
                    f"Record {idx} (sample_id='{r.sample_id}') lacks reference SIC label.",
                    details={"sample_id": r.sample_id},
                )
            if not r.reference.soc_method:
                raise SchemaError(
                    f"Record {idx} (sample_id='{r.sample_id}') lacks reference laboratory method provenance.",
                    details={"sample_id": r.sample_id},
                )


def generate_data_quality_report(dataset: SoilDataset) -> dict[str, Any]:
    """Generate a Dataset Card and Data Quality Report (Section 31, Page 33)."""
    manifest = dataset.build_manifest()
    soc_vals = [r.reference.soc_pct for r in dataset.records if r.reference and r.reference.soc_pct is not None]
    sic_vals = [r.reference.sic_pct for r in dataset.records if r.reference and r.reference.sic_pct is not None]
    tc_vals = [r.reference.tc_pct for r in dataset.records if r.reference and r.reference.tc_pct is not None]
    missing_bd_count = sum(1 for r in dataset.records if not r.covariates.has_bulk_density)

    def _stats(arr: list[float]) -> dict[str, float | None]:
        if not arr:
            return {"min": None, "max": None, "mean": None, "std": None, "median": None, "iqr": None}
        a = np.asarray(arr, dtype=float)
        q75, q25 = np.percentile(a, [75, 25])
        return {
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "median": float(np.median(a)),
            "iqr": float(q75 - q25),
        }

    return {
        "manifest": manifest.to_dict(),
        "is_synthetic": dataset.dataset_source == "synthetic",
        "site_distribution": {
            site: sum(1 for r in dataset.records if r.site_id == site)
            for site in dataset.unique_site_ids
        },
        "moisture_scenarios": {
            scen: sum(1 for r in dataset.records if r.moisture_scenario == scen)
            for scen in sorted({r.moisture_scenario for r in dataset.records})
        },
        "missing_bulk_density_count": missing_bd_count,
        "target_distributions_pct": {
            "soc_pct": _stats(soc_vals),
            "sic_pct": _stats(sic_vals),
            "tc_pct": _stats(tc_vals),
        },
    }


def build_dataset_card(dataset: SoilDataset) -> dict[str, Any]:
    """Build a structured Dataset Card summarizing lineage, coverage, and limitations (Section 31)."""
    dq = generate_data_quality_report(dataset)
    textures = sorted({r.soil_texture_class for r in dataset.records if r.soil_texture_class})
    preps = sorted({r.sample_preparation for r in dataset.records})
    soc_methods = sorted({r.reference.soc_method for r in dataset.records if r.reference and r.reference.soc_method})
    sic_methods = sorted({r.reference.sic_method for r in dataset.records if r.reference and r.reference.sic_method})
    return {
        "dataset_card_version": "1.0.0",
        "manifest": dq["manifest"],
        "is_synthetic": dq["is_synthetic"],
        "intended_use": (
            "Software pipeline verification and smoke testing only; NOT field evidence."
            if dq["is_synthetic"]
            else "Calibration, validation, and model card reporting on verified laboratory samples."
        ),
        "lineage": {
            "site_count": len(dataset.unique_site_ids),
            "site_distribution": dq["site_distribution"],
            "texture_classes": textures,
            "sample_preparations": preps,
            "moisture_scenarios": dq["moisture_scenarios"],
            "soc_methods": soc_methods,
            "sic_methods": sic_methods,
        },
        "data_quality": {
            "missing_bulk_density_count": dq["missing_bulk_density_count"],
            "target_distributions_pct": dq["target_distributions_pct"],
        },
    }


def save_dataset_json(dataset: SoilDataset, path: str | Path) -> Path:
    """Save a SoilDataset and its manifest to a JSON file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": dataset.build_manifest().to_dict(),
        "records": [r.to_dict() for r in dataset.records],
    }
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def measurement_from_dict(item: Mapping[str, Any]) -> Measurement:
    """Reconstruct and validate a canonical `Measurement` from a serialized dict."""
    try:
        spec_d = item["spectrum"]
        spectrum = Spectrum(
            values=np.asarray(spec_d["values"], dtype=float),
            wavelength_grid_nm=tuple(float(w) for w in spec_d["wavelength_grid_nm"]),
            channel_ids=tuple(str(c) for c in spec_d["channel_ids"]),
            mode=SpectralSignalMode(spec_d["mode"]),
            dark_reference=tuple(spec_d["dark_reference"]) if spec_d.get("dark_reference") is not None else None,
            white_reference=tuple(spec_d["white_reference"]) if spec_d.get("white_reference") is not None else None,
            integration_time_ms=spec_d.get("integration_time_ms"),
            gain=spec_d.get("gain"),
            sensor_temperature_c=spec_d.get("sensor_temperature_c"),
            reference_age_s=spec_d.get("reference_age_s"),
            replicate_id=spec_d.get("replicate_id", "rep_1"),
            saturation_flags=tuple(spec_d["saturation_flags"]) if spec_d.get("saturation_flags") is not None else None,
            valid_mask=tuple(spec_d["valid_mask"]) if spec_d.get("valid_mask") is not None else None,
        )
        cov_d = item["covariates"]
        eis_obj = None
        if cov_d.get("eis") is not None:
            eis_d = cov_d["eis"]
            eis_obj = EISMeasurement(
                frequencies_hz=tuple(float(f) for f in eis_d["frequencies_hz"]),
                z_real_ohm=tuple(float(r) for r in eis_d["z_real_ohm"]),
                z_imag_ohm=tuple(float(im) for im in eis_d["z_imag_ohm"]),
                protocol_id=eis_d.get("protocol_id", "eis_sweep_v1"),
            )
        covariates = CovariateSet(
            ph=cov_d.get("ph"),
            moisture_volumetric_pct=cov_d.get("moisture_volumetric_pct"),
            moisture_gravimetric_pct=cov_d.get("moisture_gravimetric_pct"),
            temperature_c=cov_d.get("temperature_c"),
            ec_ds_m=cov_d.get("ec_ds_m"),
            bulk_density_g_cm3=cov_d.get("bulk_density_g_cm3"),
            bulk_density_uncertainty_g_cm3=cov_d.get("bulk_density_uncertainty_g_cm3"),
            bulk_density_source=cov_d.get("bulk_density_source"),
            coarse_fragments_fraction=cov_d.get("coarse_fragments_fraction"),
            eis=eis_obj,
            missing_reasons=dict(cov_d.get("missing_reasons", {})),
        )
        ref_obj = None
        if item.get("reference") is not None:
            ref_d = item["reference"]
            ref_obj = SoilReference(
                soc_pct=ref_d.get("soc_pct"),
                sic_pct=ref_d.get("sic_pct"),
                tc_pct=ref_d.get("tc_pct"),
                dry_matter_basis=ref_d.get("dry_matter_basis", True),
                soc_method=ref_d.get("soc_method"),
                sic_method=ref_d.get("sic_method"),
                tc_method=ref_d.get("tc_method"),
                is_soc_derived_from_difference=ref_d.get("is_soc_derived_from_difference", False),
                lab_uncertainty_soc_pct=ref_d.get("lab_uncertainty_soc_pct"),
                lab_uncertainty_sic_pct=ref_d.get("lab_uncertainty_sic_pct"),
                lab_uncertainty_tc_pct=ref_d.get("lab_uncertainty_tc_pct"),
                soil_class_id=ref_d.get("soil_class_id"),
                soil_class_name=ref_d.get("soil_class_name"),
                taxonomy_version=ref_d.get("taxonomy_version", "provisional_taxonomy_v0"),
            )
        depth_d = item["depth_interval"]
        coord_obj = None
        if item.get("coordinates") is not None:
            c_d = item["coordinates"]
            coord_obj = SpatialCoordinates(
                lat=c_d["lat"],
                lon=c_d["lon"],
                crs_epsg=c_d.get("crs_epsg", "EPSG:4326"),
                elevation_m=c_d.get("elevation_m"),
            )
        qual_d = item.get("quality", {})
        quality = QualityMetadata(
            status=QualityStatus(qual_d.get("status", "ok")),
            quality_score=float(qual_d.get("quality_score", 1.0)),
            flags=[QualityFlag(f) for f in qual_d.get("flags", [])],
            rejected_bands=list(qual_d.get("rejected_bands", [])),
            rejected_shots_count=int(qual_d.get("rejected_shots_count", 0)),
            missing_covariates=dict(qual_d.get("missing_covariates", {})),
            notes=list(qual_d.get("notes", [])),
        )
        return Measurement(
            sample_id=item["sample_id"],
            site_id=item.get("site_id", "unknown_site"),
            session_id=item.get("session_id", "session_1"),
            timestamp_utc=item["timestamp_utc"],
            instrument_id=item["instrument_id"],
            instrument_profile_id=item["instrument_profile_id"],
            calibration_id=item["calibration_id"],
            soil_texture_class=item.get("soil_texture_class"),
            sample_preparation=item.get("sample_preparation", "air_dried_sieved_2mm"),
            moisture_scenario=item.get("moisture_scenario", "dry"),
            depth_interval=DepthInterval(depth_d["top_cm"], depth_d["bottom_cm"]),
            coordinates=coord_obj,
            spectrum=spectrum,
            covariates=covariates,
            reference=ref_obj,
            dataset_source=item["dataset_source"],
            quality=quality,
            schema_version=item.get("schema_version", SCHEMA_VERSION),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SchemaError(
            f"Malformed Measurement payload: {exc}",
            details={"sample_id": item.get("sample_id") if isinstance(item, Mapping) else None},
        ) from exc


def load_dataset_json(path: str | Path) -> SoilDataset:
    """Load and validate a SoilDataset from a JSON file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    manifest = raw["manifest"]
    records = [measurement_from_dict(item) for item in raw["records"]]
    dataset = SoilDataset(
        dataset_id=manifest["dataset_id"],
        records=records,
        dataset_source=manifest["dataset_source"],
        description=manifest.get("description", ""),
    )
    if dataset.compute_sha256() != manifest["sha256_hash"]:
        raise SchemaError(
            "Dataset SHA-256 integrity check failed on load_dataset_json!",
            details={"expected": manifest["sha256_hash"], "actual": dataset.compute_sha256()},
        )
    return dataset


def _dataset_to_dataframe(dataset: SoilDataset) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    first_spec = dataset.records[0].spectrum
    wl_cols = [f"wl_{w:.2f}nm" for w in first_spec.wavelength_grid_nm]

    for r in dataset.records:
        if r.spectrum.values.ndim != 1:
            raise SchemaError("Tabular export requires 1D aggregated spectra; use JSON for 2D multi-shot LIBS.")
        row: dict[str, Any] = {
            "sample_id": r.sample_id,
            "site_id": r.site_id,
            "session_id": r.session_id,
            "replicate_id": r.spectrum.replicate_id,
            "timestamp_utc": r.timestamp_utc,
            "instrument_id": r.instrument_id,
            "instrument_profile_id": r.instrument_profile_id,
            "calibration_id": r.calibration_id,
            "soil_texture_class": r.soil_texture_class,
            "sample_preparation": r.sample_preparation,
            "moisture_scenario": r.moisture_scenario,
            "dataset_source": r.dataset_source,
            "depth_top_cm": r.depth_interval.top_cm,
            "depth_bottom_cm": r.depth_interval.bottom_cm,
            "lat": r.coordinates.lat if r.coordinates else None,
            "lon": r.coordinates.lon if r.coordinates else None,
            "crs_epsg": r.coordinates.crs_epsg if r.coordinates else None,
            "ph": r.covariates.ph,
            "moisture_volumetric_pct": r.covariates.moisture_volumetric_pct,
            "moisture_gravimetric_pct": r.covariates.moisture_gravimetric_pct,
            "temperature_c": r.covariates.temperature_c,
            "ec_ds_m": r.covariates.ec_ds_m,
            "bulk_density_g_cm3": r.covariates.bulk_density_g_cm3,
            "bulk_density_source": r.covariates.bulk_density_source,
            "coarse_fragments_fraction": r.covariates.coarse_fragments_fraction,
            "soc_pct": r.reference.soc_pct if r.reference else None,
            "sic_pct": r.reference.sic_pct if r.reference else None,
            "tc_pct": r.reference.tc_pct if r.reference else None,
            "soc_method": r.reference.soc_method if r.reference else None,
            "sic_method": r.reference.sic_method if r.reference else None,
            "tc_method": r.reference.tc_method if r.reference else None,
            "soil_class_id": r.reference.soil_class_id if r.reference else None,
            "soil_class_name": r.reference.soil_class_name if r.reference else None,
        }
        for col_name, val in zip(wl_cols, r.spectrum.values):
            row[col_name] = float(val)
        rows.append(row)

    return pd.DataFrame(rows), wl_cols


def _dataframe_to_dataset(
    df: pd.DataFrame,
    *,
    dataset_id: str,
    declared_source: DatasetSourceType,
    wl_grid: tuple[float, ...],
    ch_ids: tuple[str, ...],
    spec_cols: list[str],
    mode: SpectralSignalMode,
) -> SoilDataset:
    def _opt_float(val: Any) -> float | None:
        if val is None or (isinstance(val, float) and np.isnan(val)) or pd.isna(val):
            return None
        return float(val)

    def _opt_str(val: Any) -> str | None:
        if val is None or pd.isna(val):
            return None
        return str(val)

    records: list[Measurement] = []
    for _, row in df.iterrows():
        spec_vals = np.asarray([float(row[c]) for c in spec_cols], dtype=float)
        spectrum = Spectrum(
            values=spec_vals,
            wavelength_grid_nm=wl_grid,
            channel_ids=ch_ids,
            mode=mode,
            replicate_id=str(row.get("replicate_id", "rep_1")),
        )
        covariates = CovariateSet(
            ph=_opt_float(row.get("ph")),
            moisture_volumetric_pct=_opt_float(row.get("moisture_volumetric_pct")),
            moisture_gravimetric_pct=_opt_float(row.get("moisture_gravimetric_pct")),
            temperature_c=_opt_float(row.get("temperature_c")),
            ec_ds_m=_opt_float(row.get("ec_ds_m")),
            bulk_density_g_cm3=_opt_float(row.get("bulk_density_g_cm3")),
            bulk_density_source=_opt_str(row.get("bulk_density_source")),  # type: ignore[arg-type]
            coarse_fragments_fraction=_opt_float(row.get("coarse_fragments_fraction")),
        )
        soc = _opt_float(row.get("soc_pct"))
        sic = _opt_float(row.get("sic_pct"))
        tc = _opt_float(row.get("tc_pct"))
        ref = None
        if soc is not None or sic is not None or tc is not None:
            cls_id_raw = row.get("soil_class_id")
            cls_id = int(cls_id_raw) if cls_id_raw is not None and not pd.isna(cls_id_raw) else None
            ref = SoilReference(
                soc_pct=soc,
                sic_pct=sic,
                tc_pct=tc,
                soc_method=_opt_str(row.get("soc_method")) or "dry_combustion_acid_pretreated_iso10694",
                sic_method=_opt_str(row.get("sic_method")) or "calcimeter_iso10693",
                tc_method=_opt_str(row.get("tc_method")) or "dry_combustion_dumas_iso10694",
                soil_class_id=cls_id,
                soil_class_name=_opt_str(row.get("soil_class_name")),
            )
        lat = _opt_float(row.get("lat"))
        lon = _opt_float(row.get("lon"))
        coords = (
            SpatialCoordinates(lat=lat, lon=lon, crs_epsg=_opt_str(row.get("crs_epsg")) or "EPSG:4326")
            if lat is not None and lon is not None
            else None
        )
        records.append(
            Measurement(
                sample_id=str(row["sample_id"]),
                site_id=str(row["site_id"]),
                session_id=str(row.get("session_id", "session_1")),
                timestamp_utc=str(row["timestamp_utc"]),
                instrument_id=str(row["instrument_id"]),
                instrument_profile_id=str(row["instrument_profile_id"]),
                calibration_id=str(row["calibration_id"]),
                soil_texture_class=_opt_str(row.get("soil_texture_class")),
                sample_preparation=_opt_str(row.get("sample_preparation")) or "air_dried_sieved_2mm",
                moisture_scenario=str(row.get("moisture_scenario", "dry")),  # type: ignore[arg-type]
                dataset_source=str(row["dataset_source"]),  # type: ignore[arg-type]
                depth_interval=DepthInterval(
                    float(row.get("depth_top_cm", 0.0)),
                    float(row.get("depth_bottom_cm", 30.0)),
                ),
                coordinates=coords,
                spectrum=spectrum,
                covariates=covariates,
                reference=ref,
            )
        )

    return SoilDataset(
        dataset_id=dataset_id,
        records=records,
        dataset_source=declared_source,
    )


def save_dataset_csv(dataset: SoilDataset, path: str | Path) -> Path:
    """Export a 1D-spectral SoilDataset to a flat CSV file plus a companion `.meta.json` sidecar."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    first_spec = dataset.records[0].spectrum
    df, wl_cols = _dataset_to_dataframe(dataset)
    df.to_csv(target, index=False)

    sidecar = target.with_suffix(target.suffix + ".meta.json")
    sidecar_payload = {
        "manifest": dataset.build_manifest().to_dict(),
        "wavelength_grid_nm": list(first_spec.wavelength_grid_nm),
        "channel_ids": list(first_spec.channel_ids),
        "spectral_mode": first_spec.mode.value,
        "spectral_columns": wl_cols,
    }
    sidecar.write_text(json.dumps(sidecar_payload, indent=2), encoding="utf-8")
    return target


def load_dataset_csv(
    path: str | Path,
    *,
    profile: InstrumentProfile | None = None,
) -> SoilDataset:
    """Load a SoilDataset from a CSV file and its companion `.meta.json` sidecar (or explicit profile)."""
    csv_path = Path(path)
    sidecar = csv_path.with_suffix(csv_path.suffix + ".meta.json")
    df = pd.read_csv(csv_path)

    if sidecar.exists():
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        wl_grid = tuple(float(w) for w in meta["wavelength_grid_nm"])
        ch_ids = tuple(str(c) for c in meta["channel_ids"])
        spec_cols = list(meta["spectral_columns"])
        mode = SpectralSignalMode(meta.get("spectral_mode", "reflectance"))
        dataset_id = meta["manifest"]["dataset_id"]
        declared_source: DatasetSourceType = meta["manifest"]["dataset_source"]
    elif profile is not None:
        wl_grid = profile.wavelength_grid_nm
        ch_ids = profile.channel_ids
        spec_cols = [f"wl_{w:.2f}nm" for w in wl_grid]
        mode = SpectralSignalMode.REFLECTANCE
        dataset_id = csv_path.stem
        declared_source = str(df["dataset_source"].iloc[0])  # type: ignore[assignment]
    else:
        raise SchemaError(
            "Loading CSV requires either a '.meta.json' sidecar or an explicit InstrumentProfile."
        )

    return _dataframe_to_dataset(
        df,
        dataset_id=dataset_id,
        declared_source=declared_source,
        wl_grid=wl_grid,
        ch_ids=ch_ids,
        spec_cols=spec_cols,
        mode=mode,
    )


def save_dataset_parquet(dataset: SoilDataset, path: str | Path) -> Path:
    """Export a 1D-spectral SoilDataset to Parquet (requires pyarrow) + `.meta.json` sidecar."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    first_spec = dataset.records[0].spectrum
    df, wl_cols = _dataset_to_dataframe(dataset)
    df.to_parquet(target, index=False)

    sidecar = target.with_suffix(target.suffix + ".meta.json")
    sidecar_payload = {
        "manifest": dataset.build_manifest().to_dict(),
        "wavelength_grid_nm": list(first_spec.wavelength_grid_nm),
        "channel_ids": list(first_spec.channel_ids),
        "spectral_mode": first_spec.mode.value,
        "spectral_columns": wl_cols,
    }
    sidecar.write_text(json.dumps(sidecar_payload, indent=2), encoding="utf-8")
    return target


def load_dataset_parquet(
    path: str | Path,
    *,
    profile: InstrumentProfile | None = None,
) -> SoilDataset:
    """Load a SoilDataset from a Parquet file and its `.meta.json` sidecar."""
    pq_path = Path(path)
    sidecar = pq_path.with_suffix(pq_path.suffix + ".meta.json")
    df = pd.read_parquet(pq_path)

    if sidecar.exists():
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        wl_grid = tuple(float(w) for w in meta["wavelength_grid_nm"])
        ch_ids = tuple(str(c) for c in meta["channel_ids"])
        spec_cols = list(meta["spectral_columns"])
        mode = SpectralSignalMode(meta.get("spectral_mode", "reflectance"))
        dataset_id = meta["manifest"]["dataset_id"]
        declared_source: DatasetSourceType = meta["manifest"]["dataset_source"]
    elif profile is not None:
        wl_grid = profile.wavelength_grid_nm
        ch_ids = profile.channel_ids
        spec_cols = [f"wl_{w:.2f}nm" for w in wl_grid]
        mode = SpectralSignalMode.REFLECTANCE
        dataset_id = pq_path.stem
        declared_source = str(df["dataset_source"].iloc[0])  # type: ignore[assignment]
    else:
        raise SchemaError(
            "Loading Parquet requires either a '.meta.json' sidecar or an explicit InstrumentProfile."
        )

    return _dataframe_to_dataset(
        df,
        dataset_id=dataset_id,
        declared_source=declared_source,
        wl_grid=wl_grid,
        ch_ids=ch_ids,
        spec_cols=spec_cols,
        mode=mode,
    )
