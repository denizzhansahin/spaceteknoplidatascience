"""Physics-informed Synthetic Generator v2 (Section 31, Page 33).

Replaces legacy unverified `synthetic_csv_builder.py` with a deterministic, parameterized
generator that models:
- Water O-H overtone & combination bands (~1430 nm, ~1920 nm) driven by volumetric moisture
- Clay Al-OH band (~2200 nm)
- Organic C-H / N-H functional group band (~2170 nm + visible darkening) driven by SOC
- Carbonate CO3 overtone bands (~2335 nm, ~2520 nm) driven by SIC
- Strict physical closure: TC = SOC + SIC
- Paired wet/dry calibration sample generation for EPO (Faz 4) testing
- Mandatory `dataset_source="synthetic"` and `QualityFlag.SYNTHETIC_DATA_SOURCE` tagging
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from st_ds.domain.instrument import InstrumentProfile, get_builtin_profile
from st_ds.domain.schemas import (
    CovariateSet,
    DepthInterval,
    EISMeasurement,
    Measurement,
    MoistureScenarioType,
    SoilReference,
    SpatialCoordinates,
    SpectralSignalMode,
    Spectrum,
)
from st_ds.io.dataset import SoilDataset


@dataclass(frozen=True)
class SyntheticConfig:
    """Configuration for deterministic physics-informed synthetic soil dataset generation."""

    dataset_id: str = "stds_synthetic_v2"
    profile_id: str = "visnir_24ch_v1"
    n_samples: int = 120
    n_sites: int = 6
    seed: int = 42
    noise_std_reflectance: float = 0.0025
    include_eis: bool = True
    missing_bulk_density_ratio: float = 0.0


class SyntheticGeneratorV2:
    """Deterministic generator for synthetic soil spectra, covariates, and paired wet/dry sets."""

    PROVISIONAL_CLASSES: tuple[tuple[int, str], ...] = (
        (0, "low_organic_low_carbonate"),
        (1, "low_organic_calcareous"),
        (2, "humus_rich_non_calcareous"),
        (3, "humus_rich_calcareous"),
    )

    def __init__(self, config: SyntheticConfig | None = None) -> None:
        self.config = config or SyntheticConfig()
        self.profile: InstrumentProfile = get_builtin_profile(self.config.profile_id)

    @staticmethod
    def _gaussian_band(wl_nm: np.ndarray, center_nm: float, sigma_nm: float) -> np.ndarray:
        return np.exp(-0.5 * ((wl_nm - center_nm) / sigma_nm) ** 2)

    @classmethod
    def synthesize_absorbance(
        cls,
        wl_nm: np.ndarray,
        *,
        soc_pct: float,
        sic_pct: float,
        moisture_vol_pct: float,
        clay_fraction: float,
        baseline_offset: float,
        baseline_slope: float,
    ) -> np.ndarray:
        """Synthesize a physically plausible soil absorbance curve A(lambda) = -log10(R(lambda))."""
        wl_norm = (wl_nm - 400.0) / 2150.0

        # 1. Baseline scattering and iron-oxide/humus visible-NIR continuum
        base_abs = (
            baseline_offset
            + baseline_slope * wl_norm
            + 0.045 * soc_pct * np.exp(-np.maximum(wl_nm - 400.0, 0.0) / 650.0)
        )

        # 2. Organic matter functional group features (C-H stretch & overtones)
        org_abs = soc_pct * (
            0.032 * cls._gaussian_band(wl_nm, 1725.0, 35.0)
            + 0.055 * cls._gaussian_band(wl_nm, 2170.0, 42.0)
            + 0.025 * cls._gaussian_band(wl_nm, 940.0, 55.0)
        )

        # 3. Carbonate mineral CO3 features (SIC)
        carb_abs = sic_pct * (
            0.068 * cls._gaussian_band(wl_nm, 2335.0, 28.0)
            + 0.052 * cls._gaussian_band(wl_nm, 2520.0, 35.0)
            + 0.015 * cls._gaussian_band(wl_nm, 1995.0, 30.0)
        )

        # 4. Water O-H absorption bands (1430 nm, 1920 nm, plus weaker 970 nm overtone)
        water_abs = (moisture_vol_pct / 100.0) * (
            0.12 * cls._gaussian_band(wl_nm, 970.0, 45.0)
            + 0.58 * cls._gaussian_band(wl_nm, 1430.0, 48.0)
            + 0.85 * cls._gaussian_band(wl_nm, 1925.0, 52.0)
            + 0.18 * wl_norm
        )

        # 5. Clay mineral Al-OH absorption (~2205 nm)
        clay_abs = clay_fraction * (
            0.14 * cls._gaussian_band(wl_nm, 2205.0, 25.0)
        )

        total_abs = np.clip(base_abs + org_abs + carb_abs + water_abs + clay_abs, 0.02, 2.2)
        return total_abs

    @staticmethod
    def _assign_provisional_class(soc_pct: float, sic_pct: float) -> tuple[int, str]:
        high_soc = soc_pct >= 2.0
        high_sic = sic_pct >= 1.5
        if not high_soc and not high_sic:
            return SyntheticGeneratorV2.PROVISIONAL_CLASSES[0]
        if not high_soc and high_sic:
            return SyntheticGeneratorV2.PROVISIONAL_CLASSES[1]
        if high_soc and not high_sic:
            return SyntheticGeneratorV2.PROVISIONAL_CLASSES[2]
        return SyntheticGeneratorV2.PROVISIONAL_CLASSES[3]

    def _build_measurement(
        self,
        *,
        rng: np.random.Generator,
        sample_id: str,
        site_id: str,
        session_id: str,
        soc_pct: float,
        sic_pct: float,
        moisture_vol_pct: float,
        moisture_scenario: MoistureScenarioType,
        clay_fraction: float,
        ph: float,
        temperature_c: float,
        bulk_density_g_cm3: float | None,
        coarse_fragments: float,
        baseline_offset: float,
        baseline_slope: float,
        replicate_id: str = "rep_1",
    ) -> Measurement:
        wl_nm = np.asarray(self.profile.wavelength_grid_nm, dtype=float)
        abs_true = self.synthesize_absorbance(
            wl_nm,
            soc_pct=soc_pct,
            sic_pct=sic_pct,
            moisture_vol_pct=moisture_vol_pct,
            clay_fraction=clay_fraction,
            baseline_offset=baseline_offset,
            baseline_slope=baseline_slope,
        )
        refl_true = np.power(10.0, -abs_true)
        noise = rng.normal(0.0, self.config.noise_std_reflectance, size=wl_nm.shape)
        refl_noisy = np.clip(refl_true + noise, 0.005, 0.995)

        dark_ref = tuple(float(x) for x in np.full(wl_nm.shape, 0.01))
        white_ref = tuple(float(x) for x in np.full(wl_nm.shape, 0.99))

        spectrum = Spectrum(
            values=refl_noisy,
            wavelength_grid_nm=self.profile.wavelength_grid_nm,
            channel_ids=self.profile.channel_ids,
            mode=SpectralSignalMode.REFLECTANCE,
            dark_reference=dark_ref,
            white_reference=white_ref,
            integration_time_ms=50.0,
            gain=1.0,
            sensor_temperature_c=temperature_c,
            reference_age_s=30.0,
            replicate_id=replicate_id,
        )

        ec_ds_m = float(np.clip(0.15 + 0.045 * moisture_vol_pct + 0.08 * sic_pct, 0.05, 25.0))
        eis_obj = None
        if self.config.include_eis:
            freqs = (100.0, 1000.0, 10000.0, 100000.0)
            base_r = 1200.0 / (1.0 + 0.08 * moisture_vol_pct + 0.2 * ec_ds_m)
            z_real = tuple(float(base_r * (1.0 - 0.06 * idx)) for idx in range(len(freqs)))
            z_imag = tuple(float(-0.25 * base_r * (1.0 + 0.1 * idx)) for idx in range(len(freqs)))
            eis_obj = EISMeasurement(frequencies_hz=freqs, z_real_ohm=z_real, z_imag_ohm=z_imag)

        covariates = CovariateSet(
            ph=ph,
            moisture_volumetric_pct=moisture_vol_pct,
            moisture_gravimetric_pct=float(moisture_vol_pct * 0.72),
            temperature_c=temperature_c,
            ec_ds_m=ec_ds_m,
            bulk_density_g_cm3=bulk_density_g_cm3,
            bulk_density_uncertainty_g_cm3=0.05 if bulk_density_g_cm3 is not None else None,
            bulk_density_source="core_sample" if bulk_density_g_cm3 is not None else None,
            coarse_fragments_fraction=coarse_fragments,
            eis=eis_obj,
        )

        cls_id, cls_name = self._assign_provisional_class(soc_pct, sic_pct)
        tc_pct = float(soc_pct + sic_pct)
        reference = SoilReference(
            soc_pct=soc_pct,
            sic_pct=sic_pct,
            tc_pct=tc_pct,
            dry_matter_basis=True,
            soil_class_id=cls_id,
            soil_class_name=cls_name,
        )

        return Measurement(
            sample_id=sample_id,
            site_id=site_id,
            session_id=session_id,
            timestamp_utc="2026-11-12T12:00:00Z",
            instrument_id=f"INST_{self.profile.profile_id.upper()}",
            instrument_profile_id=self.profile.profile_id,
            calibration_id="cal_synth_v2",
            soil_texture_class="loam" if clay_fraction < 0.35 else "clay_loam",
            moisture_scenario=moisture_scenario,
            depth_interval=DepthInterval(0.0, 30.0),
            coordinates=SpatialCoordinates(lat=39.90 + 0.01 * hash(site_id) % 0.5, lon=32.85),
            spectrum=spectrum,
            covariates=covariates,
            reference=reference,
            dataset_source="synthetic",
        )

    def generate_dataset(self) -> SoilDataset:
        """Generate a synthetic SoilDataset with realistic site grouping and physical closure."""
        rng = np.random.default_rng(self.config.seed)
        records: list[Measurement] = []
        sites = [f"site_{i + 1:02d}" for i in range(self.config.n_sites)]

        for idx in range(self.config.n_samples):
            site_id = sites[idx % len(sites)]
            sample_id = f"SYN_{idx + 1:04d}"
            # Site-dependent baseline soil chemistry + sample variation
            site_idx = idx % len(sites)
            soc_pct = float(np.clip(rng.uniform(0.3, 4.8) + 0.15 * site_idx, 0.2, 6.0))
            sic_pct = float(np.clip(rng.uniform(0.05, 3.8) + 0.10 * (len(sites) - site_idx), 0.0, 5.5))
            moisture_vol = float(rng.uniform(4.0, 34.0))
            clay_frac = float(rng.uniform(0.12, 0.55))
            ph = float(np.clip(6.2 + 0.38 * sic_pct - 0.12 * soc_pct + rng.normal(0.0, 0.15), 4.5, 8.8))
            temp_c = float(rng.uniform(14.0, 28.0))
            coarse_frac = float(rng.uniform(0.02, 0.22))

            if rng.random() < self.config.missing_bulk_density_ratio:
                bd: float | None = None
            else:
                # Pedotransfer-consistent synthetic bulk density (decreases with higher SOC)
                bd = float(np.clip(1.58 - 0.065 * soc_pct + rng.normal(0.0, 0.04), 0.90, 1.85))

            base_offset = float(rng.uniform(0.18, 0.32))
            base_slope = float(rng.uniform(0.02, 0.09))
            scenario: MoistureScenarioType = (
                "dry" if moisture_vol < 10.0 else ("field_moist" if moisture_vol < 24.0 else "wet")
            )

            records.append(
                self._build_measurement(
                    rng=rng,
                    sample_id=sample_id,
                    site_id=site_id,
                    session_id=f"sess_{site_id}",
                    soc_pct=soc_pct,
                    sic_pct=sic_pct,
                    moisture_vol_pct=moisture_vol,
                    moisture_scenario=scenario,
                    clay_fraction=clay_frac,
                    ph=ph,
                    temperature_c=temp_c,
                    bulk_density_g_cm3=bd,
                    coarse_fragments=coarse_frac,
                    baseline_offset=base_offset,
                    baseline_slope=base_slope,
                )
            )

        return SoilDataset(
            dataset_id=self.config.dataset_id,
            records=records,
            dataset_source="synthetic",
            description=f"Physics-informed synthetic soil dataset (seed={self.config.seed})",
        )

    def generate_paired_wet_dry_dataset(
        self,
        *,
        n_pairs: int = 40,
        dry_moisture_range: tuple[float, float] = (2.0, 5.0),
        wet_moisture_range: tuple[float, float] = (22.0, 38.0),
    ) -> tuple[SoilDataset, SoilDataset]:
        """Generate matched (X_dry, X_wet) soil datasets on identical sample_ids for EPO calibration."""
        rng = np.random.default_rng(self.config.seed + 1000)
        dry_records: list[Measurement] = []
        wet_records: list[Measurement] = []
        sites = [f"cal_site_{i + 1:02d}" for i in range(max(2, self.config.n_sites))]

        for idx in range(n_pairs):
            site_id = sites[idx % len(sites)]
            sample_id = f"EPO_PAIR_{idx + 1:04d}"
            soc_pct = float(rng.uniform(0.4, 4.5))
            sic_pct = float(rng.uniform(0.1, 3.5))
            clay_frac = float(rng.uniform(0.15, 0.50))
            ph = float(np.clip(6.4 + 0.3 * sic_pct - 0.1 * soc_pct, 4.8, 8.6))
            temp_c = float(rng.uniform(18.0, 24.0))
            bd = float(np.clip(1.55 - 0.06 * soc_pct, 0.95, 1.80))
            coarse_frac = float(rng.uniform(0.03, 0.15))
            base_offset = float(rng.uniform(0.20, 0.30))
            base_slope = float(rng.uniform(0.03, 0.08))

            moist_dry = float(rng.uniform(*dry_moisture_range))
            moist_wet = float(rng.uniform(*wet_moisture_range))

            dry_records.append(
                self._build_measurement(
                    rng=rng,
                    sample_id=sample_id,
                    site_id=site_id,
                    session_id="epo_cal_dry",
                    soc_pct=soc_pct,
                    sic_pct=sic_pct,
                    moisture_vol_pct=moist_dry,
                    moisture_scenario="dry",
                    clay_fraction=clay_frac,
                    ph=ph,
                    temperature_c=temp_c,
                    bulk_density_g_cm3=bd,
                    coarse_fragments=coarse_frac,
                    baseline_offset=base_offset,
                    baseline_slope=base_slope,
                )
            )
            wet_records.append(
                self._build_measurement(
                    rng=rng,
                    sample_id=sample_id,
                    site_id=site_id,
                    session_id="epo_cal_wet",
                    soc_pct=soc_pct,
                    sic_pct=sic_pct,
                    moisture_vol_pct=moist_wet,
                    moisture_scenario="wet",
                    clay_fraction=clay_frac,
                    ph=ph,
                    temperature_c=temp_c,
                    bulk_density_g_cm3=bd,
                    coarse_fragments=coarse_frac,
                    baseline_offset=base_offset,
                    baseline_slope=base_slope,
                )
            )

        dry_ds = SoilDataset(
            dataset_id=f"{self.config.dataset_id}_dry_pairs",
            records=dry_records,
            dataset_source="synthetic",
            description="Paired dry calibration samples for EPO subspace estimation",
        )
        wet_ds = SoilDataset(
            dataset_id=f"{self.config.dataset_id}_wet_pairs",
            records=wet_records,
            dataset_source="synthetic",
            description="Paired wet calibration samples for EPO subspace estimation",
        )
        return dry_ds, wet_ds

    def generate_multi_replicate_dataset(self, *, n_replicates: int = 3) -> SoilDataset:
        """Generate a synthetic dataset where each physical sample_id has multiple replicate scans."""
        base_ds = self.generate_dataset()
        rng = np.random.default_rng(self.config.seed + 2000)
        rep_records: list[Measurement] = []
        for rec in base_ds.records:
            for rep_idx in range(n_replicates):
                noisy_vals = np.clip(
                    rec.spectrum.values + rng.normal(0.0, self.config.noise_std_reflectance, size=rec.spectrum.n_bands),
                    0.005,
                    0.995,
                )
                rep_spec = Spectrum(
                    values=noisy_vals,
                    wavelength_grid_nm=rec.spectrum.wavelength_grid_nm,
                    channel_ids=rec.spectrum.channel_ids,
                    mode=rec.spectrum.mode,
                    dark_reference=rec.spectrum.dark_reference,
                    white_reference=rec.spectrum.white_reference,
                    integration_time_ms=rec.spectrum.integration_time_ms,
                    gain=rec.spectrum.gain,
                    sensor_temperature_c=rec.spectrum.sensor_temperature_c,
                    reference_age_s=rec.spectrum.reference_age_s,
                    replicate_id=f"rep_{rep_idx + 1}",
                )
                rep_records.append(
                    Measurement(
                        sample_id=rec.sample_id,
                        site_id=rec.site_id,
                        session_id=rec.session_id,
                        timestamp_utc=rec.timestamp_utc,
                        instrument_id=rec.instrument_id,
                        instrument_profile_id=rec.instrument_profile_id,
                        calibration_id=rec.calibration_id,
                        soil_texture_class=rec.soil_texture_class,
                        sample_preparation=rec.sample_preparation,
                        moisture_scenario=rec.moisture_scenario,
                        depth_interval=rec.depth_interval,
                        coordinates=rec.coordinates,
                        spectrum=rep_spec,
                        covariates=rec.covariates,
                        reference=rec.reference,
                        dataset_source="synthetic",
                    )
                )
        return SoilDataset(
            dataset_id=f"{self.config.dataset_id}_replicates",
            records=rep_records,
            dataset_source="synthetic",
            description=f"Multi-replicate ({n_replicates}x) synthetic dataset for split leakage verification",
        )
