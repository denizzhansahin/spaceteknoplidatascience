"""Faz 2 Unit & Contract Tests: SoilDataset, Source Isolation, Grouped Splits, and SyntheticGeneratorV2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from st_ds.domain import (
    CovariateSet,
    Measurement,
    QualityFlag,
    SchemaError,
    SoilReference,
    Spectrum,
    get_builtin_profile,
)
from st_ds.io import (
    DataValidator,
    ReferenceLabelStore,
    SoilDataset,
    SyntheticConfig,
    SyntheticGeneratorV2,
    build_dataset_card,
    generate_data_quality_report,
    load_dataset_csv,
    load_dataset_json,
    load_dataset_parquet,
    save_dataset_csv,
    save_dataset_json,
    save_dataset_parquet,
)
from st_ds.training import GroupedSplitBuilder


class TestSyntheticGeneratorV2:
    """Verify deterministic, physics-informed synthetic dataset and paired wet/dry generation."""

    def test_deterministic_reproducibility_and_closure(self) -> None:
        cfg = SyntheticConfig(
            dataset_id="synth_test_01",
            profile_id="visnir_24ch_v1",
            n_samples=60,
            n_sites=5,
            seed=12345,
            missing_bulk_density_ratio=0.25,
        )
        ds1 = SyntheticGeneratorV2(cfg).generate_dataset()
        ds2 = SyntheticGeneratorV2(cfg).generate_dataset()

        assert ds1.compute_sha256() == ds2.compute_sha256()
        assert ds1.dataset_source == "synthetic"
        assert len(ds1) == 60

        for r in ds1.records:
            assert r.dataset_source == "synthetic"
            assert QualityFlag.SYNTHETIC_DATA_SOURCE in r.quality.flags
            assert r.reference is not None
            assert pytest.approx(r.reference.closure_residual_pct, abs=1e-12) == 0.0
            assert r.reference.soc_pct >= 0.0
            assert r.reference.sic_pct >= 0.0

        report = generate_data_quality_report(ds1)
        assert report["is_synthetic"] is True
        assert report["missing_bulk_density_count"] > 0

    def test_paired_wet_dry_spectra_water_band_physics(self) -> None:
        cfg = SyntheticConfig(
            dataset_id="epo_pairs_test",
            profile_id="ftnir_1300_2550nm_v1",
            seed=99,
            noise_std_reflectance=0.0005,
        )
        gen = SyntheticGeneratorV2(cfg)
        dry_ds, wet_ds = gen.generate_paired_wet_dry_dataset(n_pairs=20)

        assert len(dry_ds) == 20
        assert len(wet_ds) == 20
        water_band_indices = gen.profile.bands_in_window(1415.0, 1450.0) + gen.profile.bands_in_window(1910.0, 1945.0)
        assert len(water_band_indices) > 0

        for dry_rec, wet_rec in zip(dry_ds.records, wet_ds.records):
            assert dry_rec.sample_id == wet_rec.sample_id
            assert dry_rec.reference.soc_pct == wet_rec.reference.soc_pct
            assert dry_rec.reference.sic_pct == wet_rec.reference.sic_pct
            assert wet_rec.covariates.moisture_volumetric_pct > dry_rec.covariates.moisture_volumetric_pct

            # Higher moisture increases O-H absorbance -> decreases reflectance at 1430nm & 1925nm
            abs_dry = -np.log10(dry_rec.spectrum.values[water_band_indices])
            abs_wet = -np.log10(wet_rec.spectrum.values[water_band_indices])
            assert float(np.mean(abs_wet)) > float(np.mean(abs_dry)) + 0.08


class TestDatasetSourceIsolationAndRoundTripIO:
    """Verify strict separation of synthetic vs real_lab data and lossless JSON/CSV round-trips."""

    def test_mixing_synthetic_and_real_lab_raises_schema_error(self) -> None:
        cfg = SyntheticConfig(n_samples=10, n_sites=2, seed=7)
        ds_synth = SyntheticGeneratorV2(cfg).generate_dataset()

        # Attempt to declare synthetic records as real_lab dataset -> must fail
        with pytest.raises(SchemaError, match="Dataset source mismatch or contamination"):
            SoilDataset(
                dataset_id="fake_real_lab",
                records=ds_synth.records,
                dataset_source="real_lab",
            )

    def test_json_and_csv_round_trip_integrity(self, tmp_path: Path) -> None:
        cfg = SyntheticConfig(
            dataset_id="io_roundtrip_ds",
            profile_id="visnir_24ch_v1",
            n_samples=24,
            n_sites=4,
            seed=2026,
        )
        ds = SyntheticGeneratorV2(cfg).generate_dataset()

        # 1. JSON round-trip (exact SHA-256 match)
        json_path = save_dataset_json(ds, tmp_path / "dataset.json")
        loaded_json_ds = load_dataset_json(json_path)
        assert loaded_json_ds.compute_sha256() == ds.compute_sha256()

        # 2. CSV + sidecar round-trip
        csv_path = save_dataset_csv(ds, tmp_path / "dataset.csv")
        loaded_csv_ds = load_dataset_csv(csv_path)
        assert len(loaded_csv_ds) == len(ds)
        assert loaded_csv_ds.dataset_source == "synthetic"
        np.testing.assert_allclose(
            loaded_csv_ds.records[0].spectrum.values,
            ds.records[0].spectrum.values,
            rtol=1e-6,
            atol=1e-8,
        )


class TestGroupedSplitBuilder:
    """Verify zero sample_id and zero site_id leakage across folds (DEF-009)."""

    def test_group_kfold_and_leave_one_site_out_zero_leakage(self) -> None:
        cfg = SyntheticConfig(n_samples=60, n_sites=6, seed=42)
        ds = SyntheticGeneratorV2(cfg).generate_dataset()
        ds.validate_for_training()

        gkf_folds = GroupedSplitBuilder.group_kfold_split(ds, n_splits=3, group_by="site_id")
        assert len(gkf_folds) == 3
        for fold in gkf_folds:
            assert set(fold.train_groups).isdisjoint(set(fold.val_groups))
            assert set(fold.train_sample_ids).isdisjoint(set(fold.val_sample_ids))

        loso_folds = GroupedSplitBuilder.leave_one_site_out_split(ds)
        assert len(loso_folds) == 6
        for fold in loso_folds:
            assert len(fold.val_groups) == 1
            assert set(fold.train_groups).isdisjoint(set(fold.val_groups))
            assert set(fold.train_sample_ids).isdisjoint(set(fold.val_sample_ids))

    def test_deliberate_leakage_detected_and_rejected(self) -> None:
        cfg = SyntheticConfig(n_samples=20, n_sites=4, seed=42)
        ds = SyntheticGeneratorV2(cfg).generate_dataset()

        # Put record 0 in both train and eval -> must raise SchemaError
        with pytest.raises(SchemaError, match="Data leakage detected"):
            GroupedSplitBuilder.verify_no_leakage(ds.records[:10], ds.records[0:5])

    def test_multi_replicate_dataset_keeps_replicates_together_across_folds(self) -> None:
        cfg = SyntheticConfig(n_samples=18, n_sites=6, seed=77)
        rep_ds = SyntheticGeneratorV2(cfg).generate_multi_replicate_dataset(n_replicates=3)
        assert len(rep_ds) == 54  # 18 samples * 3 replicates

        folds = GroupedSplitBuilder.group_kfold_split(rep_ds, n_splits=3, group_by="site_id")
        for fold in folds:
            assert set(fold.train_sample_ids).isdisjoint(set(fold.val_sample_ids))
            assert set(fold.train_groups).isdisjoint(set(fold.val_groups))


class TestReferenceStoreValidatorParquetAndCard:
    """Verify Faz 2 completeness: ReferenceLabelStore, DataValidator, Parquet IO, and DatasetCard."""

    def test_reference_label_store_attach_and_synthetic_guard(self, tmp_path: Path) -> None:
        prof = get_builtin_profile("led_5band_v1")
        spec = Spectrum(
            values=np.array([0.1, 0.2, 0.3, 0.25, 0.2]),
            wavelength_grid_nm=prof.wavelength_grid_nm,
            channel_ids=prof.channel_ids,
        )
        cov = CovariateSet.from_raw({"ph": {"value": 6.8, "unit": "dimensionless"}})
        real_meas = Measurement(
            sample_id="LAB_101",
            site_id="field_A",
            timestamp_utc="2026-11-12T10:00:00Z",
            instrument_id="DEV_1",
            instrument_profile_id="led_5band_v1",
            calibration_id="cal_1",
            spectrum=spec,
            covariates=cov,
            dataset_source="real_lab",
        )
        store = ReferenceLabelStore(
            store_id="lab_batch_1",
            labels={"LAB_101": SoilReference.from_units(soc=1.9, sic=0.6, unit="pct")},
        )
        csv_ref_path = store.save_csv(tmp_path / "refs.csv")
        loaded_store = ReferenceLabelStore.load_csv(csv_ref_path)
        attached = loaded_store.attach_to_measurements([real_meas], strict=True)
        assert attached[0].reference is not None
        assert pytest.approx(attached[0].reference.tc_pct) == 2.5

        # Guard: real label store cannot be attached to synthetic measurements
        synth_ds = SyntheticGeneratorV2(SyntheticConfig(n_samples=3, n_sites=2, seed=5)).generate_dataset()
        with pytest.raises(SchemaError, match="synthetic"):
            loaded_store.attach_to_measurements(synth_ds.records)

    def test_parquet_round_trip_validator_and_dataset_card(self, tmp_path: Path) -> None:
        cfg = SyntheticConfig(n_samples=18, n_sites=3, seed=55)
        ds = SyntheticGeneratorV2(cfg).generate_dataset()

        summary = DataValidator.validate_dataset(ds, for_training=True)
        assert summary.is_valid is True
        assert summary.n_errors == 0

        pq_path = save_dataset_parquet(ds, tmp_path / "dataset.parquet")
        loaded_pq = load_dataset_parquet(pq_path)
        assert len(loaded_pq) == len(ds)
        assert loaded_pq.records[0].soil_texture_class == ds.records[0].soil_texture_class
        assert loaded_pq.records[0].reference.soc_method == ds.records[0].reference.soc_method

        card = build_dataset_card(ds)
        assert card["is_synthetic"] is True
        assert card["lineage"]["site_count"] == 3
