"""STDS (`st_ds`) — 03: Sıfırdan CSV Veri Seti Hazırlama ve Toprak Dışı Alanlar (Gıda / Hava).

Bu betik iki gerçek hayat senaryosunu tek başına çalıştırır:
  Senaryo A (Toprak Ham CSV): Excel/pandas ile üretilmiş yan dosyası (.meta.json)
    OLMAYAN bir eğitim CSV'si + hedef sütunu (soc_pct) olmayan bir tahmin CSV'sini
    `st-ds` CLI komutlarına `--instrument-profile visnir_24ch_v1` ile verip çalıştırır.
  Senaryo B (Toprak Dışı Alan — Gıda NIR Protein Analizi): Toprak karbonu
    olmayan `protein_pct` hedefi için özel bir 9-bantlı `InstrumentProfile` JSON'u
    ve ham CSV oluşturup hem Python API hem CLI (`--target protein_pct --model plsr`)
    üzerinden analiz ve tam PDF/grafik paketi üretir.

Çalıştırma:
    PYTHONPATH=. /opt/anaconda3/bin/python examples/03_custom_csv_and_non_soil_domains.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from st_ds.domain.instrument import InstrumentProfile, SpectralModality, get_builtin_profile
from st_ds.features.fusion import FeatureFusionConfig
from st_ds.io.dataset import load_dataset_csv
from st_ds.training.splits import GroupedSplitBuilder
from st_ds.training.trainer import STSoilTrainer, TrainerConfig


def _run_cli(*args: str) -> None:
    cmd = [sys.executable, "-m", "st_ds.cli", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"CLI error ({proc.returncode}): {' '.join(cmd)}\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}")


def scenario_a_raw_soil_csv(work_dir: Path) -> None:
    print("\n=== SENARYO A: Sıfırdan Ham Toprak CSV (Eğitim + Hedefsiz Tahmin) ===")
    prof = get_builtin_profile("visnir_24ch_v1")
    rng = np.random.default_rng(7)

    # 1) Eğitim CSV satırları (her satır benzersiz sample_id + replicate_id + moisture_scenario)
    rows_train = []
    wl_arr = np.asarray(prof.wavelength_grid_nm)
    for i in range(32):
        site_id = f"tarla_{i // 8 + 1:02d}"
        soc = 1.2 + 0.08 * i + float(rng.normal(0.0, 0.05))
        sic = 0.4 + 0.03 * (i % 5)
        tc = soc + sic
        # Fiziksel kurallara uygun spektrum (wl_{w:.2f}nm sütunları)
        abs_curve = 0.25 + 0.06 * soc * np.exp(-0.5 * ((wl_arr - 2170.0) / 45.0) ** 2)
        refl = np.clip(10.0 ** (-abs_curve) + rng.normal(0.0, 0.002, size=wl_arr.size), 0.02, 0.98)

        row = {
            "sample_id": f"SOIL_{i + 1:03d}",
            "site_id": site_id,
            "replicate_id": "rep_1",
            "moisture_scenario": "dry",
            "timestamp_utc": "2026-11-12T09:00:00Z",
            "instrument_id": "lab_spec_01",
            "instrument_profile_id": prof.profile_id,
            "calibration_id": "cal_2026_v1",
            "dataset_source": "real_lab",     # Gerçek veri için 'real_lab', sentetik için 'synthetic'
            "depth_top_cm": 0.0,
            "depth_bottom_cm": 30.0,
            "lat": 39.90 + 0.005 * (i // 8) + 0.0004 * (i % 8),
            "lon": 32.80 + 0.006 * (i % 8),
            "ph": round(6.5 + 0.02 * i, 2),
            "moisture_volumetric_pct": round(6.0 + 0.4 * (i % 6), 2),
            "temperature_c": 22.0,
            "ec_ds_m": 0.45,
            "bulk_density_g_cm3": 1.32,       # Stok hesabı için şart (yoksa stock=None döner)
            "soc_pct": round(soc, 4),
            "sic_pct": round(sic, 4),
            "tc_pct": round(tc, 4),
        }
        for w, val in zip(prof.wavelength_grid_nm, refl):
            row[f"wl_{w:.2f}nm"] = round(float(val), 6)
        rows_train.append(row)

    train_csv = work_dir / "ham_toprak_egitim.csv"
    pd.DataFrame(rows_train).to_csv(train_csv, index=False)

    # 2) Tahmin (Inference) CSV satırları: soc_pct / sic_pct / tc_pct SÜTUNLARI HİÇ YOK!
    rows_pred = []
    for j in range(4):
        row = dict(rows_train[j])
        row["sample_id"] = f"YENI_SAHA_{j + 1:02d}"
        for drop_col in ("soc_pct", "sic_pct", "tc_pct"):
            row.pop(drop_col, None)
        rows_pred.append(row)
    pred_csv = work_dir / "yeni_saha_tahmin.csv"
    pd.DataFrame(rows_pred).to_csv(pred_csv, index=False)

    # 3) Ham CSV'yi doğrudan CLI ile doğrula, eğit ve hedefsiz dosya üzerinde tahmin yap
    _run_cli(
        "dataset", "validate",
        "--input", str(train_csv),
        "--instrument-profile", "visnir_24ch_v1",
        "--for-training",
    )
    bundle_dir = work_dir / "soil_csv_bundle"
    _run_cli(
        "train",
        "--input", str(train_csv),
        "--instrument-profile", "visnir_24ch_v1",
        "--target", "soc_pct",
        "--model", "plsr",
        "--multi-target",
        "--output", str(bundle_dir),
    )
    pred_out = work_dir / "soil_csv_preds.json"
    _run_cli(
        "predict",
        "--artifact", str(bundle_dir),
        "--input", str(pred_csv),
        "--instrument-profile", "visnir_24ch_v1",
        "--output", str(pred_out),
    )
    preds = json.loads(pred_out.read_text(encoding="utf-8"))
    print(
        f"  Eğitim & hedefsiz tahmin CSV tamamlandı: {preds['summary']['n_succeeded']}/{preds['summary']['n_total']} başarılı."
    )


def scenario_b_non_soil_food_and_air_domain(work_dir: Path) -> None:
    print("\n=== SENARYO B: Toprak Dışı Alan (9-Bant Gıda Protein NIR + Özel Hedef) ===")
    # 1) Toprak dışı özel sensör profili (700–1020 nm, 9 bant) -> JSON dosyasına kaydet
    wl_grid = tuple(float(w) for w in range(700, 1021, 40))
    ch_ids = tuple(f"FOOD_{int(w)}" for w in wl_grid)
    food_profile = InstrumentProfile(
        profile_id="gida_nir_9ch_v1",
        sensor_model="GrainProtein-NIR9",
        modality=SpectralModality.VIS_NIR_SWIR,
        wavelength_grid_nm=wl_grid,
        channel_ids=ch_ids,
        fwhm_nm=tuple(15.0 for _ in wl_grid),
        metadata={"domain": "food_protein_spectroscopy"},
    )
    profile_json = work_dir / "gida_profil.json"
    profile_json.write_text(json.dumps(food_profile.to_dict(), indent=2), encoding="utf-8")

    # 2) Ham CSV oluştur — hedef kolonumuz toprak değil: `protein_pct` (ör. %9.0 – %16.5)
    rng = np.random.default_rng(101)
    wl_arr = np.asarray(wl_grid)
    rows = []
    for i in range(36):
        silo = f"silo_{i // 9 + 1:02d}"
        protein_pct = 9.5 + 0.18 * i + float(rng.normal(0.0, 0.15))
        moisture = 10.5 + 0.3 * (i % 4)
        # N-H protein bandı ~910 nm, O-H nem bandı ~980 nm simülasyonu
        absorb = (
            0.18
            + 0.025 * protein_pct * np.exp(-0.5 * ((wl_arr - 900.0) / 35.0) ** 2)
            + 0.012 * moisture * np.exp(-0.5 * ((wl_arr - 980.0) / 28.0) ** 2)
        )
        refl = np.clip(10.0 ** (-absorb) + rng.normal(0.0, 0.0015, size=wl_arr.size), 0.02, 0.98)

        row = {
            "sample_id": f"BUGDAY_{i + 1:03d}",
            "site_id": silo,
            "timestamp_utc": "2026-11-12T12:00:00Z",
            "instrument_id": "tahil_nir_02",
            "instrument_profile_id": food_profile.profile_id,
            "calibration_id": "gida_cal_v1",
            "dataset_source": "real_lab",
            "temperature_c": round(21.0 + 0.4 * (i % 5), 2),
            "moisture_volumetric_pct": round(moisture, 2),
            "lat": 37.87 + 0.003 * (i // 9) + 0.0005 * (i % 9),
            "lon": 32.48 + 0.004 * (i % 9),
            "protein_pct": round(protein_pct, 3),  # <--- ÖZEL HEDEF KOLONU!
        }
        for w, val in zip(wl_grid, refl):
            row[f"wl_{w:.2f}nm"] = round(float(val), 6)
        rows.append(row)

    food_csv = work_dir / "gida_protein.csv"
    pd.DataFrame(rows).to_csv(food_csv, index=False)

    # 3) Python API üzerinden özel hedef ve profil ile yükleme + kıyaslama
    ds = load_dataset_csv(food_csv, profile=food_profile, target_columns=["protein_pct"])
    folds = GroupedSplitBuilder.group_kfold_split(ds, n_splits=4, group_by="site_id")
    bench = STSoilTrainer(
        TrainerConfig(
            target_name="protein_pct",
            fusion_config=FeatureFusionConfig(covariate_names=("moisture_volumetric_pct", "temperature_c")),
            seed=101,
        )
    ).run_benchmark(
        ds, split_folds=folds, include_optional_models=False
    )
    print("  Python API Benchmark (protein_pct):", bench.summary_table())

    # 4) Aynı işlemi sıfır kodlu CLI (`--instrument-profile gida_profil.json --target protein_pct --covariates ...`) ile çalıştır!
    food_bundle = work_dir / "food_bundle"
    _run_cli(
        "train",
        "--input", str(food_csv),
        "--instrument-profile", str(profile_json),
        "--target", "protein_pct",
        "--covariates", "moisture_volumetric_pct,temperature_c",
        "--model", "plsr",
        "--folds", "3",
        "--output", str(food_bundle),
    )
    food_viz = work_dir / "food_viz"
    _run_cli(
        "visualize",
        "--input", str(food_csv),
        "--artifact", str(food_bundle),
        "--instrument-profile", str(profile_json),
        "--target", "protein_pct",
        "--model", "plsr",
        "--dpi", "90",
        "--no-interactive-3d",
        "--lang", "tr",
        "--output", str(food_viz),
    )
    pdf_file = food_viz / "reports" / "stds_analysis_report.pdf"
    print(f"  CLI Toprak Dışı (`protein_pct`) PDF Raporu Üretildi: {pdf_file.name} ({pdf_file.stat().st_size} bayt)")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="stds_custom_domains_") as tmp:
        work = Path(tmp)
        scenario_a_raw_soil_csv(work)
        scenario_b_non_soil_food_and_air_domain(work)
    print("\nSenaryo A ve Senaryo B temiz biçimde doğrulandı!")


if __name__ == "__main__":
    main()
