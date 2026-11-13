"""STDS (`st_ds`) — 02: Tüm Python API Yapılandırma Sınıfları ve Parametre Örnekleri.

Bu betik, bash komut dosyalarına ihtiyaç duymadan STDS kütüphanesindeki TÜM temel
yapılandırma sınıflarının (`SyntheticConfig`, `PreprocessConfig`, `FeatureFusionConfig`,
tüm 5 regresyon modeli, `TrainerConfig`, `ConformalConfig`, `OODConfig`,
`STSoilAnalyzer` ve `VisualizationConfig`) parametrelerini yorum satırlarıyla açıklayarak
çalıştırır ve farklı değerlerin çıktıyı nasıl değiştirdiğini gösterir.

Çalıştırma:
    PYTHONPATH=. /opt/anaconda3/bin/python examples/02_all_python_api_parameters.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from st_ds import (
    SCHEMA_VERSION,
    STSoilAnalyzer,
    VisualizationConfig,
    save_artifact,
)
from st_ds.artifact.bundle import collect_dependency_versions
from st_ds.chemometrics.pipeline import ChemometricPipeline, PreprocessConfig
from st_ds.domain.instrument import BUILTIN_PROFILES, get_builtin_profile
from st_ds.domain.schemas import ArtifactMetadata
from st_ds.features.fusion import FeatureFusionConfig, FeatureFusionPipeline
from st_ds.io.synthetic import SyntheticConfig, SyntheticGeneratorV2
from st_ds.models.baselines import (
    KernelRidgeRegressor,
    LocalPLSRRegressor,
    PLSRRegressor,
    RandomForestSoilRegressor,
)
from st_ds.models.hybrid_net import STSoilHybridNetRegressor
from st_ds.training.splits import GroupedSplitBuilder
from st_ds.training.trainer import STSoilTrainer, TrainerConfig, _extract_target_vector
from st_ds.uncertainty.conformal import ConformalConfig, ConformalUncertaintyCalibrator
from st_ds.uncertainty.ood import OODConfig, OutOfDomainDetector


def main() -> None:
    print("Yerleşik Cihaz Profilleri:", list(BUILTIN_PROFILES.keys()))

    # --------------------------------------------------------------------------
    # 1) SyntheticConfig parametreleri
    # --------------------------------------------------------------------------
    syn_cfg = SyntheticConfig(
        dataset_id="py_param_demo",
        profile_id="visnir_24ch_v1",       # led_5band_v1 | multispectral_18ch_v1 | visnir_24ch_v1 | ftnir_1300_2550nm_v1 | libs_ci_hi_v1
        n_samples=64,                      # Toplam örnek satır sayısı
        n_sites=8,                         # Benzersiz saha grubu sayısı (>= 2)
        seed=42,                           # Deterministik rastgelelik tohumu
        noise_std_reflectance=0.0025,      # Yansıma gürültü standart sapması
        include_eis=True,                  # 12 frekanslı EIS elektrokimyasal ölçüm üret
        missing_bulk_density_ratio=0.10,   # Örneklerin %10'unda hacim ağırlığını eksik bırak (DEF-005 testi)
    )
    gen = SyntheticGeneratorV2(syn_cfg)
    dataset = gen.generate_dataset()
    dry_ds, wet_ds = gen.generate_paired_wet_dry_dataset(
        n_pairs=18,
        dry_moisture_range=(2.0, 5.0),
        wet_moisture_range=(22.0, 38.0),
    )
    print(f"[1] Üretilen veri: {len(dataset)} ölçüm, {len(dataset.unique_site_ids)} saha.")

    # --------------------------------------------------------------------------
    # 2) PreprocessConfig parametreleri (Absorbans -> SNV -> SavGol -> EPO/GLSW)
    # --------------------------------------------------------------------------
    pre_epo = PreprocessConfig(
        use_absorbance=True,     # True: A = -log10(R); False: ham spektrum zaten absorbanssa
        use_snv=True,            # True: Satır bazlı ort. çıkar, std böl (saçılma düzeltmesi)
        use_savgol=True,         # True: Savitzky-Golay türev filtresi uygular
        sg_window=7,             # Tek tamsayı pencere boyutu (5, 7, 9...)
        sg_polyorder=2,          # Polinom derecesi (< sg_window)
        sg_deriv=1,              # 0 (yumuşatma), 1 (1. türev: taban kayması siler), 2 (2. türev)
        suppression_mode="epo",  # "none" | "epo" (dik projeksiyon) | "glsw" (kovaryans sönümleme)
        epo_q=2,                 # EPO rankı (None -> otomatik tekil değer eşiği)
        epo_max_q=6,             # Otomatik rank üst sınırı
    )
    pre_glsw = PreprocessConfig(suppression_mode="glsw", glsw_alpha=0.1)

    chemo_epo = ChemometricPipeline(pre_epo).fit_from_datasets(wet_ds, dry_ds)
    chemo_glsw = ChemometricPipeline(pre_glsw).fit_from_datasets(wet_ds, dry_ds)
    assert chemo_epo.epo is not None and chemo_glsw.glsw is not None
    print(f"[2] EPO q={chemo_epo.epo.q} invariant max_err={max(chemo_epo.epo.check_invariants().values()):.2e}")

    # --------------------------------------------------------------------------
    # 3) FeatureFusionConfig parametreleri (Spektral + Kovaryat dalları)
    # --------------------------------------------------------------------------
    fusion_cfg = FeatureFusionConfig(
        # Modelin kovaryat dalına girecek sütunlar (boş tuple () = sadece spektrum)
        covariate_names=("ph", "moisture_volumetric_pct", "temperature_c", "ec_ds_m"),
        missing_strategy="median_with_indicator",  # "median_with_indicator" | "fail_closed"
        spectral_scaler="standard",                # "none" | "standard"
        covariate_scaler="robust",                 # "standard" | "robust" (medyan/IQR)
        include_visnir_bands=True,                 # Fiziksel kil/organik/karbonat bant öznitelikleri
        include_derived=True,                      # Türetilmiş oran indeksleri
        include_eis=False,                         # EIS Nyquist öznitelikleri (varsa True yapılabilir)
    )
    fusion = FeatureFusionPipeline(fusion_cfg).fit(dataset.records, chemometric_pipeline=chemo_epo)
    fused = fusion.build_feature_vector(dataset.records, chemometric_pipeline=chemo_epo)
    print(f"[3] Fused feature boyutu: X_spectral={fused.X_spectral.shape}, X_covariates={fused.X_covariates.shape}")

    # --------------------------------------------------------------------------
    # 4) Beş Aday Model Sınıfı ve Özel Hiperparametreleri
    # --------------------------------------------------------------------------
    y_tc = _extract_target_vector(dataset.records, "tc_pct")  # Örnek hedef: tc_pct (veya soc_pct / sic_pct)
    groups = np.asarray([r.site_id for r in dataset.records], dtype=str)

    models = {
        "plsr_k6": PLSRRegressor(
            max_components=6,        # İç CV'de denenecek maks latent bileşen sayısı
            inner_cv_folds=4,        # Eğitim-içi hiperparametre CV kat sayısı
            scale=True,
            target_names=("tc_pct",),
            clip_negative=True,      # Fiziksel negatif tahmin kilidi
        ),
        "krr_custom": KernelRidgeRegressor(
            alpha_grid=(1e-2, 1e-1, 1.0),         # L2 düzenlileştirme adayları
            sigma_multipliers=(0.5, 1.0, 2.0),    # RBF bant genişliği çarpanları
            inner_cv_folds=4,
            target_names=("tc_pct",),
        ),
        "rf_100": RandomForestSoilRegressor(
            n_estimators=100,        # Ormandaki karar ağacı sayısı
            max_depth=10,            # Ağaç derinlik sınırı (None = sınırsız)
            min_samples_leaf=2,
            random_state=42,
            target_names=("tc_pct",),
        ),
        "local_mbl_k15": LocalPLSRRegressor(
            n_neighbors=15,          # Her test noktası için en yakın spektral komşu sayısı
            n_components=3,
            bandwidth=None,          # None -> eğitim medyan uzaklığından otomatik Gauss genişliği
            min_library_size=8,
            target_names=("tc_pct",),
        ),
        "hybrid_net": STSoilHybridNetRegressor(
            n_spectral_features=fused.X_spectral.shape[1],
            spectral_hidden_dim=48,  # Spektral encoder gizli genişliği
            covariate_hidden_dim=24, # Kovaryat encoder gizli genişliği
            fusion_hidden_dim=48,
            dropout=0.10,
            epochs=10,
            learning_rate=2e-3,
            batch_size=16,
            seed=42,
            target_names=("tc_pct",),
        ),
    }

    for name, mdl in models.items():
        mdl.set_metadata(
            feature_names=fused.fused_feature_names,
            target_names=("tc_pct",),
            dataset_source=dataset.dataset_source,
        )
        if isinstance(mdl, STSoilHybridNetRegressor):
            mdl.fit_multitask(fused.X_spectral, fused.X_covariates, soc=y_tc)
        else:
            try:
                mdl.fit(fused.X_fused, y_tc, groups=groups)  # type: ignore[call-arg]
            except TypeError:
                mdl.fit(fused.X_fused, y_tc)
        preds, stds = mdl.predict_distribution(
            fused.X_fused if not isinstance(mdl, STSoilHybridNetRegressor) else fused.X_fused[:5]
        )
        print(f"    Model {name:14s}: ilk 3 tahmin={np.round(np.asarray(preds).ravel()[:3], 3)}")

    # --------------------------------------------------------------------------
    # 5) TrainerConfig + ConformalConfig + OODConfig + STSoilAnalyzer.predict(..., alpha=...)
    # --------------------------------------------------------------------------
    trainer_cfg = TrainerConfig(
        target_name="tc_pct",
        preprocess_config=pre_epo,
        fusion_config=fusion_cfg,
        hybrid_epochs=10,
        seed=42,
    )
    trainer = STSoilTrainer(trainer_cfg)
    folds = GroupedSplitBuilder.group_kfold_split(dataset, n_splits=4, group_by="site_id")
    bench = trainer.run_benchmark(
        dataset,
        split_folds=folds,
        include_optional_models=True,
        wet_dataset=wet_ds,
        dry_dataset=dry_ds,
    )
    best_name = bench.evidence_ranking[0]
    best_entry = bench.entries[best_name]
    print(f"[5] CV Kıyaslama lideri (tc_pct): {best_name} (OOF RMSE={best_entry.oof_metrics.rmse:.4f}%)")

    # Conformal default_alpha değeri %90 (0.10) vs %95 (0.05) aralık genişliğini kontrol eder
    conf_90 = ConformalUncertaintyCalibrator(
        ConformalConfig(default_alpha=0.10, method="normalized")
    ).fit(y_tc, best_entry.oof_predictions, pred_std_cal=best_entry.oof_uncertainties)

    ood = OutOfDomainDetector(
        OODConfig(
            mahalanobis_quantile=0.99,
            knn_k=5,
            knn_quantile=0.99,
            shrinkage=0.15,
            fail_on_ood=False,
        )
    ).fit(fused.X_fused, measurements=dataset.records)

    with tempfile.TemporaryDirectory(prefix="stds_py_params_") as tmp_dir:
        art_dir = save_artifact(
            Path(tmp_dir) / "bundle",
            metadata=ArtifactMetadata(
                artifact_id="py_param_demo_bundle",
                semantic_version="0.1.0",
                schema_version=SCHEMA_VERSION,
                instrument_profile_id=dataset.instrument_profile_id,
                calibration_id=dataset.records[0].calibration_id,
                pipeline_version=pre_epo.pipeline_version,
                dataset_hash=dataset.compute_sha256(),
                dependency_versions=collect_dependency_versions(),
                seed=42,
            ),
            instrument_profile=get_builtin_profile(dataset.instrument_profile_id),
            model=models["plsr_k6"],
            chemometric_pipeline=chemo_epo,
            fusion_pipeline=fusion,
            conformal_calibrator=conf_90,
            ood_detector=ood,
            validation_metrics=best_entry.oof_metrics.to_dict(),
            background_features=fused.X_fused,
            background_targets=y_tc,
        )
        analyzer = STSoilAnalyzer.from_artifact(art_dir)
        # Aynı modelle hem %90 (alpha=0.10) hem %95 (alpha=0.05) güven aralığı karşılaştırması:
        r90 = analyzer.predict(dataset.records[0], include_xai=True, include_stock=True, alpha=0.10)
        r95 = analyzer.predict(dataset.records[0], include_xai=False, include_stock=False, alpha=0.05)
        ci90 = r90.carbon_metrics["confidence_interval"]
        ci95 = r95.carbon_metrics["confidence_interval"]
        print(
            f"[6] alpha=0.10 yarı-genişlik={ci90['half_width_pct']:.3f}% vs "
            f"alpha=0.05 yarı-genişlik={ci95['half_width_pct']:.3f}%"
        )

    # VisualizationConfig örneği (tüm alanların görünümü)
    viz_cfg = VisualizationConfig(
        target_name="tc_pct",
        model_name="plsr",
        n_folds=4,
        formats=("png",),
        dpi=130,
        include_html_dashboard=True,
        include_interactive_3d=False,
        include_optional_models=True,
        include_pdf_report=True,
        lang="tr",
        seed=42,
    )
    print(f"[7] VisualizationConfig örnek nesnesi: target={viz_cfg.target_name}, model={viz_cfg.model_name}, lang={viz_cfg.lang}")


if __name__ == "__main__":
    main()
