"""Spectral, chemometric, EPO moisture suppression, QC, and covariate visualizations (Plots 01-07)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from st_ds.chemometrics.epo import EPO, EPOConfig
from st_ds.chemometrics.pipeline import ChemometricPipeline
from st_ds.chemometrics.savgol import savgol_derivative
from st_ds.chemometrics.transforms import snv_transform, to_absorbance
from st_ds.i18n import pick_text, t
from st_ds.io.dataset import SoilDataset
from st_ds.signal.qc import SignalQCReport, evaluate_signal_qc
from st_ds.visualization.base import (
    PALETTE,
    DatasetArrays,
    PlotArtifactRecord,
    VisualizationConfig,
    sanitize_label_map,
    save_figure,
)

# Diagnostic Vis-NIR-SWIR wavelengths (nm) referenced in soil spectroscopy literature
_DIAGNOSTIC_BANDS_NM: tuple[tuple[float, float, str, str, str], ...] = (
    (450.0, 580.0, "Demir Oksit / Organik C", "Iron Oxide / Organic C", "#E69F00"),
    (1380.0, 1500.0, "O-H Nem Bandı (~1450 nm)", "O-H Moisture Band (~1450 nm)", "#56B4E9"),
    (1880.0, 2000.0, "H2O Kombinasyon (~1940 nm)", "H2O Combination (~1940 nm)", "#0072B2"),
    (2160.0, 2240.0, "Kil Al-OH (~2200 nm)", "Clay Al-OH (~2200 nm)", "#009E73"),
    (2300.0, 2380.0, "Karbonat CO3 (~2340 nm)", "Carbonate CO3 (~2340 nm)", "#CC79A7"),
)


def _downsample_band_indices(n_bands: int, max_bands: int = 36) -> np.ndarray:
    if n_bands <= max_bands:
        return np.arange(n_bands, dtype=int)
    return np.unique(np.linspace(0, n_bands - 1, max_bands, dtype=int))


def plot_01_raw_spectra_and_bands(
    arrays: DatasetArrays,
    *,
    plots_dir: Path,
    config: VisualizationConfig,
    dataset_source: str,
) -> PlotArtifactRecord:
    """Plot 01: Raw spectra envelopes grouped by moisture/soil category with diagnostic bands."""
    wl = arrays.wavelengths_nm
    X = arrays.spectra
    groups = arrays.moisture_scenarios
    uniq_groups = sorted(set(groups))
    if len(uniq_groups) <= 1:
        groups = arrays.soil_classes
        uniq_groups = sorted(set(groups))

    color_map = sanitize_label_map(groups)
    fig, (ax_main, ax_hist) = plt.subplots(
        1,
        2,
        figsize=(13.5, 5.2),
        gridspec_kw={"width_ratios": [2.4, 1.0]},
    )
    fig.suptitle(
        pick_text(
            config.lang,
            tr="01. Ham Spektrumlar, Grup Zarfları ve Tanısal Kimyasal Bantlar",
            en="01. Raw Spectra, Group Envelopes & Diagnostic Chemical Bands",
        ),
        fontsize=13,
        fontweight="bold",
        color=PALETTE["navy"],
    )

    # Shaded soil absorption bands if overlapping dataset wavelength window
    wl_min, wl_max = float(np.min(wl)), float(np.max(wl))
    for lo, hi, label_tr, label_en, col in _DIAGNOSTIC_BANDS_NM:
        if hi >= wl_min and lo <= wl_max:
            ax_main.axvspan(max(lo, wl_min), min(hi, wl_max), color=col, alpha=0.10, label=pick_text(config.lang, tr=label_tr, en=label_en))

    # Overlay light individual traces
    rng = np.random.default_rng(config.seed)
    sample_idx = np.arange(X.shape[0])
    if sample_idx.size > config.max_spectra_overlay:
        sample_idx = rng.choice(sample_idx, size=config.max_spectra_overlay, replace=False)
    for idx in sample_idx:
        grp = groups[idx]
        ax_main.plot(wl, X[idx], color=color_map.get(grp, PALETTE["slate"]), alpha=0.18, linewidth=0.8)

    # Group mean +/- 1 SD envelope
    for grp in uniq_groups[:7]:
        mask = np.array([g == grp for g in groups], dtype=bool)
        if not np.any(mask):
            continue
        mu = np.mean(X[mask], axis=0)
        sd = np.std(X[mask], axis=0)
        col = color_map.get(grp, PALETTE["navy"])
        ax_main.plot(
            wl,
            mu,
            color=col,
            linewidth=2.2,
            label=pick_text(config.lang, tr=f"Ort ± SS: {grp} (n={int(np.sum(mask))})", en=f"Mean ± SD: {grp} (n={int(np.sum(mask))})"),
        )
        ax_main.fill_between(wl, mu - sd, mu + sd, color=col, alpha=0.18)

    ax_main.set_title(pick_text(config.lang, tr="Dalga Boyu Bazında Yansıma / Sinyal Dağılımı", en="Reflectance / Signal Distribution across Wavelengths"))
    ax_main.set_xlabel(pick_text(config.lang, tr="Dalga Boyu (nm)", en="Wavelength (nm)"))
    ax_main.set_ylabel(pick_text(config.lang, tr="Ham Spektral Değer", en="Raw Spectral Value"))
    ax_main.grid(True, alpha=0.5)
    ax_main.legend(loc="upper right", fontsize=7.6, ncol=2)

    # Right panel: Bandwise mean SNR proxy / Coefficient of Variation + Target SOC histogram
    finite_soc = arrays.soc[np.isfinite(arrays.soc)]
    if finite_soc.size > 0:
        ax_hist.hist(finite_soc, bins=min(15, max(5, finite_soc.size // 3)), color=PALETTE["teal"], edgecolor="#FFFFFF", alpha=0.88)
        ax_hist.axvline(float(np.mean(finite_soc)), color=PALETTE["vermilion"], linestyle="--", linewidth=1.6, label=pick_text(config.lang, tr=f"Ort={np.mean(finite_soc):.2f}%", en=f"Mean={np.mean(finite_soc):.2f}%"))
        ax_hist.axvline(float(np.median(finite_soc)), color=PALETTE["navy"], linestyle=":", linewidth=1.6, label=f"Med={np.median(finite_soc):.2f}%")
        ax_hist.set_title(pick_text(config.lang, tr="Referans SOC (%) Dağılımı", en="Reference SOC (%) Distribution"))
        ax_hist.set_xlabel(pick_text(config.lang, tr="SOC (% kuru madde)", en="SOC (% dry weight)"))
        ax_hist.set_ylabel(pick_text(config.lang, tr="Örnek Sayısı", en="Sample Count"))
        ax_hist.legend(loc="upper right")
    else:
        band_cv = np.std(X, axis=0) / np.maximum(np.abs(np.mean(X, axis=0)), 1e-6)
        ax_hist.plot(wl, band_cv, color=PALETTE["vermilion"], linewidth=1.6)
        ax_hist.set_title(pick_text(config.lang, tr="Bant Bazında Varyasyon Katsayısı", en="Bandwise Coefficient of Variation"))
        ax_hist.set_xlabel(pick_text(config.lang, tr="Dalga Boyu (nm)", en="Wavelength (nm)"))
        ax_hist.set_ylabel(pick_text(config.lang, tr="CV (SS / |Ort|)", en="CV (SD / |Mean|)"))
    ax_hist.grid(True, alpha=0.5)

    return save_figure(
        fig,
        plots_dir,
        "01_ham_spektrumlar_ve_kimyasal_bantlar",
        config=config,
        plot_id="01_raw_spectra_and_bands",
        title=pick_text(config.lang, tr="01. Ham Spektrumlar ve Tanısal Bantlar", en="01. Raw Spectra & Diagnostic Bands"),
        category=t("cat_spectral", config.lang),
        description=pick_text(
            config.lang,
            tr="Grup bazlı ortalama ± standart sapma zarfları, su/kil/karbonat bantları ve SOC dağılımı.",
            en="Group mean ± standard deviation envelopes, water/clay/carbonate absorption bands, and SOC distribution.",
        ),
        dataset_source=dataset_source,
    )


def plot_02_chemometric_pipeline_stages(
    arrays: DatasetArrays,
    *,
    plots_dir: Path,
    config: VisualizationConfig,
    dataset_source: str,
) -> PlotArtifactRecord:
    """Plot 02: Step-by-step chemometric progression (Raw -> Absorbance -> SNV -> SavGol 1st Deriv)."""
    wl = arrays.wavelengths_nm
    X_raw = arrays.spectra
    rng = np.random.default_rng(config.seed)
    idx = rng.choice(X_raw.shape[0], size=min(12, X_raw.shape[0]), replace=False)

    X_sub = X_raw[idx]
    X_abs = to_absorbance(X_sub)
    X_snv = snv_transform(X_abs)
    window = min(7, X_raw.shape[1] if X_raw.shape[1] % 2 == 1 else max(3, X_raw.shape[1] - 1))
    window = max(5, window) if X_raw.shape[1] >= 5 else 3
    poly = min(2, window - 1)
    X_sg = savgol_derivative(X_snv, wl, window=window, polyorder=poly, deriv=1)

    stages = [
        (pick_text(config.lang, tr="1) Ham Spektrum R(λ)", en="1) Raw Spectrum R(λ)"), X_sub, pick_text(config.lang, tr="Ham Yansıma", en="Raw Reflectance"), PALETTE["navy"]),
        (pick_text(config.lang, tr="2) Görünür Absorbans: log10(1 / R)", en="2) Apparent Absorbance: log10(1 / R)"), X_abs, pick_text(config.lang, tr="Absorbans (AU)", en="Absorbance (AU)"), PALETTE["blue"]),
        (pick_text(config.lang, tr="3) Standart Normal Variate (SNV)", en="3) Standard Normal Variate (SNV)"), X_snv, pick_text(config.lang, tr="SNV Normalize", en="SNV Normalized"), PALETTE["teal"]),
        (pick_text(config.lang, tr=f"4) Savitzky-Golay 1. Türev (w={window})", en=f"4) Savitzky-Golay 1st Deriv (w={window})"), X_sg, "dA / dλ", PALETTE["vermilion"]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), sharex=True)
    fig.suptitle(
        pick_text(
            config.lang,
            tr="02. Kemometrik Ön İşleme Dönüşüm Aşamaları (Ham → Absorbans → SNV → Türev)",
            en="02. Chemometric Preprocessing Transformation Stages (Raw → Absorbance → SNV → Derivative)",
        ),
        fontsize=13,
        fontweight="bold",
        color=PALETTE["navy"],
    )

    for ax, (st_title, mat, ylab, col) in zip(axes.ravel(), stages):
        for r in range(mat.shape[0]):
            ax.plot(wl, mat[r], color=col, alpha=0.35, linewidth=0.9)
        ax.plot(wl, np.mean(mat, axis=0), color="#1A202C", linewidth=2.0, label=pick_text(config.lang, tr="Ortalama Eğri", en="Mean Curve"))
        ax.set_title(st_title)
        ax.set_ylabel(ylab)
        ax.set_xlabel(pick_text(config.lang, tr="Dalga Boyu (nm)", en="Wavelength (nm)"))
        ax.grid(True, alpha=0.5)
        ax.legend(loc="upper right", fontsize=8)

    return save_figure(
        fig,
        plots_dir,
        "02_kemometri_donusum_asamalari",
        config=config,
        plot_id="02_chemometric_stages",
        title=pick_text(config.lang, tr="02. Kemometrik Dönüşüm Aşamaları", en="02. Chemometric Transformation Stages"),
        category=t("cat_spectral", config.lang),
        description=pick_text(
            config.lang,
            tr="Saçılma ve taban çizgisi sürüklenmesinin Absorbans, SNV ve SavGol türevi ile adım adım giderilmesi.",
            en="Step-by-step scatter and baseline drift removal via Absorbance, SNV, and Savitzky-Golay derivative.",
        ),
        dataset_source=dataset_source,
    )


def plot_03_epo_moisture_suppression(
    arrays: DatasetArrays,
    *,
    plots_dir: Path,
    config: VisualizationConfig,
    dataset_source: str,
    chem_pipeline: ChemometricPipeline | None = None,
    wet_dataset: SoilDataset | None = None,
    dry_dataset: SoilDataset | None = None,
) -> PlotArtifactRecord:
    """Plot 03: EPO / moisture suppression diagnostics comparing wet vs dry before/after projection."""
    wl = arrays.wavelengths_nm
    pipe = chem_pipeline
    X_wet: np.ndarray | None = None
    X_dry: np.ndarray | None = None

    if wet_dataset is not None and dry_dataset is not None and len(wet_dataset.records) > 0 and len(dry_dataset.records) > 0:
        X_wet = np.vstack([np.asarray(r.spectrum.values, dtype=float) for r in wet_dataset.records])
        X_dry = np.vstack([np.asarray(r.spectrum.values, dtype=float) for r in dry_dataset.records])
    else:
        scen_lower = np.array([s.lower() for s in arrays.moisture_scenarios])
        wet_mask = np.array(["wet" in s or "moist" in s or "nem" in s or "islak" in s for s in scen_lower])
        dry_mask = np.array(["dry" in s or "kuru" in s for s in scen_lower])
        if np.sum(wet_mask) >= 2 and np.sum(dry_mask) >= 2:
            X_wet = arrays.spectra[wet_mask]
            X_dry = arrays.spectra[dry_mask]
        elif np.all(np.isfinite(arrays.moisture_pct)) and len(arrays.records) >= 6:
            med_m = float(np.median(arrays.moisture_pct))
            X_wet = arrays.spectra[arrays.moisture_pct >= med_m]
            X_dry = arrays.spectra[arrays.moisture_pct < med_m]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    fig.suptitle(
        pick_text(
            config.lang,
            tr="03. Dış Parametre Ortogonalizasyonu (EPO) ve Nem Etkisi Baskılama Analizi",
            en="03. External Parameter Orthogonalization (EPO) & Moisture Suppression Analysis",
        ),
        fontsize=13,
        fontweight="bold",
        color=PALETTE["navy"],
    )

    if X_wet is not None and X_dry is not None and X_wet.shape[0] >= 2 and X_dry.shape[0] >= 2:
        tmp_pipe = pipe if pipe is not None and pipe.is_fitted else ChemometricPipeline()
        sig = tmp_pipe.space_signature or tmp_pipe.compute_space_signature(wl)
        Z_wet_before = tmp_pipe.apply_deterministic_transforms(X_wet, wl)
        Z_dry_before = tmp_pipe.apply_deterministic_transforms(X_dry, wl)

        if tmp_pipe.epo is not None:
            epo_model = tmp_pipe.epo
        else:
            n_pair = min(Z_wet_before.shape[0], Z_dry_before.shape[0])
            epo_model = EPO(EPOConfig(q=min(3, max(1, n_pair - 1)), max_q=5)).fit(
                Z_wet_before[:n_pair],
                Z_dry_before[:n_pair],
                space_signature=sig,
            )

        Z_wet_after = epo_model.transform(Z_wet_before, space_signature=sig)
        Z_dry_after = epo_model.transform(Z_dry_before, space_signature=sig)

        diff_before = np.mean(Z_wet_before, axis=0) - np.mean(Z_dry_before, axis=0)
        diff_after = np.mean(Z_wet_after, axis=0) - np.mean(Z_dry_after, axis=0)

        norm_before = float(np.linalg.norm(diff_before))
        norm_after = float(np.linalg.norm(diff_after))
        reduction_pct = (1.0 - norm_after / max(norm_before, 1e-12)) * 100.0

        ax1.plot(wl, diff_before, color=PALETTE["vermilion"], linewidth=1.8, label=pick_text(config.lang, tr=f"EPO Öncesi Fark (||Δ||={norm_before:.3f})", en=f"Pre-EPO Diff (||Δ||={norm_before:.3f})"))
        ax1.plot(wl, diff_after, color=PALETTE["teal"], linewidth=2.0, label=pick_text(config.lang, tr=f"EPO Sonrası Artık (||Δ||={norm_after:.3f})", en=f"Post-EPO Residual (||Δ||={norm_after:.3f})"))
        ax1.axhline(0.0, color="#718096", linestyle=":", linewidth=1.0)
        ax1.set_title(pick_text(config.lang, tr=f"Islak − Kuru Ortalama Fark Spektrumu (Baskılama: %{reduction_pct:.1f})", en=f"Wet − Dry Mean Difference Spectrum (Suppression: {reduction_pct:.1f}%)"))
        ax1.set_xlabel(pick_text(config.lang, tr="Dalga Boyu (nm)", en="Wavelength (nm)"))
        ax1.set_ylabel(pick_text(config.lang, tr="Fark Genliği (Ön İşlenmiş Uzay)", en="Diff Amplitude (Preprocessed Space)"))
        ax1.grid(True, alpha=0.5)
        ax1.legend(loc="best")

        ratios = np.asarray(epo_model.explained_variance_ratios_ if hasattr(epo_model, "explained_variance_ratios_") else [], dtype=float)
        if ratios.size == 0 and getattr(epo_model, "singular_values_", None) is not None:
            sv2 = np.square(np.asarray(epo_model.singular_values_, dtype=float))
            ratios = sv2 / max(float(np.sum(sv2)), 1e-12)
        if ratios.size > 0:
            comps = [f"EPO-{i+1}" for i in range(ratios.size)]
            bars = ax2.bar(comps, ratios * 100.0, color=PALETTE["blue"], edgecolor="#FFFFFF")
            for b, v in zip(bars, ratios * 100.0):
                ax2.text(b.get_x() + b.get_width() / 2.0, v + 0.8, f"{v:.1f}%", ha="center", va="bottom", fontsize=8.5)
            ax2.set_title(pick_text(config.lang, tr=f"Nem Farkı Tekil Değer Açıklanan Varyans (q={ratios.size})", en=f"Moisture Difference SVD Explained Variance (q={ratios.size})"))
            ax2.set_ylabel(pick_text(config.lang, tr="Açıklanan Nem Varyansı (%)", en="Explained Moisture Variance (%)"))
        else:
            ax2.bar([pick_text(config.lang, tr="EPO Öncesi", en="Pre-EPO"), pick_text(config.lang, tr="EPO Sonrası", en="Post-EPO")], [norm_before, norm_after], color=[PALETTE["vermilion"], PALETTE["teal"]])
            ax2.set_title(pick_text(config.lang, tr="L2 Fark Normu Karşılaştırması", en="L2 Difference Norm Comparison"))
            ax2.set_ylabel("||Δ||₂")
        ax2.grid(True, axis="y", alpha=0.5)
    else:
        ax1.text(
            0.5,
            0.5,
            pick_text(
                config.lang,
                tr="Ayrı Islak/Kuru alt grupları bulunamadı;\nmevcut veri tekil nem koşulu altında.",
                en="Separate Wet/Dry subgroups not found;\ncurrent dataset represents a single moisture state.",
            ),
            ha="center",
            va="center",
        )
        ax1.axis("off")
        ax2.axis("off")

    return save_figure(
        fig,
        plots_dir,
        "03_epo_nem_baskilama_karsilastirma",
        config=config,
        plot_id="03_epo_moisture_suppression",
        title=pick_text(config.lang, tr="03. EPO Nem Etkisi Baskılama Analizi", en="03. EPO Moisture Suppression Analysis"),
        category=t("cat_spectral", config.lang),
        description=pick_text(
            config.lang,
            tr="EPO projeksiyonu öncesi ve sonrası ıslak-kuru fark spektrumu ve nem bileşeni varyans oranı.",
            en="Wet-vs-dry difference spectrum before/after EPO projection and moisture component variance ratio.",
        ),
        dataset_source=dataset_source,
    )


def plot_04_spectral_3d_surface(
    arrays: DatasetArrays,
    *,
    plots_dir: Path,
    config: VisualizationConfig,
    dataset_source: str,
) -> PlotArtifactRecord:
    """Plot 04 (3D #1): 3D Spectral Landscape sorted by SOC concentration."""
    wl = arrays.wavelengths_nm
    X = arrays.spectra
    if np.all(np.isfinite(arrays.soc)):
        order = np.argsort(arrays.soc)
        sort_note = pick_text(config.lang, tr="Örnekler artan SOC (%) sırasına göre dizilmiştir", en="Samples ordered by increasing SOC (%)")
    else:
        order = np.arange(X.shape[0])
        sort_note = pick_text(config.lang, tr="Örnekler veri seti sırasına göre dizilmiştir", en="Samples ordered by dataset index")

    max_rows = min(40, order.size)
    row_sel = order[np.linspace(0, order.size - 1, max_rows, dtype=int)]
    col_sel = _downsample_band_indices(wl.size, max_bands=48)

    W, S = np.meshgrid(wl[col_sel], np.arange(row_sel.size))
    Z = X[np.ix_(row_sel, col_sel)]

    fig = plt.figure(figsize=(11.5, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(W, S, Z, cmap="viridis", edgecolor="none", alpha=0.92, antialiased=True)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.62, pad=0.08)
    cbar.set_label(pick_text(config.lang, tr="Ham Spektral Değer", en="Raw Spectral Value"), fontsize=9.5)

    ax.set_title(
        pick_text(
            config.lang,
            tr="04. 3D Spektral Yüzey Topografyası (Dalga Boyu × Örnek İndeksi × Yansıma)",
            en="04. 3D Spectral Surface Topography (Wavelength × Sample Index × Reflectance)",
        ),
        fontsize=12.5,
        fontweight="bold",
        color=PALETTE["navy"],
        pad=12,
    )
    ax.set_xlabel(pick_text(config.lang, tr="Dalga Boyu (nm)", en="Wavelength (nm)"), labelpad=8)
    ax.set_ylabel(pick_text(config.lang, tr="SOC Sıralı Örnek İndeksi", en="SOC-Sorted Sample Index"), labelpad=8)
    ax.set_zlabel(pick_text(config.lang, tr="Spektral Değer", en="Spectral Value"), labelpad=8)
    ax.view_init(elev=27, azim=-125)

    return save_figure(
        fig,
        plots_dir,
        "04_spektral_3d_yuzey",
        config=config,
        plot_id="04_spectral_3d_surface",
        title=pick_text(config.lang, tr="04. 3D Spektral Yüzey Topografyası (3D)", en="04. 3D Spectral Surface Topography (3D)"),
        category=t("cat_spectral", config.lang),
        is_3d=True,
        description=pick_text(
            config.lang,
            tr=f"Dalga boyu ve SOC sıralı örnekler boyunca üç boyutlu spektral yüzey. {sort_note}.",
            en=f"Three-dimensional spectral landscape across wavelengths and SOC-sorted samples. {sort_note}.",
        ),
        dataset_source=dataset_source,
        subtitle=sort_note,
    )


def plot_05_wavelength_correlation_matrix(
    arrays: DatasetArrays,
    *,
    plots_dir: Path,
    config: VisualizationConfig,
    dataset_source: str,
) -> PlotArtifactRecord:
    """Plot 05: Band-by-band Pearson correlation heatmap + per-wavelength correlation with SOC."""
    wl = arrays.wavelengths_nm
    X = arrays.spectra
    sel = _downsample_band_indices(wl.size, max_bands=32)
    X_sub = X[:, sel]
    wl_sub = wl[sel]

    corr = np.corrcoef(X_sub, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)

    fig, (ax_heat, ax_soc) = plt.subplots(
        1,
        2,
        figsize=(13.5, 5.4),
        gridspec_kw={"width_ratios": [1.3, 1.0]},
    )
    fig.suptitle(
        pick_text(
            config.lang,
            tr="05. Dalga Boyu İçi Eşdoğrusallık Matrisi ve SOC Hedefi ile Korelasyon Profili",
            en="05. Wavelength Collinearity Matrix & SOC Target Correlation Profile",
        ),
        fontsize=13,
        fontweight="bold",
        color=PALETTE["navy"],
    )

    im = ax_heat.imshow(corr, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r")
    tick_step = max(1, wl_sub.size // 8)
    ticks = np.arange(0, wl_sub.size, tick_step)
    labels = [f"{wl_sub[t]:.0f}" for t in ticks]
    ax_heat.set_xticks(ticks)
    ax_heat.set_xticklabels(labels, rotation=45, ha="right")
    ax_heat.set_yticks(ticks)
    ax_heat.set_yticklabels(labels)
    ax_heat.set_title(pick_text(config.lang, tr="Bantlar Arası Korelasyon Matrisi", en="Inter-Band Correlation Matrix"))
    ax_heat.set_xlabel(pick_text(config.lang, tr="Dalga Boyu (nm)", en="Wavelength (nm)"))
    ax_heat.set_ylabel(pick_text(config.lang, tr="Dalga Boyu (nm)", en="Wavelength (nm)"))

    # Right panel: correlation of every wavelength with target SOC
    mask_soc = np.isfinite(arrays.soc)
    if np.sum(mask_soc) >= 4 and float(np.std(arrays.soc[mask_soc])) > 1e-8:
        y = arrays.soc[mask_soc]
        X_m = X[mask_soc]
        y_c = y - np.mean(y)
        X_c = X_m - np.mean(X_m, axis=0)
        denom = np.sqrt(np.sum(np.square(X_c), axis=0) * np.sum(np.square(y_c)))
        r_vals = np.where(denom > 1e-12, np.sum(X_c * y_c[:, None], axis=0) / np.maximum(denom, 1e-12), 0.0)
        ax_soc.plot(wl, r_vals, color=PALETTE["navy"], linewidth=1.8)
        ax_soc.fill_between(wl, 0.0, r_vals, where=(r_vals >= 0), color=PALETTE["teal"], alpha=0.25, label=pick_text(config.lang, tr="Pozitif r", en="Positive r"))
        ax_soc.fill_between(wl, 0.0, r_vals, where=(r_vals < 0), color=PALETTE["vermilion"], alpha=0.25, label=pick_text(config.lang, tr="Negatif r", en="Negative r"))
        peak_idx = int(np.argmax(np.abs(r_vals)))
        ax_soc.scatter([wl[peak_idx]], [r_vals[peak_idx]], color=PALETTE["orange"], zorder=5, s=55)
        ax_soc.annotate(
            pick_text(config.lang, tr=f"Tepe: {wl[peak_idx]:.0f} nm\n(r={r_vals[peak_idx]:.2f})", en=f"Peak: {wl[peak_idx]:.0f} nm\n(r={r_vals[peak_idx]:.2f})"),
            xy=(wl[peak_idx], r_vals[peak_idx]),
            xytext=(12, 12 if r_vals[peak_idx] < 0.7 else -24),
            textcoords="offset points",
            fontsize=8.2,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "#FFFFFF", "edgecolor": PALETTE["orange"]},
        )
        ax_soc.axhline(0.0, color="#718096", linestyle=":")
        ax_soc.set_ylim(-1.05, 1.05)
        ax_soc.legend(loc="best")
        ax_soc.set_title(pick_text(config.lang, tr="Her Dalga Boyunun SOC (%) ile Korelasyonu", en="Per-Wavelength Correlation with SOC (%)"))
        ax_soc.set_xlabel(pick_text(config.lang, tr="Dalga Boyu (nm)", en="Wavelength (nm)"))
        ax_soc.set_ylabel(pick_text(config.lang, tr="Pearson r (Spektrum vs SOC)", en="Pearson r (Spectrum vs SOC)"))
    else:
        ax_soc.text(0.5, 0.5, pick_text(config.lang, tr="SOC korelasyonu için yeterli hedef değer yok.", en="Insufficient reference SOC values for correlation."), ha="center", va="center")
        ax_soc.axis("off")
    ax_soc.grid(True, alpha=0.5)

    return save_figure(
        fig,
        plots_dir,
        "05_dalga_boyu_korelasyon_matrisi",
        config=config,
        plot_id="05_wavelength_correlation",
        title=pick_text(config.lang, tr="05. Dalga Boyu Korelasyon Matrisi", en="05. Wavelength Correlation Matrix"),
        category=t("cat_spectral", config.lang),
        description=pick_text(
            config.lang,
            tr="Spektral bant eşdoğrusallık ısı haritası ve SOC korelasyon profili.",
            en="Spectral band collinearity heatmap and SOC correlation profile.",
        ),
        dataset_source=dataset_source,
    )


def plot_06_signal_qc_diagnostics(
    arrays: DatasetArrays,
    *,
    plots_dir: Path,
    config: VisualizationConfig,
    dataset_source: str,
) -> PlotArtifactRecord:
    """Plot 06: Signal QC score distribution, SNR vs SOC, and QC status breakdown."""
    qc_reports: list[SignalQCReport] = [evaluate_signal_qc(rec.spectrum) for rec in arrays.records]
    qc_scores = np.array([q.qc_score for q in qc_reports], dtype=float)
    snr_vals = np.array([q.snr_estimate for q in qc_reports], dtype=float)
    statuses = [q.status.value for q in qc_reports]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.0))
    fig.suptitle(
        pick_text(
            config.lang,
            tr="06. Spektral Sinyal Kalite Kontrol (QC) ve SNR Teşhis Analizi",
            en="06. Spectral Signal Quality Control (QC) & SNR Diagnostics",
        ),
        fontsize=13,
        fontweight="bold",
        color=PALETTE["navy"],
    )

    ax1.hist(qc_scores, bins=12, range=(0.0, 1.0), color=PALETTE["blue"], edgecolor="#FFFFFF", alpha=0.88)
    ax1.axvline(0.5, color=PALETTE["vermilion"], linestyle="--", linewidth=1.6, label=pick_text(config.lang, tr="Uyarı Eşiği (0.50)", en="Warning Threshold (0.50)"))
    ax1.set_title(pick_text(config.lang, tr=f"QC Skoru Dağılımı (Ort={np.mean(qc_scores):.3f}, Min={np.min(qc_scores):.3f})", en=f"QC Score Distribution (Mean={np.mean(qc_scores):.3f}, Min={np.min(qc_scores):.3f})"))
    ax1.set_xlabel(pick_text(config.lang, tr="Sinyal QC Skoru [0-1]", en="Signal QC Score [0-1]"))
    ax1.set_ylabel(pick_text(config.lang, tr="Örnek Sayısı", en="Sample Count"))
    ax1.grid(True, alpha=0.5)
    ax1.legend(loc="upper left")

    status_colors = {"OK": PALETTE["teal"], "WARNING": PALETTE["orange"], "UNRELIABLE": PALETTE["vermilion"]}
    if np.all(np.isfinite(arrays.soc)):
        for st_val in sorted(set(statuses)):
            m = np.array([s == st_val for s in statuses])
            ax2.scatter(
                arrays.soc[m],
                snr_vals[m],
                label=f"{st_val} (n={int(np.sum(m))})",
                color=status_colors.get(st_val, PALETTE["navy"]),
                s=46,
                alpha=0.85,
                edgecolor="#FFFFFF",
            )
        ax2.set_xlabel(pick_text(config.lang, tr="Referans SOC (%)", en="Reference SOC (%)"))
    else:
        for st_val in sorted(set(statuses)):
            m = np.array([s == st_val for s in statuses])
            ax2.scatter(
                np.where(m)[0],
                snr_vals[m],
                label=f"{st_val} (n={int(np.sum(m))})",
                color=status_colors.get(st_val, PALETTE["navy"]),
                s=46,
                alpha=0.85,
            )
        ax2.set_xlabel(pick_text(config.lang, tr="Örnek İndeksi", en="Sample Index"))
    ax2.set_ylabel(pick_text(config.lang, tr="Tahmini Sinyal / Gürültü Oranı (SNR)", en="Estimated Signal-to-Noise Ratio (SNR)"))
    ax2.set_title(pick_text(config.lang, tr="SNR Profili ve Kalite Kapısı Statüsü", en="SNR Profile & Quality Gate Status"))
    ax2.grid(True, alpha=0.5)
    ax2.legend(loc="best")

    return save_figure(
        fig,
        plots_dir,
        "06_sinyal_qc_teshis",
        config=config,
        plot_id="06_signal_qc_diagnostics",
        title=pick_text(config.lang, tr="06. Sinyal QC ve SNR Teşhis Analizi", en="06. Signal QC & SNR Diagnostics"),
        category=t("cat_spectral", config.lang),
        description=pick_text(
            config.lang,
            tr="Her ölçüm için sinyal kalite skoru, SNR tahmini ve kalite kapısı dağılımı.",
            en="Per-measurement signal QC score, SNR estimate, and quality gate distribution.",
        ),
        dataset_source=dataset_source,
    )


def plot_07_eis_or_covariate_matrix(
    arrays: DatasetArrays,
    *,
    plots_dir: Path,
    config: VisualizationConfig,
    dataset_source: str,
) -> PlotArtifactRecord:
    """Plot 07: EIS Nyquist & Bode (if EIS is present) paired with Covariate & Carbon correlation matrix."""
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13.6, 5.4))
    fig.suptitle(
        pick_text(
            config.lang,
            tr="07. Kovaryat İlişki Matrisi ve Toprak Sensör / EIS Analizi",
            en="07. Covariate Correlation Matrix & Soil Sensor / EIS Analysis",
        ),
        fontsize=13,
        fontweight="bold",
        color=PALETTE["navy"],
    )

    if arrays.eis_frequencies is not None and arrays.eis_z_real is not None and arrays.eis_z_imag is not None:
        z_real = arrays.eis_z_real
        z_imag_neg = -arrays.eis_z_imag
        n_plot = min(15, z_real.shape[0])
        for i in range(n_plot):
            ax_left.plot(z_real[i], z_imag_neg[i], marker="o", markersize=3, alpha=0.55, linewidth=1.2)
        ax_left.plot(np.mean(z_real, axis=0), np.mean(z_imag_neg, axis=0), color=PALETTE["navy"], linewidth=2.3, label=pick_text(config.lang, tr="Ortalama Nyquist", en="Mean Nyquist"))
        ax_left.set_title(pick_text(config.lang, tr="EIS Nyquist Diyagramı (Z' vs −Z'')", en="EIS Nyquist Diagram (Z' vs −Z'')"))
        ax_left.set_xlabel(pick_text(config.lang, tr="Reel Empedans Z' (Ohm)", en="Real Impedance Z' (Ohm)"))
        ax_left.set_ylabel(pick_text(config.lang, tr="Sanal Empedans −Z'' (Ohm)", en="Imaginary Impedance −Z'' (Ohm)"))
        ax_left.legend(loc="best")
        ax_left.grid(True, alpha=0.5)
    else:
        # Plot Moisture vs SOC scatter colored by pH (or texture) when EIS is not present
        valid = np.isfinite(arrays.moisture_pct) & np.isfinite(arrays.soc)
        if np.any(valid):
            sc = ax_left.scatter(
                arrays.moisture_pct[valid],
                arrays.soc[valid],
                c=np.where(np.isfinite(arrays.ph[valid]), arrays.ph[valid], 7.0),
                cmap="plasma",
                s=52,
                alpha=0.88,
                edgecolor="#FFFFFF",
            )
            cbar = fig.colorbar(sc, ax=ax_left, pad=0.03)
            cbar.set_label(pick_text(config.lang, tr="Toprak pH", en="Soil pH"))
            ax_left.set_title(pick_text(config.lang, tr="Volümetrik Nem (%) vs SOC (%) (Renk: pH)", en="Volumetric Moisture (%) vs SOC (%) (Color: pH)"))
            ax_left.set_xlabel(pick_text(config.lang, tr="Nem (%)", en="Moisture (%)"))
            ax_left.set_ylabel(pick_text(config.lang, tr="Referans SOC (%)", en="Reference SOC (%)"))
        else:
            ax_left.text(0.5, 0.5, pick_text(config.lang, tr="EIS ve eşleşmiş nem verisi bulunamadı.", en="No EIS or paired moisture data found."), ha="center", va="center")
            ax_left.axis("off")
        ax_left.grid(True, alpha=0.5)

    # Right: Covariate + Target Carbon Pearson correlation heatmap
    candidates: list[tuple[str, np.ndarray]] = [
        ("SOC (%)", arrays.soc),
        ("SIC (%)", arrays.sic),
        ("TC (%)", arrays.tc),
        (pick_text(config.lang, tr="Nem (%)", en="Moisture (%)"), arrays.moisture_pct),
        ("pH", arrays.ph),
        (pick_text(config.lang, tr="Sıcaklık (°C)", en="Temp (°C)"), arrays.temperature_c),
        ("EC (dS/m)", arrays.ec_ds_m),
        (pick_text(config.lang, tr="Hacim Ağ. (g/cm³)", en="Bulk Dens. (g/cm³)"), arrays.bulk_density),
        (pick_text(config.lang, tr="Kaba Parça", en="Coarse Frag."), arrays.coarse_fraction),
    ]
    kept_names: list[str] = []
    kept_cols: list[np.ndarray] = []
    for cname, arr in candidates:
        fin = np.isfinite(arr)
        if np.sum(fin) >= 4 and float(np.std(arr[fin])) > 1e-8:
            # Impute occasional NaN with column median purely for pairwise visualization stability
            filled = np.where(fin, arr, float(np.median(arr[fin])))
            kept_names.append(cname)
            kept_cols.append(filled)

    if len(kept_cols) >= 2:
        mat = np.column_stack(kept_cols)
        corr = np.nan_to_num(np.corrcoef(mat, rowvar=False), nan=0.0)
        im = ax_right.imshow(corr, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        fig.colorbar(im, ax=ax_right, fraction=0.046, pad=0.04).set_label("Pearson r")
        ax_right.set_xticks(np.arange(len(kept_names)))
        ax_right.set_xticklabels(kept_names, rotation=35, ha="right", fontsize=8.4)
        ax_right.set_yticks(np.arange(len(kept_names)))
        ax_right.set_yticklabels(kept_names, fontsize=8.4)
        for r_i in range(len(kept_names)):
            for c_i in range(len(kept_names)):
                val = corr[r_i, c_i]
                ax_right.text(
                    c_i,
                    r_i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.3,
                    color="#FFFFFF" if abs(val) > 0.55 else "#1A202C",
                )
        ax_right.set_title(pick_text(config.lang, tr="Kovaryat ve Karbon Fraksiyonu Korelasyon Matrisi", en="Covariate & Carbon Fraction Correlation Matrix"))
    else:
        ax_right.text(0.5, 0.5, pick_text(config.lang, tr="Korelasyon matrisi için yeterli sayısal değişken yok.", en="Insufficient numeric variables for correlation matrix."), ha="center", va="center")
        ax_right.axis("off")

    return save_figure(
        fig,
        plots_dir,
        "07_eis_veya_kovaryat_matrisi",
        config=config,
        plot_id="07_eis_covariate_matrix",
        title=pick_text(config.lang, tr="07. EIS ve Kovaryat Korelasyon Matrisi", en="07. EIS & Covariate Correlation Matrix"),
        category=t("cat_spectral", config.lang),
        description=pick_text(
            config.lang,
            tr="EIS Nyquist spektrumu (varsa) veya nem-pH-SOC saçılımı ile kovaryat korelasyon matrisi.",
            en="EIS Nyquist spectrum (when present) or moisture-pH-SOC scatter alongside covariate correlation heatmap.",
        ),
        dataset_source=dataset_source,
    )


def generate_spectral_plots(
    arrays: DatasetArrays,
    *,
    plots_dir: Path,
    config: VisualizationConfig,
    dataset_source: str,
    chem_pipeline: ChemometricPipeline | None = None,
    wet_dataset: SoilDataset | None = None,
    dry_dataset: SoilDataset | None = None,
) -> list[PlotArtifactRecord]:
    """Generate all 7 spectral, chemometric, EPO, QC, and covariate plots."""
    return [
        plot_01_raw_spectra_and_bands(arrays, plots_dir=plots_dir, config=config, dataset_source=dataset_source),
        plot_02_chemometric_pipeline_stages(arrays, plots_dir=plots_dir, config=config, dataset_source=dataset_source),
        plot_03_epo_moisture_suppression(
            arrays,
            plots_dir=plots_dir,
            config=config,
            dataset_source=dataset_source,
            chem_pipeline=chem_pipeline,
            wet_dataset=wet_dataset,
            dry_dataset=dry_dataset,
        ),
        plot_04_spectral_3d_surface(arrays, plots_dir=plots_dir, config=config, dataset_source=dataset_source),
        plot_05_wavelength_correlation_matrix(arrays, plots_dir=plots_dir, config=config, dataset_source=dataset_source),
        plot_06_signal_qc_diagnostics(arrays, plots_dir=plots_dir, config=config, dataset_source=dataset_source),
        plot_07_eis_or_covariate_matrix(arrays, plots_dir=plots_dir, config=config, dataset_source=dataset_source),
    ]
