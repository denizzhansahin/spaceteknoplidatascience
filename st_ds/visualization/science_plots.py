"""Explainable AI (XAI), carbon partitioning/stock uncertainty, variogram & Kriging plots (Plots 17-23)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from st_ds.carbon.stock import STOCK_PARAM_NAMES
from st_ds.domain.schemas import PredictionReport
from st_ds.i18n import pick_text, t
from st_ds.spatial.kriging import KrigingPrediction, ordinary_kriging
from st_ds.spatial.variogram import evaluate_variogram
from st_ds.visualization.base import (
    PALETTE,
    DatasetArrays,
    PlotArtifactRecord,
    VisualizationConfig,
    render_notice_figure,
    save_figure,
)
from st_ds.xai.engine import XAIExplanationReport


@dataclass(frozen=True)
class SpatialPlotBundle:
    """Computed Ordinary Kriging grid and source coordinates (if spatial data allows fitting)."""

    kriging: KrigingPrediction
    sample_coords_m: np.ndarray
    sample_values: np.ndarray
    grid_x_m: np.ndarray
    grid_y_m: np.ndarray


def compute_spatial_kriging_bundle(
    arrays: DatasetArrays,
    target_values: np.ndarray,
    *,
    grid_size: int = 24,
) -> SpatialPlotBundle | None:
    """Aggregate unique spatial sites and compute a 2D Ordinary Kriging surface if feasible."""
    valid = np.isfinite(arrays.longitudes) & np.isfinite(arrays.latitudes) & np.isfinite(target_values)
    if np.sum(valid) < 8:
        return None

    coords_raw = np.column_stack((arrays.longitudes[valid], arrays.latitudes[valid]))
    vals_raw = target_values[valid]

    # Deduplicate identical or sub-1e-5 degree coordinates by averaging measurements per location.
    rounded: dict[tuple[float, float], list[float]] = {}
    for (lon, lat), v in zip(coords_raw, vals_raw):
        key = (round(float(lon), 5), round(float(lat), 5))
        rounded.setdefault(key, []).append(float(v))

    if len(rounded) < 8:
        return None

    uniq_coords = np.array(list(rounded.keys()), dtype=np.float64)
    uniq_vals = np.array([float(np.mean(vs)) for vs in rounded.values()], dtype=np.float64)

    lon_min, lon_max = float(np.min(uniq_coords[:, 0])), float(np.max(uniq_coords[:, 0]))
    lat_min, lat_max = float(np.min(uniq_coords[:, 1])), float(np.max(uniq_coords[:, 1]))
    if (lon_max - lon_min) < 1e-6 and (lat_max - lat_min) < 1e-6:
        return None

    lon_pad = max(1e-4, 0.08 * (lon_max - lon_min))
    lat_pad = max(1e-4, 0.08 * (lat_max - lat_min))
    gx = np.linspace(lon_min - lon_pad, lon_max + lon_pad, grid_size)
    gy = np.linspace(lat_min - lat_pad, lat_max + lat_pad, grid_size)
    GX, GY = np.meshgrid(gx, gy)
    query_coords = np.column_stack((GX.ravel(), GY.ravel()))

    try:
        krig = ordinary_kriging(
            uniq_coords,
            uniq_vals,
            query_coords,
            crs_epsg="EPSG:4326",
            coord_order="lon_lat",
            model="spherical",
            n_lags=6,
            min_points=8,
        )
    except Exception:
        return None

    sample_m = krig.variogram.coordinate_frame.to_local_meters(uniq_coords) if krig.variogram else uniq_coords
    qx_m = krig.query_coords_m[:, 0].reshape(grid_size, grid_size)
    qy_m = krig.query_coords_m[:, 1].reshape(grid_size, grid_size)
    return SpatialPlotBundle(
        kriging=krig,
        sample_coords_m=sample_m,
        sample_values=uniq_vals,
        grid_x_m=qx_m,
        grid_y_m=qy_m,
    )


def plot_17_xai_spectral_sensitivity_ig(
    xai_report: XAIExplanationReport | None,
    arrays: DatasetArrays,
    *,
    plots_dir: Path,
    config: VisualizationConfig,
    dataset_source: str,
) -> PlotArtifactRecord:
    """Plot 17: Spectral Integrated Gradients & Local Sensitivity overlaid with mean spectrum."""
    if xai_report is None:
        return render_notice_figure(
            plots_dir,
            "17_xai_duyarlilik_ve_ig_spektral",
            config=config,
            plot_id="17_xai_spectral",
            title=pick_text(config.lang, tr="17. XAI Spektral Duyarlılık ve Integrated Gradients", en="17. XAI Spectral Sensitivity & Integrated Gradients"),
            category=t("cat_science", config.lang),
            notice_tr=pick_text(
                config.lang,
                tr="XAI raporu oluşturulamadığı için spektral önem grafiği atlandı.",
                en="Spectral XAI plot skipped because no XAI report could be generated.",
            ),
            dataset_source=dataset_source,
        )

    wl = arrays.wavelengths_nm
    sens = xai_report.local_sensitivity
    ig = xai_report.integrated_gradients
    n_spec = wl.size

    # First n_spec entries in fused features correspond to preprocessed spectral bands
    sens_spec = np.abs(np.asarray(sens.normalized_sensitivity[:n_spec], dtype=float))
    ig_raw = ig.attributions if ig.attributions is not None else ig.normalized_sensitivity
    ig_spec = np.asarray(ig_raw[:n_spec], dtype=float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.0, 7.6), sharex=True)
    fig.suptitle(
        pick_text(
            config.lang,
            tr=f"17. Spektral XAI: Integrated Gradients ve Yerel Duyarlılık Profili ({xai_report.target_name})",
            en=f"17. Spectral XAI: Integrated Gradients & Local Sensitivity Profile ({xai_report.target_name})",
        ),
        fontsize=13,
        fontweight="bold",
        color=PALETTE["navy"],
    )

    ax1.bar(wl, ig_spec, width=max(4.0, float(np.median(np.diff(wl))) * 0.8), color=np.where(ig_spec >= 0, PALETTE["teal"], PALETTE["vermilion"]), alpha=0.82, label=pick_text(config.lang, tr="Integrated Gradients Katkısı", en="Integrated Gradients Attribution"))
    ax1.axhline(0.0, color="#718096", linewidth=1.0)
    ax1.set_title(pick_text(config.lang, tr="Integrated Gradients Spektral Atıfları (Referans Tabana Göre Katkı)", en="Integrated Gradients Spectral Attributions (Relative to Baseline)"))
    ax1.set_ylabel(pick_text(config.lang, tr="IG Katkısı (%)", en="IG Contribution (%)"))
    ax1.grid(True, alpha=0.5)
    ax1.legend(loc="upper right")

    ax2_twin = ax2.twinx()
    mean_spec = np.mean(arrays.spectra, axis=0)
    ax2_twin.plot(wl, mean_spec, color=PALETTE["slate"], linestyle="--", alpha=0.55, linewidth=1.4, label=pick_text(config.lang, tr="Ortalama Ham Spektrum", en="Mean Raw Spectrum"))
    ax2_twin.set_ylabel(pick_text(config.lang, tr="Ortalama Ham Değer", en="Mean Raw Value"), color=PALETTE["slate"])

    ax2.plot(wl, sens_spec, color=PALETTE["navy"], linewidth=2.0, label=pick_text(config.lang, tr="Normalize Yerel Duyarlılık |∂Ŷ/∂x|", en="Normalized Local Sensitivity |∂Ŷ/∂x|"))
    top_k = np.argsort(sens_spec)[-min(5, sens_spec.size):]
    ax2.scatter(wl[top_k], sens_spec[top_k], color=PALETTE["orange"], zorder=5, s=48, label=pick_text(config.lang, tr="En Duyarlı 5 Bant", en="Top 5 Sensitive Bands"))
    for idx in top_k:
        ax2.annotate(f"{wl[idx]:.0f} nm", (wl[idx], sens_spec[idx]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7.8)
    ax2.set_title(pick_text(config.lang, tr="Yerel Model Duyarlılığı ve Baskın Dalga Boyları", en="Local Model Sensitivity & Dominant Wavelengths"))
    ax2.set_xlabel(pick_text(config.lang, tr="Dalga Boyu (nm)", en="Wavelength (nm)"))
    ax2.set_ylabel(pick_text(config.lang, tr="Normalize Duyarlılık", en="Normalized Sensitivity"))
    ax2.grid(True, alpha=0.5)
    ax2.legend(loc="upper left")

    return save_figure(
        fig,
        plots_dir,
        "17_xai_duyarlilik_ve_ig_spektral",
        config=config,
        plot_id="17_xai_spectral",
        title=pick_text(config.lang, tr="17. XAI Spektral Duyarlılık ve Integrated Gradients", en="17. XAI Spectral Sensitivity & Integrated Gradients"),
        category=t("cat_science", config.lang),
        description=pick_text(
            config.lang,
            tr="Modelin tahmin üretirken hangi dalga boylarına duyarlı olduğunu gösteren IG ve gradyan profili.",
            en="Integrated Gradients and gradient sensitivity profile showing influential spectral bands.",
        ),
        dataset_source=dataset_source,
    )


def plot_18_xai_hsic_and_component_ranking(
    xai_report: XAIExplanationReport | None,
    *,
    plots_dir: Path,
    config: VisualizationConfig,
    dataset_source: str,
) -> PlotArtifactRecord:
    """Plot 18: Dataset-level normalized HSIC dependence + grouped physical domain ranking."""
    if xai_report is None:
        return render_notice_figure(
            plots_dir,
            "18_xai_hsic_ve_kovaryat_onemi",
            config=config,
            plot_id="18_xai_hsic_ranking",
            title=pick_text(config.lang, tr="18. XAI HSIC Bağımlılığı ve Bileşen Sıralaması", en="18. XAI HSIC Dependence & Component Ranking"),
            category=t("cat_science", config.lang),
            notice_tr=pick_text(config.lang, tr="XAI raporu bulunamadı.", en="XAI report unavailable."),
            dataset_source=dataset_source,
        )

    hsic = xai_report.dataset_dependence_hsic
    ranking = xai_report.component_ranking

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.4))
    fig.suptitle(
        pick_text(
            config.lang,
            tr="18. Küresel HSIC İstatistiksel Bağımlılık ve Fiziksel Bileşen Önem Sıralaması",
            en="18. Global HSIC Statistical Dependence & Physical Component Importance Ranking",
        ),
        fontsize=13,
        fontweight="bold",
        color=PALETTE["navy"],
    )

    if hsic.feature_dependence:
        items = sorted(hsic.feature_dependence.items(), key=lambda kv: kv[1], reverse=True)[:12]
        names = [k for k, _ in items][::-1]
        vals = [v for _, v in items][::-1]
        ax1.barh(names, vals, color=PALETTE["blue"], edgecolor="#FFFFFF")
        ax1.set_title(
            pick_text(
                config.lang,
                tr=f"En Yüksek 12 Değişken — nHSIC (n={hsic.n_samples}, durum={hsic.status})",
                en=f"Top 12 Variables — nHSIC (n={hsic.n_samples}, status={hsic.status})",
            )
        )
        ax1.set_xlabel(pick_text(config.lang, tr="Normalize HSIC (Bağımlılık Gücü [0-1])", en="Normalized HSIC (Dependence Strength [0-1])"))
        ax1.grid(True, axis="x", alpha=0.5)
    else:
        ax1.text(0.5, 0.5, pick_text(config.lang, tr=f"HSIC hesaplanamadı: {hsic.status} (B={hsic.n_samples})", en=f"HSIC unavailable: {hsic.status} (B={hsic.n_samples})"), ha="center", va="center")
        ax1.axis("off")

    # Right panel: separated physical component groups (ADR-006)
    bars_labels: list[str] = []
    bars_scores: list[float] = []
    bars_cols: list[str] = []
    spec_tag = pick_text(config.lang, tr="[Spektral]", en="[Spectral]")
    cov_tag = pick_text(config.lang, tr="[Kovaryat]", en="[Covariate]")
    for item in ranking.spectral_ranking[:6]:
        bars_labels.append(f"{spec_tag} {item.name}")
        bars_scores.append(float(item.score))
        bars_cols.append(PALETTE["teal"])
    for item in ranking.covariate_ranking[:6]:
        bars_labels.append(f"{cov_tag} {item.name} ({item.unit})")
        bars_scores.append(float(item.score))
        bars_cols.append(PALETTE["orange"])
    for item in ranking.derived_eis_ranking[:4]:
        bars_labels.append(f"[EIS] {item.name}")
        bars_scores.append(float(item.score))
        bars_cols.append(PALETTE["purple"])

    if bars_labels:
        y_pos = np.arange(len(bars_labels))
        ax2.barh(y_pos, bars_scores[::-1], color=bars_cols[::-1], edgecolor="#FFFFFF")
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(bars_labels[::-1], fontsize=8.3)
        ax2.set_title(pick_text(config.lang, tr="Birim-Ayrıştırılmış Öznitelik Grubu Sıralaması", en="Unit-Separated Feature Group Ranking"))
        ax2.set_xlabel(pick_text(config.lang, tr="Grup İçi Önem Skoru", en="Within-Group Importance Score"))
        ax2.grid(True, axis="x", alpha=0.5)
    else:
        ax2.axis("off")

    return save_figure(
        fig,
        plots_dir,
        "18_xai_hsic_ve_kovaryat_onemi",
        config=config,
        plot_id="18_xai_hsic_ranking",
        title=pick_text(config.lang, tr="18. XAI HSIC Bağımlılığı ve Bileşen Sıralaması", en="18. XAI HSIC Dependence & Component Ranking"),
        category=t("cat_science", config.lang),
        description=pick_text(
            config.lang,
            tr="RBF HSIC bağımlılık sıralaması (nedensellik değildir) ve spektral/kovaryat bileşen önemleri.",
            en="RBF HSIC statistical dependence ranking (non-causal) and spectral/covariate component importances.",
        ),
        dataset_source=dataset_source,
        subtitle=pick_text(
            config.lang,
            tr="Not: HSIC nedensellik değil istatistiksel bağımlılık ölçer",
            en="Note: HSIC measures statistical dependence, not causality",
        ),
    )


def plot_19_xai_interactions_heatmap(
    xai_report: XAIExplanationReport | None,
    *,
    plots_dir: Path,
    config: VisualizationConfig,
    dataset_source: str,
) -> PlotArtifactRecord:
    """Plot 19: Symmetric 2nd-order pairwise interaction matrix + top coupling bar chart."""
    if xai_report is None or not xai_report.feature_interactions:
        return render_notice_figure(
            plots_dir,
            "19_xai_ikili_etkilesim_isi_haritasi",
            config=config,
            plot_id="19_xai_interactions",
            title=pick_text(config.lang, tr="19. XAI İkinci Derece Öznitelik Etkileşim Matrisi", en="19. XAI Second-Order Feature Interaction Matrix"),
            category=t("cat_science", config.lang),
            notice_tr=pick_text(
                config.lang,
                tr="Etkileşim analizi sonucu bulunamadı veya model tamamen doğrusal/toplamsal davrandı.",
                en="No interaction results found or the model behaved strictly linearly/additively.",
            ),
            dataset_source=dataset_source,
        )

    inter = xai_report.feature_interactions
    feat_names = sorted({p.feature_i for p in inter} | {p.feature_j for p in inter})[:12]
    idx_map = {n: i for i, n in enumerate(feat_names)}
    mat = np.zeros((len(feat_names), len(feat_names)), dtype=float)
    for p in inter:
        if p.feature_i in idx_map and p.feature_j in idx_map:
            i, j = idx_map[p.feature_i], idx_map[p.feature_j]
            mat[i, j] = max(mat[i, j], float(p.interaction_score))
            mat[j, i] = mat[i, j]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.4))
    fig.suptitle(
        pick_text(
            config.lang,
            tr="19. İkinci Derece İkili Öznitelik Etkileşimi (Karma Kısmi Türev / 2D Pertürbasyon)",
            en="19. Second-Order Pairwise Feature Interactions (Mixed Partial Derivative / 2D Perturbation)",
        ),
        fontsize=13,
        fontweight="bold",
        color=PALETTE["navy"],
    )

    im = ax1.imshow(mat, cmap="YlGnBu", aspect="auto")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04).set_label(pick_text(config.lang, tr="Etkileşim Gücü |∂²f/∂xᵢ∂xⱼ|·σᵢ·σⱼ", en="Interaction Strength |∂²f/∂xᵢ∂xⱼ|·σᵢ·σⱼ"))
    ax1.set_xticks(np.arange(len(feat_names)))
    ax1.set_xticklabels(feat_names, rotation=35, ha="right", fontsize=8.0)
    ax1.set_yticks(np.arange(len(feat_names)))
    ax1.set_yticklabels(feat_names, fontsize=8.0)
    ax1.set_title(pick_text(config.lang, tr="İkili Etkileşim Isı Haritası", en="Pairwise Interaction Heatmap"))

    top_pairs = sorted(inter, key=lambda p: p.interaction_score, reverse=True)[:10]
    labels = [f"{p.feature_i} × {p.feature_j}" for p in top_pairs][::-1]
    scores = [float(p.interaction_score) for p in top_pairs][::-1]
    ax2.barh(labels, scores, color=PALETTE["purple"], edgecolor="#FFFFFF")
    ax2.set_title(pick_text(config.lang, tr="En Güçlü 10 Öznitelik Çifti Etkileşimi", en="Top 10 Strongest Feature Pair Interactions"))
    ax2.set_xlabel(pick_text(config.lang, tr="Ölçeklenmiş Etkileşim Skoru", en="Scaled Interaction Score"))
    ax2.grid(True, axis="x", alpha=0.5)

    return save_figure(
        fig,
        plots_dir,
        "19_xai_ikili_etkilesim_isi_haritasi",
        config=config,
        plot_id="19_xai_interactions",
        title=pick_text(config.lang, tr="19. XAI İkili Öznitelik Etkileşim Haritası", en="19. XAI Pairwise Interaction Heatmap"),
        category=t("cat_science", config.lang),
        description=pick_text(
            config.lang,
            tr="Spektral bantlar ve toprak kovaryatları arasındaki doğrusal olmayan ikinci derece etkileşimler.",
            en="Nonlinear second-order interactions between spectral bands and soil covariates.",
        ),
        dataset_source=dataset_source,
    )


def plot_20_carbon_fractions_and_closure(
    reports: tuple[PredictionReport | None, ...],
    arrays: DatasetArrays,
    *,
    plots_dir: Path,
    config: VisualizationConfig,
    dataset_source: str,
) -> PlotArtifactRecord:
    """Plot 20: Carbon fraction breakdown (SOC + SIC vs TC) and mass-balance closure residuals."""
    valid_reps = [r for r in reports if r is not None]
    sids: list[str] = []
    socs: list[float] = []
    sics: list[float] = []
    tcs: list[float] = []
    residuals: list[float] = []

    for r in valid_reps:
        cm = r.carbon_metrics
        soc_v = cm.get("soc_pct")
        sic_v = cm.get("sic_pct")
        tc_v = cm.get("tc_pct")
        res_v = cm.get("closure_residual_pct")
        if soc_v is not None:
            sids.append(str(r.identity.get("sample_id", len(sids))))
            socs.append(float(soc_v))
            sics.append(float(sic_v) if sic_v is not None else 0.0)
            tcs.append(float(tc_v) if tc_v is not None else (float(soc_v) + (float(sic_v) if sic_v is not None else 0.0)))
            residuals.append(float(res_v) if res_v is not None else 0.0)

    # Supplement SIC / TC from dataset references if single-head baseline didn't predict SIC/TC
    if sics and all(v == 0.0 for v in sics) and np.any(np.isfinite(arrays.sic)):
        sid_to_idx = {sid: i for i, sid in enumerate(arrays.sample_ids)}
        for k, sid in enumerate(sids):
            idx = sid_to_idx.get(sid)
            if idx is not None and np.isfinite(arrays.sic[idx]):
                sics[k] = float(arrays.sic[idx])
                if np.isfinite(arrays.tc[idx]):
                    tcs[k] = float(arrays.tc[idx])
                    residuals[k] = tcs[k] - (socs[k] + sics[k])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.2))
    fig.suptitle(
        pick_text(
            config.lang,
            tr="20. Karbon Fraksiyonları (SOC + SIC vs TC) ve Kütle Dengesi Kapatma Teşhisi",
            en="20. Carbon Fractions (SOC + SIC vs TC) & Mass-Balance Closure Diagnostics",
        ),
        fontsize=13,
        fontweight="bold",
        color=PALETTE["navy"],
    )

    n_show = min(22, len(sids))
    x = np.arange(n_show)
    ax1.bar(x, socs[:n_show], color=PALETTE["teal"], label=pick_text(config.lang, tr="Organik Karbon (SOC %)", en="Organic Carbon (SOC %)"))
    ax1.bar(x, sics[:n_show], bottom=socs[:n_show], color=PALETTE["orange"], label=pick_text(config.lang, tr="İnorganik Karbon (SIC %)", en="Inorganic Carbon (SIC %)"))
    ax1.plot(x, tcs[:n_show], color=PALETTE["navy"], marker="D", markersize=4.5, linewidth=1.3, label=pick_text(config.lang, tr="Toplam Karbon (TC %)", en="Total Carbon (TC %)"))
    ax1.set_xticks(x)
    ax1.set_xticklabels(sids[:n_show], rotation=45, ha="right", fontsize=7.6)
    ax1.set_title(pick_text(config.lang, tr=f"Örnek Bazlı Karbon Ayrıştırması (İlk {n_show} Örnek)", en=f"Sample-Wise Carbon Speciation (First {n_show} Samples)"))
    ax1.set_ylabel(pick_text(config.lang, tr="Karbon Konsantrasyonu (% kuru madde)", en="Carbon Concentration (% dry matter)"))
    ax1.grid(True, axis="y", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8.2)

    res_arr = np.asarray(residuals, dtype=float)
    ax2.hist(res_arr, bins=12, color=PALETTE["blue"], edgecolor="#FFFFFF", alpha=0.85)
    ax2.axvline(0.0, color="#1A202C", linestyle="-", linewidth=1.2)
    ax2.axvline(0.05, color=PALETTE["vermilion"], linestyle="--", linewidth=1.4, label=pick_text(config.lang, tr="Tolerans (±0.05%)", en="Tolerance (±0.05%)"))
    ax2.axvline(-0.05, color=PALETTE["vermilion"], linestyle="--", linewidth=1.4)
    ax2.set_title(pick_text(config.lang, tr=f"Kapatma Artığı: TC − (SOC + SIC) (Ort={np.mean(res_arr):+.4f}%)", en=f"Closure Residual: TC − (SOC + SIC) (Mean={np.mean(res_arr):+.4f}%)"))
    ax2.set_xlabel(pick_text(config.lang, tr="Kapatma Artığı (%)", en="Closure Residual (%)"))
    ax2.set_ylabel(pick_text(config.lang, tr="Örnek Sayısı", en="Sample Count"))
    ax2.grid(True, alpha=0.5)
    ax2.legend(loc="upper right")

    return save_figure(
        fig,
        plots_dir,
        "20_karbon_fraksiyonlari_ve_kapatma",
        config=config,
        plot_id="20_carbon_fractions",
        title=pick_text(config.lang, tr="20. Karbon Fraksiyonları ve Kapatma Teşhisi", en="20. Carbon Fractions & Closure Diagnostics"),
        category=t("cat_science", config.lang),
        description=pick_text(
            config.lang,
            tr="SOC, SIC ve TC yığılmış fraksiyonları ile TC = SOC + SIC kütle dengesi sapması.",
            en="Stacked SOC, SIC, and TC fractions alongside TC = SOC + SIC mass-balance closure residuals.",
        ),
        dataset_source=dataset_source,
    )


def plot_21_carbon_stock_and_tornado(
    reports: tuple[PredictionReport | None, ...],
    *,
    plots_dir: Path,
    config: VisualizationConfig,
    dataset_source: str,
) -> PlotArtifactRecord:
    """Plot 21: Layer Carbon Stock (Mg C ha^-1) + First-order variance attribution tornado."""
    valid_reps = [r for r in reports if r is not None]
    stock_vals: list[float] = []
    stock_stds: list[float] = []
    sample_names: list[str] = []
    missing_bd_count = 0

    param_labels = {
        "carbon_pct": pick_text(config.lang, tr="Karbon (%)", en="Carbon (%)"),
        "bulk_density": pick_text(config.lang, tr="Hacim Ağırlığı (g/cm³)", en="Bulk Density (g/cm³)"),
        "depth_cm": pick_text(config.lang, tr="Derinlik Kalınlığı (cm)", en="Depth Thickness (cm)"),
        "coarse_fragments": pick_text(config.lang, tr="Kaba Parça Oranı", en="Coarse Fragments"),
    }
    var_sums = {k: 0.0 for k in STOCK_PARAM_NAMES}
    n_var_items = 0

    for r in valid_reps:
        cs = r.carbon_stock
        s_val = cs.get("stock_mg_ha")
        if s_val is None:
            missing_bd_count += 1
            continue
        stock_vals.append(float(s_val))
        stock_stds.append(float(cs.get("stock_std_mg_ha") or 0.0))
        sample_names.append(str(r.identity.get("sample_id", len(sample_names))))
        unc = cs.get("uncertainty") or {}
        vc = unc.get("variance_contributions") or {}
        if vc:
            n_var_items += 1
            for k in STOCK_PARAM_NAMES:
                var_sums[k] += max(0.0, float(vc.get(k, 0.0)))

    if not stock_vals:
        return render_notice_figure(
            plots_dir,
            "21_karbon_stoku_ve_varyans_tornado",
            config=config,
            plot_id="21_carbon_stock_tornado",
            title=pick_text(config.lang, tr="21. Karbon Stoku ve Varyans Ayrıştırması", en="21. Carbon Stock & Variance Decomposition"),
            category=t("cat_science", config.lang),
            notice_tr=pick_text(
                config.lang,
                tr=(
                    f"Hacim ağırlığı (bulk_density_g_cm3) {missing_bd_count} örnekte eksik olduğu için "
                    "ADR-004 fail-closed kuralı gereği stok hesabı bastırıldı (INSUFFICIENT_DATA)."
                ),
                en=(
                    f"Bulk density (bulk_density_g_cm3) was missing in {missing_bd_count} samples; "
                    "stock computation suppressed per ADR-004 fail-closed rule (INSUFFICIENT_DATA)."
                ),
            ),
            dataset_source=dataset_source,
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.2))
    fig.suptitle(
        pick_text(
            config.lang,
            tr="21. Toprak Katmanı Karbon Stoku (Mg C ha⁻¹) ve Birinci Derece Belirsizlik Payı",
            en="21. Soil Layer Carbon Stock (Mg C ha⁻¹) & First-Order Variance Attribution",
        ),
        fontsize=13,
        fontweight="bold",
        color=PALETTE["navy"],
    )

    n_show = min(22, len(stock_vals))
    x = np.arange(n_show)
    ax1.bar(x, stock_vals[:n_show], yerr=stock_stds[:n_show], capsize=3, color=PALETTE["teal"], ecolor=PALETTE["navy"], edgecolor="#FFFFFF")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sample_names[:n_show], rotation=45, ha="right", fontsize=7.6)
    ax1.set_title(
        pick_text(
            config.lang,
            tr=f"Katman Karbon Stoku ± 1σ (Ort={np.mean(stock_vals):.1f} Mg C ha⁻¹, Eksik HA={missing_bd_count})",
            en=f"Layer Carbon Stock ± 1σ (Mean={np.mean(stock_vals):.1f} Mg C ha⁻¹, Missing BD={missing_bd_count})",
        )
    )
    ax1.set_ylabel(pick_text(config.lang, tr="Karbon Stoku (Mg C ha⁻¹)", en="Carbon Stock (Mg C ha⁻¹)"))
    ax1.grid(True, axis="y", alpha=0.5)

    tot_var = max(sum(var_sums.values()), 1e-12)
    pct_shares = [(param_labels[k], (var_sums[k] / tot_var) * 100.0) for k in STOCK_PARAM_NAMES]
    pct_shares.sort(key=lambda kv: kv[1])
    labels_t = [p[0] for p in pct_shares]
    vals_t = [p[1] for p in pct_shares]
    bars = ax2.barh(labels_t, vals_t, color=[PALETTE["sky"], PALETTE["blue"], PALETTE["orange"], PALETTE["vermilion"]], edgecolor="#FFFFFF")
    for b, v in zip(bars, vals_t):
        ax2.text(v + 1.0, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", ha="left", fontsize=8.8)
    ax2.set_xlim(0, min(115.0, max(vals_t) + 18.0))
    ax2.set_title(pick_text(config.lang, tr="Taylor Yayılımı Baskın Varyans Kaynağı Tornado Grafiği", en="Taylor Propagation Dominant Variance Tornado"))
    ax2.set_xlabel(pick_text(config.lang, tr="Toplam Stok Varyansına Ortalama Katkı (%)", en="Mean Contribution to Total Stock Variance (%)"))
    ax2.grid(True, axis="x", alpha=0.5)

    return save_figure(
        fig,
        plots_dir,
        "21_karbon_stoku_ve_varyans_tornado",
        config=config,
        plot_id="21_carbon_stock_tornado",
        title=pick_text(config.lang, tr="21. Karbon Stoku ve Varyans Ayrıştırması", en="21. Carbon Stock & Variance Decomposition"),
        category=t("cat_science", config.lang),
        description=pick_text(
            config.lang,
            tr="Katman karbon stoku (Mg C ha⁻¹) dağılımı ve hacim ağırlığı/karbon belirsizlik payları.",
            en="Layer carbon stock (Mg C ha⁻¹) distribution and bulk density / carbon variance shares.",
        ),
        dataset_source=dataset_source,
    )


def plot_22_spatial_variogram(
    spatial: SpatialPlotBundle | None,
    target_name: str,
    *,
    plots_dir: Path,
    config: VisualizationConfig,
    dataset_source: str,
) -> PlotArtifactRecord:
    """Plot 22: Empirical Matheron semivariogram points + fitted parametric variogram model."""
    if spatial is None or spatial.kriging.variogram is None:
        return render_notice_figure(
            plots_dir,
            "22_mekansal_variogram_modeli",
            config=config,
            plot_id="22_spatial_variogram",
            title=pick_text(config.lang, tr="22. Mekansal Semivariogram Modeli", en="22. Spatial Semivariogram Model"),
            category=t("cat_science", config.lang),
            notice_tr=pick_text(
                config.lang,
                tr="Variogram uyumu için en az 8 farklı koordinat noktası (lon, lat) gereklidir.",
                en="At least 8 distinct coordinate points (lon, lat) are required for variogram fitting.",
            ),
            dataset_source=dataset_source,
        )

    v = spatial.kriging.variogram
    emp = v.empirical
    lags = emp.lag_centers_m
    semis = emp.semivariances
    counts = emp.pair_counts

    h_grid = np.linspace(0.0, float(np.max(lags)) * 1.08, 120)
    gamma_fit = evaluate_variogram(h_grid, v)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.2), gridspec_kw={"width_ratios": [1.8, 1.0]})
    fig.suptitle(
        pick_text(
            config.lang,
            tr=f"22. Mekansal Deneysel Semivariogram ve Uyarlanmış '{v.model_name.title()}' Modeli ({target_name})",
            en=f"22. Empirical Spatial Semivariogram & Fitted '{v.model_name.title()}' Model ({target_name})",
        ),
        fontsize=12.8,
        fontweight="bold",
        color=PALETTE["navy"],
    )

    sizes = np.clip(counts.astype(float) * 8.0, 35.0, 260.0)
    ax1.scatter(lags, semis, s=sizes, color=PALETTE["orange"], edgecolor=PALETTE["navy"], zorder=5, label=pick_text(config.lang, tr="Ampirik Matheron (Boyut ~ Çift Sayısı)", en="Empirical Matheron (Size ~ Pair Count)"))
    ax1.plot(h_grid, gamma_fit, color=PALETTE["blue"], linewidth=2.3, label=pick_text(config.lang, tr=f"Parametrik Model ({v.model_name})", en=f"Parametric Model ({v.model_name})"))
    ax1.axhline(v.nugget, color=PALETTE["slate"], linestyle=":", linewidth=1.3, label=f"Nugget (C₀) = {v.nugget:.4f}")
    ax1.axhline(v.sill, color=PALETTE["teal"], linestyle="--", linewidth=1.3, label=f"Sill (C₀+C₁) = {v.sill:.4f}")
    ax1.axvline(v.range_m, color=PALETTE["vermilion"], linestyle="--", linewidth=1.3, label=pick_text(config.lang, tr=f"Erişim (Range) = {v.range_m:.1f} m", en=f"Range = {v.range_m:.1f} m"))
    ax1.set_title(pick_text(config.lang, tr=f"Semivariogram Eğrisi (Uyum RMSE = {v.fit_rmse:.4f})", en=f"Semivariogram Curve (Fit RMSE = {v.fit_rmse:.4f})"))
    ax1.set_xlabel(pick_text(config.lang, tr="Gecikme Mesafesi h (metre)", en="Lag Distance h (meters)"))
    ax1.set_ylabel(pick_text(config.lang, tr="Semivarians γ(h)", en="Semivariance γ(h)"))
    ax1.grid(True, alpha=0.5)
    ax1.legend(loc="lower right", fontsize=8.0)

    ax2.bar(np.arange(lags.size), counts, color=PALETTE["sky"], edgecolor=PALETTE["navy"])
    ax2.set_xticks(np.arange(lags.size))
    ax2.set_xticklabels([f"{h:.0f}m" for h in lags], rotation=35, ha="right", fontsize=8.0)
    ax2.set_title(pick_text(config.lang, tr=f"Lag Kutusu Çift Sayıları (Toplam Saha Noktası={v.n_samples})", en=f"Lag Bin Pair Counts (Total Field Sites={v.n_samples})"))
    ax2.set_xlabel(pick_text(config.lang, tr="Gecikme Merkezi", en="Lag Bin Center"))
    ax2.set_ylabel(pick_text(config.lang, tr="Nokta Çifti Sayısı", en="Point Pair Count"))
    ax2.grid(True, axis="y", alpha=0.5)

    return save_figure(
        fig,
        plots_dir,
        "22_mekansal_variogram_modeli",
        config=config,
        plot_id="22_spatial_variogram",
        title=pick_text(config.lang, tr="22. Mekansal Semivariogram Modeli", en="22. Spatial Semivariogram Model"),
        category=t("cat_science", config.lang),
        description=pick_text(
            config.lang,
            tr="Matheron ampirik variogram noktaları, nugget/sill/range parametreleri ve model uyumu.",
            en="Matheron empirical variogram points, nugget/sill/range parameters, and model fit.",
        ),
        dataset_source=dataset_source,
    )


def plot_23_ordinary_kriging_2d_3d(
    spatial: SpatialPlotBundle | None,
    target_name: str,
    *,
    plots_dir: Path,
    config: VisualizationConfig,
    dataset_source: str,
) -> PlotArtifactRecord:
    """Plot 23 (3D #3): 2D Ordinary Kriging uncertainty contour + 3D spatial carbon surface."""
    if spatial is None:
        return render_notice_figure(
            plots_dir,
            "23_ordinary_kriging_2d_ve_3d",
            config=config,
            plot_id="23_ordinary_kriging_2d_3d",
            title=pick_text(config.lang, tr="23. Ordinary Kriging 2D & 3D Mekansal Yüzey (3D)", en="23. Ordinary Kriging 2D & 3D Spatial Surface (3D)"),
            category=t("cat_science", config.lang),
            notice_tr=pick_text(
                config.lang,
                tr="2D/3D Ordinary Kriging enterpolasyonu için en az 8 ayrı saha koordinatı gereklidir.",
                en="At least 8 distinct spatial coordinates are required for 2D/3D Ordinary Kriging interpolation.",
            ),
            dataset_source=dataset_source,
            is_3d=True,
        )

    kr = spatial.kriging
    GX = spatial.grid_x_m
    GY = spatial.grid_y_m
    Z_pred = kr.predicted_values.reshape(GX.shape)
    Z_std = kr.kriging_std.reshape(GX.shape)
    extrap = kr.is_extrapolated_mask.reshape(GX.shape)

    fig = plt.figure(figsize=(14.4, 6.4))
    fig.suptitle(
        pick_text(
            config.lang,
            tr=f"23. 2D & 3D Ordinary Kriging Mekansal {target_name} Yüzeyi ve Belirsizlik Haritası",
            en=f"23. 2D & 3D Ordinary Kriging Spatial {target_name} Surface & Uncertainty Map",
        ),
        fontsize=13,
        fontweight="bold",
        color=PALETTE["navy"],
    )

    ax2d = fig.add_subplot(1, 2, 1)
    cf = ax2d.contourf(GX, GY, Z_pred, levels=16, cmap="YlGn")
    fig.colorbar(cf, ax=ax2d, fraction=0.046, pad=0.04).set_label(pick_text(config.lang, tr=f"Kriging Tahmini {target_name} (%)", en=f"Kriging Estimate {target_name} (%)"))
    # Overlay uncertainty contours & extrapolation boundary
    cs_std = ax2d.contour(GX, GY, Z_std, levels=6, colors="#1A202C", linestyles="--", linewidths=0.8)
    ax2d.clabel(cs_std, inline=True, fontsize=7.2, fmt="±%.2f")
    if np.any(extrap):
        ax2d.contour(GX, GY, extrap.astype(float), levels=[0.5], colors=[PALETTE["vermilion"]], linewidths=1.8, linestyles="-.")
    ax2d.scatter(
        spatial.sample_coords_m[:, 0],
        spatial.sample_coords_m[:, 1],
        c=spatial.sample_values,
        cmap="YlGn",
        edgecolor="#1A202C",
        s=55,
        zorder=5,
        label=pick_text(config.lang, tr="Örnekleme Noktaları", en="Sampling Locations"),
    )
    ax2d.set_title(pick_text(config.lang, tr="2D Kriging Yüzeyi (Kesikli Çizgi = Kriging Std, Kırmızı = Ekstrapolasyon)", en="2D Kriging Surface (Dashed = Kriging Std, Red = Extrapolation)"))
    ax2d.set_xlabel(pick_text(config.lang, tr="Doğu-Batı Mesafe (m)", en="East-West Distance (m)"))
    ax2d.set_ylabel(pick_text(config.lang, tr="Kuzey-Güney Mesafe (m)", en="North-South Distance (m)"))
    ax2d.legend(loc="upper right", fontsize=8.0)

    ax3d = fig.add_subplot(1, 2, 2, projection="3d")
    surf = ax3d.plot_surface(GX, GY, Z_pred, cmap="viridis", edgecolor="none", alpha=0.88)
    ax3d.scatter(
        spatial.sample_coords_m[:, 0],
        spatial.sample_coords_m[:, 1],
        spatial.sample_values,
        color=PALETTE["vermilion"],
        s=34,
        depthshade=False,
        label=pick_text(config.lang, tr="Gözlem", en="Observation"),
    )
    fig.colorbar(surf, ax=ax3d, shrink=0.58, pad=0.08).set_label(f"{target_name} (%)")
    ax3d.set_title(pick_text(config.lang, tr="3D Kriging Arazi Yüzeyi", en="3D Kriging Terrain Surface"))
    ax3d.set_xlabel("X (m)", labelpad=6)
    ax3d.set_ylabel("Y (m)", labelpad=6)
    ax3d.set_zlabel(f"{target_name} (%)", labelpad=6)
    ax3d.view_init(elev=28, azim=-130)

    return save_figure(
        fig,
        plots_dir,
        "23_ordinary_kriging_2d_ve_3d",
        config=config,
        plot_id="23_ordinary_kriging_2d_3d",
        title=pick_text(config.lang, tr="23. Ordinary Kriging 2D & 3D Mekansal Yüzey (3D)", en="23. Ordinary Kriging 2D & 3D Spatial Surface (3D)"),
        category=t("cat_science", config.lang),
        is_3d=True,
        description=pick_text(
            config.lang,
            tr="Yerel metrik koordinat düzleminde 2D Kriging belirsizlik konturları ve 3D karbon dağılım yüzeyi.",
            en="2D Kriging uncertainty contours and 3D carbon surface in local metric coordinates.",
        ),
        dataset_source=dataset_source,
        subtitle=pick_text(
            config.lang,
            tr=f"Ekstrapolasyon dışbükey zarf dışı noktalar işaretlendi; maksimum ağırlık toplam hatası={kr.weights_sum_max_error:.2e}",
            en=f"Extrapolation marks points outside convex hull; max weight sum error={kr.weights_sum_max_error:.2e}",
        ),
    )


def generate_science_plots(
    *,
    xai_report: XAIExplanationReport | None,
    reports: tuple[PredictionReport | None, ...],
    arrays: DatasetArrays,
    target_name: str,
    spatial: SpatialPlotBundle | None,
    plots_dir: Path,
    config: VisualizationConfig,
    dataset_source: str,
) -> list[PlotArtifactRecord]:
    """Generate Plots 17-23 covering XAI, carbon accounting, uncertainty tornado, variogram & Kriging."""
    return [
        plot_17_xai_spectral_sensitivity_ig(xai_report, arrays, plots_dir=plots_dir, config=config, dataset_source=dataset_source),
        plot_18_xai_hsic_and_component_ranking(xai_report, plots_dir=plots_dir, config=config, dataset_source=dataset_source),
        plot_19_xai_interactions_heatmap(xai_report, plots_dir=plots_dir, config=config, dataset_source=dataset_source),
        plot_20_carbon_fractions_and_closure(reports, arrays, plots_dir=plots_dir, config=config, dataset_source=dataset_source),
        plot_21_carbon_stock_and_tornado(reports, plots_dir=plots_dir, config=config, dataset_source=dataset_source),
        plot_22_spatial_variogram(spatial, target_name, plots_dir=plots_dir, config=config, dataset_source=dataset_source),
        plot_23_ordinary_kriging_2d_3d(spatial, target_name, plots_dir=plots_dir, config=config, dataset_source=dataset_source),
    ]
