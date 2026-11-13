"""Model comparison, parity, Bland-Altman, residuals, conformal, OOD & 3D PCA plots (Plots 08-16)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from st_ds.api.analyzer import BatchPredictionResult
from st_ds.domain.schemas import PredictionReport
from st_ds.i18n import pick_text, t
from st_ds.training.metrics import compute_regression_metrics
from st_ds.training.trainer import BenchmarkReport, ModelBenchmarkEntry
from st_ds.uncertainty.conformal import compute_calibration_curve
from st_ds.uncertainty.ood import OODBatchResult
from st_ds.visualization.base import (
    PALETTE,
    DatasetArrays,
    PlotArtifactRecord,
    VisualizationConfig,
    render_notice_figure,
    sanitize_label_map,
    save_figure,
)


@dataclass(frozen=True)
class ModelPlotContext:
    """Bundled model evaluation, inference, uncertainty, and latent-space arrays for Plots 08-16."""

    target_name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    pred_std: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    benchmark: BenchmarkReport | None
    best_entry: ModelBenchmarkEntry | None
    reports: tuple[PredictionReport | None, ...]
    batch_result: BatchPredictionResult | None
    fused_features: np.ndarray | None
    ood_batch: OODBatchResult | None
    conformal_calibrator: object | None


def _select_valid_predictions(ctx: ModelPlotContext) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = np.isfinite(ctx.y_true) & np.isfinite(ctx.y_pred)
    return (
        ctx.y_true[mask],
        ctx.y_pred[mask],
        ctx.pred_std[mask],
        ctx.ci_lower[mask],
        ctx.ci_upper[mask],
    )


def plot_08_model_comparison_metrics(
    ctx: ModelPlotContext,
    *,
    plots_dir: Path,
    config: VisualizationConfig,
    dataset_source: str,
) -> PlotArtifactRecord:
    """Plot 08: Cross-validated model benchmark comparison (RMSE, MAE, R², RPIQ)."""
    if ctx.benchmark is None or not ctx.benchmark.entries:
        return render_notice_figure(
            plots_dir,
            "08_model_kiyaslama_metrikleri",
            config=config,
            plot_id="08_model_comparison",
            title=pick_text(config.lang, tr="08. Model Kıyaslama Metrikleri", en="08. Model Benchmark Metrics"),
            category=t("cat_model", config.lang),
            notice_tr=pick_text(
                config.lang,
                tr="Çoklu model kıyaslaması için BenchmarkReport bulunamadı.",
                en="BenchmarkReport unavailable for multi-model comparison.",
            ),
            dataset_source=dataset_source,
        )

    names = list(ctx.benchmark.evidence_ranking)
    entries = [ctx.benchmark.entries[n] for n in names]
    rmses = [float(e.oof_metrics.rmse) for e in entries]
    maes = [float(e.oof_metrics.mae) for e in entries]
    r2s = [float(e.oof_metrics.r2) if e.oof_metrics.r2 is not None else 0.0 for e in entries]
    rpiqs = [float(e.oof_metrics.rpiq) if e.oof_metrics.rpiq is not None else 0.0 for e in entries]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.0))
    fig.suptitle(
        pick_text(
            config.lang,
            tr=f"08. Gruplanmış Çapraz Doğrulama Model Kıyaslaması (Hedef: {ctx.target_name})",
            en=f"08. Grouped Cross-Validation Model Benchmark (Target: {ctx.target_name})",
        ),
        fontsize=13,
        fontweight="bold",
        color=PALETTE["navy"],
    )

    x = np.arange(len(names))
    w = 0.36
    b1 = axes[0].bar(x - w / 2, rmses, width=w, label="RMSE (%)", color=PALETTE["navy"])
    b2 = axes[0].bar(x + w / 2, maes, width=w, label="MAE (%)", color=PALETTE["sky"])
    for bar, v in zip(b1, rmses):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7.6)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=24, ha="right", fontsize=8.2)
    axes[0].set_title(pick_text(config.lang, tr="Hata Metrikleri (Düşük = Daha İyi)", en="Error Metrics (Lower = Better)"))
    axes[0].set_ylabel(pick_text(config.lang, tr="Hata (%)", en="Error (%)"))
    axes[0].grid(True, axis="y", alpha=0.5)
    axes[0].legend(loc="upper right")

    colors_r2 = [PALETTE["teal"] if i == 0 else PALETTE["blue"] for i in range(len(names))]
    br2 = axes[1].bar(x, r2s, color=colors_r2, edgecolor="#FFFFFF")
    for bar, v in zip(br2, r2s):
        axes[1].text(bar.get_x() + bar.get_width() / 2, max(v, 0.0) + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=7.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=24, ha="right", fontsize=8.2)
    axes[1].set_title(pick_text(config.lang, tr="Belirtme Katsayısı R² (Yüksek = Daha İyi)", en="Coefficient of Determination R² (Higher = Better)"))
    axes[1].set_ylabel("OOF R²")
    axes[1].set_ylim(min(0.0, min(r2s) - 0.1), 1.12)
    axes[1].grid(True, axis="y", alpha=0.5)

    brp = axes[2].bar(x, rpiqs, color=PALETTE["orange"], edgecolor="#FFFFFF")
    for bar, v in zip(brp, rpiqs):
        axes[2].text(bar.get_x() + bar.get_width() / 2, v + 0.04, f"{v:.2f}", ha="center", va="bottom", fontsize=7.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=24, ha="right", fontsize=8.2)
    axes[2].set_title(pick_text(config.lang, tr="RPIQ = IQR / RMSE (Yüksek = Daha İyi)", en="RPIQ = IQR / RMSE (Higher = Better)"))
    axes[2].set_ylabel("RPIQ")
    axes[2].grid(True, axis="y", alpha=0.5)

    return save_figure(
        fig,
        plots_dir,
        "08_model_kiyaslama_metrikleri",
        config=config,
        plot_id="08_model_comparison",
        title=pick_text(config.lang, tr="08. Model Kıyaslama Metrikleri", en="08. Model Benchmark Metrics"),
        category=t("cat_model", config.lang),
        description=pick_text(
            config.lang,
            tr="PLSR, Kernel Ridge, Local PLSR, Random Forest ve STSoilHybridNet OOF performans kıyaslaması.",
            en="Out-of-fold benchmark comparison across PLSR, Kernel Ridge, Local PLSR, Random Forest, and STSoilHybridNet.",
        ),
        dataset_source=dataset_source,
    )


def plot_09_parity_and_ci90(
    ctx: ModelPlotContext,
    arrays: DatasetArrays,
    *,
    plots_dir: Path,
    config: VisualizationConfig,
    dataset_source: str,
) -> PlotArtifactRecord:
    """Plot 09: Observed vs Predicted parity plot with 1:1 line, trendline, and ±90% Conformal CI."""
    mask = np.isfinite(ctx.y_true) & np.isfinite(ctx.y_pred)
    yt = ctx.y_true[mask]
    yp = ctx.y_pred[mask]
    lo = ctx.ci_lower[mask]
    hi = ctx.ci_upper[mask]

    if yt.size < 2:
        return render_notice_figure(
            plots_dir,
            "09_gercek_vs_tahmin_ve_ci90",
            config=config,
            plot_id="09_parity_ci90",
            title=pick_text(config.lang, tr="09. Gerçek vs Tahmin ve %90 Güven Aralığı", en="09. Observed vs Predicted & 90% CI"),
            category=t("cat_model", config.lang),
            notice_tr=pick_text(
                config.lang,
                tr="Parite grafiği için referans hedef değerleri eksik.",
                en="Reference target values missing for parity plot.",
            ),
            dataset_source=dataset_source,
        )

    metrics = compute_regression_metrics(yt, yp)
    scenarios = [arrays.moisture_scenarios[i] for i in np.where(mask)[0]]
    col_map = sanitize_label_map(scenarios)

    fig, ax = plt.subplots(figsize=(8.6, 6.8))
    fig.suptitle(
        pick_text(
            config.lang,
            tr=f"09. Referans vs Tahmin Parite Grafiği ve %90 Conformal Güven Aralığı ({ctx.target_name})",
            en=f"09. Observed vs Predicted Parity & 90% Conformal CI ({ctx.target_name})",
        ),
        fontsize=12.8,
        fontweight="bold",
        color=PALETTE["navy"],
    )

    lo_err = np.maximum(0.0, yp - lo)
    hi_err = np.maximum(0.0, hi - yp)

    for grp in sorted(set(scenarios)):
        gm = np.array([s == grp for s in scenarios])
        col = col_map.get(grp, PALETTE["blue"])
        ax.errorbar(
            yt[gm],
            yp[gm],
            yerr=np.vstack((lo_err[gm], hi_err[gm])),
            fmt="o",
            color=col,
            ecolor=col,
            elinewidth=1.0,
            capsize=2.5,
            alpha=0.82,
            markersize=6.2,
            label=f"{grp} (n={int(np.sum(gm))})",
        )

    vmin = float(min(np.min(yt), np.min(lo)))
    vmax = float(max(np.max(yt), np.max(hi)))
    pad = max(0.1, 0.06 * (vmax - vmin))
    grid = np.linspace(vmin - pad, vmax + pad, 50)
    ax.plot(grid, grid, linestyle="--", color="#1A202C", linewidth=1.6, label=pick_text(config.lang, tr="1:1 İdeal Doğru", en="1:1 Ideal Line"))

    slope, intercept = np.polyfit(yt, yp, 1)
    ax.plot(grid, slope * grid + intercept, linestyle="-", color=PALETTE["vermilion"], linewidth=1.5, label=pick_text(config.lang, tr=f"Regresyon: y={slope:.2f}x+{intercept:.2f}", en=f"Regression: y={slope:.2f}x+{intercept:.2f}"))

    coverage = float(np.mean((yt >= lo) & (yt <= hi)))
    stats_box = (
        f"n = {metrics.n_samples}\n"
        f"R² = {metrics.r2:.3f}\n" if metrics.r2 is not None else f"n = {metrics.n_samples}\nR² = N/A\n"
    ) + (
        f"RMSE = {metrics.rmse:.3f}%\n"
        f"MAE = {metrics.mae:.3f}%\n"
        f"Bias = {metrics.bias:+.3f}%\n"
        f"RPIQ = {metrics.rpiq:.2f}\n" if metrics.rpiq is not None else f"RMSE = {metrics.rmse:.3f}%\nBias = {metrics.bias:+.3f}%\n"
    ) + pick_text(config.lang, tr=f"%90 CI Kapsamı = %{coverage * 100:.1f}", en=f"90% CI Coverage = {coverage * 100:.1f}%")

    ax.text(
        0.04,
        0.96,
        stats_box,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.6,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#FFFFFF", "edgecolor": PALETTE["border"]},
    )
    ax.set_xlim(vmin - pad, vmax + pad)
    ax.set_ylim(vmin - pad, vmax + pad)
    ax.set_xlabel(pick_text(config.lang, tr=f"Laboratuvar / Referans {ctx.target_name} (%)", en=f"Laboratory / Reference {ctx.target_name} (%)"))
    ax.set_ylabel(pick_text(config.lang, tr=f"Model Tahmini {ctx.target_name} (%) ± %90 Conformal CI", en=f"Predicted {ctx.target_name} (%) ± 90% Conformal CI"))
    ax.grid(True, alpha=0.5)
    ax.legend(loc="lower right", fontsize=8.2)

    return save_figure(
        fig,
        plots_dir,
        "09_gercek_vs_tahmin_ve_ci90",
        config=config,
        plot_id="09_parity_ci90",
        title=pick_text(config.lang, tr="09. Gerçek vs Tahmin ve %90 Conformal Güven Aralığı", en="09. Observed vs Predicted & 90% Conformal CI"),
        category=t("cat_model", config.lang),
        description=pick_text(
            config.lang,
            tr="1:1 parite çizgisi, doğrusal uyum eğrisi ve kalibre edilmiş %90 tahmin aralıkları.",
            en="1:1 parity line, linear trend fit, and calibrated 90% prediction intervals.",
        ),
        dataset_source=dataset_source,
    )


def plot_10_bland_altman(
    ctx: ModelPlotContext,
    *,
    plots_dir: Path,
    config: VisualizationConfig,
    dataset_source: str,
) -> PlotArtifactRecord:
    """Plot 10: Bland-Altman agreement diagram showing Bias and ±1.96 SD Limits of Agreement."""
    yt, yp, _, _, _ = _select_valid_predictions(ctx)
    if yt.size < 3:
        return render_notice_figure(
            plots_dir,
            "10_bland_altman_uyum",
            config=config,
            plot_id="10_bland_altman",
            title=pick_text(config.lang, tr="10. Bland-Altman Uyum Analizi", en="10. Bland-Altman Agreement Analysis"),
            category=t("cat_model", config.lang),
            notice_tr=pick_text(
                config.lang,
                tr="Bland-Altman analizi için en az 3 eşleşmiş örnek gereklidir.",
                en="At least 3 paired samples are required for Bland-Altman analysis.",
            ),
            dataset_source=dataset_source,
        )

    metrics = compute_regression_metrics(yt, yp)
    means = 0.5 * (yt + yp)
    diffs = yp - yt

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    fig.suptitle(
        pick_text(
            config.lang,
            tr="10. Bland-Altman Uyum Diyagramı (Bias ve ±1.96 SD Uyum Sınırları)",
            en="10. Bland-Altman Agreement Plot (Bias & ±1.96 SD Limits of Agreement)",
        ),
        fontsize=12.8,
        fontweight="bold",
        color=PALETTE["navy"],
    )

    ax.scatter(means, diffs, color=PALETTE["blue"], edgecolor="#FFFFFF", s=54, alpha=0.85, label=pick_text(config.lang, tr="Örnek Farkı (Tahmin − Referans)", en="Sample Diff (Pred − Ref)"))
    ax.axhline(metrics.bland_altman_mean_diff, color=PALETTE["navy"], linewidth=2.0, label=pick_text(config.lang, tr=f"Ortalama Bias = {metrics.bland_altman_mean_diff:+.3f}%", en=f"Mean Bias = {metrics.bland_altman_mean_diff:+.3f}%"))
    ax.axhline(metrics.bland_altman_loa_upper, color=PALETTE["vermilion"], linestyle="--", linewidth=1.6, label=pick_text(config.lang, tr=f"Üst LoA (+1.96 SD) = {metrics.bland_altman_loa_upper:+.3f}%", en=f"Upper LoA (+1.96 SD) = {metrics.bland_altman_loa_upper:+.3f}%"))
    ax.axhline(metrics.bland_altman_loa_lower, color=PALETTE["vermilion"], linestyle="--", linewidth=1.6, label=pick_text(config.lang, tr=f"Alt LoA (−1.96 SD) = {metrics.bland_altman_loa_lower:+.3f}%", en=f"Lower LoA (−1.96 SD) = {metrics.bland_altman_loa_lower:+.3f}%"))
    ax.axhspan(metrics.bland_altman_loa_lower, metrics.bland_altman_loa_upper, color=PALETTE["sky"], alpha=0.10)
    ax.axhline(0.0, color="#718096", linestyle=":", linewidth=1.0)

    ax.set_xlabel(pick_text(config.lang, tr=f"Ortalama ((Referans + Tahmin) / 2) [{ctx.target_name} %]", en=f"Mean ((Ref + Pred) / 2) [{ctx.target_name} %]"))
    ax.set_ylabel(pick_text(config.lang, tr="Fark (Tahmin − Referans) [%]", en="Difference (Pred − Ref) [%]"))
    ax.set_title(pick_text(config.lang, tr="Sistematik Kayma ve Orantısal Hata Kontrolü", en="Systematic Bias & Proportional Error Check"))
    ax.grid(True, alpha=0.5)
    ax.legend(loc="best", fontsize=8.4)

    return save_figure(
        fig,
        plots_dir,
        "10_bland_altman_uyum",
        config=config,
        plot_id="10_bland_altman",
        title=pick_text(config.lang, tr="10. Bland-Altman Uyum Analizi", en="10. Bland-Altman Agreement Analysis"),
        category=t("cat_model", config.lang),
        description=pick_text(
            config.lang,
            tr="Tahmin ile referans laboratuvar yöntemi arasındaki ortalama sapma ve %95 uyum sınırları.",
            en="Mean bias and 95% limits of agreement between predicted and reference lab values.",
        ),
        dataset_source=dataset_source,
    )


def plot_11_residual_diagnostics_4panel(
    ctx: ModelPlotContext,
    arrays: DatasetArrays,
    *,
    plots_dir: Path,
    config: VisualizationConfig,
    dataset_source: str,
) -> PlotArtifactRecord:
    """Plot 11: 4-panel residual diagnostic suite (Res vs Pred, Histogram, Q-Q, By Scenario)."""
    mask = np.isfinite(ctx.y_true) & np.isfinite(ctx.y_pred)
    yt = ctx.y_true[mask]
    yp = ctx.y_pred[mask]
    if yt.size < 4:
        return render_notice_figure(
            plots_dir,
            "11_artik_teshis_4panel",
            config=config,
            plot_id="11_residual_diagnostics",
            title=pick_text(config.lang, tr="11. 4 Panelli Artık Hata Teşhisi", en="11. 4-Panel Residual Diagnostics"),
            category=t("cat_model", config.lang),
            notice_tr=pick_text(
                config.lang,
                tr="Artık hata teşhisi için en az 4 referans ölçüm gereklidir.",
                en="At least 4 reference measurements are required for residual diagnostics.",
            ),
            dataset_source=dataset_source,
        )

    res = yp - yt
    scenarios = [arrays.moisture_scenarios[i] for i in np.where(mask)[0]]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.2))
    fig.suptitle(
        pick_text(
            config.lang,
            tr="11. 4 Panelli Artık Hata Teşhis Paneli (Eşvaryanslılık ve Normallik Kontrolü)",
            en="11. 4-Panel Residual Diagnostics (Homoscedasticity & Normality Check)",
        ),
        fontsize=13,
        fontweight="bold",
        color=PALETTE["navy"],
    )

    # 1. Residuals vs Predicted
    axes[0, 0].scatter(yp, res, color=PALETTE["blue"], edgecolor="#FFFFFF", s=45, alpha=0.85)
    axes[0, 0].axhline(0.0, color=PALETTE["vermilion"], linestyle="--", linewidth=1.4)
    axes[0, 0].set_title(pick_text(config.lang, tr="Artık vs Tahmin (Eşvaryanslılık)", en="Residual vs Predicted (Homoscedasticity)"))
    axes[0, 0].set_xlabel(pick_text(config.lang, tr="Tahmini Değer (%)", en="Predicted Value (%)"))
    axes[0, 0].set_ylabel(pick_text(config.lang, tr="Artık (Tahmin − Gerçek)", en="Residual (Pred − Obs)"))
    axes[0, 0].grid(True, alpha=0.5)

    # 2. Residual Histogram + Gaussian fit
    axes[0, 1].hist(res, bins=min(14, max(5, res.size // 3)), density=True, color=PALETTE["teal"], edgecolor="#FFFFFF", alpha=0.8)
    mu, sd = float(np.mean(res)), max(float(np.std(res)), 1e-6)
    gx = np.linspace(mu - 3.2 * sd, mu + 3.2 * sd, 100)
    axes[0, 1].plot(gx, stats.norm.pdf(gx, mu, sd), color=PALETTE["navy"], linewidth=2.0, label=f"N({mu:.3f}, {sd:.3f}²)")
    axes[0, 1].set_title(pick_text(config.lang, tr="Artık Dağılım Histogramı", en="Residual Distribution Histogram"))
    axes[0, 1].set_xlabel(pick_text(config.lang, tr="Artık (%)", en="Residual (%)"))
    axes[0, 1].set_ylabel(pick_text(config.lang, tr="Yoğunluk", en="Density"))
    axes[0, 1].legend(loc="best")
    axes[0, 1].grid(True, alpha=0.5)

    # 3. Normal Q-Q Plot
    osm, osr = stats.probplot(res, dist="norm", fit=False)
    slope_q, int_q = np.polyfit(osm, osr, 1)
    axes[1, 0].scatter(osm, osr, color=PALETTE["purple"], edgecolor="#FFFFFF", s=42)
    axes[1, 0].plot(osm, slope_q * osm + int_q, color=PALETTE["navy"], linestyle="--", linewidth=1.5)
    axes[1, 0].set_title(pick_text(config.lang, tr="Normal Q-Q Olasılık Grafiği", en="Normal Q-Q Probability Plot"))
    axes[1, 0].set_xlabel(pick_text(config.lang, tr="Teorik Standart Normal Kantiller", en="Theoretical Standard Normal Quantiles"))
    axes[1, 0].set_ylabel(pick_text(config.lang, tr="Sıralı Artıklar (%)", en="Ordered Residuals (%)"))
    axes[1, 0].grid(True, alpha=0.5)

    # 4. Boxplot by Moisture Scenario
    uniq_sc = sorted(set(scenarios))
    grouped_res = [res[np.array([s == k for s in scenarios])] for k in uniq_sc]
    axes[1, 1].boxplot(grouped_res, tick_labels=uniq_sc, patch_artist=True, boxprops={"facecolor": PALETTE["sky"], "alpha": 0.6})
    axes[1, 1].axhline(0.0, color=PALETTE["vermilion"], linestyle=":", linewidth=1.3)
    axes[1, 1].set_title(pick_text(config.lang, tr="Nem Senaryosuna Göre Artık Dağılımı", en="Residuals by Moisture Scenario"))
    axes[1, 1].set_ylabel(pick_text(config.lang, tr="Artık (%)", en="Residual (%)"))
    axes[1, 1].grid(True, axis="y", alpha=0.5)

    return save_figure(
        fig,
        plots_dir,
        "11_artik_teshis_4panel",
        config=config,
        plot_id="11_residual_diagnostics",
        title=pick_text(config.lang, tr="11. 4 Panelli Artık Hata Teşhisi", en="11. 4-Panel Residual Diagnostics"),
        category=t("cat_model", config.lang),
        description=pick_text(
            config.lang,
            tr="Tahmine göre artık saçılımı, histogram, Q-Q normallik testi ve nem senaryosu kutu grafiği.",
            en="Residual vs predicted scatter, histogram, normal Q-Q test, and moisture scenario boxplot.",
        ),
        dataset_source=dataset_source,
    )


def plot_12_subgroup_error_heatmap(
    ctx: ModelPlotContext,
    arrays: DatasetArrays,
    *,
    plots_dir: Path,
    config: VisualizationConfig,
    dataset_source: str,
) -> PlotArtifactRecord:
    """Plot 12: Subgroup RMSE & Bias heatmap stratified by site_id × moisture_scenario."""
    mask = np.isfinite(ctx.y_true) & np.isfinite(ctx.y_pred)
    if np.sum(mask) < 3:
        return render_notice_figure(
            plots_dir,
            "12_alt_grup_hata_isi_haritasi",
            config=config,
            plot_id="12_subgroup_error_heatmap",
            title=pick_text(config.lang, tr="12. Alt Grup Hata Isı Haritası", en="12. Subgroup Error Heatmap"),
            category=t("cat_model", config.lang),
            notice_tr=pick_text(
                config.lang,
                tr="Alt grup hata matrisi için referans veri eksik.",
                en="Reference data missing for subgroup error matrix.",
            ),
            dataset_source=dataset_source,
        )

    sites = sorted(set(arrays.site_ids))
    scenarios = sorted(set(arrays.moisture_scenarios))
    if len(scenarios) <= 1 and len(set(arrays.soil_classes)) > 1:
        col_labels = sorted(set(arrays.soil_classes))
        col_vals = arrays.soil_classes
        col_title = pick_text(config.lang, tr="Toprak Sınıfı", en="Soil Class")
    else:
        col_labels = scenarios
        col_vals = arrays.moisture_scenarios
        col_title = pick_text(config.lang, tr="Nem Senaryosu", en="Moisture Scenario")

    rmse_mat = np.full((len(sites), len(col_labels)), np.nan, dtype=float)
    bias_mat = np.full((len(sites), len(col_labels)), np.nan, dtype=float)
    cnt_mat = np.zeros((len(sites), len(col_labels)), dtype=int)

    for r_i, s_id in enumerate(sites):
        for c_i, c_id in enumerate(col_labels):
            cell_m = mask & np.array([arrays.site_ids[k] == s_id and col_vals[k] == c_id for k in range(len(arrays.records))])
            if np.any(cell_m):
                diff = ctx.y_pred[cell_m] - ctx.y_true[cell_m]
                rmse_mat[r_i, c_i] = float(np.sqrt(np.mean(np.square(diff))))
                bias_mat[r_i, c_i] = float(np.mean(diff))
                cnt_mat[r_i, c_i] = int(np.sum(cell_m))

    fig, (ax_r, ax_b) = plt.subplots(1, 2, figsize=(13.5, max(4.8, 0.7 * len(sites) + 2.2)))
    fig.suptitle(
        pick_text(
            config.lang,
            tr=f"12. Saha × {col_title} Alt Grup Hata Dağılım Isı Haritası",
            en=f"12. Site × {col_title} Subgroup Error Distribution Heatmap",
        ),
        fontsize=13,
        fontweight="bold",
        color=PALETTE["navy"],
    )

    for ax, mat, cmap, title, center_zero in [
        (ax_r, rmse_mat, "YlOrRd", pick_text(config.lang, tr="Alt Grup RMSE (%)", en="Subgroup RMSE (%)"), False),
        (ax_b, bias_mat, "RdBu_r", pick_text(config.lang, tr="Alt Grup Ortalama Bias (%)", en="Subgroup Mean Bias (%)"), True),
    ]:
        finite = mat[np.isfinite(mat)]
        if center_zero and finite.size > 0:
            vmax = max(0.05, float(np.max(np.abs(finite))))
            im = ax.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        else:
            im = ax.imshow(mat, cmap=cmap, aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=25, ha="right")
        ax.set_yticks(np.arange(len(sites)))
        ax.set_yticklabels(sites)
        ax.set_title(title)
        for ri in range(len(sites)):
            for ci in range(len(col_labels)):
                if np.isfinite(mat[ri, ci]):
                    ax.text(
                        ci,
                        ri,
                        f"{mat[ri, ci]:.3f}\n(n={cnt_mat[ri, ci]})",
                        ha="center",
                        va="center",
                        fontsize=8.0,
                        color="#1A202C",
                    )
                else:
                    ax.text(ci, ri, "-", ha="center", va="center", color="#A0AEC0")

    return save_figure(
        fig,
        plots_dir,
        "12_alt_grup_hata_isi_haritasi",
        config=config,
        plot_id="12_subgroup_error_heatmap",
        title=pick_text(config.lang, tr="12. Alt Grup Hata Isı Haritası", en="12. Subgroup Error Heatmap"),
        category=t("cat_model", config.lang),
        description=pick_text(
            config.lang,
            tr="Saha ve nem/toprak alt gruplarında modelin RMSE ve sistematik sapma dağılımı.",
            en="Model RMSE and systematic bias distribution across site and moisture/soil subgroups.",
        ),
        dataset_source=dataset_source,
    )


def plot_13_conformal_calibration_coverage(
    ctx: ModelPlotContext,
    *,
    plots_dir: Path,
    config: VisualizationConfig,
    dataset_source: str,
) -> PlotArtifactRecord:
    """Plot 13: Conformal prediction nominal vs empirical coverage reliability diagram."""
    yt, yp, sd, _, _ = _select_valid_predictions(ctx)
    if ctx.conformal_calibrator is None or yt.size < 5:
        return render_notice_figure(
            plots_dir,
            "13_conformal_kalibrasyon_kapsami",
            config=config,
            plot_id="13_conformal_calibration",
            title=pick_text(config.lang, tr="13. Conformal Güvenilirlik Kalibrasyon Eğrisi", en="13. Conformal Reliability Calibration Curve"),
            category=t("cat_model", config.lang),
            notice_tr=pick_text(
                config.lang,
                tr="Conformal kalibratör veya yeterli kalibrasyon örneği bulunamadı.",
                en="Conformal calibrator or sufficient calibration samples unavailable.",
            ),
            dataset_source=dataset_source,
        )

    levels = (0.50, 0.70, 0.80, 0.90, 0.95)
    curve = compute_calibration_curve(ctx.conformal_calibrator, yt, yp, pred_std=sd, coverage_levels=levels)
    nom = [p.nominal_coverage * 100.0 for p in curve]
    emp = [p.empirical_coverage * 100.0 for p in curve]
    widths = [p.mean_interval_width for p in curve]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.0))
    fig.suptitle(
        pick_text(
            config.lang,
            tr="13. Conformal Belirsizlik Kalibrasyon Güvenilirliği ve Aralık Genişliği",
            en="13. Conformal Uncertainty Calibration Reliability & Interval Width",
        ),
        fontsize=13,
        fontweight="bold",
        color=PALETTE["navy"],
    )

    ax1.plot([45, 100], [45, 100], linestyle="--", color="#1A202C", label=pick_text(config.lang, tr="İdeal Kalibrasyon (1:1)", en="Ideal Calibration (1:1)"))
    ax1.plot(nom, emp, marker="o", linewidth=2.2, markersize=7, color=PALETTE["teal"], label=pick_text(config.lang, tr="Gözlenen Conformal Kapsam", en="Observed Conformal Coverage"))
    ax1.fill_between(nom, nom, emp, color=PALETTE["sky"], alpha=0.20)
    for nx, ex in zip(nom, emp):
        ax1.annotate(f"{ex:.1f}%", (nx, ex), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.2)
    ax1.set_xlim(45, 100)
    ax1.set_ylim(35, 103)
    ax1.set_xlabel(pick_text(config.lang, tr="Nominal Hedef Kapsam (%)", en="Nominal Target Coverage (%)"))
    ax1.set_ylabel(pick_text(config.lang, tr="Gerçekleşen Ampirik Kapsam (%)", en="Empirical Achieved Coverage (%)"))
    ax1.set_title(pick_text(config.lang, tr="Nominal vs Ampirik Kapsam Eğrisi", en="Nominal vs Empirical Coverage Curve"))
    ax1.grid(True, alpha=0.5)
    ax1.legend(loc="lower right")

    bars = ax2.bar([f"{int(n)}%" for n in nom], widths, color=PALETTE["blue"], edgecolor="#FFFFFF")
    for b, w in zip(bars, widths):
        ax2.text(b.get_x() + b.get_width() / 2, w, f"±{w / 2:.3f}%", ha="center", va="bottom", fontsize=8.2)
    ax2.set_title(pick_text(config.lang, tr="Güven Düzeyine Göre Ortalama Tahmin Aralığı Genişliği", en="Mean Prediction Interval Width by Confidence Level"))
    ax2.set_xlabel(pick_text(config.lang, tr="Güven Düzeyi (1 − α)", en="Confidence Level (1 − α)"))
    ax2.set_ylabel(pick_text(config.lang, tr="Ortalama Tam Aralık Genişliği (%)", en="Mean Full Interval Width (%)"))
    ax2.grid(True, axis="y", alpha=0.5)

    return save_figure(
        fig,
        plots_dir,
        "13_conformal_kalibrasyon_kapsami",
        config=config,
        plot_id="13_conformal_calibration",
        title=pick_text(config.lang, tr="13. Conformal Kalibrasyon Kapsamı", en="13. Conformal Calibration Coverage"),
        category=t("cat_model", config.lang),
        description=pick_text(
            config.lang,
            tr="Farklı güven düzeylerinde nominal vs ampirik kapsam oranları ve aralık genişlikleri.",
            en="Nominal vs empirical coverage rates and interval widths across confidence levels.",
        ),
        dataset_source=dataset_source,
    )


def plot_14_uncertainty_decomposition(
    ctx: ModelPlotContext,
    arrays: DatasetArrays,
    *,
    plots_dir: Path,
    config: VisualizationConfig,
    dataset_source: str,
) -> PlotArtifactRecord:
    """Plot 14: Aleatoric vs Epistemic / Model Std decomposition & uncertainty vs error check."""
    idx = np.arange(len(arrays.records))
    half_widths = 0.5 * (ctx.ci_upper - ctx.ci_lower)
    pred_sd = np.where(np.isfinite(ctx.pred_std), ctx.pred_std, half_widths / 1.645)

    # Separate heteroscedastic core std vs conformal inflation overhead
    conformal_margin = np.maximum(0.0, half_widths - pred_sd)
    order = np.argsort(half_widths)[::-1]
    top_n = min(30, order.size)
    sel = order[:top_n]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.2))
    fig.suptitle(
        pick_text(
            config.lang,
            tr="14. Belirsizlik Ayrıştırması (Model İçsel Std vs Conformal Genişleme) ve Hata İlişkisi",
            en="14. Uncertainty Decomposition (Intrinsic Model Std vs Conformal Margin) & Error Check",
        ),
        fontsize=12.8,
        fontweight="bold",
        color=PALETTE["navy"],
    )

    x_pos = np.arange(top_n)
    ax1.bar(x_pos, pred_sd[sel], color=PALETTE["blue"], label=pick_text(config.lang, tr="Model Tahmin Std (Aleatorik/Epistemik)", en="Model Pred Std (Aleatoric/Epistemic)"))
    ax1.bar(x_pos, conformal_margin[sel], bottom=pred_sd[sel], color=PALETTE["orange"], label=pick_text(config.lang, tr="Conformal %90 Kalibrasyon Payı", en="Conformal 90% Calibration Margin"))
    ax1.set_title(pick_text(config.lang, tr=f"En Yüksek Belirsizliğe Sahip İlk {top_n} Örnek", en=f"Top {top_n} Samples with Highest Uncertainty"))
    ax1.set_xlabel(pick_text(config.lang, tr="Örnek Sırası (Yüksekten Düşüğe)", en="Sample Rank (High to Low)"))
    ax1.set_ylabel(pick_text(config.lang, tr=f"Yarı Genişlik / Std ({ctx.target_name} %)", en=f"Half-Width / Std ({ctx.target_name} %)"))
    ax1.grid(True, axis="y", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8.2)

    valid = np.isfinite(ctx.y_true) & np.isfinite(ctx.y_pred)
    if np.sum(valid) >= 3:
        abs_err = np.abs(ctx.y_pred[valid] - ctx.y_true[valid])
        hw_v = half_widths[valid]
        ax2.scatter(hw_v, abs_err, color=PALETTE["teal"], edgecolor="#FFFFFF", s=54, alpha=0.85, label=pick_text(config.lang, tr="Örnekler", en="Samples"))
        max_lim = max(float(np.max(hw_v)), float(np.max(abs_err))) * 1.08
        ax2.plot([0, max_lim], [0, max_lim], linestyle="--", color=PALETTE["vermilion"], label=pick_text(config.lang, tr="|Hata| = %90 Yarı Genişlik Sınırı", en="|Error| = 90% Half-Width Boundary"))
        inside = int(np.sum(abs_err <= hw_v))
        ax2.set_title(pick_text(config.lang, tr=f"Tahmin Belirsizliği vs Gerçekleşen |Hata| ({inside}/{int(np.sum(valid))} sınır içinde)", en=f"Predicted Uncertainty vs Observed |Error| ({inside}/{int(np.sum(valid))} inside)"))
        ax2.set_xlabel(pick_text(config.lang, tr="%90 Conformal Yarı Genişlik (%)", en="90% Conformal Half-Width (%)"))
        ax2.set_ylabel(pick_text(config.lang, tr="|Tahmin − Gerçek| (%)", en="|Pred − Obs| (%)"))
        ax2.legend(loc="upper left")
    else:
        ax2.hist(half_widths, bins=12, color=PALETTE["teal"], edgecolor="#FFFFFF")
        ax2.set_title(pick_text(config.lang, tr="%90 Conformal Yarı Genişlik Dağılımı", en="90% Conformal Half-Width Distribution"))
        ax2.set_xlabel(pick_text(config.lang, tr="Yarı Genişlik (%)", en="Half-Width (%)"))
        ax2.set_ylabel(pick_text(config.lang, tr="Örnek Sayısı", en="Sample Count"))
    ax2.grid(True, alpha=0.5)

    return save_figure(
        fig,
        plots_dir,
        "14_belirsizlik_aleatoric_epistemic",
        config=config,
        plot_id="14_uncertainty_decomposition",
        title=pick_text(config.lang, tr="14. Belirsizlik Ayrıştırması", en="14. Uncertainty Decomposition"),
        category=t("cat_model", config.lang),
        description=pick_text(
            config.lang,
            tr="Model standart sapması, Conformal kapsam payı ve belirsizlik-hata uyumluluğu.",
            en="Intrinsic model standard deviation, Conformal margin, and uncertainty-error calibration.",
        ),
        dataset_source=dataset_source,
    )


def plot_15_ood_applicability_domain(
    ctx: ModelPlotContext,
    arrays: DatasetArrays,
    *,
    plots_dir: Path,
    config: VisualizationConfig,
    dataset_source: str,
) -> PlotArtifactRecord:
    """Plot 15: Applicability domain scatter (Mahalanobis ratio vs kNN ratio) with OOD thresholds."""
    if ctx.ood_batch is None or not ctx.ood_batch.assessments:
        return render_notice_figure(
            plots_dir,
            "15_ood_etki_alani_haritasi",
            config=config,
            plot_id="15_ood_applicability_domain",
            title=pick_text(config.lang, tr="15. OOD Uygulanabilirlik Alanı Haritası", en="15. OOD Applicability Domain Map"),
            category=t("cat_model", config.lang),
            notice_tr=pick_text(
                config.lang,
                tr="Yüklenen artefakta OOD dedektörü bulunamadı.",
                en="OOD detector unavailable in loaded artifact.",
            ),
            dataset_source=dataset_source,
        )

    m_rat = np.array([a.mahalanobis_ratio for a in ctx.ood_batch.assessments], dtype=float)
    k_rat = np.array([a.knn_ratio for a in ctx.ood_batch.assessments], dtype=float)
    is_ood = np.array([a.is_ood for a in ctx.ood_batch.assessments], dtype=bool)
    scores = np.array([a.ood_score for a in ctx.ood_batch.assessments], dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.2))
    fig.suptitle(
        pick_text(
            config.lang,
            tr="15. Alan Dışı (OOD) Dedektörü ve Model Uygulanabilirlik Alanı Haritası",
            en="15. Out-of-Domain (OOD) Detector & Applicability Domain Map",
        ),
        fontsize=13,
        fontweight="bold",
        color=PALETTE["navy"],
    )

    in_m = ~is_ood
    if np.any(in_m):
        ax1.scatter(m_rat[in_m], k_rat[in_m], color=PALETTE["teal"], s=54, edgecolor="#FFFFFF", label=pick_text(config.lang, tr=f"Alan İçi (n={int(np.sum(in_m))})", en=f"In-Domain (n={int(np.sum(in_m))})"))
    if np.any(is_ood):
        ax1.scatter(m_rat[is_ood], k_rat[is_ood], color=PALETTE["vermilion"], marker="X", s=78, edgecolor="#FFFFFF", label=pick_text(config.lang, tr=f"OOD Uyarı (n={int(np.sum(is_ood))})", en=f"OOD Flag (n={int(np.sum(is_ood))})"))
    ax1.axvline(1.0, color=PALETTE["navy"], linestyle="--", linewidth=1.3, label=pick_text(config.lang, tr="Eşik Oranı = 1.0", en="Threshold Ratio = 1.0"))
    ax1.axhline(1.0, color=PALETTE["navy"], linestyle="--", linewidth=1.3)
    ax1.axvspan(0.0, 1.0, ymin=0.0, ymax=min(1.0, 1.0 / max(1.2, float(np.max(k_rat)) * 1.1)), color=PALETTE["teal"], alpha=0.05)
    ax1.set_title(pick_text(config.lang, tr="Mahalanobis Oranı vs kNN Mesafe Oranı", en="Mahalanobis Ratio vs kNN Distance Ratio"))
    ax1.set_xlabel(pick_text(config.lang, tr="Mahalanobis Mesafe Oranı (d / d_eşik)", en="Mahalanobis Distance Ratio (d / d_thresh)"))
    ax1.set_ylabel(pick_text(config.lang, tr="kNN Mesafe Oranı (d_kNN / d_eşik)", en="kNN Distance Ratio (d_kNN / d_thresh)"))
    ax1.grid(True, alpha=0.5)
    ax1.legend(loc="upper left")

    ax2.hist(scores, bins=12, color=PALETTE["blue"], edgecolor="#FFFFFF", alpha=0.86)
    ax2.axvline(1.0, color=PALETTE["vermilion"], linestyle="--", linewidth=1.7, label=pick_text(config.lang, tr=f"OOD Sınırı (OOD oranı=%{ctx.ood_batch.ood_rate * 100:.1f})", en=f"OOD Boundary (OOD rate={ctx.ood_batch.ood_rate * 100:.1f}%)"))
    ax2.set_title(pick_text(config.lang, tr="Bileşik OOD Skoru Dağılımı", en="Composite OOD Score Distribution"))
    ax2.set_xlabel(pick_text(config.lang, tr="OOD Skoru (> 1.0 Alan Dışı Şüphesi)", en="OOD Score (> 1.0 Signals Extrapolation)"))
    ax2.set_ylabel(pick_text(config.lang, tr="Örnek Sayısı", en="Sample Count"))
    ax2.grid(True, alpha=0.5)
    ax2.legend(loc="upper right")

    return save_figure(
        fig,
        plots_dir,
        "15_ood_etki_alani_haritasi",
        config=config,
        plot_id="15_ood_applicability_domain",
        title=pick_text(config.lang, tr="15. OOD Uygulanabilirlik Alanı Haritası", en="15. OOD Applicability Domain Map"),
        category=t("cat_model", config.lang),
        description=pick_text(
            config.lang,
            tr="Mahalanobis ve kNN oranlarına göre eğitim uzayı dışındaki ölçümlerin tespiti.",
            en="Detection of out-of-distribution field measurements via Mahalanobis and kNN ratios.",
        ),
        dataset_source=dataset_source,
    )


def plot_16_pca_3d_latent_space(
    ctx: ModelPlotContext,
    arrays: DatasetArrays,
    *,
    plots_dir: Path,
    config: VisualizationConfig,
    dataset_source: str,
) -> PlotArtifactRecord:
    """Plot 16 (3D #2): 3D PCA of fused spectral+covariate space colored by SOC (%)."""
    X_feat = ctx.fused_features if ctx.fused_features is not None else arrays.spectra
    if X_feat.shape[0] < 4 or X_feat.shape[1] < 3:
        return render_notice_figure(
            plots_dir,
            "16_pca_3d_gizli_uzay",
            config=config,
            plot_id="16_pca_3d_latent_space",
            title=pick_text(config.lang, tr="16. 3D PCA Gizli Öznitelik Uzayı (3D)", en="16. 3D PCA Latent Feature Space (3D)"),
            category=t("cat_model", config.lang),
            notice_tr=pick_text(
                config.lang,
                tr="3D PCA izdüşümü için en az 4 örnek ve 3 öznitelik gereklidir.",
                en="At least 4 samples and 3 features are required for 3D PCA projection.",
            ),
            dataset_source=dataset_source,
            is_3d=True,
        )

    X_centered = X_feat - np.mean(X_feat, axis=0, keepdims=True)
    col_sd = np.std(X_centered, axis=0, keepdims=True)
    X_std = np.where(col_sd > 1e-12, X_centered / np.maximum(col_sd, 1e-12), 0.0)
    _, s_vals, vt = np.linalg.svd(X_std, full_matrices=False)
    coords_3d = np.dot(X_std, vt[:3].T)
    tot_var = max(float(np.sum(np.square(s_vals))), 1e-12)
    exp_var = (np.square(s_vals[:3]) / tot_var) * 100.0

    c_vals = np.where(np.isfinite(ctx.y_true), ctx.y_true, ctx.y_pred)
    fig = plt.figure(figsize=(11.0, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        coords_3d[:, 0],
        coords_3d[:, 1],
        coords_3d[:, 2],
        c=c_vals,
        cmap="viridis",
        s=58,
        alpha=0.90,
        edgecolor="#FFFFFF",
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.62, pad=0.08)
    cbar.set_label(f"{ctx.target_name} (%)", fontsize=9.5)

    ax.set_title(
        pick_text(
            config.lang,
            tr="16. 3D Birleşik Öznitelik Uzayı PCA Kümelemesi (Spektral + Kovaryat)",
            en="16. 3D Fused Feature Space PCA Clustering (Spectral + Covariate)",
        ),
        fontsize=12.5,
        fontweight="bold",
        color=PALETTE["navy"],
        pad=12,
    )
    ax.set_xlabel(f"PC1 ({exp_var[0]:.1f}%)", labelpad=8)
    ax.set_ylabel(f"PC2 ({exp_var[1]:.1f}%)", labelpad=8)
    ax.set_zlabel(f"PC3 ({exp_var[2]:.1f}%)", labelpad=8)
    ax.view_init(elev=24, azim=-55)

    return save_figure(
        fig,
        plots_dir,
        "16_pca_3d_gizli_uzay",
        config=config,
        plot_id="16_pca_3d_latent_space",
        title=pick_text(config.lang, tr="16. 3D PCA Gizli Öznitelik Uzayı (3D)", en="16. 3D PCA Latent Feature Space (3D)"),
        category=t("cat_model", config.lang),
        is_3d=True,
        description=pick_text(
            config.lang,
            tr="Birleşik spektral-kovaryat öznitelik uzayının ilk 3 temel bileşeninde SOC gradyanı.",
            en="SOC gradient across the first 3 principal components of fused spectral-covariate space.",
        ),
        dataset_source=dataset_source,
        subtitle=pick_text(
            config.lang,
            tr=f"İlk 3 bileşen toplam açıklanan varyans: %{float(np.sum(exp_var)):.1f}",
            en=f"Top 3 components cumulative explained variance: {float(np.sum(exp_var)):.1f}%",
        ),
    )


def generate_model_plots(
    ctx: ModelPlotContext,
    arrays: DatasetArrays,
    *,
    plots_dir: Path,
    config: VisualizationConfig,
    dataset_source: str,
) -> list[PlotArtifactRecord]:
    """Generate all 9 model evaluation, uncertainty, OOD, and 3D PCA plots."""
    return [
        plot_08_model_comparison_metrics(ctx, plots_dir=plots_dir, config=config, dataset_source=dataset_source),
        plot_09_parity_and_ci90(ctx, arrays, plots_dir=plots_dir, config=config, dataset_source=dataset_source),
        plot_10_bland_altman(ctx, plots_dir=plots_dir, config=config, dataset_source=dataset_source),
        plot_11_residual_diagnostics_4panel(ctx, arrays, plots_dir=plots_dir, config=config, dataset_source=dataset_source),
        plot_12_subgroup_error_heatmap(ctx, arrays, plots_dir=plots_dir, config=config, dataset_source=dataset_source),
        plot_13_conformal_calibration_coverage(ctx, plots_dir=plots_dir, config=config, dataset_source=dataset_source),
        plot_14_uncertainty_decomposition(ctx, arrays, plots_dir=plots_dir, config=config, dataset_source=dataset_source),
        plot_15_ood_applicability_domain(ctx, arrays, plots_dir=plots_dir, config=config, dataset_source=dataset_source),
        plot_16_pca_3d_latent_space(ctx, arrays, plots_dir=plots_dir, config=config, dataset_source=dataset_source),
    ]
