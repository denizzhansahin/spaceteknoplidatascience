"""Visual summary/audit tables, 3D covariate surface, CSV/MD exports & HTML dashboard (Plots 24-26)."""

from __future__ import annotations

import base64
import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from st_ds import __version__
from st_ds.config import resolve_lang
from st_ds.i18n import pick_text, t
from st_ds.io.dataset import SoilDataset
from st_ds.visualization.base import (
    PALETTE,
    DatasetArrays,
    PlotArtifactRecord,
    TableArtifactRecord,
    VisualizationConfig,
    export_table_files,
    extract_row_summary_stats,
    render_notice_figure,
    render_table_figure,
    save_figure,
)
from st_ds.visualization.model_plots import ModelPlotContext
from st_ds.visualization.science_plots import SpatialPlotBundle
from st_ds.xai.engine import XAIExplanationReport


def plot_26_covariate_3d_response_surface(
    ctx: ModelPlotContext,
    arrays: DatasetArrays,
    *,
    plots_dir: Path,
    config: VisualizationConfig,
    dataset_source: str,
) -> PlotArtifactRecord:
    """Plot 26 (3D #4): 3D Covariate response surface (Moisture % × pH × Predicted/True Carbon)."""
    lang = resolve_lang(config.lang)
    z_vals = np.where(np.isfinite(ctx.y_true), ctx.y_true, ctx.y_pred)
    x_candidates = [
        (pick_text(lang, tr="Volümetrik Nem (%)", en="Volumetric Moisture (%)"), arrays.moisture_pct),
        (pick_text(lang, tr="Toprak pH", en="Soil pH"), arrays.ph),
        (pick_text(lang, tr="Sıcaklık (°C)", en="Temperature (°C)"), arrays.temperature_c),
        ("EC (dS/m)", arrays.ec_ds_m),
        (pick_text(lang, tr="Hacim Ağırlığı (g/cm³)", en="Bulk Density (g/cm³)"), arrays.bulk_density),
    ]
    usable = [
        (lbl, arr)
        for lbl, arr in x_candidates
        if np.sum(np.isfinite(arr)) >= 6 and float(np.std(arr[np.isfinite(arr)])) > 1e-8
    ]
    if len(usable) < 2:
        return render_notice_figure(
            plots_dir,
            "26_kovaryat_3d_iliski_yuzeyi",
            config=config,
            plot_id="26_covariate_3d_surface",
            title=pick_text(lang, tr="26. 3D Kovaryat İlişki Yüzeyi (3D)", en="26. 3D Covariate Relationship Surface (3D)"),
            category=t("cat_tables_3d", lang),
            notice_tr=pick_text(
                lang,
                tr="3D kovaryat yüzey grafiği için en az iki değişken kovaryat gereklidir.",
                en="At least two varying covariates are required for the 3D covariate response surface.",
            ),
            dataset_source=dataset_source,
            is_3d=True,
        )

    (x_label, x_arr), (y_label, y_arr) = usable[0], usable[1]
    mask = np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(z_vals)
    xv, yv, zv = x_arr[mask], y_arr[mask], z_vals[mask]

    # Fit smooth quadratic response surface Z ~ b0 + b1*x + b2*y + b3*x^2 + b4*x*y + b5*y^2
    xn = (xv - np.mean(xv)) / max(float(np.std(xv)), 1e-6)
    yn = (yv - np.mean(yv)) / max(float(np.std(yv)), 1e-6)
    design = np.column_stack((np.ones_like(xn), xn, yn, xn**2, xn * yn, yn**2))
    beta, *_ = np.linalg.lstsq(design, zv, rcond=None)

    gx = np.linspace(float(np.min(xv)), float(np.max(xv)), 22)
    gy = np.linspace(float(np.min(yv)), float(np.max(yv)), 22)
    GX, GY = np.meshgrid(gx, gy)
    GXn = (GX - np.mean(xv)) / max(float(np.std(xv)), 1e-6)
    GYn = (GY - np.mean(yv)) / max(float(np.std(yv)), 1e-6)
    GZ = (
        beta[0]
        + beta[1] * GXn
        + beta[2] * GYn
        + beta[3] * (GXn**2)
        + beta[4] * (GXn * GYn)
        + beta[5] * (GYn**2)
    )

    fig = plt.figure(figsize=(11.2, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(GX, GY, GZ, cmap="coolwarm", alpha=0.52, edgecolor="none")
    sc = ax.scatter(
        xv,
        yv,
        zv,
        c=zv,
        cmap="viridis",
        s=52,
        edgecolor="#1A202C",
        depthshade=False,
        label=pick_text(lang, tr="Ölçümler", en="Samples"),
    )
    fig.colorbar(sc, ax=ax, shrink=0.62, pad=0.08).set_label(f"{ctx.target_name} (%)")

    ax.set_title(
        pick_text(
            lang,
            tr=f"26. 3D Kovaryat Tepki Yüzeyi ({x_label} × {y_label} × {ctx.target_name})",
            en=f"26. 3D Covariate Response Surface ({x_label} × {y_label} × {ctx.target_name})",
        ),
        fontsize=12.4,
        fontweight="bold",
        color=PALETTE["navy"],
        pad=12,
    )
    ax.set_xlabel(x_label, labelpad=8)
    ax.set_ylabel(y_label, labelpad=8)
    ax.set_zlabel(f"{ctx.target_name} (%)", labelpad=8)
    ax.view_init(elev=25, azim=-50)

    return save_figure(
        fig,
        plots_dir,
        "26_kovaryat_3d_iliski_yuzeyi",
        config=config,
        plot_id="26_covariate_3d_surface",
        title=pick_text(lang, tr="26. 3D Kovaryat Tepki Yüzeyi (3D)", en="26. 3D Covariate Response Surface (3D)"),
        category=t("cat_tables_3d", lang),
        is_3d=True,
        description=pick_text(
            lang,
            tr=f"En bilgilendirici iki toprak kovaryatı ({x_label}, {y_label}) üzerinde ikinci derece tepki yüzeyi.",
            en=f"Quadratic response surface over the two most informative soil covariates ({x_label}, {y_label}).",
        ),
        dataset_source=dataset_source,
    )


def build_and_export_tables(
    *,
    dataset: SoilDataset,
    arrays: DatasetArrays,
    ctx: ModelPlotContext,
    xai_report: XAIExplanationReport | None,
    tables_dir: Path,
    plots_dir: Path,
    config: VisualizationConfig,
    dataset_source: str,
) -> tuple[list[TableArtifactRecord], list[PlotArtifactRecord]]:
    """Create 5 synchronized CSV/MD tables plus visual Table PNGs #24 and #25."""
    lang = resolve_lang(config.lang)
    tables: list[TableArtifactRecord] = []
    visual_tables: list[PlotArtifactRecord] = []

    # Table 1: Dataset & Covariate Summary Statistics
    t1_headers = (
        ["Değişken / Parametre", "Birim", "Geçerli (n)", "Minimum", "Medyan", "Ort ± SS", "Maksimum"]
        if lang == "tr"
        else ["Variable / Parameter", "Unit", "Valid (n)", "Min", "Median", "Mean ± SD", "Max"]
    )
    summary_vars = [
        (pick_text(lang, tr="Organik Karbon (SOC)", en="Soil Organic Carbon (SOC)"), "%", arrays.soc),
        (pick_text(lang, tr="İnorganik Karbon (SIC)", en="Soil Inorganic Carbon (SIC)"), "%", arrays.sic),
        (pick_text(lang, tr="Toplam Karbon (TC)", en="Total Carbon (TC)"), "%", arrays.tc),
        (pick_text(lang, tr="Volümetrik Nem", en="Volumetric Moisture"), "%", arrays.moisture_pct),
        (pick_text(lang, tr="Toprak pH", en="Soil pH"), "pH", arrays.ph),
        (pick_text(lang, tr="Toprak Sıcaklığı", en="Soil Temperature"), "°C", arrays.temperature_c),
        (pick_text(lang, tr="Elektriksel İletkenlik (EC)", en="Electrical Conductivity (EC)"), "dS/m", arrays.ec_ds_m),
        (pick_text(lang, tr="Hacim Ağırlığı (Bulk Density)", en="Bulk Density"), "g/cm³", arrays.bulk_density),
        (pick_text(lang, tr="Kaba Parça Oranı", en="Coarse Fragments"), "0-1", arrays.coarse_fraction),
        (pick_text(lang, tr="Katman Alt Derinliği", en="Layer Bottom Depth"), "cm", arrays.depth_bottom_cm),
    ]
    t1_rows: list[list[str]] = []
    for name, unit, arr in summary_vars:
        st = extract_row_summary_stats(arr)
        t1_rows.append([name, unit, st["n"], st["min"], st["median"], st["mean_sd"], st["max"]])

    tables.append(
        export_table_files(
            tables_dir,
            "01_veri_seti_ozet_istatistikleri",
            "01_dataset_summary",
            pick_text(lang, tr="Tablo 01: Veri Seti Karbon ve Kovaryat Özet İstatistikleri", en="Table 01: Dataset Carbon & Covariate Summary Statistics"),
            t1_headers,
            t1_rows,
        )
    )
    visual_tables.append(
        render_table_figure(
            t1_headers,
            t1_rows,
            plots_dir=plots_dir,
            stem="24_veri_seti_ozet_tablosu",
            config=config,
            plot_id="24_dataset_summary_table",
            title=pick_text(lang, tr="24. Veri Seti Karbon ve Kovaryat Özet İstatistik Tablosu", en="24. Dataset Carbon & Covariate Summary Statistics Table"),
            category=t("cat_tables_3d", lang),
            description=pick_text(lang, tr="Veri setindeki hedef karbon fraksiyonları ve sensör kovaryatlarının özet dağılım tablosu.", en="Summary distribution of target carbon fractions and sensor covariates."),
            dataset_source=dataset_source,
            subtitle=pick_text(
                lang,
                tr=f"Veri seti={dataset.dataset_id} | Toplam ölçüm={len(dataset.records)} | Saha={len(dataset.unique_site_ids)}",
                en=f"Dataset={dataset.dataset_id} | Samples={len(dataset.records)} | Sites={len(dataset.unique_site_ids)}",
            ),
        )
    )

    # Table 2: Model Comparison Metrics
    t2_headers = (
        ["Sıra", "Model Adı", "Model Ailesi", "OOF RMSE (%)", "OOF MAE (%)", "Bias (%)", "R²", "RPIQ", "LoA Alt / Üst"]
        if lang == "tr"
        else ["Rank", "Model Name", "Family", "OOF RMSE (%)", "OOF MAE (%)", "Bias (%)", "R²", "RPIQ", "LoA Low / High"]
    )
    t2_rows: list[list[str]] = []
    if ctx.benchmark is not None:
        for rank, m_name in enumerate(ctx.benchmark.evidence_ranking, start=1):
            e = ctx.benchmark.entries[m_name]
            m = e.oof_metrics
            t2_rows.append(
                [
                    str(rank),
                    e.model_name,
                    e.model_family,
                    f"{m.rmse:.4f}",
                    f"{m.mae:.4f}",
                    f"{m.bias:+.4f}",
                    f"{m.r2:.4f}" if m.r2 is not None else "-",
                    f"{m.rpiq:.3f}" if m.rpiq is not None else "-",
                    f"[{m.bland_altman_loa_lower:+.3f}, {m.bland_altman_loa_upper:+.3f}]",
                ]
            )
    if not t2_rows:
        t2_rows.append(["1", pick_text(lang, tr="Tekil Artefakt Modeli", en="Single Artifact Model"), "-", "-", "-", "-", "-", "-", "-"])
    tables.append(
        export_table_files(
            tables_dir,
            "02_model_karsilastirma_metrikleri",
            "02_model_benchmark",
            pick_text(lang, tr="Tablo 02: Gruplanmış Çapraz Doğrulama Model Kıyaslama Metrikleri", en="Table 02: Grouped Cross-Validation Model Benchmark Metrics"),
            t2_headers,
            t2_rows,
        )
    )

    # Table 3: Subgroup Performance Matrix
    t3_headers = (
        ["Saha (site_id)", "Nem Senaryosu", "Örnek Sayısı", "RMSE (%)", "MAE (%)", "Ortalama Bias (%)"]
        if lang == "tr"
        else ["Site (site_id)", "Moisture Scenario", "Samples", "RMSE (%)", "MAE (%)", "Mean Bias (%)"]
    )
    t3_rows: list[list[str]] = []
    mask_eval = np.isfinite(ctx.y_true) & np.isfinite(ctx.y_pred)
    for s_id in sorted(set(arrays.site_ids)):
        for sc in sorted(set(arrays.moisture_scenarios)):
            sub_m = mask_eval & np.array(
                [arrays.site_ids[i] == s_id and arrays.moisture_scenarios[i] == sc for i in range(len(arrays.records))]
            )
            if np.any(sub_m):
                diff = ctx.y_pred[sub_m] - ctx.y_true[sub_m]
                t3_rows.append(
                    [
                        s_id,
                        sc,
                        str(int(np.sum(sub_m))),
                        f"{float(np.sqrt(np.mean(np.square(diff)))):.4f}",
                        f"{float(np.mean(np.abs(diff))):.4f}",
                        f"{float(np.mean(diff)):+.4f}",
                    ]
                )
    if not t3_rows:
        t3_rows.append(["-", "-", "0", "-", "-", "-"])
    tables.append(
        export_table_files(
            tables_dir,
            "03_alt_grup_hata_matrisi",
            "03_subgroup_errors",
            pick_text(lang, tr="Tablo 03: Saha ve Nem Senaryosuna Göre Alt Grup Hata Dağılımı", en="Table 03: Subgroup Error Distribution by Site & Moisture Scenario"),
            t3_headers,
            t3_rows,
        )
    )

    # Table 4: XAI Component Ranking
    t4_headers = (
        ["Fiziksel Alan", "Sıra", "Değişken Adı", "Birim", "Önem Skoru", "Hedef Değişken"]
        if lang == "tr"
        else ["Physical Domain", "Rank", "Feature Name", "Unit", "Importance Score", "Target"]
    )
    t4_rows: list[list[str]] = []
    if xai_report is not None:
        cr = xai_report.component_ranking
        for grp_label, lst in [
            (pick_text(lang, tr="Spektral Bant", en="Spectral Band"), cr.spectral_ranking[:10]),
            (pick_text(lang, tr="Toprak Kovaryatı", en="Soil Covariate"), cr.covariate_ranking[:10]),
            (pick_text(lang, tr="Türetilmiş EIS", en="Derived EIS"), cr.derived_eis_ranking[:6]),
        ]:
            for item in lst:
                t4_rows.append([grp_label, str(item.rank), item.name, item.unit, f"{item.score:.5f}", item.target_name])
    if not t4_rows:
        t4_rows.append(["-", "1", pick_text(lang, tr="XAI verisi yok", en="No XAI data"), "-", "-", ctx.target_name])
    tables.append(
        export_table_files(
            tables_dir,
            "04_xai_bilesen_onem_siralamasi",
            "04_xai_ranking",
            pick_text(lang, tr="Tablo 04: Birim-Ayrıştırılmış XAI Bileşen Önem Sıralaması", en="Table 04: Unit-Separated XAI Component Importance Ranking"),
            t4_headers,
            t4_rows,
        )
    )

    # Table 5: Sample-Level Prediction & Stock Audit Table (+ Visual Table Figure #25)
    t5_headers = (
        ["Örnek ID", "Saha", "Tahmin (%)", "%90 CI Aralığı", "SIC / TC (%)", "Stok (Mg/ha)", "Baskın Belirsizlik", "OOD Skoru", "Durum"]
        if lang == "tr"
        else ["Sample ID", "Site", "Pred (%)", "90% CI Interval", "SIC / TC (%)", "Stock (Mg/ha)", "Dominant Unc.", "OOD Score", "Status"]
    )
    t5_rows: list[list[str]] = []
    for idx, rep in enumerate(ctx.reports):
        if rep is None:
            t5_rows.append([arrays.sample_ids[idx], arrays.site_ids[idx], t("error", lang), "-", "-", "-", "-", "-", "FAILED"])
            continue
        cm = rep.carbon_metrics
        ci = cm.get("confidence_interval") or {}
        cs = rep.carbon_stock or {}
        unc = cs.get("uncertainty") or {}
        pred_val = cm.get(ctx.target_name)
        if pred_val is None and ctx.target_name in ("soc_pct", "sic_pct", "tc_pct"):
            pred_val = cm.get("soc_pct")
        pred_str = f"{float(pred_val):.3f}" if pred_val is not None else "-"
        ci_str = (
            f"[{float(ci['lower_pct']):.2f}, {float(ci['upper_pct']):.2f}]"
            if "lower_pct" in ci and "upper_pct" in ci
            else "-"
        )
        if cm.get("sic_pct") is not None or cm.get("tc_pct") is not None:
            sic_str = f"{float(cm['sic_pct']):.2f}" if cm.get("sic_pct") is not None else "-"
            tc_str = f"{float(cm['tc_pct']):.2f}" if cm.get("tc_pct") is not None else "-"
            sic_tc_str = f"{sic_str} / {tc_str}"
        elif np.isfinite(arrays.sic[idx]) or np.isfinite(arrays.tc[idx]):
            ref_sic = f"{float(arrays.sic[idx]):.2f}" if np.isfinite(arrays.sic[idx]) else "-"
            ref_tc = f"{float(arrays.tc[idx]):.2f}" if np.isfinite(arrays.tc[idx]) else "-"
            sic_tc_str = f"Ref {ref_sic}/{ref_tc}"
        else:
            sic_tc_str = pick_text(lang, tr="Yok (tek hedef)", en="N/A (soc-only)")
        st_val = cs.get("stock_mg_ha")
        st_str = f"{float(st_val):.1f}" if st_val is not None else t("missing_bd", lang)
        dom_src = str(unc.get("dominant_uncertainty_source") or "-")
        ood_sc = rep.diagnostics.get("ood_score")
        ood_str = f"{float(ood_sc):.2f}" if ood_sc is not None else "-"
        w_codes = list(rep.diagnostics.get("warning_codes") or ())
        if rep.status.value == "ok":
            status_str = "ok"
        else:
            short_tags: list[str] = []
            for wc in w_codes:
                wcs = str(wc)
                if "synthetic" in wcs:
                    short_tags.append(pick_text(lang, tr="sentetik", en="synthetic"))
                elif "bulk_density" in wcs:
                    short_tags.append(pick_text(lang, tr="eksik_bd", en="missing_bd"))
                elif "ood" in wcs:
                    short_tags.append("ood")
                elif "closure" in wcs:
                    short_tags.append("closure")
                else:
                    short_tags.append(wcs[:12])
            base_st = pick_text(
                lang,
                tr="uyarı" if rep.status.value == "warning" else rep.status.value,
                en=rep.status.value,
            )
            tag_label = ",".join(dict.fromkeys(short_tags[:2]))
            status_str = f"{base_st} [{tag_label}]" if tag_label else base_st
        t5_rows.append(
            [
                str(rep.identity.get("sample_id", arrays.sample_ids[idx])),
                str(rep.identity.get("site_id") or "-"),
                pred_str,
                ci_str,
                sic_tc_str,
                st_str,
                dom_src,
                ood_str,
                status_str,
            ]
        )

    tables.append(
        export_table_files(
            tables_dir,
            "05_tahmin_ve_stok_denetim_raporu",
            "05_prediction_audit",
            pick_text(lang, tr="Tablo 05: Örnek Bazlı Tahmin, Güven Aralığı, Stok ve Kalite Denetim Raporu", en="Table 05: Sample Prediction, Confidence Interval, Stock & Quality Audit Report"),
            t5_headers,
            t5_rows,
        )
    )
    visual_tables.append(
        render_table_figure(
            t5_headers,
            t5_rows[:18],
            plots_dir=plots_dir,
            stem="25_tahmin_denetim_tablosu",
            config=config,
            plot_id="25_prediction_audit_table",
            title=pick_text(lang, tr="25. Örnek Bazlı Tahmin, Stok ve Kalite Denetim Tablosu (İlk 18 Örnek)", en="25. Sample-Level Prediction, Stock & Quality Audit Table (First 18 Samples)"),
            category=t("cat_tables_3d", lang),
            description=pick_text(lang, tr="Her ölçüm için tahmin, %90 Conformal CI, karbon stoku, baskın belirsizlik kaynağı ve OOD durumu.", en="Per-sample prediction, 90% conformal CI, carbon stock, dominant uncertainty source, and OOD status."),
            dataset_source=dataset_source,
            subtitle=pick_text(
                lang,
                tr=f"Tam liste (n={len(t5_rows)}) tables/05_tahmin_ve_stok_denetim_raporu.csv içindedir",
                en=f"Full table (n={len(t5_rows)}) exported to tables/05_tahmin_ve_stok_denetim_raporu.csv",
            ),
        )
    )

    return tables, visual_tables


def generate_optional_interactive_3d_html(
    *,
    interactive_dir: Path,
    arrays: DatasetArrays,
    ctx: ModelPlotContext,
    spatial: SpatialPlotBundle | None,
    lang: str | None = None,
) -> list[dict[str, str]]:
    """Optionally export interactive Plotly 3D HTML files if `plotly` is installed."""
    eff_lang = resolve_lang(lang)
    try:
        import plotly.graph_objects as go
    except ImportError:
        return []

    interactive_dir.mkdir(parents=True, exist_ok=True)
    generated: list[dict[str, str]] = []

    def _write_plotly(fig: Any, filename: str, title: str, first: bool) -> None:
        out_path = interactive_dir / filename
        fig.write_html(
            str(out_path),
            include_plotlyjs="directory" if first or (interactive_dir / "plotly.min.js").exists() else True,
            full_html=True,
        )
        generated.append({"title": title, "path": str(out_path), "filename": filename})

    wl = arrays.wavelengths_nm
    col_sel = np.unique(np.linspace(0, wl.size - 1, min(48, wl.size), dtype=int))
    order = np.argsort(arrays.soc) if np.all(np.isfinite(arrays.soc)) else np.arange(arrays.spectra.shape[0])
    row_sel = order[np.linspace(0, order.size - 1, min(35, order.size), dtype=int)]
    fig_spec = go.Figure(
        data=[
            go.Surface(
                x=wl[col_sel],
                y=np.arange(row_sel.size),
                z=arrays.spectra[np.ix_(row_sel, col_sel)],
                colorscale="Viridis",
            )
        ]
    )
    fig_spec.update_layout(
        title=pick_text(eff_lang, tr="Etkileşimli 3D Spektral Yüzey", en="Interactive 3D Spectral Surface"),
        scene={
            "xaxis_title": pick_text(eff_lang, tr="Dalga Boyu (nm)", en="Wavelength (nm)"),
            "yaxis_title": pick_text(eff_lang, tr="SOC Sıralı Örnek İndeksi", en="SOC-Sorted Sample Index"),
            "zaxis_title": pick_text(eff_lang, tr="Spektral Değer", en="Spectral Value"),
        },
        template="plotly_white",
    )
    _write_plotly(
        fig_spec,
        "3d_spektral_yuzey.html",
        pick_text(eff_lang, tr="3D Spektral Yüzey (Etkileşimli)", en="3D Spectral Surface (Interactive)"),
        first=True,
    )

    X_feat = ctx.fused_features if ctx.fused_features is not None else arrays.spectra
    if X_feat.shape[0] >= 4 and X_feat.shape[1] >= 3:
        Xc = X_feat - np.mean(X_feat, axis=0, keepdims=True)
        sd = np.std(Xc, axis=0, keepdims=True)
        Xs = np.where(sd > 1e-12, Xc / np.maximum(sd, 1e-12), 0.0)
        _, _, vt = np.linalg.svd(Xs, full_matrices=False)
        coords = np.dot(Xs, vt[:3].T)
        cvals = np.where(np.isfinite(ctx.y_true), ctx.y_true, ctx.y_pred)
        hover = [
            f"{pick_text(eff_lang, tr='Örnek', en='Sample')}: {sid}<br>{pick_text(eff_lang, tr='Saha', en='Site')}: {site}<br>{ctx.target_name}: {cv:.3f}%"
            for sid, site, cv in zip(arrays.sample_ids, arrays.site_ids, cvals)
        ]
        fig_pca = go.Figure(
            data=[
                go.Scatter3d(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    z=coords[:, 2],
                    mode="markers",
                    text=hover,
                    hoverinfo="text",
                    marker={"size": 6, "color": cvals, "colorscale": "Viridis", "showscale": True},
                )
            ]
        )
        fig_pca.update_layout(
            title=pick_text(eff_lang, tr="Etkileşimli 3D PCA Birleşik Öznitelik Uzayı", en="Interactive 3D PCA Fused Feature Space"),
            scene={"xaxis_title": "PC1", "yaxis_title": "PC2", "zaxis_title": "PC3"},
            template="plotly_white",
        )
        _write_plotly(
            fig_pca,
            "3d_pca_gizli_uzay.html",
            pick_text(eff_lang, tr="3D PCA Öznitelik Uzayı (Etkileşimli)", en="3D PCA Feature Space (Interactive)"),
            first=False,
        )

    if spatial is not None:
        Z_pred = spatial.kriging.predicted_values.reshape(spatial.grid_x_m.shape)
        fig_kr = go.Figure(
            data=[
                go.Surface(x=spatial.grid_x_m, y=spatial.grid_y_m, z=Z_pred, colorscale="YlGn"),
                go.Scatter3d(
                    x=spatial.sample_coords_m[:, 0],
                    y=spatial.sample_coords_m[:, 1],
                    z=spatial.sample_values,
                    mode="markers",
                    marker={"size": 5, "color": "#D55E00"},
                    name=pick_text(eff_lang, tr="Gözlem", en="Observation"),
                ),
            ]
        )
        fig_kr.update_layout(
            title=pick_text(eff_lang, tr="Etkileşimli 3D Ordinary Kriging Yüzeyi", en="Interactive 3D Ordinary Kriging Surface"),
            scene={"xaxis_title": "X (m)", "yaxis_title": "Y (m)", "zaxis_title": f"{ctx.target_name} (%)"},
            template="plotly_white",
        )
        _write_plotly(
            fig_kr,
            "3d_kriging_yuzey.html",
            pick_text(eff_lang, tr="3D Ordinary Kriging Yüzeyi (Etkileşimli)", en="3D Ordinary Kriging Surface (Interactive)"),
            first=False,
        )

    return generated


def render_html_dashboard(
    *,
    output_dir: Path,
    dataset: SoilDataset,
    plots: list[PlotArtifactRecord],
    tables: list[TableArtifactRecord],
    interactive_3d: list[dict[str, str]],
    ctx: ModelPlotContext,
    artifact_info: dict[str, Any],
    pdf_report_path: str | Path | None = None,
    lang: str | None = None,
) -> Path:
    """Build a self-contained bilingual (`tr`/`en`) offline HTML dashboard (`index.html`)."""
    eff_lang = resolve_lang(lang)
    is_synth = dataset.dataset_source.lower() == "synthetic"
    banner_cls = "banner-synth" if is_synth else "banner-real"
    banner_txt = t("dash_banner_synth" if is_synth else "dash_banner_real", eff_lang)

    best_rmse = "-"
    best_r2 = "-"
    best_model = str(artifact_info.get("model_name", "-"))
    if ctx.best_entry is not None:
        best_model = ctx.best_entry.model_name
        best_rmse = f"{ctx.best_entry.oof_metrics.rmse:.4f}%"
        best_r2 = f"{ctx.best_entry.oof_metrics.r2:.3f}" if ctx.best_entry.oof_metrics.r2 is not None else "-"

    ood_rate_txt = (
        f"%{ctx.ood_batch.ood_rate * 100:.1f}"
        if (ctx.ood_batch is not None and eff_lang == "tr")
        else (f"{ctx.ood_batch.ood_rate * 100:.1f}%" if ctx.ood_batch is not None else "-")
    )

    categories: dict[str, list[PlotArtifactRecord]] = {}
    for p in plots:
        categories.setdefault(p.category, []).append(p)

    plot_sections_html: list[str] = []
    badge_label = t("plot_3d_badge", eff_lang)
    open_svg_label = t("open_svg", eff_lang)
    for cat_name, cat_plots in categories.items():
        cards: list[str] = []
        for p in cat_plots:
            primary_file = p.files[0]
            rel_path = Path(primary_file).relative_to(output_dir).as_posix()
            badge_3d = f'<span class="pill-3d">{html.escape(badge_label)}</span>' if p.is_3d else ""
            img_tag = (
                f'<a href="{html.escape(rel_path)}" target="_blank">'
                f'<img src="{html.escape(rel_path)}" alt="{html.escape(p.title)}" loading="lazy"/></a>'
                if rel_path.endswith(".png")
                else f'<p><a href="{html.escape(rel_path)}" target="_blank">{html.escape(open_svg_label)} ({html.escape(rel_path)})</a></p>'
            )
            extra_links = " | ".join(
                f'<a href="{html.escape(Path(f).relative_to(output_dir).as_posix())}" target="_blank">{Path(f).suffix.upper().lstrip(".")}</a>'
                for f in p.files
            )
            cards.append(
                f"""
                <div class="plot-card">
                  <div class="plot-header">
                    <h3>{html.escape(p.title)} {badge_3d}</h3>
                    <span class="file-links">{extra_links}</span>
                  </div>
                  <p class="plot-desc">{html.escape(p.description)}</p>
                  <div class="plot-img-wrap">{img_tag}</div>
                </div>
                """
            )
        plot_sections_html.append(
            f'<section class="category-sec"><h2>{html.escape(cat_name)}</h2><div class="plot-grid">{"".join(cards)}</div></section>'
        )

    rows_lbl = t("rows_unit", eff_lang)
    csv_lbl = t("download_csv", eff_lang)
    md_lbl = t("view_markdown", eff_lang)
    table_links_html = "".join(
        f'<li><strong>{html.escape(t_rec.title)}</strong> ({t_rec.n_rows} {html.escape(rows_lbl)}): '
        f'<a href="{html.escape(Path(t_rec.csv_path).relative_to(output_dir).as_posix())}">{html.escape(csv_lbl)}</a> · '
        f'<a href="{html.escape(Path(t_rec.markdown_path).relative_to(output_dir).as_posix())}">{html.escape(md_lbl)}</a></li>'
        for t_rec in tables
    )

    inter_html = ""
    if interactive_3d:
        links = "".join(
            f'<a class="inter-btn" href="{html.escape(Path(item["path"]).relative_to(output_dir).as_posix())}" target="_blank">{html.escape(item["title"])}</a>'
            for item in interactive_3d
        )
        inter_html = f'<section class="category-sec"><h2>{html.escape(t("interactive_3d_h2", eff_lang))}</h2><div class="inter-row">{links}</div></section>'

    pdf_btn_html = ""
    if pdf_report_path is not None and Path(pdf_report_path).is_file():
        pdf_rel = Path(pdf_report_path).relative_to(output_dir).as_posix()
        pdf_btn_html = f'<div style="margin-bottom:18px;"><a class="inter-btn" style="background:#1B365D;" href="{html.escape(pdf_rel)}" target="_blank">📄 {html.escape(t("pdf_download_btn", eff_lang))} ({html.escape(pdf_rel)})</a></div>'

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    full_html = f"""<!DOCTYPE html>
<html lang="{eff_lang}">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>STDS v{html.escape(__version__)} — {html.escape(t("dash_title", eff_lang))}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; background: #F4F7FA; color: #1A202C; }}
    header {{ background: linear-gradient(120deg, #1B365D, #0072B2); color: #FFFFFF; padding: 24px 36px; }}
    header h1 {{ margin: 0 0 6px 0; font-size: 1.65rem; }}
    header p {{ margin: 0; opacity: 0.90; font-size: 0.92rem; }}
    .container {{ max-width: 1440px; margin: 0 auto; padding: 22px 28px 48px; }}
    .banner-synth {{ background: #FEFCBF; border-left: 6px solid #D69E2E; color: #744210; padding: 12px 16px; border-radius: 6px; font-weight: 600; margin-bottom: 18px; }}
    .banner-real {{ background: #C6F6D5; border-left: 6px solid #2F855A; color: #22543D; padding: 12px 16px; border-radius: 6px; font-weight: 600; margin-bottom: 18px; }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; margin-bottom: 24px; }}
    .kpi-card {{ background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }}
    .kpi-label {{ font-size: 0.78rem; text-transform: uppercase; color: #4A5568; letter-spacing: 0.03em; }}
    .kpi-val {{ font-size: 1.35rem; font-weight: 700; color: #1B365D; margin-top: 4px; }}
    .category-sec {{ margin-top: 28px; }}
    .category-sec h2 {{ color: #1B365D; border-bottom: 2px solid #CBD5E0; padding-bottom: 6px; font-size: 1.25rem; }}
    .plot-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(560px, 1fr)); gap: 18px; margin-top: 14px; }}
    .plot-card {{ background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px 16px; box-shadow: 0 2px 4px rgba(0,0,0,0.04); }}
    .plot-header {{ display: flex; justify-content: space-between; align-items: center; gap: 10px; }}
    .plot-header h3 {{ margin: 0; font-size: 1.02rem; color: #1A202C; }}
    .plot-desc {{ margin: 6px 0 10px; font-size: 0.86rem; color: #4A5568; }}
    .plot-img-wrap img {{ width: 100%; height: auto; border-radius: 4px; border: 1px solid #EDF2F7; }}
    .pill-3d {{ background: #E6FFFA; color: #234E52; border: 1px solid #81E6D9; font-size: 0.72rem; padding: 2px 7px; border-radius: 999px; vertical-align: middle; margin-left: 6px; }}
    .file-links a {{ color: #0072B2; font-size: 0.8rem; text-decoration: none; font-weight: 600; }}
    .inter-row {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px; }}
    .inter-btn {{ display: inline-block; background: #0072B2; color: #FFFFFF; padding: 9px 14px; border-radius: 6px; text-decoration: none; font-weight: 600; font-size: 0.88rem; }}
    .tables-box {{ background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px 20px; }}
    .tables-box li {{ margin: 7px 0; font-size: 0.92rem; }}
    footer {{ margin-top: 34px; font-size: 0.8rem; color: #718096; border-top: 1px solid #CBD5E0; padding-top: 12px; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(t("dash_header_h1", eff_lang, version=__version__))}</h1>
    <p>{t("dash_subtitle", eff_lang, dataset_id=html.escape(dataset.dataset_id), target=html.escape(ctx.target_name), lang_code=eff_lang.upper(), now=now_utc)}</p>
  </header>
  <div class="container">
    <div class="{banner_cls}">{html.escape(banner_txt)}</div>
    {pdf_btn_html}
    <div class="kpi-grid">
      <div class="kpi-card"><div class="kpi-label">{html.escape(pick_text(eff_lang, tr="Toplam Grafik", en="Total Plots"))}</div><div class="kpi-val">{len(plots)} ({sum(1 for p in plots if p.is_3d)} 3D)</div></div>
      <div class="kpi-card"><div class="kpi-label">{html.escape(pick_text(eff_lang, tr="Örnek / Saha", en="Samples / Sites"))}</div><div class="kpi-val">{len(dataset.records)} / {len(dataset.unique_site_ids)}</div></div>
      <div class="kpi-card"><div class="kpi-label">{html.escape(pick_text(eff_lang, tr="Öne Çıkan Model", en="Selected Model"))}</div><div class="kpi-val">{html.escape(best_model)}</div></div>
      <div class="kpi-card"><div class="kpi-label">OOF RMSE / R²</div><div class="kpi-val">{best_rmse} / {best_r2}</div></div>
      <div class="kpi-card"><div class="kpi-label">{html.escape(pick_text(eff_lang, tr="OOD Oranı", en="OOD Rate"))}</div><div class="kpi-val">{ood_rate_txt}</div></div>
      <div class="kpi-card"><div class="kpi-label">{html.escape(pick_text(eff_lang, tr="Analitik Tablolar", en="Analytical Tables"))}</div><div class="kpi-val">{len(tables)} CSV + MD</div></div>
    </div>

    {inter_html}

    <section class="category-sec">
      <h2>{html.escape(t("tables_h2", eff_lang))}</h2>
      <div class="tables-box"><ul>{table_links_html}</ul></div>
    </section>

    {"".join(plot_sections_html)}

    <footer>
      {t("dash_footer", eff_lang, artifact_id=html.escape(str(artifact_info.get("artifact_id", "-"))), semver=html.escape(str(artifact_info.get("semantic_version", "-"))), sha=html.escape(str(artifact_info.get("bundle_sha256", "-")))[:16])}
    </footer>
  </div>
</body>
</html>
"""
    dash_path = output_dir / "index.html"
    dash_path.write_text(full_html, encoding="utf-8")
    return dash_path
