"""Multi-page scientific PDF report generator (`PdfPages` backend, zero extra deps).

Generates comprehensive, offline-renderable TR/EN PDF reports capturing:
1. Cover, provenance, Risk 9 data-source disclaimer, and executive KPIs
2. End-to-end data lifecycle: what inputs arrived, how they were processed,
   what outputs were produced, and what spectral/covariate relationships (XAI/HSIC) emerged
3. Quantitative summary, benchmark ranking, XAI component, and prediction audit tables
4. Every generated PNG figure (01..26+) embedded with contextual titles and descriptions
"""

from __future__ import annotations

import csv
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
import numpy as np

from st_ds import SCHEMA_VERSION, __version__
from st_ds.config import LangCode, resolve_lang
from st_ds.i18n import pick_text, t

_PAGE_SIZE = (11.69, 8.27)  # A4 landscape (inches)
_NAVY = "#1B365D"
_BLUE = "#0072B2"
_TEAL = "#009E73"
_AMBER = "#D69E2E"
_SLATE = "#4A5568"
_TEXT = "#1A202C"


def _configure_pdf_fonts() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.unicode_minus": False,
        }
    )


def _new_pdf_page(
    header_title: str,
    header_sub: str,
    *,
    lang: LangCode,
    page_num: int,
    dataset_source: str = "synthetic",
) -> tuple[Figure, Any]:
    fig = plt.figure(figsize=_PAGE_SIZE, facecolor="#FFFFFF")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    # Top banner strip
    ax.add_patch(mpatches.Rectangle((0.0, 0.91), 1.0, 0.09, facecolor=_NAVY, edgecolor="none"))
    ax.text(0.03, 0.962, header_title, color="#FFFFFF", fontsize=13.5, fontweight="bold", va="center")
    ax.text(0.03, 0.928, header_sub, color="#CBD5E0", fontsize=8.8, va="center")
    ax.text(
        0.97,
        0.950,
        f"STDS v{__version__}  |  {lang.upper()}",
        color="#90CDF4",
        fontsize=9.2,
        fontweight="bold",
        ha="right",
        va="center",
    )

    # Bottom footer with provenance & page index
    is_synth = str(dataset_source).lower() == "synthetic"
    badge = t("badge_synthetic" if is_synth else "badge_real", lang)
    badge_color = "#B7791F" if is_synth else "#276749"
    ax.plot([0.03, 0.97], [0.045, 0.045], color="#E2E8F0", linewidth=0.9)
    ax.text(0.03, 0.022, badge, color=badge_color, fontsize=7.8, fontweight="bold", va="center")
    page_lbl = pick_text(lang, tr=f"Sayfa {page_num}", en=f"Page {page_num}")
    ax.text(0.97, 0.022, page_lbl, color=_SLATE, fontsize=8.5, fontweight="bold", ha="right", va="center")
    return fig, ax


def _draw_card(
    ax: Any,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    title: str,
    lines: Sequence[str],
    accent: str = _BLUE,
    bg: str = "#F8FAFC",
    font_size: float = 8.4,
    wrap_width: int = 62,
) -> None:
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            facecolor=bg,
            edgecolor="#CBD5E0",
            linewidth=1.0,
        )
    )
    ax.add_patch(mpatches.Rectangle((x, y + h - 0.036), w, 0.036, facecolor=accent, edgecolor="none"))
    ax.text(x + 0.015, y + h - 0.018, title, color="#FFFFFF", fontsize=9.6, fontweight="bold", va="center")

    wrapped_lines: list[str] = []
    for line in lines:
        if not line:
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(textwrap.wrap(str(line), width=wrap_width) or [""])

    cursor_y = y + h - 0.058
    step = min(0.026, max(0.016, (h - 0.068) / max(1, len(wrapped_lines))))
    for wline in wrapped_lines:
        if cursor_y < y + 0.012:
            break
        ax.text(x + 0.015, cursor_y, wline, color=_TEXT, fontsize=font_size, va="top")
        cursor_y -= step


def _draw_kpi_row(ax: Any, y: float, items: Sequence[tuple[str, str, str]]) -> None:
    n = max(1, len(items))
    gap = 0.015
    total_w = 0.94
    card_w = (total_w - gap * (n - 1)) / n
    for idx, (label, val, accent) in enumerate(items):
        cx = 0.03 + idx * (card_w + gap)
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (cx, y),
                card_w,
                0.115,
                boxstyle="round,pad=0.005,rounding_size=0.010",
                facecolor="#FFFFFF",
                edgecolor="#CBD5E0",
                linewidth=1.0,
            )
        )
        ax.add_patch(mpatches.Rectangle((cx, y + 0.106), card_w, 0.009, facecolor=accent, edgecolor="none"))
        ax.text(cx + 0.012, y + 0.080, label, color=_SLATE, fontsize=7.8, fontweight="bold", va="center")
        ax.text(cx + 0.012, y + 0.036, val, color=_NAVY, fontsize=11.5, fontweight="bold", va="center")


def _render_table_section(
    pdf: PdfPages,
    *,
    title: str,
    subtitle: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    lang: LangCode,
    page_counter: list[int],
    dataset_source: str,
    max_rows_per_page: int = 20,
) -> None:
    if not rows:
        rows = [[t("no_data", lang)] + ["-"] * max(0, len(headers) - 1)]
    chunks = [rows[i : i + max_rows_per_page] for i in range(0, len(rows), max_rows_per_page)]
    for part_idx, chunk in enumerate(chunks):
        page_counter[0] += 1
        part_suffix = f" ({part_idx + 1}/{len(chunks)})" if len(chunks) > 1 else ""
        fig, _ = _new_pdf_page(
            f"{title}{part_suffix}",
            subtitle,
            lang=lang,
            page_num=page_counter[0],
            dataset_source=dataset_source,
        )
        ax_tbl = fig.add_axes([0.04, 0.08, 0.92, 0.79])
        ax_tbl.axis("off")
        tbl = ax_tbl.table(
            cellText=[list(map(str, r)) for r in chunk],
            colLabels=[str(h) for h in headers],
            loc="upper center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        n_cols = max(1, len(headers))
        tbl.set_fontsize(8.0 if n_cols <= 7 else (7.0 if n_cols <= 9 else 6.2))
        tbl.scale(1.0, 1.48)
        for (r_idx, _c_idx), cell in tbl.get_celld().items():
            cell.set_edgecolor("#CBD5E0")
            cell.set_linewidth(0.6)
            if r_idx == 0:
                cell.set_facecolor(_NAVY)
                cell.set_text_props(color="#FFFFFF", fontweight="bold")
            elif r_idx % 2 == 1:
                cell.set_facecolor("#FFFFFF")
            else:
                cell.set_facecolor("#F7FAFC")
        pdf.savefig(fig, dpi=150)
        plt.close(fig)


def _read_csv_rows(csv_path: Path) -> tuple[list[str], list[list[str]]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = list(csv.reader(handle))
    if not reader:
        return [], []
    return reader[0], reader[1:]


def _summarize_arrays_for_lifecycle(
    dataset: Any,
    arrays: Any,
    ctx: Any,
    xai_report: Any,
    artifact_info: Mapping[str, Any],
    chem_config: Mapping[str, Any] | None,
    lang: LangCode,
) -> tuple[list[str], list[str], list[str], list[str]]:
    wl = arrays.wavelengths_nm
    n_rec = len(dataset.records)
    n_sites = len(dataset.unique_site_ids)
    n_bd_miss = int(np.sum(~np.isfinite(arrays.bulk_density)))
    has_eis = arrays.eis_frequencies is not None

    def _fmt_range(arr: np.ndarray, unit: str = "") -> str:
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            return pick_text(lang, tr="ölçülmedi", en="absent")
        return f"{float(np.min(valid)):.2f}–{float(np.max(valid)):.2f} (ort/mean {float(np.mean(valid)):.2f}){unit}"

    in_lines = [
        pick_text(lang, tr=f"• Veri Kimliği: {dataset.dataset_id} ({n_rec} ölçüm, {n_sites} saha)", en=f"• Dataset ID: {dataset.dataset_id} ({n_rec} records, {n_sites} sites)"),
        pick_text(lang, tr=f"• Spektrum: {wl.size} kanal ({float(np.min(wl)):.0f}–{float(np.max(wl)):.0f} nm)", en=f"• Spectrum: {wl.size} bands ({float(np.min(wl)):.0f}–{float(np.max(wl)):.0f} nm)"),
        pick_text(lang, tr=f"• Toprak pH: {_fmt_range(arrays.ph)}", en=f"• Soil pH: {_fmt_range(arrays.ph)}"),
        pick_text(lang, tr=f"• Volümetrik Nem: {_fmt_range(arrays.moisture_pct, '%')}", en=f"• Volumetric Moisture: {_fmt_range(arrays.moisture_pct, '%')}"),
        pick_text(lang, tr=f"• Hacim Ağırlığı (ρb): {_fmt_range(arrays.bulk_density, ' g/cm³')} | Eksik: {n_bd_miss}", en=f"• Bulk Density (ρb): {_fmt_range(arrays.bulk_density, ' g/cm³')} | Missing: {n_bd_miss}"),
        pick_text(lang, tr=f"• EIS Empedans Kanalı: {'Mevcut' if has_eis else 'Yok'}", en=f"• EIS Impedance Channel: {'Present' if has_eis else 'None'}"),
    ]

    cc = dict(chem_config or {})
    sup_mode = str(cc.get("suppression_mode", "none")).upper()
    proc_lines = [
        pick_text(lang, tr="• Aşama 1 (Alan & Birim): UnitRegistry + InstrumentProfile doğrulaması", en="• Stage 1 (Domain & Units): UnitRegistry + InstrumentProfile validation"),
        pick_text(lang, tr="• Aşama 2 (Radyometri & QC): Karanlık/Beyaz normalizasyon + SNR/doygunluk", en="• Stage 2 (Radiometry & QC): Dark/White normalization + SNR/saturation"),
        pick_text(lang, tr=f"• Aşama 3 (Kemometri): Absorbance={cc.get('use_absorbance', True)}, SNV={cc.get('use_snv', True)}, Fiziksel SG (pencere={cc.get('sg_window', 7)})", en=f"• Stage 3 (Chemometrics): Absorbance={cc.get('use_absorbance', True)}, SNV={cc.get('use_snv', True)}, Physical SG (win={cc.get('sg_window', 7)})"),
        pick_text(lang, tr=f"• Aşama 4 (Nem Bastırma): {sup_mode} (eğitim uzayı SHA-256 mühürlü)", en=f"• Stage 4 (Moisture Suppression): {sup_mode} (fit space SHA-256 sealed)"),
        pick_text(lang, tr=f"• Aşama 5 (Füzyon & Bölme): Vis-NIR bantları + kovaryatlar + saha-ayrık GroupKFold", en="• Stage 5 (Fusion & Split): Vis-NIR bands + covariates + site-disjoint GroupKFold"),
        pick_text(lang, tr=f"• Aşama 6 (Paket Güvenliği): Sıfır-pickle .safetensors ({str(artifact_info.get('bundle_sha256', '-'))[:12]}...)", en=f"• Stage 6 (Bundle Security): Zero-pickle .safetensors ({str(artifact_info.get('bundle_sha256', '-'))[:12]}...)"),
    ]

    best_entry = ctx.best_entry
    best_name = best_entry.model_name if best_entry is not None else str(artifact_info.get("model_name", "-"))
    r2_str = f"{best_entry.oof_metrics.r2:.3f}" if (best_entry and best_entry.oof_metrics.r2 is not None) else "-"
    rmse_str = f"{best_entry.oof_metrics.rmse:.4f}%" if best_entry else "-"
    ci_cov = "-"
    if np.all(np.isfinite(ctx.y_true)) and ctx.y_true.size > 0:
        inside = np.mean((ctx.y_true >= ctx.ci_lower) & (ctx.y_true <= ctx.ci_upper))
        ci_cov = f"%{100.0 * float(inside):.1f}" if lang == "tr" else f"{100.0 * float(inside):.1f}%"
    ood_pct = f"%{100.0 * ctx.ood_batch.ood_rate:.1f}" if (ctx.ood_batch is not None and lang == "tr") else (f"{100.0 * ctx.ood_batch.ood_rate:.1f}%" if ctx.ood_batch is not None else "-")
    stocks = [
        float(r.carbon_stock["stock_mg_ha"])
        for r in ctx.reports
        if r is not None and r.carbon_stock and r.carbon_stock.get("stock_mg_ha") is not None
    ]
    stock_str = f"{min(stocks):.1f}–{max(stocks):.1f} Mg/ha (n={len(stocks)})" if stocks else pick_text(lang, tr="Hesaplanmadı (eksik HA)", en="Not computed (missing BD)")

    out_lines = [
        pick_text(lang, tr=f"• Seçilen Model: {best_name} (OOF R²={r2_str}, RMSE={rmse_str})", en=f"• Selected Model: {best_name} (OOF R²={r2_str}, RMSE={rmse_str})"),
        pick_text(lang, tr=f"• Tahmin {ctx.target_name}: {_fmt_range(ctx.y_pred, '%')}", en=f"• Predicted {ctx.target_name}: {_fmt_range(ctx.y_pred, '%')}"),
        pick_text(lang, tr=f"• Referans SOC / SIC / TC ort: {np.nanmean(arrays.soc):.2f}% / {np.nanmean(arrays.sic):.2f}% / {np.nanmean(arrays.tc):.2f}%", en=f"• Reference SOC / SIC / TC mean: {np.nanmean(arrays.soc):.2f}% / {np.nanmean(arrays.sic):.2f}% / {np.nanmean(arrays.tc):.2f}%"),
        pick_text(lang, tr=f"• %90 Conformal Kapsam: {ci_cov} | OOD Risk Oranı: {ood_pct}", en=f"• 90% Conformal Coverage: {ci_cov} | OOD Risk Rate: {ood_pct}"),
        pick_text(lang, tr=f"• Karbon Stoku (0-30 cm): {stock_str}", en=f"• Carbon Stock (0-30 cm): {stock_str}"),
        pick_text(lang, tr=f"• Başarısız / Uyarı Kaydı: {sum(1 for r in ctx.reports if r is None)} hata", en=f"• Failed / Error Records: {sum(1 for r in ctx.reports if r is None)} failed"),
    ]

    rel_lines: list[str] = []
    if xai_report is not None:
        cr = xai_report.component_ranking
        top_spec = ", ".join(f"{it.name} ({it.score:.3f})" for it in cr.spectral_ranking[:4]) or "-"
        top_cov = ", ".join(f"{it.name} ({it.score:.3f})" for it in cr.covariate_ranking[:4]) or "-"
        hsic_obj = xai_report.dataset_dependence_hsic
        hsic_status = str(hsic_obj.status)
        top_inter = "-"
        if xai_report.feature_interactions:
            best_i = max(xai_report.feature_interactions, key=lambda x: abs(float(x.interaction_score)))
            top_inter = f"{best_i.feature_i} × {best_i.feature_j} ({best_i.interaction_score:.4f})"
        rel_lines = [
            pick_text(lang, tr=f"• En Etkili Spektral Bantlar (IG/Duyarlılık): {top_spec}", en=f"• Top Spectral Bands (IG/Sensitivity): {top_spec}"),
            pick_text(lang, tr=f"• En Etkili Toprak Kovaryatları: {top_cov}", en=f"• Top Soil Covariates: {top_cov}"),
            pick_text(lang, tr=f"• İstatistiksel Bağımlılık (RBF-HSIC): durum={hsic_status} (n={hsic_obj.n_samples})", en=f"• Statistical Dependence (RBF-HSIC): status={hsic_status} (n={hsic_obj.n_samples})"),
            pick_text(lang, tr=f"• Baskın 2. Derece Özellik Etkileşimi: {top_inter}", en=f"• Dominant 2nd-Order Interaction: {top_inter}"),
            pick_text(lang, tr="• Bilimsel Not (ADR-006): XAI korelasyon ve yerel model duyarlılığını gösterir, nedensellik değildir.", en="• Scientific Note (ADR-006): XAI quantifies correlation and local model sensitivity, not causality."),
        ]
    else:
        rel_lines = [
            pick_text(lang, tr="• Bu çalıştırmada ayrı XAI raporu üretilmedi; spektral/kovaryat korelasyonları grafiklerde yer alır.", en="• Standalone XAI report was not attached; spectral/covariate correlations appear in figures."),
        ]
    return in_lines, proc_lines, out_lines, rel_lines


def generate_analysis_pdf_report(
    output_pdf: str | Path,
    *,
    dataset: Any,
    arrays: Any,
    ctx: Any,
    plots: Sequence[Any],
    tables: Sequence[Any],
    xai_report: Any = None,
    artifact_info: Mapping[str, Any] | None = None,
    chem_config: Mapping[str, Any] | None = None,
    lang: str | None = None,
) -> Path:
    """Generate the full multi-page scientific PDF report embedding tables and ALL PNG figures."""
    _configure_pdf_fonts()
    eff_lang = resolve_lang(lang)
    pdf_path = Path(output_pdf)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    art = dict(artifact_info or {})
    ds_source = str(getattr(dataset, "dataset_source", "synthetic"))
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    page_counter = [1]

    best_entry = ctx.best_entry
    best_model = best_entry.model_name if best_entry is not None else str(art.get("model_name", "-"))
    r2_val = f"{best_entry.oof_metrics.r2:.3f}" if (best_entry and best_entry.oof_metrics.r2 is not None) else "-"
    rmse_val = f"{best_entry.oof_metrics.rmse:.4f}%" if best_entry else "-"
    ood_val = (
        f"%{ctx.ood_batch.ood_rate * 100:.1f}"
        if (ctx.ood_batch is not None and eff_lang == "tr")
        else (f"{ctx.ood_batch.ood_rate * 100:.1f}%" if ctx.ood_batch is not None else "-")
    )

    with PdfPages(pdf_path) as pdf:
        # ---- Page 1: Cover & Executive Summary ----
        fig, ax = _new_pdf_page(
            t("pdf_cover_title", eff_lang),
            t("pdf_cover_sub", eff_lang),
            lang=eff_lang,
            page_num=page_counter[0],
            dataset_source=ds_source,
        )
        kpis = [
            (pick_text(eff_lang, tr="ÖRNEK / SAHA", en="SAMPLES / SITES"), f"{len(dataset.records)} / {len(dataset.unique_site_ids)}", _BLUE),
            (pick_text(eff_lang, tr="HEDEF DEĞİŞKEN", en="TARGET FRACTION"), str(ctx.target_name), _TEAL),
            (pick_text(eff_lang, tr="SEÇİLEN MODEL", en="SELECTED MODEL"), best_model[:18], _NAVY),
            ("OOF R² / RMSE", f"{r2_val} | {rmse_val}", "#805AD5"),
            (pick_text(eff_lang, tr="OOD RİSK ORANI", en="OOD RISK RATE"), ood_val, _AMBER),
            (pick_text(eff_lang, tr="GRAFİK / TABLO", en="PLOTS / TABLES"), f"{len(plots)} PNG | {len(tables)} CSV", "#D55E00"),
        ]
        _draw_kpi_row(ax, 0.75, kpis)

        in_lines, proc_lines, out_lines, rel_lines = _summarize_arrays_for_lifecycle(
            dataset, arrays, ctx, xai_report, art, chem_config, eff_lang
        )
        _draw_card(
            ax,
            0.03,
            0.40,
            0.455,
            0.31,
            title=pick_text(eff_lang, tr="Gelen Veri Profili & Kapsam", en="Incoming Data Profile & Scope"),
            lines=in_lines,
            accent=_BLUE,
        )
        _draw_card(
            ax,
            0.515,
            0.40,
            0.455,
            0.31,
            title=pick_text(eff_lang, tr="Uygulanan Analiz ve Kemometri Hattı", en="Applied Calibration & Chemometrics"),
            lines=proc_lines,
            accent=_TEAL,
        )
        _draw_card(
            ax,
            0.03,
            0.08,
            0.455,
            0.29,
            title=pick_text(eff_lang, tr="Üretilen Tahmin, Stok ve Kalite Çıktıları", en="Predictions, Carbon Stock & Quality"),
            lines=out_lines,
            accent=_NAVY,
        )
        _draw_card(
            ax,
            0.515,
            0.08,
            0.455,
            0.29,
            title=pick_text(eff_lang, tr="Değişken İlişkileri & XAI Bulguları", en="Variable Relationships & XAI Insights"),
            lines=rel_lines,
            accent="#805AD5",
        )
        ax.text(
            0.03,
            0.725,
            pick_text(
                eff_lang,
                tr=f"Oluşturulma: {now_str}  |  Artefakt: {art.get('artifact_id', '-')} (v{art.get('semantic_version', __version__)})  |  Şema: {SCHEMA_VERSION}",
                en=f"Generated: {now_str}  |  Artifact: {art.get('artifact_id', '-')} (v{art.get('semantic_version', __version__)})  |  Schema: {SCHEMA_VERSION}",
            ),
            color=_SLATE,
            fontsize=8.2,
            fontweight="bold",
        )
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ---- Page 2..K: Exported Tables ----
        for tbl in tables:
            csv_p = Path(tbl.csv_path)
            if not csv_p.is_file():
                continue
            headers, rows = _read_csv_rows(csv_p)
            _render_table_section(
                pdf,
                title=t("pdf_sec_tables", eff_lang),
                subtitle=str(tbl.title),
                headers=headers,
                rows=rows,
                lang=eff_lang,
                page_counter=page_counter,
                dataset_source=ds_source,
                max_rows_per_page=18,
            )

        # ---- Remaining Pages: Embed ALL Generated PNG Figures (2 per page or 1 large 3D/table) ----
        png_entries: list[tuple[Any, Path]] = []
        for p_rec in plots:
            png_candidates = [Path(f) for f in p_rec.files if str(f).lower().endswith(".png") and Path(f).is_file()]
            if png_candidates:
                png_entries.append((p_rec, png_candidates[0]))

        idx = 0
        while idx < len(png_entries):
            p_first, path_first = png_entries[idx]
            use_single = bool(p_first.is_3d or "table" in str(p_first.plot_id).lower() or idx + 1 >= len(png_entries))
            if not use_single and idx + 1 < len(png_entries):
                p_second, _ = png_entries[idx + 1]
                if p_second.is_3d or "table" in str(p_second.plot_id).lower():
                    use_single = True

            page_counter[0] += 1
            fig, ax = _new_pdf_page(
                t("pdf_sec_plots", eff_lang),
                str(p_first.category),
                lang=eff_lang,
                page_num=page_counter[0],
                dataset_source=ds_source,
            )

            if use_single:
                ax.text(0.04, 0.878, str(p_first.title), color=_NAVY, fontsize=11.0, fontweight="bold")
                desc = textwrap.shorten(str(p_first.description), width=140, placeholder="...")
                ax.text(0.04, 0.854, desc, color=_SLATE, fontsize=8.3)
                img_ax = fig.add_axes([0.04, 0.065, 0.92, 0.77])
                img_ax.axis("off")
                img_ax.imshow(mpimg.imread(str(path_first)))
                idx += 1
            else:
                p_second, path_second = png_entries[idx + 1]
                slots = [
                    (p_first, path_first, [0.04, 0.485, 0.92, 0.365], 0.875, 0.854),
                    (p_second, path_second, [0.04, 0.060, 0.92, 0.365], 0.450, 0.429),
                ]
                for rec, img_p, rect, ty, dy in slots:
                    ax.text(0.04, ty, str(rec.title), color=_NAVY, fontsize=10.0, fontweight="bold")
                    ax.text(0.04, dy, textwrap.shorten(str(rec.description), width=135, placeholder="..."), color=_SLATE, fontsize=7.9)
                    sub_ax = fig.add_axes(rect)
                    sub_ax.axis("off")
                    sub_ax.imshow(mpimg.imread(str(img_p)))
                idx += 2

            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    return pdf_path


def generate_prediction_pdf_report(
    payload: Mapping[str, Any],
    output_pdf: str | Path,
    *,
    lang: str | None = None,
) -> Path:
    """Export a clean multi-page PDF report directly from `predict`, `train`, or Appendix B JSON."""
    _configure_pdf_fonts()
    eff_lang = resolve_lang(lang)
    pdf_path = Path(output_pdf)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Normalize payload shape (single PredictionReport, batch_prediction_result, or CLI train summary)
    if "results" in payload and isinstance(payload["results"], list):
        records = list(payload["results"])
        failures = list(payload.get("failures") or [])
        ds_source = "synthetic"
        if records:
            ds_source = str((records[0].get("provenance") or {}).get("dataset_source", "synthetic"))
    elif "carbon_metrics" in payload:
        records = [dict(payload)]
        failures = []
        ds_source = str((payload.get("provenance") or {}).get("dataset_source", "synthetic"))
    else:
        records = []
        failures = []
        ds_source = str(payload.get("dataset_source", "synthetic"))

    page_counter = [1]
    with PdfPages(pdf_path) as pdf:
        fig, ax = _new_pdf_page(
            pick_text(eff_lang, tr="STDS — Analiz ve Tahmin PDF Raporu", en="STDS — Analysis & Prediction PDF Report"),
            pick_text(eff_lang, tr="Appendix B Kanonik Sözleşme Çıktısı & İzlenebilirlik Özeti", en="Appendix B Canonical Contract Output & Provenance Summary"),
            lang=eff_lang,
            page_num=page_counter[0],
            dataset_source=ds_source,
        )

        n_ok = len(records)
        n_fail = len(failures)
        cmd_name = str(payload.get("command") or ("batch_predict" if n_ok > 1 else "prediction_report"))
        art_id = str(
            payload.get("artifact_id")
            or ((records[0].get("provenance") or {}).get("artifact_id") if records else "-")
        )
        model_name = str(
            payload.get("selected_model")
            or ((records[0].get("provenance") or {}).get("model_name") if records else "-")
        )

        _draw_kpi_row(
            ax,
            0.75,
            [
                (pick_text(eff_lang, tr="İŞLEM TÜRÜ", en="OPERATION"), cmd_name[:18], _BLUE),
                (pick_text(eff_lang, tr="BAŞARILI / HATA", en="OK / FAILED"), f"{n_ok} / {n_fail}", _TEAL),
                (pick_text(eff_lang, tr="MODEL", en="MODEL"), model_name[:18], _NAVY),
                ("ARTEFAKT", art_id[:16], "#805AD5"),
            ],
        )

        meta_lines = [
            pick_text(eff_lang, tr=f"• Rapor Tarihi (UTC): {now_str}", en=f"• Report Timestamp (UTC): {now_str}"),
            pick_text(eff_lang, tr=f"• STDS Sürüm / Şema: v{__version__} / {SCHEMA_VERSION}", en=f"• STDS Version / Schema: v{__version__} / {SCHEMA_VERSION}"),
            pick_text(eff_lang, tr=f"• Veri Kaynağı Politikası: {ds_source}", en=f"• Dataset Source Policy: {ds_source}"),
            pick_text(eff_lang, tr=f"• Toplam Kayıt / Kısmi Hata: {n_ok} / {n_fail}", en=f"• Total Records / Partial Failures: {n_ok} / {n_fail}"),
        ]
        if "selected_model" in payload:
            meta_lines.append(
                pick_text(
                    eff_lang,
                    tr=f"• Seçilen Model & Hedef: {payload.get('selected_model')} ({payload.get('target', 'soc_pct')})",
                    en=f"• Selected Model & Target: {payload.get('selected_model')} ({payload.get('target', 'soc_pct')})",
                )
            )
            meta_lines.append(f"• Bundle SHA-256: {str(payload.get('bundle_sha256', '-'))[:24]}...")

        _draw_card(
            ax,
            0.03,
            0.40,
            0.94,
            0.31,
            title=pick_text(eff_lang, tr="Çalıştırma Kökeni ve Özet Metadata", en="Execution Provenance & Summary Metadata"),
            lines=meta_lines,
            accent=_BLUE,
            wrap_width=115,
        )

        # Relationships / Highlights card
        hi_lines: list[str] = []
        if records:
            socs = [float(r["carbon_metrics"]["soc_pct"]) for r in records if r.get("carbon_metrics", {}).get("soc_pct") is not None]
            stocks = [
                float(r["carbon_stock"]["stock_mg_ha"])
                for r in records
                if isinstance(r.get("carbon_stock"), dict) and r["carbon_stock"].get("stock_mg_ha") is not None
            ]
            hi_lines.append(
                pick_text(
                    eff_lang,
                    tr=f"• Tahmin Edilen SOC Aralığı: {min(socs):.3f}% – {max(socs):.3f}% (ort {np.mean(socs):.3f}%)" if socs else "• SOC yok",
                    en=f"• Predicted SOC Range: {min(socs):.3f}% – {max(socs):.3f}% (mean {np.mean(socs):.3f}%)" if socs else "• No SOC values",
                )
            )
            hi_lines.append(
                pick_text(
                    eff_lang,
                    tr=f"• Karbon Stoku Aralığı: {min(stocks):.1f} – {max(stocks):.1f} Mg C/ha" if stocks else "• Stok hesaplanmadı (eksik hacim ağırlığı fail-closed kuralı DEF-005).",
                    en=f"• Carbon Stock Range: {min(stocks):.1f} – {max(stocks):.1f} Mg C/ha" if stocks else "• Stock not computed (missing bulk density fail-closed DEF-005).",
                )
            )
            xai_first = next((r.get("xai") for r in records if r.get("xai")), None)
            if isinstance(xai_first, dict):
                cr = xai_first.get("component_ranking") or {}
                spec_top = (cr.get("spectral_ranking") or [])[:3]
                if spec_top:
                    top_txt = ", ".join(f"{it.get('name')} ({float(it.get('score', 0.0)):.3f})" for it in spec_top)
                    hi_lines.append(
                        pick_text(eff_lang, tr=f"• Baskın Spektral Bantlar (XAI): {top_txt}", en=f"• Top Spectral Bands (XAI): {top_txt}")
                    )
        elif "oof_ranking" in payload and isinstance(payload["oof_ranking"], list):
            for item in payload["oof_ranking"][:4]:
                hi_lines.append(
                    f"• {item.get('model_name')}: RMSE={item.get('rmse'):.4f}% | R²={item.get('r2'):.3f} | RPIQ={item.get('rpiq'):.2f}"
                )

        _draw_card(
            ax,
            0.03,
            0.08,
            0.94,
            0.29,
            title=pick_text(eff_lang, tr="Analitik Çıktılar & Özellik İlişkileri Özeti", en="Analytical Outputs & Feature Highlights"),
            lines=hi_lines or [pick_text(eff_lang, tr="• Ayrıntılı kayıtlar takip eden sayfalardadır.", en="• Detailed records appear on following pages.")],
            accent=_TEAL,
            wrap_width=115,
        )
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        if records:
            headers = (
                ["Örnek ID", "Saha", "SOC (%)", "%90 CI", "SIC (%)", "TC (%)", "Stok (Mg/ha)", "OOD", "Durum"]
                if eff_lang == "tr"
                else ["Sample ID", "Site", "SOC (%)", "90% CI", "SIC (%)", "TC (%)", "Stock (Mg/ha)", "OOD", "Status"]
            )
            rows: list[list[str]] = []
            for r in records:
                ident = r.get("identity") or {}
                cm = r.get("carbon_metrics") or {}
                ci = cm.get("confidence_interval") or {}
                cs = r.get("carbon_stock") or {}
                diag = r.get("diagnostics") or {}
                ci_txt = f"[{ci.get('lower_pct'):.2f}, {ci.get('upper_pct'):.2f}]" if "lower_pct" in ci and "upper_pct" in ci else "-"
                st_val = cs.get("stock_mg_ha") if isinstance(cs, dict) else None
                rows.append(
                    [
                        str(ident.get("sample_id", "-")),
                        str(ident.get("site_id") or "-"),
                        f"{float(cm['soc_pct']):.3f}" if cm.get("soc_pct") is not None else "-",
                        ci_txt,
                        f"{float(cm['sic_pct']):.2f}" if cm.get("sic_pct") is not None else "-",
                        f"{float(cm['tc_pct']):.2f}" if cm.get("tc_pct") is not None else "-",
                        f"{float(st_val):.1f}" if st_val is not None else t("missing_bd", eff_lang),
                        f"{float(diag['ood_score']):.2f}" if diag.get("ood_score") is not None else "-",
                        str(r.get("status", "-")),
                    ]
                )
            _render_table_section(
                pdf,
                title=pick_text(eff_lang, tr="Tahmin ve Stok Denetim Tablosu", en="Prediction & Stock Audit Table"),
                subtitle=pick_text(eff_lang, tr="Örnek bazında Appendix B tahmin çıktıları", en="Sample-level Appendix B prediction outputs"),
                headers=headers,
                rows=rows,
                lang=eff_lang,
                page_counter=page_counter,
                dataset_source=ds_source,
                max_rows_per_page=18,
            )

    return pdf_path
