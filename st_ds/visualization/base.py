"""Shared configuration, styling, headless backend setup, and export helpers for `st_ds.visualization`."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

# Always enforce headless rendering so CLI and CI workflows never require a display server.
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from st_ds.config import get_runtime_config, resolve_lang
from st_ds.domain.schemas import Measurement
from st_ds.i18n import t
from st_ds.io.dataset import SoilDataset

# Colorblind-safe Okabe-Ito inspired scientific palette
PALETTE: dict[str, str] = {
    "navy": "#1B365D",
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "teal": "#009E73",
    "orange": "#E69F00",
    "vermilion": "#D55E00",
    "purple": "#CC79A7",
    "gold": "#F0E442",
    "slate": "#4A5568",
    "light_gray": "#EDF2F7",
    "border": "#CBD5E0",
}

CATEGORY_COLORS: tuple[str, ...] = (
    PALETTE["blue"],
    PALETTE["vermilion"],
    PALETTE["teal"],
    PALETTE["orange"],
    PALETTE["purple"],
    PALETTE["sky"],
    PALETTE["navy"],
)


def configure_matplotlib_style() -> None:
    """Apply consistent high-readability scientific styling with Turkish UTF-8 glyphs."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.unicode_minus": False,
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FAFBFC",
            "axes.edgecolor": "#CBD5E0",
            "axes.linewidth": 0.9,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.0,
            "axes.labelcolor": "#1A202C",
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.5,
            "legend.frameon": True,
            "legend.facecolor": "#FFFFFF",
            "legend.edgecolor": "#E2E8F0",
            "grid.color": "#E2E8F0",
            "grid.linestyle": "--",
            "grid.linewidth": 0.65,
            "savefig.facecolor": "#FFFFFF",
            "savefig.bbox": "tight",
        }
    )


configure_matplotlib_style()


@dataclass(frozen=True)
class PlotArtifactRecord:
    """Metadata describing a single generated plot figure on disk."""

    plot_id: str
    title: str
    category: str
    is_3d: bool
    files: tuple[str, ...]
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "plot_id": self.plot_id,
            "title": self.title,
            "category": self.category,
            "is_3d": self.is_3d,
            "files": list(self.files),
            "description": self.description,
        }


@dataclass(frozen=True)
class TableArtifactRecord:
    """Metadata describing an analytical CSV/Markdown table exported to disk."""

    table_id: str
    title: str
    csv_path: str
    markdown_path: str
    n_rows: int
    n_cols: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "title": self.title,
            "csv_path": self.csv_path,
            "markdown_path": self.markdown_path,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
        }


@dataclass(frozen=True)
class VisualizationConfig:
    """Configuration for generating the STDS multi-plot scientific visualization suite."""

    target_name: str = "soc_pct"
    model_name: str = "auto"
    covariate_names: tuple[str, ...] | None = None
    n_folds: int = 4
    formats: tuple[str, ...] = ("png",)
    dpi: int = 160
    include_html_dashboard: bool = True
    include_interactive_3d: bool = True
    include_optional_models: bool = False
    include_pdf_report: bool | None = None
    lang: str | None = None
    seed: int = 42
    max_spectra_overlay: int = 25

    def __post_init__(self) -> None:
        normalized = tuple(fmt.strip().lower().lstrip(".") for fmt in self.formats if fmt.strip())
        if not normalized:
            raise ValueError("formats must contain at least one output format ('png' or 'svg').")
        for fmt in normalized:
            if fmt not in {"png", "svg"}:
                raise ValueError(f"Unsupported figure format '{fmt}'. Supported: png, svg.")
        object.__setattr__(self, "formats", normalized)
        object.__setattr__(self, "lang", resolve_lang(self.lang))
        if self.include_pdf_report is None:
            object.__setattr__(self, "include_pdf_report", get_runtime_config().pdf_export)
        if self.dpi < 72:
            raise ValueError("dpi must be >= 72.")
        if self.n_folds < 2:
            raise ValueError("n_folds must be >= 2.")


@dataclass(frozen=True)
class DatasetArrays:
    """Aligned numeric/categorical arrays extracted once from a SoilDataset for visualization."""

    records: tuple[Measurement, ...]
    wavelengths_nm: np.ndarray
    spectra: np.ndarray
    sample_ids: tuple[str, ...]
    site_ids: tuple[str, ...]
    moisture_scenarios: tuple[str, ...]
    soil_classes: tuple[str, ...]
    soc: np.ndarray
    sic: np.ndarray
    tc: np.ndarray
    ph: np.ndarray
    moisture_pct: np.ndarray
    temperature_c: np.ndarray
    ec_ds_m: np.ndarray
    bulk_density: np.ndarray
    coarse_fraction: np.ndarray
    latitudes: np.ndarray
    longitudes: np.ndarray
    depth_top_cm: np.ndarray
    depth_bottom_cm: np.ndarray
    eis_frequencies: np.ndarray | None
    eis_z_real: np.ndarray | None
    eis_z_imag: np.ndarray | None


def extract_dataset_arrays(dataset: SoilDataset) -> DatasetArrays:
    """Extract aligned spectra, targets, covariates, coordinates, and EIS arrays from a SoilDataset."""
    recs = tuple(dataset.records)
    if not recs:
        raise ValueError("Cannot visualize an empty SoilDataset.")

    wavelengths = np.asarray(recs[0].spectrum.wavelength_grid_nm, dtype=np.float64)
    spectra = np.vstack([np.asarray(r.spectrum.values, dtype=np.float64) for r in recs])

    def _opt_float(val: float | None) -> float:
        return float(val) if val is not None and math.isfinite(float(val)) else np.nan

    soc = np.array([_opt_float(r.reference.soc_pct if r.reference else None) for r in recs], dtype=np.float64)
    sic = np.array([_opt_float(r.reference.sic_pct if r.reference else None) for r in recs], dtype=np.float64)
    tc = np.array([_opt_float(r.reference.tc_pct if r.reference else None) for r in recs], dtype=np.float64)

    ph = np.array([_opt_float(r.covariates.ph) for r in recs], dtype=np.float64)
    moisture = np.array([_opt_float(r.covariates.moisture_volumetric_pct) for r in recs], dtype=np.float64)
    temp = np.array([_opt_float(r.covariates.temperature_c) for r in recs], dtype=np.float64)
    ec = np.array([_opt_float(r.covariates.ec_ds_m) for r in recs], dtype=np.float64)
    bd = np.array([_opt_float(r.covariates.bulk_density_g_cm3) for r in recs], dtype=np.float64)
    coarse = np.array([_opt_float(r.covariates.coarse_fragments_fraction) for r in recs], dtype=np.float64)

    lats = np.array(
        [_opt_float(r.coordinates.lat if r.coordinates else None) for r in recs],
        dtype=np.float64,
    )
    lons = np.array(
        [_opt_float(r.coordinates.lon if r.coordinates else None) for r in recs],
        dtype=np.float64,
    )
    d_top = np.array([float(r.depth_interval.top_cm) for r in recs], dtype=np.float64)
    d_bot = np.array([float(r.depth_interval.bottom_cm) for r in recs], dtype=np.float64)

    sample_ids = tuple(r.sample_id for r in recs)
    site_ids = tuple(r.site_id or "bilinmeyen_saha" for r in recs)
    scenarios = tuple(r.moisture_scenario or "belirtilmemis" for r in recs)
    soil_classes = tuple(
        (r.reference.soil_class_name if r.reference and r.reference.soil_class_name else None)
        or r.soil_texture_class
        or "belirtilmemis"
        for r in recs
    )

    eis_freqs: np.ndarray | None = None
    eis_real_list: list[np.ndarray] = []
    eis_imag_list: list[np.ndarray] = []
    for r in recs:
        if r.covariates.eis is not None:
            if eis_freqs is None:
                eis_freqs = np.asarray(r.covariates.eis.frequencies_hz, dtype=np.float64)
            eis_real_list.append(np.asarray(r.covariates.eis.z_real_ohm, dtype=np.float64))
            eis_imag_list.append(np.asarray(r.covariates.eis.z_imag_ohm, dtype=np.float64))
    eis_real = np.vstack(eis_real_list) if eis_real_list else None
    eis_imag = np.vstack(eis_imag_list) if eis_imag_list else None

    return DatasetArrays(
        records=recs,
        wavelengths_nm=wavelengths,
        spectra=spectra,
        sample_ids=sample_ids,
        site_ids=site_ids,
        moisture_scenarios=scenarios,
        soil_classes=soil_classes,
        soc=soc,
        sic=sic,
        tc=tc,
        ph=ph,
        moisture_pct=moisture,
        temperature_c=temp,
        ec_ds_m=ec,
        bulk_density=bd,
        coarse_fraction=coarse,
        latitudes=lats,
        longitudes=lons,
        depth_top_cm=d_top,
        depth_bottom_cm=d_bot,
        eis_frequencies=eis_freqs,
        eis_z_real=eis_real,
        eis_z_imag=eis_imag,
    )


def add_dataset_badge(
    fig: Figure,
    dataset_source: str,
    *,
    subtitle: str | None = None,
    lang: str | None = None,
) -> None:
    """Stamp provenance footer showing synthetic vs field/lab status and optional subtitle."""
    is_synth = str(dataset_source).lower() == "synthetic"
    badge_text = t("badge_synthetic" if is_synth else "badge_real", lang)
    badge_color = "#B7791F" if is_synth else "#276749"
    footer = f"{badge_text}   |   {subtitle}" if subtitle else badge_text
    fig.text(
        0.01,
        0.008,
        footer,
        fontsize=7.6,
        color=badge_color,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def save_figure(
    fig: Figure,
    plots_dir: Path,
    stem: str,
    *,
    config: VisualizationConfig,
    plot_id: str,
    title: str,
    category: str,
    description: str,
    is_3d: bool = False,
    dataset_source: str = "synthetic",
    subtitle: str | None = None,
) -> PlotArtifactRecord:
    """Apply provenance badge, export figure in requested format(s), close it, and return metadata."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    add_dataset_badge(fig, dataset_source, subtitle=subtitle, lang=config.lang)
    try:
        fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.96))
    except Exception:
        # 3D axes or complex gridspecs occasionally warn on tight_layout; proceed safely.
        fig.subplots_adjust(bottom=0.10, top=0.90, left=0.08, right=0.96)

    saved_paths: list[str] = []
    for fmt in config.formats:
        out_path = plots_dir / f"{stem}.{fmt}"
        fig.savefig(out_path, format=fmt, dpi=config.dpi, bbox_inches="tight", pad_inches=0.18)
        saved_paths.append(str(out_path))
    plt.close(fig)

    return PlotArtifactRecord(
        plot_id=plot_id,
        title=title,
        category=category,
        is_3d=is_3d,
        files=tuple(saved_paths),
        description=description,
    )


def render_notice_figure(
    plots_dir: Path,
    stem: str,
    *,
    config: VisualizationConfig,
    plot_id: str,
    title: str,
    category: str,
    notice_tr: str,
    dataset_source: str = "synthetic",
    is_3d: bool = False,
) -> PlotArtifactRecord:
    """Render a graceful explanatory panel when optional data (e.g., spatial coords or EIS) is absent."""
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    ax.axis("off")
    fig.suptitle(title, fontsize=12.5, fontweight="bold", color=PALETTE["navy"])
    ax.text(
        0.5,
        0.52,
        notice_tr,
        ha="center",
        va="center",
        fontsize=10.5,
        color=PALETTE["slate"],
        bbox={
            "boxstyle": "round,pad=0.8",
            "facecolor": "#FFFBEB",
            "edgecolor": "#D69E2E",
            "linewidth": 1.2,
        },
        wrap=True,
    )
    return save_figure(
        fig,
        plots_dir,
        stem,
        config=config,
        plot_id=plot_id,
        title=title,
        category=category,
        description=notice_tr,
        is_3d=is_3d,
        dataset_source=dataset_source,
    )


def render_table_figure(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    plots_dir: Path,
    stem: str,
    config: VisualizationConfig,
    plot_id: str,
    title: str,
    category: str,
    description: str,
    dataset_source: str = "synthetic",
    subtitle: str | None = None,
) -> PlotArtifactRecord:
    """Render a crisp publication-grade visual table PNG/SVG using matplotlib Table."""
    n_rows = max(1, len(rows))
    n_cols = max(1, len(headers))
    fig_w = max(11.0, min(18.0, 1.55 * n_cols))
    fig_h = max(3.6, min(14.0, 1.3 + 0.44 * (n_rows + 1)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    fig.suptitle(title, fontsize=12.5, fontweight="bold", color=PALETTE["navy"], y=0.97)

    safe_rows = [list(r) for r in rows] if rows else [[t("no_data", config.lang)] + [""] * (n_cols - 1)]
    col_chars = [
        max(
            [len(str(headers[c]))] + [len(str(r[c])) if c < len(r) else 0 for r in safe_rows] + [4]
        )
        for c in range(n_cols)
    ]
    raw_weights = [max(5.5, float(ch) + 2.2) for ch in col_chars]
    total_weight = float(sum(raw_weights)) or 1.0
    col_widths = [w / total_weight for w in raw_weights]

    tbl = ax.table(
        cellText=safe_rows,
        colLabels=list(headers),
        colWidths=col_widths,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    total_chars = sum(col_chars)
    font_sz = 6.8 if total_chars > 125 else (7.4 if (total_chars > 95 or n_cols > 9) else 8.2)
    tbl.set_fontsize(font_sz)
    tbl.scale(1.0, 1.45)

    for (r_idx, c_idx), cell in tbl.get_celld().items():
        cell.set_edgecolor("#CBD5E0")
        cell.set_linewidth(0.7)
        if r_idx == 0:
            cell.set_facecolor(PALETTE["navy"])
            cell.set_text_props(color="#FFFFFF", fontweight="bold")
        elif r_idx % 2 == 1:
            cell.set_facecolor("#FFFFFF")
        else:
            cell.set_facecolor("#F7FAFC")

    return save_figure(
        fig,
        plots_dir,
        stem,
        config=config,
        plot_id=plot_id,
        title=title,
        category=category,
        description=description,
        is_3d=False,
        dataset_source=dataset_source,
        subtitle=subtitle,
    )


def export_table_files(
    tables_dir: Path,
    stem: str,
    table_id: str,
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
) -> TableArtifactRecord:
    """Write synchronized CSV (UTF-8) and Markdown tables to disk."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / f"{stem}.csv"
    md_path = tables_dir / f"{stem}.md"

    str_headers = [str(h) for h in headers]
    str_rows = [[str(cell) for cell in row] for row in rows]

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(str_headers)
        writer.writerows(str_rows)

    md_lines = [
        f"# {title}",
        "",
        "| " + " | ".join(str_headers) + " |",
        "| " + " | ".join(["---"] * len(str_headers)) + " |",
    ]
    for r in str_rows:
        escaped = [c.replace("|", "\\|").replace("\n", " ") for c in r]
        md_lines.append("| " + " | ".join(escaped) + " |")
    md_lines.append("")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return TableArtifactRecord(
        table_id=table_id,
        title=title,
        csv_path=str(csv_path),
        markdown_path=str(md_path),
        n_rows=len(str_rows),
        n_cols=len(str_headers),
    )


def finite_target_mask(*arrays: np.ndarray) -> np.ndarray:
    """Return boolean mask of indices where all supplied 1D arrays are finite."""
    mask = np.ones(arrays[0].shape[0], dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return mask


def sanitize_label_map(items: Sequence[str], max_groups: int = 7) -> dict[str, str]:
    """Build deterministic color assignment map for categorical variables."""
    uniq = sorted(set(items))[:max_groups]
    return {
        label: CATEGORY_COLORS[i % len(CATEGORY_COLORS)]
        for i, label in enumerate(uniq)
    }


def extract_row_summary_stats(values: np.ndarray) -> Mapping[str, str]:
    """Format min, median, mean ± std, max for a numeric array ignoring NaNs."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"n": "0", "min": "-", "median": "-", "mean_sd": "-", "max": "-"}
    return {
        "n": str(int(finite.size)),
        "min": f"{float(np.min(finite)):.3f}",
        "median": f"{float(np.median(finite)):.3f}",
        "mean_sd": f"{float(np.mean(finite)):.3f} ± {float(np.std(finite)):.3f}",
        "max": f"{float(np.max(finite)):.3f}",
    }
