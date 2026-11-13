"""Integration tests for `st_ds.visualization` (26+ 2D/3D plots, 5 tables, HTML dashboard, CLI)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from st_ds.cli.main import main as cli_main
from st_ds.domain.schemas import SpatialCoordinates
from st_ds.io.dataset import SoilDataset, save_dataset_json
from st_ds.io.synthetic import SyntheticConfig, SyntheticGeneratorV2
from st_ds.visualization import VisualizationConfig, generate_full_visualization_suite


def _build_spatial_synthetic_dataset(*, n_samples: int = 36, n_sites: int = 12, seed: int = 17) -> SoilDataset:
    """Create a synthetic SoilDataset guaranteed to have >= 10 distinct coordinates for Kriging."""
    gen = SyntheticGeneratorV2(
        SyntheticConfig(
            dataset_id="viz_spatial_synth_v1",
            profile_id="visnir_24ch_v1",
            n_samples=n_samples,
            n_sites=n_sites,
            seed=seed,
        )
    )
    ds = gen.generate_dataset()
    enriched = []
    for idx, rec in enumerate(ds.records):
        # Assign distinct coordinates across the Ankara basin test bbox
        lon = 32.78 + 0.015 * (idx % 6) + 0.002 * (idx // 6)
        lat = 39.88 + 0.012 * (idx // 6)
        enriched.append(
            replace(
                rec,
                coordinates=SpatialCoordinates(lat=lat, lon=lon, crs_epsg="EPSG:4326"),
            )
        )
    return SoilDataset(
        dataset_id=ds.dataset_id,
        records=enriched,
        dataset_source=ds.dataset_source,
    )


def test_full_visualization_suite_generates_26_plus_plots_4_3d_5_tables_and_dashboard(
    tmp_path: Path,
) -> None:
    ds = _build_spatial_synthetic_dataset(n_samples=36, n_sites=12, seed=17)
    wet_ds, dry_ds = SyntheticGeneratorV2(
        SyntheticConfig(dataset_id="viz_pairs", n_samples=16, n_sites=4, seed=19)
    ).generate_paired_wet_dry_dataset(n_pairs=10)

    out_dir = tmp_path / "viz_full"
    cfg = VisualizationConfig(
        target_name="soc_pct",
        n_folds=3,
        formats=("png", "svg"),
        dpi=96,
        include_html_dashboard=True,
        include_interactive_3d=True,
        seed=17,
    )
    report = generate_full_visualization_suite(
        ds,
        out_dir,
        wet_dataset=wet_ds,
        dry_dataset=dry_ds,
        config=cfg,
    )

    assert report.n_plots >= 26
    assert report.n_3d_plots == 4
    assert report.n_tables == 5

    # Verify manifest JSON exists and matches counts
    manifest_file = Path(report.manifest_path)
    assert manifest_file.is_file()
    manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest_data["n_plots"] == report.n_plots
    assert manifest_data["n_3d_plots"] == 4
    assert manifest_data["n_tables"] == 5

    # Verify every PNG & SVG plot artifact exists and is non-empty
    for plot_rec in report.plots:
        assert len(plot_rec.files) == 2  # png and svg
        for file_str in plot_rec.files:
            fp = Path(file_str)
            assert fp.is_file(), f"Missing plot output: {fp}"
            assert fp.stat().st_size > 1000, f"Plot output unexpectedly small: {fp}"

    # Verify all 5 CSV + Markdown tables exist
    for tbl_rec in report.tables:
        assert Path(tbl_rec.csv_path).is_file()
        assert Path(tbl_rec.markdown_path).is_file()
        assert Path(tbl_rec.csv_path).stat().st_size > 20
        assert Path(tbl_rec.markdown_path).stat().st_size > 20

    # Verify offline Turkish HTML dashboard exists and references expected banners & plots
    assert report.dashboard_path is not None
    dash_html = Path(report.dashboard_path).read_text(encoding="utf-8")
    assert "SENTETİK VERİ DOĞRULAMA ÇIKTISI" in dash_html
    assert "04_spektral_3d_yuzey.png" in dash_html
    assert "23_ordinary_kriging_2d_ve_3d.png" in dash_html
    assert "26_kovaryat_3d_iliski_yuzeyi.png" in dash_html


def test_visualization_suite_graceful_fallback_without_coords_or_bulk_density(
    tmp_path: Path,
) -> None:
    gen = SyntheticGeneratorV2(
        SyntheticConfig(
            dataset_id="viz_minimal_v1",
            n_samples=6,
            n_sites=1,
            seed=9,
        )
    )
    raw = gen.generate_dataset()
    recs = [
        replace(
            r,
            coordinates=None,
            covariates=replace(r.covariates, bulk_density_g_cm3=None),
        )
        for r in raw.records
    ]
    minimal_ds = SoilDataset(dataset_id="viz_minimal_v1", records=recs, dataset_source="synthetic")

    out_dir = tmp_path / "viz_minimal"
    report = generate_full_visualization_suite(
        minimal_ds,
        out_dir,
        config=VisualizationConfig(dpi=85, include_interactive_3d=False),
    )

    assert report.n_plots == 26
    assert report.n_3d_plots == 4
    assert report.n_tables == 5
    for p in report.plots:
        for f in p.files:
            assert Path(f).is_file() and Path(f).stat().st_size > 500


def test_cli_visualize_subcommand_runs_cleanly(tmp_path: Path, capsys) -> None:
    ds = _build_spatial_synthetic_dataset(n_samples=24, n_sites=6, seed=21)
    in_path = tmp_path / "cli_viz_input.json"
    save_dataset_json(ds, in_path)
    out_dir = tmp_path / "cli_viz_out"

    rc = cli_main(
        [
            "visualize",
            "--input",
            str(in_path),
            "--output",
            str(out_dir),
            "--folds",
            "3",
            "--dpi",
            "90",
            "--no-interactive-3d",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "visualize"
    assert payload["n_plots"] >= 26
    assert payload["n_3d_plots"] == 4
    assert payload["n_tables"] == 5
    assert Path(payload["dashboard_path"]).is_file()


def test_default_synthetic_has_2d_coordinates_for_kriging(tmp_path: Path) -> None:
    from st_ds.visualization.base import extract_dataset_arrays
    from st_ds.visualization.science_plots import compute_spatial_kriging_bundle

    gen = SyntheticGeneratorV2(SyntheticConfig(dataset_id="krig_2d_v1", n_samples=40, n_sites=6, seed=42))
    ds = gen.generate_dataset()
    wet_ds, dry_ds = gen.generate_paired_wet_dry_dataset(n_pairs=12)

    coords = {(round(r.coordinates.lon, 6), round(r.coordinates.lat, 6)) for r in ds.records if r.coordinates}
    assert len(coords) >= 8
    lons = {c[0] for c in coords}
    lats = {c[1] for c in coords}
    assert len(lons) >= 3
    assert len(lats) >= 3

    arrays = extract_dataset_arrays(ds)
    bundle = compute_spatial_kriging_bundle(arrays, arrays.soc, grid_size=16)
    assert bundle is not None
    assert bundle.grid_x_m.shape == (16, 16)
    assert bundle.kriging is not None

    out_dir = tmp_path / "viz_2d_krig"
    report = generate_full_visualization_suite(
        ds,
        out_dir,
        wet_dataset=wet_ds,
        dry_dataset=dry_ds,
        config=VisualizationConfig(lang="en", dpi=90, include_interactive_3d=False),
    )
    audit_csv = (out_dir / "tables" / "05_tahmin_ve_stok_denetim_raporu.csv").read_text(encoding="utf-8")
    assert "warning [synthetic]" in audit_csv
    assert "N/A (soc-only)" not in audit_csv
    assert "- / -" not in audit_csv
    assert report.n_plots == 26

