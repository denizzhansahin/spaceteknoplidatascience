"""Bilingual localization helpers (`tr` / `en`) for STDS plots, tables, HTML, PDF, and CLI.

Language selection follows `.env` / runtime precedence:
    explicit `lang` > `os.environ['STDS_LANG']` > optional `.env` > `'tr'`
"""

from __future__ import annotations

from typing import Any

from st_ds.config import LangCode, normalize_lang, resolve_lang

MESSAGES: dict[str, dict[LangCode, str]] = {
    # Provenance & Risk 9 banners
    "badge_synthetic": {
        "tr": "VERİ KAYNAĞI: SENTETİK (Sadece doğrulama amaçlıdır; saha kalibrasyonu yerine geçmez)",
        "en": "DATA SOURCE: SYNTHETIC (Pipeline verification only; NOT field calibration)",
    },
    "badge_real": {
        "tr": "VERİ KAYNAĞI: GERÇEK SAHA / LABORATUVAR ÖLÇÜMÜ",
        "en": "DATA SOURCE: REAL FIELD / LABORATORY MEASUREMENT",
    },
    "dash_banner_synth": {
        "tr": "SENTETİK VERİ DOĞRULAMA ÇIKTISI — Yalnızca yazılım ve matematik doğrulama içindir; saha kalibrasyonu yerine geçmez!",
        "en": "SYNTHETIC DATA VERIFICATION OUTPUT — For software and mathematical verification only; NOT field calibration performance!",
    },
    "dash_banner_real": {
        "tr": "GERÇEK SAHA / LABORATUVAR ANALİZ ÇIKTISI",
        "en": "REAL FIELD / LABORATORY ANALYSIS OUTPUT",
    },
    # Categories
    "cat_spectral": {
        "tr": "Spektrum, Kemometri & Sensör Kalitesi",
        "en": "Spectra, Chemometrics & Sensor Quality",
    },
    "cat_model": {
        "tr": "Model Kıyaslama, Belirsizlik & OOD",
        "en": "Model Benchmark, Uncertainty & OOD",
    },
    "cat_science": {
        "tr": "XAI Açıklanabilirlik, Karbon Stoku & Mekânsal Kriging",
        "en": "XAI Explainability, Carbon Stock & Spatial Kriging",
    },
    "cat_tables_3d": {
        "tr": "Tablolar, Çok Değişkenli & Pano",
        "en": "Tables, Multivariate & Dashboard",
    },
    # Common labels
    "no_data": {"tr": "Veri yok", "en": "No data"},
    "missing_bd": {"tr": "Eksik HA", "en": "Missing BD"},
    "error": {"tr": "HATA", "en": "ERROR"},
    "plot_3d_badge": {"tr": "3D GRAFİK", "en": "3D PLOT"},
    "rows_unit": {"tr": "satır", "en": "rows"},
    "download_csv": {"tr": "CSV İndir", "en": "Download CSV"},
    "view_markdown": {"tr": "Markdown Görüntüle", "en": "View Markdown"},
    "open_svg": {"tr": "SVG Grafiği Aç", "en": "Open SVG Plot"},
    # HTML dashboard & PDF headings
    "dash_title": {
        "tr": "Toprak Karbon Analizi ve Bilimsel Görselleştirme Panosu",
        "en": "Soil Carbon Analytics & Scientific Visualization Dashboard",
    },
    "dash_header_h1": {
        "tr": "STDS v{version} — Toprak Karbon Analizi ve Bilimsel Görselleştirme Raporu",
        "en": "STDS v{version} — Soil Carbon Analytics & Scientific Visualization Report",
    },
    "dash_subtitle": {
        "tr": "Veri Seti: <strong>{dataset_id}</strong> | Hedef: <strong>{target}</strong> | Dil: <strong>{lang_code}</strong> | Oluşturulma: {now}",
        "en": "Dataset: <strong>{dataset_id}</strong> | Target: <strong>{target}</strong> | Language: <strong>{lang_code}</strong> | Generated: {now}",
    },
    "pdf_download_btn": {
        "tr": "Tam Kapsamlı PDF Raporunu Aç",
        "en": "Open Complete PDF Report",
    },
    "interactive_3d_h2": {
        "tr": "Etkileşimli 3D Plotly Grafikleri",
        "en": "Interactive 3D Plotly Visualizations",
    },
    "tables_h2": {
        "tr": "Analitik CSV ve Markdown Tabloları",
        "en": "Analytical CSV and Markdown Tables",
    },
    "dash_footer": {
        "tr": "Artefakt ID: <code>{artifact_id}</code> | SemVer: <code>{semver}</code> | SHA-256: <code>{sha}...</code> | Bu pano yerel disk üzerinden çevrimdışı (ADR-001 uyumlu) görüntülenmek üzere üretilmiştir.",
        "en": "Artifact ID: <code>{artifact_id}</code> | SemVer: <code>{semver}</code> | SHA-256: <code>{sha}...</code> | Generated for offline local inspection (ADR-001 compliant).",
    },
    # PDF Report structural strings
    "pdf_report_filename": {
        "tr": "STDS_Analiz_Raporu.pdf",
        "en": "STDS_Analysis_Report.pdf",
    },
    "pdf_cover_title": {
        "tr": "STDS — Kapsamlı Toprak Karbon Analizi ve Bilimsel Rapor",
        "en": "STDS — Comprehensive Soil Carbon Analytics & Scientific Report",
    },
    "pdf_cover_sub": {
        "tr": "Çok Sensörlü Spektroskopi • Kemometri • Model Çıkarımı • Belirsizlik • XAI • Karbon Stoku & Jeoistatistik",
        "en": "Multi-Sensor Spectroscopy • Chemometrics • Model Inference • Uncertainty • XAI • Carbon Stock & Geostatistics",
    },
    "pdf_sec_lifecycle": {
        "tr": "1. Veri Yaşam Döngüsü: Ne Geldi, Nasıl İşlendi, Ne Çıktı & İlişkiler",
        "en": "1. Data Lifecycle: Inputs Received, Processing Applied, Outputs & Relationships",
    },
    "pdf_sec_tables": {
        "tr": "2. Sayısal Denetim, Kıyaslama ve Örnek Bazlı Özet Tabloları",
        "en": "2. Quantitative Audit, Benchmark & Sample-Level Summary Tables",
    },
    "pdf_sec_plots": {
        "tr": "3. Bilimsel Görselleştirme Galerisi (Tüm Üretilen 2D & 3D PNG Grafikler)",
        "en": "3. Scientific Visualization Gallery (All Generated 2D & 3D PNG Figures)",
    },
}


def pick_text(lang: str | None, *, tr: str, en: str) -> str:
    """Return Turkish or English string according to resolved effective language."""
    eff = resolve_lang(lang)
    return en if eff == "en" else tr


def t(key: str, lang: str | None = None, **kwargs: Any) -> str:
    """Lookup localized message by key and optional `.format(**kwargs)` substitutions."""
    eff: LangCode = resolve_lang(lang) if lang is None else normalize_lang(lang)
    entry = MESSAGES.get(key)
    if entry is None:
        return key
    template = entry.get(eff) or entry["tr"]
    return template.format(**kwargs) if kwargs else template
