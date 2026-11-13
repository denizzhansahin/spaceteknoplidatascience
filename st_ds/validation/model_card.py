"""Eğitilen STDS paketleri için Türkçe Model Kartı (Model Card) üretimi ve şablon doğrulaması."""

from __future__ import annotations

from typing import Any, Mapping

from st_ds.domain.errors import SchemaError

REQUIRED_MODEL_CARD_SECTIONS: tuple[str, ...] = (
    "Model Kimliği",
    "Veri Seti ve Saha Kapsamı",
    "Değerlendirme Metrikleri",
    "Sentetik ve Gerçek Saha Ayrımı",
    "Kullanım Amacı ve Fail-Closed Kapılar",
    "Çalışma Ortamı",
)


def build_model_card_markdown(
    *,
    artifact_id: str,
    semantic_version: str,
    model_name: str,
    model_family: str,
    instrument_profile_id: str,
    dataset_source: str,
    n_train_samples: int,
    train_sites: list[str] | tuple[str, ...],
    holdout_sites: list[str] | tuple[str, ...],
    holdout_metrics: Mapping[str, Any],
    environment: Mapping[str, Any],
    bundle_sha256: str = "",
) -> str:
    """Eğitilmiş bir STDS artifact paketi için denetime hazır Türkçe Model Kartı üretir."""
    r2_val = holdout_metrics.get("r2")
    rpiq_val = holdout_metrics.get("rpiq")
    r2_str = f"{float(r2_val):.4f}" if isinstance(r2_val, (int, float)) else "Yok"
    rpiq_str = f"{float(rpiq_val):.4f}" if isinstance(rpiq_val, (int, float)) else "Yok"

    lines = [
        f"# STDS Model Kartı — `{artifact_id}`",
        "",
        "## Model Kimliği",
        f"- **Artifact Kimliği:** `{artifact_id}`",
        f"- **Semantik Sürüm:** `{semantic_version}`",
        f"- **Model Adı / Ailesi:** `{model_name}` (`{model_family}`)",
        f"- **Enstrüman Profili:** `{instrument_profile_id}`",
        f"- **Bundle SHA-256 Özeti:** `{bundle_sha256 or 'disa_aktarilmadi'}`",
        "",
        "## Veri Seti ve Saha Kapsamı",
        f"- **Veri Kaynağı:** `{dataset_source}`",
        f"- **Eğitim Örnek Sayısı:** `{n_train_samples}`",
        f"- **Kalibrasyon Sahaları (Görülen):** `{', '.join(train_sites)}`",
        f"- **Holdout Sahaları (Hiç Görülmemiş):** `{', '.join(holdout_sites)}`",
        "",
        "## Değerlendirme Metrikleri",
        "| Metrik | Değer |",
        "| --- | --- |",
        f"| `n_samples` | `{holdout_metrics.get('n_samples')}` |",
        f"| `rmse` | `{float(holdout_metrics.get('rmse', 0.0)):.4f}` |",
        f"| `mae` | `{float(holdout_metrics.get('mae', 0.0)):.4f}` |",
        f"| `bias` | `{float(holdout_metrics.get('bias', 0.0)):+.4f}` |",
        f"| `r2` | `{r2_str}` |",
        f"| `rpiq` | `{rpiq_str}` |",
        f"| `bland_altman_95_loa` | `[{float(holdout_metrics.get('bland_altman_loa_lower', 0.0)):.4f}, {float(holdout_metrics.get('bland_altman_loa_upper', 0.0)):.4f}]` |",
        "",
        "## Sentetik ve Gerçek Saha Ayrımı",
        "- Sentetik ölçütler **yalnızca** matematiksel sözleşme, veri sızıntısız çapraz doğrulama, "
        "kalibrasyon, belirsizlik, XAI ve artifact güvenlik kapılarını doğrular.",
        "- Sentetik sonuçlar hiçbir koşulda gerçek laboratuvar veya tarla doğruluğu olarak raporlanamaz (`Risk 9`).",
        "",
        "## Kullanım Amacı ve Fail-Closed Kapılar",
        "- Operasyonel raporlama için `QualityStatus.OK` (veya incelenmiş `WARNING`) durumu gereklidir.",
        "- Eksik hacim ağırlığında (`bulk_density`) `stock_mg_ha = None` ve `QualityFlag.MISSING_BULK_DENSITY` döner (`DEF-005`).",
        "- Kalibrasyon alanı dışındaki girdiler `QualityStatus.OUT_OF_DOMAIN` (`OOD_HIGH_RISK`) bayrağını tetikler.",
        "- `model_weights.safetensors`, `preprocess_state.npz` veya `manifest.json` üzerindeki her byte değişikliği `ArtifactCompatibilityError` ile reddedilir (`DEF-006`).",
        "",
        "## Çalışma Ortamı",
        f"- **Platform:** `{environment.get('platform', 'bilinmiyor')}` (`{environment.get('machine', 'bilinmiyor')}`)",
        f"- **Python:** `{environment.get('python_version', 'bilinmiyor')}`",
        f"- **Bağımlılıklar:** `{environment.get('dependencies', {})}`",
        "",
    ]
    return "\n".join(lines)


def validate_model_card_sections(markdown_text: str) -> list[str]:
    """Zorunlu Türkçe Model Kartı başlıklarının eksiksiz olduğunu doğrular; eksik başlıkları döner."""
    if not isinstance(markdown_text, str) or not markdown_text.strip():
        raise SchemaError("Model Kartı Markdown metni boş olmayan bir dize olmalıdır.")
    missing = [
        sec for sec in REQUIRED_MODEL_CARD_SECTIONS if f"## {sec}" not in markdown_text
    ]
    return missing
