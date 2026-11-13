# STDS (`st_ds`) — Yorumlu Çalıştırma Örnekleri (`CLI` & `Python API`)

Bu klasör, `scripts/` altındaki otomatik bash hatlarından **bağımsız** olarak hem tüm **`st-ds` CLI komutlarını & bayraklarını** hem de tüm **Python API sınıflarını, parametrelerini ve sıfırdan CSV/toprak-dışı veri hazırlamayı** satır satır yorumlarla öğrenip doğrudan çalıştırabilmeniz için hazırlanmıştır.

## Dosya İndeksi

| Dosya | Tür | İçerik | Nasıl Çalıştırılır? |
| --- | --- | --- | --- |
| **[synthetic_python_quickstart.py](synthetic_python_quickstart.py)** | Python | Hızlı uçtan uca sentetik eğitim, `.safetensors` paketleme, `STSoilAnalyzer`, stok & 26 grafik üretimi. | `PYTHONPATH=. /opt/anaconda3/bin/python examples/synthetic_python_quickstart.py` |
| **[cli_workflow.sh](cli_workflow.sh)** | CLI | Temel 8 adımlı `st-ds` komut zinciri (doğrulama → EPO → CV → eğitim → tahmin → PDF → görselleştirme). | `bash examples/cli_workflow.sh` |
| **[01_all_cli_commands_and_params.sh](01_all_cli_commands_and_params.sh)** | CLI (Parametre Kataloğu) | Tüm `8` CLI alt komutunu farklı **`--target`** (`soc_pct`/`tc_pct`/`sic_pct`), **`--model`** (`auto`/`plsr`/`random_forest`/`hybrid_net`), **`--suppression-mode`** (`epo`/`glsw`/`none`), `--include-xai`, `--no-stock`, `--format`, `--dpi`, `--lang` seçenekleriyle ayrıntılı açıklamalı çalıştırır. | `bash examples/01_all_cli_commands_and_params.sh` |
| **[02_all_python_api_parameters.py](02_all_python_api_parameters.py)** | Python (Parametre Kataloğu) | `SyntheticConfig`, `PreprocessConfig`, `FeatureFusionConfig`, 5 model sınıfı (`PLSR`, `KernelRidge`, `RandomForest`, `LocalPLSR`, `HybridNet`), `TrainerConfig`, `ConformalConfig`, `OODConfig`, `STSoilAnalyzer` ve `VisualizationConfig` parametrelerinin tamamını yorumlu olarak çalıştırır. | `PYTHONPATH=. /opt/anaconda3/bin/python examples/02_all_python_api_parameters.py` |
| **[03_custom_csv_and_non_soil_domains.py](03_custom_csv_and_non_soil_domains.py)** | Python + CLI | **Sıfırdan kendi ham CSV'nizi oluşturma** (eğitim CSV + hedefsiz tahmin CSV) ve **toprak dışı alanlar** (Gıda Proteini / Hava Kalitesi `pm25_ugm3`) için özel `InstrumentProfile` JSON + `--target <ozel_hedef>` çalıştırma örneği. | `PYTHONPATH=. /opt/anaconda3/bin/python examples/03_custom_csv_and_non_soil_domains.py` |

## İlgili Dokümantasyon
- [Veri Seti Hazırlama & Şema Kuralları (TR & EN)](../docs/DATA_PREPARATION_AND_SCHEMA_GUIDE.md)
- [Tüm CLI & Python Parametreleri Referansı (TR & EN)](../docs/CLI_AND_PYTHON_PARAMETERS_GUIDE.md)
- [Toprak Dışı & Genel Veri Analizi Rehberi (TR & EN)](../docs/GENERAL_DATA_ANALYSIS_GUIDE.md)
