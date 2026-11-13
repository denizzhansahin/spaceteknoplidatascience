"""Reference laboratory label store (Section 8, 19, 31).

Stores verified laboratory ground-truth labels (SOC, SIC, TC, method provenance,
uncertainty, soil class) keyed by `sample_id` and safely attaches them to spectral
measurements without mixing synthetic and real laboratory labels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import pandas as pd

from st_ds.domain.errors import SchemaError
from st_ds.domain.schemas import Measurement, SoilReference


@dataclass
class ReferenceLabelStore:
    """In-memory and file-backed store of laboratory reference carbon labels."""

    store_id: str
    labels: dict[str, SoilReference]

    def __post_init__(self) -> None:
        if not self.store_id or not self.store_id.strip():
            raise SchemaError("ReferenceLabelStore.store_id cannot be empty.")
        for sample_id, ref in self.labels.items():
            if not sample_id or not sample_id.strip():
                raise SchemaError("ReferenceLabelStore sample_id keys cannot be empty.")
            if not isinstance(ref, SoilReference):
                raise SchemaError(f"Invalid SoilReference value for sample_id '{sample_id}'.")

    def __len__(self) -> int:
        return len(self.labels)

    def add_label(self, sample_id: str, reference: SoilReference, *, overwrite: bool = False) -> None:
        if not sample_id or not sample_id.strip():
            raise SchemaError("sample_id cannot be empty.")
        if not overwrite and sample_id in self.labels:
            raise SchemaError(
                f"Duplicate sample_id '{sample_id}' in ReferenceLabelStore '{self.store_id}'.",
                details={"sample_id": sample_id},
            )
        self.labels[sample_id] = reference

    def get(self, sample_id: str) -> SoilReference | None:
        return self.labels.get(sample_id)

    def attach_to_measurements(
        self,
        measurements: Iterable[Measurement],
        *,
        strict: bool = True,
    ) -> list[Measurement]:
        """Return new Measurement instances with laboratory reference labels attached by sample_id."""
        attached: list[Measurement] = []
        for m in measurements:
            if m.dataset_source == "synthetic":
                raise SchemaError(
                    f"Refusing to attach real laboratory reference labels to synthetic measurement '{m.sample_id}'.",
                    details={"sample_id": m.sample_id},
                )
            ref = self.labels.get(m.sample_id)
            if ref is None:
                if strict:
                    raise SchemaError(
                        f"Missing reference label for sample_id '{m.sample_id}' in store '{self.store_id}'.",
                        details={"sample_id": m.sample_id},
                    )
                attached.append(m)
            else:
                attached.append(replace(m, reference=ref))
        return attached

    def save_csv(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for sid, ref in sorted(self.labels.items()):
            d = ref.to_dict()
            d["sample_id"] = sid
            rows.append(d)
        pd.DataFrame(rows).to_csv(target, index=False)
        return target

    @classmethod
    def load_csv(cls, path: str | Path, *, store_id: str | None = None) -> ReferenceLabelStore:
        csv_path = Path(path)
        df = pd.read_csv(csv_path)
        if "sample_id" not in df.columns:
            raise SchemaError("Reference label CSV must contain a 'sample_id' column.")

        def _opt_float(val: object) -> float | None:
            if val is None or pd.isna(val):  # type: ignore[arg-type]
                return None
            return float(val)  # type: ignore[arg-type]

        def _opt_str(val: object) -> str | None:
            if val is None or pd.isna(val):  # type: ignore[arg-type]
                return None
            return str(val)

        labels: dict[str, SoilReference] = {}
        for _, row in df.iterrows():
            sid = str(row["sample_id"]).strip()
            if sid in labels:
                raise SchemaError(f"Duplicate sample_id '{sid}' in reference CSV '{csv_path}'.")
            unit = _opt_str(row.get("unit")) or "pct"
            derive = bool(row.get("is_soc_derived_from_difference", False))
            cls_id_raw = row.get("soil_class_id")
            cls_id = int(cls_id_raw) if cls_id_raw is not None and not pd.isna(cls_id_raw) else None

            ref = SoilReference.from_units(
                soc=_opt_float(row.get("soc_pct")),
                sic=_opt_float(row.get("sic_pct")),
                tc=_opt_float(row.get("tc_pct")),
                unit=unit,
                dry_matter_basis=bool(row.get("dry_matter_basis", True)),
                soc_method=_opt_str(row.get("soc_method")) or "dry_combustion_acid_pretreated_iso10694",
                sic_method=_opt_str(row.get("sic_method")) or "calcimeter_iso10693",
                tc_method=_opt_str(row.get("tc_method")) or "dry_combustion_dumas_iso10694",
                derive_soc_from_difference=derive,
                soil_class_id=cls_id,
                soil_class_name=_opt_str(row.get("soil_class_name")),
            )
            labels[sid] = ref
        return cls(store_id=store_id or csv_path.stem, labels=labels)

    def save_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "store_id": self.store_id,
            "labels": {sid: ref.to_dict() for sid, ref in sorted(self.labels.items())},
        }
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return target
