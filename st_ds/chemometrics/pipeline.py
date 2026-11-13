"""Versioned Chemometric Preprocessing Pipeline enforcing fit/transform space identity (ADR-003).

Guarantees that absorbance, SNV, Savitzky-Golay derivative, and EPO/GLSW suppression
always execute in the exact same preprocessing space and on the exact same wavelength grid.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from st_ds.chemometrics.epo import EPO, EPOConfig, GLSW
from st_ds.chemometrics.savgol import savgol_derivative
from st_ds.chemometrics.transforms import snv_transform, to_absorbance, validate_spectral_grid
from st_ds.domain.errors import CalibrationError, SchemaError
from st_ds.domain.schemas import Spectrum
from st_ds.io.dataset import SoilDataset


@dataclass(frozen=True)
class PreprocessConfig:
    """Versioned configuration of the chemometric preprocessing graph (Section 11 & 32)."""

    pipeline_version: str = "chem_v1"
    use_absorbance: bool = True
    use_snv: bool = True
    use_savgol: bool = True
    sg_window: int = 7
    sg_polyorder: int = 2
    sg_deriv: int = 1
    suppression_mode: Literal["none", "epo", "glsw"] = "epo"
    epo_q: int | None = None
    epo_max_q: int = 6
    glsw_alpha: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ChemometricPipeline:
    """Stateful, versioned preprocessing pipeline separating calibration fit from inference transform."""

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()
        self.wavelength_grid_nm: tuple[float, ...] | None = None
        self.space_signature: str | None = None
        self.epo: EPO | None = None
        self.glsw: GLSW | None = None
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def compute_space_signature(self, wavelengths: np.ndarray | tuple[float, ...]) -> str:
        """Compute deterministic SHA-256 signature of deterministic transforms + wavelength grid."""
        wl = validate_spectral_grid(wavelengths)
        payload = {
            "pipeline_version": self.config.pipeline_version,
            "use_absorbance": self.config.use_absorbance,
            "use_snv": self.config.use_snv,
            "use_savgol": self.config.use_savgol,
            "sg_window": self.config.sg_window if self.config.use_savgol else None,
            "sg_polyorder": self.config.sg_polyorder if self.config.use_savgol else None,
            "sg_deriv": self.config.sg_deriv if self.config.use_savgol else None,
            "wavelength_grid_nm": [round(float(w), 6) for w in wl],
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return f"{self.config.pipeline_version}:{digest[:16]}"

    def apply_deterministic_transforms(
        self,
        X: np.ndarray,
        wavelengths: np.ndarray | tuple[float, ...],
    ) -> np.ndarray:
        """Apply row-wise deterministic preprocessing steps (Absorbance -> SNV -> SavGol)."""
        wl = validate_spectral_grid(wavelengths)
        out = np.asarray(X, dtype=float)
        if out.shape[-1] != wl.size:
            raise SchemaError(
                f"Spectral band count ({out.shape[-1]}) does not match wavelength grid ({wl.size})."
            )
        if self.config.use_absorbance:
            out = to_absorbance(out)
        if self.config.use_snv:
            out = snv_transform(out)
        if self.config.use_savgol:
            out = savgol_derivative(
                out,
                wl,
                window=self.config.sg_window,
                polyorder=self.config.sg_polyorder,
                deriv=self.config.sg_deriv,
            )
        return out

    def fit(
        self,
        wavelengths: np.ndarray | tuple[float, ...],
        *,
        X_wet: np.ndarray | None = None,
        X_dry: np.ndarray | None = None,
        calibration_dataset_hash: str = "untracked",
    ) -> ChemometricPipeline:
        """Fit calibration-dependent steps (EPO or GLSW) inside the exact deterministic transform space."""
        wl = validate_spectral_grid(wavelengths)
        sig = self.compute_space_signature(wl)
        self.wavelength_grid_nm = tuple(float(w) for w in wl)
        self.space_signature = sig

        if self.config.suppression_mode == "none":
            self.epo = None
            self.glsw = None
            self._is_fitted = True
            return self

        if X_wet is None or X_dry is None:
            raise CalibrationError(
                f"ChemometricPipeline with suppression_mode='{self.config.suppression_mode}' "
                "requires paired X_wet and X_dry calibration matrices during fit."
            )

        # CRITICAL FIX (DEF-001, ADR-003): Transform wet/dry into the EXACT same space before fitting!
        Z_wet = self.apply_deterministic_transforms(X_wet, wl)
        Z_dry = self.apply_deterministic_transforms(X_dry, wl)

        if self.config.suppression_mode == "epo":
            epo = EPO(EPOConfig(q=self.config.epo_q, max_q=self.config.epo_max_q))
            epo.fit(
                Z_wet,
                Z_dry,
                space_signature=sig,
                q=self.config.epo_q,
                calibration_dataset_hash=calibration_dataset_hash,
            )
            self.epo = epo
            self.glsw = None
        elif self.config.suppression_mode == "glsw":
            glsw = GLSW(alpha=self.config.glsw_alpha)
            glsw.fit(Z_wet, Z_dry, space_signature=sig, alpha=self.config.glsw_alpha)
            self.glsw = glsw
            self.epo = None
        else:
            raise SchemaError(f"Unsupported suppression_mode '{self.config.suppression_mode}'.")

        self._is_fitted = True
        return self

    def fit_from_datasets(
        self,
        wet_dataset: SoilDataset,
        dry_dataset: SoilDataset,
    ) -> ChemometricPipeline:
        """Fit pipeline from matched paired wet and dry SoilDataset objects."""
        if len(wet_dataset) != len(dry_dataset):
            raise CalibrationError(
                f"Paired wet ({len(wet_dataset)}) and dry ({len(dry_dataset)}) datasets must have equal length."
            )
        wet_by_id = {r.sample_id: r for r in wet_dataset.records}
        dry_by_id = {r.sample_id: r for r in dry_dataset.records}
        if set(wet_by_id.keys()) != set(dry_by_id.keys()):
            raise CalibrationError("Paired wet and dry datasets must contain identical sample_ids.")

        ordered_ids = sorted(wet_by_id.keys())
        wl = wet_by_id[ordered_ids[0]].spectrum.wavelength_grid_nm
        X_wet = np.vstack([wet_by_id[sid].spectrum.values for sid in ordered_ids])
        X_dry = np.vstack([dry_by_id[sid].spectrum.values for sid in ordered_ids])
        cal_hash = f"{wet_dataset.compute_sha256()[:12]}_{dry_dataset.compute_sha256()[:12]}"
        return self.fit(wl, X_wet=X_wet, X_dry=X_dry, calibration_dataset_hash=cal_hash)

    def transform_array(
        self,
        X: np.ndarray,
        wavelengths: np.ndarray | tuple[float, ...],
    ) -> np.ndarray:
        """Apply deterministic transforms followed by fitted EPO/GLSW suppression."""
        if not self._is_fitted or self.space_signature is None or self.wavelength_grid_nm is None:
            raise CalibrationError(
                "ChemometricPipeline must be fitted before calling transform (Section 32)."
            )
        wl = validate_spectral_grid(wavelengths)
        if wl.size != len(self.wavelength_grid_nm) or not np.allclose(
            wl, self.wavelength_grid_nm, atol=1e-3, rtol=0.0
        ):
            raise CalibrationError(
                "Input wavelength grid does not match fitted ChemometricPipeline wavelength grid."
            )

        sig = self.compute_space_signature(wl)
        Z = self.apply_deterministic_transforms(X, wl)

        if self.config.suppression_mode == "epo":
            if self.epo is None:
                raise CalibrationError("EPO state is missing in fitted ChemometricPipeline.")
            return self.epo.transform(Z, space_signature=sig)
        if self.config.suppression_mode == "glsw":
            if self.glsw is None:
                raise CalibrationError("GLSW state is missing in fitted ChemometricPipeline.")
            return self.glsw.transform(Z, space_signature=sig)
        return Z

    def transform_spectrum(self, spectrum: Spectrum) -> np.ndarray:
        """Transform a single Spectrum object into a clean 1D/2D feature array."""
        return self.transform_array(spectrum.values, spectrum.wavelength_grid_nm)

    def transform_dataset(self, dataset: SoilDataset) -> np.ndarray:
        """Transform all 1D spectra in a SoilDataset into a clean 2D matrix (N, P)."""
        X = np.vstack([r.spectrum.values for r in dataset.records])
        wl = dataset.records[0].spectrum.wavelength_grid_nm
        return self.transform_array(X, wl)
