"""Faz 4 Unit & Mathematical Invariant Tests:
Absorbance, SNV, Wavelength-Aware Savitzky-Golay (DEF-002),
LIBS ARPLS & Pseudo-Voigt, EPO & GLSW (DEF-001, ADR-003).
"""

from __future__ import annotations

import numpy as np
import pytest

from st_ds.chemometrics import (
    EPO,
    GLSW,
    ChemometricPipeline,
    EPOConfig,
    PreprocessConfig,
    savgol_derivative,
    snv_transform,
    to_absorbance,
    validate_spectral_grid,
)
from st_ds.domain import (
    CalibrationError,
    SchemaError,
    SpectralQualityError,
    SpectralSignalMode,
    Spectrum,
    get_builtin_profile,
)
from st_ds.io import SyntheticConfig, SyntheticGeneratorV2
from st_ds.signal import (
    correct_baseline,
    estimate_arpls_baseline,
    extract_libs_peak_features,
    fit_pseudovoigt_peaks,
    pseudovoigt_profile,
)


class TestAbsorbanceAndSNV:
    """Verify reflectance-to-absorbance and row-wise deterministic SNV transforms."""

    def test_to_absorbance_and_negative_reflectance_rejection(self) -> None:
        refl = np.array([1.0, 0.1, 0.01])
        abs_vals = to_absorbance(refl)
        np.testing.assert_allclose(abs_vals, [0.0, 1.0, 2.0], atol=1e-12)

        with pytest.raises(SpectralQualityError, match="strictly positive reflectance"):
            to_absorbance(np.array([0.2, 0.0, 0.5]))

        with pytest.raises(SpectralQualityError, match="strictly positive reflectance"):
            to_absorbance(np.array([0.2, -0.05, 0.5]))

    def test_snv_transform_zero_mean_unit_std(self) -> None:
        x = np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])
        z = snv_transform(x)
        np.testing.assert_allclose(np.mean(z, axis=1), [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(np.std(z, axis=1), [1.0, 1.0], atol=1e-10)
        # Multiplicative scaling invariance
        np.testing.assert_allclose(z[0], z[1], atol=1e-10)


class TestWavelengthAwareSavitzkyGolay:
    """Verify DEF-002 fix: physical d^k p(lambda)/d(lambda)^k on uniform & non-uniform grids."""

    def test_exact_analytic_derivative_on_uniform_and_nonuniform_grids(self) -> None:
        # Polynomial A(lambda) = 0.002 * lambda^2 - 0.5 * lambda + 3.0
        # Exact dA/dlambda = 0.004 * lambda - 0.5
        # Exact d^2A/dlambda^2 = 0.004
        a, b, c = 0.002, -0.5, 3.0

        # 1. Uniform grid (10 nm spacing)
        wl_unif = np.linspace(1300.0, 2500.0, 25)
        y_unif = a * wl_unif**2 + b * wl_unif + c
        dy_unif = savgol_derivative(y_unif, wl_unif, window=7, polyorder=2, deriv=1)
        d2y_unif = savgol_derivative(y_unif, wl_unif, window=7, polyorder=2, deriv=2)
        np.testing.assert_allclose(dy_unif, 2 * a * wl_unif + b, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(d2y_unif, np.full_like(wl_unif, 2 * a), rtol=1e-5, atol=1e-6)

        # 2. Non-uniform grid (exact 24-channel Vis-SWIR array with variable spacing)
        prof24 = get_builtin_profile("visnir_24ch_v1")
        assert prof24.is_uniform_grid is False
        wl_nonunif = np.asarray(prof24.wavelength_grid_nm, dtype=float)
        y_nonunif = a * wl_nonunif**2 + b * wl_nonunif + c

        dy_nonunif = savgol_derivative(y_nonunif, wl_nonunif, window=7, polyorder=2, deriv=1)
        d2y_nonunif = savgol_derivative(y_nonunif, wl_nonunif, window=7, polyorder=2, deriv=2)
        np.testing.assert_allclose(dy_nonunif, 2 * a * wl_nonunif + b, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(d2y_nonunif, np.full_like(wl_nonunif, 2 * a), rtol=1e-4, atol=1e-6)

    def test_invalid_window_or_too_few_bands_raises_schema_error(self) -> None:
        led5 = get_builtin_profile("led_5band_v1")
        spec = Spectrum(
            values=np.array([0.1, 0.2, 0.3, 0.25, 0.15]),
            wavelength_grid_nm=led5.wavelength_grid_nm,
            channel_ids=led5.channel_ids,
        )
        with pytest.raises(SchemaError, match="smaller than Savitzky-Golay window"):
            savgol_derivative(spec, window=7, polyorder=2, deriv=1)


class TestLIBSBaselineAndPseudoVoigt:
    """Verify ARPLS continuum baseline removal and Pseudo-Voigt peak parameter recovery."""

    def test_arpls_baseline_removal_and_pseudovoigt_peak_recovery(self) -> None:
        prof = get_builtin_profile("libs_ci_hi_v1")
        wl = np.asarray(prof.wavelength_grid_nm, dtype=float)

        # Smooth bremsstrahlung continuum baseline
        continuum = 400.0 + 180.0 * np.exp(-(wl - 190.0) / 150.0)

        # Known emission peaks inside C I (193.1, 229.7, 247.8 nm) and H I (656.3 nm)
        true_peaks = (
            (193.12, 2200.0, 0.14, 0.12, 0.4),
            (229.70, 950.0, 0.12, 0.10, 0.3),
            (247.82, 3400.0, 0.15, 0.14, 0.5),
            (656.30, 1600.0, 0.18, 0.15, 0.35),
        )
        signal = continuum.copy()
        for mu, amp, sig, gam, eta in true_peaks:
            signal += pseudovoigt_profile(wl, amp, mu, sig, gam, eta, offset=0.0) - 0.0

        spec = Spectrum(
            values=signal,
            wavelength_grid_nm=prof.wavelength_grid_nm,
            channel_ids=prof.channel_ids,
            mode=SpectralSignalMode.LIBS_EMISSION,
        )
        corrected_spec, est_base = correct_baseline(spec, lam=1e3, ratio=0.005, max_iter=80)

        # Estimated baseline should closely track true continuum away from peaks
        rel_base_err = float(np.median(np.abs(est_base - continuum) / continuum))
        assert rel_base_err < 0.06

        peak_info = extract_libs_peak_features(corrected_spec)
        assert peak_info["all_converged"] is True
        fits = {f["name"]: f for f in peak_info["peak_fits"]}
        assert abs(fits["C_I_193_1nm"]["center_nm"] - 193.12) < 0.03
        assert abs(fits["C_I_247_8nm"]["center_nm"] - 247.82) < 0.03
        assert abs(fits["H_I_656_3nm"]["center_nm"] - 656.30) < 0.03
        assert fits["C_I_247_8nm"]["amplitude"] > 2800.0


class TestEPOAndGLSWInvariantsAndSpaceConsistency:
    """Verify DEF-001 & ADR-003: matrix invariants, identical preprocessing space, and moisture removal."""

    def test_epo_matrix_invariants_and_space_mismatch_rejection(self) -> None:
        cfg = SyntheticConfig(
            dataset_id="epo_inv_test",
            profile_id="ftnir_1300_2550nm_v1",
            seed=42,
            noise_std_reflectance=0.0005,
        )
        gen = SyntheticGeneratorV2(cfg)
        dry_ds, wet_ds = gen.generate_paired_wet_dry_dataset(n_pairs=30)

        pipe_cfg = PreprocessConfig(
            pipeline_version="chem_v1",
            use_absorbance=True,
            use_snv=True,
            use_savgol=True,
            sg_window=7,
            sg_polyorder=2,
            sg_deriv=1,
            suppression_mode="epo",
            epo_q=3,
        )
        pipeline = ChemometricPipeline(pipe_cfg).fit_from_datasets(wet_ds, dry_ds)
        assert pipeline.is_fitted is True
        assert pipeline.epo is not None

        # 1. Mathematical Invariants: P^T = P, P^2 = P, V_s^T V_s = I_q
        inv = pipeline.epo.check_invariants()
        assert inv["symmetry_frobenius_error"] < 1e-11
        assert inv["idempotence_frobenius_error"] < 1e-11
        assert inv["orthonormality_frobenius_error"] < 1e-11

        # 2. Space mismatch rejection (DEF-001 guard)
        raw_X = np.vstack([r.spectrum.values for r in dry_ds.records])
        with pytest.raises(CalibrationError, match="EPO space mismatch"):
            pipeline.epo.transform(raw_X, space_signature="raw_unpreprocessed_space")

        # 3. Unfitted pipeline must reject transform (Section 32 gate)
        unfitted = ChemometricPipeline(pipe_cfg)
        with pytest.raises(CalibrationError, match="must be fitted"):
            unfitted.transform_dataset(dry_ds)

    def test_epo_reduces_moisture_variance_while_retaining_carbon_signal(self) -> None:
        cfg = SyntheticConfig(
            dataset_id="epo_physics_test",
            profile_id="ftnir_1300_2550nm_v1",
            seed=7,
            noise_std_reflectance=0.0002,
        )
        gen = SyntheticGeneratorV2(cfg)
        dry_cal, wet_cal = gen.generate_paired_wet_dry_dataset(n_pairs=35)
        dry_test, wet_test = SyntheticGeneratorV2(
            SyntheticConfig(profile_id="ftnir_1300_2550nm_v1", seed=999, noise_std_reflectance=0.0002)
        ).generate_paired_wet_dry_dataset(n_pairs=20)

        base_cfg = PreprocessConfig(
            use_absorbance=True,
            use_snv=True,
            use_savgol=True,
            sg_window=7,
            sg_polyorder=2,
            sg_deriv=1,
            suppression_mode="none",
        )
        epo_cfg = PreprocessConfig(
            use_absorbance=True,
            use_snv=True,
            use_savgol=True,
            sg_window=7,
            sg_polyorder=2,
            sg_deriv=1,
            suppression_mode="epo",
            epo_q=None,  # Dynamic rank selection (not hardcoded 2)
            epo_max_q=4,
        )

        wl = gen.profile.wavelength_grid_nm
        pipe_none = ChemometricPipeline(base_cfg).fit(wl)
        pipe_epo = ChemometricPipeline(epo_cfg).fit_from_datasets(wet_cal, dry_cal)

        z_dry_before = pipe_none.transform_dataset(dry_test)
        z_wet_before = pipe_none.transform_dataset(wet_test)
        moisture_diff_before = float(np.mean(np.linalg.norm(z_wet_before - z_dry_before, axis=1)))

        z_dry_after = pipe_epo.transform_dataset(dry_test)
        z_wet_after = pipe_epo.transform_dataset(wet_test)
        moisture_diff_after = float(np.mean(np.linalg.norm(z_wet_after - z_dry_after, axis=1)))

        # Moisture perturbation variance must drop substantially (> 75% reduction)
        assert pipe_epo.epo is not None and pipe_epo.epo.q is not None and pipe_epo.epo.q >= 1
        assert moisture_diff_after < 0.25 * moisture_diff_before

        # Between-sample variance across distinct SOC/SIC samples must remain non-trivial
        between_sample_std = float(np.mean(np.std(z_dry_after, axis=0)))
        assert between_sample_std > 0.05 * float(np.mean(np.std(z_dry_before, axis=0)))

    def test_glsw_suppression_reduces_wet_dry_divergence(self) -> None:
        cfg = SyntheticConfig(profile_id="visnir_24ch_v1", seed=11, noise_std_reflectance=0.0005)
        gen = SyntheticGeneratorV2(cfg)
        dry_cal, wet_cal = gen.generate_paired_wet_dry_dataset(n_pairs=25)

        glsw_cfg = PreprocessConfig(
            use_absorbance=True,
            use_snv=True,
            use_savgol=False,
            suppression_mode="glsw",
            glsw_alpha=0.01,
        )
        pipe_glsw = ChemometricPipeline(glsw_cfg).fit_from_datasets(wet_cal, dry_cal)
        assert pipe_glsw.glsw is not None

        # P_GLSW must be symmetric
        P = pipe_glsw.glsw.P
        assert P is not None
        np.testing.assert_allclose(P, P.T, atol=1e-12)

        z_dry_raw = pipe_glsw.apply_deterministic_transforms(
            np.vstack([r.spectrum.values for r in dry_cal.records]),
            gen.profile.wavelength_grid_nm,
        )
        z_wet_raw = pipe_glsw.apply_deterministic_transforms(
            np.vstack([r.spectrum.values for r in wet_cal.records]),
            gen.profile.wavelength_grid_nm,
        )
        diff_before = float(np.mean(np.linalg.norm(z_wet_raw - z_dry_raw, axis=1)))
        diff_after = float(
            np.mean(
                np.linalg.norm(
                    pipe_glsw.transform_dataset(wet_cal) - pipe_glsw.transform_dataset(dry_cal),
                    axis=1,
                )
            )
        )
        assert diff_after < 0.5 * diff_before
