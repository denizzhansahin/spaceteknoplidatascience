"""Faz 3 Unit & Contract Tests: Radiometry, ISS, Wavelength Alignment, Signal QC, LIBS QC, and DLIA."""

from __future__ import annotations

import numpy as np
import pytest

from st_ds.calibration import (
    align_wavelengths,
    apply_iss,
    calibrate_reflectance,
    fit_iss,
)
from st_ds.domain import (
    CalibrationError,
    InstrumentProfile,
    QualityFlag,
    QualityStatus,
    SpectralModality,
    SpectralQualityError,
    SpectralSignalMode,
    Spectrum,
    get_builtin_profile,
)
from st_ds.signal import (
    DLIAReadout,
    LIBSShotQCConfig,
    SignalQCConfig,
    evaluate_signal_qc,
    filter_libs_shots,
    summarize_dlia_outputs,
)


class TestRadiometricCalibration:
    """Verify white/dark reference calibration, saturation, denominator, and reference age rules."""

    def test_calibrate_reflectance_exact_fixture_and_flags(self) -> None:
        wl = (450.0, 650.0, 850.0, 1450.0, 2200.0)
        ch = tuple(f"CH_{i}" for i in range(5))
        dark = Spectrum(
            values=np.array([100.0, 100.0, 100.0, 100.0, 100.0]),
            wavelength_grid_nm=wl,
            channel_ids=ch,
            mode=SpectralSignalMode.RAW_INTENSITY,
            integration_time_ms=50.0,
            gain=1.0,
        )
        white = Spectrum(
            values=np.array([10100.0, 10100.0, 100.0, 65535.0, 20100.0]),  # ch2 zero denom, ch3 saturated
            wavelength_grid_nm=wl,
            channel_ids=ch,
            mode=SpectralSignalMode.RAW_INTENSITY,
            integration_time_ms=50.0,
            gain=1.0,
            reference_age_s=2400.0,  # exceeds default 1800s
        )
        raw = Spectrum(
            values=np.array([2600.0, 5100.0, 500.0, 10000.0, 10100.0]),
            wavelength_grid_nm=wl,
            channel_ids=ch,
            mode=SpectralSignalMode.RAW_INTENSITY,
            integration_time_ms=50.0,
            gain=1.0,
        )

        cal = calibrate_reflectance(raw, dark, white, saturation_limit=65535.0, max_reference_age_s=1800.0)
        assert cal.spectrum.mode == SpectralSignalMode.REFLECTANCE
        assert cal.valid_mask == (True, True, False, False, True)
        assert cal.clipping_count == 1
        assert cal.reference_age_s == 2400.0
        assert QualityFlag.SATURATED_BANDS in cal.quality_flags
        assert QualityFlag.DARK_WHITE_INVALID in cal.quality_flags
        assert QualityFlag.REFERENCE_AGED in cal.quality_flags

        # Exact reflectance check on valid bands: (2600-100)/(10100-100)=0.25, 0.50, 0.50
        assert pytest.approx(cal.spectrum.values[0]) == 0.25
        assert pytest.approx(cal.spectrum.values[1]) == 0.50
        assert pytest.approx(cal.spectrum.values[4]) == 0.50

    def test_integration_time_or_gain_mismatch_raises_calibration_error(self) -> None:
        wl = (450.0, 650.0, 850.0)
        ch = ("B1", "B2", "B3")
        raw = Spectrum(values=np.array([500.0, 600.0, 700.0]), wavelength_grid_nm=wl, channel_ids=ch, integration_time_ms=50.0)
        dark = Spectrum(values=np.array([10.0, 10.0, 10.0]), wavelength_grid_nm=wl, channel_ids=ch, integration_time_ms=100.0)
        white = Spectrum(values=np.array([5000.0, 5000.0, 5000.0]), wavelength_grid_nm=wl, channel_ids=ch, integration_time_ms=50.0)

        with pytest.raises(CalibrationError, match="Integration time mismatch"):
            calibrate_reflectance(raw, dark, white)


class TestISSCalibration:
    """Verify DEF-007 sign/direction convention for ISS correction."""

    def test_fit_and_apply_iss_recovers_standard_reference(self) -> None:
        prof = get_builtin_profile("led_5band_v1")
        s_std = Spectrum(
            values=np.array([0.80, 0.85, 0.90, 0.88, 0.82]),
            wavelength_grid_nm=prof.wavelength_grid_nm,
            channel_ids=prof.channel_ids,
        )
        # Secondary instrument has band-wise multiplicative drift (0.90, 1.05, 0.80, 1.10, 0.95)
        drift = np.array([0.90, 1.05, 0.80, 1.10, 0.95])
        m_meas = Spectrum(
            values=s_std.values * drift,
            wavelength_grid_nm=prof.wavelength_grid_nm,
            channel_ids=prof.channel_ids,
        )

        iss_model = fit_iss(s_std, m_meas)
        np.testing.assert_allclose(iss_model.correction_factors, drift, rtol=1e-9)

        corrected = apply_iss(m_meas, iss_model)
        np.testing.assert_allclose(corrected.values, s_std.values, rtol=1e-9)

        # Also check on a soil sample scanned by the secondary instrument
        soil_true = np.array([0.20, 0.30, 0.40, 0.35, 0.25])
        soil_sec = Spectrum(
            values=soil_true * drift,
            wavelength_grid_nm=prof.wavelength_grid_nm,
            channel_ids=prof.channel_ids,
        )
        soil_corr = apply_iss(soil_sec, iss_model)
        np.testing.assert_allclose(soil_corr.values, soil_true, rtol=1e-9)


class TestWavelengthAlignment:
    """Verify interpolation mask tracking and strict no-extrapolation rule (Section 9)."""

    def test_align_wavelengths_interior_interpolation_and_no_extrapolation(self) -> None:
        src_wl = (1300.0, 1500.0, 1700.0, 1900.0, 2100.0, 2300.0, 2550.0)
        src_spec = Spectrum(
            values=np.linspace(0.20, 0.50, len(src_wl)),
            wavelength_grid_nm=src_wl,
            channel_ids=tuple(f"S_{i}" for i in range(len(src_wl))),
        )
        # Target profile inside [1300, 2550] nm with a mix of exact and interpolated bands
        inner_target = InstrumentProfile(
            profile_id="inner_test",
            sensor_model="Test",
            modality=SpectralModality.VIS_NIR_SWIR,
            wavelength_grid_nm=(1300.0, 1400.0, 1500.0, 2000.0, 2300.0),
            channel_ids=("T1", "T2", "T3", "T4", "T5"),
        )
        aligned = align_wavelengths(src_spec, inner_target, method="linear")
        assert aligned.interpolated_mask == (False, True, False, True, False)
        assert pytest.approx(aligned.measured_band_ratio) == 3.0 / 5.0
        assert aligned.spectrum.n_bands == 5

        # Extrapolation attempt -> must raise CalibrationError
        ftnir_outside = InstrumentProfile(
            profile_id="outside_test",
            sensor_model="Test",
            modality=SpectralModality.VIS_NIR_SWIR,
            wavelength_grid_nm=(1100.0, 1500.0, 2600.0),
            channel_ids=("O1", "O2", "O3"),
        )
        with pytest.raises(CalibrationError, match="Extrapolation is forbidden"):
            align_wavelengths(src_spec, ftnir_outside)


class TestSignalQCAndLIBSShotQC:
    """Verify general signal QC, LIBS shot filtering, and DLIA output summarization."""

    def test_signal_qc_flags_healthy_vs_corrupted_spectra(self) -> None:
        prof = get_builtin_profile("visnir_24ch_v1")
        wl = np.asarray(prof.wavelength_grid_nm)
        clean_vals = 0.35 - 0.12 * np.exp(-0.5 * ((wl - 1920.0) / 80.0) ** 2)
        clean_spec = Spectrum(values=clean_vals, wavelength_grid_nm=prof.wavelength_grid_nm, channel_ids=prof.channel_ids)
        qc_ok = evaluate_signal_qc(clean_spec)
        assert qc_ok.is_valid_for_reporting is True
        assert qc_ok.status == QualityStatus.OK
        assert qc_ok.qc_score > 0.7

        # Flatline / disconnected sensor
        flat_spec = Spectrum(
            values=np.full(prof.n_channels, 0.25),
            wavelength_grid_nm=prof.wavelength_grid_nm,
            channel_ids=prof.channel_ids,
        )
        qc_flat = evaluate_signal_qc(flat_spec)
        assert qc_flat.is_valid_for_reporting is False
        assert qc_flat.status == QualityStatus.NOT_VALID_FOR_REPORTING
        assert QualityFlag.LOW_SNR in qc_flat.flags

        with pytest.raises(SpectralQualityError):
            evaluate_signal_qc(flat_spec, config=SignalQCConfig(reject_policy="raise"))

    def test_filter_libs_shots_rejects_saturated_shifted_disconnect_and_bad_energy(self) -> None:
        prof = get_builtin_profile("libs_ci_hi_v1")
        wl = np.asarray(prof.wavelength_grid_nm)
        rng = np.random.default_rng(42)

        def _clean_shot(peak_center: float = 247.8, amp: float = 4000.0) -> np.ndarray:
            sig = 120.0 + amp * np.exp(-0.5 * ((wl - peak_center) / 0.15) ** 2)
            sig += 2500.0 * np.exp(-0.5 * ((wl - 193.1) / 0.15) ** 2)
            sig += 1800.0 * np.exp(-0.5 * ((wl - 656.3) / 0.18) ** 2)
            return sig + rng.normal(0.0, 5.0, size=wl.size)

        shots = [_clean_shot() for _ in range(12)]
        energies = [50.0 for _ in range(12)]

        # Inject 4 bad shots:
        shots[2][40] = 65535.0  # saturated
        shots[5] = np.full_like(wl, 5.0)  # optical disconnect
        shots[7] = _clean_shot(peak_center=248.6)  # > 0.5 nm peak shift
        energies[9] = 10.0  # laser energy drop (-80%)

        libs_spec = Spectrum(
            values=np.vstack(shots),
            wavelength_grid_nm=prof.wavelength_grid_nm,
            channel_ids=prof.channel_ids,
            mode=SpectralSignalMode.LIBS_EMISSION,
        )
        res = filter_libs_shots(
            libs_spec,
            shot_energies=energies,
            config=LIBSShotQCConfig(min_surviving_shots=6),
        )
        assert set(res.rejected_indices) == {2, 5, 7, 9}
        assert len(res.kept_indices) == 8
        assert res.qc_report.is_valid_for_reporting is True
        assert res.averaged_spectrum.values.shape == (prof.n_channels,)

    def test_summarize_dlia_outputs(self) -> None:
        readouts = [
            DLIAReadout(
                channel_id="LED_460nm",
                wavelength_nm=460.0,
                in_phase=(3.0, 3.01, 2.99, 3.0),
                quadrature=(4.0, 4.01, 3.99, 4.0),
                reference_freq_hz=217.0,
            ),
            DLIAReadout(
                channel_id="LED_1450nm",
                wavelength_nm=1450.0,
                in_phase=(1.0, 1.0, 1.0, 1.0),
                quadrature=(0.0, 0.0, 0.0, 0.0),
                reference_freq_hz=217.0,
            ),
        ]
        summary = summarize_dlia_outputs(readouts)
        assert summary["n_channels"] == 2
        assert summary["all_locked"] is True
        assert pytest.approx(summary["amplitudes"][0], abs=1e-2) == 5.0
        assert pytest.approx(summary["amplitudes"][1], abs=1e-6) == 1.0
