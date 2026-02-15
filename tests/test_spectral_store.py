"""Tests for spectral data pipeline (store, simulated instruments, capture hook)."""
from __future__ import annotations

import uuid

import pytest

from app.adapters.simulated_instrument import SimulatedAdapter
from app.services.spectral_store import SpectralRecord, SpectralStore, _flatten_to_floats
from app.services.metrics import _capture_spectral_data
from app.core.db import init_db, utcnow_iso


@pytest.fixture(autouse=True)
def _setup_db(tmp_path, monkeypatch):
    """Point the DB at a temp directory for every test."""
    from app.core import config as _cfg

    class _FakeSettings:
        data_dir = tmp_path
        object_store_dir = tmp_path / "objects"
        db_path = str(tmp_path / "test.db")

    monkeypatch.setattr(_cfg, "get_settings", lambda: _FakeSettings())
    init_db()


# ---------------------------------------------------------------------------
# SpectralStore basics
# ---------------------------------------------------------------------------


class TestSpectralStore:
    def test_store_and_query(self):
        store = SpectralStore()
        rec = _make_record(campaign_id="camp-1", technique="eis")
        rid = store.store(rec)
        assert rid == rec.record_id

        results = store.query("camp-1")
        assert len(results) == 1
        assert results[0].record_id == rec.record_id
        assert results[0].technique == "eis"
        assert results[0].raw_data == rec.raw_data

    def test_query_by_technique(self):
        store = SpectralStore()
        store.store(_make_record(campaign_id="camp-2", technique="eis"))
        store.store(_make_record(campaign_id="camp-2", technique="lsv"))
        store.store(_make_record(campaign_id="camp-2", technique="eis"))

        eis = store.query("camp-2", technique="eis")
        assert len(eis) == 2
        lsv = store.query("camp-2", technique="lsv")
        assert len(lsv) == 1

    def test_get_for_run(self):
        store = SpectralStore()
        run_id = "run-42"
        store.store(_make_record(run_id=run_id, campaign_id="c"))
        store.store(_make_record(run_id=run_id, campaign_id="c"))
        store.store(_make_record(run_id="run-other", campaign_id="c"))

        results = store.get_for_run(run_id)
        assert len(results) == 2

    def test_export_for_nexus(self):
        store = SpectralStore()
        cid = "camp-export"
        store.store(_make_record(
            campaign_id=cid,
            technique="eis",
            raw_data={"frequencies_hz": [1.0, 10.0], "z_real": [100.0, 90.0], "z_imag": [-5.0, -15.0]},
        ))
        store.store(_make_record(
            campaign_id=cid,
            technique="eis",
            raw_data={"frequencies_hz": [1.0, 10.0], "z_real": [105.0, 95.0], "z_imag": [-6.0, -16.0]},
        ))
        matrix = store.export_for_nexus(cid)
        assert len(matrix) == 2
        # Each row: frequencies(2) + z_imag(2) + z_real(2) = 6 floats (sorted keys)
        assert len(matrix[0]) == 6
        assert all(isinstance(v, float) for v in matrix[0])


# ---------------------------------------------------------------------------
# Simulated EIS spectrum
# ---------------------------------------------------------------------------


class TestSimulatedEIS:
    def test_eis_returns_spectrum(self):
        adapter = SimulatedAdapter()
        result = adapter.execute_primitive(
            instrument_id="sim-1",
            primitive="eis",
            params={"duration_s": 0.01},
        )
        assert result["ok"] is True
        # Backward compat
        assert "impedance_ohm" in result
        assert isinstance(result["impedance_ohm"], float)
        # New spectrum data
        spec = result["spectrum"]
        assert spec["technique"] == "eis"
        assert len(spec["frequencies_hz"]) == 15
        assert len(spec["z_real"]) == 15
        assert len(spec["z_imag"]) == 15
        assert "r_sol_ohm" in spec
        assert "r_ct_ohm" in spec


# ---------------------------------------------------------------------------
# Simulated LSV curve
# ---------------------------------------------------------------------------


class TestSimulatedLSV:
    def test_lsv_returns_curve(self):
        adapter = SimulatedAdapter()
        result = adapter.execute_primitive(
            instrument_id="sim-1",
            primitive="lsv",
            params={"duration_s": 0.01, "e_start_v": 0.0, "e_end_v": 1.0},
        )
        assert result["ok"] is True
        spec = result["spectrum"]
        assert spec["technique"] == "lsv"
        assert len(spec["potential_v"]) == 20
        assert len(spec["current_ma"]) == 20


# ---------------------------------------------------------------------------
# Spectral capture from step results
# ---------------------------------------------------------------------------


class TestSpectralCapture:
    def test_capture_from_step_result(self):
        store = SpectralStore()
        step_result = {
            "instrument_id": "sim-1",
            "primitive": "eis",
            "impedance_ohm": 100.0,
            "spectrum": {
                "technique": "eis",
                "frequencies_hz": [1.0, 10.0],
                "z_real": [100.0, 90.0],
                "z_imag": [-5.0, -10.0],
            },
            "ok": True,
        }
        _capture_spectral_data("run-cap-1", "camp-cap", step_result)

        records = store.get_for_run("run-cap-1")
        assert len(records) == 1
        assert records[0].technique == "eis"
        assert records[0].campaign_id == "camp-cap"

    def test_capture_skips_without_spectrum(self):
        store = SpectralStore()
        step_result = {"instrument_id": "sim-1", "ok": True}
        _capture_spectral_data("run-cap-2", "camp-cap", step_result)

        records = store.get_for_run("run-cap-2")
        assert len(records) == 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    *,
    record_id: str | None = None,
    run_id: str = "run-1",
    campaign_id: str = "camp-1",
    technique: str = "eis",
    raw_data: dict | None = None,
    metadata: dict | None = None,
) -> SpectralRecord:
    return SpectralRecord(
        record_id=record_id or str(uuid.uuid4()),
        run_id=run_id,
        campaign_id=campaign_id,
        technique=technique,
        raw_data=raw_data or {"frequencies_hz": [1.0, 10.0], "z_real": [100.0, 90.0], "z_imag": [-5.0, -10.0]},
        metadata=metadata or {"sample": "test"},
        timestamp=utcnow_iso(),
    )
