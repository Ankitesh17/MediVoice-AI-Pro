import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from types import SimpleNamespace

import pytest

from app.adapters.stt.whisper import FasterWhisperSTT
from app.services import appointments


def test_whisper_preload_is_reused_across_concurrent_sessions(monkeypatch):
    from app.adapters.stt import preload_stt_model, whisper

    loads = []

    def load(size, *, device, compute_type):
        loads.append((size, device, compute_type))
        time.sleep(0.02)
        return object()

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=load))
    monkeypatch.setattr(whisper, "_models", {})
    monkeypatch.setenv("STT_PROVIDER", "whisper")
    monkeypatch.setenv("WHISPER_MODEL", "tiny.en")
    with ThreadPoolExecutor(max_workers=4) as pool:
        preload = pool.submit(preload_stt_model)
        sessions = list(pool.map(FasterWhisperSTT, [None] * 3))
        preload.result()
    assert loads == [("tiny.en", "cpu", "int8")]
    assert all(session._model is sessions[0]._model for session in sessions)
    assert sessions[0] is not sessions[1]


def test_sqlite_path_preserves_appointment_lifecycle(tmp_path, monkeypatch: pytest.MonkeyPatch):
    database = tmp_path / "persist" / "appointments.sqlite3"
    monkeypatch.setenv("MEDIVOICE_DB_PATH", str(database))
    day = date.today() + timedelta(days=7)
    while day.weekday() >= 5:
        day += timedelta(days=1)

    booked = appointments.book_appointment("Demo", day.isoformat(), "09:00", "Demo visit")
    assert database.is_file()
    assert booked["success"]
    reference = booked["appointment_id"]
    appointments.init_db()  # Startup on a subsequent run keeps the existing row.
    assert appointments.reschedule_appointment(reference, day.isoformat(), "10:00")["success"]
    assert appointments.get_appointment(reference)["appointment_time"] == "10:00"
    assert appointments.cancel_appointment(reference)["success"]
    assert appointments.get_appointment(reference)["status"] == "cancelled"
