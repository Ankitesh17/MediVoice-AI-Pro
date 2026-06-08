from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ws_echo() -> None:
    with client.websocket_connect("/ws/echo") as ws:
        ready = ws.receive_json()
        assert ready == {"type": "state", "value": "ready"}
        ws.send_bytes(b"\x01\x02\x03")
        assert ws.receive_bytes() == b"\x01\x02\x03"
        ws.send_text("hello")
        assert ws.receive_json() == {"type": "echo", "text": "hello"}
