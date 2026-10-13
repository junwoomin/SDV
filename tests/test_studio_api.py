"""API and configuration regressions; all writes remain in pytest temp roots."""
from __future__ import annotations

import io
import socket
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio.config import Config
from studio.api import create_app

WRITE_HEADERS = {"X-SDV-Request": "1"}


@pytest.fixture
def client(tmp_path):
    app = create_app(root=tmp_path)
    with TestClient(app, base_url="http://127.0.0.1:8765") as browser:
        yield browser


@pytest.mark.parametrize("path", ["../escape", "/etc/passwd", "."])
def test_project_containment_blocks_external_paths(tmp_path, path):
    config = Config(tmp_path)
    with pytest.raises(ValueError):
        config.save_settings({"data_root": path})


def test_project_containment_resolves_symlinks(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (root / "escape").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError):
        Config(root).save_settings({"results_root": "escape/results"})
    assert not (external / "results").exists()


@pytest.mark.parametrize("bad", [True, "nan", "Infinity", 1.5, -1, 65536])
def test_invalid_numeric_configuration_is_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        Config(tmp_path).save_settings({"carla_port": bad})


def test_invalid_configuration_does_not_overwrite_previous(tmp_path):
    config = Config(tmp_path)
    good = config.save_settings({"project_name": "QA durable settings"})
    with pytest.raises(ValueError):
        config.save_settings({"data_root": "../outside"})
    assert Config(tmp_path).settings() == good


def test_training_requires_real_supported_inputs(tmp_path):
    config = Config(tmp_path)
    invalid = config.training_defaults()
    invalid["sensor_use"]["front_cam"] = False
    with pytest.raises(ValueError):
        config.save_training(invalid)
    invalid = config.training_defaults()
    invalid["opt_checkboxes"]["SGD"] = True
    with pytest.raises(ValueError):
        config.save_training(invalid)
    invalid = config.training_defaults()
    invalid["model_conf"]["BEV"] = True
    with pytest.raises(ValueError):
        config.save_training(invalid)


def test_settings_persist_through_fresh_browser_session(client):
    result = client.put("/api/settings", json={"project_name": "QA browser persistence"}, headers=WRITE_HEADERS)
    assert result.status_code == 200, result.text
    result = client.get("/api/settings")
    assert result.status_code == 200
    assert result.json()["project_name"] == "QA browser persistence"


def test_write_requires_csrf_header(client):
    response = client.put("/api/settings", json={"project_name": "Unexpected write"})
    assert response.status_code == 403


def test_cross_origin_write_rejected_even_with_header(client):
    response = client.put("/api/settings", json={"project_name": "Unexpected write"}, headers={**WRITE_HEADERS, "Origin": "https://evil.example"})
    assert response.status_code == 403


def test_invalid_settings_has_explanatory_4xx(client):
    response = client.put("/api/settings", json={"carla_port": "wrong"}, headers=WRITE_HEADERS)
    assert response.status_code in (400, 422)
    assert response.json().get("detail")


def test_no_fake_jobs_in_fresh_application(client):
    response = client.get("/api/jobs")
    assert response.status_code == 200
    assert response.json() == {"jobs": []}


def test_local_read_endpoints_return_truthful_empty_state_and_headers(client):
    for path in ("/", "/api/status", "/api/config/sensors", "/api/config/training", "/api/experiments", "/api/datasets"):
        response = client.get(path)
        assert response.status_code == 200, (path, response.text)
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"
    status = client.get("/api/status").json()
    assert status["active_job"] is None
    assert status["carla"]["connected"] is False
    assert status["jobs"] == []
    assert client.get("/api/experiments").json() == {"experiments": []}
    assert client.get("/api/datasets").json() == {"datasets": []}


def test_missing_carla_and_training_inputs_block_real_execution(client):
    # Bind but never listen, so the isolated port cannot be a CARLA service.
    with socket.socket() as unused_server:
        unused_server.bind(("127.0.0.1", 0))
        port = unused_server.getsockname()[1]
        changed = client.put("/api/settings", json={"carla_port": port}, headers=WRITE_HEADERS)
        assert changed.status_code == 200
        response = client.post("/api/carla/connect", headers=WRITE_HEADERS)
        assert response.status_code == 200
        state = response.json()
        assert state["connected"] is False
        assert state["error"]
        assert state["hint"]
        for kind in ("collection", "training"):
            check = client.post("/api/preflight", json={"kind": kind}, headers=WRITE_HEADERS)
            assert check.status_code == 200
            assert check.json()["ready"] is False
            assert check.json()["blockers"]
        launch = client.post("/api/jobs", json={"kind": "training"}, headers=WRITE_HEADERS)
        assert launch.status_code == 409
        assert launch.json()["detail"]
    assert client.get("/api/jobs").json() == {"jobs": []}


def test_unknown_job_is_404_and_cannot_be_stopped(client):
    assert client.get("/api/jobs/not-a-job").status_code == 404
    assert client.post("/api/jobs/not-a-job/stop", headers=WRITE_HEADERS).status_code == 404


def test_foreign_host_and_fetch_site_writes_rejected(client):
    response = client.get("/api/settings", headers={"Host": "evil.example"})
    assert response.status_code == 403
    response = client.put("/api/settings", json={"project_name": "Unexpected write"}, headers={**WRITE_HEADERS, "Sec-Fetch-Site": "cross-site"})
    assert response.status_code == 403


def test_recorded_results_download_and_export_cannot_escape_run(client, tmp_path):
    run = tmp_path / "results" / "qa_fixture"
    run.mkdir(parents=True)
    source = "epoch,train_loss,val_loss\n1,0.75,0.90\n2,0.50,0.60\n"
    (run / "train_metrics.csv").write_text(source)
    external = tmp_path / "private.txt"
    external.write_text("QA PRIVATE FIXTURE - NEVER EXPORT")
    (run / "escape.txt").symlink_to(external)
    rows = client.get("/api/experiments").json()["experiments"]
    assert len(rows) == 1
    eid = rows[0]["id"]
    detail = client.get(f"/api/experiments/{eid}")
    assert detail.status_code == 200
    assert detail.json()["latest_metrics"] == {"train_loss": 0.5, "val_loss": 0.6}
    response = client.get(f"/api/experiments/{eid}/artifact", params={"path": "train_metrics.csv"})
    assert response.status_code == 200
    assert response.text == source
    for attack in ("../private.txt", str(external), "escape.txt", "..\\private.txt"):
        response = client.get(f"/api/experiments/{eid}/artifact", params={"path": attack})
        assert response.status_code in (400, 403, 404, 422), attack
    exported = client.get(f"/api/experiments/{eid}/export")
    assert exported.status_code == 200
    with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
        assert "train_metrics.csv" in archive.namelist()
        assert "escape.txt" not in archive.namelist()
        assert not any("PRIVATE FIXTURE" in archive.read(name).decode("utf-8", errors="ignore") for name in archive.namelist())
    assert client.get("/api/experiments/not-a-run").status_code == 404
