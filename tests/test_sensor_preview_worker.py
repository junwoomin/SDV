"""Explicit fake-CARLA fixtures; these tests do not claim CARLA/GPU coverage."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from studio import sensor_preview_worker as preview


class FakeBlueprint:
    def __init__(self, name):
        self.name, self.attributes = name, {}

    def has_attribute(self, name):
        return True

    def set_attribute(self, name, value):
        self.attributes[name] = value


class FakeActor:
    def __init__(self, world, blueprint, transform=None, parent=None):
        self.id = len(world.actors) + 100
        self.world, self.blueprint, self.transform, self.parent = world, blueprint, transform, parent
        self.callback = None
        self.destroyed = False
        self.physics = None
        self.stopped = False
        world.actors.append(self)

    def set_simulate_physics(self, value):
        self.physics = value

    def listen(self, callback):
        self.callback = callback

    def stop(self):
        self.stopped = True

    def destroy(self):
        self.destroyed = True
        self.world.destroyed_ids.append(self.id)
        return True


class FakeWorld:
    def __init__(self, synchronous=False, fail_sensor_at=None):
        self.actors = []
        self.destroyed_ids = []
        self.settings = SimpleNamespace(synchronous_mode=synchronous, fixed_delta_seconds=0.05)
        self.tick_calls = []
        self.spawned_sensors = 0
        self.fail_sensor_at = fail_sensor_at
        self.map = SimpleNamespace(name="/Game/Carla/Maps/Town04", get_spawn_points=lambda: ["available-spawn"])

    def get_settings(self):
        return self.settings

    def apply_settings(self, value):
        pytest.fail("preview must not modify shared world settings")

    def get_map(self):
        return self.map

    def get_blueprint_library(self):
        return SimpleNamespace(find=FakeBlueprint)

    def try_spawn_actor(self, blueprint, transform):
        return FakeActor(self, blueprint, transform)

    def spawn_actor(self, blueprint, transform, attach_to):
        self.spawned_sensors += 1
        if self.spawned_sensors == self.fail_sensor_at:
            raise RuntimeError("fixture sensor spawn failed")
        return FakeActor(self, blueprint, transform, attach_to)

    def tick(self, timeout):
        self.tick_calls.append(timeout)


class FakeClient:
    def __init__(self, world):
        self.world = world
        self.timeout = None

    def set_timeout(self, seconds):
        self.timeout = seconds

    def get_world(self):
        return self.world

    def get_client_version(self):
        return "0.9.15"

    def get_server_version(self):
        return "0.9.15"


def configuration():
    camera = dict(x=1.3, y=0, z=2.3, pitch=0, roll=0, yaw=0,
                  width=1920, height=1080, fov=100, seg=True, depth=True, od=True)
    lidar = dict(x=0, y=0, z=2.5, pitch=0, roll=0, yaw=0,
                 channels=16, range=40, points_per_second=20000, rotation_frequency=10)
    return {"towns": ["Town01"], "data_num": 10, "sensors": {"front_cam": camera, "lidar": lidar}}


def make_worker(tmp_path, synchronous=False, owns_clock=False, fail_sensor_at=None, fps=None):
    world = FakeWorld(synchronous, fail_sensor_at)
    client = FakeClient(world)
    carla = SimpleNamespace(
        Client=lambda host, port: client,
        Location=lambda **kwargs: SimpleNamespace(**kwargs),
        Rotation=lambda **kwargs: SimpleNamespace(**kwargs),
        Transform=lambda location, rotation: SimpleNamespace(location=location, rotation=rotation))
    snapshot = {"settings": {"carla_host": "127.0.0.1", "carla_port": 2000,
                              "vehicle": "vehicle.tesla.model3"},
                "sensors": configuration(), "revision": 0, "view": "perspective",
                "exclusive_tick": owns_clock}
    if fps is not None:
        snapshot["settings"]["preview_fps"] = fps
    worker = preview.PreviewWorker(tmp_path, snapshot, carla, np, Image)
    return worker, world


def rgb_measurement(frame=23):
    return SimpleNamespace(raw_data=bytes([0, 0, 255, 255] * 4), width=2, height=2,
                           frame=frame, timestamp=1.25)


def test_actual_selected_mount_configuration_stationary_vehicle_and_current_map(tmp_path):
    worker, world = make_worker(tmp_path)
    worker.connect()
    assert worker.vehicle.physics is False
    assert worker.vehicle.blueprint.attributes["role_name"].startswith("sdv_preview_")
    assert worker.state["map"].endswith("Town04")  # collection towns must not load another map
    camera, lidar = worker.sensors
    assert camera.parent is worker.vehicle and camera.transform.location.x == 1.3
    assert camera.blueprint.attributes["image_size_x"] == "640"
    assert camera.blueprint.attributes["image_size_y"] == "360"
    assert float(camera.blueprint.attributes["sensor_tick"]) == pytest.approx(1/24)
    assert lidar.blueprint.attributes["channels"] == "16"
    assert lidar.blueprint.attributes["range"] == "40"
    assert lidar.blueprint.attributes["points_per_second"] == "20000"
    assert worker.fps == 24
    assert set(worker.channels) == {"front_cam_rgb", "lidar"}
    assert world.spawned_sensors == len(configuration()["sensors"])
    assert worker.channels["front_cam_rgb"]["configured_width"] == 1920
    assert worker.channels["front_cam_rgb"]["scaled"] is True
    assert not world.tick_calls
    assert worker.cleanup() == []


def test_real_pixel_and_point_buffers_publish_latest_jpeg_with_observed_metadata(tmp_path, capsys):
    worker, world = make_worker(tmp_path)
    worker.connect()
    capsys.readouterr()
    worker.sensors[0].callback(rgb_measurement())
    points = np.array([[0, 0, 0, 1], [10, 5, 0, 0.5]], dtype=np.float32)
    worker.sensors[1].callback(SimpleNamespace(raw_data=points.tobytes(), frame=24, timestamp=1.3))
    worker.publish()
    assert capsys.readouterr().out == ""  # live frames do not flood worker.log
    camera_file = tmp_path / "frames/front_cam_rgb.jpg"
    assert Image.open(str(camera_file)).getpixel((0, 0))[0] > 240
    state = json.loads((tmp_path / "status.json").read_text())
    channels = {entry["id"]: entry for entry in state["channels"]}
    assert channels["front_cam_rgb"]["frame"] == 23
    assert channels["front_cam_rgb"]["timestamp"] == 1.25
    assert channels["lidar"]["point_count"] == 2
    assert channels["lidar"]["displayed_points"] == 2
    assert state["ready"] is True
    assert channels["front_cam_rgb"]["measured_fps"] is None
    assert channels["front_cam_rgb"]["requested_fps"] == 24
    assert sorted(path.name for path in (tmp_path / "frames").iterdir()) == ["front_cam_rgb.jpg", "lidar.jpg"]
    assert not list((tmp_path / "frames").glob("*.tmp"))
    worker.cleanup()


def test_lidar_axes_and_nan_filter_are_based_on_real_points():
    points = np.array([[5, -5, 0, 1], [np.nan, 0, 0, 1], [100, 0, 0, 1]], dtype=np.float32)
    image, metadata = preview.lidar_image(np, Image, {"data": points.tobytes()}, 10, size=101)
    assert image.getpixel((25, 25)) != (9, 15, 25)
    assert image.getpixel((75, 75)) == (9, 15, 25)
    assert metadata["point_count"] == 3 and metadata["displayed_points"] == 1


def test_revision_rebuild_moves_sensors_ignores_legacy_view_and_drops_stale_callbacks(tmp_path):
    worker, world = make_worker(tmp_path)
    worker.connect()
    previous = list(worker.sensors)
    previous[0].callback(rgb_measurement())
    worker.publish()
    new = configuration()
    new["sensors"]["front_cam"]["x"] = 2.7
    preview.atomic_json(tmp_path / "command.json", {"revision": 1, "sensors": new, "view": "top"})
    worker.commands()
    assert worker.revision == 1 and worker.view == "top"
    assert all(actor.destroyed and actor.stopped for actor in previous)
    assert worker.sensors[0].transform.location.x == 2.7
    assert worker.sensors[-1].transform.rotation.pitch == 0
    assert set(worker.channels) == {"front_cam_rgb", "lidar"}
    assert not (tmp_path / "frames/front_cam_rgb.jpg").exists()
    previous[0].callback(rgb_measurement(99))
    assert not worker.pending
    worker.sensors[0].callback(rgb_measurement(31))
    worker.publish()
    assert worker.channels["front_cam_rgb"]["revision"] == 1
    assert worker.channels["front_cam_rgb"]["frame"] == 31
    worker.cleanup()


@pytest.mark.parametrize("synchronous,owned,expected_ticks", [(False, False, 0), (True, False, 0), (True, True, 1)])
def test_clock_ownership_is_explicit_and_sync_tick_rate_is_bounded(tmp_path, synchronous, owned, expected_ticks):
    worker, world = make_worker(tmp_path, synchronous, owned)
    worker.connect()
    worker.advance()
    worker.advance()
    assert len(world.tick_calls) == expected_ticks
    assert worker.state["owns_clock"] == bool(synchronous and owned)
    assert worker.state["world_settings_modified"] is False
    worker.cleanup()


@pytest.mark.parametrize("fps", [1, 2, 24, 60])
def test_integer_preview_fps_bounds_are_respected(tmp_path, fps):
    worker, world = make_worker(tmp_path, fps=fps)
    worker.connect()
    assert worker.fps == fps
    assert worker.frame_timeout == 15
    assert float(worker.sensors[0].blueprint.attributes["sensor_tick"]) == pytest.approx(1/fps)
    worker.cleanup()


@pytest.mark.parametrize("fps", [0, 0.1, 1.5, 61, True, float("nan"), "invalid"])
def test_noninteger_or_out_of_range_preview_fps_fails_before_spawning(tmp_path, fps):
    with pytest.raises(ValueError, match="1~60"):
        make_worker(tmp_path, fps=fps)


def test_apply_updates_running_two_fps_worker_to_saved_target(tmp_path, monkeypatch):
    worker, world = make_worker(tmp_path, fps=2)
    worker.connect()
    previous = list(worker.sensors)
    preview.atomic_json(tmp_path / "command.json", {
        "revision": 1, "sensors": configuration(), "preview_fps": 24})
    worker.commands()
    assert worker.fps == 24
    assert worker.state["requested_fps"] == 24
    assert worker.state["preview_fps"] == 24
    assert all(actor.destroyed for actor in previous)
    assert float(worker.sensors[0].blueprint.attributes["sensor_tick"]) == pytest.approx(1/24)
    clock = [100.0]
    monkeypatch.setattr(preview.time, "monotonic", lambda: clock[0])
    for index in range(11):  # A 10 FPS world remains 10 FPS after requesting 24.
        clock[0] = 100.0 + index / 10.0
        worker.sensors[0].callback(rgb_measurement(index))
        worker.publish()
    worker.status()
    assert worker.channels["front_cam_rgb"]["requested_fps"] == 24
    assert worker.channels["front_cam_rgb"]["received_fps"] == 10
    assert worker.channels["front_cam_rgb"]["measured_fps"] == 10
    worker.cleanup()


def test_observed_fps_and_latest_only_buffer_do_not_fabricate_target_rate(tmp_path, monkeypatch):
    worker, world = make_worker(tmp_path, fps=24)
    worker.connect()
    clock = [100.0]
    monkeypatch.setattr(preview.time, "monotonic", lambda: clock[0])
    for index in range(21):  # Explicit fixture: CARLA supplies only 20 real frames/s.
        clock[0] = 100.0 + index / 20.0
        worker.sensors[0].callback(rgb_measurement(index))
        assert len(worker.pending) == 1
        worker.publish()
    worker.status()
    row = worker.channels["front_cam_rgb"]
    assert row["requested_fps"] == 24
    assert row["received_fps"] == 20
    assert row["measured_fps"] == 20
    assert row["frame"] == 20
    clock[0] += 3
    worker.status()
    assert row["received_fps"] == 0 and row["measured_fps"] == 0
    worker.cleanup()


def test_callback_pacing_keeps_fractional_input_rate_without_half_rate_aliasing(tmp_path, monkeypatch):
    worker, world = make_worker(tmp_path, fps=24)
    worker.connect()
    clock = [100.0]
    monkeypatch.setattr(preview.time, "monotonic", lambda: clock[0])
    accepted = []
    for index in range(60):  # A 30 FPS fixture should produce about 24, not 15 FPS.
        clock[0] = 100.0 + index / 30.0
        worker.sensors[0].callback(rgb_measurement(index))
        if worker.pending:
            accepted.append(worker.pending.pop("front_cam_rgb")["frame"])
    assert 47 <= len(accepted) <= 49
    assert sorted(accepted) == accepted
    worker.cleanup()


def test_external_sync_missing_frames_has_actionable_error_without_ticking(tmp_path, monkeypatch):
    worker, world = make_worker(tmp_path, synchronous=True)
    worker.connect()
    worker.frame_deadline_start = 0
    monkeypatch.setattr(preview.time, "monotonic", lambda: 20)
    with pytest.raises(RuntimeError, match="원래 클라이언트가 tick"):
        worker.check_frames()
    assert not world.tick_calls
    worker.cleanup()


def test_failure_cleans_only_own_actors_before_terminal_status(tmp_path, capsys):
    worker, world = make_worker(tmp_path, fail_sensor_at=2)
    foreign = FakeActor(world, FakeBlueprint("vehicle.foreign"))
    assert worker.execute() == 1
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == "Failed" and "fixture sensor spawn failed" in status["error"]
    assert status["actor_ids"] == []
    assert all(actor.destroyed for actor in world.actors if actor is not foreign)
    assert not foreign.destroyed
    assert "Failed / stopped / revision 0: fixture sensor spawn failed" in capsys.readouterr().out


def test_batched_cleanup_never_ticks_or_destroys_unowned_actor(tmp_path):
    worker, world = make_worker(tmp_path)
    foreign = FakeActor(world, FakeBlueprint("vehicle.foreign"))
    worker.connect()
    selected = [actor.id for actor in worker.sensors]
    calls = []
    worker.carla.command = SimpleNamespace(DestroyActor=lambda actor_id: actor_id)
    def batch(actor_ids, tick):
        calls.append((actor_ids, tick))
        for actor in world.actors:
            if actor.id in actor_ids:
                actor.destroy()
        return [SimpleNamespace(error=None) for actor_id in actor_ids]
    worker.client.apply_batch_sync = batch
    assert worker.cleanup() == []
    assert calls == [(selected, False)]
    assert not foreign.destroyed
    assert worker.vehicle is None and worker.sensors == []


def test_status_write_failure_does_not_skip_actor_cleanup(tmp_path, monkeypatch):
    worker, world = make_worker(tmp_path)
    worker.connect()
    owned = list(world.actors)
    def failed_status(**kwargs):
        raise OSError("fixture full disk")
    monkeypatch.setattr(worker, "status", failed_status)
    errors = worker.cleanup()
    assert any("상태 파일 기록 실패" in error for error in errors)
    assert all(actor.destroyed for actor in owned)


def test_stop_file_stops_without_touching_existing_world(tmp_path):
    worker, world = make_worker(tmp_path)
    (tmp_path / "stop.json").write_text("{}")
    assert worker.execute() == 0
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "Stopped"
    assert not world.actors and not world.tick_calls


def test_missing_package_names_the_selected_python_requirement(monkeypatch):
    def missing(name):
        raise ImportError("No module named carla")
    monkeypatch.setattr(preview.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="carla==0.9.15"):
        preview.load_dependencies()


def test_no_capture_buffer_accepts_nonfinite_or_unsupported_sensor_config():
    config = configuration()
    config["sensors"]["front_cam"]["x"] = float("nan")
    with pytest.raises(ValueError, match="좌표"):
        preview.sensor_configuration(config)
