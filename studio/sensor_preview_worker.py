"""Real CARLA sensor preview worker; run with the selected research Python.

Only CARLA, NumPy and Pillow are runtime dependencies. Frames are latest-only;
this worker does not run collection, load a map, or change world settings.
"""
import argparse
from collections import deque
import importlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
import uuid

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sdv.collection.sensor_attributes import configure_camera_blueprint, configure_lidar_blueprint


CAMERAS = ("front_cam", "left_cam", "right_cam", "rear_cam", "rear_left_cam", "rear_right_cam", "front_cam_pv")
POSE_FIELDS = ("x", "y", "z", "pitch", "roll", "yaw")
# Accepted only for old API callers; no external-view camera is created.
VIEWS = frozenset(("perspective", "top", "side"))
FRAME_TIMEOUT = 15.0
FRAME_RATE_NOTE = "실제 FPS는 CARLA tick·렌더링 속도에 따라 요청한 값보다 낮아질 수 있습니다."


def log_message(message):
    try:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        print("[{}] {}".format(stamp, message), flush=True)
    except OSError:
        # A closed log pipe must never prevent ownership-scoped actor cleanup.
        pass


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, allow_nan=False)
            output.flush()
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_dependencies():
    modules = []
    for name, package in (("carla", "carla==0.9.15"), ("numpy", "numpy"), ("PIL.Image", "Pillow")):
        try:
            modules.append(importlib.import_module(name))
        except ImportError as error:
            raise RuntimeError("선택한 연구 Python에 {} 패키지가 없거나 불러올 수 없습니다: {}".format(package, error))
    return tuple(modules)


def preview_dimensions(width, height, maximum=640):
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("카메라 너비와 높이는 0보다 커야 합니다.")
    scale = min(1.0, float(maximum) / max(width, height))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def preview_frame_rate(value=24):
    try:
        parsed = float(value)
        if isinstance(value, bool) or not math.isfinite(parsed) or parsed != int(parsed) or not 1 <= parsed <= 60:
            raise ValueError()
    except (TypeError, ValueError, OverflowError):
        raise ValueError("미리보기 FPS는 1~60 범위의 정수여야 합니다.")
    return int(parsed)


def measured_rate(samples, now):
    recent = [stamp for stamp in samples if now - stamp <= 2.0]
    if not recent:
        return 0.0 if samples else None
    if len(recent) < 2 or recent[-1] <= recent[0]:
        return None
    return round((len(recent) - 1) / (recent[-1] - recent[0]), 2)


def sensor_configuration(value):
    sensors = value.get("sensors", value) if isinstance(value, dict) else None
    if not isinstance(sensors, dict) or not sensors:
        raise ValueError("미리보기할 센서 설정이 없습니다.")
    for name, config in sensors.items():
        if name not in CAMERAS + ("lidar",) or not isinstance(config, dict):
            raise ValueError("지원하지 않는 미리보기 센서: " + str(name))
        for field in POSE_FIELDS:
            number = config.get(field)
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
                raise ValueError("{}.{}: 유한한 센서 좌표/각도가 필요합니다.".format(name, field))
        if name != "lidar":
            preview_dimensions(config["width"], config["height"])
    return sensors


def camera_image(np, Image, sample):
    pixels = np.frombuffer(sample["data"], dtype=np.uint8)
    expected = sample["width"] * sample["height"] * 4
    if pixels.size != expected:
        raise ValueError("CARLA 카메라 프레임 크기가 메타데이터와 다릅니다.")
    bgra = pixels.reshape((sample["height"], sample["width"], 4))
    return Image.fromarray(bgra[:, :, 2::-1].copy(), "RGB"), {}


def lidar_image(np, Image, sample, sensor_range, size=640):
    points = np.frombuffer(sample["data"], dtype=np.float32)
    if points.size % 4:
        raise ValueError("CARLA LiDAR 프레임은 x, y, z, intensity 네 열이어야 합니다.")
    points = points.reshape((-1, 4))
    total = len(points)
    stride = max(1, int(math.ceil(total / 100000.0)))
    points = points[::stride]
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    extent = float(sensor_range)
    valid = (np.abs(points[:, 0]) <= extent) & (np.abs(points[:, 1]) <= extent)
    points = points[valid]
    canvas = np.empty((size, size, 3), dtype=np.uint8)
    canvas[:] = (9, 15, 25)
    if len(points):
        # CARLA sensor-local x is forward (image up), y right (image right).
        x = np.rint((points[:, 1] / extent + 1) * (size - 1) / 2).astype(np.int32)
        y = np.rint((1 - points[:, 0] / extent) * (size - 1) / 2).astype(np.int32)
        intensity = np.clip(points[:, 3], 0, 1)
        colors = np.stack((40 + intensity * 55, 145 + intensity * 90, 180 + intensity * 75), axis=1).astype(np.uint8)
        canvas[y, x] = colors
    return Image.fromarray(canvas, "RGB"), {
        "point_count": total, "displayed_points": len(points), "sample_stride": stride,
        "range_m": extent, "coordinate_frame": "sensor-local: x forward/up, y right",
    }


class PreviewWorker:
    def __init__(self, run, snapshot, carla, np, Image, cancel=None):
        self.run = Path(run).resolve()
        self.run.mkdir(parents=True, exist_ok=True)
        self.frames = self.run / "frames"
        self.frames.mkdir(exist_ok=True)
        self.snapshot = snapshot
        self.settings = snapshot["settings"]
        self.carla, self.np, self.Image = carla, np, Image
        self.cancel = cancel if cancel is not None else threading.Event()
        self.fps = preview_frame_rate(self.settings.get("preview_fps", 24))
        self.frame_timeout = max(FRAME_TIMEOUT, 3 / self.fps)
        self.tick_interval = 0.05
        self.next_tick_due = 0
        self.revision = int(snapshot.get("revision", 0))
        self.view = snapshot.get("view", "perspective")
        if self.view not in VIEWS:
            raise ValueError("지원하지 않는 미리보기 시점입니다.")
        self.config = sensor_configuration(snapshot["sensors"])
        self.exclusive_tick = snapshot.get("exclusive_tick") is True
        self.client = self.world = self.vehicle = None
        self.sensors = []
        self.channels = {}
        self.pending = {}
        self.lock = threading.Lock()
        self.frame_event = threading.Event()
        self.generation = 0
        self.last_received = {}
        self.last_callback = {}
        self.last_written = {}
        self.received_times = {}
        self.published_times = {}
        self.status_dirty = False
        self.started = time.time()
        self.frame_deadline_start = time.monotonic()
        self.last_status_write = 0
        self.command_signature = None
        self.callback_error = None
        self.last_log_signature = None
        self.state = {
            "status": "Running", "phase": "connecting", "message": "CARLA 센서 미리보기를 연결합니다.",
            "error": None, "started_at": self.started, "pid": os.getpid(), "revision": self.revision,
            "view": self.view, "preview_fps": self.fps, "requested_fps": self.fps,
            "frame_rate_note": FRAME_RATE_NOTE, "map": None,
            "synchronous_mode": None, "owns_clock": False, "world_settings_modified": False,
            "actor_ids": [], "channels": [], "cleanup_errors": [],
        }

    def stopped(self):
        return self.cancel.is_set() or (self.run / "stop.json").exists()

    def status(self, **changes):
        self.state.update(changes)
        now = time.monotonic()
        with self.lock:
            for name, channel in self.channels.items():
                channel.update(requested_fps=self.fps,
                               received_fps=measured_rate(self.received_times.get(name, ()), now),
                               measured_fps=measured_rate(self.published_times.get(name, ()), now))
        self.state.update(updated_at=time.time(), revision=self.revision, view=self.view,
                          preview_fps=self.fps, requested_fps=self.fps,
                          actor_ids=[a.id for a in self.sensors] + ([self.vehicle.id] if self.vehicle is not None else []),
                          channels=list(self.channels.values()),
                          ready=self.state["status"] == "Running" and self.state["phase"] == "live"
                          and bool(self.channels) and all(c.get("frame") is not None for c in self.channels.values()))
        signature = (self.state["status"], self.state["phase"], self.revision, self.state.get("error"))
        if signature != self.last_log_signature:
            self.last_log_signature = signature
            message = self.state.get("message", "")
            error = self.state.get("error")
            if error and error != message:
                message += " · " + str(error)
            log_message("{} / {} / revision {}: {}".format(signature[0], signature[1], self.revision, message))
        atomic_json(self.run / "status.json", self.state)
        self.last_status_write = time.monotonic()

    def transform(self, config):
        return self.carla.Transform(
            self.carla.Location(x=float(config["x"]), y=float(config["y"]), z=float(config["z"])),
            self.carla.Rotation(pitch=float(config["pitch"]), roll=float(config["roll"]), yaw=float(config["yaw"])))

    def connect(self):
        self.status()
        self.client = self.carla.Client(str(self.settings["carla_host"]), int(self.settings["carla_port"]))
        self.client.set_timeout(3.0)
        self.world = self.client.get_world()
        version = self.client.get_server_version()
        client_version = self.client.get_client_version()
        if version != client_version:
            raise RuntimeError("CARLA 버전 불일치: 서버 {}, Python {}".format(version, client_version))
        world_settings = self.world.get_settings()
        delta = float(getattr(world_settings, "fixed_delta_seconds", None) or 0.05)
        self.tick_interval = max(0.01, delta) if math.isfinite(delta) else 0.05
        self.state.update(map=self.world.get_map().name, server_version=version,
                          synchronous_mode=bool(world_settings.synchronous_mode),
                          owns_clock=bool(world_settings.synchronous_mode and self.exclusive_tick))
        # Do not load the collection's towns or change shared world/traffic settings.
        blueprint = self.world.get_blueprint_library().find(self.settings.get("vehicle", "vehicle.tesla.model3"))
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "sdv_preview_" + uuid.uuid4().hex[:12])
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("현재 맵에 차량 spawn point가 없습니다.")
        deadline = time.monotonic() + 10
        for spawn in spawn_points:
            if self.stopped():
                return
            if time.monotonic() > deadline:
                break
            self.vehicle = self.world.try_spawn_actor(blueprint, spawn)
            if self.vehicle is not None:
                break
        if self.vehicle is None:
            raise RuntimeError("빈 차량 생성 위치가 없습니다. 현재 맵의 차량 점유 상태를 확인하세요.")
        # This vehicle belongs only to the preview and never joins traffic.
        self.vehicle.set_simulate_physics(False)
        self.rebuild(self.config, self.revision, self.view)

    def callback(self, channel, generation, camera):
        def receive(measurement):
            try:
                now = time.monotonic()
                with self.lock:
                    if generation != self.generation or self.stopped():
                        return
                    self.last_received[channel] = now
                    self.received_times.setdefault(channel, deque(maxlen=256)).append(now)
                    due = self.last_callback.get(channel, now)
                    if now + 1e-9 < due:
                        return
                    interval = 1 / self.fps
                    self.last_callback[channel] = due + (int(max(0.0, now - due) / interval) + 1) * interval
                    sample = {"data": bytes(measurement.raw_data), "frame": int(measurement.frame),
                              "timestamp": float(measurement.timestamp), "received_at": time.time(),
                              "generation": generation}
                    if camera:
                        sample.update(width=int(measurement.width), height=int(measurement.height))
                    self.pending[channel] = sample
                self.frame_event.set()
            except Exception as error:
                self.callback_error = "{} 센서 콜백 오류: {}".format(channel, error)
        return receive

    def spawn_sensor(self, channel, blueprint, config, metadata, camera):
        if blueprint.has_attribute("sensor_tick"):
            blueprint.set_attribute("sensor_tick", str(1 / self.fps))
        actor = self.world.spawn_actor(blueprint, self.transform(config), attach_to=self.vehicle)
        self.sensors.append(actor)
        metadata.update(id=channel, channel=channel, frame=None, timestamp=None, received_at=None,
                        revision=self.revision, path="frames/" + channel + ".jpg",
                        pose={key: config[key] for key in POSE_FIELDS}, sensor_tick=1 / self.fps)
        self.channels[channel] = metadata
        actor.listen(self.callback(channel, self.generation, camera))

    def destroy_sensors(self):
        errors = []
        owned = list(self.sensors)
        for actor in owned:
            try:
                actor.stop()
            except Exception:
                pass  # Destruction below still handles a stopped/lost sensor.
        command = getattr(self.carla, "command", None)
        if owned and command is not None and hasattr(self.client, "apply_batch_sync"):
            try:
                responses = self.client.apply_batch_sync([command.DestroyActor(actor.id) for actor in owned], False)
                if len(responses) != len(owned):
                    raise RuntimeError("센서 정리 응답 개수가 일치하지 않습니다.")
                for actor, response in zip(owned, responses):
                    if response.error:
                        errors.append("sensor {}: {}".format(actor.id, response.error))
                    else:
                        self.sensors.remove(actor)
            except Exception as error:
                errors.append("센서 batch 정리 확인 실패: " + str(error))
        else:
            for actor in owned:
                try:
                    if actor.destroy() is False:
                        raise RuntimeError("destroy 반환값 false")
                    self.sensors.remove(actor)
                except Exception as error:
                    errors.append("sensor {}: {}".format(actor.id, error))
        return errors

    def rebuild(self, config, revision, view, fps=None):
        config = sensor_configuration(config)
        fps = preview_frame_rate(self.fps if fps is None else fps)
        if view not in VIEWS:
            raise ValueError("지원하지 않는 미리보기 시점입니다.")
        self.status(phase="applying", message="센서 위치와 설정을 실제 CARLA actor에 적용합니다.", error=None)
        with self.lock:
            self.generation += 1
            self.pending.clear()
            self.last_received.clear()
            self.last_callback.clear()
            self.last_written.clear()
            self.received_times.clear()
            self.published_times.clear()
            self.frame_event.clear()
            self.fps = fps
            self.frame_timeout = max(FRAME_TIMEOUT, 3 / self.fps)
        self.client.set_timeout(1.0)
        try:
            errors = self.destroy_sensors()
        finally:
            self.client.set_timeout(3.0)
        if errors:
            raise RuntimeError("기존 미리보기 센서 정리 실패: " + "; ".join(errors))
        for channel in self.channels:
            try:
                (self.frames / (channel + ".jpg")).unlink()
            except FileNotFoundError:
                pass
        self.channels = {}
        self.config, self.revision, self.view = config, int(revision), view
        library = self.world.get_blueprint_library()
        for name, original in config.items():
            if self.stopped():
                return
            if name == "lidar":
                blueprint = library.find("sensor.lidar.ray_cast")
                applied = configure_lidar_blueprint(blueprint, original)
                self.spawn_sensor("lidar", blueprint, original, {
                    "label": "LiDAR BEV", "kind": "lidar", "width": 640, "height": 640,
                    "configured_attributes": dict(applied), "capture_attributes": dict(applied),
                    "range_m": float(applied["range"]), "scaled": False,
                }, False)
            else:
                width, height = preview_dimensions(original["width"], original["height"])
                display_config = dict(original, width=width, height=height)
                blueprint = library.find("sensor.camera.rgb")
                applied = configure_camera_blueprint(blueprint, display_config)
                self.spawn_sensor(name + "_rgb", blueprint, original, {
                    "label": name + " RGB", "kind": "rgb", "width": width, "height": height,
                    "configured_width": int(original["width"]), "configured_height": int(original["height"]),
                    "capture_width": width, "capture_height": height,
                    "scaled": (width, height) != (int(original["width"]), int(original["height"])),
                    "fov": original["fov"], "capture_attributes": applied,
                }, True)
        self.frame_deadline_start = time.monotonic()
        self.status(phase="live", message="실제 센서 프레임을 기다립니다. 수집 해상도는 변경하지 않았습니다.")

    def commands(self):
        path = self.run / "command.json"
        if not path.exists():
            return
        signature = (path.stat().st_mtime_ns, path.stat().st_size)
        if signature == self.command_signature:
            return
        self.command_signature = signature
        value = json.loads(path.read_text(encoding="utf-8"))
        revision = int(value["revision"])
        if revision <= self.revision:
            return
        self.rebuild(value.get("sensors", self.config), revision, value.get("view", self.view),
                     fps=value.get("preview_fps", self.fps))

    def publish(self):
        with self.lock:
            pending, self.pending = self.pending, {}
        first_frame = False
        for channel, sample in pending.items():
            if sample["generation"] != self.generation or channel not in self.channels:
                continue
            metadata = self.channels[channel]
            first_frame = first_frame or metadata.get("frame") is None
            if metadata["kind"] == "lidar":
                image, additional = lidar_image(self.np, self.Image, sample, metadata["range_m"])
            else:
                image, additional = camera_image(self.np, self.Image, sample)
            temporary = self.frames / (channel + ".jpg.tmp")
            try:
                image.save(str(temporary), format="JPEG", quality=78)
                os.replace(str(temporary), str(self.frames / (channel + ".jpg")))
            finally:
                if temporary.exists():
                    temporary.unlink()
            metadata.update(additional)
            metadata.update(frame=sample["frame"], timestamp=sample["timestamp"], received_at=sample["received_at"],
                            width=image.width, height=image.height, revision=self.revision)
            self.last_written[channel] = time.monotonic()
            self.published_times.setdefault(channel, deque(maxlen=256)).append(self.last_written[channel])
            self.status_dirty = True
        elapsed = time.monotonic() - self.last_status_write
        if first_frame or (self.status_dirty and elapsed >= 1 / self.fps) or elapsed > 1:
            self.status(phase="live", message="실제 CARLA 센서 미리보기 · 요청 {} FPS. {}".format(self.fps, FRAME_RATE_NOTE))
            self.status_dirty = False

    def advance(self):
        # Only a manager-authorized, exclusively owned synchronous server may be
        # ticked. External servers retain their existing clock and settings.
        if self.state["owns_clock"]:
            now = time.monotonic()
            if now < self.next_tick_due:
                self.cancel.wait(min(0.05, self.next_tick_due - now))
                return
            self.world.tick(3.0)
            self.next_tick_due = time.monotonic() + self.tick_interval
        else:
            # New sensor frames wake the loop immediately. Retain at most one
            # pending frame per channel rather than accumulating an image queue.
            self.frame_event.wait(min(0.05, 1 / self.fps))
            self.frame_event.clear()

    def check_frames(self):
        if self.callback_error:
            raise RuntimeError(self.callback_error)
        now = time.monotonic()
        missing = [name for name in self.channels
                   if now - self.last_received.get(name, self.frame_deadline_start) > self.frame_timeout]
        if missing:
            hint = (" 외부 동기 모드 서버는 원래 클라이언트가 tick을 공급해야 합니다."
                    if self.state["synchronous_mode"] and not self.state["owns_clock"]
                    else " CARLA 렌더링 상태와 GPU/서버 로그를 확인하세요.")
            raise RuntimeError("{:g}초 동안 프레임을 받지 못한 센서: ".format(self.frame_timeout) + ", ".join(missing) + "." + hint)

    def cleanup(self):
        with self.lock:
            self.generation += 1
        errors = []
        try:
            self.status(phase="stopping", message="이번 미리보기에서 만든 센서와 차량을 정리합니다.")
        except OSError as error:
            errors.append("정리 상태 파일 기록 실패: " + str(error))
        if self.client is not None:
            try:
                self.client.set_timeout(1.0)
            except Exception as error:
                errors.append("CARLA 정리 timeout 설정 실패: " + str(error))
        errors.extend(self.destroy_sensors())
        if self.vehicle is not None:
            try:
                if self.vehicle.destroy() is False:
                    raise RuntimeError("destroy 반환값 false")
                self.vehicle = None
            except Exception as error:
                errors.append("vehicle {}: {}".format(self.vehicle.id, error))
        return errors

    def execute(self):
        error = None
        try:
            self.connect()
            while not self.stopped():
                self.commands()
                self.advance()
                self.publish()
                self.check_frames()
        except Exception as caught:
            error = str(caught)
        finally:
            try:
                cleanup_errors = self.cleanup()
            except Exception as caught:
                cleanup_errors = [str(caught)]
            if cleanup_errors:
                error = (error + " · " if error else "") + "actor 정리 확인 실패: " + "; ".join(cleanup_errors)
            self.status(status="Failed" if error else "Stopped", phase="stopped", error=error,
                        cleanup_errors=cleanup_errors, message=error or "센서 미리보기를 종료했습니다.")
        return 1 if error else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="SDV real CARLA sensor preview worker")
    parser.add_argument("--run", required=True)
    args = parser.parse_args(argv)
    run = Path(args.run)
    if not run.is_absolute():
        parser.error("--run 경로는 절대 경로여야 합니다.")
    cancel = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda signum, frame: cancel.set())
    try:
        snapshot = json.loads((run / "snapshot.json").read_text(encoding="utf-8"))
        carla, np, Image = load_dependencies()
        return PreviewWorker(run, snapshot, carla, np, Image, cancel).execute()
    except Exception as error:
        log_message("Failed / stopped: " + str(error))
        atomic_json(run / "status.json", {"status": "Failed", "phase": "stopped", "error": str(error),
                    "message": str(error), "updated_at": time.time(), "pid": os.getpid(),
                    "actor_ids": [], "channels": [], "revision": 0})
        return 1


if __name__ == "__main__":
    sys.exit(main())
