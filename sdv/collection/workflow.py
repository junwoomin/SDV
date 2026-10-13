# ────────────────────────────────────────────────────────────────────────────────
#  기본 라이브러리
# ────────────────────────────────────────────────────────────────────────────────
from sdv.runtime.processes import prepare, stop_all, launch_carla, connect_carla
prepare()
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import sys
import time
import yaml
import random
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import numpy.random as npr
import cv2
import carla
import glob
import os
import shutil
import json

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from sdv.runtime.ports import get_random_ports
from sdv.collection.encoding import convert_semantics_to_color, get_depth, depth_outputs
from sdv.collection.carla_engine.map_utils import encode_npy_to_pil
from sdv.collection.carla_engine.main import World, get_nearby_lights
from sdv.collection.driving_runtime import DrivingRuntime, driving_settings
import torch
# ────────────────────────────────────────────────────────────────────────────────
#  전역 상수
# ────────────────────────────────────────────────────────────────────────────────

from sdv.telemetry import configured_path, emit
from sdv.collection.progress import CollectionProgress, save_image
from sdv.collection.metadata import MetadataRecorder, aligned_packets, collection_schema, atomic_json, assert_world_frame
from sdv.collection.bev_adapter import (CarlaBEVAdapter, configured_settings, save_ground_truth,
                                         compatibility_images, legacy_route_pixels,
                                         validate_global_maps, annotate_scene_map)
OUTPUT_ROOT = configured_path("SDV_COLLECTION_OUTPUT", "data")
RUN_TAG=time.strftime('%Y%m%d_%H%M%S')
FPS          = max(1, min(60, int(os.environ.get("SDV_FPS", "30"))))
WEATHER_SWAP_INTERVAL = 2_000          # tick 단위
TICK_DT               = 1.0 / FPS
def location_to_dict(loc):
    return {"x": loc.x, "y": loc.y, "z": loc.z} if loc else None

def control_to_dict(ctrl):
    return {
        "throttle": ctrl.throttle,
        "steer": ctrl.steer,
        "brake": ctrl.brake
    }
def set_next_destination(agent, spawn_points):
    explicit = os.environ.get('SDV_DESTINATION_INDEX')
    if explicit is not None:
        index = int(explicit)
        if not 0 <= index < len(spawn_points):
            raise ValueError('SDV_DESTINATION_INDEX is outside the map spawn list')
        candidates = [spawn_points[index]]
    else:
        ego_location = agent.ego.get_location()
        candidates = [point for point in spawn_points if point.location.distance(ego_location) >= 30]
        random.shuffle(candidates)
    for point in candidates[:30]:
        try:
            agent.set_destination(point.location)
            return
        except (RuntimeError, ValueError):
            if explicit is not None:
                raise
    agent.emergency_stop('no_route')
    raise RuntimeError('No valid destination and global route could be generated')

# CARLA 날씨 프리셋
WEATHERS = [
    carla.WeatherParameters.ClearNoon,
    carla.WeatherParameters.CloudyNoon,
    carla.WeatherParameters.WetNoon,
    carla.WeatherParameters.WetCloudyNoon,
    carla.WeatherParameters.MidRainyNoon,
    carla.WeatherParameters.HardRainNoon,
    carla.WeatherParameters.SoftRainNoon,
    carla.WeatherParameters.ClearSunset,
    carla.WeatherParameters.CloudySunset,
    carla.WeatherParameters.WetSunset,
    carla.WeatherParameters.WetCloudySunset,
    carla.WeatherParameters.MidRainSunset,
    carla.WeatherParameters.HardRainSunset,
    carla.WeatherParameters.SoftRainSunset,
]

# ────────────────────────────────────────────────────────────────────────────────
#  설정 로드 및 CARLA 서버 기동
# ────────────────────────────────────────────────────────────────────────────────
def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def scene_name(idx: int) -> str:
    return f"scene_{RUN_TAG}_{idx:04d}"

def make_scene_dirs(town: str, scene: str, cam_names: List[str],
                    lidar_on: bool, od: bool, seg: bool, depth: bool, bev_adapter=None) -> Path:
    """
    data/{town}/{scene}/ 이하에 모든 하위 폴더를 만든다.
    반환값: base_scene(Path)
    """
    from sdv.collection.metadata import collection_schema
    base_scene = OUTPUT_ROOT / town / scene
    (base_scene / "das").mkdir(parents=True, exist_ok=True)
    (base_scene / "lane").mkdir(parents=True, exist_ok=True)
    (base_scene / "stopline").mkdir(parents=True, exist_ok=True)
    (base_scene / "route").mkdir(parents=True, exist_ok=True)
    (base_scene / "light_topview").mkdir(parents=True, exist_ok=True)
    (base_scene / "walker_mask").mkdir(parents=True, exist_ok=True)
    (base_scene / "vehicle_mask").mkdir(parents=True, exist_ok=True)
    (base_scene / "info").mkdir(parents=True, exist_ok=True)

    if lidar_on:
        (base_scene / "lidar").mkdir(parents=True, exist_ok=True)

    from sdv.config.validation import read_sensor_config
    snapshot = read_sensor_config()
    bev_settings = configured_settings(snapshot)
    if bev_settings['mode'] in ('height_aware', 'global_raster'):
        (base_scene / 'bev_gt').mkdir(parents=True, exist_ok=True)
        (base_scene / 'bev').mkdir(parents=True, exist_ok=True)
    for cam in cam_names:
        camera_options = snapshot['sensors'].get(cam, {'od': od, 'seg': seg, 'depth': depth})
        (base_scene / f"{cam}_rgb").mkdir(parents=True, exist_ok=True)
        if camera_options.get('od'):
            (base_scene / f"{cam}_bbs").mkdir(parents=True, exist_ok=True)
        if camera_options.get('seg'):
            (base_scene / f"{cam}_seg").mkdir(parents=True, exist_ok=True)
            (base_scene / f"{cam}_seg_ids").mkdir(parents=True, exist_ok=True)
        if camera_options.get('depth'):
            (base_scene / f"{cam}_depth").mkdir(parents=True, exist_ok=True)
            (base_scene / f"{cam}_depth_m").mkdir(parents=True, exist_ok=True)

    (base_scene / 'sensor_config.yaml').write_text(
        yaml.safe_dump(snapshot, allow_unicode=True), encoding='utf-8')
    (base_scene / 'collection_manifest.json').write_text(json.dumps({
        **collection_schema(bev_settings), 'run': RUN_TAG, 'town': town, 'scene': scene,
        'fps': FPS, 'save_every_ticks': 2, 'rendering': {'profile': os.environ.get('SDV_CARLA_RENDER_PROFILE','external-unspecified'), 'no_rendering_mode': False, 'graphics_api':'Vulkan'}, 'driving': driving_settings(snapshot.get('driving')),
        'control_semantics': 'state/sensors at frame F; command submitted at F for physics F+1; previous_vehicle_control produced state F', 'vehicle': os.environ.get('SDV_VEHICLE', 'vehicle.tesla.model3'),
        'weather': os.environ.get('SDV_WEATHER', 'cycle'),
        'note': 'Independent binary masks preserve overlaps; versioned numeric BEV files and their per-frame metadata are authoritative where present.'
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    annotate_scene_map(base_scene, bev_adapter)

    return base_scene

def launch_carla_server(carla_root: Path, port: int, town="Town01"):
    return launch_carla(carla_root,port,town)


# ────────────────────────────────────────────────────────────────────────────────
#  스폰 헬퍼
# ────────────────────────────────────────────────────────────────────────────────
def get_blueprints(world: carla.World, pattern: str, generation: str = "All") -> List:
    """세대 필터까지 고려한 블루프린트 목록 반환"""
    bps = world.get_blueprint_library().filter(pattern)
    if generation.lower() == "all" or len(bps) == 1:
        return bps
    try:
        gen = int(generation)
        return [bp for bp in bps if int(bp.get_attribute("generation")) == gen]
    except ValueError:
        print(f"[WARN] 잘못된 generation='{generation}' → 전체 사용")
        return bps

def save_xyxycls_npy(bbs, out_dir="bbox_npy", filename="000001.npy"):


    np.save(os.path.join(out_dir, filename), np.asarray(bbs, dtype=np.float32).reshape(-1, 5), allow_pickle=False)

def spawn_vehicles(
    client,
    world: carla.World,
    tm: carla.TrafficManager,
    num_veh: int,
    ownership=None,

) -> List[int]:
    """랜덤 자동차 스폰 + 오토파일럿 활성 / actor_ids 반환"""
    blueprints = get_blueprints(world, "vehicle.*")
    blueprints = [
        bp
        for bp in blueprints
        if int(bp.get_attribute("number_of_wheels")) == 4
        and not bp.id.endswith(
            (
                "microlino",
                "carlacola",
                "cybertruck",
                "t2",
                "sprinter",
                "firetruck",
                "ambulance",
                "bus",
            )
        )
    ]
    spawn_points = world.get_map().get_spawn_points()
    npr.shuffle(spawn_points)
    num_veh = min(num_veh, len(spawn_points))
    batch = []
    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor

    for transform in spawn_points[:num_veh]:
        bp = random.choice(blueprints)
        if bp.has_attribute("color"):
            bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
        if bp.has_attribute("driver_id"):
            bp.set_attribute("driver_id", random.choice(bp.get_attribute("driver_id").recommended_values))
        bp.set_attribute("role_name", "autopilot")
        batch.append(SpawnActor(bp, transform)
            .then(SetAutopilot(FutureActor, True, tm.get_port())))
    resp = client.apply_batch_sync(batch, True)
    ids = [r.actor_id for r in resp if not r.error]
    if ownership: ownership.own(ids)
    for a in world.get_actors(ids):
        tm.update_vehicle_lights(a, True)
    return ids


def spawn_walkers(
        client,
    world: carla.World,
    num_walk: int,
    run_ratio: float = 0.0,
    cross_ratio: float = 0.0,
    ownership=None,
) -> Tuple[List[int], List[int]]:
    """보행자 Object + 컨트롤러 스폰 / (walker_ids, controller_ids)"""
    walker_bps = get_blueprints(world, "walker.pedestrian.*")
    Spawn = carla.command.SpawnActor

    spawn_pts, speeds = [], []
    for _ in range(num_walk):
        loc = world.get_random_location_from_navigation()
        if not loc:
            continue
        pt = carla.Transform(loc)
        bp = random.choice(walker_bps)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        if bp.has_attribute("speed"):
            speed_vals = bp.get_attribute("speed").recommended_values
            speeds.append(speed_vals[2] if npr.rand() < run_ratio else speed_vals[1])
        else:
            speeds.append(0.0)
        spawn_pts.append((bp, pt))

    # spawn walkers
    walkers, controllers = [], []
    batch = [Spawn(bp, tf) for bp, tf in spawn_pts]
    for r in client.apply_batch_sync(batch, True):
        if not r.error:
            walkers.append(r.actor_id)
    if ownership: ownership.own(walkers)

    # spawn controllers
    ctrl_bp = world.get_blueprint_library().find("controller.ai.walker")
    batch = [Spawn(ctrl_bp, carla.Transform(), wid) for wid in walkers]
    for r in client.apply_batch_sync(batch, True):
        if not r.error:
            controllers.append(r.actor_id)
    if ownership: ownership.own(controllers)

    # init controllers
    all_actors = world.get_actors(walkers + controllers)
    world.set_pedestrians_cross_factor(cross_ratio)
    for idx in range(0, len(controllers)):
        ctrl = all_actors.find(controllers[idx])
        walker = all_actors.find(walkers[idx])
        ctrl.start()
        ctrl.go_to_location(world.get_random_location_from_navigation())
        ctrl.set_max_speed(float(speeds[idx]))
    return walkers, controllers

# ────────────────────────────────────────────────────────────────────────────────
#  데이터 디렉터리 준비
# ────────────────────────────────────────────────────────────────────────────────
def make_data_dirs(town: str, cam_names: List[str], lidar_on: bool,od: bool, seg: bool, depth: bool) -> None:
    base = Path("data") / town
    (base / "das").mkdir(parents=True, exist_ok=True)
    (base / "lane").mkdir(parents=True, exist_ok=True)
    (base / "stopline").mkdir(parents=True, exist_ok=True)
    (base / "route").mkdir(parents=True, exist_ok=True)
    (base / "light_topview").mkdir(parents=True, exist_ok=True)
    (base / "walker_mask").mkdir(parents=True, exist_ok=True)
    (base / "vehicle_mask").mkdir(parents=True, exist_ok=True)
    (base / "info").mkdir(parents=True, exist_ok=True)



    if lidar_on:
        (base / "lidar").mkdir(parents=True, exist_ok=True)
    for cam in cam_names:
        (base / f"{cam}_rgb").mkdir(parents=True, exist_ok=True)
        if od:
            (base / f"{cam}_bbs").mkdir(parents=True, exist_ok=True)
        if seg:
            (base / f"{cam}_seg").mkdir(parents=True, exist_ok=True)
        if depth:
            (base / f"{cam}_depth").mkdir(parents=True, exist_ok=True)

town_npc_settings = {
    "Town01": (30, 20),  # (차량 수, 보행자 수)
    "Town02": (30, 25),
    "Town03": (40, 25),
    "Town04": (50, 30),
    "Town05": (50, 30),
    "Town06": (25, 15),
    "Town07": (20, 10),
    "Town10": (40, 30),
}

# ────────────────────────────────────────────────────────────────────────────────
#  주 루프
# ────────────────────────────────────────────────────────────────────────────────
def main():
    # ── 1) 설정 로드 ───────────────────────────────────────────────────────────

    from sdv.config.validation import read_sensor_config
    cfg = read_sensor_config()
    bev_settings = configured_settings(cfg)
    driving = driving_settings(cfg.get('driving'))
    seed = int(driving.get('seed', 2026))
    random.seed(seed)
    npr.seed(seed)
    policy = None
    if driving['agent'] == 'roach':
        # Fail before starting a server or moving a vehicle when weights are absent.
        from sdv.driving.policy import RoachPolicy
        policy = RoachPolicy(checkpoint=driving['checkpoint'], device=driving['device'],
                             deterministic=bool(driving['deterministic']))
        emit('driving_policy_loaded', policy=policy.metadata)

    towns: List[str]  = cfg["towns"]
    sensor_cfg: Dict  = cfg["sensors"]
    cam_sensors = {k: v for k, v in sensor_cfg.items() if "_cam" in k}
    per_town_samples: int = cfg["data_num"]
    validate_global_maps(towns, bev_settings)

    digit_width = max(5, len(str(per_town_samples)) + 1)

    # ── 2) CARLA 서버 기동 ───────────────────────────────────────────────────────
    PORT, TM_PORT = get_random_ports()
    carla_root = Path.cwd() / "carla"
    os.environ["CARLA_ROOT"]   = str(carla_root)
    os.environ["CARLA_SERVER"] = str(carla_root / "CarlaUE4.sh")

    from sdv.collection.session import connect_external, CollectionSession
    from sdv.runtime.processes import CLEANUPS
    external = os.environ.get("SDV_CARLA_MODE", "managed") == "external"
    if external:
        server_proc = None
        client, initial_world = connect_external()
        TM_PORT = int(os.environ.get("SDV_CARLA_TM_PORT", "8000"))
    else:
        server_proc = launch_carla_server(carla_root, PORT, towns[0])
        client, initial_world = connect_carla(server_proc, PORT, 75)
    tm = client.get_trafficmanager(TM_PORT)
    session = CollectionSession(client, tm, server_proc)
    CLEANUPS.append(session.close)
    progress = CollectionProgress(towns, per_town_samples)
    emit("collection_started", towns=towns, **progress.snapshot(),
         output=str(OUTPUT_ROOT), mode="external" if external else "managed")
    tm.set_random_device_seed(seed)
    tm.set_global_distance_to_leading_vehicle(2.5)
    tm.set_synchronous_mode(True)
    tm.set_hybrid_physics_mode(True)
    tm.set_respawn_dormant_vehicles(True)
    world = client.get_world()
    # Do not remove actors belonging to other clients, including on an
    # externally managed CARLA server. Only session.own() actors are cleaned up.
    # 센서 이름 정리
    camera_names = [k for k in sensor_cfg.keys() if k != "lidar"]
    lidar_enabled = "lidar" in sensor_cfg
    # ── 3) Town 루프 ────────────────────────────────────────────────────────────
    for town in towns:
        emit("collection_town_started", **progress.snapshot(town),
             message="{} 준비 중 · 전체 {} / {} 샘플 저장".format(
                 town, progress.snapshot()["saved"], progress.snapshot()["total"]))
        if town in town_npc_settings:
            NUM_VEHICLES, NUM_WALKERS = town_npc_settings[town]
        else:
            NUM_VEHICLES, NUM_WALKERS = 40, 20
        NUM_VEHICLES=max(0, min(200, int(os.environ.get('SDV_NPC_VEHICLES', str(min(NUM_VEHICLES, 4))))))
        NUM_WALKERS=max(0, min(200, int(os.environ.get('SDV_NPC_WALKERS', str(min(NUM_WALKERS, 2))))))
        scene_idx = 1
        current_scene = scene_name(scene_idx)
        bev_adapter = None
        # [추가] 첫 씬 폴더 생성
        base_scene = make_scene_dirs(
            town, current_scene, camera_names,
            lidar_enabled,
            sensor_cfg[camera_names[0]]['od'],
            sensor_cfg[camera_names[0]]['seg'],
            sensor_cfg[camera_names[0]]['depth'], bev_adapter=bev_adapter,
        )
        # 월드 로드 & 동기화 설정
        emit('collection_stage',stage='Loading Map',town=town)
        client.set_timeout(120.0)
        if client.get_world().get_map().name.split("/")[-1] != town:
            client.load_world(town)
        client.set_timeout(5.0)
        world = client.get_world()
        session.begin_world(world)
        emit('collection_stage',stage='Ego / Sensor Spawn',town=town)
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = TICK_DT
        settings.no_rendering_mode = False
        settings.substepping = True
        settings.max_substep_delta_time = min(.01, TICK_DT)
        settings.max_substeps = max(1, int(np.ceil(TICK_DT/settings.max_substep_delta_time)))
        world.apply_settings(settings)
        bev_adapter = None
        if bev_settings['mode'] in ('height_aware', 'global_raster'):
            from sdv.paths import RUNTIME
            map_label = 'Global PNG BEV 지도' if bev_settings['mode'] == 'global_raster' else 'Height-aware global BEV cache'
            emit('collection_map_preparing', town=town, message=map_label + ' 준비 중')
            bev_adapter = CarlaBEVAdapter(world, os.environ.get('SDV_BEV_CACHE', str(RUNTIME / 'bev')), bev_settings)
            annotate_scene_map(base_scene, bev_adapter)
            emit('collection_map_ready', town=town, message=map_label + ' 준비 완료')
        weather_name = os.environ.get("SDV_WEATHER", "cycle")
        if weather_name != "cycle":
            weather = getattr(carla.WeatherParameters, weather_name, None)
            if weather is None: raise ValueError(f"지원하지 않는 날씨: {weather_name}")
            world.set_weather(weather)

        world.set_pedestrians_seed(seed)

        # 액터 스폰
        vehicle_ids = spawn_vehicles(client,world, tm, NUM_VEHICLES, ownership=session)
        session.own(vehicle_ids)
        walker_ids, ctrl_ids = spawn_walkers(client,world, NUM_WALKERS, ownership=session)
        session.own(walker_ids + ctrl_ids)


        # Ego 월드/에이전트 생성
        ego_world = World(client, sensor_cfg, server_job=server_proc, bev_mode=bev_settings['mode'])
        spawn_pts = ego_world.map.get_spawn_points()
        def create_agent(ego):
            runtime = DrivingRuntime(world, ego.player, driving, fixed_delta_seconds=TICK_DT,
                collision_sensor=ego.collision_sensor, lane_sensor=ego.lane_invasion_sensor,
                traffic_manager=tm, bev_settings=bev_settings, policy=policy)
            CLEANUPS.insert(0, runtime.close)
            set_next_destination(runtime, spawn_pts)
            return runtime
        agent = create_agent(ego_world)
        # 루프 변수
        tick_cnt   = 0
        sample_idx = 0
        metadata_recorder = MetadataRecorder()
        consecutive_sync_failures = 0

        try:
            while sample_idx < int(per_town_samples):

                tick_cnt += 1
                native_frame = world.tick()
                snapshot = world.get_snapshot()
                if snapshot.frame != native_frame:
                    agent.emergency_stop('world_frame_advanced')
                    raise RuntimeError('Another CARLA client advanced the world; collection stopped to preserve frame alignment')
                metadata_recorder.observe_signals(ego_world.traffic_lights, native_frame)

                if agent.done():
                    atomic_json(base_scene / 'events' / ('arrival_{}.json'.format(native_frame)),
                        dict(event='destination_reached', carla_frame=native_frame,
                             metrics=agent.metrics, navigation=agent.last_navigation))
                    set_next_destination(agent, spawn_pts)
                try:
                    ctrl, info_dict = agent.run_step(snapshot)
                except BaseException:
                    atomic_json(base_scene / 'events' / ('driving_failure_{}.json'.format(native_frame)),
                        dict(event='driving_failure', carla_frame=native_frame, policy=agent.last_policy,
                             navigation=agent.last_navigation, metrics=agent.metrics))
                    raise
                emit('driving_status', carla_frame=native_frame, timestamp_s=snapshot.timestamp.elapsed_seconds,
                     policy=agent.last_policy, navigation=agent.last_navigation, metrics=agent.metrics,
                     speed_kmh=info_dict['current_speed'], control=info_dict['next_control'],
                     collision=info_dict['collision'], applied_control=info_dict['next_control'],
                     traffic=dict(signal_state=agent.last_navigation.get('traffic_light_state'), collision=info_dict['collision']))
                if info_dict.get("collision", False):
                    atomic_json(base_scene / 'events' / ('collision_{}.json'.format(native_frame)), {
                        'schema_version': 1, 'sequence_id': '{}/{}'.format(town, current_scene), 'carla_frame': native_frame,
                        'timestamp_s': snapshot.timestamp.elapsed_seconds, 'event': 'collision_respawn',
                        'source': 'CARLA collision sensor; image sample may not be saved for terminal tick',
                        'collisions': list(getattr(ego_world.collision_sensor, 'events', [])),
                        'policy': agent.last_policy, 'metrics': agent.metrics})
                    agent.close()
                    if agent.close in CLEANUPS: CLEANUPS.remove(agent.close)
                    ego_world.destroy()
                    ego_world = World(client, sensor_cfg, server_job=server_proc, bev_mode=bev_settings['mode'])
                    agent = create_agent(ego_world)
                    scene_idx += 1
                    current_scene = scene_name(scene_idx)
                    base_scene = make_scene_dirs(
                        town, current_scene, camera_names,
                        lidar_enabled,
                        sensor_cfg[camera_names[0]]['od'],
                        sensor_cfg[camera_names[0]]['seg'],
                        sensor_cfg[camera_names[0]]['depth'], bev_adapter=bev_adapter,
                    )

                    time.sleep(0.2)
                    continue

                # ── 3-1) 날씨 변경 + Ego 재스폰 ────────────────────────────────
                elif weather_name == "cycle" and tick_cnt % WEATHER_SWAP_INTERVAL == 0:
                    new_weather = random.choice(WEATHERS)

                    ego_world.world.set_weather(new_weather)
                    scene_idx += 1
                    current_scene = scene_name(scene_idx)
                    base_scene = make_scene_dirs(
                        town, current_scene, camera_names,
                        lidar_enabled,
                        sensor_cfg[camera_names[0]]['od'],
                        sensor_cfg[camera_names[0]]['seg'],
                        sensor_cfg[camera_names[0]]['depth'], bev_adapter=bev_adapter,
                    )

                    # 기존 Ego + 센서 파괴
                    agent.close()
                    if agent.close in CLEANUPS: CLEANUPS.remove(agent.close)
                    ego_world.destroy()
                    # 새 월드(센서 포함) 재생성
                    ego_world = World(client, sensor_cfg, server_job=server_proc, bev_mode=bev_settings['mode'])
                    agent = create_agent(ego_world)
                    time.sleep(0.1)
                    continue

                # ── 3-2) 데이터 저장 (격프레임) ────────────────────────────────
                if tick_cnt % 2 == 0:

                    required_keys = [
                        "ego_vehicle_loc", "current_speed", "current_waypoint",
                        "target_waypoint", "global_waypoints", "current_control",
                        "next_control", "traffic_light", "traffic_light_20", "vehicle_in_front", "collision",
                        "ego_vehicle_rot"

                    ]
                    if not all(k in info_dict and info_dict[k] is not None for k in required_keys):
                        print(f"[경고] 필수 info_dict 데이터 누락 → 저장 생략 (sample_idx: {sample_idx})")
                        continue

                    sensor_wrappers = {}
                    for cam in camera_names:
                        sensor_wrappers[cam + '_rgb'] = getattr(ego_world, cam + '_rgb')
                        if sensor_cfg[cam].get('seg') or sensor_cfg[cam].get('od'):
                            sensor_wrappers[cam + '_seg'] = getattr(ego_world, cam + '_seg')
                        if sensor_cfg[cam].get('depth') or sensor_cfg[cam].get('od'):
                            sensor_wrappers[cam + '_depth'] = getattr(ego_world, cam + '_depth')
                    sensor_wrappers['imu'] = ego_world.imu_sensor
                    if lidar_enabled:
                        sensor_wrappers['lidar'] = ego_world.lidar_sensor
                    packets, missing_sensors = aligned_packets(sensor_wrappers, native_frame)
                    if missing_sensors:
                        consecutive_sync_failures += 1
                        emit('collection_warning', reason='sensor_frame_mismatch', carla_frame=native_frame,
                             sensors=missing_sensors, message='Current simulator frame did not arrive from every sensor; frame skipped.')
                        agent.emergency_stop('sensor_frame_mismatch')
                        raise RuntimeError('센서 Frame 동기화 실패로 안전 정지했습니다: {}. Sensor tick을 0으로 설정하거나 모든 센서 주기를 맞추세요.'.format(', '.join(missing_sensors)))
                    consecutive_sync_failures = 0
                    assert_world_frame(world, native_frame)

                    fname = f"{sample_idx:0{digit_width}d}_a"

                    bev_result = None
                    if bev_adapter is None:
                        bev_arr, light_birdview, walker_view, vehicle_view,stopline_birdview,route_birdview = ego_world.render_BEV(agent.future_waypoints,info_dict["global_waypoints"])
                    else:
                        bev_result = bev_adapter.generate(ego_world.player, snapshot, agent)
                        pose = bev_adapter.last_ego_record
                        ego_world.global_waypoints = legacy_route_pixels(info_dict['global_waypoints'], pose, bev_settings)
                        anchor = [[bev_settings['width'] * bev_settings['anchor_x'], bev_settings['height'] * bev_settings['anchor_y']]]
                        ego_world.ways_torch = [anchor] + legacy_route_pixels(agent.future_waypoints, pose, bev_settings)


                    json_info = {
                            "ego_vehicle_loc":info_dict["ego_vehicle_loc"],
                            "current_speed": info_dict["current_speed"],
                            "current_waypoint": location_to_dict(info_dict["current_waypoint"]),
                            "global_waypoints": ego_world.global_waypoints.tolist() if isinstance(ego_world.global_waypoints, torch.Tensor) else ego_world.global_waypoints,
                            "bev_future_waypoints": ego_world.ways_torch.tolist() if isinstance(ego_world.ways_torch, torch.Tensor) else ego_world.ways_torch,


                            "current_control": info_dict["current_control"],
                            "next_control": info_dict["next_control"],
                            "traffic_light":info_dict["traffic_light"],
                            "traffic_light_20":info_dict["traffic_light_20"],
                            "traffic_light_distance":info_dict["traffic_light_distance"],

                            "vehicle_in_front":info_dict["vehicle_in_front"],
                            "ego_vehicle_rot":info_dict["ego_vehicle_rot"],
                            "collision":info_dict["collision"],
                            "imu": packets['imu']['data']
                        }
                    json_info['sdv_metadata'] = metadata_recorder.capture(
                        world, ego_world.player, agent, base_scene, fname, snapshot, packets,
                        applied_control=ctrl, carla_map=ego_world.map)
                    json_info['sdv_metadata']['policy'] = dict(agent.last_policy)
                    json_info['sdv_metadata']['navigation'].update(agent.last_navigation)
                    json_info['sdv_metadata']['driving_metrics'] = dict(agent.metrics)
                    json_info['sdv_metadata']['ego']['previous_vehicle_control'] = agent.last_policy.get('previous_vehicle_control')
                    if agent.mode == 'autopilot':
                        json_info['sdv_metadata']['ego']['applied_control_semantics'] = agent.last_policy['autopilot_command_semantics']
                    if agent.last_observation is not None:
                        observation_dir = base_scene / 'roach_observation'
                        observation_dir.mkdir(parents=True, exist_ok=True)
                        observation_file = observation_dir / (fname + '.npz')
                        np.savez_compressed(observation_file, birdview=agent.last_observation['birdview'],
                            state=agent.last_observation['state'], frame=np.int64(native_frame),
                            timestamp_s=np.float64(snapshot.timestamp.elapsed_seconds))
                        json_info['sdv_metadata']['policy']['observation_file'] = 'roach_observation/' + observation_file.name
                    if bev_result is not None:
                        json_info['bev_gt'] = save_ground_truth(base_scene, fname, bev_result)
                        json_info['sdv_metadata']['bev_gt'] = json_info['bev_gt']
                        bev_images = compatibility_images(bev_result)
                    else:
                        bev_images = {
                            'das': np.array(ego_world.renderer.visualize_das_view(bev_arr)),
                            'lane': np.array(ego_world.renderer.visualize_lane_view(bev_arr)),
                            'walker_mask': np.array(ego_world.renderer.visualize_walker_view(walker_view)),
                            'vehicle_mask': np.array(ego_world.renderer.visualize_vehicle_view(vehicle_view)),
                            'light_topview': np.array(ego_world.renderer.visualize_light_birdview(light_birdview)),
                            'stopline': np.array(ego_world.renderer.visualize_stopline_view(stopline_birdview)),
                            'route': route_birdview}
                    for folder, bev_image in bev_images.items():
                        if bev_image.ndim == 3:
                            bev_image = cv2.cvtColor(bev_image, cv2.COLOR_RGB2BGR)
                        save_image(cv2.imwrite, str(base_scene / folder / f'{fname}.png'), bev_image)

                    if lidar_enabled and ego_world.lidar_sensor.point_cloud is not None:
                        np.save(base_scene / "lidar" / f"{fname}.npy",
                                packets["lidar"]["data"], allow_pickle=False)

                    if any(camera.get('od', False) for camera in cam_sensors.values()):
                        bb_3d = ego_world._get_3d_bbs(max_distance=50)

                    for cam in camera_names:
                        rgb = packets[cam + "_rgb"]["data"]
                        if rgb is not None:
                            save_image(cv2.imwrite, str(base_scene / f"{cam}_rgb" / f"{fname}.png"), rgb)

                        if sensor_cfg[cam].get("od", False):
                            bbs = []
                            seg = packets[cam + "_seg"]["data"]
                            dep = packets[cam + "_depth"]["data"]
                            if seg is not None and dep is not None:
                                ego_world._traffic_lights = get_nearby_lights(ego_world._vehicle, ego_world.traffic_lights)
                                affordances = ego_world._get_affordances()
                                traffic_lights = ego_world._find_obstacle("*traffic_light*")
                                sensor_transform = getattr(ego_world, f"{cam}_seg").sensor.get_transform()
                                _segmentation = np.copy(seg[:, :, 2])
                                camera_data = {'width': sensor_cfg[cam]['width'], 'height': sensor_cfg[cam]['height'], 'fov': sensor_cfg[cam]['fov']}
                                camera_data['calibration'] = ego_world._get_camera_to_car_calibration(camera_data)
                                bbs = ego_world._get_2d_bbs(sensor_transform, affordances, bb_3d, _segmentation, camera_data)
                            save_xyxycls_npy(bbs, str(base_scene / f"{cam}_bbs"), f"{fname}.npy")

                        if sensor_cfg[cam].get("seg", False):
                            seg = packets[cam + "_seg"]["data"]
                            if seg is not None:
                                raw_ids = seg[:, :, 2]
                                save_image(cv2.imwrite, str(base_scene / f"{cam}_seg_ids" / f"{fname}.png"), raw_ids)
                                seg_color = convert_semantics_to_color(raw_ids)
                                save_image(cv2.imwrite, str(base_scene / f"{cam}_seg" / f"{fname}.png"), seg_color)

                        if sensor_cfg[cam].get("depth", False):
                            dep = packets[cam + "_depth"]["data"]
                            if dep is not None:
                                depth_m, depth_preview = depth_outputs(dep[:, :, :3])
                                np.save(base_scene / f"{cam}_depth_m" / f"{fname}.npy", depth_m, allow_pickle=False)
                                save_image(cv2.imwrite, str(base_scene / f"{cam}_depth" / f"{fname}.png"), depth_preview)


                    assert_world_frame(world, native_frame)
                    # JSON is the final frame commit marker, after every image/annotation exists.
                    atomic_json(base_scene / "info" / f"{fname}.json", json_info)

                    # A sample is committed only after all requested files were saved.
                    sample_idx += 1
                    observed = progress.commit(town)
                    print(f'SDV_PROGRESS town={town} saved={sample_idx} target={per_town_samples} '
                          f'global_saved={observed["saved"]} global_target={observed["total"]}', flush=True)
                    emit("collection_progress", **observed, scene=current_scene,
                         sample=fname, carla_frame=native_frame, timestamp_s=json_info["sdv_metadata"]["synchronization"]["timestamp_s"],
                         metadata=json_info["sdv_metadata"], speed_kmh=info_dict["current_speed"],
                         control=info_dict["next_control"], policy=agent.last_policy, navigation=agent.last_navigation, position=info_dict["ego_vehicle_loc"],
                         traffic_light=str(info_dict["traffic_light"]),
                         preview={"camera": str(base_scene / f"{camera_names[0]}_rgb" / f"{fname}.png"),
                                  "bev": str(base_scene / ('bev' if bev_result is not None else 'route') / f"{fname}.png")},
                         info=str(base_scene / "info" / f"{fname}.json"))

        finally:
            # Any storage, sensor or geometry error must release throttle now.
            agent.close()

        # ── 3-3) Town 완료 정리 ──────────────────────────────────────────
        try:

            atomic_json(base_scene / 'driving_metrics.json', agent.metrics)
            agent.close()
            if agent.close in CLEANUPS: CLEANUPS.remove(agent.close)
            ego_world.destroy()
            session.cleanup_world()
        except Exception:
            pass

    # ── 4) 종료 ────────────────────────────────────────────────────────────────

    session.close()
    if session.close in CLEANUPS: CLEANUPS.remove(session.close)
    if server_proc is not None: server_proc.kill()
    emit("collection_completed", **progress.snapshot(towns[-1]), output=str(OUTPUT_ROOT))


# ────────────────────────────────────────────────────────────────────────────────
#  엔트리포인트
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        main()
        # train_ratio = 0.7
        # Towns = glob.glob('data/Town*')
        # for Town in Towns:
        #     town_name = os.path.basename(Town)

        #     subfolders = glob.glob(f'{Town}/*')
        #     subfolders = [folder for folder in subfolders if os.path.isdir(folder)]

        #     # 모든 파일 이름(확장자 포함) 미리 읽어오기
        #     all_files = {folder: os.listdir(folder) for folder in subfolders}

        #     # 공통 파일 번호만 추출
        #     file_numbers = set()
        #     for files in all_files.values():
        #         for file in files:
        #             base = os.path.splitext(file)[0]
        #             file_numbers.add(base)

        #     file_numbers = sorted(file_numbers)
        #     random.shuffle(file_numbers)

        #     split_idx = int(len(file_numbers) * train_ratio)
        #     val_numbers = set(file_numbers[split_idx:])

        #     train_root = os.path.join('data/train', town_name)
        #     val_root = os.path.join('data/val', town_name + '_val')
        #     os.makedirs(train_root, exist_ok=True)
        #     os.makedirs(val_root, exist_ok=True)

        #     for folder in subfolders:
        #         relative_path = os.path.relpath(folder, Town)
        #         train_subfolder = os.path.join(train_root, relative_path)
        #         val_subfolder = os.path.join(val_root, relative_path)
        #         os.makedirs(train_subfolder, exist_ok=True)
        #         os.makedirs(val_subfolder, exist_ok=True)

        #     for folder, files in all_files.items():
        #         relative_path = os.path.relpath(folder, Town)
        #         for file in files:
        #             base = os.path.splitext(file)[0]
        #             src_path = os.path.join(folder, file)
        #             if base in val_numbers:
        #                 dst_folder = os.path.join(val_root, relative_path)
        #             else:
        #                 dst_folder = os.path.join(train_root, relative_path)
        #             dst_path = os.path.join(dst_folder, file)
        #             shutil.move(src_path, dst_path)
        #     shutil.rmtree(Town)

    except KeyboardInterrupt:

        sys.exit(0)
    finally:
        stop_all()
