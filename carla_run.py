# ────────────────────────────────────────────────────────────────────────────────
#  기본 라이브러리
# ────────────────────────────────────────────────────────────────────────────────
import os
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

from utils import (
    get_random_ports,
    convert_semantics_to_color,
    get_depth,
    kill_existing_carla_processes


)
from carla_data_gan.map_utils import encode_npy_to_pil
from carla_data_gan.main import World, BehaviorAgent, get_nearby_lights
import torch
# ────────────────────────────────────────────────────────────────────────────────
#  전역 상수
# ────────────────────────────────────────────────────────────────────────────────

FPS          = 15
WEATHER_SWAP_INTERVAL = 2_000          # tick 단위
TICK_DT               = 0.05
def location_to_dict(loc):
    return {"x": loc.x, "y": loc.y, "z": loc.z} if loc else None

def control_to_dict(ctrl):
    return {
        "throttle": ctrl.throttle,
        "steer": ctrl.steer,
        "brake": ctrl.brake
    }
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
    return f"scene_{idx:04d}"

def make_scene_dirs(town: str, scene: str, cam_names: List[str],
                    lidar_on: bool, od: bool, seg: bool, depth: bool) -> Path:
    """
    data/{town}/{scene}/ 이하에 모든 하위 폴더를 만든다.
    반환값: base_scene(Path)
    """
    base_scene = Path("data") / town / scene
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

    for cam in cam_names:
        (base_scene / f"{cam}_rgb").mkdir(parents=True, exist_ok=True)
        if od:
            (base_scene / f"{cam}_bbs").mkdir(parents=True, exist_ok=True)
        if seg:
            (base_scene / f"{cam}_seg").mkdir(parents=True, exist_ok=True)
        if depth:
            (base_scene / f"{cam}_depth").mkdir(parents=True, exist_ok=True)

    return base_scene

def launch_carla_server(carla_root: Path, port: int) -> subprocess.Popen:
    """백그라운드에서 CARLA 서버를 실행하고 PID 반환"""
    server_bin = carla_root / "CarlaUE4.sh"
    cmd = [
        str(server_bin),
        "-quality_level=Low",
        "-benchmark",
        "-fps=" + str(FPS),
        "-RenderOffScreen",
        "-benchmark",
        f"-carla-world-port={port}",
    ]
    return subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )


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


    np.save(os.path.join(out_dir, filename), np.array(bbs, dtype=np.float32))

def spawn_vehicles(
    client,
    world: carla.World,
    tm: carla.TrafficManager,
    num_veh: int,

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
    for a in world.get_actors(ids):
        tm.update_vehicle_lights(a, True)
    return ids


def spawn_walkers(
        client,
    world: carla.World,
    num_walk: int,
    run_ratio: float = 0.0,
    cross_ratio: float = 0.0,
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

    # spawn controllers
    ctrl_bp = world.get_blueprint_library().find("controller.ai.walker")
    batch = [Spawn(ctrl_bp, carla.Transform(), wid) for wid in walkers]
    for r in client.apply_batch_sync(batch, True):
        if not r.error:
            controllers.append(r.actor_id)

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
    
    cfg  = load_yaml("data/sensor_config.yaml")
    
    towns: List[str]  = cfg["towns"]
    sensor_cfg: Dict  = cfg["sensors"]
    cam_sensors = {k: v for k, v in sensor_cfg.items() if "_cam" in k}
    per_town_samples: int = cfg["data_num"]

    digit_width = max(5, len(str(per_town_samples)) + 1)

    # ── 2) CARLA 서버 기동 ───────────────────────────────────────────────────────
    PORT, TM_PORT = get_random_ports()
    carla_root = Path.cwd() / "carla"
    os.environ["CARLA_ROOT"]   = str(carla_root)
    os.environ["CARLA_SERVER"] = str(carla_root / "CarlaUE4.sh")
    
    server_proc = launch_carla_server(carla_root, PORT)
    time.sleep(4)
    client = carla.Client("localhost", PORT)
    client.set_timeout(60.0)
    tm = client.get_trafficmanager(TM_PORT)
    tm.set_global_distance_to_leading_vehicle(2.5)
    tm.set_synchronous_mode(True)
    tm.set_hybrid_physics_mode(True)
    tm.set_respawn_dormant_vehicles(True)
    world = client.get_world()
    actors = world.get_actors()
    for actor in actors:
        if actor.type_id.startswith("vehicle.") or \
        actor.type_id.startswith("walker.") or \
        actor.type_id.startswith("sensor."):
            actor.destroy()
    # 센서 이름 정리
    camera_names = [k for k in sensor_cfg.keys() if k != "lidar"]
    lidar_enabled = "lidar" in sensor_cfg
    sample_idx_total = 1
    # ── 3) Town 루프 ────────────────────────────────────────────────────────────
    for town in towns:
        if town in town_npc_settings:
            NUM_VEHICLES, NUM_WALKERS = town_npc_settings[town]
        else:
            NUM_VEHICLES, NUM_WALKERS = 40, 20
        scene_idx = 1
        current_scene = scene_name(scene_idx)
        # [추가] 첫 씬 폴더 생성
        base_scene = make_scene_dirs(
            town, current_scene, camera_names,
            lidar_enabled,
            sensor_cfg[camera_names[0]]['od'],
            sensor_cfg[camera_names[0]]['seg'],
            sensor_cfg[camera_names[0]]['depth'],
        )
        # 월드 로드 & 동기화 설정
        client.load_world(town)
        world = client.get_world()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = TICK_DT
        settings.no_rendering_mode = False
        world.apply_settings(settings)

        # 액터 스폰
        vehicle_ids = spawn_vehicles(client,world, tm, NUM_VEHICLES)
        walker_ids, ctrl_ids = spawn_walkers(client,world, NUM_WALKERS)


        # Ego 월드/에이전트 생성
        ego_world = World(client, sensor_cfg)
        agent     = BehaviorAgent(ego_world.player, behavior="normal")
        agent.set_collision_sensor(ego_world.collision_sensor) 
        spawn_pts = ego_world.map.get_spawn_points()
        agent.set_destination(random.choice(spawn_pts).location)
        # 루프 변수
        tick_cnt   = 0
        sample_idx = 0

        while sample_idx < int(per_town_samples) and not os.path.exists("/data") and not os.path.isdir("/data"):

            tick_cnt += 1
            world.tick()

            if agent.done():
                agent.set_destination(random.choice(spawn_pts).location)
            ctrl ,info_dict = agent.run_step()
            ctrl.manual_gear_shift = False
            value = random.uniform(-0.2, 0.2)
            ctrl.steer = ctrl.steer+value
            ego_world.player.apply_control(ctrl)
            if info_dict.get("collision", False):
                ego_world.destroy()
                ego_world = World(client, sensor_cfg)
                agent = BehaviorAgent(ego_world.player, behavior="normal")
                agent.set_collision_sensor(ego_world.collision_sensor) 
                scene_idx += 1
                current_scene = scene_name(scene_idx)
                base_scene = make_scene_dirs(
                    town, current_scene, camera_names,
                    lidar_enabled,
                    sensor_cfg[camera_names[0]]['od'],
                    sensor_cfg[camera_names[0]]['seg'],
                    sensor_cfg[camera_names[0]]['depth'],
                )

                agent.set_destination(random.choice(spawn_pts).location)
                time.sleep(0.2)
                continue  

            # ── 3-1) 날씨 변경 + Ego 재스폰 ────────────────────────────────
            elif tick_cnt % WEATHER_SWAP_INTERVAL == 0:
                new_weather = random.choice(WEATHERS)

                ego_world.world.set_weather(new_weather)
                scene_idx += 1
                current_scene = scene_name(scene_idx)
                base_scene = make_scene_dirs(
                    town, current_scene, camera_names,
                    lidar_enabled,
                    sensor_cfg[camera_names[0]]['od'],
                    sensor_cfg[camera_names[0]]['seg'],
                    sensor_cfg[camera_names[0]]['depth'],
                )

                # 기존 Ego + 센서 파괴
                ego_world.destroy()
                # 새 월드(센서 포함) 재생성
                ego_world = World(client, sensor_cfg)
                agent     = BehaviorAgent(ego_world.player, behavior="normal")
                agent.set_collision_sensor(ego_world.collision_sensor) 
                agent.set_destination(random.choice(spawn_pts).location)
                time.sleep(0.1)

            # ── 3-2) 데이터 저장 (격프레임) ────────────────────────────────
            if tick_cnt % 2 == 0:

                required_keys = [
                    "ego_vehicle_loc", "current_speed", "current_waypoint",
                    "target_waypoint", "global_waypoints", "current_control",
                    "next_control", "traffic_light", "traffic_light_20","traffic_light_distance", "vehicle_in_front", "collision",
                    "ego_vehicle_rot"
                    
                ]
                if not all(k in info_dict and info_dict[k] is not None for k in required_keys):
                    print(f"[경고] 필수 info_dict 데이터 누락 → 저장 생략 (sample_idx: {sample_idx})")
                    continue

                cam_ready = True
                for cam in camera_names:
                    if getattr(ego_world, f"{cam}_rgb").img is None:
                        cam_ready = False
                        break
                    if sensor_cfg[cam].get("seg", False) and getattr(ego_world, f"{cam}_seg").img is None:
                        cam_ready = False
                        break
                    if sensor_cfg[cam].get("depth", False) and getattr(ego_world, f"{cam}_depth").img is None:
                        cam_ready = False
                        break

                if lidar_enabled and ego_world.lidar_sensor.point_cloud is None:
                    cam_ready = False

                if not cam_ready:
                    print(f"[경고] 센서 데이터 누락 → 저장 생략 (sample_idx: {sample_idx})")
                    continue


                fname = f"{sample_idx:0{digit_width}d}_a"

                bev_arr, light_birdview, walker_view, vehicle_view,stopline_birdview,route_birdview = ego_world.render_BEV(agent.future_waypoints,info_dict["global_waypoints"])
                
                
                if len(ego_world.global_waypoints) < 6:
                    new_weather = random.choice(WEATHERS)

                    ego_world.world.set_weather(new_weather)
                    scene_idx += 1
                    current_scene = scene_name(scene_idx)
                    base_scene = make_scene_dirs(
                        town, current_scene, camera_names,
                        lidar_enabled,
                        sensor_cfg[camera_names[0]]['od'],
                        sensor_cfg[camera_names[0]]['seg'],
                        sensor_cfg[camera_names[0]]['depth'],
                    )

                    # 기존 Ego + 센서 파괴
                    ego_world.destroy()
                    # 새 월드(센서 포함) 재생성
                    ego_world = World(client, sensor_cfg)
                    agent     = BehaviorAgent(ego_world.player, behavior="normal")
                    agent.set_collision_sensor(ego_world.collision_sensor) 
                    agent.set_destination(random.choice(spawn_pts).location)
                    time.sleep(0.1)
                    print(f"경로 6개 미만으로 종료")
                    continue
                
                sample_idx += 1
                sample_idx_total += 1  
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
                        "imu": ego_world.imu_sensor.imu_data if ego_world.imu_sensor.imu_data else {
                            "accelerometer": None,
                            "gyroscope": None,
                            "compass": None
                        }
                    }
                with open(base_scene / "info" / f"{fname}.json", "w") as f:
                    json.dump(json_info, f, indent=2)

                das_img     = np.array(ego_world.renderer.visualize_das_view(bev_arr))
                lane_img    = np.array(ego_world.renderer.visualize_lane_view(bev_arr))
                walker_img  = np.array(ego_world.renderer.visualize_walker_view(walker_view))
                vehicle_img = np.array(ego_world.renderer.visualize_vehicle_view(vehicle_view))
                light_img   = np.array(ego_world.renderer.visualize_light_birdview(light_birdview))
                stopline_img= np.array(ego_world.renderer.visualize_stopline_view(stopline_birdview))

                cv2.imwrite(str(base_scene / "light_topview" / f"{fname}.png"), cv2.cvtColor(light_img, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(base_scene / "vehicle_mask" / f"{fname}.png"), vehicle_img)
                cv2.imwrite(str(base_scene / "walker_mask" / f"{fname}.png"),  walker_img)
                cv2.imwrite(str(base_scene / "das" / f"{fname}.png"),           das_img)
                cv2.imwrite(str(base_scene / "lane" / f"{fname}.png"),          lane_img)
                cv2.imwrite(str(base_scene / "stopline" / f"{fname}.png"),      stopline_img)
                cv2.imwrite(str(base_scene / "route" / f"{fname}.png"),         route_birdview)

                if lidar_enabled and ego_world.lidar_sensor.point_cloud is not None:
                    np.save(base_scene / "lidar" / f"{fname}.npy",
                            ego_world.lidar_sensor.point_cloud, allow_pickle=True)

                if list(cam_sensors.items())[0][1]['od']:
                    bb_3d = ego_world._get_3d_bbs(max_distance=50)

                for cam in camera_names:
                    rgb = getattr(ego_world, f"{cam}_rgb").img
                    if rgb is not None:
                        cv2.imwrite(str(base_scene / f"{cam}_rgb" / f"{fname}.png"), rgb)

                    if sensor_cfg[cam].get("od", False):
                        seg = getattr(ego_world, f"{cam}_seg").img
                        dep = getattr(ego_world, f"{cam}_depth").img
                        if seg is not None and dep is not None:
                            ego_world._traffic_lights = get_nearby_lights(ego_world._vehicle, ego_world.traffic_lights)
                            affordances = ego_world._get_affordances()
                            traffic_lights = ego_world._find_obstacle("*traffic_light*")
                            sensor_transform = getattr(ego_world, f"{cam}_seg").sensor.get_transform()
                            _segmentation = np.copy(seg[:, :, 2])
                            ego_world._change_seg_tl(sensor_transform, _segmentation, get_depth(dep[:, :, :3]), traffic_lights)
                            bbs = ego_world._get_2d_bbs(sensor_transform, affordances, bb_3d, _segmentation)
                        save_xyxycls_npy(bbs, str(base_scene / f"{cam}_bbs"), f"{fname}.npy")

                    if sensor_cfg[cam].get("seg", False):
                        seg = getattr(ego_world, f"{cam}_seg").img
                        if seg is not None:
                            seg_color = convert_semantics_to_color(seg[:, :, 2])
                            cv2.imwrite(str(base_scene / f"{cam}_seg" / f"{fname}.png"), seg_color)

                    if sensor_cfg[cam].get("depth", False):
                        dep = getattr(ego_world, f"{cam}_depth").img
                        if dep is not None:
                            depth_img = get_depth(dep[:, :, :3])
                            cv2.imwrite(str(base_scene / f"{cam}_depth" / f"{fname}.png"), depth_img)


        # ── 3-3) Town 완료 정리 ──────────────────────────────────────────
        try:

            ego_world.destroy()
            actors_to_destroy = vehicle_ids + walker_ids + ctrl_ids
            #actors_to_destroy = vehicle_ids 
            
            client.apply_batch([carla.command.DestroyActor(x) for x in actors_to_destroy])
        except Exception:
            pass
        
    # ── 4) 종료 ────────────────────────────────────────────────────────────────

    server_proc.kill()


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
