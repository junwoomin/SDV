"""Live preview vehicle and sensor ownership; no UI rendering."""
import carla
from sdv.collection.sensor_attributes import configure_camera_blueprint, configure_lidar_blueprint
import copy
import random
import time
import weakref
import numpy as np

class World:
    def __init__(self, client, conf, server_job=None):

        self.client = client
        self.server_job = server_job
        self.world = client.get_world()
        self.map = self.world.get_map()
        self.player = None
        self._actor_filter = 'vehicle.*'
        self.cam_sensor=None
        from sdv.runtime.processes import CLEANUPS
        CLEANUPS.append(self.destroy)
        self.restart(conf)

    def restart(self, conf):
        self._destroy_sensors_only()

        self.sensors = []

        # 2. 차량이 없으면 최초 1회만 생성
        if self.player is None:
            vehicle_bp = self.world.get_blueprint_library().find("vehicle.tesla.model3")
            if vehicle_bp.has_attribute("role_name"):
                vehicle_bp.set_attribute("role_name", "hero")
            color = vehicle_bp.get_attribute("color").recommended_values[0]
            vehicle_bp.set_attribute("color", color)

            spawn_deadline = time.monotonic() + 10
            while self.player is None:
                if time.monotonic() > spawn_deadline:
                    raise RuntimeError('차량 생성 시간 초과. 다른 타운으로 다시 시도하세요.')
                spawn_points = self.map.get_spawn_points()
                spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
                self.player = self.world.try_spawn_actor(vehicle_bp, spawn_point)

            self.modify_vehicle_physics(self.player)
            self._vehicle = self.player

        self.world.tick()

        # 3. 센서 재설정
        for name in list(conf.keys()):
            if name == 'lidar':
                continue
            rgb_sensor = CameraSensor_RGB(self.player, conf[name])
            setattr(self, f"{name}_rgb", rgb_sensor)
            self.sensors.append(rgb_sensor.sensor)

        if 'lidar' in conf:
            self.lidar_sensor = LidarSensor(self.player, conf['lidar'])
            self.sensors.append(self.lidar_sensor.sensor)

    def _destroy_sensors_only(self):
        from sdv.runtime.actors import destroy_owned_actors
        actors, self.sensors = getattr(self, 'sensors', []), []
        destroy_owned_actors(self.client, actors, self.server_job)

    def modify_vehicle_physics(self, actor):
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def destroy(self):
        from sdv.runtime.actors import destroy_owned_actors
        from sdv.runtime.processes import CLEANUPS
        actors = list(getattr(self, 'sensors', [])) + [getattr(self, 'player', None)]
        self.sensors, self.player = [], None
        if self.destroy in CLEANUPS: CLEANUPS.remove(self.destroy)
        destroy_owned_actors(self.client, actors, self.server_job)

class CameraSensor_RGB:
    def __init__(self, parent_actor,conf):
        self.sensor = None
        self._parent = parent_actor
        world = self._parent.get_world()

        bp = world.get_blueprint_library().find('sensor.camera.rgb')
        configure_camera_blueprint(bp, conf)

        self.sensor = world.spawn_actor(bp, carla.Transform(
        carla.Location(x=conf['x'], y=conf['y'], z=conf['z']),
        carla.Rotation(pitch=conf['pitch'], roll=conf['roll'], yaw=conf['yaw'])
        ), attach_to=self._parent)

        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CameraSensor_RGB._on_image(weak_self, image))
        self.img=None
    @staticmethod
    def _on_image(weak_self, image):
        self = weak_self()
        if not self:
            return

        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        self.img = array

class LidarSensor:
    def __init__(self, parent_actor,conf):
        self.sensor = None
        self._parent = parent_actor
        world = self._parent.get_world()

        bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')

        configure_lidar_blueprint(bp, conf)

        self.sensor = world.spawn_actor(bp, carla.Transform(
        carla.Location(x=conf['x'], y=conf['y'], z=conf['z']),
        carla.Rotation(pitch=conf['pitch'], roll=conf['roll'], yaw=conf['yaw'])
        ), attach_to=self._parent)


        weak_self = weakref.ref(self)
        self.sensor.listen(lambda point_cloud: LidarSensor._on_point_cloud(weak_self, point_cloud))
        self.point_cloud=None

    @staticmethod
    def _on_point_cloud(weak_self, lidar_data):
        self = weak_self()
        if not self:
            return
        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        self.point_cloud=points
