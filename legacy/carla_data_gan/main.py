import carla
import numpy as np
import copy

from carla_data_gan.agents.navigation.basic_agent import BasicAgent
from carla_data_gan.agents.navigation.local_planner import RoadOption
from carla_data_gan.agents.navigation.behavior_types import Cautious, Aggressive, Normal
from carla_data_gan.agents.tools.misc import get_speed, positive
from carla_data_gan.map_utils import MapImage, encode_npy_to_pil, PIXELS_PER_METER
from carla_data_gan import lts_rendering

import numpy.random as random
import math
import weakref
import yaml
import torch
import pygame
import cv2
from collections import deque

MIN_WIDTH = 5
MIN_HEIGHT = 5

def carla_rot_to_mat(carla_rotation):
    """
    Transform rpy in carla.Rotation to rotation matrix in np.array

    :param carla_rotation: carla.Rotation 
    :return: np.array rotation matrix
    """
    roll = np.deg2rad(carla_rotation.roll)
    pitch = np.deg2rad(carla_rotation.pitch)
    yaw = np.deg2rad(carla_rotation.yaw)

    yaw_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    pitch_matrix = np.array([
        [np.cos(pitch), 0, -np.sin(pitch)],
        [0, 1, 0],
        [np.sin(pitch), 0, np.cos(pitch)]
    ])
    roll_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(roll), np.sin(roll)],
        [0, -np.sin(roll), np.cos(roll)]
    ])

    rotation_matrix = yaw_matrix.dot(pitch_matrix).dot(roll_matrix)
    return rotation_matrix
def vec_global_to_ref(target_vec_in_global, ref_rot_in_global):
    """
    :param target_vec_in_global: carla.Vector3D in global coordinate (world, actor)
    :param ref_rot_in_global: carla.Rotation in global coordinate (world, actor)
    :return: carla.Vector3D in ref coordinate
    """
    R = carla_rot_to_mat(ref_rot_in_global)
    np_vec_in_global = np.array([[target_vec_in_global.x],
                                 [target_vec_in_global.y],
                                 [target_vec_in_global.z]])
    np_vec_in_ref = R.T.dot(np_vec_in_global)
    target_vec_in_ref = carla.Vector3D(x=np_vec_in_ref[0, 0], y=np_vec_in_ref[1, 0], z=np_vec_in_ref[2, 0])
    return target_vec_in_ref
def loc_global_to_ref(target_loc_in_global, ref_trans_in_global):
    """
    :param target_loc_in_global: carla.Location in global coordinate (world, actor)
    :param ref_trans_in_global: carla.Transform in global coordinate (world, actor)
    :return: carla.Location in ref coordinate
    """
    x = target_loc_in_global.x - ref_trans_in_global.location.x
    y = target_loc_in_global.y - ref_trans_in_global.location.y
    z = target_loc_in_global.z - ref_trans_in_global.location.z
    vec_in_global = carla.Vector3D(x=x, y=y, z=z)
    vec_in_ref = vec_global_to_ref(vec_in_global, ref_trans_in_global.rotation)

    target_loc_in_ref = carla.Location(x=vec_in_ref.x, y=vec_in_ref.y, z=vec_in_ref.z)
    return target_loc_in_ref

def _get_traffic_light_waypoints(traffic_light, carla_map):
    """
    get area of a given traffic light
    adapted from "carla-simulator/scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_criteria.py"
    """
    base_transform = traffic_light.get_transform()
    tv_loc = traffic_light.trigger_volume.location
    tv_ext = traffic_light.trigger_volume.extent

    # Discretize the trigger box into points
    x_values = np.arange(-0.9 * tv_ext.x, 0.9 * tv_ext.x, 1.0)  # 0.9 to avoid crossing to adjacent lanes
    area = []
    for x in x_values:
        point_location = base_transform.transform(tv_loc + carla.Location(x=x))
        area.append(point_location)

    # Get the waypoints of these points, removing duplicates
    ini_wps = []
    for pt in area:
        wpx = carla_map.get_waypoint(pt)
        # As x_values are arranged in order, only the last one has to be checked
        if not ini_wps or ini_wps[-1].road_id != wpx.road_id or ini_wps[-1].lane_id != wpx.lane_id:
            ini_wps.append(wpx)

    # Leaderboard: Advance them until the intersection
    stopline_wps = []
    stopline_vertices = []
    junction_wps = []
    for wpx in ini_wps:
        while not wpx.is_intersection:
            next_wp = wpx.next(0.5)[0]
            if next_wp and not next_wp.is_intersection:
                wpx = next_wp
            else:
                break
        junction_wps.append(wpx)

        stopline_wps.append(wpx)
        vec_forward = wpx.transform.get_forward_vector()
        vec_right = carla.Vector3D(x=-vec_forward.y, y=vec_forward.x, z=0)

        loc_left = wpx.transform.location - 0.4 * wpx.lane_width * vec_right
        loc_right = wpx.transform.location + 0.4 * wpx.lane_width * vec_right
        stopline_vertices.append([loc_left, loc_right])

    # all paths at junction for this traffic light
    junction_paths = []
    path_wps = []
    wp_queue = deque(junction_wps)
    while len(wp_queue) > 0:
        current_wp = wp_queue.pop()
        path_wps.append(current_wp)
        next_wps = current_wp.next(1.0)
        for next_wp in next_wps:
            if next_wp.is_junction:
                wp_queue.append(next_wp)
            else:
                junction_paths.append(path_wps)
                path_wps = []

    return carla.Location(base_transform.transform(tv_loc)), stopline_wps, stopline_vertices, junction_paths

class TrafficLightHandler:
    num_tl = 0
    list_tl_actor = []
    list_tv_loc = []
    list_stopline_wps = []
    list_stopline_vtx = []
    list_junction_paths = []
    carla_map = None

    @staticmethod
    def reset(world):
        TrafficLightHandler.carla_map = world.get_map()

        TrafficLightHandler.num_tl = 0
        TrafficLightHandler.list_tl_actor = []
        TrafficLightHandler.list_tv_loc = []
        TrafficLightHandler.list_stopline_wps = []
        TrafficLightHandler.list_stopline_vtx = []
        TrafficLightHandler.list_junction_paths = []

        all_actors = world.get_actors()
        for _actor in all_actors:
            if 'traffic_light' in _actor.type_id:
                tv_loc, stopline_wps, stopline_vtx, junction_paths = _get_traffic_light_waypoints(
                    _actor, TrafficLightHandler.carla_map)

                TrafficLightHandler.list_tl_actor.append(_actor)
                TrafficLightHandler.list_tv_loc.append(tv_loc)
                TrafficLightHandler.list_stopline_wps.append(stopline_wps)
                TrafficLightHandler.list_stopline_vtx.append(stopline_vtx)
                TrafficLightHandler.list_junction_paths.append(junction_paths)

                TrafficLightHandler.num_tl += 1

    @staticmethod
    def get_light_state(vehicle, offset=0.0, dist_threshold=15.0):
        '''
        vehicle: carla.Vehicle
        '''
        vec_tra = vehicle.get_transform()
        veh_dir = vec_tra.get_forward_vector()

        hit_loc = vec_tra.transform(carla.Location(x=offset))
        hit_wp = TrafficLightHandler.carla_map.get_waypoint(hit_loc)

        light_loc = None
        light_state = None
        light_id = None
        for i in range(TrafficLightHandler.num_tl):
            traffic_light = TrafficLightHandler.list_tl_actor[i]
            tv_loc = 0.5*TrafficLightHandler.list_stopline_wps[i][0].transform.location \
                + 0.5*TrafficLightHandler.list_stopline_wps[i][-1].transform.location

            distance = np.sqrt((tv_loc.x-hit_loc.x)**2 + (tv_loc.y-hit_loc.y)**2)
            if distance > dist_threshold:
                continue

            for wp in TrafficLightHandler.list_stopline_wps[i]:

                wp_dir = wp.transform.get_forward_vector()
                dot_ve_wp = veh_dir.x * wp_dir.x + veh_dir.y * wp_dir.y + veh_dir.z * wp_dir.z

                wp_1 = wp.previous(4.0)[0]
                same_road = (hit_wp.road_id == wp.road_id) and (hit_wp.lane_id == wp.lane_id)
                same_road_1 = (hit_wp.road_id == wp_1.road_id) and (hit_wp.lane_id == wp_1.lane_id)

                # if (wp.road_id != wp_1.road_id) or (wp.lane_id != wp_1.lane_id):
                #     print(f'Traffic Light Problem: {wp.road_id}={wp_1.road_id}, {wp.lane_id}={wp_1.lane_id}')

                if (same_road or same_road_1) and dot_ve_wp > 0:
                    # This light is red and is affecting our lane
                    loc_in_ev = loc_global_to_ref(wp.transform.location, vec_tra)
                    light_loc = np.array([loc_in_ev.x, loc_in_ev.y, loc_in_ev.z], dtype=np.float32)
                    light_state = traffic_light.state
                    light_id = traffic_light.id
                    break

        return light_state, light_loc, light_id

    @staticmethod
    def get_junctoin_paths(veh_loc, color=0, dist_threshold=50.0):
        if color == 0:
            tl_state = carla.TrafficLightState.Green
        elif color == 1:
            tl_state = carla.TrafficLightState.Yellow
        elif color == 2:
            tl_state = carla.TrafficLightState.Red

        junctoin_paths = []
        for i in range(TrafficLightHandler.num_tl):
            traffic_light = TrafficLightHandler.list_tl_actor[i]
            tv_loc = TrafficLightHandler.list_tv_loc[i]
            if tv_loc.distance(veh_loc) > dist_threshold:
                continue
            if traffic_light.state != tl_state:
                continue

            junctoin_paths += TrafficLightHandler.list_junction_paths[i]

        return junctoin_paths

    @staticmethod
    def get_stopline_vtx(veh_loc, dist_threshold=60.0):
        stopline_vtx = []
        for i in range(TrafficLightHandler.num_tl):
            traffic_light = TrafficLightHandler.list_tl_actor[i]
            tv_loc = TrafficLightHandler.list_tv_loc[i]

            if tv_loc.distance(veh_loc) > dist_threshold:
                continue

            stopline_vtx += TrafficLightHandler.list_stopline_vtx[i]

        return stopline_vtx

class BehaviorAgent(BasicAgent):

    def __init__(self, vehicle, behavior='normal'):

        super(BehaviorAgent, self).__init__(vehicle)
        self._look_ahead_steps = 0

        # Vehicle information
        self._speed = 0
        self._speed_limit = 0
        self._direction = None
        self._incoming_direction = None
        self._incoming_waypoint = None
        self._min_speed = 5
        self._behavior = None
        self._sampling_resolution = 1.0

        # 운전 스타일
        if behavior == 'cautious':
            self._behavior = Cautious()
        elif behavior == 'normal':
            self._behavior = Normal()

        elif behavior == 'aggressive':
            self._behavior = Aggressive()

    def _update_information(self):
        """
        This method updates the information regarding the ego
        vehicle based on the surrounding world.
        """
        self._speed = get_speed(self._vehicle)
        self._speed_limit = self._vehicle.get_speed_limit()
        self._local_planner.set_speed(self._speed_limit)
        self._direction = self._local_planner.target_road_option
        if self._direction is None:
            self._direction = RoadOption.LANEFOLLOW

        self._look_ahead_steps = int((self._speed_limit) / 10)

        self._incoming_waypoint, self._incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self._look_ahead_steps)
        if self._incoming_direction is None:
            self._incoming_direction = RoadOption.LANEFOLLOW

    def traffic_light_manager(self):
        """
        This method is in charge of behaviors for red lights.
        """
        actor_list = self._world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")
        affected, _ = self._affected_by_traffic_light(lights_list)

        return affected
    def traffic_light_manager2(self):
        """
        This method is in charge of behaviors for red lights.
        """
        actor_list = self._world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")
        affected, _ = self._affected_by_traffic_light2(lights_list)

        return affected
    


    def set_collision_sensor(self, collision_sensor):
        self._collision_sensor = collision_sensor
    def _tailgating(self, waypoint, vehicle_list):
        """
        This method is in charge of tailgating behaviors.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :param vehicle_list: list of all the nearby vehicles
        """

        left_turn = waypoint.left_lane_marking.lane_change
        right_turn = waypoint.right_lane_marking.lane_change

        left_wpt = waypoint.get_left_lane()
        right_wpt = waypoint.get_right_lane()

        behind_vehicle_state, behind_vehicle, _ = self._vehicle_obstacle_detected(vehicle_list, max(
            self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, low_angle_th=160)
        if behind_vehicle_state and self._speed < get_speed(behind_vehicle):
            if (right_turn == carla.LaneChange.Right or right_turn ==
                    carla.LaneChange.Both) and waypoint.lane_id * right_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
                if not new_vehicle_state:
                    print("Tailgating, moving to the right!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         right_wpt.transform.location)
            elif left_turn == carla.LaneChange.Left and waypoint.lane_id * left_wpt.lane_id > 0 and left_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
                if not new_vehicle_state:
                    print("Tailgating, moving to the left!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         left_wpt.transform.location)

    def collision_and_car_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        and managing possible tailgating chances.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a vehicle nearby, False if not
            :return vehicle: nearby vehicle
            :return distance: distance to nearby vehicle
        """

        vehicle_list = self._world.get_actors().filter("*vehicle*")
        def dist(v): return v.get_location().distance(waypoint.transform.location)
        vehicle_list = [v for v in vehicle_list if dist(v) < 45 and v.id != self._vehicle.id]

        if self._direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
        else:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=30)

            # Check for tailgating
            if not vehicle_state and self._direction == RoadOption.LANEFOLLOW \
                    and not waypoint.is_junction and self._speed > 10 \
                    and self._behavior.tailgate_counter == 0:
                self._tailgating(waypoint, vehicle_list)

        return vehicle_state, vehicle, distance

    def pedestrian_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        with any pedestrian.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a walker nearby, False if not
            :return vehicle: nearby walker
            :return distance: distance to nearby walker
        """

        walker_list = self._world.get_actors().filter("*walker.pedestrian*")
        def dist(w): return w.get_location().distance(waypoint.transform.location)
        walker_list = [w for w in walker_list if dist(w) < 10]

        if self._direction == RoadOption.CHANGELANELEFT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1)
        else:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=60)

        return walker_state, walker, distance

    def car_following_manager(self, vehicle, distance, debug=False):
        """
        Module in charge of car-following behaviors when there's
        someone in front of us.

            :param vehicle: car to follow
            :param distance: distance from vehicle
            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """

        vehicle_speed = get_speed(vehicle)
        delta_v = max(1, (self._speed - vehicle_speed) / 3.6)
        ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0., 1.)

        # Under safety time distance, slow down.
        if self._behavior.safety_time > ttc > 0.0:
            target_speed = min([
                positive(vehicle_speed - self._behavior.speed_decrease),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Actual safety distance area, try to follow the speed of the vehicle in front.
        elif 2 * self._behavior.safety_time > ttc >= self._behavior.safety_time:
            target_speed = min([
                max(self._min_speed, vehicle_speed),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Normal behavior.
        else:
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        return control
    def location_to_dict(self,loc):
        return {"x": loc.x, "y": loc.y, "z": loc.z} if loc else None
    def _nearest_global_index(self, loc: 'carla.Location') -> int:
        """현재 위치에 가장 가까운 글로벌 경로 인덱스 리턴."""
        if not self._global_route:
            return -1
        lx, ly = loc.x, loc.y
        best_i, best_d2 = 0, float('inf')
        for i, (wp, _opt) in enumerate(self._global_route):
            w = wp.transform.location
            dx, dy = (w.x - lx), (w.y - ly)
            d2 = dx*dx + dy*dy
            if d2 < best_d2:
                best_d2, best_i = d2, i
        return best_i
    def get_global_future_waypoints(self, count=20, stride=1):

        ego_loc = self._vehicle.get_transform().location
        idx = self._nearest_global_index(ego_loc)
        if idx < 0:
            return []

        out = []
        # 너무 가까운 첫 포인트는 스킵하고 약간 앞에서 시작하도록 보정(선택)
        start = max(idx + 1, 0)
        # stride 간격으로 포인트 샘플링
        for k in range(start, min(len(self._global_route), start + count*stride), stride):
            wp, _ = self._global_route[k]
            loc = wp.transform.location
            out.append(self.location_to_dict2(loc))
        return out
    def location_to_dict2(self,loc):
        return (loc.x,loc.y) if loc else None

    def run_step(self, debug=False):
        """
        Execute one step of navigation.

        :param debug: boolean for debugging
        :return: Tuple(carla.VehicleControl, dict) - control command and info dictionary
        """
        self._update_information()

        if self._behavior.tailgate_counter > 0:
            self._behavior.tailgate_counter -= 1

        ego_vehicle_tf = self._vehicle.get_transform()
        ego_vehicle_loc = ego_vehicle_tf.location
        ego_vehicle_rot = ego_vehicle_tf.rotation  # pitch, yaw, roll 정보 포함
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)
        vehicle_state, vehicle, distance = self.collision_and_car_avoid_manager(ego_vehicle_wp)

        info_dict = {
            "current_speed": self._speed,
            "current_waypoint": ego_vehicle_wp.transform.location,
            "target_waypoint": self._local_planner.target_waypoint.transform.location if self._local_planner.target_waypoint else None,
            "global_waypoints": [],
            "current_control": {
                "throttle": self._vehicle.get_control().throttle,
                "steer": self._vehicle.get_control().steer,
                "brake": self._vehicle.get_control().brake
            },
            "next_control": {
                "throttle": 0.0,
                "steer": 0.0,
                "brake": 0.0
            },
            "traffic_light": self.traffic_light_manager(),
            "traffic_light_distance": self.traffic_light_distance(),
            "traffic_light_20": False,
            "collision": False,
            "vehicle_in_front": vehicle_state,
            "ego_vehicle_loc": self.location_to_dict(ego_vehicle_loc),
            "ego_vehicle_rot": {
                "pitch": ego_vehicle_rot.pitch,
                "yaw": ego_vehicle_rot.yaw,
                "roll": ego_vehicle_rot.roll
            }
        }
        collision_detected = False
        if hasattr(self, "_collision_sensor"):
            history = self._collision_sensor.get_collision_history()
            recent_frames = [k for k in history if k > self._vehicle.get_world().get_snapshot().frame - 5]
            if len(recent_frames) > 0:
                collision_detected = True

        if collision_detected:

            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

            # Save final control values
            info_dict["next_control"] = {
                "throttle": control.throttle,
                "steer": control.steer,
                "brake": control.brake
            }

            info_dict.update({
                "collision": True,
                "traffic_light": self.traffic_light_manager(),
                "vehicle_in_front": vehicle_state,
            "ego_vehicle_loc":self.location_to_dict(ego_vehicle_loc),
            })
            return control, info_dict

        global_fwps = self.get_global_future_waypoints(count=6, stride=5)
        if global_fwps:
            info_dict["global_waypoints"] = global_fwps

        if self.traffic_light_manager2():
            info_dict['traffic_light_20']=True

        upcoming_waypoints = self._local_planner._waypoints_queue
        ways=[]
        for i in range(min(5, len(upcoming_waypoints))):
            ways.append((upcoming_waypoints[i][0].transform.location.x,upcoming_waypoints[i][0].transform.location.y))

        self.future_waypoints=ways
        control = None


        # 2.1: Pedestrian
        walker_state, walker, w_distance = self.pedestrian_avoid_manager(ego_vehicle_wp)
        if walker_state:
            
            distance = w_distance - max(
                walker.bounding_box.extent.y, walker.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)
            if distance < self._behavior.braking_distance:
                control = self.emergency_stop()
                info_dict["next_control"] = {
                    "throttle": control.throttle,
                    "steer": control.steer,
                    "brake": control.brake
                }
                return control, info_dict

        # 2.2: Vehicle collision or tailgating
        vehicle_state, vehicle, distance = self.collision_and_car_avoid_manager(ego_vehicle_wp)
        if vehicle_state:
            distance = distance - max(
                vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)
            if distance < self._behavior.braking_distance:
                control = self.emergency_stop()
            else:
                control = self.car_following_manager(vehicle, distance)
            info_dict["next_control"] = {
                "throttle": control.throttle,
                "steer": control.steer,
                "brake": control.brake
            }
            return control, info_dict

        # 3: Intersection
        elif self._incoming_waypoint.is_junction and (self._incoming_direction in [RoadOption.LEFT, RoadOption.RIGHT]):
            target_speed = min([self._behavior.max_speed, self._speed_limit - 5])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # 4: Normal case
        else:
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # 1: Red light
        if self.traffic_light_manager():
            control = self.emergency_stop()
            info_dict["next_control"] = {
                "throttle": control.throttle,
                "steer": control.steer,
                "brake": control.brake
            }
            return control, info_dict
        
        # Save final control values
        info_dict["next_control"] = {
            "throttle": control.throttle,
            "steer": control.steer,
            "brake": control.brake
        }

        return control, info_dict

    def emergency_stop(self):

        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control

def angle_diff(v0, v1):

    v0_xy = v0[:2]
    v1_xy = v1[:2]
    v0_xy_norm = np.linalg.norm(v0_xy)
    v1_xy_norm = np.linalg.norm(v1_xy)
    if v0_xy_norm == 0 or v1_xy_norm == 0:
        return 0

    v0_xy_u = v0_xy / v0_xy_norm
    v1_xy_u = v1_xy / v1_xy_norm
    dot_product = np.dot(v0_xy_u, v1_xy_u)
    angle = np.arccos(dot_product)

    # Calculate the sign of the angle using the cross product
    cross_product = np.cross(v0_xy_u, v1_xy_u)
    if cross_product < 0:
        angle = -angle
    if abs(angle) >= 2.3:
        return 0
    return round(angle, 2)

def vector(v):

    if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
        return np.array([v.x, v.y, v.z])
    elif isinstance(v, carla.Rotation):
        return np.array([v.pitch, v.yaw, v.roll])
def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
class World:
    def __init__(self, client, conf):

        self.client = client
        self.world = client.get_world()
        self.map = self.world.get_map()
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self._actor_filter = 'vehicle.*'
        self.cam_sensor=None
        self.conf=conf
        self.detection_radius = 60.0 
        self.light_radius = 30.0

        cfg  = load_yaml("data/sensor_config.yaml")
        sensor_cfg  = cfg["sensors"]
        cam_sensors = {k: v for k, v in sensor_cfg.items() if "_cam" in k}
        cam_cfg=list(cam_sensors.items())[0][1]
        self._sensor_data = {"width": cam_cfg['width'], "height": cam_cfg['height'], "fov": cam_cfg['fov']}
        self._sensor_data["calibration"] = self._get_camera_to_car_calibration(
            self._sensor_data
        )
        self._traffic_lights = list()
        self.restart()
        self.TH=TrafficLightHandler()
        self.TH.reset(self.world)

    def render_BEV(self,ways,global_waypoints):
        semantic_grid = self.global_map
        
        vehicle_position = self._vehicle.get_location()
        ego_pos_list =  [self._vehicle.get_transform().location.x, self._vehicle.get_transform().location.y]
        ego_yaw_list =  [self._vehicle.get_transform().rotation.yaw/180*np.pi]

        # fetch local birdview per agent
        ego_pos =  torch.tensor([self._vehicle.get_transform().location.x, self._vehicle.get_transform().location.y], device='cpu', dtype=torch.float32)
        ego_yaw =  torch.tensor([self._vehicle.get_transform().rotation.yaw/180*np.pi], device='cpu', dtype=torch.float32)
        birdview = self.renderer.get_local_birdview(
            semantic_grid,
            ego_pos,
            ego_yaw
        )
        vehicle_view = torch.zeros_like(birdview)
        walker_view = torch.zeros_like(birdview)
        light_birdview = torch.zeros_like(birdview)
        stopline_birdview = torch.zeros_like(birdview)
        B,C,W,H=birdview.shape

        route_birdview = np.zeros((H, W), dtype=np.uint8)

        self._actors = self.world.get_actors()
        vehicles = self._actors.filter('*vehicle*')
        for vehicle in vehicles:
            if (vehicle.get_location().distance(self._vehicle.get_location()) < self.detection_radius):
                if (vehicle.id != self._vehicle.id):

                    pos =  torch.tensor([vehicle.get_transform().location.x, vehicle.get_transform().location.y], device='cpu', dtype=torch.float32)
                    yaw =  torch.tensor([vehicle.get_transform().rotation.yaw/180*np.pi], device='cpu', dtype=torch.float32)
                    veh_x_extent = int(max(vehicle.bounding_box.extent.x*2, 1) * PIXELS_PER_METER)
                    veh_y_extent = int(max(vehicle.bounding_box.extent.y*2, 1) * PIXELS_PER_METER)

                    self.vehicle_template = torch.ones(1, 1, veh_x_extent, veh_y_extent, device='cpu')
                    self.renderer.render_agent_bv(
                        vehicle_view,
                        ego_pos,
                        ego_yaw,
                        self.vehicle_template,
                        pos,
                        yaw,
                        channel=5
                    )

        ego_pos_batched = []
        ego_yaw_batched = []
        pos_batched = []
        yaw_batched = []
        template_batched = []
        channel_batched = []

        # -----------------------------------------------------------
        # Pedestrian rendering
        # -----------------------------------------------------------
        walkers = self._actors.filter('*walker*')
        for walker in walkers:
            ego_pos_batched.append(ego_pos_list)
            ego_yaw_batched.append(ego_yaw_list)
            pos_batched.append([walker.get_transform().location.x, walker.get_transform().location.y])
            yaw_batched.append([walker.get_transform().rotation.yaw/180*np.pi])
            channel_batched.append(6)
            template_batched.append(np.ones([7, 7]))

        if len(ego_pos_batched)>0:
            ego_pos_batched_torch = torch.tensor(ego_pos_batched, device='cpu', dtype=torch.float32).unsqueeze(1)
            ego_yaw_batched_torch = torch.tensor(ego_yaw_batched, device='cpu', dtype=torch.float32).unsqueeze(1)
            pos_batched_torch = torch.tensor(pos_batched, device='cpu', dtype=torch.float32).unsqueeze(1)
            yaw_batched_torch = torch.tensor(yaw_batched, device='cpu', dtype=torch.float32).unsqueeze(1)
            template_batched_torch = torch.tensor(template_batched, device='cpu', dtype=torch.float32).unsqueeze(1)
            channel_batched_torch = torch.tensor(channel_batched, device='cpu', dtype=torch.float32)

            self.renderer.render_agent_bv_batched(
                walker_view,
                ego_pos_batched_torch,
                ego_yaw_batched_torch,
                template_batched_torch,
                pos_batched_torch,
                yaw_batched_torch,
                channel=channel_batched_torch,
            )

        ego_pos_batched = []
        ego_yaw_batched = []
        pos_batched = []
        yaw_batched = []
        template_batched = []
        channel_batched = []

        self.traffic_lights = self._actors.filter('*traffic_light*')
        for traffic_light in self.traffic_lights:

            trigger_box_global_pos = traffic_light.get_transform().transform(traffic_light.trigger_volume.location)
            trigger_box_global_pos = carla.Location(x=trigger_box_global_pos.x, y=trigger_box_global_pos.y, z=trigger_box_global_pos.z)
            if (trigger_box_global_pos.distance(vehicle_position) > self.light_radius):
                continue
            ego_pos_batched.append(ego_pos_list)
            ego_yaw_batched.append(ego_yaw_list)
            pos_batched.append([traffic_light.get_transform().location.x, traffic_light.get_transform().location.y])
            yaw_batched.append([traffic_light.get_transform().rotation.yaw/180*np.pi])

            template_batched.append(np.ones([6, 6]))
            if str(traffic_light.state) == 'Green':
                channel_batched.append(4)
            elif str(traffic_light.state) == 'Yellow':
                channel_batched.append(3)
            elif str(traffic_light.state) == 'Red':
                channel_batched.append(2)


        if len(ego_pos_batched)>0:
            ego_pos_batched_torch = torch.tensor(ego_pos_batched, device='cpu', dtype=torch.float32).unsqueeze(1)
            ego_yaw_batched_torch = torch.tensor(ego_yaw_batched, device='cpu', dtype=torch.float32).unsqueeze(1)
            pos_batched_torch = torch.tensor(pos_batched, device='cpu', dtype=torch.float32).unsqueeze(1)
            yaw_batched_torch = torch.tensor(yaw_batched, device='cpu', dtype=torch.float32).unsqueeze(1)
            template_batched_torch = torch.tensor(template_batched, device='cpu', dtype=torch.float32).unsqueeze(1)
            channel_batched_torch = torch.tensor(channel_batched, device='cpu', dtype=torch.int)


            self.renderer.render_agent_bv_batched(
                light_birdview,
                ego_pos_batched_torch,
                ego_yaw_batched_torch,
                template_batched_torch,
                pos_batched_torch,
                yaw_batched_torch,
                channel=channel_batched_torch,
            )

        ego_pos_batched = []
        ego_yaw_batched = []
        pos_batched = []
        channel_batched = []

        self.stoplines = self.TH.get_stopline_vtx(self._vehicle.get_transform().location)
        for stopline in self.stoplines:

            ego_pos_batched.append(ego_pos_list)
            ego_yaw_batched.append(ego_yaw_list)
            stoplines=[]
            for stop in stopline:

                stoplines.append((stop.x, stop.y))
            pos_batched.append(stoplines)
            
            channel_batched.append(7)

        if len(ego_pos_batched)>0:
            ego_pos_batched_torch = torch.tensor(ego_pos_batched, device='cpu', dtype=torch.float32).unsqueeze(1)
            ego_yaw_batched_torch = torch.tensor(ego_yaw_batched, device='cpu', dtype=torch.float32).unsqueeze(1)
            pos_batched = torch.tensor(pos_batched, device='cpu', dtype=torch.float32).unsqueeze(1)
            pos_batched =pos_batched.view(pos_batched.shape[0], 1, 4)
            channel_batched_torch = torch.tensor(channel_batched, device='cpu', dtype=torch.int)

            self.renderer.render_agent_stopline_bv_batched(
                stopline_birdview,
                ego_pos_batched_torch,
                ego_yaw_batched_torch,
                pos_batched,

                channel=channel_batched_torch,
            )
        ways_torch = torch.tensor(ways[:5], device='cpu', dtype=torch.float32).unsqueeze(1)
        ego_pos_batched = []
        ego_yaw_batched = []
        pos_batched = []

        for _ in range(ways_torch.shape[0]):
            ego_pos_batched.append(ego_pos_list)
            ego_yaw_batched.append(ego_yaw_list)

        ego_pos_batched_torch = torch.tensor(ego_pos_batched, device='cpu', dtype=torch.float32).unsqueeze(1)
        ego_yaw_batched_torch = torch.tensor(ego_yaw_batched, device='cpu', dtype=torch.float32).unsqueeze(1)


        ways_torch=self.renderer.world_to_pix_crop_batched(ways_torch, ego_pos_batched_torch, ego_yaw_batched_torch)

        ways_torch = torch.cat([torch.tensor([[[250.0, 250.0]]], device=ways_torch.device), ways_torch], dim=0)

        cv2.polylines(
            route_birdview,
            [np.round(ways_torch.cpu().numpy()).astype(np.int32)],
            False,
            1,
            thickness=16
        )


        ways_torch = torch.tensor(ways, device='cpu', dtype=torch.float32).unsqueeze(1)
        ego_pos_batched = []
        ego_yaw_batched = []
        pos_batched = []

        for _ in range(ways_torch.shape[0]):
            ego_pos_batched.append(ego_pos_list)
            ego_yaw_batched.append(ego_yaw_list)

        ego_pos_batched_torch = torch.tensor(ego_pos_batched, device='cpu', dtype=torch.float32).unsqueeze(1)
        ego_yaw_batched_torch = torch.tensor(ego_yaw_batched, device='cpu', dtype=torch.float32).unsqueeze(1)


        ways_torch=self.renderer.world_to_pix_crop_batched(ways_torch, ego_pos_batched_torch, ego_yaw_batched_torch)

        self.ways_torch = torch.cat([torch.tensor([[[250.0, 250.0]]], device=ways_torch.device), ways_torch], dim=0)


        ways_torch = torch.tensor(global_waypoints, device='cpu', dtype=torch.float32).unsqueeze(1)
        ego_pos_batched = []
        ego_yaw_batched = []
        pos_batched = []

        for _ in range(ways_torch.shape[0]):
            ego_pos_batched.append(ego_pos_list)
            ego_yaw_batched.append(ego_yaw_list)

        ego_pos_batched_torch = torch.tensor(ego_pos_batched, device='cpu', dtype=torch.float32).unsqueeze(1)
        ego_yaw_batched_torch = torch.tensor(ego_yaw_batched, device='cpu', dtype=torch.float32).unsqueeze(1)


        ways_torch=self.renderer.world_to_pix_crop_batched(ways_torch, ego_pos_batched_torch, ego_yaw_batched_torch)

        self.global_waypoints = ways_torch





        return birdview ,light_birdview ,walker_view, vehicle_view,stopline_birdview,route_birdview*255

    def restart(self):
        self.sensors=[]
        vehicle_bp = self.world.get_blueprint_library().find("vehicle.tesla.model3")
        
        if vehicle_bp.has_attribute("role_name"):
            vehicle_bp.set_attribute("role_name", "hero")
        
        color = vehicle_bp.get_attribute("color").recommended_values[0]
        vehicle_bp.set_attribute("color", color)
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            self.modify_vehicle_physics(self.player)
        self._vehicle= self.player
        self.world.tick()


        town_name = self.world.get_map().name.split('/')[-1]
        with open(f"carla/CarlaUE4/Content/Carla/Maps/OpenDrive/{town_name}.xodr", "r") as f:
            xodr_str = f.read()
        self.world_map = carla.Map("RouteMap", xodr_str)

        self.vehicle_template = torch.ones(1, 1, 22, 9, device='cpu')
        self.walker_template = torch.ones(1, 1, 10, 7, device='cpu')
        self.traffic_light_template = torch.ones(1, 1, 4, 4, device='cpu')

        map_image = MapImage(self.world, self.world_map, PIXELS_PER_METER)
        make_image = lambda x: np.swapaxes(pygame.surfarray.array3d(x), 0, 1).mean(axis=-1)
        road = make_image(map_image.map_surface)
        lane = make_image(map_image.lane_surface)
        
        self.global_map = np.zeros((1, 15,) + road.shape)
        self.global_map[:, 0, ...] = road / 255.
        self.global_map[:, 1, ...] = lane / 255.

        self.global_map = torch.tensor(self.global_map, device='cpu', dtype=torch.float32)
        world_offset = torch.tensor(map_image._world_offset, device='cpu', dtype=torch.float32)
        self.map_dims = self.global_map.shape[2:4]

        self.renderer = lts_rendering.Renderer(world_offset, self.map_dims, data_generation=True)



        self.collision_sensor = CollisionSensor(self.player)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = ImuSensor(self.player)

        self.sensors.append(self.imu_sensor.sensor)
        self.sensors.append(self.gnss_sensor.sensor)
        self.sensors.append(self.lane_invasion_sensor.sensor)
        self.sensors.append(self.collision_sensor.sensor)


        for name in list(self.conf.keys()):
            if name == 'lidar':
                continue

            rgb_sensor = CameraSensor_RGB(self.player, self.conf[name])
            setattr(self, f"{name}_rgb", rgb_sensor)
            self.sensors.append(rgb_sensor.sensor)

            if self.conf[name].get('seg', False) or self.conf[name].get('od', False):
                seg_sensor = CameraSensor_Seg(self.player, self.conf[name])
                setattr(self, f"{name}_seg", seg_sensor)
                self.sensors.append(seg_sensor.sensor)

            if self.conf[name].get('depth', False) or self.conf[name].get('od', False):
                depth_sensor = CameraSensor_Depth(self.player, self.conf[name])
                setattr(self, f"{name}_depth", depth_sensor)
                self.sensors.append(depth_sensor.sensor)

        if 'lidar' in self.conf:
            self.lidar_sensor = LidarSensor(self.player,self.conf['lidar'])
            self.sensors.append(self.lidar_sensor.sensor)

    def _get_2d_bbs(self, sensor_transform, affordances, bb_3d, seg_img):
        results = []

        # traffic_light = self._vehicle.get_traffic_light()
        # if traffic_light is not None and affordances["traffic_light"] is not None:
        #     baseline = self._get_2d_bb_baseline(sensor_transform,traffic_light, distance=8)
        #     tl_bb = self._baseline_to_box(baseline, sensor_transform, height=0.5)
        #     if tl_bb is not None:
        #         x1, y1 = tl_bb[0]
        #         x2, y2 = tl_bb[1]
        #         results.append([x1, y1, x2, y2, 2])  # class 2 = traffic light

        # --- Vehicles ---
        for vehicle in bb_3d["vehicles"]:
            trig_loc_world = self._create_bb_points(vehicle).T
            cords_x_y_z = self._world_to_sensor(
                trig_loc_world, sensor_transform, False
            )
            cords_x_y_z = np.array(cords_x_y_z)[:3, :]
            veh_bb = self._coords_to_2d_bb(cords_x_y_z)

            if veh_bb is not None:
                x1, y1 = veh_bb[0]
                x2, y2 = veh_bb[1]
                width = x2 - x1
                height = y2 - y1
                if width < MIN_WIDTH or height < MIN_HEIGHT:
                    continue  # 너무 작으면 무시
                region = seg_img[y1:y2, x1:x2]
                visible_ratio = np.count_nonzero(region == 10) / region.size if region.size else 0
                if visible_ratio > 0.15:  
                    results.append([x1, y1, x2, y2, 0])  

        # --- Pedestrians ---
        for pedestrian in bb_3d["pedestrians"]:
            trig_loc_world = self._create_bb_points(pedestrian).T
            cords_x_y_z = self._world_to_sensor(
                trig_loc_world, sensor_transform, False
            )
            cords_x_y_z = np.array(cords_x_y_z)[:3, :]
            ped_bb = self._coords_to_2d_bb(cords_x_y_z)

            if ped_bb is not None:
                x1, y1 = ped_bb[0]
                x2, y2 = ped_bb[1]
                
                width = x2 - x1
                height = y2 - y1
                if width < MIN_WIDTH or height < MIN_HEIGHT:
                    continue  # 너무 작으면 무시

                region = seg_img[y1:y2, x1:x2]
                visible_ratio = np.count_nonzero(region == 4) / region.size if region.size else 0
                if visible_ratio > 0.1:
                    results.append([x1, y1, x2, y2, 1])

        return results
    
    def _get_3d_bbs(self, max_distance=50):
        bounding_boxes = {
            "traffic_lights": [],
            "stop_signs": [],
            "vehicles": [],
            "trucks": [],
            "bicycles": [],
            "pedestrians": [],
        }

        bounding_boxes["traffic_lights"] = self._find_obstacle_3dbb("*traffic_light*", max_distance)
        bounding_boxes["stop_signs"] = self._find_obstacle_3dbb("*stop*", max_distance)
        bounding_boxes["pedestrians"] = self._find_obstacle_3dbb("*walker*", max_distance)

        vehicle_bbs = self._find_obstacle_3dbb("*vehicle*", max_distance, return_with_type=True)
        for bb, type_id in vehicle_bbs:
            if any(x in type_id for x in ["cargo", "truck"]):
                bounding_boxes["trucks"].append(bb)
            elif any(x in type_id for x in ["bike", "bicycle", "motorcycle"]):
                bounding_boxes["bicycles"].append(bb)
            else:
                bounding_boxes["vehicles"].append(bb)

        return bounding_boxes
    def _change_seg_tl(self,sensor_transform, seg_img, depth_img, traffic_lights, _region_size=4):

        if traffic_lights:  # Check if the list is not empty
            for tl in traffic_lights:  # Iterate over each traffic light
                _dist = self._get_distance_from_camera(tl.get_transform().location,sensor_transform)
                _region = np.abs(depth_img - _dist)

                # Assign the correct traffic light state
                if tl.get_state() == carla.TrafficLightState.Red:
                    state = 23
                elif tl.get_state() == carla.TrafficLightState.Yellow:
                    state = 24
                else:  # Green or Unknown state
                    state = 18

                # Update segmentation image
                seg_img[(_region < _region_size) & (seg_img == 18)] = state

    def _get_distance_from_camera(self, target,sensor_transform):


        distance = np.sqrt(
                (sensor_transform.location.x - target.x) ** 2 +
                (sensor_transform.location.y - target.y) ** 2 +
                (sensor_transform.location.z - target.z) ** 2)

        return distance

    def _find_obstacle_3dbb(self, obstacle_type, max_distance=50, return_with_type=False):
        obst = []

        _actors = self.world.get_actors()
        _obstacles = _actors.filter(obstacle_type)

        for _obstacle in _obstacles:
            distance_to_car = _obstacle.get_transform().location.distance(self._vehicle.get_location())

            if 0 < distance_to_car <= max_distance:
                if hasattr(_obstacle, "bounding_box"):
                    loc = _obstacle.bounding_box.location
                    _obstacle.get_transform().transform(loc)

                    extent = _obstacle.bounding_box.extent
                    _rotation_matrix = self.get_matrix(
                        carla.Transform(carla.Location(0, 0, 0), _obstacle.get_transform().rotation)
                    )

                    rotated_extent = np.squeeze(
                        np.array((np.array([[extent.x, extent.y, extent.z, 1]]) @ _rotation_matrix)[:3])
                    )

                    bb = np.array([[loc.x, loc.y, loc.z], [rotated_extent[0], rotated_extent[1], rotated_extent[2]]])
                else:
                    loc = _obstacle.get_transform().location
                    bb = np.array([[loc.x, loc.y, loc.z], [0.5, 0.5, 2]])

                if return_with_type:
                    obst.append((bb, _obstacle.type_id))
                else:
                    obst.append(bb)

        return obst

    def _create_bb_points(self, bb):
        cords = np.zeros((8, 4))
        extent = bb[1]
        loc = bb[0]
        cords[0, :] = np.array(
            [loc[0] + extent[0], loc[1] + extent[1], loc[2] - extent[2], 1]
        )
        cords[1, :] = np.array(
            [loc[0] - extent[0], loc[1] + extent[1], loc[2] - extent[2], 1]
        )
        cords[2, :] = np.array(
            [loc[0] - extent[0], loc[1] - extent[1], loc[2] - extent[2], 1]
        )
        cords[3, :] = np.array(
            [loc[0] + extent[0], loc[1] - extent[1], loc[2] - extent[2], 1]
        )
        cords[4, :] = np.array(
            [loc[0] + extent[0], loc[1] + extent[1], loc[2] + extent[2], 1]
        )
        cords[5, :] = np.array(
            [loc[0] - extent[0], loc[1] + extent[1], loc[2] + extent[2], 1]
        )
        cords[6, :] = np.array(
            [loc[0] - extent[0], loc[1] - extent[1], loc[2] + extent[2], 1]
        )
        cords[7, :] = np.array(
            [loc[0] + extent[0], loc[1] - extent[1], loc[2] + extent[2], 1]
        )
        return cords
    def _get_camera_to_car_calibration(self, sensor):

        calibration = np.identity(3)
        calibration[0, 2] = sensor["width"] / 2.0
        calibration[1, 2] = sensor["height"] / 2.0
        calibration[0, 0] = calibration[1, 1] = sensor["width"] / (
            2.0 * np.tan(sensor["fov"] * np.pi / 360.0)
        )
        return calibration

    def _baseline_to_box(self, baseline, sensor_transform, height=1):

        cords_x_y_z = np.array(
            self._world_to_sensor(baseline, sensor_transform)[:3, :]
        )

        cords = np.hstack(
            (cords_x_y_z, np.fliplr(cords_x_y_z + np.array([[0], [0], [height]])))
        )

        return self._coords_to_2d_bb(cords)
    def _get_2d_bb_baseline(self, sensor_transform,obstacle, distance=2, cam="seg_front"):

        trigger = obstacle.trigger_volume
        bb = self._create_2d_bb_points(trigger)
        trig_loc_world = self._trig_to_world(bb, obstacle, trigger)

        cords_x_y_z = np.array(
            self._world_to_sensor(trig_loc_world, sensor_transform)
        )
        indices = (-cords_x_y_z[0]).argsort()

        # check crooked up boxes
        if self._get_dist(
            cords_x_y_z[:, indices[0]], cords_x_y_z[:, indices[1]]
        ) < self._get_dist(cords_x_y_z[:, indices[0]], cords_x_y_z[:, indices[2]]):
            cords = cords_x_y_z[:, [indices[0], indices[2]]] + np.array(
                [[distance], [0], [0], [0]]
            )
        else:
            cords = cords_x_y_z[:, [indices[0], indices[1]]] + np.array(
                [[distance], [0], [0], [0]]
            )

        sensor_world_matrix = self.get_matrix(sensor_transform)
        baseline = np.dot(sensor_world_matrix, cords)

        return baseline


    def _create_2d_bb_points(self, actor_bb, scale_factor=1):

        cords = np.zeros((4, 4))
        extent = actor_bb.extent
        x = extent.x * scale_factor
        y = extent.y * scale_factor
        z = extent.z * scale_factor
        cords[0, :] = np.array([x, y, 0, 1])
        cords[1, :] = np.array([-x, y, 0, 1])
        cords[2, :] = np.array([-x, -y, 0, 1])
        cords[3, :] = np.array([x, -y, 0, 1])
        return cords
    def _get_dist(self, p1, p2):

        distance = np.sqrt(
            (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2
        )

        return distance

    def _world_to_sensor(self, cords, sensor, move_cords=False):

        sensor_world_matrix = self.get_matrix(sensor)
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)

        if move_cords:
            _num_cords = range(sensor_cords.shape[1])
            modified_cords = np.array([])
            for i in _num_cords:
                if sensor_cords[0, i] < 0:
                    for j in _num_cords:
                        if sensor_cords[0, j] > 0:
                            _direction = sensor_cords[:, i] - sensor_cords[:, j]
                            _distance = -sensor_cords[0, j] / _direction[0]
                            new_cord = (
                                sensor_cords[:, j]
                                + _distance[0, 0] * _direction * 0.9999
                            )
                            modified_cords = (
                                np.hstack([modified_cords, new_cord])
                                if modified_cords.size
                                else new_cord
                            )
                else:
                    modified_cords = (
                        np.hstack([modified_cords, sensor_cords[:, i]])
                        if modified_cords.size
                        else sensor_cords[:, i]
                    )

            return modified_cords
        else:
            return sensor_cords
    def get_matrix(self, transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix

    def _trig_to_world(self, bb, parent, trigger):
        bb_transform = carla.Transform(trigger.location)
        bb_vehicle_matrix = self.get_matrix(bb_transform)
        vehicle_world_matrix = self.get_matrix(parent.get_transform())
        bb_world_matrix = vehicle_world_matrix @ bb_vehicle_matrix
        world_cords = bb_world_matrix @ bb.T
        return world_cords
    def _coords_to_2d_bb(self, cords):

        cords_y_minus_z_x = np.vstack((cords[1, :], -cords[2, :], cords[0, :]))

        bbox = (self._sensor_data["calibration"] @ cords_y_minus_z_x).T

        camera_bbox = np.vstack(
            [bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]]
        ).T

        if np.any(camera_bbox[:, 2] > 0):

            camera_bbox = np.array(camera_bbox)
            _positive_bb = camera_bbox[camera_bbox[:, 2] > 0]

            min_x = int(
                np.clip(np.min(_positive_bb[:, 0]), 0, self._sensor_data["width"])
            )
            min_y = int(
                np.clip(np.min(_positive_bb[:, 1]), 0, self._sensor_data["height"])
            )
            max_x = int(
                np.clip(np.max(_positive_bb[:, 0]), 0, self._sensor_data["width"])
            )
            max_y = int(
                np.clip(np.max(_positive_bb[:, 1]), 0, self._sensor_data["height"])
            )

            return [(min_x, min_y), (max_x, max_y)]
        else:
            return None

    def _find_obstacle(self, obstacle_type="*traffic_light*"):

        obst = list()

        _actors = self.world.get_actors()
        _obstacles = _actors.filter(obstacle_type)

        for _obstacle in _obstacles:
            trigger = _obstacle.trigger_volume

            _obstacle.get_transform().transform(trigger.location)
            distance_to_car = trigger.location.distance(self._vehicle.get_location())

            a = np.sqrt(
                trigger.extent.x**2 + trigger.extent.y**2 + trigger.extent.z**2
            )
            b = np.sqrt(
                self._vehicle.bounding_box.extent.x**2
                + self._vehicle.bounding_box.extent.y**2
                + self._vehicle.bounding_box.extent.z**2
            )

            s = a + b + 10

            if distance_to_car <= s:
                # the actor is affected by this obstacle.
                obst.append(_obstacle)

        return obst
    
    def _get_affordances(self):
    
        affordances = {}
        affordances["traffic_light"] = True

        affecting = self._vehicle.get_traffic_light()
        if affecting is not None:
            for light in self._traffic_lights:
                if light.id == affecting.id:
                    affordances["traffic_light"] = self._translate_tl_state(
                        self._vehicle.get_traffic_light_state()
                    )

        return affordances
    def _translate_tl_state(self, state):

        if state == carla.TrafficLightState.Red:
            return 0
        elif state == carla.TrafficLightState.Yellow:
            return 1
        elif state == carla.TrafficLightState.Green:
            return 2
        elif state == carla.TrafficLightState.Off:
            return 3
        elif state == carla.TrafficLightState.Unknown:
            return 4
        else:
            return None

    def modify_vehicle_physics(self, actor):
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def destroy(self):
        actors = []

        if self.collision_sensor is not None and self.collision_sensor.sensor is not None:
            actors.append(self.collision_sensor.sensor)
        if self.lane_invasion_sensor is not None and self.lane_invasion_sensor.sensor is not None:
            actors.append(self.lane_invasion_sensor.sensor)
        if self.gnss_sensor is not None and self.gnss_sensor.sensor is not None:
            actors.append(self.gnss_sensor.sensor)
        if self.player is not None:
            actors.append(self.player)

        if hasattr(self, "sensors"):
            actors.extend(self.sensors)

        for actor in actors:
            if actor is not None:
                try:
                    actor.destroy()
                except:
                    pass

        # 리스트 초기화 (안 해주면 중복 destroy 시도할 수 있음)
        self.npc_vehicle_list = []
        self.walker_actors = []
        self.walker_controllers = []
        self.sensors = []


class GnssSensor:
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        blueprint = self._parent.get_world().get_blueprint_library().find('sensor.other.gnss')
        self.sensor = self._parent.get_world().spawn_actor(blueprint, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                                           attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

class CollisionSensor:
    def __init__(self, parent_actor):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = {}
        for frame, intensity in self.history:
            history[frame] = history.get(frame, 0) + intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)

class LaneInvasionSensor:
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        bp = self._parent.get_world().get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = self._parent.get_world().spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        pass

class CameraSensor_RGB:
    def __init__(self, parent_actor,conf):
        self.sensor = None
        self._parent = parent_actor
        world = self._parent.get_world()

        bp = world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(conf['width']))
        bp.set_attribute('image_size_y', str(conf['height']))
        bp.set_attribute('fov', str(conf['fov']))

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

class CameraSensor_Seg:
    def __init__(self, parent_actor,conf):
        self.sensor = None
        self._parent = parent_actor
        world = self._parent.get_world()

        bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        bp.set_attribute('image_size_x', str(conf['width']))
        bp.set_attribute('image_size_y', str(conf['height']))
        bp.set_attribute('fov', str(conf['fov']))
        self.sensor = world.spawn_actor(bp, carla.Transform(
        carla.Location(x=conf['x'], y=conf['y'], z=conf['z']),                                              
        carla.Rotation(pitch=conf['pitch'], roll=conf['roll'], yaw=conf['yaw'])
        ), attach_to=self._parent)

        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CameraSensor_Seg._on_image(weak_self, image))
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


class CameraSensor_Depth:
    def __init__(self, parent_actor,conf):
        self.sensor = None
        self._parent = parent_actor
        world = self._parent.get_world()

        bp = world.get_blueprint_library().find('sensor.camera.depth')

        bp.set_attribute('image_size_x', str(conf['width']))
        bp.set_attribute('image_size_y', str(conf['height']))
        bp.set_attribute('fov', str(conf['fov']))
        
        self.sensor = world.spawn_actor(bp, carla.Transform(
        carla.Location(x=conf['x'], y=conf['y'], z=conf['z']),                                              
        carla.Rotation(pitch=conf['pitch'], roll=conf['roll'], yaw=conf['yaw'])
        ), attach_to=self._parent)

        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CameraSensor_Depth._on_image(weak_self, image))
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

        bp.set_attribute('channels', '64')
        bp.set_attribute('range', '100')
        bp.set_attribute('points_per_second', '1200000')
        bp.set_attribute('rotation_frequency', '60')

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
        self.point_cloud=points*2.0


class Lidar_seg_Sensor:
    def __init__(self, parent_actor,conf):
        self.sensor = None
        self._parent = parent_actor
        world = self._parent.get_world()

        bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')

        bp.set_attribute('channels', '64')
        bp.set_attribute('range', '100')
        bp.set_attribute('points_per_second', '1200000')
        bp.set_attribute('rotation_frequency', '60')

        self.sensor = world.spawn_actor(bp, carla.Transform(
        carla.Location(x=conf['x'], y=conf['y'], z=conf['z']),                                              
        carla.Rotation(pitch=conf['pitch'], roll=conf['roll'], yaw=conf['yaw'])
        ), attach_to=self._parent)

        weak_self = weakref.ref(self)
        self.sensor.listen(lambda point_cloud: LidarSensor._on_point_cloud(weak_self, point_cloud))
        self.point_seg_cloud=None

    @staticmethod
    def _on_point_cloud(weak_self, lidar_data):
        self = weak_self()
        if not self:
            return
        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 6), 6))
        self.point_seg_cloud=points

class ImuSensor:
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.imu_data = None

        blueprint = self._parent.get_world().get_blueprint_library().find('sensor.other.imu')
        self.sensor = self._parent.get_world().spawn_actor(
            blueprint,
            carla.Transform(carla.Location(x=0.0, z=2.0)),  # 위치 조정 가능
            attach_to=self._parent
        )

        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: ImuSensor._on_imu_event(weak_self, event))

    @staticmethod
    def _on_imu_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.imu_data = {
            "accelerometer": {
                "x": event.accelerometer.x,
                "y": event.accelerometer.y,
                "z": event.accelerometer.z
            },
            "gyroscope": {
                "x": event.gyroscope.x,
                "y": event.gyroscope.y,
                "z": event.gyroscope.z
            },
            "compass": event.compass  # 방향성 (yaw)
        }


def get_nearby_lights(vehicle, lights, pixels_per_meter=5.5, size=512, radius=5):
    result = list()

    transform = vehicle.get_transform()
    pos = transform.location
    theta = np.radians(90 + transform.rotation.yaw)
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )

    for light in lights:
        delta = light.get_transform().location - pos

        target = R.T.dot([delta.x, delta.y])
        target *= pixels_per_meter
        target += size // 2

        if min(target) < 0 or max(target) >= size:
            continue

        trigger = light.trigger_volume
        light.get_transform().transform(trigger.location)
        dist = trigger.location.distance(vehicle.get_location())
        a = np.sqrt(
            trigger.extent.x**2 + trigger.extent.y**2 + trigger.extent.z**2
        )
        b = np.sqrt(
            vehicle.bounding_box.extent.x**2
            + vehicle.bounding_box.extent.y**2
            + vehicle.bounding_box.extent.z**2
        )

        if dist > a + b:
            continue

        result.append(light)

    return result