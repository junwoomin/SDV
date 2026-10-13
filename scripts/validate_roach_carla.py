#!/usr/bin/env python3
"""Run a reproducible *live* CARLA comparison, or record why it is blocked.

No CARLA installation is required for --preflight (the default). --run reloads
its configured Town for EACH episode, discarding existing world actors. Run on
a dedicated CARLA server. No result is marked passed merely because a vehicle
moved. Scenario coverage and driving metrics are recorded separately.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import socket
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sdv.evaluation.driving_metrics import DrivingMetrics


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def preflight(args, cases):
    modules = {name: importlib.util.find_spec(name) is not None
               for name in ('carla', 'torch', 'numpy', 'cv2', 'pygame', 'yaml')}
    blockers = ['Missing Python module: ' + name for name, present in modules.items() if not present]
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if 'roach' in args.agents and not checkpoint.is_file():
        blockers.append('Roach checkpoint is absent: ' + str(checkpoint))
    server = {'host': args.host, 'port': args.port, 'tcp_reachable': False}
    try:
        with socket.create_connection((args.host, args.port), timeout=args.timeout):
            server['tcp_reachable'] = True
    except OSError as error:
        server['error'] = str(error)
        blockers.append('CARLA RPC port is unreachable; live execution is not possible')
    for case in cases:
        if case.get('start_spawn_index') is None or case.get('destination_spawn_index') is None:
            blockers.append('Route is not configured for scenario: ' + case['name'])
    return {'status': 'blocked' if blockers else 'preflight_ready', 'live_execution': False,
            'python': sys.version, 'modules': modules, 'server': server,
            'checkpoint': str(checkpoint), 'checkpoint_exists': checkpoint.is_file(),
            'blockers': blockers, 'unverified': [
                'Checkpoint architecture, integrity and original LRS observation compatibility require the runtime loader',
                'CUDA availability and model execution require the selected runtime',
                'CARLA client/server API version, Town and spawn indices are checked only after RPC connection',
                'All real driving scenarios, metrics and controller comparison remain unmeasured until --run'],
            'configured_scenarios': cases}


def location_list(location):
    return [float(getattr(location, axis)) for axis in ('x', 'y', 'z')]


def collect_stoplines(world):
    """Native CARLA affected-lane stop points only; never lamp-pole positions."""
    entries = []
    missing_api = []
    for light in world.get_actors().filter('traffic.traffic_light*'):
        if not hasattr(light, 'get_stop_waypoints'):
            missing_api.append(light.id)
            continue
        for waypoint in light.get_stop_waypoints():
            forward = waypoint.transform.get_forward_vector()
            entries.append({'id': '{}:{}:{}:{:.3f}'.format(light.id, waypoint.road_id, waypoint.lane_id, waypoint.s),
                            'center': location_list(waypoint.transform.location),
                            'forward': [forward.x, forward.y], 'half_width_m': waypoint.lane_width / 2,
                            'actor': light, 'road_id': waypoint.road_id, 'lane_id': waypoint.lane_id})
    return entries, missing_api


def line_states(entries):
    return [dict({key: value for key, value in line.items() if key != 'actor'},
                 state=str(line['actor'].get_state()).split('.')[-1]) for line in entries]


def make_fixture(world, route, case, carla):
    """Optional deterministic physical lead vehicle / crossing pedestrian.

    Fixtures are explicit. Their occurrence is logged, not interpreted as proof
    that Roach responded correctly. No actor pose is overwritten during a run.
    """
    fixtures = []
    for name in ('lead_vehicle', 'pedestrian'):
        config = case.get(name)
        if not config:
            continue
        distance, waypoint = 0.0, None
        for (previous, _), (current, _) in zip(route, route[1:]):
            distance += previous.transform.location.distance(current.transform.location)
            if distance >= float(config.get('route_distance_m', 25)):
                waypoint = current
                break
        if waypoint is None:
            raise ValueError(name + ' fixture lies beyond the planned route')
        transform = carla.Transform(waypoint.transform.location, waypoint.transform.rotation)
        transform.location.z += 0.7
        right = waypoint.transform.get_right_vector()
        if name == 'pedestrian':
            offset = float(config.get('lateral_offset_m', 4.0))
            transform.location.x += right.x * offset
            transform.location.y += right.y * offset
            blueprint = sorted(world.get_blueprint_library().filter('walker.pedestrian.*'), key=lambda item: item.id)[0]
        else:
            blueprint = world.get_blueprint_library().find(config.get('blueprint', 'vehicle.tesla.model3'))
        actor = world.try_spawn_actor(blueprint, transform)
        if actor is None:
            raise RuntimeError('Could not spawn required ' + name + ' fixture')
        fixtures.append({'kind': name, 'actor': actor, 'config': config, 'right': right, 'triggered': False,
                         'spawn_transform': {'location': location_list(transform.location), 'yaw': transform.rotation.yaw}})
    return fixtures


def step_fixtures(fixtures, elapsed, ego, carla):
    for fixture in fixtures:
        config, actor = fixture['config'], fixture['actor']
        if fixture['kind'] == 'lead_vehicle':
            brake = elapsed >= float(config.get('brake_after_s', 8.0))
            velocity = actor.get_velocity()
            speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
            throttle = max(0.0, min(0.5, (float(config.get('speed_mps', 3.0)) - speed) * .25))
            actor.apply_control(carla.VehicleControl(throttle=0.0 if brake else throttle, brake=1.0 if brake else 0.0))
            fixture['triggered'] |= brake
        elif fixture['triggered'] or ego.get_location().distance(actor.get_location()) < float(config.get('trigger_distance_m', 15)):
            right = fixture['right']
            actor.apply_control(carla.WalkerControl(direction=carla.Vector3D(x=-right.x, y=-right.y),
                                                    speed=float(config.get('speed_mps', 1.4))))
            fixture['triggered'] = True


def run_episode(client, args, case, mode, output):
    import carla
    from sdv.collection.driving_runtime import DrivingRuntime
    from sdv.collection.carla_engine.agents.navigation.global_route_planner import GlobalRoutePlanner
    world = client.load_world(case.get('town', args.town))
    previous_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = args.fixed_delta
    settings.substepping = True
    settings.max_substep_delta_time = .01
    settings.max_substeps = max(10, int(math.ceil(args.fixed_delta / .01)))
    world.apply_settings(settings)
    weather_name = case.get('weather', 'ClearNoon')
    world.set_weather(getattr(carla.WeatherParameters, weather_name))
    if hasattr(world, 'set_pedestrians_seed'):
        world.set_pedestrians_seed(args.seed)
    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(args.seed)
    owned, sensors, runtime, stream, metrics = [], [], None, None, None
    collisions, invasions = [], []
    result = {'scenario': case['name'], 'agent': mode, 'live_execution': True,
              'status': 'error', 'seed': args.seed, 'town': world.get_map().name,
              'weather': weather_name, 'fixed_delta_seconds': args.fixed_delta,
              'client_version': client.get_client_version(), 'server_version': client.get_server_version(),
              'baseline': 'migrated SDV BehaviorAgent with random steering removed and corrected PID dt' if mode == 'existing' else mode}
    try:
        if client.get_client_version().split('-')[0] != client.get_server_version().split('-')[0]:
            raise RuntimeError('CARLA Python client/server version mismatch')
        spawn_points = world.get_map().get_spawn_points()
        start_index, destination_index = int(case['start_spawn_index']), int(case['destination_spawn_index'])
        if not (0 <= start_index < len(spawn_points) and 0 <= destination_index < len(spawn_points)):
            raise ValueError('Scenario spawn index is outside this Town spawn point list')
        start, destination = spawn_points[start_index], spawn_points[destination_index]
        route = GlobalRoutePlanner(world.get_map(), 1.0).trace_route(start.location, destination.location)
        if len(route) < 3:
            raise ValueError('Route is too short to evaluate')
        route_data = [{'world': location_list(wp.transform.location), 'road_option': option.name,
                       'road_id': wp.road_id, 'lane_id': wp.lane_id, 'is_junction': bool(wp.is_junction)}
                      for wp, option in route]
        result['route_sha256'] = hashlib.sha256(json.dumps(route_data, sort_keys=True).encode()).hexdigest()
        result['start_spawn_index'], result['destination_spawn_index'] = start_index, destination_index
        result['route_options'] = sorted(set(point['road_option'] for point in route_data))
        requirements = case.get('requires', {})
        missing = set(requirements.get('route_options', [])) - set(result['route_options'])
        if missing:
            raise ValueError('Configured route does not contain required maneuvers: ' + ', '.join(sorted(missing)))
        forbidden = set(requirements.get('forbidden_route_options', [])) & set(result['route_options'])
        if forbidden:
            raise ValueError('Configured route contains forbidden maneuvers: ' + ', '.join(sorted(forbidden)))
        if requirements.get('any_route_options') and not set(requirements['any_route_options']) & set(result['route_options']):
            raise ValueError('Configured route lacks a requested alternative maneuver')
        if requirements.get('junction') and not any(point['is_junction'] for point in route_data):
            raise ValueError('Configured route does not cross a junction')
        write_json(output / 'global_route.json', route_data)
        blueprint = world.get_blueprint_library().find(args.ego_blueprint)
        blueprint.set_attribute('role_name', 'sdv_evaluation_ego')
        ego = world.try_spawn_actor(blueprint, start)
        if ego is None:
            raise RuntimeError('Ego spawn failed; evaluation never chooses a different spawn silently')
        owned.append(ego)
        ego.set_autopilot(False)
        ego.apply_control(carla.VehicleControl(brake=1.0))
        rng = random.Random(args.seed)
        indices = list(range(len(spawn_points)))
        rng.shuffle(indices)
        npc_manifest = []
        choices = sorted(world.get_blueprint_library().filter('vehicle.*'), key=lambda item: item.id)
        requested_traffic = int(case.get('traffic_count', args.traffic_count))
        for index in indices:
            if len(npc_manifest) >= requested_traffic:
                break
            if spawn_points[index].location.distance(start.location) < 12:
                continue
            candidate = choices[rng.randrange(len(choices))]
            npc = world.try_spawn_actor(candidate, spawn_points[index])
            if npc is not None:
                owned.append(npc)
                npc.set_autopilot(True, args.tm_port)
                npc_manifest.append({'spawn_index': index, 'blueprint': candidate.id})
        if len(npc_manifest) != requested_traffic:
            raise RuntimeError('Unable to create the exact requested traffic fixture')
        result['traffic_manifest'] = npc_manifest
        fixtures = make_fixture(world, route, case, carla)
        owned.extend(fixture['actor'] for fixture in fixtures)
        lines, missing_api = collect_stoplines(world)
        result['native_stopline_count'] = len(lines)
        result['stopline_api_unavailable_actor_ids'] = missing_api
        if requirements.get('red_stop'):
            if not lines:
                raise RuntimeError('No native traffic-light stopline is available for this scenario')
            for light in world.get_actors().filter('traffic.traffic_light*'):
                light.set_state(carla.TrafficLightState.Red)
                light.freeze(True)
        collision = world.spawn_actor(world.get_blueprint_library().find('sensor.other.collision'), carla.Transform(), attach_to=ego)
        sensors.append(collision)
        collision.listen(lambda event: collisions.append({'frame': int(event.frame), 'other_actor_id': event.other_actor.id,
                                                          'other_actor_type': event.other_actor.type_id,
                                                          'normal_impulse': location_list(event.normal_impulse)}))
        lane = world.spawn_actor(world.get_blueprint_library().find('sensor.other.lane_invasion'), carla.Transform(), attach_to=ego)
        sensors.append(lane)
        lane.listen(lambda event: invasions.append({'frame': int(event.frame), 'markings': [str(m.type) for m in event.crossed_lane_markings]}))
        for _ in range(15):
            world.tick()
        if collisions:
            raise RuntimeError('Collision during fixture initialization; the initial condition is invalid')
        invasions.clear()
        config = dict(agent=mode, checkpoint=str(Path(args.checkpoint).resolve()), device=args.device,
                      deterministic=True, policy_ppm=5, control_hz=args.control_hz, seed=args.seed)
        bev_settings = json.loads(Path(args.bev_config).read_text()) if args.bev_config else None
        collision_events = SimpleNamespace(events=collisions, get_collision_history=lambda: {event['frame']: math.sqrt(sum(v*v for v in event['normal_impulse'])) for event in collisions})
        runtime = DrivingRuntime(world, ego, config, fixed_delta_seconds=args.fixed_delta, traffic_manager=tm, bev_settings=bev_settings,
                                 collision_sensor=collision_events, lane_sensor=SimpleNamespace(events=invasions))
        runtime.set_global_plan(route)
        if runtime.mode != mode or any(runtime.config.get(key) != value for key, value in config.items()):
            raise RuntimeError('SDV_DRIVING_CONFIG overrode explicit comparison settings; unset it for evaluation')
        result['effective_driving_config'] = dict(runtime.config)
        metrics = DrivingMetrics([point['world'] for point in route_data], args.fixed_delta)
        metrics.collisions, metrics.lane_invasions = collisions, invasions
        output.mkdir(parents=True, exist_ok=True)
        stream = (output / 'frames.jsonl').open('w', encoding='utf-8')
        duration = float(case.get('duration_s', args.duration))
        max_steps = int(math.ceil(duration / args.fixed_delta))
        reached_options, junction_seen, red_stop_s = set(), False, 0.0
        lead_stop_s, maximum_speed_mps = 0.0, 0.0
        started_sim = world.get_snapshot().timestamp.elapsed_seconds
        for step_index in range(max_steps + 1):
            snapshot = world.get_snapshot()
            if snapshot.find(ego.id) is None:
                raise RuntimeError('Ego missing from current WorldSnapshot')
            state = snapshot.find(ego.id)
            transform, velocity = state.get_transform(), state.get_velocity()
            position = transform.location
            front = transform.transform(carla.Location(x=ego.bounding_box.location.x + ego.bounding_box.extent.x,
                                                        y=ego.bounding_box.location.y, z=ego.bounding_box.location.z))
            current_lines = line_states(lines)
            heading = transform.get_forward_vector()
            # A crossing of another approach's line does not apply to ego.
            current_lines = [line for line in current_lines
                             if heading.x * line['forward'][0] + heading.y * line['forward'][1] > .5]
            waypoint = world.get_map().get_waypoint(position, project_to_road=False, lane_type=carla.LaneType.Driving)
            speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
            maximum_speed_mps = max(maximum_speed_mps, speed)
            for fixture in fixtures:
                if fixture['kind'] == 'lead_vehicle' and fixture['triggered'] and speed < .2:
                    gap = position.distance(fixture['actor'].get_location()) - ego.bounding_box.extent.x - fixture['actor'].bounding_box.extent.x
                    if 1 <= gap <= 20:
                        lead_stop_s += args.fixed_delta
            began = time.perf_counter()
            runtime.run_step(snapshot)
            latency = (time.perf_counter() - began) * 1000
            row = metrics.sample(frame=snapshot.frame, timestamp=snapshot.timestamp.elapsed_seconds,
                                 position=location_list(position), front_position=location_list(front), speed_mps=speed,
                                 offroad=waypoint is None, stoplines=current_lines,
                                 inference_ms=runtime.last_policy.get('inference_ms') if runtime.last_policy.get('inference_ran') else None,
                                 control_latency_ms=latency)
            point = route_data[min(row['segment'], len(route_data) - 1)]
            if row['deviation_m'] is not None and row['deviation_m'] < 3:
                reached_options.add(point['road_option'])
                junction_seen |= point['is_junction']
            if requirements.get('red_stop') and speed < .2:
                for line in current_lines:
                    delta = [front.x - line['center'][0], front.y - line['center'][1]]
                    along = sum(a * b for a, b in zip(delta, line['forward']))
                    lateral = abs(-delta[0] * line['forward'][1] + delta[1] * line['forward'][0])
                    if line['state'] == 'Red' and -12 <= along < 0 and lateral < line['half_width_m'] and abs(front.z - line['center'][2]) < 2:
                        red_stop_s += args.fixed_delta
                        break
            row.update(policy=runtime.last_policy, navigation=runtime.last_navigation)
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            elapsed = snapshot.timestamp.elapsed_seconds - started_sim
            step_fixtures(fixtures, elapsed, ego, carla)
            if red_stop_s >= 2:
                for light in world.get_actors().filter('traffic.traffic_light*'):
                    light.set_state(carla.TrafficLightState.Green)
            if runtime.done() or step_index == max_steps:
                break
            world.tick()
        result['metrics'] = metrics.report()
        result['arrived'] = runtime.done()
        result['fixture_events'] = [{key: value for key, value in fixture.items() if key not in ('actor', 'right')} for fixture in fixtures]
        coverage = {'route_options': sorted(reached_options), 'junction_seen': junction_seen,
                    'red_stop_seconds': red_stop_s, 'elapsed_simulation_s': elapsed,
                    'lead_stop_seconds': lead_stop_s, 'maximum_speed_mps': maximum_speed_mps,
                    'lead_brake_triggered': any(f['kind'] == 'lead_vehicle' and f['triggered'] for f in fixtures),
                    'pedestrian_triggered': any(f['kind'] == 'pedestrian' and f['triggered'] for f in fixtures),
                    'overpass_actor_separation': 'not_automatically_verified'}
        result['coverage'] = coverage
        unverified = []
        if not set(requirements.get('route_options', [])).issubset(reached_options): unverified.append('required turn maneuver not reached')
        if requirements.get('any_route_options') and not set(requirements['any_route_options']) & reached_options: unverified.append('required alternative maneuver not reached')
        if requirements.get('junction') and not junction_seen: unverified.append('junction not reached')
        if requirements.get('red_stop') and red_stop_s < 2: unverified.append('ego did not stop at a red stopline for 2 seconds')
        if requirements.get('lead_braking') and (not coverage['lead_brake_triggered'] or lead_stop_s < 2 or maximum_speed_mps < .5 or metrics.route.progress_m < 5): unverified.append('lead braking encounter and ego stop response not demonstrated')
        if requirements.get('pedestrian') and not coverage['pedestrian_triggered']: unverified.append('pedestrian fixture not triggered')
        if requirements.get('overpass'): unverified.append('requires review of per-frame HBLR actor inclusion against independently labelled upper/lower roads')
        if requirements.get('destination') and not result['arrived']: unverified.append('destination not reached')
        if requirements.get('continuous_seconds', 0) > coverage['elapsed_simulation_s']: unverified.append('requested continuous duration not reached')
        if missing_api: unverified.append('native stopline API missing; red-light metric is incomplete')
        result['unverified'] = unverified
        result['status'] = 'measured_with_unverified_items' if unverified else 'measured'
        result['safety_criteria_met'] = (result['metrics']['collision_count'] == 0 and
                                       result['metrics']['traffic_light_violation_count'] == 0 and
                                       result['metrics']['offroad_center_frames'] == 0 and
                                       result['metrics']['route_height_mismatch_frames'] == 0 and
                                       result['metrics']['route_deviation_max_m'] is not None and
                                       result['metrics']['route_deviation_max_m'] <= float(case.get('max_route_deviation_m', 3)))
        result['scenario_passed'] = not unverified and result['safety_criteria_met']
    except Exception as error:
        result['error'] = '{}: {}'.format(type(error).__name__, error)
        result['traceback'] = traceback.format_exc()
        if runtime is not None:
            result['last_policy'] = runtime.last_policy
        if metrics is not None:
            result['metrics'] = metrics.report()
        result['scenario_passed'] = False
    finally:
        if runtime is not None:
            runtime.close()
        if stream is not None:
            stream.close()
        for sensor in sensors:
            try: sensor.stop()
            except RuntimeError: pass
        for actor in list(reversed(sensors)) + list(reversed(owned)):
            try: actor.destroy()
            except RuntimeError: pass
        for light in world.get_actors().filter('traffic.traffic_light*'):
            light.freeze(False)
        tm.set_synchronous_mode(False)
        world.apply_settings(previous_settings)
        write_json(output / 'report.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--run', action='store_true', help='Reload Town for each controller and run live physics')
    action.add_argument('--preflight', action='store_true', help='Default: inspect dependencies and RPC availability only')
    action.add_argument('--list-spawns', action='store_true', help='Read current CARLA Town spawn indices without resetting world')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--tm-port', type=int, default=8000)
    parser.add_argument('--timeout', type=float, default=5)
    parser.add_argument('--town', default='Town03')
    parser.add_argument('--start-spawn', type=int)
    parser.add_argument('--destination-spawn', type=int)
    parser.add_argument('--scenario-file', type=Path)
    parser.add_argument('--case', action='append', help='Names to select from scenario file; can be repeated')
    parser.add_argument('--agents', nargs='+', choices=['existing', 'roach', 'autopilot'], default=['existing', 'roach'])
    parser.add_argument('--checkpoint', default=str(ROOT / 'roach/log/ckpt_11833344.pth'))
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--bev-config', help='JSON file with same BEV settings as the collector')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--duration', type=float, default=120)
    parser.add_argument('--fixed-delta', type=float, default=1 / 30)
    parser.add_argument('--control-hz', type=float, default=30)
    parser.add_argument('--traffic-count', type=int, default=0)
    parser.add_argument('--ego-blueprint', default='vehicle.lincoln.mkz_2017')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/roach_live')
    args = parser.parse_args(argv)
    if args.fixed_delta <= 0 or args.duration <= 0 or args.traffic_count < 0:
        parser.error('fixed-delta and duration must be positive, traffic-count nonnegative')
    if args.list_spawns:
        import carla
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)
        world = client.get_world()
        data = {'town': world.get_map().name, 'spawns': [{'index': i, 'world': location_list(t.location), 'yaw': t.rotation.yaw}
                for i, t in enumerate(world.get_map().get_spawn_points())]}
        write_json(args.output / 'spawn_points.json', data)
        print(json.dumps(data, indent=2))
        return 0
    if args.scenario_file:
        cases = json.loads(args.scenario_file.read_text(encoding='utf-8'))['scenarios']
        if args.case:
            cases = [case for case in cases if case['name'] in args.case]
            if set(args.case) - {case['name'] for case in cases}:
                parser.error('Unknown scenario name')
    else:
        cases = [dict(name='configured_route', town=args.town, start_spawn_index=args.start_spawn,
                      destination_spawn_index=args.destination_spawn, requires={'destination': True})]
    for case in cases:
        if args.start_spawn is not None: case['start_spawn_index'] = args.start_spawn
        if args.destination_spawn is not None: case['destination_spawn_index'] = args.destination_spawn
    readiness = preflight(args, cases)
    write_json(args.output / 'preflight.json', readiness)
    if not args.run or readiness['blockers']:
        print(json.dumps(readiness, indent=2, ensure_ascii=False))
        return 2 if readiness['blockers'] else 0
    import carla
    client = carla.Client(args.host, args.port)
    client.set_timeout(max(30, args.timeout))
    episodes = []
    for case in cases:
        for mode in args.agents:
            print('Running {} / {} on dedicated CARLA server'.format(case['name'], mode), flush=True)
            episodes.append(run_episode(client, args, case, mode, args.output / case['name'] / mode))
    pairs = []
    for case in cases:
        selected = [result for result in episodes if result['scenario'] == case['name']]
        route_hashes = {result.get('route_sha256') for result in selected}
        traffic = {json.dumps(result.get('traffic_manifest'), sort_keys=True) for result in selected}
        pairs.append({'scenario': case['name'], 'same_route': len(route_hashes) == 1 and None not in route_hashes,
                      'same_initial_traffic': len(traffic) == 1 and 'null' not in traffic,
                      'conditions': 'Same seed, spawn endpoints, weather and NPC initial fixtures; divergent closed-loop traffic is expected',
                      'agents': {result['agent']: result.get('metrics') for result in selected},
                      'interpretation': 'Compare measured values only; stochastic/closed-loop runs need repetitions before claiming improved stability'})
    summary = {'live_execution': True, 'episodes': episodes, 'comparisons': pairs,
               'validation_limitations': ['Evaluation uses native actor/collision/lane data, not rendered RGB sensor validation',
                   'This harness does not certify collector RGB/segmentation/depth/BEV disk alignment; use collection frame metadata checks',
                   'Passing route movement cannot establish red-light, pedestrian, lead-braking or overpass coverage without the fixture evidence',
                   'An overpass scenario always remains unverified until actor layer inclusion is independently reviewed']}
    write_json(args.output / 'comparison.json', summary)
    print(str(args.output / 'comparison.json'))
    return 0 if all(result.get('scenario_passed') for result in episodes) else 1


if __name__ == '__main__':
    raise SystemExit(main())
