#!/usr/bin/env python3
"""Validate real CARLA Map geometry without running a simulator server.

Example (CARLA0.9.15 Python package required):
  python scripts/validate_bev_offline_carla.py --xodr-root /path/to/LRS/sim_using_data/Town --town Town01 --town Town04 --town Town05 --cache-root /tmp/sdv-bev-validation-cache

The map/elevation observations are real CARLA/OpenDRIVE data. Actors used for
the performance stress test are explicitly synthetic boxes placed on real
waypoints, never recorded/simulated traffic or proof of sensor synchronization.
"""
import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import textwrap
import types
from types import SimpleNamespace

import numpy as np
import carla
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sdv.collection.bev_adapter import build_cache
from sdv.bev.generator import HeightAwareBEVGenerator
from sdv.bev.cache import _barycentric


def verify_optimized_equivalence(render, ego, actors):
    """Compare bounded graph/vector selection with unbounded/full-frame work.

    This deliberately slow reference differs only in those optimization
    operations. It does not claim to reproduce an unrelated earlier release.
    """
    import sdv.bev.generator as module
    source = textwrap.dedent(inspect.getsource(HeightAwareBEVGenerator.generate))
    bounded = "candidate_ids,max_distance=diagonal+4."
    vector = "np.isin(arrays['surface_id'][rr,cc],affiliated)"
    if bounded not in source or vector not in source:
        raise ValueError('Generator changed: review optimization reference before running equivalence check')
    source = source.replace(bounded, 'candidate_ids').replace(vector, "np.isin(arrays['surface_id'],affiliated)[rr,cc]")
    namespace = dict(module.__dict__)
    exec(compile(source, '<unbounded_full_frame_reference>', 'exec'), namespace)
    reference = HeightAwareBEVGenerator(render.cache, render.settings)
    reference.generate = types.MethodType(namespace['generate'], reference)
    actual = render.generate(ego, actors, frame_id=0, timestamp=0)
    expected = reference.generate(ego, actors, frame_id=0, timestamp=0)
    channels = {}
    for name, array in actual['arrays'].items():
        target = expected['arrays'][name]
        equal = (array == target) | (np.isnan(array) & np.isnan(target)) if array.dtype.kind == 'f' else array == target
        channels[name] = int(np.count_nonzero(~equal))
    if any(channels.values()):
        raise AssertionError('Performance optimization changed labels: ' + str(channels))
    return {'reference': 'unbounded graph traversal and full-frame surface membership',
            'all_channels_identical_including_nan': True, 'mismatch_pixels': channels}


def pose_from_waypoint(wp, actor_id=1):
    t, location = wp.transform, wp.transform.location
    theta = math.radians(t.rotation.yaw)
    rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    footprint = np.array([[-2.25, -1], [2.25, -1], [2.25, 1], [-2.25, 1]]).dot(rotation.T) + (location.x, location.y)
    return {'actor_id': actor_id, 'object_class': 'vehicle', 'x': location.x, 'y': location.y,
        'z': location.z + 0.3, 'yaw': t.rotation.yaw, 'ground_z_m': location.z,
        'ground_source': 'synthetic box on actual CARLA OpenDRIVE waypoint',
        'road_id': wp.road_id, 'lane_id': wp.lane_id, 'section_id': wp.section_id,
        'bbox_min_z_m': location.z + .05, 'bbox_max_z_m': location.z + 1.8,
        'object_height_m': 1.75,
        'footprint_world_xyz': [[float(x), float(y), location.z] for x, y in footprint]}


def elevation_at(cache, patch, x, y):
    for tri in cache._triangulations[patch['id']]:
        vertices = [patch['polygon_xyz'][i] for i in tri]
        weights = _barycentric(x, y, *vertices)
        if weights is not None and min(weights) >= -1e-7:
            return sum(w * v[2] for w, v in zip(weights, vertices))
    return None


def verify_map(town, xodr_path, cache_root, spacing=1, benchmark_runs=10, performance=False):
    raw = xodr_path.read_text(encoding='utf-8')
    carla_map = carla.Map(town, raw)
    started = time.perf_counter()
    cache = build_cache(SimpleNamespace(get_map=lambda: carla_map), cache_root, {'meters_per_pixel': .2})
    build_ms = (time.perf_counter() - started) * 1000
    waypoints = carla_map.generate_waypoints(spacing)
    failures, errors = [], []
    slopes = 0
    junctions = 0
    for wp in waypoints:
        loc = wp.transform.location
        hit = cache.lookup_ground(loc.x, loc.y, loc.z, road_id=wp.road_id,
                                  lane_id=wp.lane_id, section_id=wp.section_id)
        if hit is None:
            failures.append({'road_id': wp.road_id, 'lane_id': wp.lane_id,
                             'section_id': wp.section_id, 's_m': wp.s,
                             'xyz': [loc.x, loc.y, loc.z]})
        else:
            errors.append(abs(hit['ground_z_m'] - loc.z))
        pitch = (float(wp.transform.rotation.pitch) + 180) % 360 - 180
        slopes += abs(pitch) > 1
        junctions += bool(wp.is_junction)
    # Find actual overlapping different-height road surfaces, without assuming
    # a Town-specific Z threshold or inventing overpass locations.
    crossings = []
    seen = set()
    for wp in waypoints:
        loc = wp.transform.location
        if loc.z < 2:
            continue
        own = cache.lookup_ground(loc.x, loc.y, loc.z, road_id=wp.road_id,
                                  lane_id=wp.lane_id, section_id=wp.section_id)
        if own is None:
            continue
        for patch in cache.query((loc.x, loc.y, loc.x, loc.y)):
            height = elevation_at(cache, patch, loc.x, loc.y)
            if height is None or loc.z - height < 2:
                continue
            pair = (wp.road_id, patch['road_id'])
            if pair in seen:
                continue
            low = cache.lookup_ground(loc.x, loc.y, height, road_id=patch['road_id'],
                                      lane_id=patch['lane_id'], section_id=patch['section_id'])
            if low is None:
                continue
            high_unconstrained = cache.lookup_ground(loc.x, loc.y, loc.z)
            low_unconstrained = cache.lookup_ground(loc.x, loc.y, height)
            crossings.append({'xy': [loc.x, loc.y],
                'upper_ground_z_m': own['ground_z_m'], 'lower_ground_z_m': low['ground_z_m'],
                'upper_road_id': wp.road_id, 'lower_road_id': patch['road_id'],
                'upper_surface_kind': cache.patches[own['patch_id']]['kind'],
                'lower_surface_kind': patch['kind'],
                'upper_surface': own['patch_id'], 'lower_surface': low['patch_id'],
                'unconstrained_ground_correct': high_unconstrained is not None and low_unconstrained is not None
                    and abs(high_unconstrained['ground_z_m']-loc.z) < .1
                    and abs(low_unconstrained['ground_z_m']-height) < .1})
            seen.add(pair)
        if len(crossings) >= 12:
            break
    result = {'town': town, 'input_xodr_sha256': hashlib.sha256(raw.encode('utf-8')).hexdigest(),
        'cache_signature': cache.metadata['signature'], 'patches': len(cache.patches),
        'cache_build_options': cache.metadata.get('build_options'),
        'cache_build_or_reuse_ms': build_ms, 'sample_spacing_m': spacing,
        'driving_waypoints': len(waypoints), 'resolved': len(errors),
        'unresolved_count': len(failures), 'unresolved': failures,
        'maximum_ground_error_m': max(errors) if errors else None,
        'p95_ground_error_m': float(np.percentile(errors, 95)) if errors else None,
        'sloped_waypoints_pitch_over_1_degree': slopes, 'junction_waypoints': junctions,
        'actual_stacked_road_locations': crossings,
        'build_report': cache.metadata.get('build_report'), 'synthetic_actor_benchmarks': []}
    if performance:
        center = waypoints[len(waypoints)//2].transform.location
        nearest = sorted(waypoints, key=lambda point: point.transform.location.distance(center))[:100]
        actors = [pose_from_waypoint(point, index+1) for index, point in enumerate(nearest)]
        render = HeightAwareBEVGenerator(cache)
        result['performance_equivalence'] = verify_optimized_equivalence(render, actors[0], actors)
        for count in (6, 20, 100):
            selected = actors[:count]
            for _ in range(2):
                output = render.generate(selected[0], selected, frame_id=0, timestamp=0)
            duration = []
            for _ in range(benchmark_runs):
                start = time.perf_counter()
                output = render.generate(selected[0], selected, frame_id=0, timestamp=0)
                duration.append((time.perf_counter()-start)*1000)
            result['synthetic_actor_benchmarks'].append({
                'actor_provenance': 'synthetic 4.5 m x 2 m boxes on actual CARLA waypoints; not live actors',
                'input_actors': len(selected),
                'included_actors': sum(row['included'] for row in output['metadata']['actors']),
                'output_shape': list(output['arrays']['semantic'].shape),
                'runs': benchmark_runs, 'samples_ms': duration,
                'median_ms': statistics.median(duration), 'p95_ms': float(np.percentile(duration,95)),
                'excludes': ['CARLA simulation', 'sensor processing', 'disk writes', 'web transfer']})
    return result


def runtime_environment():
    cpu = None
    try:
        cpu = next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                   if line.startswith('model name'))
    except (OSError,StopIteration):
        pass
    root = Path(__file__).resolve().parents[1]
    sources = ('sdv/bev/geometry.py','sdv/bev/cache.py','sdv/bev/generator.py','sdv/collection/bev_adapter.py')
    return {'python':platform.python_version(),'numpy':np.__version__,'opencv':cv2.__version__,
        'machine':platform.machine(),'cpu':cpu,'logical_cpus':os.cpu_count(),
        'source_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources},
        'timing_limitations':'Shared execution environment; warm-cache generation only; not an end-to-end collection FPS guarantee'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xodr-root', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--town', action='append')
    parser.add_argument('--output-dir', default=str(Path(__file__).resolve().parents[1]/'docs'/'validation'))
    parser.add_argument('--sample-spacing', type=float, default=1)
    parser.add_argument('--benchmark-runs', type=int, default=10)
    args = parser.parse_args()
    if not .5 <= args.sample_spacing <= 5 or not 1 <= args.benchmark_runs <= 100:
        parser.error('sample spacing must be0.5..5m and benchmark runs1..100')
    output = Path(args.output_dir); output.mkdir(parents=True,exist_ok=True)
    towns = args.town or ['Town01', 'Town04', 'Town05']
    reports = []
    for index, town in enumerate(towns):
        if not town.replace('_','').isalnum():
            parser.error('Town must be a simple map name')
        report = verify_map(town, Path(args.xodr_root)/(town+'.xodr'), args.cache_root,
                            args.sample_spacing, args.benchmark_runs, performance=index==0)
        reports.append(report)
        result = {'scope':'CARLA0.9.15 offline Map on actual supplied OpenDRIVE; no simulator server',
                  'runtime_environment':runtime_environment(),
                  'maps': reports, 'live_sensor_synchronization_verified': False}
        (output/'bev-offline-carla-validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print('{}: {}/{} waypoint surfaces resolved, max ground error={}m, overpass pairs={}'.format(
            town, report['resolved'], report['driving_waypoints'], report['maximum_ground_error_m'],
            len(report['actual_stacked_road_locations'])),flush=True)


if __name__ == '__main__':
    main()
