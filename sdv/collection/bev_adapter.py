"""CARLA 0.9.15 observations for the height-aware BEV ground-truth generator.

This boundary is import-safe without CARLA. Pose/velocity and all eight box
vertices are taken from one WorldSnapshot. Traffic-light states are read inside
the collector's tick guard (ActorSnapshot does not expose signal state).
Native Map.get_waypoint affiliation must agree with the snapshot bounding-box
bottom-face estimate. PNG maps contain no metric ground-elevation channel;
the optional geometry-cache generator also verifies native raster elevation.
"""
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from sdv.collection.metadata import call, enum_name, number, rotation, vector


def _xyz(value):
    if value is None:
        return None
    values = [number(getattr(value, axis, None)) for axis in ('x', 'y', 'z')]
    return values if all(v is not None for v in values) else None


def _identity(wp):
    return {key: getattr(wp, key, None) for key in ('road_id', 'section_id', 'lane_id')}


def _lane_key(wp):
    return tuple(getattr(wp, key, None) for key in ('road_id', 'section_id', 'lane_id'))


def _waypoint_id(wp):
    return '{}:{}:{}:{:.4f}'.format(*_lane_key(wp), float(wp.s))


def _lane_edges(wp):
    """Banking/grade are retained through CARLA's actual right vector."""
    center = _xyz(wp.transform.location)
    right = _xyz(call(wp.transform, 'get_right_vector'))
    width = number(getattr(wp, 'lane_width', None))
    if center is None or right is None or width is None or width <= 0:
        raise ValueError('CARLA waypoint lacks a valid 3D transform or lane width')
    center, right = np.asarray(center), np.asarray(right)
    return (center - right * width / 2).tolist(), (center + right * width / 2).tolist()


def _last_same_lane_waypoint(first, distance, carla_map=None):
    """Find the native boundary without crossing Road/Section/Lane identity.

    The CARLA topology pair may point to a *successor* lane entry, not the
    current lane exit. Bisection calls the actual cached CARLA geometry API;
    it never interpolates or extends the road itself. At most 32 local calls.
    """
    low, high, endpoint = 0.0, float(distance), first
    for _ in range(32):
        if high-low <= 1e-5:
            break
        middle = (low+high)/2
        candidates = [wp for wp in first.next(middle) if _lane_key(wp) == _lane_key(first)]
        if candidates:
            endpoint = min(candidates, key=_waypoint_id)
            low = middle
        else:
            high = middle
    if first.lane_id > 0 and 0 <= float(endpoint.s) <= 1e-5 and carla_map is not None:
        # Bisection approaches s=0 from inside the road. Exact native road
        # starts otherwise fall a few micrometres outside this polygon.
        # Retrieve the endpoint from CARLA; never extend ground tolerances.
        exact = call(carla_map, 'get_waypoint_xodr', None, first.road_id, first.lane_id, 0.0)
        if exact is not None and _lane_key(exact) == _lane_key(first):
            current_xyz, exact_xyz = _xyz(endpoint.transform.location), _xyz(exact.transform.location)
            if current_xyz is not None and exact_xyz is not None and np.linalg.norm(np.asarray(current_xyz)-exact_xyz) <= 1e-5:
                endpoint = exact
    return endpoint


def _lateral_surface(first, side):
    """Native same-level visibility adjacency, not a lane-change permission.

    CARLA can place unrasterized shoulder/median lanes between the two
    carriageways. Traverse their actual lane links without labelling the
    intervening region as drivable. Different roads, sections or levels stop
    traversal immediately, including a level change in an intermediate lane.
    """
    current = first
    seen = {_waypoint_id(first)}
    for _ in range(128):
        adjacent = call(current, 'get_' + side + '_lane')
        if adjacent is None or _xyz(adjacent.transform.location) is None:
            return None
        identity = _waypoint_id(adjacent)
        if identity in seen or _lane_key(adjacent)[:2] != _lane_key(first)[:2]:
            return None
        seen.add(identity)
        if abs(adjacent.transform.location.z - first.transform.location.z) > 0.5:
            return None
        if enum_name(getattr(adjacent, 'lane_type', None)) in ('Driving', 'Sidewalk'):
            return adjacent
        current = adjacent
    return None


def surface_patches(carla_map, spacing_m=2.0, diagnostics=None):
    """Bounded, sampled lane ribbons; no unbounded road-following loop.

    Each graph node is a short physical road strip, not an entire road or a
    global Z bin. Traversal across a ramp therefore follows local adjacency.
    Lateral links require CARLA lane adjacency and compatible surface heights.
    Lane markings represent OpenDRIVE geometry, not texture/decal observations.
    """
    spacing_m = float(spacing_m)
    if not math.isfinite(spacing_m) or spacing_m <= 0:
        raise ValueError('waypoint spacing must be positive and finite')
    driving = list(carla_map.generate_waypoints(spacing_m))
    # Fixed-s samples omit the final fractional interval at a road's end.
    # Positive lanes drive toward decreasing s, so next(spacing) cannot cover
    # that entry tail. Topology supplies the actual endpoints, without guessed
    # extrapolation or enlarged ground-matching tolerance.
    for entry, successor_or_exit in call(carla_map, 'get_topology', []) or []:
        driving.extend((entry, successor_or_exit))
    # CARLA 0.9.15 Map.cpp GenerateWaypoints calls ForEachDrivableLaneAt.
    # Walk actual left/right lane links to include adjacent sidewalks; don't
    # assume the generated list contains them or synthesize sidewalk widths.
    discovered = {}
    for seed in driving:
        pending, seen = [seed], set()
        while pending:
            point = pending.pop()
            identity = _waypoint_id(point)
            if identity in seen:
                continue
            seen.add(identity)
            if len(seen) > 128:
                raise ValueError('CARLA lateral lane traversal exceeded 128 distinct lanes at one road section')
            discovered[identity] = point
            for side in ('left', 'right'):
                neighbor = call(point, 'get_' + side + '_lane')
                if neighbor is not None:
                    pending.append(neighbor)
    waypoints = list(discovered.values())
    by_lane = defaultdict(list)
    for wp in waypoints:
        kind = enum_name(getattr(wp, 'lane_type', None))
        if kind in ('Driving', 'Sidewalk'):
            by_lane[_lane_key(wp)].append(wp)
    for samples in by_lane.values():
        samples.sort(key=lambda wp: float(wp.s))
    patches, source_waypoints, starts, ends = [], {}, defaultdict(list), defaultdict(list)
    successor_points, boundary_points = defaultdict(list), {}
    for samples in by_lane.values():
        for first in samples:
            for last in list(first.next(spacing_m)):
                # Native next() can cross a road/section boundary. Cut its
                # polygon below, and retain the successor only as a graph edge.
                if enum_name(getattr(last, 'lane_type', None)) not in ('Driving', 'Sidewalk'):
                    continue
                original_next = last
                start, finish = _xyz(first.transform.location), _xyz(last.transform.location)
                if start is None or finish is None:
                    continue
                segment_distance = float(np.linalg.norm(np.asarray(start) - finish))
                discontinuous = segment_distance > spacing_m * 2.0
                if discontinuous:
                    # Actual Town01 sidewalk links can jump >7 m for next(2).
                    # Keep the gap unknown rather than inventing a connecting
                    # polygon or discarding all valid geometry in the Town.
                    if diagnostics is not None:
                        diagnostics.append({'reason': 'discontinuous_successor_geometry',
                            'source_waypoint': _waypoint_id(first), 'target_waypoint': _waypoint_id(last),
                            'source_xyz': start, 'target_xyz': finish, 'distance_m': segment_distance,
                            'requested_distance_m': spacing_m})
                crossing = _lane_key(first) != _lane_key(last)
                if crossing:
                    identity = _waypoint_id(first)
                    if identity not in boundary_points:
                        boundary_points[identity] = _last_same_lane_waypoint(first, spacing_m, carla_map)
                    last = boundary_points[identity]
                    finish = _xyz(last.transform.location)
                elif discontinuous:
                    continue
                if np.linalg.norm(np.asarray(start) - finish) < 1e-6:
                    continue
                left_a, right_a = _lane_edges(first)
                left_b, right_b = _lane_edges(last)
                patch_id = _waypoint_id(first) + '>' + _waypoint_id(last)
                if crossing and not discontinuous:
                    successor_points[patch_id].append(original_next)
                if patch_id in source_waypoints:
                    # A junction may have several successors for the same
                    # terminal strip. Preserve all graph links, one polygon.
                    continue
                marks, boundaries = [], []
                for side, points in (('left', [left_a, left_b]), ('right', [right_a, right_b])):
                    marking = getattr(first, side + '_lane_marking', None)
                    marking_type = enum_name(getattr(marking, 'type', None))
                    if marking_type not in ('None', 'NONE'):
                        marking_width = number(getattr(marking, 'width', None))
                        known_width = marking_width is not None and marking_width > 0
                        marks.append({'points_xyz': points,
                            'width_m': marking_width if known_width else 0.2,
                            # Unknown width defines an exclusion band, never
                            # guessed solid paint. Cache lane_validity records it.
                            'type': (marking_type or 'Unknown') if known_width else 'UnknownWidth',
                            'source': 'CARLA waypoint OpenDRIVE lane marking'})
                    adjacent = call(first, 'get_' + side + '_lane')
                    if adjacent is None or enum_name(getattr(adjacent, 'lane_type', None)) != 'Driving':
                        boundaries.append({'points_xyz': points, 'width_m': 0.2})
                patches.append(dict(_identity(first), id=patch_id,
                    polygon_xyz=[left_a, right_a, right_b, left_b],
                    center_xyz=((np.asarray(start) + finish) / 2).tolist(),
                    lane_markings=marks, boundary_lines=boundaries,
                    kind='driving' if enum_name(first.lane_type) == 'Driving' else 'sidewalk',
                    is_junction=bool(first.is_junction), neighbors=[],
                    source='CARLA sampled OpenDRIVE lane geometry',
                    start_s_m=float(first.s), end_s_m=float(last.s)))
                source_waypoints[patch_id] = (first, last)
                starts[_lane_key(first)].append((first, patch_id))
                ends[_lane_key(last)].append((last, patch_id))
    indexed = {patch['id']: patch for patch in patches}

    def nearest_connections(point, index, tolerance):
        candidates = index.get(_lane_key(point), [])
        if not candidates:
            return []
        # Same OpenDRIVE lane identity prevents XY-crossing overpass links.
        distances = [(abs(float(sample.s) - float(point.s)), sample, pid) for sample, pid in candidates]
        best = min(distance[0] for distance in distances)
        if best > tolerance:
            return []
        pos = np.asarray(_xyz(point.transform.location))
        return [pid for ds, sample, pid in distances if ds <= best + 0.01 and
                np.linalg.norm(np.asarray(_xyz(sample.transform.location)) - pos) <= tolerance * 2 + 0.1]

    for patch in patches:
        first, last = source_waypoints[patch['id']]
        neighbors = set(nearest_connections(last, starts, spacing_m * 0.55))
        neighbors.update(nearest_connections(first, ends, spacing_m * 0.55))
        for successor in successor_points[patch['id']]:
            neighbors.update(nearest_connections(successor, starts, spacing_m * 0.55))
        for side in ('left', 'right'):
            adjacent = _lateral_surface(first, side)
            if adjacent is not None:
                neighbors.update(nearest_connections(adjacent, starts, spacing_m * 0.75))
        neighbors.discard(patch['id'])
        patch['neighbors'] = sorted(neighbors)
    for patch in patches:
        for neighbor in patch['neighbors']:
            if patch['id'] not in indexed[neighbor]['neighbors']:
                indexed[neighbor]['neighbors'].append(patch['id'])
    if not patches:
        raise ValueError('No CARLA Driving/Sidewalk geometry was available for BEV rasterization')
    return patches


def build_cache(world, cache_dir, settings):
    from sdv.bev.cache import GlobalRasterCache
    carla_map = world.get_map()
    town, xodr = carla_map.name.split('/')[-1], carla_map.to_opendrive()
    resolution = float(settings.get('meters_per_pixel', 0.2))
    options = {'builder': 'carla_waypoint_ribbons_v8', 'spacing_m': 2.0,
               'endpoints': 'actual topology entry and exit waypoints',
               'identity_boundary': 'native next bisection; Road/Section/Lane retained; 1e-5 m interval',
               'exact_road_start': 'native get_waypoint_xodr s=0; same full key; within 1e-5 m',
               'lateral_visibility': 'native same-road/section links through unrasterized lanes; every height difference <= 0.5 m',
               'lanes': 'driving plus actual lateral sidewalk links',
               'marking_semantics': 'known-width solid paint only; broken/compound/unknown marking support invalid'}
    found = GlobalRasterCache.find(cache_dir, town, xodr, resolution, build_options=options)
    if found is not None:
        return found
    diagnostics = []
    patches = surface_patches(carla_map, diagnostics=diagnostics)
    return GlobalRasterCache.build(cache_dir, town, xodr, patches, resolution,
        build_options=options, build_report={'skipped_edges': diagnostics,
            'skipped_edge_count': len(diagnostics), 'surface_count': len(patches),
            'geometry_limitations': 'OpenDRIVE strips sampled every 2 m; discontinuous successor geometry excluded, solid painted markings only'})


def _ground_candidate(carla_map, location, contact_world_xyz=None):
    """Reject a native XY waypoint on a different deck, without using the roof.

    The 0.75 m guard accommodates bbox/pavement clearance but rejects a nearby
    bridge selected by the API. A bottom-face center is preferred to actor
    origin and the lowest corner (which is displaced on a banked road).
    This remains an estimate, not a measured wheel-to-surface contact.
    """
    reference = contact_world_xyz if contact_world_xyz is not None else _xyz(location)
    tolerance = .75
    provenance = {
        'ground_z_m': None, 'ground_verified': False, 'ground_candidate_z_m': None,
        'road_id': None, 'section_id': None, 'lane_id': None,
        'ground_lane_type': None, 'ground_road_s_m': None,
        'ground_reference_xyz': reference, 'ground_height_tolerance_m': tolerance,
        'ground_reference_source': 'bbox_bottom_face_center_estimate' if contact_world_xyz is not None else 'snapshot_actor_origin',
        'ground_source': 'unresolved; no exact CARLA waypoint',
    }
    if reference is None:
        return provenance
    try:
        import carla
        at = carla.Location(x=reference[0], y=reference[1], z=reference[2])
        waypoint = carla_map.get_waypoint(at, project_to_road=False, lane_type=carla.LaneType.Any)
    except ImportError:
        # CARLA-shaped fixtures expose the same keyword boundary without enum.
        from types import SimpleNamespace
        at = SimpleNamespace(x=reference[0], y=reference[1], z=reference[2])
        waypoint = carla_map.get_waypoint(at, project_to_road=False)
    except (RuntimeError, AttributeError, TypeError):
        waypoint = None
    if waypoint is None:
        return provenance
    ground = number(waypoint.transform.location.z)
    error = None if ground is None else abs(ground-reference[2])
    verified = error is not None and error <= tolerance
    provenance.update(_identity(waypoint))
    provenance.update(ground_z_m=ground if verified else None,
        ground_candidate_z_m=ground, ground_height_error_m=error,
        ground_verified=verified, ground_lane_type=enum_name(getattr(waypoint, 'lane_type', None)),
        ground_road_s_m=number(getattr(waypoint, 's', None)),
        ground_source='carla_waypoint_exact' if verified else 'rejected; native waypoint height disagrees with snapshot ground estimate')
    return provenance


def actor_record(actor, snapshot, carla_map, is_ego=False):
    state = call(snapshot, 'find', None, actor.id)
    if state is None:
        return None
    transform = call(state, 'get_transform')
    position = _xyz(getattr(transform, 'location', None))
    if position is None:
        return None
    bbox = getattr(actor, 'bounding_box', None)
    vertices = call(bbox, 'get_world_vertices', None, transform)
    vertices = None if vertices is None else [_xyz(point) for point in vertices]
    if vertices is None or len(vertices) != 8 or any(point is None for point in vertices):
        return None
    points = np.asarray(vertices, dtype=np.float64)
    # Convex hull of all eight projected vertices includes pitched/rolled boxes.
    import cv2
    footprint_xy = cv2.convexHull(points[:, :2].astype(np.float32)).reshape(-1, 2)
    object_class = 'pedestrian' if str(actor.type_id).startswith('walker.pedestrian.') else 'vehicle'
    # Two wheels also includes motorcycles. CARLA 0.9.15 semantic Bicycle=19
    # is the supported distinction; wheel count alone is not a cyclist label.
    if object_class == 'vehicle' and 19 in (getattr(actor, 'semantic_tags', []) or []):
        object_class = 'cyclist'
    # CARLA 0.9.15 BoundingBox.h GetLocalVertices orders local -extent.z at
    # indices 0,2,4,6. GetWorldVertices preserves that order after BOTH bbox
    # rotation and actor transform. The lowest world corner is not contact
    # height at the center on a slope, so keep these quantities independent.
    contact = points[[0, 2, 4, 6]].mean(axis=0).tolist()
    ground = _ground_candidate(carla_map, transform.location, contact)
    return dict(ground, id=int(actor.id), actor_id=int(actor.id), type_id=str(actor.type_id),
        object_class=object_class, is_ego=bool(is_ego), x=position[0], y=position[1], z=position[2],
        yaw=float(transform.rotation.yaw), position=position,
        rotation_deg=rotation(transform.rotation), velocity_m_s=vector(call(state, 'get_velocity')),
        bbox_extent_m=vector(getattr(bbox, 'extent', None)),
        bbox_center_local_m=vector(getattr(bbox, 'location', None)),
        bbox_rotation_deg=rotation(getattr(bbox, 'rotation', None)), bbox_vertices_world_xyz=vertices,
        footprint_world_xyz=[[float(x), float(y), float(points[:, 2].min())] for x, y in footprint_xy],
        bbox_min_z_m=float(points[:, 2].min()), bbox_max_z_m=float(points[:, 2].max()),
        object_height_m=float(points[:, 2].max() - points[:, 2].min()),
        ground_contact_world_xyz=contact, ground_contact_source='bbox_bottom_face_center_estimate',
        frame_id=int(snapshot.frame), ground_contact_measured=False,
        footprint_source='orthographic convex hull of snapshot-transformed 3D bounding box; not mesh contact')


class CarlaBEVAdapter:
    def __init__(self, world, cache_dir, settings):
        self.world, self.settings = world, dict(settings)
        self.carla_map = world.get_map()
        if self.settings['mode'] == 'global_raster':
            from sdv.bev.global_map import load_global_map
            from sdv.bev.global_generator import GlobalMapBEVGenerator
            town = self.carla_map.name.split('/')[-1]
            self.global_map = load_global_map(resolved_map_root(settings), town, settings['global_ppm'])
            self.cache = None
            self.generator = GlobalMapBEVGenerator(self.global_map, settings)
            self.map_metadata = dict(self.global_map.metadata,
                files=dict(self.global_map.record['files']), online_geometry_rebuilt=False)
        else:
            from sdv.bev.generator import HeightAwareBEVGenerator
            self.cache = build_cache(world, cache_dir, settings)
            self.generator = HeightAwareBEVGenerator(self.cache, settings)
            self.map_metadata = {'source': 'OpenDRIVE sampled surface cache',
                'signature': self.cache.metadata['signature'], 'town': self.cache.metadata.get('town'),
                'meters_per_pixel': self.cache.meters_per_pixel}
        self.stop_geometry = {}

    def traffic(self, actors):
        output = []
        for actor in actors:
            if not str(getattr(actor, 'type_id', '')).startswith('traffic.traffic_light'):
                continue
            if actor.id not in self.stop_geometry:
                records = []
                for waypoint in call(actor, 'get_stop_waypoints', []) or []:
                    left, right = _lane_edges(waypoint)
                    records.append(dict(_identity(waypoint), segments=[[left, right]],
                        ground_verified=True, ground_source='carla_native_stop_waypoint',
                        ground_lane_type=enum_name(getattr(waypoint, 'lane_type', None)),
                        ground_road_s_m=number(getattr(waypoint, 's', None)),
                        source='CARLA TrafficLight.get_stop_waypoints + lane width; API stop position, not painted decal'))
                self.stop_geometry[actor.id] = records
            state = enum_name(call(actor, 'get_state')) or 'Unknown'
            for geometry in self.stop_geometry[actor.id]:
                output.append(dict(geometry, id=int(actor.id), state=state))
        return output

    def generate(self, ego, snapshot, agent):
        actors = list(self.world.get_actors())
        records, skipped = [], []
        for actor in actors:
            if not str(getattr(actor, 'type_id', '')).startswith(('vehicle.', 'walker.pedestrian.')):
                continue
            record = actor_record(actor, snapshot, self.carla_map, actor.id == ego.id)
            if record is None:
                skipped.append({'actor_id': int(actor.id), 'reason': 'snapshot pose or 3D bounding box unavailable'})
            else:
                records.append(record)
        ego_record = next((record for record in records if record['is_ego']), None)
        if ego_record is None:
            raise RuntimeError('BEV ego snapshot or bounding box is unavailable; refusing unaligned ground truth')
        self.last_ego_record = ego_record
        planner = getattr(agent, '_local_planner', None)
        route_waypoints = [waypoint for waypoint, _ in list(getattr(planner, '_waypoints_queue', []) or [])]
        route = [_xyz(waypoint.transform.location) for waypoint in route_waypoints]
        route = [point for point in route if point is not None]
        options = {}
        if self.settings['mode'] == 'global_raster':
            options['route_ground_verified'] = bool(route_waypoints) and all(
                enum_name(getattr(wp, 'lane_type', None)) == 'Driving' for wp in route_waypoints)
        result = self.generator.generate(ego_record, records, self.traffic(actors), route,
            frame_id=int(snapshot.frame), timestamp=float(snapshot.timestamp.elapsed_seconds), **options)
        result['metadata']['skipped_snapshot_actors'] = skipped
        result['metadata']['route_source'] = 'BehaviorAgent current local waypoint queue; online plan, not future observed ground truth'
        result['metadata']['traffic_source'] = 'CARLA signal state in collector tick guard; static stop geometry cached by actor ID'
        return result


def save_ground_truth(scene, stem, result):
    """Write the numeric GT atomically before info/{frame}.json is committed."""
    destination = Path(scene) / 'bev_gt' / (str(stem) + '.npz')
    from sdv.bev.io import save_frame
    save_frame(destination, result['arrays'], metadata=result['metadata'])
    return dict(result['metadata'], path='bev_gt/{}.npz'.format(stem),
                role='simulator_ground_truth', online_input=False)


def configured_settings(config=None):
    from sdv.bev.schema import normalize_bev_settings
    value = os.environ.get('SDV_BEV_GENERATOR')
    return normalize_bev_settings(json.loads(value) if value else (config or {}).get('bev_generator'))


def resolved_map_root(settings):
    """Resolve the immutable job path independently of the caller's CWD."""
    from sdv.paths import ROOT
    selected = os.environ.get('SDV_BEV_MAP_ROOT') or settings.get('map_root', 'assets/bev_maps')
    path = Path(selected).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def validate_global_maps(towns, settings):
    """Validate source files before CARLA is launched or its world is modified."""
    if settings['mode'] != 'global_raster':
        return
    from sdv.bev.global_map import load_global_map
    root = resolved_map_root(settings)
    for town in towns:
        try:
            load_global_map(root, town, settings['global_ppm'])
        except (OSError, ValueError, KeyError) as error:
            raise ValueError('Global BEV PNG 준비 실패: {} / {} PPM / {}. 제공된 지도 폴더와 catalog.json을 확인하세요. {}'.format(
                town, settings['global_ppm'], root, error)) from error


def annotate_scene_map(scene, adapter):
    """Persist the actual map source once per scene, outside per-frame images."""
    if adapter is None:
        return
    from sdv.collection.metadata import atomic_json
    path = Path(scene) / 'collection_manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    manifest['bev_map'] = adapter.map_metadata
    atomic_json(path, manifest)


def compatibility_images(result):
    """Legacy binary PNG names are derived from authoritative numeric channels."""
    arrays = result['arrays']
    images = {name: (arrays[channel] > 0).astype(np.uint8) * 255 for name, channel in (
        ('das', 'drivable'), ('lane', 'lane'), ('vehicle_mask', 'vehicle'),
        ('walker_mask', 'pedestrian'), ('route', 'route'))}
    stops = arrays['stop_red'] | arrays['stop_yellow'] | arrays['stop_green']
    if 'static_stopline' in arrays:
        stops = stops | arrays['static_stopline']
    images['stopline'] = (stops > 0).astype(np.uint8) * 255
    light = np.zeros(arrays['semantic'].shape + (3,), dtype=np.uint8)
    for channel, color in (('stop_green', (0, 255, 0)), ('stop_yellow', (255, 255, 0)), ('stop_red', (255, 0, 0))):
        light[arrays[channel] > 0] = color
    images['light_topview'] = light
    images['bev'] = result['preview_rgb']
    return images


def legacy_route_pixels(points_xy, ego_record, settings):
    """Preserve old info pixel fields using the selected, declared BEV geometry."""
    from sdv.bev.geometry import BEVGeometry, world_to_local
    geometry = BEVGeometry(**{key: settings[key] for key in ('width', 'height', 'meters_per_pixel', 'anchor_x', 'anchor_y')})
    points = np.asarray(points_xy, dtype=np.float64)
    if not points.size:
        return []
    pose = dict(ego_record)
    if settings.get('mode') == 'global_raster' and not settings.get('rotate_with_ego', True):
        pose['yaw'] = 0.0
    return world_to_local(points.reshape(-1, 2), pose, geometry).reshape(-1, 1, 2).tolist()
