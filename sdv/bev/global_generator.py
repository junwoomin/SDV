"""Local labels cropped directly from supplied full-town PNG rasters.

No OpenDRIVE traversal, waypoint ribbons, synthetic elevation, or static
rerasterization occurs here. HBLR's town-specific high/low selector remains a
documented asset policy, not a claim of continuous road-surface reconstruction.
"""
import copy
import math
import time
from pathlib import Path

import numpy as np

from .geometry import BEVGeometry, world_to_local, local_to_world, warp_global
from .generator import HeightAwareBEVGenerator
from .schema import (GLOBAL_RASTER_SCHEMA_VERSION, BINARY_CHANNELS,
                     HEIGHT_CHANNELS, STATIC_CHANNELS, SEMANTIC_ORDER,
                     channel_schema, normalize_bev_settings)
from .io import colorize, semantic_from_channels


def _finite(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float, np.number))
            and math.isfinite(float(value)))


class GlobalMapBEVGenerator:
    # Reuse the already tested 3D-box footprint projection and subpixel polygon
    # fill. These methods need only geometry; no fake patch-cache object is used.
    _polygon_indices = HeightAwareBEVGenerator._polygon_indices

    def __init__(self, global_map, settings=None):
        self.global_map = global_map
        self.settings = normalize_bev_settings(settings)
        if self.settings['mode'] != 'global_raster':
            raise ValueError('GlobalMapBEVGenerator requires global_raster mode')
        self.geometry = BEVGeometry(**{key:self.settings[key] for key in
            ('width', 'height', 'meters_per_pixel', 'anchor_x', 'anchor_y')})
        self._coverage_images = {}

    def _selection(self, pose, channels):
        return self.global_map.select_layer(pose, policy='hblr',
            height_aware=self.settings['height_aware'], channels=channels)

    def _coverage(self, shape, pose, offset, mpp):
        # Allocate once for a town canvas; never allocate a full global image
        # for each frame. Warping ones exactly matches OpenCV's border/nearest
        # rounding, including the last pixel at fractional pose coordinates.
        shape = tuple(shape)
        if shape not in self._coverage_images:
            self._coverage_images[shape] = np.ones(shape, dtype=np.uint8)
        return warp_global(self._coverage_images[shape], pose, self.geometry, offset, mpp)

    @staticmethod
    def _selection_metadata(selection):
        meta = selection.get('metadata')
        return meta if isinstance(meta, dict) else {}

    def _line(self, points, pose, coverage, selected_layer, threshold, width_m=.3,
              ground_verified=False, lane_type=None):
        """Snapshot/planner XYZ polylines; static road pixels remain untouched."""
        import cv2
        output = np.zeros(coverage.shape, dtype=np.uint8)
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            return output
        for start, finish in zip(points[:-1], points[1:]):
            uv = world_to_local(np.array([start[:2], finish[:2]]), pose, self.geometry)
            segment = np.zeros_like(output)
            # Bound integer conversion even for malformed, distant planner data.
            uv = np.clip(np.rint(uv), -1000000, 1000000).astype(np.int32)
            cv2.line(segment, tuple(uv[0]), tuple(uv[1]), 1,
                max(1, int(round(width_m/self.geometry.meters_per_pixel))), cv2.LINE_8)
            rows, cols = np.nonzero(segment)
            keep = coverage[rows, cols].astype(bool)
            if self.settings['height_aware'] and threshold is not None:
                xy = local_to_world(np.column_stack((cols, rows)), pose, self.geometry)
                delta = finish[:2]-start[:2]
                norm = float(delta.dot(delta))
                fraction = np.zeros(len(rows)) if norm < 1e-12 else np.clip((xy-start[:2]).dot(delta)/norm, 0, 1)
                height = start[2] + fraction*(finish[2]-start[2])
                classified = self.global_map.layer_for_points(np.column_stack((xy,height)),
                    ground_verified=ground_verified, lane_type=lane_type)
                keep &= classified['high'] if selected_layer == 'high' else ~classified['high']
            output[rows[keep], cols[keep]] = 1
        return output

    def generate(self, ego_pose, actors, traffic=None, route=None, frame_id=None, timestamp=None,
                 route_ground_verified=False):
        start = time.perf_counter()
        requested = set(self.settings['semantic_layers']) | {'drivable', 'stopline'}
        selection = self._selection(ego_pose, sorted(requested))
        source = selection['arrays']
        source_meta = self._selection_metadata(selection)
        layer = selection['layer_id']
        offset = selection.get('world_offset', self.global_map.world_offset)
        mpp = selection.get('meters_per_pixel', self.global_map.meters_per_pixel)
        shape = tuple(source_meta.get('shape', getattr(self.global_map, 'shape', ())))
        if len(shape) != 2:
            raise ValueError('Global raster must declare its native height and width')
        lookup_end = time.perf_counter()

        pose = dict(ego_pose)
        pose['yaw'] = float(ego_pose['yaw']) if self.settings['rotate_with_ego'] else 0.
        # Freeze and validate the actual affine used by every static/dynamic
        # channel. Training can invert the displayed world orientation safely.
        origin, axis_x, axis_y = world_to_local(np.array([[0.,0.], [1.,0.], [0.,1.]]), pose, self.geometry)
        affine = np.column_stack((axis_x-origin, axis_y-origin, origin))
        geometry = self.geometry.metadata()
        geometry.update(world_to_bev_pixel=affine.tolist(),
            rotation_yaw_degrees=pose['yaw'], rotate_with_ego=self.settings['rotate_with_ego'],
            orientation='ego_forward_up_ego_right_right' if self.settings['rotate_with_ego'] else 'world_x_up_world_y_right')
        coord_end = time.perf_counter()

        local_shape = (self.geometry.height, self.geometry.width)
        arrays = {name:np.zeros(local_shape, dtype=np.uint8) for name in BINARY_CHANNELS}
        arrays.update({name:np.full(local_shape, np.nan, dtype=np.float32) for name in HEIGHT_CHANNELS})
        arrays['elevation_validity'] = np.zeros(local_shape, dtype=np.uint8)
        arrays['actor_id'] = np.zeros(local_shape, dtype=np.int32)
        arrays['surface_id'] = np.full(local_shape, -1, dtype=np.int32)
        coverage = self._coverage(shape, pose, offset, mpp)
        arrays['validity'] = coverage
        arrays['occupancy_validity'] = coverage.copy()
        available, missing = [], []
        for channel in STATIC_CHANNELS:
            original = 'drivable' if channel == 'road' else channel
            if original not in source:
                if channel in self.settings['semantic_layers']: missing.append(channel)
                continue
            if original not in available: available.append(original)
            if channel != 'road' and channel not in self.settings['semantic_layers']:
                continue
            sampled = warp_global(source[original], pose, self.geometry, offset, mpp)
            arrays[channel] = ((sampled > 0) & coverage.astype(bool)).astype(np.uint8)
        if 'lane' in available and 'lane' in self.settings['semantic_layers']:
            arrays['lane_validity'] = coverage.copy()
        # A source stop-line mask has no signal state. Preserve it independently;
        # only the live state below may populate red/yellow/green channels.
        if 'stopline' in source:
            arrays['static_stopline'] = ((warp_global(source['stopline'], pose, self.geometry, offset, mpp)>0)
                & coverage.astype(bool)).astype(np.uint8)
        index = 1 if layer == 'high' else 0
        arrays['surface_id'][coverage > 0] = index
        crop_end = time.perf_counter()

        warnings = list(selection.get('warnings', []))
        if self.geometry.meters_per_pixel > float(mpp)+1e-12:
            warnings.append('Output resolution is coarser than the source PNG: nearest-neighbor downsampling can remove thin lane/stopline pixels; labels are not dilated or smoothed.')
        if missing: warnings.append('Requested static channels are absent in the selected PNG layer: '+', '.join(sorted(missing)))
        warnings.append('Global PNG assets have no metric road elevation; elevation_m is NaN and elevation_validity is zero.')
        threshold = source_meta.get('height_threshold_m')
        threshold = float(threshold) if _finite(threshold) else None
        separated = bool(source_meta.get('height_separation_available', False))
        if not separated: threshold = None
        if self.settings['height_aware'] and not separated:
            warnings.append('This map has no supported high/low asset separation; dynamic actors use the recorded single-plane view.')
        if not self.settings['height_aware']:
            warnings.append('Height filtering is disabled; actors from different road levels may share this 2D view.')
        records, object_top = [], np.full(local_shape, np.nan, dtype=np.float32)
        ego_id = ego_pose.get('actor_id', ego_pose.get('id'))
        source_actors = list(actors)
        if ego_pose.get('footprint_world_xyz') is not None and not any(a.get('actor_id', a.get('id')) == ego_id for a in source_actors):
            source_actors.append(dict(ego_pose, object_class='vehicle'))
        for actor in source_actors:
            kind = actor.get('object_class', actor.get('class'))
            identity = actor.get('actor_id', actor.get('id'))
            is_ego = ego_id is not None and identity == ego_id
            if kind not in self.settings['actor_classes'] and not is_ego: continue
            rows, cols = self._polygon_indices(actor.get('footprint_world_xyz', []), pose)
            if not len(rows): continue
            record = copy.deepcopy(actor)
            ground = actor.get('ground_z_m')
            if not _finite(ground): ground = None
            record.update(ground_z_m=ground, map_layer_id=None, is_ego=is_ego,
                height_assignment='native_map_ground_height_then_HBLR_asset_threshold' if separated else 'single_plane_asset_no_vertical_separation')
            if self.settings['height_aware'] and separated and ground is None:
                arrays['occupancy_validity'][rows, cols] = 0
                record.update(included=False, reason='ground_height_unavailable_for_layer_selection')
                records.append(record); continue
            actor_layer = layer
            if self.settings['height_aware'] and separated:
                # The selector consumes ground_z_m, never a tall object's top.
                actor_selection = self._selection(dict(actor, ground_z_m=ground), [])
                actor_layer = actor_selection['layer_id']
                actor_selection_meta = self._selection_metadata(actor_selection)
                record['layer_selection_method'] = actor_selection_meta.get('layer_selection_method')
                record['threshold_overridden_by_support'] = actor_selection_meta.get('threshold_overridden_by_support',False)
                if actor_layer != layer:
                    record.update(map_layer_id=actor_layer, included=False, reason='different_height_layer')
                    records.append(record); continue
            record['map_layer_id'] = actor_layer
            on_canvas = coverage[rows, cols] > 0
            rr, cc = rows[on_canvas], cols[on_canvas]
            values = [actor.get('bbox_min_z_m'), actor.get('bbox_max_z_m'), actor.get('object_height_m')]
            if (not all(_finite(v) for v in values) or values[1] < values[0] or values[2] < 0
                    or type(identity) is not int or not 0 < identity < 2147483647):
                arrays['occupancy_validity'][rows, cols] = 0
                record.update(included=False, reason='invalid_bbox_or_actor_id'); records.append(record); continue
            if not len(rr):
                record.update(included=False, reason='outside_global_canvas'); records.append(record); continue
            channel = 'ego' if is_ego else kind
            if channel not in arrays: continue
            arrays[channel][rr, cc] = 1
            above = ~np.isfinite(object_top[rr, cc]) | (float(values[1]) > object_top[rr, cc])
            ar, ac = rr[above], cc[above]
            object_top[ar, ac] = values[1]
            arrays['actor_id'][ar, ac] = identity
            if self.settings['object_height']:
                if ground is not None: arrays['object_ground_z_m'][ar, ac] = ground
                for name, value in zip(('object_min_z_m', 'object_max_z_m', 'object_height_m'), values):
                    arrays[name][ar, ac] = value
            record.update(included=True, occupancy_pixels=int(len(rr)))
            records.append(record)
        signals = []
        for signal in traffic or []:
            state = str(signal.get('state', 'Unknown')).split('.')[-1].lower()
            pixels = 0
            if state in ('red', 'yellow', 'green'):
                for segment in signal.get('segments', []):
                    mask = self._line(segment, pose, coverage, layer, threshold,
                        ground_verified=signal.get('ground_verified') is True,
                        lane_type=signal.get('ground_lane_type',signal.get('lane_type')))
                    arrays['stop_'+state] |= mask
                    pixels += int(mask.sum())
            signals.append(dict(signal, rasterized_pixels=pixels))
        if route is not None and len(route):
            arrays['route'] = self._line(route, pose, coverage, layer, threshold, width_m=.5,
                ground_verified=route_ground_verified is True,
                lane_type='Driving' if route_ground_verified is True else None)
        actors_end = time.perf_counter()

        arrays['semantic'] = semantic_from_channels(arrays)
        visible = dict(arrays)
        enabled = set(self.settings['visual_overlays'])
        for name, _class_id in SEMANTIC_ORDER:
            if name not in enabled and not (name.startswith('stop_') and 'stoplines' in enabled):
                visible[name] = np.zeros(local_shape, dtype=np.uint8)
        preview = colorize(semantic_from_channels(visible))
        root_meta = getattr(self.global_map, 'metadata', {}) or {}
        global_meta = dict(root_meta, **source_meta)
        map_root = getattr(self.global_map, 'root', global_meta.get('root'))
        source_directory = global_meta.get('source_directory', global_meta.get('source_path'))
        source_path = str(Path(map_root)/source_directory) if map_root is not None and source_directory else source_directory
        file_records = getattr(self.global_map, 'record', {}).get('files', {})
        selected_files = {name:{'path':str(Path(source_path)/filename) if source_path else filename,
            'sha256':file_records.get(filename, {}).get('sha256')}
            for name,filename in global_meta.get('source_files', {}).items()}
        global_info = {
            'town':getattr(self.global_map, 'town', global_meta.get('town')),
            'root':str(map_root) if map_root is not None else None, 'source_path':source_path,
            'native_ppm':getattr(self.global_map, 'ppm', 1./float(mpp)),
            'meters_per_pixel':float(mpp), 'height_layers':global_meta.get('height_layers'),
            'selected_height_layer':layer,
            'source_signature':global_meta.get('source_signature', global_meta.get('signature')),
            'files':selected_files,
            'layer_selection_method':source_meta.get('layer_selection_method'),
            'threshold_overridden_by_support':source_meta.get('threshold_overridden_by_support',False),
        }
        fields = channel_schema()
        fields['elevation_validity'] = {'dtype':'uint8', 'range':[0,1], 'meaning':'finite metric road elevation coverage'}
        if 'static_stopline' in arrays:
            fields['static_stopline'] = {'dtype':'uint8', 'range':[0,1], 'meaning':'provided static stopline PNG; signal state unknown'}
        finished = time.perf_counter()
        timings = {'map_lookup_ms':(lookup_end-start)*1000., 'coordinates_ms':(coord_end-lookup_end)*1000.,
            'crop_rotation_ms':(crop_end-coord_end)*1000., 'actors_ms':(actors_end-crop_end)*1000.,
            'compose_ms':(finished-actors_end)*1000., 'total_ms':(finished-start)*1000.}
        metadata = {'schema_version':GLOBAL_RASTER_SCHEMA_VERSION, 'label_source':'carla_simulator_ground_truth',
            'static_label_source':'provided_global_raster_png', 'online_input':False,
            'geometry':geometry, 'settings':copy.deepcopy(self.settings),
            'frame_id':frame_id, 'timestamp':timestamp,
            'ego_pose':{key:ego_pose[key] for key in ('x','y','z','yaw')},
            'ground_z_m':float(ego_pose['ground_z_m']) if _finite(ego_pose.get('ground_z_m')) else None,
            'ground_source':ego_pose.get('ground_source'), 'map_layer_id':layer,
            'road_id':ego_pose.get('road_id'), 'lane_id':ego_pose.get('lane_id'),
            'surface_table':{layer:{'index':index, 'meaning':'global PNG asset layer; not a measured road surface'}},
            'global_map':global_info, 'available_static_channels':sorted(available),
            'unavailable_static_channels':sorted(missing), 'elevation_available':False,
            'height_separation_available':separated, 'height_threshold_m':threshold,
            'layer_policy':'verified native Driving ground + exclusive supplied DAS support; HBLR ground-height threshold at overlaps',
            'layer_selection_method':source_meta.get('layer_selection_method'),
            'route_ground_verified':route_ground_verified is True,
            'actors':records, 'traffic':signals, 'channel_schema':fields,
            'valid_surface_fraction':float(coverage.mean()),
            'validity_meaning':'inside the supplied global PNG canvas; not measured elevation validity',
            'warnings':warnings, 'timings':timings, 'generation_ms':timings['total_ms']}
        return {'arrays':arrays, 'metadata':metadata, 'preview_rgb':preview}
