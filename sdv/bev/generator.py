"""Online ground-truth composition from an immutable offline surface cache.

Road surfaces are connected physical patches, not fixed global Z bins. Local
overlaps select the shortest connected surface path from the ego contact road.
Unknown/disconnected surfaces remain invalid, rather than becoming free space.
"""
import math
import time
import numpy as np
from .geometry import BEVGeometry, world_to_local, local_to_world, local_to_global_affine
from .schema import (SCHEMA_VERSION, STATIC_CHANNELS, BINARY_CHANNELS, HEIGHT_CHANNELS, SEMANTIC_ORDER,
                     channel_schema, normalize_bev_settings)
from .io import semantic_from_channels, colorize

def _finite(value):
    try: return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError): return False

class HeightAwareBEVGenerator:
    def __init__(self, cache, settings=None):
        self.cache = cache
        self.settings = normalize_bev_settings(settings)
        self.geometry = BEVGeometry(**{k:self.settings[k] for k in
            ('width','height','meters_per_pixel','anchor_x','anchor_y')})
        # This index is immutable for the entire map cache and recorded once
        # in the collection manifest; per-frame metadata lists only used IDs.
        self.surface_table = {sid:i for i,sid in enumerate(sorted(cache.patches))}

    def _ground(self, record):
        # Lowest bbox corner is not located at actor-center XY on a slope.
        # Use the transformed bottom-face center when available, otherwise
        # CARLA actor reference XYZ; bbox top is NEVER a road selector.
        contact=record.get('ground_contact_world_xyz')
        if isinstance(contact,(tuple,list)) and len(contact)==3 and all(_finite(x) for x in contact):
            x,y,z=contact
        else: x,y,z=record.get('x'),record.get('y'),record.get('z')
        if not all(_finite(v) for v in (x,y,z)):
            return None
        options = {k:record.get(k) for k in ('road_id','lane_id','section_id')}
        result = self.cache.lookup_ground(x,y,float(z), **options)
        if result is None and any(v is not None for v in options.values()):
            result = self.cache.lookup_ground(x,y,float(z))
        return result

    def _crop_patch(self, patch, ego_pose):
        """Warp only the patch's local bounding rectangle, not the whole frame."""
        import cv2
        g = self.geometry
        x0,y0,x1,y1 = patch['bounds_world']
        corners = world_to_local(np.array([[x0,y0],[x0,y1],[x1,y0],[x1,y1]]),ego_pose,g)
        lo = np.maximum(np.floor(corners.min(axis=0)).astype(int)-2,0)
        hi = np.minimum(np.ceil(corners.max(axis=0)).astype(int)+3,(g.width,g.height))
        if np.any(hi<=lo): return None
        matrix = local_to_global_affine(ego_pose,g,patch['offset_world'],self.cache.meters_per_pixel)
        matrix[:,2] += matrix[:,:2].dot(lo.astype(np.float32))
        source = self.cache.arrays(patch['id'])
        names = STATIC_CHANNELS + ('validity','lane_validity','height')
        packed = np.stack([source.get(name,source['validity'] if name=='lane_validity' else np.zeros_like(source['validity']))
                           for name in names],axis=-1).astype(np.float32)
        out = cv2.warpAffine(packed,matrix,tuple((hi-lo).tolist()),
            flags=cv2.INTER_NEAREST|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        return lo,hi,{name:out[...,i] for i,name in enumerate(names)}

    def _polygon(self, points, pose):
        import cv2
        g = self.geometry
        out = np.zeros((g.height,g.width),dtype=np.uint8)
        points = np.asarray(points,dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2 or not np.isfinite(points).all():
            return out
        uv = world_to_local(points[:,:2],pose,g)
        # Safe subpixel integer rasterization, no antialiasing in class masks.
        uv = np.clip(uv,-1e6,1e6)
        hull = cv2.convexHull(np.rint(uv*256).astype(np.int32))
        cv2.fillConvexPoly(out,hull,1,lineType=cv2.LINE_8,shift=8)
        return out

    def _polygon_indices(self, points, pose):
        """Equivalent polygon pixels, allocating only its clipped rectangle."""
        import cv2
        g=self.geometry
        points=np.asarray(points,dtype=np.float64)
        empty=np.empty(0,dtype=np.intp)
        if points.ndim!=2 or points.shape[0]<3 or points.shape[1]<2 or not np.isfinite(points).all():
            return empty,empty
        uv=np.clip(world_to_local(points[:,:2],pose,g),-1e6,1e6)
        lo=np.maximum(np.floor(uv.min(axis=0)).astype(int)-1,0)
        hi=np.minimum(np.ceil(uv.max(axis=0)).astype(int)+2,(g.width,g.height))
        if np.any(hi<=lo): return empty,empty
        out=np.zeros((hi[1]-lo[1],hi[0]-lo[0]),dtype=np.uint8)
        hull=cv2.convexHull(np.rint(uv*256).astype(np.int32)-lo.astype(np.int32)*256)
        cv2.fillConvexPoly(out,hull,1,lineType=cv2.LINE_8,shift=8)
        rows,cols=np.nonzero(out)
        return rows+lo[1],cols+lo[0]

    def _line(self, points, pose, arrays, width_m=.3):
        import cv2
        g = self.geometry
        result = np.zeros((g.height,g.width),dtype=np.uint8)
        points = np.asarray(points,dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            return result
        # Per-segment elevation prevents a route/signal on an upper road from
        # being projected onto a lower one. Raster nearest labels only.
        for a,b in zip(points[:-1],points[1:]):
            uv = world_to_local(np.array([a[:2],b[:2]]),pose,g)
            segment = np.zeros_like(result)
            cv2.line(segment,tuple(np.rint(uv[0]).astype(int)),tuple(np.rint(uv[1]).astype(int)),1,
                     max(1,int(round(width_m/g.meters_per_pixel))),cv2.LINE_8)
            rows,cols = np.nonzero(segment)
            if not len(rows): continue
            xy = local_to_world(np.column_stack((cols,rows)),pose,g)
            delta = b[:2]-a[:2]; length2 = float(delta.dot(delta))
            t = np.zeros(len(rows)) if length2 < 1e-12 else np.clip((xy-a[:2]).dot(delta)/length2,0,1)
            z = a[2]+t*(b[2]-a[2])
            accept = arrays['validity'][rows,cols].astype(bool)
            if self.settings['height_aware']:
                accept &= np.abs(arrays['elevation_m'][rows,cols]-z) <= .75
            result[rows[accept],cols[accept]] = 1
        return result

    def generate(self, ego_pose, actors, traffic=None, route=None, frame_id=None, timestamp=None):
        started = time.perf_counter()
        g = self.geometry; shape=(g.height,g.width)
        arrays = {name:np.zeros(shape,dtype=np.uint8) for name in BINARY_CHANNELS}
        arrays.update({name:np.full(shape,np.nan,dtype=np.float32) for name in HEIGHT_CHANNELS})
        arrays['actor_id'] = np.zeros(shape,dtype=np.int32)
        object_top = np.full(shape,np.nan,dtype=np.float32)
        arrays['surface_id'] = np.full(shape,-1,dtype=np.int32)
        ground = self._ground(ego_pose)
        bounds = local_to_world(np.array([[0,0],[g.width,0],[0,g.height],[g.width,g.height]]),ego_pose,g)
        lo=bounds.min(axis=0);hi=bounds.max(axis=0)
        # A local connectivity halo includes lane transitions just outside the
        # picture; it does not connect distant overpass ramps across the town.
        halo=max(10.,min(30.,max(g.width,g.height)*g.meters_per_pixel*.25))
        candidates = self.cache.query((lo[0]-halo,lo[1]-halo,hi[0]+halo,hi[1]+halo))
        candidate_ids = {p['id'] for p in candidates}
        distances = self.cache.graph_distances(ground['patch_id'],candidate_ids) if ground else {}
        if not self.settings['height_aware']:
            distances = {p['id']:abs(float(p['center_xyz'][2])-float(ego_pose['z'])) for p in candidates}
        score = np.full(shape,np.inf,dtype=np.float32)
        used={}; warnings=[]
        if ground is None: warnings.append('Ego ground surface is unresolved; height-aware labels remain invalid.')
        for patch in candidates:
            if patch['id'] not in distances: continue
            cropped = self._crop_patch(patch,ego_pose)
            if cropped is None: continue
            xy0,xy1,local = cropped; x0,y0=xy0;x1,y1=xy1
            valid = (local['validity'] > 0) & np.isfinite(local['height'])
            # Metric topology distance plus distance within this small patch.
            cu,cv = world_to_local(patch['center_xyz'][:2],ego_pose,g)
            yy,xx = np.ogrid[y0:y1,x0:x1]
            cost = distances[patch['id']] + np.hypot(xx-cu,yy-cv)*g.meters_per_pixel
            take = valid & (cost < score[y0:y1,x0:x1])
            if not take.any(): continue
            score[y0:y1,x0:x1][take]=cost[take]
            for name in STATIC_CHANNELS:
                enabled = name=='road' or name in self.settings['semantic_layers']
                arrays[name][y0:y1,x0:x1][take] = (local[name][take]>0).astype(np.uint8) if enabled else 0
            arrays['validity'][y0:y1,x0:x1][take]=1
            arrays['lane_validity'][y0:y1,x0:x1][take]=(local['lane_validity'][take]>0).astype(np.uint8)
            arrays['elevation_m'][y0:y1,x0:x1][take]=local['height'][take]
            arrays['surface_id'][y0:y1,x0:x1][take]=self.surface_table[patch['id']]
            used[patch['id']]={'index':self.surface_table[patch['id']], 'road_id':patch['road_id'],
                'section_id':patch['section_id'],'lane_id':patch['lane_id']}
        arrays['occupancy_validity'][:]=arrays['validity']
        records=[]
        ego_id=ego_pose.get('actor_id',ego_pose.get('id'))
        source_actors=list(actors)
        if ego_pose.get('footprint_world_xyz') is not None and not any(a.get('actor_id',a.get('id'))==ego_id for a in source_actors):
            source_actors.append(dict(ego_pose,object_class='vehicle'))
        for actor in source_actors:
            kind=actor.get('object_class',actor.get('class'))
            aid=actor.get('actor_id',actor.get('id'))
            is_ego=ego_id is not None and aid==ego_id
            if kind not in self.settings['actor_classes'] and not is_ego: continue
            record=dict(actor)
            fr,fc=self._polygon_indices(actor.get('footprint_world_xyz',[]),ego_pose)
            if not len(fr): continue
            aground=self._ground(actor)
            record.update(map_layer_id=aground['layer_id'] if aground else None,
                ground_z_m=aground['ground_z_m'] if aground else None,
                ground_source='map_surface_triangle' if aground else None)
            on_surface=arrays['validity'][fr,fc].astype(bool)
            rr,cc=fr[on_surface],fc[on_surface]
            if aground is None:
                arrays['occupancy_validity'][fr,fc]=0
                record.update(included=False,reason='ground_surface_unresolved');records.append(record);continue
            if self.settings['height_aware']:
                # A pitched vehicle spans changing road elevation. Compare the
                # road surface plane, not one flat actor-center Z across it.
                polygon=np.asarray(self.cache.patches[aground['patch_id']]['polygon_xyz'],dtype=np.float64)
                coeff=np.linalg.lstsq(np.column_stack((polygon[:,:2],np.ones(len(polygon)))),polygon[:,2],rcond=None)[0]
                xy=local_to_world(np.column_stack((cc,rr)),ego_pose,g)
                expected=xy.dot(coeff[:2])+coeff[2]
                same_height=np.abs(arrays['elevation_m'][rr,cc]-expected)<=.75
                # Similar absolute Z alone is insufficient: a separate deck
                # can be close vertically. Require local physical affiliation.
                points=np.asarray(actor['footprint_world_xyz'],dtype=np.float64)
                diagonal=float(np.linalg.norm(points[:,:2].max(axis=0)-points[:,:2].min(axis=0)))
                paths=self.cache.graph_distances(aground['patch_id'],candidate_ids,max_distance=diagonal+4.)
                affiliated=[self.surface_table[sid] for sid,distance in paths.items() if distance<=diagonal+4.]
                keep=same_height & np.isin(arrays['surface_id'][rr,cc],affiliated)
                rr,cc=rr[keep],cc[keep]
            if not len(rr):
                record.update(included=False,reason='outside_selected_surface');records.append(record);continue
            values=[actor.get('bbox_min_z_m'),actor.get('bbox_max_z_m'),actor.get('object_height_m')]
            if not all(_finite(v) for v in values) or values[1]<values[0] or values[2]<0 or not isinstance(aid,int) or not 0<aid<2147483647:
                arrays['occupancy_validity'][fr,fc]=0
                record.update(included=False,reason='invalid_bbox_or_actor_id');records.append(record);continue
            channel='ego' if is_ego else kind
            if channel not in arrays: continue
            arrays[channel][rr,cc]=1
            # Overlapping instances remain independent in metadata; an integer
            # ID raster uses the uppermost bbox, never additive corrupt IDs.
            above=(~np.isfinite(object_top[rr,cc]) | (float(values[1])>object_top[rr,cc]))
            ar,ac=rr[above],cc[above]
            object_top[ar,ac]=values[1]
            arrays['actor_id'][ar,ac]=aid
            if self.settings['object_height']:
                arrays['object_ground_z_m'][ar,ac]=arrays['elevation_m'][ar,ac]
                for name,value in [('object_min_z_m',values[0]),
                                   ('object_max_z_m',values[1]),('object_height_m',values[2])]:
                    arrays[name][ar,ac]=value
            record.update(included=True,occupancy_pixels=int(len(rr)),is_ego=is_ego)
            records.append(record)
        signals=[]
        for light in traffic or []:
            state=str(light.get('state','Unknown')).split('.')[-1].lower()
            count=0
            if state in ('red','yellow','green'):
                for segment in light.get('segments',[]):
                    mask=self._line(segment,ego_pose,arrays)
                    arrays['stop_'+state] |= mask;count+=int(mask.sum())
            signals.append(dict(light,rasterized_pixels=count))
        if route: arrays['route']=self._line(route,ego_pose,arrays,width_m=.5)
        arrays['semantic']=semantic_from_channels(arrays)
        visible=dict(arrays)
        enabled=set(self.settings['visual_overlays'])
        for name,_ in SEMANTIC_ORDER:
            if name not in enabled and not (name.startswith('stop_') and 'stoplines' in enabled):
                visible[name]=np.zeros(shape,dtype=np.uint8)
        if not self.settings['height_aware']: warnings.append('Height filtering disabled: 2D projection can merge road levels.')
        disconnected=sum(p['id'] not in distances for p in candidates)
        if disconnected: warnings.append('{} local patches have no verified local topology path and are omitted.'.format(disconnected))
        skipped=(self.cache.metadata.get('build_report') or {}).get('skipped_edge_count',0)
        if skipped: warnings.append('{} discontinuous map edges were excluded during cache construction; see map build report.'.format(skipped))
        metadata={'schema_version':SCHEMA_VERSION,'label_source':'carla_simulator_ground_truth',
            'online_input':False,'geometry':g.metadata(),'settings':dict(self.settings),
            'frame_id':frame_id,'timestamp':timestamp,'ego_pose':{k:ego_pose[k] for k in ('x','y','z','yaw')},
            'ground_z_m':ground['ground_z_m'] if ground else None,'map_layer_id':ground['layer_id'] if ground else None,
            'road_id':ground['road_id'] if ground else None,'lane_id':ground['lane_id'] if ground else None,
            'map_cache_signature':self.cache.metadata['signature'],'surface_table':used,
            'actors':records,'traffic':signals,'channel_schema':channel_schema(),
            'valid_surface_fraction':float(arrays['validity'].mean()),'warnings':warnings,
            'ground_assignment':'bbox_bottom_face_center_or_actor_origin_then_map_surface; topology-local; no bbox-top threshold',
            'layer_policy':'shortest_connected_surface_path_within_local_roi',
            'ground_match_tolerance_m':2.0,'actor_surface_tolerance_m':.75,
            'generation_ms':(time.perf_counter()-started)*1000.}
        return {'arrays':arrays,'metadata':metadata,'preview_rgb':colorize(semantic_from_channels(visible))}
