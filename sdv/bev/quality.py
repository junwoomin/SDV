"""Frame-level evidence checks for simulator BEV ground truth."""
import math
import numpy as np
from .io import validate_arrays, semantic_from_channels

def validate_frame(arrays, metadata, frame_id=None, timestamp=None):
    issues=[]
    def issue(code,message,severity='error'):
        result={'code':code,'message':message}
        if severity!='error':result['severity']=severity
        issues.append(result)
    if not isinstance(metadata,dict):
        issue('bev_metadata','BEV metadata must be an object');return issues
    try: validate_arrays(arrays,metadata.get('geometry'),metadata=metadata)
    except ValueError as error:
        issue('bev_array_contract',str(error));return issues
    if metadata.get('schema_version') not in (3,4): issue('bev_schema','Expected BEV schema version 3 or 4')
    if metadata.get('schema_version')==4:
        from .training import validate_raster_affine
        try: validate_raster_affine(metadata)
        except ValueError as error: issue('bev_raster_affine',str(error))
        if metadata.get('static_label_source')!='provided_global_raster_png':
            issue('bev_static_provenance','Schema 4 must identify the provided global PNG source')
        if not isinstance(metadata.get('available_static_channels'),list):
            issue('bev_static_availability','Schema 4 must identify actual available static channels')
        if not np.isfinite(arrays['elevation_m']).any():
            issue('bev_metric_elevation_unavailable','Static PNGs contain no calibrated metric road elevation; height values remain unknown.', 'warning')
    if metadata.get('label_source')!='carla_simulator_ground_truth' or metadata.get('online_input') is not False:
        issue('bev_provenance','Simulator labels must be separated from online model inputs')
    if frame_id is not None and metadata.get('frame_id')!=frame_id:
        issue('bev_frame_alignment','BEV frame differs from RGB/world frame')
    if timestamp is not None:
        value=metadata.get('timestamp')
        if not isinstance(value,(int,float)) or not math.isfinite(value) or abs(value-timestamp)>1e-5:
            issue('bev_timestamp_alignment','BEV timestamp differs from RGB/world timestamp')
    if not np.array_equal(arrays['semantic'],semantic_from_channels(arrays)):
        issue('bev_semantic_mapping','Semantic IDs differ from independent channel composition')
    finite=np.isfinite(arrays['object_height_m'])
    if np.any(finite & (np.abs(arrays['object_height_m']-(arrays['object_max_z_m']-arrays['object_min_z_m']))>1e-3)):
        issue('bev_bbox_height','Object height differs from bounding box max Z minus min Z')
    ground=np.isfinite(arrays['object_ground_z_m']) & np.isfinite(arrays['elevation_m'])
    if np.any(ground & (np.abs(arrays['object_ground_z_m']-arrays['elevation_m'])>.751)):
        issue('bev_actor_road_height','Object ground and assigned road elevation disagree')
    table=metadata.get('surface_table') or {}
    if not isinstance(table,dict):
        issue('bev_surface_lookup','Surface table must be an object');table={}
    indices=[r.get('index') for r in table.values() if isinstance(r,dict)]
    if np.any(arrays['validity'].astype(bool) & ~np.isin(arrays['surface_id'],indices)):
        issue('bev_surface_lookup','Valid pixels reference an unknown surface ID')
    actors=metadata.get('actors',[])
    if not isinstance(actors,list) or any(not isinstance(a,dict) for a in actors):
        issue('bev_actor_metadata','Actors must be a list of objects');actors=[]
    ids={a.get('actor_id',a.get('id')) for a in actors if a.get('included') and isinstance(a.get('actor_id',a.get('id')),int)}
    actual=set(int(x) for x in np.unique(arrays['actor_id']) if x)
    if actual-ids: issue('bev_actor_metadata','Occupancy actor IDs are missing from same-frame metadata')
    return issues
