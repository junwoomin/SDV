"""Observed task labels from real train/validation files (never schema guesses)."""
from __future__ import annotations
from collections import OrderedDict, Counter
import copy
import json
import os
from pathlib import Path
import threading

import numpy as np
from sdv.data_contract import (BEV_LAYERS, OD_CLASSES, OD_COLORS, semantic_records,
    decode_segmentation, depth_encoding_for_scene, new_task_settings)
from .catalog import CAMERAS, safe_child
from .sequences import read_array, read_json, bbox_annotations, LAYER_COLORS
from .bev_preview import read_bev

_CACHE = OrderedDict()
_LOCK = threading.RLock()


def _scenes(root, settings):
    found = []
    for split, key in (('train', 'train_data'), ('val', 'val_data')):
        raw = Path(settings.get(key, 'data/' + split))
        path = raw if raw.is_absolute() else root / raw
        if path.is_symlink(): raise ValueError('Dataset symbolic links are not supported')
        path = path.resolve()
        if root not in path.parents: raise ValueError('Dataset must be inside the project')
        if not path.is_dir(): continue
        for folder, directories, _ in os.walk(str(path), followlinks=False):
            base = Path(folder)
            directories[:] = sorted(d for d in directories if not d.startswith('.') and not (base/d).is_symlink())
            if 'front_cam_rgb' in directories:
                found.append((split, base)); directories[:] = []
    return found


def task_catalog(root, settings, sensor_use=None):
    root = Path(root).resolve()
    selected = sensor_use or {'front_cam': True}
    if not isinstance(selected, dict) or any(k not in (*CAMERAS, 'lidar') or type(v) is not bool for k,v in selected.items()):
        raise ValueError('sensor_use must contain supported sensor names and booleans')
    cameras = [c for c in CAMERAS if selected.get(c)]
    scenes = _scenes(root, settings)
    inventory = []
    for split, scene in scenes:
        for camera in cameras:
            for suffix in ('_rgb','_seg_ids','_seg','_depth_m','_depth','_bbs'):
                folder = safe_child(scene, camera+suffix)
                if not folder.is_dir(): continue
                for path in sorted(folder.iterdir()):
                    if path.is_file() and not path.is_symlink() and path.suffix.lower() in {'.png','.npy'}:
                        st=path.stat(); inventory.append((str(path),st.st_size,st.st_mtime_ns))
        for layer in BEV_LAYERS.values():
            folder=safe_child(scene,layer['folder'])
            if folder.is_dir():
                st=folder.stat(); inventory.append((str(folder),st.st_size,st.st_mtime_ns))
        for name in ('collection_manifest.json','bev','bev_gt','info'):
            path=safe_child(scene,name)
            if path.exists():
                st=path.stat();inventory.append((str(path),st.st_size,st.st_mtime_ns))
                if name in ('bev_gt','info') and path.is_dir():
                    for child in sorted(path.iterdir()):
                        if child.is_file() and not child.is_symlink():
                            st=child.stat();inventory.append((str(child),st.st_size,st.st_mtime_ns))
    key=(str(root),tuple((s,str(p)) for s,p in scenes),tuple(cameras),tuple(inventory))
    with _LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key);return copy.deepcopy(_CACHE[key])
    seg, boxes, layers = Counter(), Counter(), Counter()
    encodings=set();warnings=[];unknown=empty=box_frames=seg_frames=0
    numeric_bev_frames=0;numeric_geometries={};height_channels=set();semantic_ids=Counter();bev_schemas=set()
    splits={k:{'scenes':0,'segmentation_pixels':{},'detection_boxes':{}} for k in ('train','val')}
    for split,scene in scenes:
        splits[split]['scenes']+=1
        numeric_keys=set()
        numeric_folder=safe_child(scene,'bev_gt')
        if numeric_folder.is_dir():
            for path in sorted(numeric_folder.glob('*.npz')):
                if path.is_symlink():continue
                try:
                    info=safe_child(scene,'info/'+path.stem+'.json')
                    raw=read_json(info) if info.is_file() and not info.is_symlink() else {}
                    metadata=raw.get('bev_gt') or {}
                    if not isinstance(metadata,dict):raise ValueError('bev_gt metadata must be an object')
                    arrays=read_bev(path,metadata)
                    collected=metadata.get('settings') or {}
                    if not isinstance(collected,dict):raise ValueError('BEV capture settings must be an object')
                    schema=metadata.get('schema_version')
                    available_static=metadata.get('available_static_channels')
                    if schema==4 and (not isinstance(available_static,list) or any(not isinstance(name,str) for name in available_static)):
                        raise ValueError('Global raster BEV must declare actual available_static_channels')
                    numeric_keys.add(path.stem);numeric_bev_frames+=1;bev_schemas.add(schema)
                    for old,new in [('das','drivable'),('lane','lane'),('vehicle','vehicle'),('walker','pedestrian')]:
                        group='semantic_layers' if new in ('drivable','lane') else 'actor_classes'
                        enabled=not isinstance(collected.get(group),list) or new in collected[group]
                        if schema==4 and group=='semantic_layers': enabled &= new in available_static
                        if new in arrays and enabled:layers[old]+=1
                    for name in ('elevation_m','object_ground_z_m','object_min_z_m','object_max_z_m','object_height_m'):
                        if name in arrays and np.isfinite(arrays[name]).any():height_channels.add(name)
                    ids,counts=np.unique(arrays['semantic'],return_counts=True)
                    for i,n in zip(ids,counts):semantic_ids[int(i)]+=int(n)
                    geometry=metadata.get('geometry')
                    if isinstance(geometry,dict):numeric_geometries[json.dumps(geometry,sort_keys=True)]=geometry
                    else:warnings.append('BEV numeric frame has no recorded geometry: '+str(path.relative_to(root)))
                except (OSError,ValueError,TypeError,KeyError) as exc:
                    if len(warnings)<30:warnings.append('BEV {}/{}: {}'.format(scene.name,path.name,exc))
        for layer, spec in BEV_LAYERS.items():
            folder=safe_child(scene,spec['folder'])
            if folder.is_dir():layers[layer]+=sum(p.is_file() and not p.is_symlink() and p.stem not in numeric_keys for p in folder.glob('*.png'))
            elif layer in ('das','lane') and (scene/'bev').is_dir():layers[layer]+=sum(p.stem not in numeric_keys for p in (scene/'bev').glob('*.png'))
        for camera in cameras:
            rgb=safe_child(scene,camera+'_rgb')
            if not rgb.is_dir():continue
            for image in sorted(rgb.glob('*.png')):
                if image.is_symlink():continue
                stem=image.stem
                try:
                    raw=safe_child(scene,camera+'_seg_ids/'+stem+'.png')
                    mask=raw if raw.is_file() else safe_child(scene,camera+'_seg/'+stem+'.png')
                    if mask.is_file():
                        ids=decode_segmentation(read_array(mask));ids,count=np.unique(ids,return_counts=True)
                        for i,n in zip(ids,count):
                            if int(i)==255:unknown+=int(n)
                            else:
                                seg[int(i)]+=int(n)
                                counts=splits[split]['segmentation_pixels'];counts[str(int(i))]=counts.get(str(int(i)),0)+int(n)
                        seg_frames+=1
                    boxfile=safe_child(scene,camera+'_bbs/'+stem+'.npy')
                    if boxfile.is_file():
                        annotations=bbox_annotations(boxfile);box_frames+=1;empty+=not annotations
                        for item in annotations:
                            i=item['class_id'];boxes[i]+=1
                            counts=splits[split]['detection_boxes'];counts[str(i)]=counts.get(str(i),0)+1
                    depth=safe_child(scene,camera+'_depth_m/'+stem+'.npy')
                    if not depth.is_file():depth=safe_child(scene,camera+'_depth/'+stem+'.png')
                    if depth.is_file():encodings.add(depth_encoding_for_scene(scene,path=depth))
                except (OSError,ValueError,TypeError) as exc:
                    if len(warnings)<30:warnings.append('{}/{}/{}: {}'.format(scene.name,camera,stem,exc))
    if unknown:warnings.append('Unknown segmentation pixels: {}'.format(unknown))
    if box_frames and not boxes:warnings.append('Bounding-box files exist but contain no objects. Detection classes cannot be selected from this dataset.')
    if 'meters_u8' in encodings:warnings.append('Legacy native depth PNG stores rounded metres; value 255 is saturated and invalid. Lost precision cannot be recovered.')
    if not scenes:warnings.append('No actual train/validation scenes were found.')
    recommendations=new_task_settings(sorted(int(i) for i in splits['train']['segmentation_pixels']),sorted(int(i) for i in splits['train']['detection_boxes']))
    recommendations['bev']['layers']=[k for k,v in BEV_LAYERS.items() if v['trainable'] and layers[k]]
    recommendations['bev']['visual_layers']=[k for k in BEV_LAYERS if layers[k]]
    result={
        'segmentation':{'classes':[dict(row,pixels=seg[row['id']],train_pixels=splits['train']['segmentation_pixels'].get(str(row['id']),0),val_pixels=splits['val']['segmentation_pixels'].get(str(row['id']),0)) for row in semantic_records(sorted(seg))],
                        'frames':seg_frames,'unknown_pixels':unknown,'encoding':'CARLA 0.9.15 semantic IDs / canonical RGB palette'},
        'detection':{'classes':[{'id':i,'name':OD_CLASSES[i],'color':list(OD_COLORS[i]),'boxes':n} for i,n in sorted(boxes.items())],
                     'annotation_count':sum(boxes.values()),'frames':box_frames,'empty_frames':empty,
                     'format':'N x 5 [x1,y1,x2,y2,class_id], 2D camera pixels; object IDs unavailable'},
        'depth':{'unit':'m','encodings':sorted(encodings),'warnings':[w for w in warnings if 'depth' in w.lower()]},
        'bev':{'layers':[dict(v,id=k,color=LAYER_COLORS[k],available=bool(layers[k]),frames=layers[k]) for k,v in BEV_LAYERS.items()],
               'geometry':{'readonly':True,'source_width_px':500,'source_height_px':500,'source_resolution_m':0.2,
                           'training_crop_px':[150,150,350,350],'output_width_px':200,'output_height_px':200,
                           'front_m':20,'rear_m':20,'left_m':20,'right_m':20,
                           'provenance':'existing_collector_and_loader_code_convention',
                           'dataset_geometry_verified':False,
                           'projection_grid_m':[-20,20],
                           'note':'Code convention, not verified calibration of arbitrary datasets. Existing label crop and projection grid cover ±20 m (stored source BEV ±50 m); BEV camera-axis calibration requires further validation. Geometry is read-only.'}},
        'splits':splits,'selected_cameras':cameras,'recommended_task_settings':recommendations,'warnings':warnings,
        'scope':'all_aligned_rgb_frames_in_selected_cameras'}
    if numeric_bev_frames:
        from sdv.bev.schema import SEMANTIC_CLASSES
        result['bev'].update(schema_version=next(iter(bev_schemas)) if len(bev_schemas)==1 else None,
            schema_versions=sorted(value for value in bev_schemas if isinstance(value,int)),numeric_frames=numeric_bev_frames,
            height_channels=sorted(height_channels),
            semantic_classes=[dict(row,pixels=semantic_ids[row['id']]) for row in SEMANTIC_CLASSES if semantic_ids[row['id']]],
            recorded_geometries=list(numeric_geometries.values()),
            provenance=('provided_global_raster_png_and_simulator_actors' if 4 in bev_schemas else 'simulator_ground_truth')+
                '; metric heights only where calibrated; height channels are not online sensor inputs')
        result['bev']['geometry']={'readonly':True,'source':'per-frame info.bev_gt.geometry',
            'recorded_geometries':list(numeric_geometries.values()),'numeric_frames':numeric_bev_frames,
            'training_projection_grid_m':[-20,20],
            'note':'Numeric ground truth is resampled in metric coordinates. Legacy datasets retain their legacy crop. Height channels are separate targets and do not change model input channels.'}
    with _LOCK:
        _CACHE[key]=copy.deepcopy(result)
        while len(_CACHE)>8:_CACHE.popitem(last=False)
    return result


def selection_blockers(root, settings, config):
    if 'task_settings' not in config:return []
    available=task_catalog(root,settings,config.get('sensor_use'))
    advanced=config['task_settings'];tasks=config['model_conf'];issues=[]
    for task,key,field in (('Seg','segmentation','class_ids'),('od','detection','class_ids')):
        if not tasks.get(task) or (task=='Seg' and advanced[key]['mode']=='legacy'):continue
        known={row['id'] for row in available[key]['classes']}
        missing=set(advanced[key][field])-known
        if missing:issues.append('{}: selected classes have no actual labels in these datasets: {}'.format(key,sorted(missing)))
        train_field='segmentation_pixels' if task=='Seg' else 'detection_boxes'
        trained=available['splits']['train'][train_field]
        absent=[i for i in advanced[key][field] if str(i) not in trained and i not in missing]
        if absent:issues.append('{}: selected classes occur only outside the training split: {}'.format(key,absent))
    if tasks.get('BEV'):
        known={r['id'] for r in available['bev']['layers'] if r['available'] and r['trainable']}
        missing=set(advanced['bev']['layers'])-known
        if missing:issues.append('BEV: selected layers have no actual labels: '+', '.join(sorted(missing)))
    return issues
