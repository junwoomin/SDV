#!/usr/bin/env python3
"""Validate direct crops of the supplied PNG maps with the real LRS affine.

This uses actual map assets and previously verified CARLA world positions;
it never fabricates sensor images, static map geometry or elevation channels.
The source audit JSON is retained and augmented with reproducible production
crop checks. No CARLA server is required.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sdv.bev.global_map import catalog, load_global_map
from sdv.bev.global_generator import GlobalMapBEVGenerator
from validate_bev_references import reference_crop


CASES = [
    {'name':'Town04 lower', 'town':'Town04','expected_layer':'low',
     'pose':{'x':17.272193908691406,'y':27.153079986572266,'z':0.,
             'ground_z_m':0.,'road_id':47,'lane_id':-4}},
    {'name':'Town04 upper', 'town':'Town04','expected_layer':'high',
     'pose':{'x':17.272193908691406,'y':27.153079986572266,'z':11.,
             'ground_z_m':11.,'road_id':39,'lane_id':3}},
    {'name':'Town05 ramp', 'town':'Town05','expected_layer':'high',
     'pose':{'x':-52.895118713378906,'y':-207.51962280273438,'z':6.094699859619141,
             'ground_z_m':6.094699859619141,'road_id':37,'lane_id':-3}},
]


def font(size):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default()


def stats(values):
    return {'median_ms':statistics.median(values),'p95_ms':float(np.percentile(values,95)),
            'samples_ms':values}


def legacy_runtime_comparison(args):
    """Optional, runtime-only comparison; never rebuild the previous cache."""
    if not args.legacy_cache:
        return None
    if not args.legacy_xodr:
        raise ValueError('--legacy-xodr is required with --legacy-cache')
    import carla
    from sdv.bev.cache import GlobalRasterCache
    from sdv.bev.generator import HeightAwareBEVGenerator
    from validate_bev_offline_carla import pose_from_waypoint
    old_cache=GlobalRasterCache(args.legacy_cache)
    town=old_cache.metadata['town']
    native=carla.Map(town,Path(args.legacy_xodr).read_text())
    waypoints=native.generate_waypoints(5)
    center=waypoints[len(waypoints)//2].transform.location
    selected=sorted(waypoints,key=lambda p:p.transform.location.distance(center))[:20]
    actors=[dict(pose_from_waypoint(w,index+1),ground_verified=True,ground_lane_type='Driving')
            for index,w in enumerate(selected)]
    settings={'width':500,'height':500,'meters_per_pixel':.2}
    old=HeightAwareBEVGenerator(old_cache,dict(settings,mode='height_aware'))
    new=GlobalMapBEVGenerator(load_global_map(args.map_root,town,5),dict(settings,mode='global_raster',global_ppm=5))
    results={}
    for name,generator in (('previous_sparse_patch',old),('direct_global_png',new)):
        for _ in range(2):generator.generate(actors[0],actors)
        samples=[];phases={};reads=[]
        original=Path.read_bytes
        def tracked(path):
            if path.suffix.lower()=='.png':reads.append(str(path))
            return original(path)
        with patch.object(Path,'read_bytes',tracked):
            for _ in range(10):
                started=time.perf_counter();result=generator.generate(actors[0],actors)
                samples.append((time.perf_counter()-started)*1000)
                for phase,value in result['metadata'].get('timings',{}).items():phases.setdefault(phase,[]).append(value)
        results[name]={'total':stats(samples),'phases':{k:stats(v) for k,v in phases.items()},
            'included_actors':sum(row['included'] for row in result['metadata']['actors']),
            'png_reads_during_warm_runs':len(reads)}
    if results['direct_global_png']['png_reads_during_warm_runs']:
        raise AssertionError('Direct PNG generator reread global PNGs after warm-up')
    return {'scope':'Runtime only; static labels differ and no accuracy equivalence is claimed',
        'town':town,'previous_cache_signature':old_cache.metadata['signature'],
        'previous_cache_builder':old_cache.metadata['build_options']['builder'],
        'input_actor_provenance':'20 synthetic boxes placed on actual CARLA native Map waypoints; not recorded traffic',
        'ego_pose':actors[0],'output_size_wh':[500,500],'local_meters_per_pixel':.2,
        'new_global_ppm':5,'runs':10,'opencv_threads':cv2.getNumThreads(),'results':results}


def run(args):
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory/'global-map-source-verification.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    reference, source_sha = reference_crop(Path(args.lrs_root)/'sim'/'Global_Dynamic_MapManager.py')
    inventory = catalog(args.map_root)
    if inventory['errors'] or any(not v['available'] for town in inventory['town_records'] for v in town['variants']):
        raise AssertionError('Supplied global PNG catalog is unavailable or changed')
    checks, timings, images, cross_resolution = [], [], {}, []
    affine_errors = []
    for town in ('Town04','Town05'):
        cases = [case for case in CASES if case['town']==town]
        for ppm in (5,10):
            start = time.perf_counter()
            source = load_global_map(args.map_root,town,ppm)
            catalog_load_ms = (time.perf_counter()-start)*1000
            # Both native-resolution outputs cover the same 40x40 metres.
            size = 40*ppm
            generator = GlobalMapBEVGenerator(source, {'mode':'global_raster','global_ppm':ppm,
                'width':size,'height':size,'meters_per_pixel':1./ppm,
                'rotate_with_ego':True,'semantic_layers':['drivable','lane','road_boundary']})
            for case in cases:
                pose = dict(case['pose'], yaw=0., ground_verified=True,
                    ground_lane_type='Driving', ground_source='actual CARLA0.9.15 native Map waypoint audit')
                first_start = time.perf_counter()
                first = generator.generate(pose, [])
                first_ms = (time.perf_counter()-first_start)*1000
                if first['metadata']['map_layer_id'] != case['expected_layer']:
                    raise AssertionError('Wrong source PNG layer at the verified native ground position: '+case['name'])
                for yaw in (0,90,180,270):
                    pose['yaw'] = float(yaw)
                    result = generator.generate(pose, [])
                    arrays = result['arrays']
                    selection = source.select_layer(pose,channels=['drivable','lane','road_boundary'])
                    center = (np.array([pose['x'],pose['y']])-source.world_offset)*ppm
                    for channel, image in selection['arrays'].items():
                        expected = reference(image,tuple(center),math.radians(yaw),
                            out_size=size,out_mpp=1./ppm,pixels_per_meter=ppm)
                        count = int(np.count_nonzero(expected != arrays[channel]))
                        checks.append({'case':case['name'],'ppm':ppm,'yaw_deg':yaw,'channel':channel,
                            'size_wh':[size,size],'field_of_view_m':[40,40],
                            'source_file':selection['metadata']['source_files'][channel],
                            'mismatch_pixels':count})
                        if count:
                            raise AssertionError('Global crop differs from reviewed LRS source: '+str(checks[-1]))
                    matrix = np.asarray(result['metadata']['geometry']['world_to_bev_pixel'])
                    # The recorded affine must place ego exactly at its anchor.
                    anchor = matrix.dot([pose['x'],pose['y'],1.])
                    affine_errors.append(float(np.linalg.norm(anchor-[size/2,size/2]))/ppm)
                    if np.isfinite(arrays['elevation_m']).any() or arrays['elevation_validity'].any():
                        raise AssertionError('No height map was supplied; finite dense heights would be fabricated')
                    if not np.isin(arrays['lane'],(0,1)).all():
                        raise AssertionError('Semantic nearest-neighbor warp changed class IDs')
                    images[(case['name'],ppm,yaw)] = result['preview_rgb'].copy()
                    low_image = images.get((case['name'],5,yaw))
                    if ppm == 10 and low_image is not None:
                        # Native PNGs were independently rasterized; edge and
                        # paint widths need not be pixel-identical at 5/10PPM.
                        cross_resolution.append({'case':case['name'],'yaw_deg':yaw,
                            'same_world_pose':dict(pose),'field_of_view_m':[40,40],
                            'rgb_mismatch_after_10ppm_decimation':int(np.any(low_image!=result['preview_rgb'][::2,::2],axis=2).sum()),
                            'interpretation':'descriptive only; original 5/10PPM edge/paint rasterization may differ'})
                pose['yaw']=0.
                for _ in range(3):
                    generator.generate(pose,[])
                values,reads = {},[]
                original=Path.read_bytes
                def tracked(path):
                    if path.suffix.lower()=='.png':reads.append(str(path))
                    return original(path)
                with patch.object(Path,'read_bytes',tracked):
                    for _ in range(0 if args.skip_benchmark else args.runs):
                        result = generator.generate(pose,[])
                        for phase,value in result['metadata']['timings'].items():
                            values.setdefault(phase,[]).append(value)
                if reads:
                    raise AssertionError('Global PNGs were reread after cache warm-up')
                timings.append({'case':case['name'],'ppm':ppm,'output_size_wh':[size,size],
                    'catalog_verify_and_construct_ms':catalog_load_ms,
                    'first_frame_including_lazy_decode_ms':first_ms,'runs':0 if args.skip_benchmark else args.runs,
                    'warm_phases':{phase:stats(samples) for phase,samples in values.items()},
                    'png_reads_during_warm_runs':len(reads),
                    'actors':0,'excludes':['CARLA simulation','camera sensor processing','disk output','web transfer']})
            # Free large native layers before loading the next resolution.
            del generator,source
    width, top = 1420, 104
    compact = Image.new('RGB',(width,top+len(CASES)*290+120),'#0b1220')
    draw = ImageDraw.Draw(compact)
    draw.text((24,16),'Supplied global PNGs / direct crop at the same world pose',font=font(26),fill='white')
    draw.text((24,54),'5 PPM and 10 PPM | native pixels | identical 40 x 40 m footprint | no waypoint rasterization',font=font(17),fill='#b6c3d7')
    # One compact row per case: each heading has a 5PPM /10PPM pair.
    for row,case in enumerate(CASES):
        y0=top+row*290
        draw.text((24,y0),case['name']+' | Z={:.3f} m'.format(case['pose']['ground_z_m']),font=font(19),fill='white')
        for column,yaw in enumerate((0,90,180,270)):
            x0=24+column*348
            draw.text((x0,y0+31),'yaw {} deg'.format(yaw),font=font(16),fill='#b6c3d7')
            for item,ppm in enumerate((5,10)):
                # Preserve aspect ratio: square 160x160, never stretch labels.
                picture=Image.fromarray(images[(case['name'],ppm,yaw)]).resize((160,160),resample=Image.NEAREST)
                compact.paste(picture,(x0+item*166,y0+60))
                draw.text((x0+item*166,y0+229),'{} PPM / {} px'.format(ppm,40*ppm),font=font(14),fill='#b6c3d7')
            draw.text((x0,y0+259),'same XY, orientation and area',font=font(13),fill='#8b9cb4')
    y0=top+len(CASES)*290
    draw.text((24,y0),'Each crop is bit-identical to the reviewed LRS affine on that exact source PNG.',font=font(17),fill='white')
    draw.text((24,y0+29),'Different PPM source rasters can have different edge pixels; no forced resampling equality or synthetic labels.',font=font(15),fill='#b6c3d7')
    draw.text((24,y0+56),'Town05 ramp uses verified native Driving ground + exclusive high-DAS support; source PNG bytes are unchanged.',font=font(15),fill='#b6c3d7')
    compact.save(str(directory/'global-map-direct-crop-comparison.png'))
    root=Path(__file__).resolve().parents[1]
    result={'scope':'Actual supplied full-town PNGs cropped by production generator; no live CARLA server',
        'source_lrs_sha256':source_sha,'python':platform.python_version(),'numpy':np.__version__,'opencv':cv2.__version__,
        'production_source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (root/'sdv/bev/global_map.py',root/'sdv/bev/global_generator.py',root/'sdv/bev/geometry.py')},
        'catalog_variants_verified':sum(len(t['variants']) for t in inventory['town_records']),
        'exact_lrs_crop_checks':checks,'all_crop_checks_exact':all(not item['mismatch_pixels'] for item in checks),
        'maximum_recorded_affine_ego_error_m':max(affine_errors),
        'dense_elevation_absent_verified':True,'cross_resolution_descriptive':cross_resolution,
        'timings':timings,'limitations':['shared CPU host timing','no live sensors, traffic or camera synchronization',
            'native 5/10 PPM raster edge differences retained','no claim that every original PNG pixel is independently correct']}
    if not args.skip_benchmark:
        result['previous_runtime_comparison']=legacy_runtime_comparison(args)
    report['production_validation']=result
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('{} direct crops match actual LRS affine; {} catalog variants verified'.format(len(checks),result['catalog_variants_verified']))
    for item in timings:
        if item['warm_phases']:
            print(item['case'],item['ppm'],'PPM','warm total median={:.3f}ms'.format(item['warm_phases']['total_ms']['median_ms']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lrs-root',required=True)
    parser.add_argument('--map-root',default=None)
    parser.add_argument('--runs',type=int,default=30)
    parser.add_argument('--skip-benchmark',action='store_true',help='Run label/geometry checks only while other CPU jobs are active')
    parser.add_argument('--legacy-cache',help='Existing sparse patch cache directory; never rebuilt')
    parser.add_argument('--legacy-xodr',help='Actual matching native CARLA map for the optional runtime comparison')
    parser.add_argument('--output-dir',default=str(Path(__file__).resolve().parents[1]/'docs'/'validation'))
    args=parser.parse_args()
    if not 1<=args.runs<=1000:parser.error('--runs must be1..1000')
    run(args)


if __name__=='__main__':main()
