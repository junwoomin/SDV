#!/usr/bin/env python3
"""Compare real LRS/HBLR maps and new SDV geometry at the same overpass poses.

Requires the reviewed source/map directories and the offline CARLA validation
report/cache. No simulator, RGB images or fabricated actors are used. Different
reference raster semantics/anchors are documented, not scored as an IoU test.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sdv.bev.cache import GlobalRasterCache
from sdv.bev.generator import HeightAwareBEVGenerator
from sdv.bev.geometry import world_to_global
from sdv.bev.io import save_frame
from validate_bev_references import reference_crop


HBLR_SOURCE_SHA256 = '9f1fa1499ff4163ff8f5d327d91c6abd9fd4a040263e2a73b4e83548bdaa8539'


def load_hblr(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != HBLR_SOURCE_SHA256:
        raise ValueError('HBLR-linked source differs from the reviewed commit')
    spec = importlib.util.spec_from_file_location('reviewed_hblr_reference', str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def static_rgb(road, lane):
    image = np.full(road.shape + (3,), (18, 25, 39), dtype=np.uint8)
    image[road > 0] = (70, 88, 112)
    image[lane > 0] = (224, 235, 249)
    return image


def font(size):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default()


def compare(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads(Path(args.validation_report).read_text())
    town = next(row for row in report['maps'] if row['town'] == 'Town04')
    crossing = town['actual_stacked_road_locations'][0]
    cache_directory = next(path.parent for path in (Path(args.cache_root)/'Town04').glob('*/metadata.json')
        if json.loads(path.read_text())['signature'] == town['cache_signature'])
    cache = GlobalRasterCache(cache_directory)
    lrs_root, hblr_root = Path(args.lrs_root), Path(args.hblr_root)
    crop, lrs_source_sha = reference_crop(lrs_root/'sim'/'Global_Dynamic_MapManager.py')
    hblr = load_hblr(hblr_root/'data_gen'/'collector'/'bev.py')
    hblr_maps = hblr.BEVMap(hblr_root/'data_gen'/'data5', 'Town04')
    lrs_maps = lrs_root/'sim_using_data'/'data5'/'Town04'
    offset = np.load(str(lrs_maps/'world_offset.npy'), allow_pickle=False)
    renderer = HeightAwareBEVGenerator(cache, {'width':200, 'height':200, 'meters_per_pixel':.2})
    rows, results = [], []
    controlled_actors = []
    for index, level in enumerate(('lower','upper')):
        patch = cache.patches[crossing[level+'_surface']]
        x,y = crossing['xy']; z = crossing[level+'_ground_z_m']
        controlled_actors.append({'actor_id':100+index,'object_class':'vehicle',
            'x':x,'y':y,'z':z+.2,'yaw':0.,'road_id':patch['road_id'],
            'lane_id':patch['lane_id'],'section_id':patch['section_id'],
            'bbox_min_z_m':z+.05,'bbox_max_z_m':z+1.8,'object_height_m':1.75,
            'footprint_world_xyz':[[x+dx,y+dy,z] for dx,dy in ((-2.25,-1),(2.25,-1),(2.25,1),(-2.25,1))]})
    for level in ('lower', 'upper'):
        patch = cache.patches[crossing[level+'_surface']]
        x, y = crossing['xy']
        z = crossing[level+'_ground_z_m']
        pose = {'x': x, 'y': y, 'z': z, 'yaw': 0.,
            'road_id': patch['road_id'], 'lane_id': patch['lane_id'], 'section_id': patch['section_id']}
        info = {'ego_vehicle_loc': {'x':x, 'y':y, 'z':z}, 'ego_vehicle_rot': {'yaw':0.}}
        suffix = '_high' if level == 'upper' else ''
        filenames = ['das_full'+suffix+'.png', 'lane_full'+suffix+'.png']
        lrs_layers = [cv2.imread(str(lrs_maps/name), cv2.IMREAD_GRAYSCALE) for name in filenames]
        if any(layer is None for layer in lrs_layers):
            raise FileNotFoundError('Reviewed LRS Town04 layers are required')
        # Original LRS caller truncates global ego pixels before the exact
        # source affine. Preserve it for this reference, not in new SDV.
        center = tuple(np.trunc(world_to_global((x,y), offset, .2)).astype(int))
        lrs = [crop(layer, center, math.radians(0)) for layer in lrs_layers]
        old_hblr = hblr_maps.static_masks(info)
        new = renderer.generate(pose, [], frame_id=None, timestamp=None)
        arrays = new['arrays']
        save_frame(output/('bev-town04-'+level+'-actual-map.npz'), arrays)
        height = arrays['elevation_m']
        valid = np.isfinite(height) & arrays['validity'].astype(bool)
        scaled = np.rint(np.clip(np.nan_to_num(height, nan=0)/12,0,1)*255).astype(np.uint8)
        height_rgb = cv2.cvtColor(cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
        height_rgb[~valid] = (18,25,39)
        rows.append([static_rgb(*lrs), static_rgb(old_hblr['das'], old_hblr['lane']),
                     new['preview_rgb'], height_rgb])
        anchor = (100,100)
        center_height = float(height[anchor[1],anchor[0]]) if valid[anchor[1],anchor[0]] else None
        if center_height is None or abs(center_height-z) > .1:
            raise AssertionError('New BEV selected the wrong overpass surface at the ego anchor')
        actor_check = renderer.generate(pose,controlled_actors)['metadata']['actors']
        included = sorted(row['actor_id'] for row in actor_check if row['included'])
        expected_id = 100 if level == 'lower' else 101
        if included != [expected_id]:
            raise AssertionError('Controlled boxes at different actual map heights merged incorrectly')
        results.append({'level': level, 'ego_pose': pose,
            'actual_surface': patch['id'], 'actual_surface_kind': patch['kind'],
            'selected_anchor_height_m': center_height,
            'anchor_ground_error_m': abs(center_height-z),
            'valid_pixels': int(valid.sum()),
            'height_range_m': [float(height[valid].min()), float(height[valid].max())],
            'semantic_class_ids': np.unique(arrays['semantic']).tolist(),
            'controlled_actor_height_check': {'provenance':'Two synthetic 4.5x2m boxes at identical XY on actual native map heights; not recorded/live actors',
                'input_ids':[100,101],'expected_included_ids':[expected_id],'actual_included_ids':included,
                'records':actor_check},
            'lrs_images': [{'file': name, 'sha256': hashlib.sha256((lrs_maps/name).read_bytes()).hexdigest()}
                           for name in filenames],
            'warnings': new['metadata'].get('warnings', [])})
    width, header, row_height, footer = 1336, 100, 386, 122
    mosaic = Image.new('RGB',(width, header+2*row_height+footer),'#0b1220')
    draw = ImageDraw.Draw(mosaic)
    draw.text((24,16),'Town04 / the same XY on two actual road levels',font=font(24),fill='white')
    draw.text((24,50),'x={:.3f} m  y={:.3f} m  yaw=0 deg | 40 x 40 m | 0.2 m/pixel'.format(*crossing['xy']),
              font=font(17),fill='#b6c3d7')
    titles = ['LRS original crop', 'HBLR-linked original crop', 'SDV semantic ground truth', 'SDV road elevation (m)']
    for row_index, tiles in enumerate(rows):
        top = header + row_index*row_height
        item = results[row_index]
        for col, rgb in enumerate(tiles):
            left = 24+col*328
            draw.text((left,top),titles[col],font=font(17),fill='white')
            image = Image.fromarray(rgb).resize((304,304),resample=Image.NEAREST)
            mosaic.paste(image,(left,top+32))
            # A UI crosshair, not part of any saved ground-truth array.
            cu,cv = left+152,top+32+152
            draw.line((cu-5,cv,cu+5,cv),fill='#f8b66c',width=1)
            draw.line((cu,cv-5,cu,cv+5),fill='#f8b66c',width=1)
            subtitle = ('{} | road {} | z={:.3f} m'.format(item['level'].upper(),
                        item['ego_pose']['road_id'],item['ego_pose']['z']))
            draw.text((left,top+345),subtitle,font=font(15),fill='#b6c3d7')
    top = header+2*row_height
    draw.text((24,top),'Real supplied maps / native OpenDRIVE surfaces. Static geometry only; no simulator or RGB capture.',
              font=font(16),fill='white')
    draw.text((24,top+27),'LRS anchor=100 px; HBLR anchor=99.5 px and integer ego pixels; SDV keeps subpixel pose.',
              font=font(15),fill='#b6c3d7')
    draw.text((24,top+51),'Legacy white=lane map; SDV includes sidewalk/junction/boundary. Broken paint is invalid, not invented.',
              font=font(15),fill='#b6c3d7')
    gradient = cv2.cvtColor(cv2.applyColorMap(np.arange(256,dtype=np.uint8)[None,:],cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)
    mosaic.paste(Image.fromarray(gradient).resize((420,12)),(24,top+83))
    for xpos, label in ((24,'0 m'), (156,'4 m'), (295,'8 m'), (414,'12 m')):
        draw.text((xpos,top+98),label,font=font(13),fill='#b6c3d7')
    draw.text((465,top+80),'Navy = unknown; no local topology path = invalid surface',
              font=font(14),fill='#b6c3d7')
    mosaic.save(str(output/'bev-town04-overpass-reference-comparison.png'))
    metadata = {'scope':'Same actual Town04 XY, lower/upper road elevation, yaw0; no live actors/sensors',
        'reference_commits': {'lrs':'fe9cbc2f8f38fa5f7808e6d161e645160958c6a2',
                             'hblr_linked_agileq':'f3eba609de1af9d07a1c3e37283afdc51e7fcf22'},
        'source_sha256': {'lrs':lrs_source_sha,'hblr_linked':HBLR_SOURCE_SHA256},
        'cache_signature': town['cache_signature'],
        'coordinate_convention':'CARLA metres; +X/yaw0 forward-up; +Y right; yaw degrees; world offset metres',
        'limitations': ['Legacy rasters and new native sampled geometry have different label definitions',
            'LRS/HBLR integer ego pixels and HBLR half-pixel anchor differ from new subpixel SDV',
            'No cross-method IoU or accuracy claim; original PNG height calibration is unverified',
            'Lane validity excludes unsupported broken/compound paint geometry',
            'Opposite carriageways without a verified local topology path are omitted/invalid',
            'No live CARLA server, recorded actors, camera synchronization or planning targets tested'],
        'poses':results}
    (output/'bev-town04-overpass-reference-comparison.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Saved actual Town04 3-way comparison; upper/lower ego anchor ground error <0.1 m')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lrs-root',required=True)
    parser.add_argument('--hblr-root',required=True,help='Reviewed AGILEQ-Training root')
    parser.add_argument('--cache-root',required=True)
    default = Path(__file__).resolve().parents[1]/'docs'/'validation'
    parser.add_argument('--validation-report',default=str(default/'bev-offline-carla-validation.json'))
    parser.add_argument('--output-dir',default=str(default))
    compare(parser.parse_args())


if __name__ == '__main__':
    main()
