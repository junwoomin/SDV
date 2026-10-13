#!/usr/bin/env python3
"""Compare the SDV affine with the actual supplied LRS function/map files.

Run from the project root:
  python scripts/validate_bev_references.py --lrs-root /path/to/extracted/LRS

No CARLA server is required. This checks crop equivalence, not correctness of
the original map labels or real-world coordinate calibration. The reference
source is executed only after its exact known Git blob identity is verified;
its SHA-256 is also recorded in the output report.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sdv.bev.geometry import BEVGeometry, warp_global


SOURCE_COMMIT = 'fe9cbc2f8f38fa5f7808e6d161e645160958c6a2'
# Git blob 5d2cf129ef940db6670710246dfe3bd4bb3a45f6, also byte-identical in
# both user-provided LRS_V1 archives. Calculate the byte hash from the pinned
# Git blob identity, so no line-ending-normalized substitution is accepted.
SOURCE_GIT_BLOB = '5d2cf129ef940db6670710246dfe3bd4bb3a45f6'


def reference_crop(source):
    raw = source.read_bytes()
    digest = hashlib.sha1(b'blob ' + str(len(raw)).encode('ascii') + b'\0' + raw).hexdigest()
    if digest != SOURCE_GIT_BLOB:
        raise ValueError('LRS source differs from the reviewed commit; review it before comparison')
    function = next(node for node in ast.parse(raw.decode('utf-8')).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'make_local_bev_ego_aligned')
    module = ast.parse('')
    module.body = [function]
    namespace = {'np': np, 'cv2': cv2, 'math': math, 'Tuple': Tuple}
    exec(compile(module, str(source), 'exec'), namespace)
    return namespace[function.name], hashlib.sha256(raw).hexdigest()


def compare(lrs_root, output_dir, runs=100):
    lrs_root, output_dir = Path(lrs_root), Path(output_dir)
    reference, source_sha256 = reference_crop(lrs_root / 'sim' / 'Global_Dynamic_MapManager.py')
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry = BEVGeometry()
    comparisons, timings, images, sources = [], [], [], []
    for town in ('Town01', 'Town04', 'Town05'):
        directory = lrs_root / 'sim_using_data' / 'data5' / town
        offset = np.load(str(directory / 'world_offset.npy'), allow_pickle=False)
        road = cv2.imread(str(directory / 'das_full.png'), cv2.IMREAD_GRAYSCALE)
        if road is None:
            raise FileNotFoundError('Actual LRS map is required: ' + str(directory))
        ys, xs = np.where(road > 100)
        if not len(xs):
            raise ValueError('Actual LRS drivable map contains no road pixels: ' + town)
        sample = len(xs) // 2
        cx, cy = float(xs[sample]), float(ys[sample])
        position = offset + np.array([cx, cy]) * 0.2
        layers = ['das_full.png', 'lane_full.png']
        if town != 'Town01':
            layers += ['das_full_high.png', 'lane_full_high.png']
        for filename in layers:
            path = directory / filename
            data = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if data is None or data.shape != road.shape:
                raise ValueError('Missing or unregistered actual LRS layer: ' + str(path))
            sources.append({'town': town, 'layer': filename,
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'shape': list(data.shape), 'world_offset': offset.tolist()})
            for yaw in (0, 90, 180, -90, 37.125, 123.5):
                pose = (float(position[0]), float(position[1]), 0, yaw)
                old = reference(data, (cx, cy), math.radians(yaw))
                new = warp_global(data, pose, geometry, offset, 0.2)
                difference = int(np.count_nonzero(old != new))
                comparisons.append({'town': town, 'layer': filename,
                    'ego_world_xy': position.tolist(), 'yaw_deg': yaw,
                    'mismatch_pixels': difference,
                    'local_sha256': hashlib.sha256(new.tobytes()).hexdigest()})
                if filename == 'das_full.png' and yaw in (0, 37.125):
                    tile = Image.new('RGB', (400, 234), '#111827')
                    tile.paste(Image.fromarray(old).convert('RGB'), (0, 34))
                    tile.paste(Image.fromarray(new).convert('RGB'), (200, 34))
                    drawing = ImageDraw.Draw(tile)
                    drawing.text((8, 4), town + ' yaw ' + str(yaw) + ' deg', fill='white')
                    drawing.text((8, 19), 'Original LRS', fill='white')
                    drawing.text((208, 19), 'New SDV', fill='white')
                    images.append(tile)
            for _ in range(5):
                warp_global(data, pose, geometry, offset, 0.2)
            samples = []
            for _ in range(runs):
                start = time.perf_counter_ns()
                warp_global(data, pose, geometry, offset, 0.2)
                samples.append((time.perf_counter_ns() - start) / 1e6)
            timings.append({'town': town, 'layer': filename, 'runs': runs,
                            'median_ms': statistics.median(samples),
                            'p95_ms': float(np.percentile(samples, 95))})
    mosaic = Image.new('RGB', (800, 234 * 3), '#111827')
    for index, tile in enumerate(images):
        mosaic.paste(tile, ((index % 2) * 400, (index // 2) * 234))
    mosaic.save(str(output_dir / 'bev-lrs-actual-comparison.png'))
    report = {'scope': 'actual supplied LRS static maps and pinned original crop function',
        'limitations': ['not a live CARLA test', 'not proof that legacy global labels are correct',
                       'timing excludes map loading, actors, storage and web delivery'],
        'source_commit': SOURCE_COMMIT, 'source_git_blob': SOURCE_GIT_BLOB,
        'source_sha256': source_sha256,
        'geometry': geometry.metadata(), 'opencv': cv2.__version__, 'numpy': np.__version__,
        'source_assets': sources, 'comparisons': comparisons,
        'all_comparisons_exact': all(row['mismatch_pixels'] == 0 for row in comparisons),
        'crop_only_cpu_timing': timings}
    (output_dir / 'bev-lrs-reference-validation.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if not report['all_comparisons_exact']:
        raise AssertionError('Actual LRS source crop comparison failed; see saved report')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lrs-root', required=True, help='Extracted LRS source root containing sim/ and sim_using_data/')
    parser.add_argument('--output-dir', default=str(Path(__file__).resolve().parents[1] / 'docs' / 'validation'))
    parser.add_argument('--runs', type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.runs <= 10000:
        parser.error('--runs must be between 1 and 10000')
    report = compare(args.lrs_root, args.output_dir, args.runs)
    print('{} actual-map crop comparisons passed with zero pixel differences'.format(len(report['comparisons'])))


if __name__ == '__main__':
    main()
