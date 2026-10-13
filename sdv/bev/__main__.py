"""Inspect supplied global maps; retain explicit legacy cache-build tooling."""
import argparse
import json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description='SDV supplied global map crop and rotation')
    sub=parser.add_subparsers(dest='action')
    maps=sub.add_parser('maps',help='Inspect supplied whole-town PNG maps and native PPM; does not generate maps')
    maps.add_argument('--root',default=None,help='Global map asset root; defaults to bundled assets/bev_maps')
    build=sub.add_parser('build',help='Explicit older OpenDRIVE cache tooling; not used by global_raster collection')
    build.add_argument('--host',default='127.0.0.1');build.add_argument('--port',type=int,default=2000)
    build.add_argument('--town',help='Optional required current map name; never changes a running world')
    build.add_argument('--xodr',help='Offline OpenDRIVE file; CARLA Python API required, no server needed')
    build.add_argument('--cache-dir',default='.runtime/bev');build.add_argument('--mpp',type=float,default=.2)
    inspect=sub.add_parser('inspect',help='Validate and describe an existing map cache')
    inspect.add_argument('path')
    args=parser.parse_args()
    if args.action=='maps':
        from .global_map import catalog
        print(json.dumps(catalog(args.root),ensure_ascii=False,indent=2))
        return
    if args.action=='inspect':
        from .cache import GlobalRasterCache
        cache=GlobalRasterCache.load(args.path)
    elif args.action=='build':
        from sdv.collection.bev_adapter import build_cache
        from .schema import normalize_bev_settings
        import carla
        if args.xodr:
            name=args.town or Path(args.xodr).stem
            road_map=carla.Map(name,Path(args.xodr).read_text(encoding='utf-8'))
            class OfflineWorld:
                def get_map(self): return road_map
            world=OfflineWorld()
        else:
            client=carla.Client(args.host,args.port);client.set_timeout(20.)
            world=client.get_world()
            if args.town and world.get_map().name.rsplit('/',1)[-1]!=args.town:
                parser.error('Connected map does not match --town; select the intended map before building')
        cache=build_cache(world,Path(args.cache_dir),normalize_bev_settings({'mode':'height_aware','meters_per_pixel':args.mpp}))
    else:
        parser.print_help();return
    print(json.dumps({'cache':str(cache.path),'signature':cache.metadata['signature'],
        'town':cache.metadata.get('town'),'patches':len(cache.patches),
        'meters_per_pixel':cache.meters_per_pixel,'height_unit':'m'},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
