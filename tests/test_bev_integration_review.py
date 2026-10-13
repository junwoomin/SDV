"""Controlled counterexamples for height/topology assignment (not CARLA runs)."""
import numpy as np
import pytest

pytest.importorskip('cv2')
from sdv.bev.cache import GlobalRasterCache
from sdv.bev.generator import HeightAwareBEVGenerator


def road(sid, height, road_id, slope=0):
    return {'id': sid, 'road_id': road_id, 'section_id': 0, 'lane_id': -1,
            'polygon_xyz': [[-20, -5, height-20*slope], [20, -5, height+20*slope],
                            [20, 5, height+20*slope], [-20, 5, height-20*slope]],
            'center_xyz': [0, 0, height], 'neighbors': [], 'kind': 'driving', 'is_junction': False}


def generator(tmp_path, surfaces):
    cache = GlobalRasterCache.build(tmp_path, 'ControlledTest', '<OpenDRIVE />', surfaces)
    return HeightAwareBEVGenerator(cache, {'width': 200, 'height': 200})


def actor(actor_id, x, ground, road_id, length=4):
    return {'actor_id': actor_id, 'object_class': 'vehicle', 'x': x, 'y': 0,
            'z': ground+0.3, 'yaw': 0, 'ground_z_m': ground,
            'ground_source': 'CARLA exact waypoint candidate; requires raster elevation verification',
            'road_id': road_id, 'section_id': 0, 'lane_id': -1,
            'bbox_min_z_m': ground+0.05, 'bbox_max_z_m': ground+2.05,
            'object_height_m': 2.0,
            'footprint_world_xyz': [[x-length/2, -1, ground], [x+length/2, -1, ground],
                                    [x+length/2, 1, ground], [x-length/2, 1, ground]]}


def test_long_pitched_truck_ground_is_surface_at_center_not_lowest_corner(tmp_path):
    render = generator(tmp_path, [road('slope', 4, 1, slope=.3)])
    truck = actor(1, 0, 4, 1, length=20)
    # 20m footprint along a30% incline: bottom corners differ by6m in Z.
    # Its lowest corner is3m below center ground, despite valid actual map
    # ground at the actor center; this does not mean the truck is below road.
    truck['bbox_min_z_m'] = 1.05
    truck['bbox_max_z_m'] = 9.05
    truck['object_height_m'] = 8.0
    truck['bbox_bottom_center_world_xyz'] = [0, 0, 4.05]
    assignment = render._ground(truck)
    assert assignment is not None
    assert assignment['layer_id'] == 'slope'
    assert assignment['ground_z_m'] == pytest.approx(4)


def test_disconnected_close_height_road_cannot_leak_actor_to_ego_surface(tmp_path):
    render = generator(tmp_path, [road('lower', 0, 1), road('separate', .6, 2)])
    ego = actor(1, -4, 0, 1)
    upper_actor = actor(2, 2, .6, 2)
    result = render.generate(ego, [ego, upper_actor], frame_id=10, timestamp=1.0)
    assert result['arrays']['validity'].any()
    assert not np.any(result['arrays']['actor_id'] == 2)
    record = next(row for row in result['metadata']['actors'] if row['actor_id'] == 2)
    assert not record['included']


def test_object_height_export_toggle_does_not_change_instance_occlusion(tmp_path):
    cache = GlobalRasterCache.build(tmp_path, 'ControlledTest', '<OpenDRIVE />', [road('flat', 0, 1)])
    low, tall = actor(2, 2, 0, 1), actor(3, 2, 0, 1)
    tall['bbox_max_z_m'] = 3.05
    tall['object_height_m'] = 3.0
    ego = actor(1, -6, 0, 1)
    outputs = []
    for enabled in (True, False):
        render = HeightAwareBEVGenerator(cache, {'width':200, 'height':200, 'object_height': enabled})
        outputs.append(render.generate(ego, [ego, tall, low])['arrays'])
    np.testing.assert_array_equal(outputs[0]['actor_id'], outputs[1]['actor_id'])
    assert (outputs[1]['actor_id'] == 3).any()


def test_long_vehicle_on_slope_keeps_front_and_rear_occupancy(tmp_path):
    render = generator(tmp_path, [road('slope', 4, 1, slope=.2)])
    ego = actor(1, -10, 2, 1)
    truck = actor(2, 0, 4, 1, length=16)
    truck['bbox_min_z_m'] = 2.45
    truck['bbox_max_z_m'] = 7.65
    truck['object_height_m'] = 5.2
    truck['bbox_bottom_center_world_xyz'] = [0, 0, 4.05]
    result = render.generate(ego, [ego, truck])
    from sdv.bev.geometry import world_to_local
    for x in (-7, 7):
        u, v = np.rint(world_to_local([x, 0], ego, render.geometry)).astype(int)
        assert result['arrays']['vehicle'][v, u] == 1
