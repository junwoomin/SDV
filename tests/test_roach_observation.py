"""Numeric source parity and CARLA-boundary checks; no simulator claims."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
cv2 = pytest.importorskip("cv2", reason="Roach observation parity requires research OpenCV")
import numpy as np

from sdv.driving.observation import (
    RoachObservationAdapter, ObservationError, vehicle_state, contract,
)

spec = importlib.util.spec_from_file_location('lrs_reference', Path(__file__).parent/'fixtures/lrs_observation_reference.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


class Map:
    def __init__(self, ppm=5, layered=False):
        self.ppm = ppm
        self.meters_per_pixel = 1/ppm
        self.world_offset = np.array([0., 0.])
        self.shape = (100*ppm, 100*ppm)
        self.layered = layered
        self.metadata = {'height_layers': ['low', 'high'] if layered else ['low'], 'source_signature': 'test-fixture'}
        self.road = np.zeros(self.shape, dtype=np.uint8)
        self.lane = np.zeros(self.shape, dtype=np.uint8)
        self.road[10*ppm:45*ppm, 8*ppm:70*ppm] = 1
        self.lane[15*ppm:16*ppm, 8*ppm:70*ppm] = 1

    def select_layer(self, ego, policy='hblr', height_aware=True, channels=None):
        layer = policy if policy in ('low', 'high') else ('high' if self.layered and ego.get('ground_z_m', 0)>5 else 'low')
        return {'layer_id': layer, 'arrays': {'drivable': self.road, 'lane': self.lane}, 'metadata': self.metadata}

    def layer_for_points(self, points, **kwargs):
        return {'high': np.asarray(points)[:, 2]>5 if self.layered else np.zeros(len(points), dtype=bool)}


def ego(frame=1, x=25.12, y=25.18, yaw=0, z=0):
    return dict(id=1, actor_id=1, is_ego=True, object_class='vehicle', x=x, y=y, z=z, yaw=yaw,
        ground_z_m=z, ground_verified=True, ground_lane_type='Driving', ground_source='fixture_native_ground',
        velocity_m_s={'x': 7, 'y': 2, 'z': 1}, rotation_deg={'pitch': 0, 'yaw': yaw, 'roll': 0}, frame_id=frame)


CONTROL = dict(throttle=.3, steer=-.2, brake=0, gear=4, reverse=False, manual_gear_shift=False)


def actor(frame=1, x=29, y=26, yaw=31, z=0, kind='vehicle', identity=2):
    value = ego(frame, x, y, yaw, z)
    value.update(id=identity, actor_id=identity, is_ego=False, object_class=kind)
    return value


def observe(adapter, frame=1, own=None, actors=None, traffic=None, route=None):
    own = own or ego(frame)
    return adapter.observe(own, actors or [own], traffic or [], route or [[20,25,0], [30,25,0], [45,25,0]],
        CONTROL, frame, frame/30)


@pytest.mark.parametrize('yaw', [0, 90, -90, 37.125])
def test_static_crop_equals_isolated_actual_lrs_function(yaw):
    source = Map()
    own = ego(yaw=yaw)
    result = observe(RoachObservationAdapter(source), own=own)
    center = reference.world_to_pixel(own['x'], own['y'], 0, 0, 5)
    for index, image in ((0, source.road), (2, source.lane)):
        expected = reference.make_local_bev_ego_aligned(image*255, center, np.deg2rad(yaw),
            out_size=192, out_mpp=.2, pixels_per_meter=5, move_px=40)
        np.testing.assert_array_equal(result['birdview'][index], expected)
    assert result['birdview'].dtype == np.uint8
    assert result['metadata']['ego_anchor_pixels'] == [96,136]


@pytest.mark.parametrize('yaw', [0, 90, -90, 37.125])
def test_actor_and_ped_global_grid_raster_match_lrs(yaw):
    source = Map()
    own, other, ped = ego(yaw=yaw), actor(), actor(x=30,y=27,kind='pedestrian',identity=3)
    result = observe(RoachObservationAdapter(source), own=own, actors=[own, other, ped])
    global_vehicle = np.zeros(source.shape, dtype=np.uint8)
    center = reference.world_to_pixel(other['x'],other['y'],0,0,5)
    pts = reference._obb_pts(*center, 12, 5, np.deg2rad(other['yaw']))
    cv2.fillConvexPoly(global_vehicle, pts, 255)
    global_ped = np.zeros_like(global_vehicle)
    px,py = reference.world_to_pixel(ped['x'],ped['y'],0,0,5)
    global_ped[py-4:py+5,px-4:px+5]=255
    ego_center = reference.world_to_pixel(own['x'],own['y'],0,0,5)
    for channel, raster in ((3,global_vehicle),(7,global_ped)):
        expected = reference.make_local_bev_ego_aligned(raster,ego_center,np.deg2rad(yaw),192,.2,5,move_px=40)
        np.testing.assert_array_equal(result['birdview'][channel],expected)


@pytest.mark.parametrize('state,value', [('Green',80),('Yellow',170),('Red',255)])
def test_traffic_values_stopline_aa_and_cold_history_match_lrs(state,value):
    source=Map()
    signal={'id':101,'state':state,'segments':[[[30,23,0],[30,27,0]]], 'ground_verified':True,'ground_lane_type':'Driving'}
    result=observe(RoachObservationAdapter(source),traffic=[signal])
    raster=np.zeros(source.shape,dtype=np.uint8)
    cv2.line(raster,(150,115),(150,135),value,6,cv2.LINE_AA)
    expected=reference.make_local_bev_ego_aligned(raster,(125,125),0,192,.2,5,move_px=40)
    for channel in range(11,15):
        np.testing.assert_array_equal(result['birdview'][channel],expected)
    assert result['birdview'][11].max()==value
    assert np.unique(result['birdview'][11]).size>2  # LRS AA edges retained, no /255 or thresholding.


def test_history_order_is_latest_first_and_reprojects_old_world_positions():
    adapter=RoachObservationAdapter(Map())
    result=None
    for frame in range(1,17):
        own=ego(frame,x=25+frame*.2,yaw=0)
        other=actor(frame,x=30+frame*.2,y=25,yaw=0)
        result=observe(adapter,frame,own,[own,other])
    assert result['metadata']['history_frames']==[16,12,7,2]
    assert result['metadata']['history_ready'] is True
    centers=[]
    for channel in range(3,7):
        rows,cols=np.nonzero(result['birdview'][channel])
        centers.append(rows.mean())
    np.testing.assert_allclose(centers,[111,115,120,125])
    assert contract()['history_seconds']==[0,4/30,9/30,14/30]


def test_cold_actor_history_is_zero_not_repeated():
    own,other=ego(),actor()
    result=observe(RoachObservationAdapter(Map()),actors=[own,other])
    assert result['birdview'][3].max()==255
    assert result['birdview'][4:7].sum()==0
    assert result['metadata']['history_frames']==[1,None,None,None]


@pytest.mark.parametrize('yaw',[0,37.125,90])
def test_route_rounding_width_and_clipping_match_actual_lrs(yaw):
    source=Map()
    own=ego(yaw=yaw)
    route=[[float(x),25.,0.] for x in range(15,65)]
    result=observe(RoachObservationAdapter(source),own=own,route=route)
    source_env=SimpleNamespace(OUT_SIZE=192,PPM=5,OFFSET_X=0,OFFSET_Y=0,move_px=40)
    expected=reference._render_path_local_bev(source_env,route,0,125,125,np.deg2rad(yaw))
    np.testing.assert_array_equal(result['birdview'][1],expected)


def test_10ppm_map_preserves_5ppm_physical_extent_and_nearest_values():
    source5,source10=Map(5),Map(10)
    own=ego(x=25.0,y=25.0)
    result5=observe(RoachObservationAdapter(source5),own=own)
    result10=observe(RoachObservationAdapter(source10),own=own)
    np.testing.assert_array_equal(result5['birdview'],result10['birdview'])
    assert set(np.unique(result10['birdview'][0]))=={0,255}
    assert result10['metadata']['policy_ppm']==5
    assert result10['metadata']['source_map_ppm']==10


def test_state_rotation_matches_lrs_and_gear_uses_drive_category():
    own=ego(yaw=37)
    own['rotation_deg'].update(pitch=8,roll=-4)
    result=vehicle_state(own,CONTROL)
    rotation=SimpleNamespace(**own['rotation_deg'])
    expected=reference.carla_rot_to_mat(rotation).T.dot(np.array([7,2,1]))
    np.testing.assert_allclose(result[4:],expected[:2],rtol=1e-6)
    np.testing.assert_allclose(result[:4],[.3,-.2,0,3])
    assert vehicle_state(own,dict(CONTROL,reverse=True))[3]==1
    assert vehicle_state(own,dict(CONTROL,manual_gear_shift=True,gear=0))[3]==2
    assert vehicle_state(own,dict(CONTROL,hand_brake=True))[3]==3


def test_overpass_removes_opposite_deck_actor_and_signal():
    own=ego(z=0)
    other=actor(z=9)
    signal={'id':101,'state':'Red','segments':[[[30,23,9],[30,27,9]]],'ground_verified':True,'ground_lane_type':'Driving'}
    result=observe(RoachObservationAdapter(Map(layered=True)),actors=[own,other],traffic=[signal])
    assert result['birdview'][3:15].sum()==0
    assert result['metadata']['excluded_height_actor_ids']==[2]


def test_uncertain_nearby_actor_without_snapshot_box_contact_fails():
    own,other=ego(),actor()
    other['ground_verified']=False
    with pytest.raises(ObservationError,match='contact'):
        observe(RoachObservationAdapter(Map()),actors=[own,other])


def test_offroad_pedestrian_contact_retains_conservative_occupancy():
    own,other=ego(),actor(kind='pedestrian')
    other.update(ground_verified=False,ground_z_m=None,ground_contact_world_xyz=[other['x'],other['y'],.1])
    result=observe(RoachObservationAdapter(Map()),actors=[own,other])
    assert result['birdview'][7].max()==255
    assert result['metadata']['uncertain_ground_actor_ids']==[2]
    assert result['metadata']['excluded_height_actor_ids']==[]


def test_unverified_opposite_deck_contact_is_excluded_and_disclosed():
    own,other=ego(),actor(kind='pedestrian',z=10)
    other.update(ground_verified=False,ground_z_m=None,ground_contact_world_xyz=[other['x'],other['y'],10])
    result=observe(RoachObservationAdapter(Map(layered=True)),actors=[own,other])
    assert result['birdview'][7].sum()==0
    assert result['metadata']['uncertain_ground_actor_ids']==[2]
    assert result['metadata']['excluded_height_actor_ids']==[2]


def test_wrong_clock_duplicate_skipped_or_mixed_frame_fails():
    with pytest.raises(ObservationError,match='30 Hz'):
        RoachObservationAdapter(Map(),.05)
    adapter=RoachObservationAdapter(Map())
    observe(adapter)
    with pytest.raises(ObservationError,match='Nonconsecutive'):
        observe(adapter,3)
    other=actor(frame=9)
    with pytest.raises(ObservationError,match='record frame'):
        observe(RoachObservationAdapter(Map()),actors=[ego(),other])


def test_nan_velocity_and_empty_route_fail():
    own=ego();own['velocity_m_s']['x']=np.nan
    with pytest.raises(ObservationError,match='velocity'):
        observe(RoachObservationAdapter(Map()),own=own)
    with pytest.raises(ObservationError,match='route'):
        RoachObservationAdapter(Map()).observe(ego(),[ego()],[],[],CONTROL,1,1/30)
