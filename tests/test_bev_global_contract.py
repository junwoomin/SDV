"""Global PNG labels need coverage without fabricated metric elevation."""
import copy
import numpy as np
import pytest

from sdv.bev.io import validate_arrays, save_frame, load_frame
from sdv.bev.schema import BINARY_CHANNELS, HEIGHT_CHANNELS, normalize_bev_settings
from studio.config import Config, atomic_json


def frame():
    shape=(64,64)
    arrays={name:np.zeros(shape,dtype=np.uint8) for name in BINARY_CHANNELS}
    arrays.update({name:np.full(shape,np.nan,dtype=np.float32) for name in HEIGHT_CHANNELS})
    arrays.update(semantic=np.zeros(shape,dtype=np.uint8),actor_id=np.zeros(shape,dtype=np.int32),
                  surface_id=np.zeros(shape,dtype=np.int32),elevation_validity=np.zeros(shape,dtype=np.uint8))
    arrays['validity'][:32]=1
    return arrays


def test_png_coverage_roundtrip_does_not_fabricate_elevation(tmp_path):
    arrays=frame(); meta={'schema_version':4}
    save_frame(tmp_path/'frame.npz',arrays,metadata=meta)
    restored=load_frame(tmp_path/'frame.npz',metadata=meta)
    assert restored['validity'].sum()==2048
    assert not restored['elevation_validity'].any()
    assert np.isnan(restored['elevation_m']).all()


@pytest.mark.parametrize('fault', ['missing','mask_dtype','mask_shape','false_height','outside'])
def test_height_validity_rejects_inconsistent_labels(fault):
    arrays=frame()
    if fault=='missing': arrays.pop('elevation_validity')
    elif fault=='mask_dtype': arrays['elevation_validity']=arrays['elevation_validity'].astype(np.float32)
    elif fault=='mask_shape': arrays['elevation_validity']=arrays['elevation_validity'][:2]
    elif fault=='false_height': arrays['elevation_m'][0,0]=0
    else:
        arrays['elevation_m'][63,0]=2
        arrays['elevation_validity'][63,0]=1
    with pytest.raises(ValueError): validate_arrays(arrays,metadata={'schema_version':4})


def test_original_metric_surface_contract_still_requires_finite_height():
    arrays=frame();arrays.pop('elevation_validity')
    with pytest.raises(ValueError,match='finite elevation'): validate_arrays(arrays)


def test_settings_separate_native_source_from_output_sampling():
    config=normalize_bev_settings({'global_ppm':10,'meters_per_pixel':.2,'width':200,'height':200})
    assert config['mode']=='global_raster'
    assert config['global_ppm']==10 and config['width']*config['meters_per_pixel']==40
    for invalid in (7,5.0,True):
        with pytest.raises(ValueError):normalize_bev_settings({'global_ppm':invalid})


def test_old_web_default_migrates_without_changing_saved_job_snapshots(tmp_path):
    cfg=Config(tmp_path)
    prior={'mode':'height_aware','layer_policy':'topology_surface','width':300,'height':300}
    stored={'bev_generator':copy.deepcopy(prior)}
    atomic_json(cfg.state/'settings.json',stored)
    assert cfg.settings()['bev_generator']['mode']=='global_raster'
    assert cfg.settings()['bev_generator']['width']==300
    assert cfg.read('settings',{})==stored
    # Explicit historical generator settings still support the original path.
    assert normalize_bev_settings(prior)['mode']=='height_aware'
