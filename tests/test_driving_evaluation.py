"""Geometric metric tests; these are not CARLA driving scenario results."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from sdv.evaluation.driving_metrics import DrivingMetrics, RedLightCrossings, RouteProgress


def line(state='Red', z=0):
    return dict(id='light1:road1:lane1', center=[10, 0, z], forward=[1, 0], half_width_m=1.75, state=state)


def test_red_line_requires_actual_front_crossing():
    counter = RedLightCrossings()
    assert counter.update([8, 0, 0], [line()], 1, .05) == []
    assert counter.update([9, 0, 0], [line()], 2, .1) == []
    assert counter.count == 0
    events = counter.update([11, 0, 0], [line()], 3, .15)
    assert len(events) == 1
    assert events[0]['crossing_world'] == [10, 0, 0]
    assert counter.count == 1
    counter.update([12, 0, 0], [line()], 4, .2)
    assert counter.count == 1


@pytest.mark.parametrize('start,end,signal', [
    ([8, 4, 0], [12, 4, 0], line()),   # neighbouring lane
    ([8, 0, 6], [12, 0, 6], line()),   # grade-separated road
    ([12, 0, 0], [8, 0, 0], line()),   # reverse crossing
    ([8, 0, 0], [12, 0, 0], line('Green')),
    ([8, 0, 0], [12, 0, 0], line('Yellow')),
])
def test_non_red_or_non_applicable_crossing_is_not_violation(start, end, signal):
    counter = RedLightCrossings()
    counter.update(start, [signal], 1, .05)
    assert counter.update(end, [signal], 2, .1) == []


def test_signal_colour_uses_start_of_transition():
    counter = RedLightCrossings()
    counter.update([9, 0, 0], [line('Green')], 1, .05)
    counter.update([11, 0, 0], [line('Red')], 2, .1)
    assert counter.count == 0


def test_route_progress_metric_length_not_sampling_density():
    route = RouteProgress([[0, 0, 0], [1, 0, 0], [10, 0, 0], [20, 0, 0]])
    assert route.update([10, 0, 0])['completion_pct'] == pytest.approx(50)
    assert route.update([8, 0, 0])['completion_pct'] == pytest.approx(50)


def test_route_progress_does_not_jump_to_future_crossing():
    route = RouteProgress([[0, 0, 0], [20, 0, 0], [20, 20, 0], [0, 20, 0], [0, 0, 0], [-20, 0, 0]])
    row = route.update([0, 0, 0])
    assert row['completion_pct'] == 0
    assert row['segment'] == 0


def test_route_progress_excludes_stacked_road_and_offroute_shortcut():
    route = RouteProgress([[0, 0, 0], [30, 0, 0]])
    assert route.update([20, 0, 7])['height_match'] is False
    row = route.update([20, 9, 0])
    assert row['deviation_m'] == pytest.approx(9)
    assert row['progress_m'] == 0


def test_metrics_incident_grouping_and_latency_units():
    metrics = DrivingMetrics([[0, 0, 0], [30, 0, 0]], .05)
    metrics.collisions.extend([{'frame': 1, 'other_actor_id': 2}, {'frame': 2, 'other_actor_id': 2},
                               {'frame': 3, 'other_actor_id': 3}, {'frame': 30, 'other_actor_id': 2}])
    metrics.sample(frame=1, timestamp=.05, position=[0, 0, 0], front_position=[2, 0, 0],
                   speed_mps=10, offroad=False, stoplines=[], inference_ms=4, control_latency_ms=7)
    result = metrics.report()
    assert result['collision_count'] == 3
    assert result['collision_callback_count'] == 4
    assert result['average_speed_kmh'] == 36
    assert result['policy_inference_mean_ms'] == 4
    assert result['control_latency_mean_ms'] == 7


def test_preflight_missing_environment_is_explicit(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('roach_validation_cli', Path(__file__).parents[1] / 'scripts/validate_roach_carla.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.importlib.util, 'find_spec', lambda name: None)
    def unavailable(*args, **kwargs):
        raise OSError('no test server')
    monkeypatch.setattr(module.socket, 'create_connection', unavailable)
    args = SimpleNamespace(checkpoint=str(tmp_path / 'missing.pth'), agents=['roach'], host='localhost', port=2000, timeout=.1)
    result = module.preflight(args, [{'name': 'straight', 'start_spawn_index': None, 'destination_spawn_index': None}])
    assert result['status'] == 'blocked'
    assert result['live_execution'] is False
    assert len(result['blockers']) >= 4
    assert 'metrics' not in result
