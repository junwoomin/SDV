"""Independent geometric evaluation; no policy output is treated as ground truth.

All positions are CARLA world coordinates (metres). Route progress is projected
onto a bounded, height-aware forward window so intersections and overpasses do
not jump to a distant part of the route. Red-light violations require an actual
forward crossing of a native stop line, not just a red lamp near the ego.
"""
import math
import statistics


def _xyz(point):
    values = tuple(float(v) for v in point[:3])
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError('Expected finite world x, y, z')
    return values


def _distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def percentile(values, quantile):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lo, hi = int(math.floor(index)), int(math.ceil(index))
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


class RouteProgress:
    """Metric arc length along the assigned route, not nearest-waypoint count."""
    def __init__(self, points, forward_window_m=40.0, height_tolerance_m=2.5):
        self.points = [_xyz(point) for point in points]
        if len(self.points) < 2:
            raise ValueError('Route requires at least two world points')
        self.cumulative = [0.0]
        for a, b in zip(self.points, self.points[1:]):
            self.cumulative.append(self.cumulative[-1] + _distance(a, b))
        self.length_m = self.cumulative[-1]
        if self.length_m <= 0:
            raise ValueError('Route length must be positive')
        self.forward_window_m = float(forward_window_m)
        self.height_tolerance_m = float(height_tolerance_m)
        self.progress_m = 0.0
        self.segment = 0

    def update(self, position):
        p = _xyz(position)
        candidates = []
        for index in range(max(0, self.segment - 10), len(self.points) - 1):
            if self.cumulative[index] > self.progress_m + self.forward_window_m:
                break
            a, b = self.points[index], self.points[index + 1]
            delta = tuple(y - x for x, y in zip(a, b))
            length2 = sum(v * v for v in delta)
            if length2 == 0:
                continue
            fraction = max(0.0, min(1.0, sum((v - x) * d for v, x, d in zip(p, a, delta)) / length2))
            projection = tuple(x + fraction * d for x, d in zip(a, delta))
            if abs(projection[2] - p[2]) > self.height_tolerance_m:
                continue
            distance = _distance(p, projection)
            arc = self.cumulative[index] + math.sqrt(length2) * fraction
            candidates.append((distance, abs(arc - self.progress_m), arc, index))
        if not candidates:
            return {'progress_m': self.progress_m, 'completion_pct': 100 * self.progress_m / self.length_m,
                    'deviation_m': None, 'height_match': False, 'segment': self.segment}
        distance, _, arc, index = min(candidates)
        # Far from the route cannot increase completion by cutting a junction.
        if distance <= 5.0:
            self.progress_m = max(self.progress_m, arc)
            self.segment = max(self.segment, index)
        return {'progress_m': self.progress_m, 'completion_pct': min(100.0, 100 * self.progress_m / self.length_m),
                'deviation_m': distance, 'height_match': True, 'segment': index}


class RedLightCrossings:
    """Stop lines: id, center[3], forward[2], half_width_m and current state.

    The crossing segment uses the front bumper, and must travel from the
    approach side to the junction side. Lamp colour is sampled for the *start*
    frame of the movement interval; a state change is not retroactive.
    """
    def __init__(self):
        self.previous = None
        self.previous_lines = {}
        self.count = 0
        self.events = []

    def update(self, front_position, lines, frame, timestamp):
        current = _xyz(front_position)
        events = []
        if self.previous is not None:
            for line_id, line in self.previous_lines.items():
                center = _xyz(line['center'])
                nx, ny = map(float, line['forward'][:2])
                norm = math.hypot(nx, ny)
                if norm <= 0:
                    continue
                nx, ny = nx / norm, ny / norm
                before = (self.previous[0] - center[0]) * nx + (self.previous[1] - center[1]) * ny
                after = (current[0] - center[0]) * nx + (current[1] - center[1]) * ny
                if not (before < 0 <= after):
                    continue
                alpha = -before / (after - before)
                crossing = tuple(a + alpha * (b - a) for a, b in zip(self.previous, current))
                lateral = abs((crossing[0] - center[0]) * -ny + (crossing[1] - center[1]) * nx)
                if lateral > float(line['half_width_m']) or abs(crossing[2] - center[2]) > 2.0:
                    continue
                if str(line['state']).split('.')[-1].lower() != 'red':
                    continue
                event = {'frame': int(frame), 'timestamp': float(timestamp), 'stopline_id': line_id,
                         'crossing_world': list(crossing), 'state_before_crossing': 'Red'}
                events.append(event)
                self.events.append(event)
                self.count += 1
        self.previous = current
        self.previous_lines = {str(line['id']): dict(line) for line in lines}
        return events


class DrivingMetrics:
    def __init__(self, route_points, fixed_delta_seconds=.05):
        self.route = RouteProgress(route_points)
        self.fixed_delta_seconds = float(fixed_delta_seconds)
        self.red_lights = RedLightCrossings()
        self.samples = []
        self.collisions = []
        self.lane_invasions = []

    def sample(self, *, frame, timestamp, position, front_position, speed_mps,
               offroad, stoplines, inference_ms=None, control_latency_ms=None):
        progress = self.route.update(position)
        row = {'frame': int(frame), 'timestamp': float(timestamp), 'position': list(_xyz(position)),
               'speed_mps': float(speed_mps), 'offroad_center': bool(offroad), **progress,
               'inference_ms': inference_ms, 'control_latency_ms': control_latency_ms,
               'red_crossings': self.red_lights.update(front_position, stoplines, frame, timestamp)}
        self.samples.append(row)
        return row

    def report(self):
        speeds = [row['speed_mps'] for row in self.samples]
        deviations = [row['deviation_m'] for row in self.samples if row['deviation_m'] is not None]
        inference = [float(row['inference_ms']) for row in self.samples if row['inference_ms'] is not None]
        latency = [float(row['control_latency_ms']) for row in self.samples if row['control_latency_ms'] is not None]
        # Repeated contact callbacks with one actor within one second = one
        # incident. Raw callback data remains available for other definitions.
        incidents, last_contact = 0, {}
        for event in sorted(self.collisions, key=lambda value: value['frame']):
            actor = event['other_actor_id']
            seconds = event['frame'] * self.fixed_delta_seconds
            if actor not in last_contact or seconds - last_contact[actor] > 1.0:
                incidents += 1
            last_contact[actor] = seconds
        return {'frames': len(self.samples), 'route_length_m': self.route.length_m,
                'route_completion_pct': 100 * self.route.progress_m / self.route.length_m,
                'collision_count': incidents, 'collision_callback_count': len(self.collisions),
                'traffic_light_violation_count': self.red_lights.count,
                'offroad_center_frames': sum(row['offroad_center'] for row in self.samples),
                'offroad_center_seconds': sum(row['offroad_center'] for row in self.samples) * self.fixed_delta_seconds,
                'lane_invasion_events': len(self.lane_invasions),
                'route_height_mismatch_frames': sum(not row['height_match'] for row in self.samples),
                'average_speed_kmh': statistics.mean(speeds) * 3.6 if speeds else None,
                'route_deviation_mean_m': statistics.mean(deviations) if deviations else None,
                'route_deviation_p95_m': percentile(deviations, .95),
                'route_deviation_max_m': max(deviations) if deviations else None,
                'policy_inference_mean_ms': statistics.mean(inference) if inference else None,
                'policy_inference_p95_ms': percentile(inference, .95),
                'control_latency_mean_ms': statistics.mean(latency) if latency else None,
                'control_latency_p95_ms': percentile(latency, .95),
                'metric_definitions': {
                    'route_completion': 'monotonic metric arc length; 40 m forward search; 2.5 m height gate; advancement only within 5 m',
                    'collision_count': 'contacts with the same actor grouped until a gap greater than 1 second',
                    'red_light_violation': 'front bumper crosses a native directed stopline forward while start-frame state is Red',
                    'offroad': 'ego center outside a native Driving waypoint; not full bounding-box road departure',
                    'lane_invasion': 'CARLA lane-marking crossing callbacks, including permitted lane changes; not automatically violations',
                    'control_latency': 'wall time for runtime.run_step including observation, inference, safety and apply_control; excludes next physics tick'}}
