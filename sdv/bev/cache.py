"""Sparse, versioned CARLA road-surface rasters; no simulator dependency.

Unlike the legacy HBLR-linked Town04/Town05 Z switch, a layer here is an
identified continuous road/lane surface patch.  Different elevations at the
same XY remain separate patches.  Geometry/topology must come from the map
adapter, never from normalized visualization PNGs.
"""
import collections
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile


FORMAT_VERSION = "sdv-surface-cache-v1"
GENERATOR_VERSION = "1.0.0"
MASK_NAMES = ("road", "drivable", "lane", "road_boundary", "sidewalk", "junction")
_SURFACE_KEYS = ("id", "road_id", "section_id", "lane_id", "polygon_xyz",
                 "lane_markings", "boundary_lines", "kind", "is_junction",
                 "neighbors", "center_xyz")


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _points(value, minimum):
    if not isinstance(value, (list, tuple)) or len(value) < minimum:
        raise ValueError("Surface geometry needs at least {} XYZ points".format(minimum))
    points = []
    for point in value:
        if len(point) != 3:
            raise ValueError("Surface points must contain X, Y and actual ground Z")
        row = [float(x) for x in point]
        if not all(math.isfinite(x) for x in row):
            raise ValueError("Surface coordinates must be finite")
        points.append(row)
    return points


def _normalize_surface(value):
    sid = str(value["id"])
    if not sid or len(sid) > 256:
        raise ValueError("Invalid surface ID")
    polygon = _points(value["polygon_xyz"], 3)
    if polygon[0] == polygon[-1]:
        polygon.pop()
    if len(polygon) < 3:
        raise ValueError("Degenerate surface polygon")
    kind = str(value.get("kind", "driving"))
    if kind not in ("driving", "sidewalk", "shoulder", "parking", "biking", "other"):
        raise ValueError("Unsupported surface kind: " + kind)
    lines = {}
    for key in ("lane_markings", "boundary_lines"):
        lines[key] = []
        for line in value.get(key, []):
            width = float(line.get("width_m", 0.15))
            if not math.isfinite(width) or not 0 < width <= 20:
                raise ValueError("Line width must be finite, positive meters")
            item = {"points_xyz": _points(line["points_xyz"], 2), "width_m": width}
            if key == "lane_markings":
                item.update(type=str(line.get("type", "Solid")),
                            paint_geometry=bool(line.get("paint_geometry", line.get("resolved", False))))
            lines[key].append(item)
    center = value.get("center_xyz")
    if center is None:
        center = [sum(p[i] for p in polygon) / len(polygon) for i in range(3)]
    center = _points([center], 1)[0]
    out = {"id": sid, "polygon_xyz": polygon, "kind": kind,
           "is_junction": bool(value.get("is_junction", False)),
           "neighbors": sorted(set(str(x) for x in value.get("neighbors", []))),
           "center_xyz": center}
    for key in ("road_id", "section_id", "lane_id"):
        raw = value.get(key)
        out[key] = None if raw is None else int(raw)
    out.update(lines)
    # Validate non-self-crossing geometry during build, not only at first use.
    _triangles(polygon)
    return out


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _barycentric(x, y, a, b, c):
    det = _cross(a, b, c)
    if abs(det) <= 1e-10:
        return None
    u = ((b[0] - x) * (c[1] - y) - (b[1] - y) * (c[0] - x)) / det
    v = ((c[0] - x) * (a[1] - y) - (c[1] - y) * (a[0] - x)) / det
    return u, v, 1.0 - u - v


def _triangles(polygon):
    """Ear clipping keeps sloped concave polygons distinct from their hull."""
    p = polygon
    area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(p, p[1:] + p[:1]))
    if abs(area) < 1e-8:
        raise ValueError("Surface polygon has zero area")
    indices = list(range(len(p)))
    if area < 0:
        indices.reverse()
    result = []
    while len(indices) > 3:
        for j, b in enumerate(indices):
            a, c = indices[j - 1], indices[(j + 1) % len(indices)]
            if _cross(p[a], p[b], p[c]) <= 1e-10:
                continue
            blocked = False
            for other in indices:
                if other in (a, b, c):
                    continue
                weights = _barycentric(p[other][0], p[other][1], p[a], p[b], p[c])
                if weights is not None and min(weights) >= -1e-10:
                    blocked = True
                    break
            if not blocked:
                result.append((a, b, c))
                indices.pop(j)
                break
        else:
            raise ValueError("Surface polygon is self-intersecting or degenerate")
    if abs(_cross(*(p[i] for i in indices))) < 1e-10:
        raise ValueError("Surface polygon has degenerate triangle")
    result.append(tuple(indices))
    return result


def _source_manifest(town, opendrive_text, surfaces, meters_per_pixel, build_options=None, build_report=None):
    if not isinstance(town, str) or not re.match(r"^[A-Za-z0-9_.-]+$", town) or town in (".", ".."):
        raise ValueError("Town must be a simple map name")
    mpp = float(meters_per_pixel)
    if not math.isfinite(mpp) or not 0.02 <= mpp <= 5.0:
        raise ValueError("Meters per pixel must be in [0.02, 5]")
    if not isinstance(opendrive_text, str) or not opendrive_text.strip():
        raise ValueError("Actual OpenDRIVE text is required for cache identity")
    source = sorted((_normalize_surface(s) for s in surfaces), key=lambda s: s["id"])
    ids = set(s["id"] for s in source)
    if not source or len(ids) != len(source):
        raise ValueError("Surface IDs must be unique and geometry cannot be empty")
    for s in source:
        missing = set(s["neighbors"]) - ids
        if missing:
            raise ValueError("Unknown surface neighbor(s): " + ", ".join(sorted(missing)))
    manifest = {"format": FORMAT_VERSION, "generator_version": GENERATOR_VERSION,
                "town": town, "opendrive_sha256": _sha(opendrive_text.encode("utf-8")),
                "meters_per_pixel": mpp, "build_options": build_options or {}, "surfaces": source}
    if build_report is not None:
        if not isinstance(build_report, dict):
            raise ValueError("Build report must be a JSON object")
        # Freeze diagnostics with the geometry, including omitted-edge reasons.
        manifest["build_report"] = json.loads(_json_bytes(build_report).decode("utf-8"))
    return manifest


def _raster(surface, origin, mpp):
    import numpy as np
    import cv2
    points = np.asarray(surface["polygon_xyz"], dtype=np.float64)
    start = np.floor((points[:, :2].min(axis=0) - origin) / mpp).astype(int) - 1
    stop = np.ceil((points[:, :2].max(axis=0) - origin) / mpp).astype(int) + 1
    width, height = (stop - start + 1).tolist()
    if width <= 0 or height <= 0 or width * height > 16000000:
        raise ValueError("A surface patch is too large; subdivide the road geometry")
    xs = origin[0] + (start[0] + np.arange(width)) * mpp
    ys = origin[1] + (start[1] + np.arange(height)) * mpp
    xx, yy = np.meshgrid(xs, ys)
    elevation = np.full((height, width), np.nan, dtype=np.float32)
    for triangle in _triangles(surface["polygon_xyz"]):
        a, b, c = points[list(triangle)]
        u, v, w = _barycentric(xx, yy, a, b, c)
        inside = (u >= -1e-7) & (v >= -1e-7) & (w >= -1e-7)
        elevation[inside] = (u * a[2] + v * b[2] + w * c[2])[inside]
    valid = np.isfinite(elevation).astype(np.uint8)
    arrays = {key: np.zeros_like(valid) for key in MASK_NAMES}
    arrays["road"] = valid.copy() if surface["kind"] in ("driving", "shoulder", "parking", "biking") else np.zeros_like(valid)
    arrays["drivable"] = valid.copy() if surface["kind"] == "driving" else np.zeros_like(valid)
    arrays["sidewalk"] = valid.copy() if surface["kind"] == "sidewalk" else np.zeros_like(valid)
    arrays["junction"] = valid.copy() if surface["is_junction"] else np.zeros_like(valid)
    lane_validity = valid.copy()
    unsupported = collections.Counter()
    for key, channel in (("lane_markings", "lane"), ("boundary_lines", "road_boundary")):
        for line in surface[key]:
            pixels = (np.asarray(line["points_xyz"], dtype=np.float64)[:, :2] - origin) / mpp - start
            thickness = max(1, int(round(line["width_m"] / mpp)))
            if channel == "lane":
                marking_type = line["type"].rsplit(".", 1)[-1].lower()
                if marking_type in ("none", "other") and not line["paint_geometry"]:
                    if marking_type == "none":
                        continue
                elif marking_type == "solid" or line["paint_geometry"]:
                    marking_type = "resolved"
                if marking_type != "resolved":
                    unsupported[line["type"]] += 1
                    # Waypoints identify a marking boundary, not dash geometry.
                    # Exclude its support band from paint supervision instead of
                    # turning Broken/compound/unknown markings into solid paint.
                    cv2.polylines(lane_validity, [np.rint(pixels).astype(np.int32)], False, 0,
                                  thickness + 4, lineType=cv2.LINE_8)
                    continue
            # Discrete values only, never antialias semantic labels.
            cv2.polylines(arrays[channel], [np.rint(pixels).astype(np.int32)], False, 1,
                          thickness, lineType=cv2.LINE_8)
        arrays[channel] *= valid
    arrays["lane"] *= lane_validity
    arrays.update(height=elevation, validity=valid, lane_validity=lane_validity)
    return arrays, [int(start[0]), int(start[1]), int(stop[0]) + 1, int(stop[1]) + 1], dict(unsupported)


class GlobalRasterCache:
    """A disk-backed sparse global map with bounded in-memory patch arrays."""

    @classmethod
    def build(cls, cache_dir, town, opendrive_text, surfaces, meters_per_pixel=0.2, build_options=None, build_report=None):
        import numpy as np
        source = _source_manifest(town, opendrive_text, surfaces, meters_per_pixel, build_options, build_report)
        signature = _sha(_json_bytes(source))
        parent = Path(cache_dir).expanduser().resolve() / town
        target = parent / signature[:24]
        if target.exists():
            loaded = cls.load(target)
            if loaded.metadata["signature"] != signature:
                raise ValueError("Cache identity collision")
            return loaded
        parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".build-", dir=str(parent)))
        try:
            (temporary / "layers").mkdir()
            mpp = source["meters_per_pixel"]
            xmin = min(p[0] for s in source["surfaces"] for p in s["polygon_xyz"])
            ymin = min(p[1] for s in source["surfaces"] for p in s["polygon_xyz"])
            origin = np.floor(np.array([xmin, ymin], dtype=np.float64) / mpp) * mpp
            np.save(str(temporary / "world_offset.npy"), origin, allow_pickle=False)
            records = []
            for surface in source["surfaces"]:
                arrays, bounds, unsupported = _raster(surface, origin, mpp)
                filename = "layers/{}.npz".format(_sha(surface["id"].encode("utf-8"))[:24])
                path = temporary / filename
                np.savez_compressed(str(path), **arrays)
                pts = surface["polygon_xyz"]
                record = dict(surface)
                record.update(layer_id=surface["id"], file=filename, sha256=_sha(path.read_bytes()),
                              bounds_px=bounds, offset_world=[float(origin[i] + bounds[i] * mpp) for i in (0, 1)],
                              bounds_world=[min(p[0] for p in pts), min(p[1] for p in pts),
                                            max(p[0] for p in pts), max(p[1] for p in pts)],
                              z_min_m=min(p[2] for p in pts), z_max_m=max(p[2] for p in pts))
                record["unsupported_markings"] = unsupported
                records.append(record)
            metadata = {key: val for key, val in source.items() if key != "surfaces"}
            metadata.update(signature=signature, world_offset=origin.tolist(), patches=records,
                            coordinate_system="CARLA world meters: X forward, Y right, Z up; global u=+X,v=+Y",
                            pixel_reference="integer pixel coordinates denote world sample locations",
                            label_source="simulator_map_ground_truth", height_unit="m", invalid_height="NaN",
                            mask_values=[0, 1], semantic_layers=list(MASK_NAMES),
                            lane_semantics="Known solid map-marking geometry or explicitly resolved paint segments; unresolved markings excluded by lane_validity. Not verified against rendered CARLA texture.")
            metadata_bytes = _json_bytes(metadata)
            (temporary / "metadata.json").write_bytes(metadata_bytes)
            (temporary / "metadata.sha256").write_text(_sha(metadata_bytes), encoding="ascii")
            try:
                os.rename(str(temporary), str(target))
            except OSError:
                if not target.is_dir():
                    raise
                # A concurrent builder may publish the exact same content key.
                existing = cls.load(target)
                if existing.metadata["signature"] != signature:
                    raise ValueError("Concurrent cache build produced another identity")
            return cls.load(target)
        finally:
            if temporary.exists():
                shutil.rmtree(str(temporary))

    @classmethod
    def load(cls, cachepath, max_cached_patches=4096, max_cache_bytes=64 * 1024 * 1024):
        return cls(cachepath, max_cached_patches=max_cached_patches, max_cache_bytes=max_cache_bytes)

    @classmethod
    def find(cls, cache_dir, town, opendrive_text, meters_per_pixel=0.2, build_options=None):
        """Reuse only an exact map/resolution/builder-options identity.

        The adapter must include its own version, waypoint spacing, marking
        policy and every geometry-affecting option in build_options.
        Multiple matching signatures are ambiguous and require a fresh build.
        """
        if not re.match(r"^[A-Za-z0-9_.-]+$", town) or town in (".", ".."):
            raise ValueError("Invalid town")
        root = Path(cache_dir).expanduser().resolve() / town
        identity = {"format": FORMAT_VERSION, "generator_version": GENERATOR_VERSION,
                    "town": town, "opendrive_sha256": _sha(opendrive_text.encode("utf-8")),
                    "meters_per_pixel": float(meters_per_pixel), "build_options": build_options or {}}
        matches = []
        if not root.is_dir():
            return None
        for folder in root.iterdir():
            if not re.match(r"^[0-9a-f]{24}$", folder.name) or not folder.is_dir():
                continue
            try:
                raw = (folder / "metadata.json").read_bytes()
                if _sha(raw) != (folder / "metadata.sha256").read_text(encoding="ascii").strip():
                    continue
                metadata = json.loads(raw.decode("utf-8"))
                if all(metadata.get(key) == value for key, value in identity.items()):
                    matches.append(folder)
            except (OSError, ValueError, KeyError):
                continue
        return cls.load(matches[0]) if len(matches) == 1 else None

    def __init__(self, cachepath, max_cached_patches=4096, max_cache_bytes=64 * 1024 * 1024):
        import numpy as np
        self.path = Path(cachepath).expanduser().resolve()
        raw = (self.path / "metadata.json").read_bytes()
        if _sha(raw) != (self.path / "metadata.sha256").read_text(encoding="ascii").strip():
            raise ValueError("BEV cache metadata checksum mismatch")
        self.metadata = json.loads(raw.decode("utf-8"))
        m = self.metadata
        if m.get("format") != FORMAT_VERSION or m.get("generator_version") != GENERATOR_VERSION:
            raise ValueError("Unsupported BEV cache format/generator version; rebuild the cache")
        source = {key: m[key] for key in ("format", "generator_version", "town", "opendrive_sha256", "meters_per_pixel", "build_options")}
        if "build_report" in m:
            source["build_report"] = m["build_report"]
        source["surfaces"] = [{key: s[key] for key in _SURFACE_KEYS} for s in m["patches"]]
        if _sha(_json_bytes(source)) != m.get("signature"):
            raise ValueError("BEV cache metadata integrity mismatch")
        self.world_offset = np.asarray(m["world_offset"], dtype=np.float64)
        stored_offset = np.load(str(self.path / "world_offset.npy"), allow_pickle=False)
        if self.world_offset.shape != (2,) or not np.isfinite(self.world_offset).all() or not np.array_equal(stored_offset, self.world_offset):
            raise ValueError("Invalid or mismatched world_offset.npy")
        self.meters_per_pixel = float(m["meters_per_pixel"])
        self.patches = {s["id"]: s for s in m["patches"]}
        self._tile_m = 32.0
        self._tiles = collections.defaultdict(set)
        for sid, record in self.patches.items():
            for tile in self._tile_keys(record["bounds_world"]):
                self._tiles[tile].add(sid)
        self._arrays = collections.OrderedDict()
        self._cache_bytes = 0
        self._max_cached_patches = max(1, int(max_cached_patches))
        self._max_cache_bytes = max(1, int(max_cache_bytes))
        self._triangulations = {sid: _triangles(s["polygon_xyz"]) for sid, s in self.patches.items()}
        self._adjacent = {sid: set(record["neighbors"]) for sid, record in self.patches.items()}
        for sid, neighbors in list(self._adjacent.items()):
            for neighbor in list(neighbors):
                if neighbor not in self._adjacent:
                    raise ValueError("Cache topology references unknown neighbor: " + neighbor)
                self._adjacent[neighbor].add(sid)
        # Frame generation queries this graph for many actors. Edge geometry is
        # immutable, so compute its Euclidean metric weight once during load.
        self._weighted_adjacent = {}
        for sid, neighbors in self._adjacent.items():
            a = self.patches[sid]["center_xyz"]
            edges = []
            for neighbor in sorted(neighbors):
                b = self.patches[neighbor]["center_xyz"]
                length = math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))
                edges.append((neighbor, length))
            self._weighted_adjacent[sid] = tuple(edges)

    def _tile_keys(self, bounds):
        xmin, ymin, xmax, ymax = [float(v) for v in bounds]
        if not all(math.isfinite(v) for v in (xmin, ymin, xmax, ymax)) or xmin > xmax or ymin > ymax:
            raise ValueError("Bounds must be finite ordered xmin,ymin,xmax,ymax")
        for tx in range(int(math.floor(xmin / self._tile_m)), int(math.floor(xmax / self._tile_m)) + 1):
            for ty in range(int(math.floor(ymin / self._tile_m)), int(math.floor(ymax / self._tile_m)) + 1):
                yield tx, ty

    def query(self, boundsworld):
        ids = set()
        for tile in self._tile_keys(boundsworld):
            ids.update(self._tiles.get(tile, ()))
        a, b, c, d = boundsworld
        return [self.patches[sid] for sid in sorted(ids)
                if self.patches[sid]["bounds_world"][0] <= c and self.patches[sid]["bounds_world"][2] >= a
                and self.patches[sid]["bounds_world"][1] <= d and self.patches[sid]["bounds_world"][3] >= b]

    def arrays(self, patch_id):
        import numpy as np
        sid = str(patch_id)
        if sid in self._arrays:
            arrays = self._arrays.pop(sid)
            self._arrays[sid] = arrays
            return arrays
        record = self.patches[sid]
        path = (self.path / record["file"]).resolve()
        if self.path not in path.parents or _sha(path.read_bytes()) != record["sha256"]:
            raise ValueError("BEV patch file integrity mismatch: " + sid)
        with np.load(str(path), allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in MASK_NAMES + ("height", "validity", "lane_validity")}
        shape = (record["bounds_px"][3] - record["bounds_px"][1], record["bounds_px"][2] - record["bounds_px"][0])
        for key, value in arrays.items():
            expected = np.dtype(np.float32 if key == "height" else np.uint8)
            if value.shape != shape or value.dtype != expected:
                raise ValueError("Invalid BEV patch array: " + key)
            value.flags.writeable = False
        size = sum(a.nbytes for a in arrays.values())
        while self._arrays and (len(self._arrays) >= self._max_cached_patches or self._cache_bytes + size > self._max_cache_bytes):
            _, old = self._arrays.popitem(last=False)
            self._cache_bytes -= sum(a.nbytes for a in old.values())
        if size <= self._max_cache_bytes:
            self._arrays[sid] = arrays
            self._cache_bytes += size
        return arrays

    def lookup_ground(self, x, y, z, road_id=None, lane_id=None, section_id=None, tolerance_m=2.0):
        """Return an identified surface at XY near ground Z; ambiguous -> None.

        z must be a ground/contact estimate, not bounding-box top or actor height.
        Provided map IDs are constraints: no fallback to a different road/lane.
        """
        values = [float(v) for v in (x, y, z, tolerance_m)]
        if not all(math.isfinite(v) for v in values) or tolerance_m < 0:
            raise ValueError("Ground lookup needs finite XYZ and nonnegative tolerance")
        candidates = []
        for patch in self.query((x, y, x, y)):
            if any(value is not None and patch[key] != int(value)
                   for key, value in (("road_id", road_id), ("lane_id", lane_id), ("section_id", section_id))):
                continue
            p = patch["polygon_xyz"]
            for triangle in self._triangulations[patch["id"]]:
                vertices = [p[i] for i in triangle]
                weights = _barycentric(x, y, *vertices)
                if weights is not None and min(weights) >= -1e-7:
                    ground_z = sum(w * v[2] for w, v in zip(weights, vertices))
                    distance = abs(ground_z - z)
                    if distance <= tolerance_m:
                        candidates.append((distance, patch["id"], ground_z, patch))
                    break
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]))
        distance, sid, ground_z, patch = candidates[0]
        if len(candidates) > 1:
            second = candidates[1]
            if abs(second[0] - distance) <= 0.05 and abs(second[2] - ground_z) > 0.05:
                return None
        return {"patch_id": sid, "layer_id": sid, "ground_z": ground_z, "ground_z_m": ground_z,
                "road_id": patch["road_id"], "section_id": patch["section_id"], "lane_id": patch["lane_id"],
                "distance_z_m": distance, "source": "map_surface_triangle", "kind": patch["kind"]}

    def graph_distances(self, seed_id, allowed_ids=None, max_distance=None):
        """Shortest physical continuity distance in meters, in either direction.

        Direction of permitted driving remains available in record.neighbors.
        Visibility needs both upstream and downstream road surfaces.
        """
        seed = str(seed_id)
        if seed not in self.patches:
            raise KeyError(seed)
        allowed = None if allowed_ids is None else set(str(x) for x in allowed_ids)
        if allowed is not None:
            allowed.add(seed)
        limit = float("inf") if max_distance is None else float(max_distance)
        if math.isnan(limit) or limit < 0:
            raise ValueError("Maximum graph distance must be nonnegative")
        distances = {seed: 0.0}
        queue = [(0.0, seed)]
        while queue:
            distance, current = heapq.heappop(queue)
            if distance > distances[current]:
                continue
            for neighbor, length in self._weighted_adjacent[current]:
                if allowed is not None and neighbor not in allowed:
                    continue
                candidate = distance + length
                if candidate <= limit and candidate < distances.get(neighbor, float("inf")):
                    distances[neighbor] = candidate
                    heapq.heappush(queue, (candidate, neighbor))
        return distances

    connected = graph_distances
