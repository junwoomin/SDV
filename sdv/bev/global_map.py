"""Direct, versioned loading of the user's pre-rendered global BEV PNGs.

No waypoint generation, OpenDRIVE rasterization or simulator dependency lives
here. PNG bytes are preserved in assets; only mask decoding and local sampling
are allowed. Height-layer names are not metric elevation images.
"""
import collections
from collections.abc import Mapping
import hashlib
import io
import json
import math
from pathlib import Path
import threading
import warnings


CATALOG_VERSION = "sdv-supplied-global-png-v1"
HIGH_THRESHOLDS = {"Town04": 5.0, "Town05": 7.0}
ALIASES = {
    "drivable": ("DAS_BEV",),
    "lane": ("all_lane_BEV", "lane_all_BEV"),
    "road_boundary": ("Boundary_BEV", "boundary_BEV"),
    "stopline": ("stop_line_BEV",),
    "crosswalk": ("crosswalk_BEV",),
    "lane_center": ("lane_center_BEV",),
    "lane_solid": ("lane_solid_BEV", "lane_BEV"),
    "lane_broken": ("lane_broken_BEV", "lane_all_broken_BEV"),
    "stencil": ("stencil_BEV",),
    "stencil_stop": ("stencil_stop_BEV",),
}


def default_map_root():
    return Path(__file__).resolve().parents[2] / "assets" / "bev_maps"


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _root(root):
    return default_map_root() if root is None or str(root).strip() == "" else Path(root).expanduser().resolve()


def _safe_file(root, relative):
    path = (root / relative).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise ValueError("Missing or unsafe global map file: " + relative)
    return path


def _manifest(root):
    path = root / "catalog.json"
    if not path.is_file():
        # An unchanged copy of the supplied folders can use their bundled
        # content manifest. Unknown files never inherit a guessed resolution.
        path = default_map_root() / "catalog.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("format") != CATALOG_VERSION:
        raise ValueError("Unsupported global PNG catalog; provide a verified source catalog")
    for record in value["variants"]:
        expected = record.get("source_signature")
        source = {key: val for key, val in record.items() if key != "source_signature"}
        if expected != _sha(_json(source)):
            raise ValueError("Global PNG catalog record integrity mismatch")
        if record.get("ppm") not in (5, 10) or record.get("meters_per_pixel") != 1.0 / record["ppm"]:
            raise ValueError("Unverified or inconsistent global raster resolution")
    return value


def _check_variant(root, record, verify_hashes=True):
    import numpy as np
    from PIL import Image
    errors = []
    for filename, entry in record["files"].items():
        try:
            path = _safe_file(root, record["directory"] + "/" + filename)
            if path.stat().st_size != entry["bytes"]:
                raise ValueError("byte length differs")
            if verify_hashes and _sha(path.read_bytes()) != entry["sha256"]:
                raise ValueError("SHA-256 differs")
            if filename.endswith(".png"):
                with warnings.catch_warnings():
                    # Verified source metadata bounds this known large image.
                    warnings.simplefilter("ignore", Image.DecompressionBombWarning)
                    with Image.open(path) as image:
                        if list(image.size) != entry["size"] or image.mode != entry["mode"]:
                            raise ValueError("PNG shape/mode differs")
        except (OSError, ValueError) as exc:
            errors.append(filename + ": " + str(exc))
    try:
        path = _safe_file(root, record["directory"] + "/world_offset.npy")
        offset = np.asarray(np.load(str(path), allow_pickle=False), dtype=np.float64).reshape(-1)
        if offset.shape != (2,) or not np.isfinite(offset).all() or not np.array_equal(offset, record["world_offset_xy"]):
            errors.append("world_offset.npy: invalid or changed XY registration")
    except (OSError, ValueError) as exc:
        errors.append("world_offset.npy: " + str(exc))
    return errors


def _layer_files(record, layer_id):
    suffix = "_high" if layer_id == "high" else ""
    result = {}
    for channel, candidates in ALIASES.items():
        for stem in candidates:
            filename = stem + suffix + ".png"
            if filename in record["files"]:
                result[channel] = filename
                break
    return result


def _variant_public(record, errors):
    low, high = _layer_files(record, "low"), _layer_files(record, "high")
    notes = list(record.get("warnings", []))
    if high:
        notes.append("High/low are supplied image layers; no per-pixel metric road elevation was supplied.")
    return {"ppm": record["ppm"], "available": not errors,
            "source_path": record["directory"], "directory": record["directory"],
            "width_px": record["width_px"], "height_px": record["height_px"],
            "meters_per_pixel": record["meters_per_pixel"], "world_offset_xy": record["world_offset_xy"],
            "static_layers": sorted(low), "height_layers": ["low", "high"] if high else ["low"],
            "layers": {"low": low, "high": high}, "warnings": notes,
            "errors": errors, "source_signature": record["source_signature"],
            "resolution_evidence": record.get("resolution_evidence"),
            "metric_elevation_available": False,
            "estimated_plane_bytes": record["width_px"] * record["height_px"]}


def catalog(root=None):
    """Return the verified supplied-map inventory without decoding full PNGs."""
    selected_root = _root(root)
    result = {"root": str(selected_root), "town_records": [], "errors": []}
    try:
        manifest = _manifest(selected_root)
    except (OSError, ValueError, KeyError) as exc:
        result["errors"].append(str(exc))
        return result
    grouped = collections.defaultdict(list)
    for record in manifest["variants"]:
        errors = _check_variant(selected_root, record)
        grouped[record["town"]].append(_variant_public(record, errors))
    for town in sorted(grouped):
        variants = sorted(grouped[town], key=lambda item: item["ppm"])
        result["town_records"].append({"town": town,
            "available_ppm": [v["ppm"] for v in variants if v["available"]], "variants": variants})
    result["archive_sha256"] = manifest.get("archive_sha256")
    result["format"] = CATALOG_VERSION
    return result


class _LayerArrays(Mapping):
    """Lazy array access prevents a catalog request from decoding every map."""
    def __init__(self, owner, files):
        self.owner = owner
        self.files = files

    def __getitem__(self, key):
        return self.owner._array(self.files[key])

    def __iter__(self):
        return iter(self.files)

    def __len__(self):
        return len(self.files)


class GlobalMap:
    def __init__(self, root, record, max_cache_bytes=768 * 1024 * 1024):
        import numpy as np
        self.root = _root(root)
        self.record = record
        errors = _check_variant(self.root, record)
        if errors:
            raise ValueError("Global BEV map is unavailable: " + "; ".join(errors))
        self.town = record["town"]
        self.ppm = record["ppm"]
        self.meters_per_pixel = record["meters_per_pixel"]
        self.world_offset = np.asarray(record["world_offset_xy"], dtype=np.float64)
        self.world_offset.flags.writeable = False
        self.shape = (record["height_px"], record["width_px"])
        self.metadata = _variant_public(record, [])
        self.metadata.update(town=self.town, shape=list(self.shape),
            source="user_supplied_global_png", source_directory=record["directory"],
            format=CATALOG_VERSION, supervision="map_annotation_ground_truth", metric_elevation_available=False,
            coordinate_system="CARLA world XY meters; global column=(X-offsetX)*PPM, row=(Y-offsetY)*PPM",
            original_bytes_preserved=True, mask_decoding="source grayscale > 0; alpha ignored as occupancy",
            height_policy_source="AGILEQ-Training f3eba609de1af9d07a1c3e37283afdc51e7fcf22 data_gen/collector/bev.py HIGH_ROAD_THRESHOLDS")
        self._cache = collections.OrderedDict()
        self._bytes = 0
        self._limit = max(1, int(max_cache_bytes))
        self._lock = threading.RLock()

    def _array(self, filename):
        import numpy as np
        from PIL import Image
        with self._lock:
            if filename in self._cache:
                value = self._cache.pop(filename)
                self._cache[filename] = value
                return value
            path = _safe_file(self.root, self.record["directory"] + "/" + filename)
            payload = path.read_bytes()
            if _sha(payload) != self.record["files"][filename]["sha256"]:
                raise ValueError("Global PNG changed after validation: " + filename)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(payload)) as image:
                    gray = image.convert("L")
                # Avoid retaining the RGBA source while converting a large
                # Town04 plane, and threshold the uint8 buffer in place.
                value = np.array(gray, dtype=np.uint8, copy=True)
                gray.close()
                np.greater(value, 0, out=value)
            if value.shape != self.shape:
                raise ValueError("Source layer dimensions differ: " + filename)
            value.flags.writeable = False
            while self._cache and self._bytes + value.nbytes > self._limit:
                _, old = self._cache.popitem(last=False)
                self._bytes -= old.nbytes
            if value.nbytes <= self._limit:
                self._cache[filename] = value
                self._bytes += value.nbytes
            return value

    def layer_for_points(self, points_xyz, ground_verified=False, lane_type=None):
        """Select supplied low/high layers without creating any static pixels.

        A native Driving surface may use exclusive DAS support to follow a
        ramp. Sidewalk/unknown surfaces cannot use this override: a high road
        can project over a low sidewalk which has no low drivable pixels.
        """
        import numpy as np
        points = np.asarray(points_xyz, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("Layer points must be finite Nx3 world ground coordinates")
        threshold = HIGH_THRESHOLDS.get(self.town)
        high_files = _layer_files(self.record, "high")
        separated = threshold is not None and "drivable" in high_files
        high = points[:, 2] > threshold if separated else np.zeros(len(points), dtype=bool)
        overridden = np.zeros(len(points), dtype=bool)
        used = np.zeros(len(points), dtype=bool)
        uv = np.rint((points[:, :2] - self.world_offset) / self.meters_per_pixel).astype(np.int64)
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < self.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < self.shape[0])
        low_support = np.zeros(len(points), dtype=bool)
        high_support = np.zeros(len(points), dtype=bool)
        driving = str(lane_type).rsplit(".", 1)[-1].lower() == "driving"
        if separated and ground_verified and driving:
            low_array = self._array(_layer_files(self.record, "low")["drivable"])
            high_array = self._array(high_files["drivable"])
            ids = np.flatnonzero(inside)
            low_support[ids] = low_array[uv[ids, 1], uv[ids, 0]] > 0
            high_support[ids] = high_array[uv[ids, 1], uv[ids, 0]] > 0
            used = inside & (low_support != high_support)
            overridden = used & (high != high_support)
            high[used] = high_support[used]
        return {"high": high, "exclusive_support_used": used, "threshold_overridden_by_support": overridden,
                "in_bounds": inside, "low_support": low_support, "high_support": high_support}

    def select_layer(self, ego_pose, policy="hblr", height_aware=True, channels=None):
        if policy not in ("hblr", "hblr_ground_height", "low", "high"):
            raise ValueError("Unsupported supplied map layer policy: " + str(policy))
        notes = list(self.metadata.get("warnings", []))
        height_source = None
        z = None
        if isinstance(ego_pose, Mapping):
            for key in ("ground_z_m", "ground_z", "z"):
                try:
                    candidate = float(ego_pose.get(key))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(candidate):
                    z, height_source = candidate, key
                    break
        else:
            try:
                z = float(ego_pose[2])
                if math.isfinite(z):
                    height_source = "ego_pose_z"
                else:
                    z = None
            except (TypeError, ValueError, IndexError):
                pass
        threshold = HIGH_THRESHOLDS.get(self.town)
        layer_id = "low"
        method = "single_layer"
        support_used = False
        threshold_override = False
        if height_aware and policy == "high":
            layer_id = "high"
            method = "explicit_layer"
        elif height_aware and policy in ("hblr", "hblr_ground_height") and threshold is not None:
            if z is None:
                raise ValueError("A finite ground/contact Z is required to select the supplied high/low map")
            layer_id = "high" if z > threshold else "low"
            method = "HBLR_ground_height_threshold"
            if isinstance(ego_pose, Mapping):
                native = ego_pose.get("ground_verified") is True
                lane_type = ego_pose.get("ground_lane_type", ego_pose.get("lane_type"))
                try:
                    xy = [float(ego_pose[k]) for k in ("x", "y")]
                except (KeyError, TypeError, ValueError):
                    xy = []
                if len(xy) == 2 and all(math.isfinite(v) for v in xy) and height_source in ("ground_z_m", "ground_z"):
                    result = self.layer_for_points([xy + [z]], ground_verified=native, lane_type=lane_type)
                    support_used = bool(result["exclusive_support_used"][0])
                    threshold_override = bool(result["threshold_overridden_by_support"][0])
                    if support_used:
                        layer_id = "high" if result["high"][0] else "low"
                        method = "native_Driving_ground_and_exclusive_supplied_DAS_support"
            notes.append("Town-specific HBLR threshold selects a supplied image layer; road topology and ramp transitions are not encoded in these PNGs.")
            if height_source not in ("ground_z_m", "ground_z"):
                notes.append("Ground Z is absent; layer selection uses ego/contact Z and is uncertain.")
        files = _layer_files(self.record, layer_id)
        if "drivable" not in files:
            raise ValueError("Requested height layer is absent: " + layer_id)
        requested = set(files) if channels is None else set(channels)
        selected = {key: value for key, value in files.items() if key in requested}
        missing = sorted(requested - set(files))
        metadata = dict(self.metadata)
        separated = threshold is not None and "drivable" in _layer_files(self.record, "high")
        metadata.update(layer_id=layer_id, layer_policy=policy, height_selection_source=height_source,
                        selection_z_m=z, height_threshold_m=threshold,
                        height_separation_available=separated,
                        ground_z_m_used=z if height_source in ("ground_z_m", "ground_z") else None,
                        ground_source=ego_pose.get("ground_source", height_source) if isinstance(ego_pose, Mapping) else height_source,
                        layer_selection_method=method, exclusive_support_used=support_used,
                        threshold_overridden_by_support=threshold_override,
                        unavailable_channels=missing, source_files=dict(selected),
                        available_channels=sorted(files), metric_elevation_available=False)
        if missing:
            notes.append("Unavailable in this image layer: " + ", ".join(missing))
        return {"layer_id": layer_id, "arrays": _LayerArrays(self, selected),
                "world_offset": self.world_offset, "meters_per_pixel": self.meters_per_pixel,
                "mpp": self.meters_per_pixel, "elevation_m": None, "metadata": metadata, "warnings": notes}


def load_global_map(root, town, ppm, max_cache_bytes=768 * 1024 * 1024):
    selected_root = _root(root)
    manifest = _manifest(selected_root)
    try:
        ppm = float(ppm)
    except (TypeError, ValueError):
        raise ValueError("Global map PPM must be 5 or 10")
    record = next((item for item in manifest["variants"] if item["town"] == town and item["ppm"] == ppm), None)
    if record is None:
        raise ValueError("No verified supplied global map for {} at {} PPM".format(town, ppm))
    return GlobalMap(selected_root, record, max_cache_bytes=max_cache_bytes)


class GlobalMapCatalog:
    def __init__(self, root=None):
        self.root = _root(root)

    def catalog(self):
        return catalog(self.root)

    def get(self, town, ppm=5, **kwargs):
        return load_global_map(self.root, town, ppm, **kwargs)
