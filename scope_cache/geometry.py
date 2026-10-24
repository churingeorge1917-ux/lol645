"""Геометрия Cache, линии обзора и приближённая оценка смока."""

from __future__ import annotations
import hashlib
import json
import math
import struct
import tempfile
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from awpy import NavMesh, VisibilityChecker
from scipy.spatial import cKDTree
from scope_cache import ASSETS

SMOKE_PARAMETERS = {
    "radius_xy": 144.0,
    "radius_z": 96.0,
    "center_height": 72.0,
    "minimum_chord": 32.0,
    "context_radius": 450.0,
    "quantization_units": 16.0,
    "sensitivity_radii": [96.0, 144.0, 192.0],
    "status": "geometric sensitivity assumptions; not calibrated engine smoke physics",
}


def segment_distances(a, b, point):
    a, b, point = np.asarray(a), np.asarray(b), np.asarray(point)
    d = b - a
    squared = np.sum(d * d, axis=-1)
    t = np.clip(np.sum((point - a) * d, axis=-1) / np.maximum(squared, 1e-12), 0, 1)
    return np.linalg.norm(a + t[..., None] * d - point, axis=-1)


def smoke_chords(a, b, center, radius_xy=144.0, radius_z=96.0):
    if radius_xy <= 0 or radius_z <= 0:
        raise ValueError("Smoke radii must be positive")
    a, b, center = (
        np.asarray(a, dtype=float),
        np.asarray(b, dtype=float),
        np.asarray(center, dtype=float),
    )
    scale = np.array([radius_xy, radius_xy, radius_z])
    delta, offset = (b - a) / scale, (a - center) / scale
    aa = np.sum(delta * delta, axis=-1)
    bb = 2 * np.sum(offset * delta, axis=-1)
    cc = np.sum(offset * offset, axis=-1) - 1
    discriminant = bb * bb - 4 * aa * cc
    root = np.sqrt(np.maximum(discriminant, 0))
    safe = np.maximum(2 * aa, 1e-20)
    lo, hi = (-bb - root) / safe, (-bb + root) / safe
    fraction = np.maximum(0, np.minimum(hi, 1) - np.maximum(lo, 0))
    return np.where(
        (discriminant > 0) & (aa > 1e-20),
        fraction * np.linalg.norm(b - a, axis=-1),
        0.0,
    )


def read_mesh(path=ASSETS / "de_cache.mesh"):
    raw = Path(path).read_bytes()
    magic, version, vertices, faces = struct.unpack("<4sIII", raw[:16])
    if magic != b"AWMH" or version != 1 or len(raw) != 16 + vertices * 12 + faces * 12:
        raise ValueError("Unsupported or truncated collision mesh")
    points = np.frombuffer(raw, dtype="<f4", count=vertices * 3, offset=16).reshape(-1, 3)
    triangles = np.frombuffer(raw, dtype="<u4", count=faces * 3, offset=16 + vertices * 12).reshape(
        -1, 3
    )
    if triangles.max() >= vertices or not np.isfinite(points).all():
        raise ValueError("Invalid mesh vertices/indices")
    return points, triangles


class MapGeometry:
    def __init__(self, assets=ASSETS):
        self.assets = Path(assets)
        self.manifest = verify_assets(self.assets)
        self.visibility = VisibilityChecker(self.assets / "de_cache.mesh")
        graph = np.load(self.assets / "sightlines.npz", allow_pickle=False)
        self.a, self.b = graph["origins"], graph["targets"]
        self.midpoint = (self.a + self.b) / 2
        self.half_length = np.linalg.norm(self.b - self.a, axis=1) / 2
        self.tree = cKDTree(self.midpoint)
        self.search_radius = float(self.half_length.max()) + SMOKE_PARAMETERS["context_radius"]

    def visible(self, a, b):
        if len(a) != 3 or len(b) != 3 or not np.isfinite([a, b]).all():
            raise ValueError("Visibility requires two finite XYZ points")
        return self.visibility.is_visible(tuple(a), tuple(b))

    def smoke_at(self, endpoint, *, quantize=True):
        endpoint = np.asarray(endpoint, dtype=float)
        if endpoint.shape != (3,) or not np.isfinite(endpoint).all():
            raise ValueError("Smoke endpoint must contain three finite coordinates")
        q = SMOKE_PARAMETERS["quantization_units"]
        if quantize:
            endpoint = np.round(endpoint / q) * q
        center = endpoint + np.array([0, 0, SMOKE_PARAMETERS["center_height"]])
        indices = np.asarray(self.tree.query_ball_point(center, self.search_radius), dtype=int)
        if len(indices):
            distance = segment_distances(self.a[indices], self.b[indices], center)
            indices = indices[distance <= SMOKE_PARAMETERS["context_radius"]]
        coverage, counts = [], []
        for radius in SMOKE_PARAMETERS["sensitivity_radii"]:
            chords = smoke_chords(
                self.a[indices],
                self.b[indices],
                center,
                radius,
                SMOKE_PARAMETERS["radius_z"],
            )
            blocked = int(np.count_nonzero(chords >= SMOKE_PARAMETERS["minimum_chord"]))
            counts.append(blocked)
            coverage.append(blocked / len(indices) if len(indices) else None)
        return {
            "potential_coverage": coverage[1],
            "coverage_radius_96": coverage[0],
            "coverage_radius_192": coverage[2],
            "blocked_static_sightlines": counts[1],
            "candidate_static_sightlines": len(indices),
            "status": "static_geometric_proxy" if len(indices) else "no_candidate_sightlines",
            "actual_enemy_view_blocked": None,
            "quantized": quantize,
        }

    def snapshot_blocking(self, smoke_endpoint, players, *, tick, smoke_begin, smoke_end):
        if not smoke_begin <= tick < smoke_end:
            return {"status": "inactive_smoke", "eligible_pairs": 0, "blocked_pairs": 0}
        if any(p.get("tick") != tick for p in players):
            raise ValueError("All player observations must have the requested tick")
        eligible, blocked = 0, 0
        center = np.asarray(smoke_endpoint) + [0, 0, SMOKE_PARAMETERS["center_height"]]
        for observer in players:
            eye, gaze = (
                np.asarray(observer["eye"], float),
                np.asarray(observer["gaze"], float),
            )
            if (
                observer["side"] not in (2, 3)
                or eye.shape != (3,)
                or gaze.shape != (3,)
                or not np.isfinite([eye, gaze]).all()
                or np.linalg.norm(gaze) == 0
            ):
                raise ValueError("Invalid observer eye/gaze/side")
            fov = float(observer["fov_degrees"])
            if not 0 < fov < 180:
                raise ValueError("FOV must be between 0 and 180 degrees")
            for target in players:
                if target["side"] == observer["side"]:
                    continue
                point = np.asarray(target["eye"], float)
                delta = point - eye
                if np.linalg.norm(delta) == 0:
                    continue
                cosine = np.dot(gaze, delta) / (np.linalg.norm(gaze) * np.linalg.norm(delta))
                if cosine < math.cos(math.radians(fov / 2)) or not self.visible(eye, point):
                    continue
                eligible += 1
                blocked += (
                    float(smoke_chords(eye, point, center)) >= SMOKE_PARAMETERS["minimum_chord"]
                )
        return {
            "status": "snapshot_geometric_proxy",
            "eligible_pairs": eligible,
            "blocked_pairs": blocked,
            "limitation": "Requires externally supplied same-tick telemetry; static geometry, cone FOV and approximate smoke volume",
        }


def verify_assets(assets=ASSETS, names=None):
    assets = Path(assets)
    manifest = json.loads((assets / "manifest.json").read_text())
    for name, meta in manifest["files"].items():
        if names is not None and name not in names:
            continue
        if hashlib.sha256((assets / name).read_bytes()).hexdigest() != meta["sha256"]:
            raise ValueError(f"Map asset checksum mismatch: {name}")
    return manifest


def build_sightlines(assets=ASSETS):
    assets = Path(assets)
    manifest = verify_assets(
        assets, names=["de_cache.mesh", "de_cache.nav", "radar_transform.json"]
    )
    vertices, triangles = read_mesh(assets / "de_cache.mesh")
    nav = NavMesh(assets / "de_cache.nav")
    areas = nav.areas.to_pandas()
    areas["gx"] = np.floor(areas.centroid_x / 128).astype(int)
    areas["gy"] = np.floor(areas.centroid_y / 128).astype(int)
    areas["gz"] = np.floor(areas.centroid_z / 64).astype(int)
    chosen = areas.sort_values("area_id").drop_duplicates(["gx", "gy", "gz"])
    points = chosen[["centroid_x", "centroid_y", "centroid_z"]].to_numpy(dtype=float)
    points[:, 2] += 64.0
    pairs = cKDTree(points).query_pairs(1600, output_type="ndarray")
    distance = np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1)
    pairs = pairs[distance >= 128]
    checker = VisibilityChecker(assets / "de_cache.mesh")
    start = time.monotonic()
    visible = np.fromiter(
        (checker.is_visible(tuple(points[a]), tuple(points[b])) for a, b in pairs),
        dtype=bool,
        count=len(pairs),
    )
    accepted = pairs[visible]
    np.savez_compressed(
        assets / "sightlines.npz",
        origins=points[accepted[:, 0]],
        targets=points[accepted[:, 1]],
        navigation_centers=points,
    )
    report = {
        "client_version": manifest["client_version"],
        "triangles": len(triangles),
        "vertices": len(vertices),
        "navigation_areas": nav.area_count,
        "sampled_eye_points": len(points),
        "tested_segments": len(pairs),
        "clear_static_sightlines": len(accepted),
        "mesh_bounds": [vertices.min(0).tolist(), vertices.max(0).tolist()],
        "eye_height_assumption": 64,
        "sample_xy_bin": 128,
        "sample_z_bin": 64,
        "segment_lengths": [128, 1600],
        "smoke_parameters": SMOKE_PARAMETERS,
        "interpretation": "Potential static lines between sampled walkable locations, not observed player sightlines or expert tactical weights",
        "version_compatibility_with_source": "unknown; source has no game version",
    }
    (assets / "geometry_report.json").write_text(json.dumps(report, indent=2) + "\n")
    for name in ["sightlines.npz", "radar_transform.json", "geometry_report.json"]:
        p = assets / name
        manifest["files"][name] = {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "bytes": p.stat().st_size,
        }
    (assets / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("Visibility build seconds", round(time.monotonic() - start, 2))


PINNED_VERSION = "2000934"


def download(url, path):
    request = urllib.request.Request(url, headers={"User-Agent": "Cache-grenade-PoC"})
    with (
        urllib.request.urlopen(request, timeout=60) as source,
        path.open("wb") as target,
    ):
        while block := source.read(1024 * 1024):
            target.write(block)


def fetch(version=PINNED_VERSION, output=ASSETS):
    if not version.isdigit():
        raise ValueError("Expected a numeric CS2 client version")
    base = f"https://github.com/pnxenopoulos/awpy-data/releases/download/{version}"
    output = Path(output)
    with tempfile.TemporaryDirectory(prefix="cache-map-") as tmp:
        staging = Path(tmp)
        download(base + "/manifest.json", staging / "manifest.json")
        manifest = json.loads((staging / "manifest.json").read_text())
        if manifest["client_version"] != version or "de_cache" not in manifest["maps"]:
            raise ValueError("Release version mismatch or missing de_cache")
        for bundle, filename in [
            ("geometry.zip", "de_cache.mesh"),
            ("navs.zip", "de_cache.nav"),
            ("images.zip", "de_cache.png"),
        ]:
            path = staging / bundle
            download(base + "/" + bundle, path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            expected = manifest["artifacts"][bundle]
            if digest != expected["sha256"] or path.stat().st_size != expected["bytes"]:
                raise ValueError(f"Release archive checksum mismatch: {bundle}")
            with zipfile.ZipFile(path) as archive:
                names = [n for n in archive.namelist() if Path(n).name == filename]
                if len(names) != 1:
                    raise ValueError(f"Ambiguous map asset {filename}")
                (staging / filename).write_bytes(archive.read(names[0]))
        download(base + "/map_data.json", staging / "map_data.json")
        transforms = json.loads((staging / "map_data.json").read_text())
        transform = transforms["de_cache"]
        for key in ["pos_x", "pos_y", "scale"]:
            if key not in transform:
                raise ValueError("Missing radar transform")
        (staging / "radar_transform.json").write_text(json.dumps(transform, indent=2) + "\n")
        report = {
            "checked_at": datetime.now(timezone.utc).date().isoformat(),
            **{
                key: manifest[key]
                for key in [
                    "client_version",
                    "patch_version",
                    "buildid",
                    "source_revision",
                ]
            },
            "release_url": f"https://github.com/pnxenopoulos/awpy-data/releases/tag/{version}",
            "release_generated_at": manifest["generated_at"],
            "coordinate_system": "Hammer XYZ, Z-up",
            "dataset_version_match": "unverified: no source game version in input",
            "geometry": "static, material-filtered and decimated collision mesh; no moving doors, smoke physics or penetration",
            "release_artifacts": manifest["artifacts"],
            "files": {},
        }
        for name in [
            "de_cache.mesh",
            "de_cache.nav",
            "de_cache.png",
            "radar_transform.json",
        ]:
            path = staging / name
            report["files"][name] = {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        output.mkdir(parents=True, exist_ok=True)
        for name in report["files"]:
            (output / name).write_bytes((staging / name).read_bytes())
        (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    build_sightlines(output)
    return report
