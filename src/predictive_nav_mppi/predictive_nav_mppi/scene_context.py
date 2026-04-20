from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class ScenePatchConfig:
    size_m: float = 6.0
    pixels: int = 32
    align_to_heading: bool = True


@dataclass(frozen=True)
class OccupancySceneMap:
    yaml_path: Path
    image_path: Path
    resolution: float
    origin_xy: Tuple[float, float]
    occupied_mask: np.ndarray  # [H, W], 1 = occupied, 0 = free

    @property
    def height(self) -> int:
        return int(self.occupied_mask.shape[0])

    @property
    def width(self) -> int:
        return int(self.occupied_mask.shape[1])


_MAP_CACHE: Dict[str, OccupancySceneMap] = {}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_scene_map_path(source_name: str) -> Path:
    maps_dir = _repo_root() / "maps"
    mapping = {
        "corridor_v2": maps_dir / "long_corridor_map.yaml",
        "labyrinth_turns": maps_dir / "labyrinth_turns_map.yaml",
        "world2": maps_dir / "nonlinear_corridor_map.yaml",
        "nonlinear_corridor": maps_dir / "nonlinear_corridor_map.yaml",
        "long_corridor": maps_dir / "long_corridor_map.yaml",
    }
    key = str(source_name).strip().lower()
    if key not in mapping:
        raise KeyError(f"No scene-map mapping for source_name={source_name!r}")
    return mapping[key]


def _parse_simple_yaml(path: Path) -> Dict[str, str]:
    payload: Dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        payload[key.strip()] = value.strip()
    return payload


def _read_pgm(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if b"\\n" in raw[:64] and b"\n" not in raw[:64]:
        raw = raw.replace(b"\\n", b"\n", 3)

    newline_count = 0
    header_end = -1
    for idx, byte in enumerate(raw):
        if byte == 10:
            newline_count += 1
            if newline_count == 3:
                header_end = idx + 1
                break
    if header_end < 0:
        raise ValueError(f"Could not parse PGM header in {path}")

    header_lines = raw[:header_end].decode("ascii", errors="strict").strip().splitlines()
    if len(header_lines) < 3:
        raise ValueError(f"Incomplete PGM header in {path}")
    magic = header_lines[0].strip()
    if magic not in ("P5", "P2"):
        raise ValueError(f"Unsupported PGM format in {path}: {magic!r}")
    width, height = [int(tok) for tok in header_lines[1].split()]
    maxval = int(header_lines[2].strip())

    if magic == "P5":
        payload = raw[header_end:]
        if maxval >= 256:
            data = np.frombuffer(payload[: width * height * 2], dtype=">u2")
        else:
            data = np.frombuffer(payload[: width * height], dtype=np.uint8)
    else:
        values = [int(tok) for tok in raw[header_end:].decode("ascii", errors="strict").split()]
        data = np.asarray(values[: width * height], dtype=np.uint16 if maxval >= 256 else np.uint8)

    if data.size != width * height:
        raise ValueError(f"Unexpected data size in {path}: got {data.size}, expected {width * height}")
    return data.reshape((height, width))


def load_occupancy_scene_map(yaml_path: str | Path) -> OccupancySceneMap:
    path = Path(yaml_path).expanduser().resolve()
    cache_key = str(path)
    cached = _MAP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    meta = _parse_simple_yaml(path)
    image_rel = meta.get("image")
    if not image_rel:
        raise ValueError(f"Map yaml {path} does not contain 'image'")
    image_path = (path.parent / image_rel).resolve()
    resolution = float(meta.get("resolution", "0.05"))
    origin_raw = meta.get("origin", "[-0.0, -0.0, 0.0]").strip()
    if not (origin_raw.startswith("[") and origin_raw.endswith("]")):
        raise ValueError(f"Map yaml {path} has invalid origin: {origin_raw}")
    origin_parts = [float(item.strip()) for item in origin_raw[1:-1].split(",")]
    origin_xy = (origin_parts[0], origin_parts[1])
    negate = int(float(meta.get("negate", "0")))
    occupied_thresh = float(meta.get("occupied_thresh", "0.65"))
    free_thresh = float(meta.get("free_thresh", "0.25"))

    pgm = _read_pgm(image_path).astype(np.float32)
    if pgm.max() <= 0:
        raise ValueError(f"Map image {image_path} is empty")

    norm = pgm / 255.0 if pgm.max() > 1.0 else pgm
    occ_prob = norm if negate else 1.0 - norm
    occupied = (occ_prob >= occupied_thresh).astype(np.float32)
    free = occ_prob <= free_thresh
    occupied = np.where(free, 0.0, occupied).astype(np.float32)

    scene_map = OccupancySceneMap(
        yaml_path=path,
        image_path=image_path,
        resolution=resolution,
        origin_xy=origin_xy,
        occupied_mask=occupied,
    )
    _MAP_CACHE[cache_key] = scene_map
    return scene_map


def _world_to_pixel(scene_map: OccupancySceneMap, world_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    wx = world_xy[..., 0]
    wy = world_xy[..., 1]
    px = (wx - scene_map.origin_xy[0]) / scene_map.resolution
    py = (wy - scene_map.origin_xy[1]) / scene_map.resolution
    row = (scene_map.height - 1) - py
    col = px
    return row, col


def extract_scene_patch(
    scene_map: OccupancySceneMap,
    center_xy: np.ndarray,
    heading_rad: float,
    cfg: ScenePatchConfig,
) -> np.ndarray:
    pixels = max(8, int(cfg.pixels))
    size_m = float(max(0.5, cfg.size_m))
    cell = size_m / float(pixels)
    half = 0.5 * size_m

    coords = (np.arange(pixels, dtype=np.float32) + 0.5) * cell - half
    grid_x, grid_y = np.meshgrid(coords, coords)
    local = np.stack([grid_x, grid_y], axis=-1)

    theta = float(heading_rad if cfg.align_to_heading else 0.0)
    c = math.cos(theta)
    s = math.sin(theta)
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    world = local @ rot.T + np.asarray(center_xy, dtype=np.float32)

    row, col = _world_to_pixel(scene_map, world)
    row_idx = np.rint(row).astype(np.int64)
    col_idx = np.rint(col).astype(np.int64)
    valid = (
        (row_idx >= 0)
        & (row_idx < scene_map.height)
        & (col_idx >= 0)
        & (col_idx < scene_map.width)
    )

    patch = np.ones((pixels, pixels), dtype=np.float32)
    patch[valid] = scene_map.occupied_mask[row_idx[valid], col_idx[valid]]
    return patch[None, ...].astype(np.float32)


def extract_patch_from_source(
    source_name: str,
    center_xy: np.ndarray,
    heading_rad: float,
    cfg: ScenePatchConfig,
) -> np.ndarray:
    scene_map = load_occupancy_scene_map(default_scene_map_path(source_name))
    return extract_scene_patch(scene_map=scene_map, center_xy=center_xy, heading_rad=heading_rad, cfg=cfg)
