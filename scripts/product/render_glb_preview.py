#!/usr/bin/env python3
"""Render a lightweight orthographic PNG preview from a GLB mesh.

This is a diagnostic helper for remote inference smoke tests. It parses GLB
positions/indices directly and avoids heavyweight viewers, Blender, or local GPU
use. The output is not a production renderer; it is a quick visual sanity check.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct

import numpy as np
from PIL import Image, ImageDraw

COMPONENT_DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
TYPE_COUNTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def load_glb(path: Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    magic, version, _length = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67 or version != 2:
        raise ValueError(f"{path} is not a GLB v2 file")

    offset = 12
    gltf = None
    bin_blob = None
    while offset + 8 <= len(data):
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset : offset + chunk_len]
        offset += chunk_len
        if chunk_type == 0x4E4F534A:
            gltf = json.loads(chunk.rstrip(b"\0").decode("utf-8"))
        elif chunk_type == 0x004E4942:
            bin_blob = chunk

    if gltf is None or bin_blob is None:
        raise ValueError(f"{path} does not contain both JSON and BIN chunks")
    return gltf, bin_blob


def read_accessor(gltf: dict, blob: bytes, accessor_index: int) -> np.ndarray:
    accessor = gltf["accessors"][accessor_index]
    buffer_view = gltf["bufferViews"][accessor["bufferView"]]
    dtype = np.dtype(COMPONENT_DTYPES[accessor["componentType"]]).newbyteorder("<")
    components = TYPE_COUNTS[accessor["type"]]
    count = int(accessor["count"])
    base = int(buffer_view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    stride = int(buffer_view.get("byteStride", dtype.itemsize * components))

    raw = np.frombuffer(blob, dtype=np.uint8, count=stride * count, offset=base)
    packed_stride = dtype.itemsize * components
    if stride == packed_stride:
        array = np.frombuffer(raw.tobytes(), dtype=dtype).reshape(count, components)
    else:
        rows = [
            np.frombuffer(raw[i * stride : i * stride + packed_stride].tobytes(), dtype=dtype)
            for i in range(count)
        ]
        array = np.vstack(rows)
    return array.reshape(-1) if components == 1 else array


def first_triangle_primitive(gltf: dict) -> dict:
    for mesh in gltf.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            if primitive.get("mode", 4) == 4 and "indices" in primitive and "POSITION" in primitive.get("attributes", {}):
                return primitive
    raise ValueError("No indexed triangle primitive with POSITION attribute found")


def render_preview(
    input_path: Path,
    output_path: Path,
    *,
    size: int = 1100,
    yaw_degrees: float = 35.0,
    pitch_degrees: float = -18.0,
    max_wire_faces: int = 6000,
) -> dict:
    gltf, blob = load_glb(input_path)
    primitive = first_triangle_primitive(gltf)
    vertices = read_accessor(gltf, blob, primitive["attributes"]["POSITION"]).astype(np.float64)
    indices = read_accessor(gltf, blob, primitive["indices"]).astype(np.int64).reshape(-1, 3)

    vertices = vertices - (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
    vertices = vertices / max(float(np.ptp(vertices, axis=0).max()), 1e-9)

    yaw = math.radians(yaw_degrees)
    pitch = math.radians(pitch_degrees)
    rotation_y = np.array(
        [[math.cos(yaw), 0.0, math.sin(yaw)], [0.0, 1.0, 0.0], [-math.sin(yaw), 0.0, math.cos(yaw)]]
    )
    rotation_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, math.cos(pitch), -math.sin(pitch)], [0.0, math.sin(pitch), math.cos(pitch)]]
    )
    projected_vertices = vertices @ (rotation_y @ rotation_x).T
    triangles = projected_vertices[indices]

    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normal_lengths = np.linalg.norm(normals, axis=1)
    keep = normal_lengths > 1e-10
    triangles = triangles[keep]
    normals = normals[keep] / normal_lengths[keep, None]

    order = np.argsort(triangles[:, :, 2].mean(axis=1))
    triangles = triangles[order]
    normals = normals[order]

    xy = triangles[:, :, :2]
    min_xy = xy.reshape(-1, 2).min(axis=0)
    max_xy = xy.reshape(-1, 2).max(axis=0)
    margin = max(24, int(size * 0.065))
    canvas = size - 2 * margin
    scale = canvas / max(float((max_xy - min_xy).max()), 1e-6)
    points = (xy - (min_xy + max_xy) / 2.0) * scale + size / 2.0
    # Image coordinates grow downward; mesh coordinates use +Y as up.
    points[:, :, 1] = size - points[:, :, 1]

    light = np.array([0.25, -0.45, 0.86])
    light = light / np.linalg.norm(light)
    shade = np.clip((normals @ light) * 0.55 + 0.55, 0.18, 1.0)
    base_color = np.array([198, 194, 181])

    image = Image.new("RGB", (size, size), (236, 233, 222))
    draw = ImageDraw.Draw(image)
    for polygon, value in zip(points, shade):
        color = tuple(np.clip(base_color * value, 0, 255).astype(np.uint8).tolist())
        draw.polygon([tuple(point) for point in polygon], fill=color)

    wire_step = max(1, len(points) // max(1, max_wire_faces))
    for polygon in points[::wire_step]:
        draw.line([tuple(polygon[0]), tuple(polygon[1]), tuple(polygon[2]), tuple(polygon[0])], fill=(84, 82, 76), width=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "vertices": int(len(vertices)),
        "faces": int(len(indices)),
        "rendered_faces": int(len(triangles)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", type=int, default=1100)
    parser.add_argument("--yaw", type=float, default=35.0)
    parser.add_argument("--pitch", type=float, default=-18.0)
    args = parser.parse_args()

    report = render_preview(args.input, args.output, size=args.size, yaw_degrees=args.yaw, pitch_degrees=args.pitch)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
