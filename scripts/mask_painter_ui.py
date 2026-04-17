"""Mask-painting UI for ClearMesh region edits and mesh surgery.

Two tabs:

  Tab 1 — 2D Region Mask Painter
    Upload a source image (the one you'd feed TRELLIS.2), paint a mask
    over the area you want to edit, preview the dilated/blurred mask,
    save as PNG. The PNG plugs directly into EditOptions.region_mask for
    Easy3EEditor.edit_from_source_image().

  Tab 2 — 3D Bounding Box Picker
    Upload a GLB, see three orthographic renders (front/top/side). Six
    sliders control the bbox in normalized coords (matches surgery.py's
    ``coords='normalized'`` convention: centroid-subtracted, divided by
    mesh.extents.max()). The bbox is drawn on each view live. Click
    "Run surgery" to call remove_by_bounding_box with the current values
    and download the fixed GLB.

Run (pod):
    cd /workspace/clearmesh
    pip install gradio==4.*
    python scripts/mask_painter_ui.py --port 7860

Then port-forward from your laptop:
    ssh -L 7860:localhost:7860 vastai_pod
and open http://localhost:7860 in a browser.

Design notes:
  * Uses Gradio 4.x ImageEditor for painting (no custom JS). This
    widget returns {background, layers, composite} — we only need the
    alpha-merged ``layers`` to build a binary mask.
  * 3D bbox preview rasterizes the mesh to 3 views with a cheap
    PIL-based orthographic projection (no GL required) so it works on
    any pod, including CPU-only. For a live render use pyrender.
  * No implicit state — every callback is pure, re-computed from
    current widget values. Makes iteration predictable.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Ensure clearmesh is on path (pod layout)
for p in ("/workspace/clearmesh",):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------
# 2D mask helpers
# ---------------------------------------------------------------------

def mask_from_editor(
    editor_value: dict | None,
    dilation: int = 2,
    blur_radius: float = 6.0,
) -> tuple[Image.Image | None, Image.Image | None]:
    """Convert Gradio ImageEditor output to a binary + preview mask.

    Gradio 4.x returns a dict with keys:
      - "background": PIL.Image (what the user uploaded)
      - "layers": list of PIL.Image (one per brush stroke layer, RGBA)
      - "composite": PIL.Image (background + layers merged)

    We OR all the layer alpha channels into one binary mask, then apply
    dilation + Gaussian blur for smoother region boundaries — matching
    the defaults in EditOptions.region_mask.
    """
    if editor_value is None:
        return None, None
    layers = editor_value.get("layers") or []
    if not layers:
        return None, None

    bg = editor_value.get("background")
    w, h = bg.size if bg is not None else layers[0].size

    # OR all layers' alpha
    acc = np.zeros((h, w), dtype=np.uint8)
    for layer in layers:
        if layer.size != (w, h):
            layer = layer.resize((w, h))
        arr = np.asarray(layer.convert("RGBA"))
        alpha = arr[..., 3]
        acc = np.maximum(acc, alpha)

    mask = Image.fromarray(acc, mode="L")

    # Dilation via MaxFilter (kernel size = 2*dilation+1)
    if dilation > 0:
        mask = mask.filter(ImageFilter.MaxFilter(size=2 * dilation + 1))

    # Gaussian blur for soft edges
    if blur_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))

    # Preview: overlay the mask in red over the source image
    preview = None
    if bg is not None:
        preview = bg.convert("RGBA").copy()
        red = Image.new("RGBA", preview.size, (255, 0, 0, 0))
        red_alpha = np.asarray(mask) // 2  # 0..127 so background still visible
        red_arr = np.zeros((*preview.size[::-1], 4), dtype=np.uint8)
        red_arr[..., 0] = 255
        red_arr[..., 3] = red_alpha
        red = Image.fromarray(red_arr, mode="RGBA")
        preview = Image.alpha_composite(preview, red).convert("RGB")

    return mask, preview


# ---------------------------------------------------------------------
# 3D bbox helpers
# ---------------------------------------------------------------------

@dataclass
class BBox3D:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float


def _load_normalized_mesh(glb_path: str):
    """Load a mesh and normalize to the same frame surgery.py expects:
    centroid-subtracted, divided by extents.max().
    """
    import trimesh

    mesh = trimesh.load(glb_path, force="mesh")
    centered = mesh.vertices - mesh.centroid
    ext = mesh.extents.max()
    if ext > 0:
        v_norm = centered / ext
    else:
        v_norm = centered
    return mesh, v_norm


def _orthographic_render(
    v_norm: np.ndarray,
    faces: np.ndarray,
    axis: str,  # "front" (xy), "top" (xz), "side" (yz)
    size: int = 512,
    bbox: BBox3D | None = None,
) -> Image.Image:
    """Cheap point-cloud-style silhouette render with bbox overlay.

    Trades quality for zero-GL dependency: projects all vertices onto a
    2D grid, draws 1-pixel dots for silhouette + bbox rectangle for the
    surgery region. Enough to pick a tight box visually.
    """
    # Pick 2 of 3 axes
    if axis == "front":
        u, v = v_norm[:, 0], v_norm[:, 1]
        label_u, label_v = "x", "y"
        bmin_u, bmax_u = (bbox.x_min, bbox.x_max) if bbox else (None, None)
        bmin_v, bmax_v = (bbox.y_min, bbox.y_max) if bbox else (None, None)
        flip_v = True  # y up
    elif axis == "top":
        u, v = v_norm[:, 0], v_norm[:, 2]
        label_u, label_v = "x", "z"
        bmin_u, bmax_u = (bbox.x_min, bbox.x_max) if bbox else (None, None)
        bmin_v, bmax_v = (bbox.z_min, bbox.z_max) if bbox else (None, None)
        flip_v = False
    elif axis == "side":
        u, v = v_norm[:, 2], v_norm[:, 1]
        label_u, label_v = "z", "y"
        bmin_u, bmax_u = (bbox.z_min, bbox.z_max) if bbox else (None, None)
        bmin_v, bmax_v = (bbox.y_min, bbox.y_max) if bbox else (None, None)
        flip_v = True
    else:
        raise ValueError(axis)

    # World range -1..1 mapped to 0..size-1
    def _w2p(a: np.ndarray | float) -> np.ndarray | float:
        return (np.asarray(a) + 1.0) * 0.5 * (size - 1)

    img = Image.new("RGB", (size, size), (20, 22, 28))
    draw = ImageDraw.Draw(img)

    # Axes lines (through origin)
    ox = int(_w2p(0.0))
    oy = int(_w2p(0.0))
    if flip_v:
        oy = size - 1 - oy
    draw.line([(ox, 0), (ox, size - 1)], fill=(50, 55, 65), width=1)
    draw.line([(0, oy), (size - 1, oy)], fill=(50, 55, 65), width=1)

    # Vertex silhouette (subsample big meshes)
    n = u.shape[0]
    if n > 200_000:
        idx = np.random.default_rng(0).choice(n, 200_000, replace=False)
        u, v = u[idx], v[idx]

    pu = _w2p(u).astype(int).clip(0, size - 1)
    pv = _w2p(v).astype(int).clip(0, size - 1)
    if flip_v:
        pv = size - 1 - pv

    # Rasterize — draw as numpy array for speed
    arr = np.asarray(img).copy()
    arr[pv, pu] = (230, 235, 245)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)

    # Bbox rectangle
    if bbox is not None:
        x0 = int(_w2p(bmin_u))
        x1 = int(_w2p(bmax_u))
        y0 = int(_w2p(bmin_v))
        y1 = int(_w2p(bmax_v))
        if flip_v:
            y0, y1 = size - 1 - y0, size - 1 - y1
        lo_x, hi_x = sorted((x0, x1))
        lo_y, hi_y = sorted((y0, y1))
        # Semi-transparent red fill via pixel blend
        arr = np.asarray(img).copy()
        region = arr[lo_y:hi_y + 1, lo_x:hi_x + 1].astype(np.int32)
        region[..., 0] = np.minimum(255, region[..., 0] + 80)
        region[..., 1] = np.maximum(0, region[..., 1] - 30)
        region[..., 2] = np.maximum(0, region[..., 2] - 30)
        arr[lo_y:hi_y + 1, lo_x:hi_x + 1] = region.astype(np.uint8)
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        draw.rectangle([(lo_x, lo_y), (hi_x, hi_y)], outline=(255, 100, 80), width=2)

    # Axis labels
    draw.text((6, 6), f"{axis.upper()} view ({label_u}, {label_v})", fill=(140, 150, 170))
    return img


def render_three_views(
    glb_path: str | None,
    bbox: BBox3D,
    size: int = 512,
) -> tuple[Image.Image | None, Image.Image | None, Image.Image | None, str]:
    if not glb_path:
        blank = Image.new("RGB", (size, size), (20, 22, 28))
        return blank, blank, blank, "(no mesh loaded)"
    try:
        mesh, v_norm = _load_normalized_mesh(glb_path)
    except Exception as e:
        err = Image.new("RGB", (size, size), (40, 20, 20))
        ImageDraw.Draw(err).text((10, 10), f"load error: {e}", fill=(255, 200, 200))
        return err, err, err, f"load failed: {e}"
    f = _orthographic_render(v_norm, mesh.faces, "front", size, bbox)
    t = _orthographic_render(v_norm, mesh.faces, "top", size, bbox)
    s = _orthographic_render(v_norm, mesh.faces, "side", size, bbox)

    info = (
        f"{len(mesh.vertices):,} verts / {len(mesh.faces):,} faces | "
        f"extents = {mesh.extents.round(3).tolist()} | "
        f"bbox = ({bbox.x_min:.2f},{bbox.y_min:.2f},{bbox.z_min:.2f}) "
        f"→ ({bbox.x_max:.2f},{bbox.y_max:.2f},{bbox.z_max:.2f})"
    )
    return f, t, s, info


def run_surgery(
    glb_path: str | None,
    x_min: float, y_min: float, z_min: float,
    x_max: float, y_max: float, z_max: float,
    fill_holes: bool,
    verbose: bool = True,
) -> tuple[str | None, str]:
    """Invoke clearmesh.mesh.surgery.remove_by_bounding_box and return a
    path to the fixed GLB.
    """
    if not glb_path:
        return None, "upload a GLB first"
    try:
        import trimesh
        from clearmesh.mesh.surgery import remove_by_bounding_box
    except Exception as e:
        return None, f"import error: {e}"

    try:
        mesh = trimesh.load(glb_path, force="mesh")
        fixed = remove_by_bounding_box(
            mesh=mesh,
            bbox_min=(x_min, y_min, z_min),
            bbox_max=(x_max, y_max, z_max),
            coords="normalized",
            fill_holes=fill_holes,
            verbose=verbose,
        )
    except Exception as e:
        return None, f"surgery failed: {e}"

    out = tempfile.NamedTemporaryFile(suffix="_fixed.glb", delete=False)
    out.close()
    fixed.export(out.name)
    msg = (
        f"OK: {len(mesh.vertices):,}v/{len(mesh.faces):,}f → "
        f"{len(fixed.vertices):,}v/{len(fixed.faces):,}f"
    )
    return out.name, msg


# ---------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------

def build_ui():
    import gradio as gr

    with gr.Blocks(title="ClearMesh Mask Painter") as demo:
        gr.Markdown(
            "# ClearMesh Mask Painter\n"
            "Paint 2D region masks for text-guided edits, or pick a 3D "
            "bounding box for artifact surgery."
        )

        with gr.Tab("2D Region Mask"):
            gr.Markdown(
                "**Use:** Upload source image → paint over the region you "
                "want the edit to affect (e.g. the wings area of a "
                "dragon). Tweak dilation/blur. Download the mask PNG and "
                "pass it as `EditOptions.region_mask`."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    editor = gr.ImageEditor(
                        label="Paint region (brush only)",
                        type="pil",
                        sources=("upload",),
                        layers=True,
                        brush=gr.Brush(colors=["#ff3355"], default_size=40),
                        eraser=gr.Eraser(default_size=30),
                        height=560,
                    )
                    dil = gr.Slider(0, 20, value=2, step=1, label="Dilation (px)")
                    blur = gr.Slider(0.0, 40.0, value=6.0, step=0.5, label="Gaussian blur radius (px)")
                with gr.Column(scale=1):
                    mask_preview = gr.Image(label="Mask overlay", interactive=False, height=280)
                    raw_mask = gr.Image(label="Binary mask (download-ready)", interactive=False, height=280, image_mode="L")

            def _recalc(ev, d, b):
                mask, overlay = mask_from_editor(ev, dilation=int(d), blur_radius=float(b))
                return overlay, mask

            for w in (editor, dil, blur):
                w.change(_recalc, inputs=[editor, dil, blur], outputs=[mask_preview, raw_mask])

        with gr.Tab("3D Bounding Box"):
            gr.Markdown(
                "**Use:** Upload a GLB, adjust the six sliders to tightly "
                "enclose the artifact you want to remove. Coordinates are "
                "**normalized** (centroid-subtracted, divided by "
                "`mesh.extents.max()`) — same convention as "
                "`surgery.remove_by_bounding_box(coords='normalized')`. "
                "Click **Run surgery** to delete faces inside the bbox and "
                "fill the resulting hole."
            )
            mesh_in = gr.File(label="Upload GLB", file_types=[".glb", ".obj", ".ply", ".stl"])

            with gr.Row():
                x_min = gr.Slider(-1.0, 1.0, value=-0.05, step=0.01, label="x_min")
                x_max = gr.Slider(-1.0, 1.0, value=0.22, step=0.01, label="x_max")
            with gr.Row():
                y_min = gr.Slider(-1.0, 1.0, value=0.15, step=0.01, label="y_min")
                y_max = gr.Slider(-1.0, 1.0, value=0.22, step=0.01, label="y_max")
            with gr.Row():
                z_min = gr.Slider(-1.0, 1.0, value=-0.42, step=0.01, label="z_min")
                z_max = gr.Slider(-1.0, 1.0, value=0.05, step=0.01, label="z_max")

            info = gr.Markdown("(no mesh loaded)")
            with gr.Row():
                front = gr.Image(label="Front (X, Y)", interactive=False, height=360)
                top = gr.Image(label="Top (X, Z)", interactive=False, height=360)
                side = gr.Image(label="Side (Z, Y)", interactive=False, height=360)

            with gr.Row():
                fill = gr.Checkbox(value=True, label="Fill hole after cut")
                run_btn = gr.Button("Run surgery", variant="primary")

            result_file = gr.File(label="Fixed GLB")
            result_msg = gr.Markdown()

            def _recalc_views(mesh_file, xl, xh, yl, yh, zl, zh):
                path = mesh_file.name if mesh_file else None
                # swap if inverted
                xl, xh = sorted((xl, xh))
                yl, yh = sorted((yl, yh))
                zl, zh = sorted((zl, zh))
                bbox = BBox3D(xl, xh, yl, yh, zl, zh)
                f, t, s, msg = render_three_views(path, bbox)
                return f, t, s, msg

            for w in (mesh_in, x_min, x_max, y_min, y_max, z_min, z_max):
                w.change(
                    _recalc_views,
                    inputs=[mesh_in, x_min, x_max, y_min, y_max, z_min, z_max],
                    outputs=[front, top, side, info],
                )

            def _run(mesh_file, xl, yl, zl, xh, yh, zh, do_fill):
                path = mesh_file.name if mesh_file else None
                xl, xh = sorted((xl, xh))
                yl, yh = sorted((yl, yh))
                zl, zh = sorted((zl, zh))
                out_path, msg = run_surgery(path, xl, yl, zl, xh, yh, zh, do_fill)
                return out_path, msg

            run_btn.click(
                _run,
                inputs=[mesh_in, x_min, y_min, z_min, x_max, y_max, z_max, fill],
                outputs=[result_file, result_msg],
            )

    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--share", action="store_true", help="Expose via gradio.live tunnel")
    args = ap.parse_args()

    demo = build_ui()
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
