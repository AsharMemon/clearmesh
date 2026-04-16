#!/usr/bin/env python3
"""Image Edit — InstructPix2Pix wrapper for text-guided image editing.

Used in the text-guided 3D editing pipeline:
  Text instruction + source mesh render → InstructPix2Pix → edited image → Easy3E

InstructPix2Pix (Brooks et al., 2023) edits images based on text
instructions while preserving the overall structure — ideal for
generating "edit target" images from source mesh renders.

Usage:
    editor = ImageEditor()
    edited = editor.edit(
        source_image="source_render.png",
        instruction="make it look like a robot",
    )
    edited.save("edited_view.png")

    # Or with a PIL Image directly:
    edited = editor.edit(
        source_image=pil_image,
        instruction="add armor plating",
        image_guidance_scale=1.5,
    )
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image


class ImageEditor:
    """InstructPix2Pix-based image editor.

    Wraps the InstructPix2Pix diffusion pipeline for text-guided
    image editing. Used to generate "edit target" images from
    source mesh renders for the Easy3E 3D editing pipeline.
    """

    def __init__(
        self,
        model_id: str = "timbrooks/instruct-pix2pix",
        device: str | None = None,
        dtype: torch.dtype = torch.float16,
    ):
        """Initialize the image editor.

        Args:
            model_id: HuggingFace model ID for InstructPix2Pix.
            device: Compute device.
            dtype: Model dtype (float16 for efficiency).
        """
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self._pipeline = None

    @property
    def pipeline(self):
        """Lazy-load the InstructPix2Pix pipeline."""
        if self._pipeline is None:
            print(f"Loading InstructPix2Pix from {self.model_id}...")
            from diffusers import StableDiffusionInstructPix2PixPipeline

            self._pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                safety_checker=None,
            )
            self._pipeline.to(self.device)
            # Enable memory optimizations
            if hasattr(self._pipeline, "enable_model_cpu_offload"):
                # Only use if VRAM is tight
                pass
            print("InstructPix2Pix loaded.")
        return self._pipeline

    def edit(
        self,
        source_image: str | Path | Image.Image,
        instruction: str,
        num_inference_steps: int = 20,
        image_guidance_scale: float = 1.5,
        guidance_scale: float = 7.5,
        seed: int | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> Image.Image:
        """Edit an image based on a text instruction.

        Args:
            source_image: Source image (path or PIL Image).
            instruction: Text editing instruction (e.g., "make it red").
            num_inference_steps: Diffusion steps (more = higher quality).
            image_guidance_scale: How much to preserve source structure.
                Higher values = more faithful to source, less change.
            guidance_scale: Text guidance scale (CFG).
                Higher values = stronger adherence to instruction.
            seed: Random seed for reproducibility.
            output_size: Resize output to (W, H). None = match input size.

        Returns:
            Edited PIL Image.
        """
        # Load image if path
        if isinstance(source_image, (str, Path)):
            source_image = Image.open(str(source_image)).convert("RGB")

        original_size = source_image.size

        # Resize to model's expected resolution (multiples of 8)
        w, h = source_image.size
        w = (w // 8) * 8
        h = (h // 8) * 8
        if (w, h) != source_image.size:
            source_image = source_image.resize((w, h), Image.LANCZOS)

        # Set seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Run InstructPix2Pix
        result = self.pipeline(
            prompt=instruction,
            image=source_image,
            num_inference_steps=num_inference_steps,
            image_guidance_scale=image_guidance_scale,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        edited = result.images[0]

        # Resize to output size or original size
        target_size = output_size or original_size
        if edited.size != target_size:
            edited = edited.resize(target_size, Image.LANCZOS)

        return edited

    def batch_edit(
        self,
        source_images: list[Image.Image],
        instruction: str,
        **kwargs,
    ) -> list[Image.Image]:
        """Edit multiple images with the same instruction.

        Useful for editing multiple views of a 3D model consistently.

        Args:
            source_images: List of source PIL Images.
            instruction: Text editing instruction.
            **kwargs: Additional arguments passed to edit().

        Returns:
            List of edited PIL Images.
        """
        return [
            self.edit(img, instruction, **kwargs)
            for img in source_images
        ]

    def render_and_edit(
        self,
        mesh_path: str | Path,
        instruction: str,
        view: str = "front",
        image_size: int = 512,
        **edit_kwargs,
    ) -> Image.Image:
        """Render a mesh view and edit it in one step.

        Convenience method for the text-guided editing pipeline.

        Args:
            mesh_path: Path to mesh file.
            instruction: Text editing instruction.
            view: View name (front/back/left/right/top/bottom).
            image_size: Render resolution.
            **edit_kwargs: Additional arguments for edit().

        Returns:
            Edited PIL Image.
        """
        # Render the source view
        source_render = self._render_view(mesh_path, view, image_size)

        # Edit with InstructPix2Pix
        return self.edit(source_render, instruction, **edit_kwargs)

    # Shared view definitions — matches render_ctrl_adapter_data.py convention
    _VIEW_CAMERAS = {
        "front":  {"eye": (0, 0, 2),  "up": (0, 1, 0)},
        "back":   {"eye": (0, 0, -2), "up": (0, 1, 0)},
        "left":   {"eye": (-2, 0, 0), "up": (0, 1, 0)},
        "right":  {"eye": (2, 0, 0),  "up": (0, 1, 0)},
        "top":    {"eye": (0, 2, 0),  "up": (0, 0, -1)},
        "bottom": {"eye": (0, -2, 0), "up": (0, 0, 1)},
    }

    def _render_view(
        self,
        mesh_path: str | Path,
        view: str,
        image_size: int,
    ) -> Image.Image:
        """Render a single view of a mesh to a PIL Image.

        Uses a fallback chain:
          1. pyrender.OffscreenRenderer with PYOPENGL_PLATFORM=egl
             (works on most headless CUDA pods).
          2. nvdiffrast (available as a TRELLIS.2 CUDA dep).
          3. trimesh Scene.save_image (pyglet-based, works on desktops only).

        On headless Vast.ai / RunPod instances, option 1 usually succeeds.
        Option 3 silently returns a grey image when it fails — the previous
        implementation of this method always landed on that code path,
        producing unusable InstructPix2Pix inputs.

        Args:
            mesh_path: Path to mesh file.
            view: View name in _VIEW_CAMERAS.
            image_size: Output resolution (square).

        Returns:
            Rendered PIL Image (RGB, image_size x image_size).
        """
        import numpy as np
        import trimesh

        # Load and normalize mesh to unit cube centered at origin.
        # Centering is important for _VIEW_CAMERAS (camera placed at distance 2).
        mesh = trimesh.load(str(mesh_path), force="mesh")
        mesh.vertices -= mesh.centroid
        scale = mesh.extents.max()
        if scale > 0:
            mesh.vertices /= scale

        cam = self._VIEW_CAMERAS.get(view, self._VIEW_CAMERAS["front"])

        # Attempt 1: pyrender with EGL (preferred for headless GPU pods)
        img = _render_with_pyrender(mesh, cam, image_size)
        if img is not None:
            return img

        # Attempt 2: nvdiffrast (CUDA-only; fast, no X server needed)
        img = _render_with_nvdiffrast(mesh, cam, image_size)
        if img is not None:
            return img

        # Attempt 3: trimesh pyglet (desktop only, returns grey on headless)
        try:
            from io import BytesIO
            scene = trimesh.Scene(mesh)
            scene.camera_transform = trimesh.transformations.look_at(
                np.array(cam["eye"]), np.array([0, 0, 0]), np.array(cam["up"])
            )
            png = scene.save_image(resolution=(image_size, image_size))
            if png is not None:
                rendered = Image.open(BytesIO(png)).convert("RGB")
                # Detect the "silently greyed out" failure mode
                arr = np.asarray(rendered)
                if arr.std() > 1.0:
                    return rendered
        except Exception:
            pass

        # Every attempt failed — return grey with a warning so callers can detect.
        import warnings
        warnings.warn(
            f"[image_edit] All renderers failed for view={view!r}; returning grey fallback. "
            "Install pyrender + pyopengl (EGL) or nvdiffrast on the pod."
        )
        return Image.new("RGB", (image_size, image_size), (128, 128, 128))


def _render_with_pyrender(mesh, cam, image_size):
    """Render with pyrender OffscreenRenderer (PYOPENGL_PLATFORM=egl).

    Returns PIL Image on success, None on any failure.
    """
    import os
    # PYOPENGL_PLATFORM must be set BEFORE pyrender/OpenGL import.
    # If the user has it set to something else, respect that; otherwise default to egl.
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    try:
        import numpy as np
        import pyrender
        import trimesh
    except ImportError:
        return None

    try:
        tri_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        scene = pyrender.Scene(
            ambient_light=(0.3, 0.3, 0.3),
            bg_color=(0, 0, 0, 0),
        )
        scene.add(tri_mesh)

        # Camera: 45° fov, placed at eye, looking at origin
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=1.0)
        cam_pose = trimesh.transformations.look_at(
            np.array(cam["eye"]),
            np.array([0.0, 0.0, 0.0]),
            np.array(cam["up"]),
        )
        scene.add(camera, pose=cam_pose)

        # One directional light from camera direction
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        scene.add(light, pose=cam_pose)

        r = pyrender.OffscreenRenderer(image_size, image_size)
        try:
            color, _ = r.render(scene)
        finally:
            r.delete()

        return Image.fromarray(color[..., :3])
    except Exception:
        return None


def _render_with_nvdiffrast(mesh, cam, image_size):
    """Render with nvdiffrast (differentiable rasterizer, CUDA-only).

    Produces a simple shaded RGB image using face normals. Returns PIL Image
    on success, None on any failure (missing CUDA, missing nvdiffrast, etc).
    """
    try:
        import numpy as np
        import torch
        import trimesh
        if not torch.cuda.is_available():
            return None
        import nvdiffrast.torch as dr
    except ImportError:
        return None

    try:
        device = "cuda"
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
        faces = torch.tensor(mesh.faces, dtype=torch.int32, device=device)

        # Build view-projection matrix
        eye = np.array(cam["eye"], dtype=np.float32)
        up = np.array(cam["up"], dtype=np.float32)
        look = trimesh.transformations.look_at(eye, np.array([0.0, 0.0, 0.0]), up)
        # trimesh look_at returns camera→world; we need world→camera
        view = np.linalg.inv(look).astype(np.float32)

        fov = np.pi / 4.0
        f = 1.0 / np.tan(fov / 2.0)
        near, far = 0.01, 100.0
        proj = np.array(
            [
                [f, 0, 0, 0],
                [0, f, 0, 0],
                [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
                [0, 0, -1, 0],
            ],
            dtype=np.float32,
        )
        mvp = torch.tensor(proj @ view, dtype=torch.float32, device=device)

        verts_h = torch.cat(
            [vertices, torch.ones(vertices.shape[0], 1, device=device)], dim=1
        )
        verts_clip = (verts_h @ mvp.T).unsqueeze(0)  # (1, V, 4)

        glctx = dr.RasterizeCudaContext()
        rast, _ = dr.rasterize(glctx, verts_clip, faces, resolution=(image_size, image_size))

        # Simple face-normal shading
        tri_verts = vertices[faces.long()]
        face_normals = torch.cross(
            tri_verts[:, 1] - tri_verts[:, 0],
            tri_verts[:, 2] - tri_verts[:, 0],
            dim=1,
        )
        face_normals = torch.nn.functional.normalize(face_normals, dim=1)
        # Light from +Z in camera space; dot with world-space normal approximates
        # camera-aligned shading (acceptable for preview, not publication-quality)
        light_dir = torch.tensor(
            [0.5, 0.5, 1.0], dtype=torch.float32, device=device
        )
        light_dir = torch.nn.functional.normalize(light_dir, dim=0)
        shading = (face_normals * light_dir).sum(dim=1).clamp(0.0, 1.0)

        # For each pixel: face index is rast[..., 3]-1 (0 = background)
        tri_id = rast[0, ..., 3].long() - 1
        bg_mask = tri_id < 0
        tri_id = tri_id.clamp(min=0)
        pixel_shade = shading[tri_id]
        pixel_shade[bg_mask] = 0.0

        img = (pixel_shade.unsqueeze(-1).repeat(1, 1, 3) * 255).clamp(0, 255)
        img = img.cpu().numpy().astype(np.uint8)
        return Image.fromarray(img)
    except Exception:
        return None
