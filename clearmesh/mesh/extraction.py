"""Mesh extraction from SDF/voxel data via NDC and FlexiCubes.

NDC (Neural Dual Contouring): Used at inference time for sharp-edge extraction.
FlexiCubes: Used in training loop (differentiable) and optionally at inference.

O-Voxel's native extraction is also supported via TRELLIS.2's built-in pipeline.
"""

import numpy as np
import torch
import trimesh


def extract_ndc(sdf_grid: np.ndarray, ndc_model_path: str | None = None) -> trimesh.Trimesh:
    """Extract mesh using Neural Dual Contouring for sharp edges.

    NDC produces significantly sharper edges than Marching Cubes,
    especially on mechanical/architectural geometry.

    Args:
        sdf_grid: (R, R, R) numpy array of SDF values
        ndc_model_path: Path to trained NDC model weights

    Returns:
        Extracted trimesh.Trimesh
    """
    try:
        # Try using the NDC library directly
        sys_path_added = False
        import sys

        ndc_dir = "/mnt/data/NDC"
        if ndc_dir not in sys.path:
            sys.path.insert(0, ndc_dir)
            sys_path_added = True

        from model import NDC as NDCModel

        if ndc_model_path:
            model = NDCModel()
            model.load_state_dict(torch.load(ndc_model_path, weights_only=True))
            model.eval()
            model.cuda()

            sdf_tensor = torch.from_numpy(sdf_grid).float().unsqueeze(0).unsqueeze(0).cuda()
            with torch.no_grad():
                vertices, faces = model(sdf_tensor)

            vertices = vertices.cpu().numpy()
            faces = faces.cpu().numpy()
        else:
            # Without trained NDC, fall back to Dual Contouring approximation
            vertices, faces = _dual_contouring_fallback(sdf_grid)

        if sys_path_added:
            sys.path.remove(ndc_dir)

        return trimesh.Trimesh(vertices=vertices, faces=faces)

    except ImportError:
        print("NDC not available, falling back to Marching Cubes")
        return extract_marching_cubes(sdf_grid)


def extract_flexicubes(
    sdf_grid: torch.Tensor,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract mesh using FlexiCubes (differentiable).

    For training loop integration (gradients flow back to SDF predictor).

    Args:
        sdf_grid: (R, R, R) tensor of SDF values (requires_grad=True for training)
        device: CUDA device

    Returns:
        vertices: (V, 3) tensor
        faces: (F, 3) tensor
    """
    from kaolin.non_commercial import FlexiCubes

    R = sdf_grid.shape[0]
    fc = FlexiCubes(device=device)

    # Create grid coordinates
    x = torch.linspace(-1, 1, R, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(x, x, x, indexing="ij")
    x_nx3 = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)

    sdf_flat = sdf_grid.reshape(-1)

    # Generate cube indices
    cubes = []
    for i in range(R - 1):
        for j in range(R - 1):
            for k in range(R - 1):
                v0 = i * R * R + j * R + k
                cubes.append([v0, v0 + 1, v0 + R, v0 + R + 1, v0 + R * R, v0 + R * R + 1, v0 + R * R + R, v0 + R * R + R + 1])

    cube_fx8 = torch.tensor(cubes, dtype=torch.long, device=device)

    vertices, faces = fc(x_nx3, sdf_flat, cube_fx8, R)
    return vertices, faces


def extract_marching_cubes(sdf_grid: np.ndarray, level: float = 0.0) -> trimesh.Trimesh:
    """Standard Marching Cubes extraction (baseline, no sharp edges).

    Args:
        sdf_grid: (R, R, R) numpy array of SDF values
        level: Isosurface level

    Returns:
        Extracted trimesh.Trimesh
    """
    try:
        from skimage.measure import marching_cubes

        vertices, faces, normals, _ = marching_cubes(sdf_grid, level=level)
    except ImportError:
        # Fall back to trimesh's built-in
        import mcubes

        vertices, faces = mcubes.marching_cubes(sdf_grid, level)

    # Normalize to [-1, 1]
    R = sdf_grid.shape[0]
    vertices = vertices / R * 2 - 1

    return trimesh.Trimesh(vertices=vertices, faces=faces)


def extract_from_ovoxel(ovoxel_mesh) -> trimesh.Trimesh:
    """Extract mesh from TRELLIS.2's O-Voxel representation.

    O-Voxel natively handles sharp features and complex topology.
    This uses TRELLIS.2's built-in extraction.

    Args:
        ovoxel_mesh: Output from TRELLIS.2 pipeline

    Returns:
        trimesh.Trimesh
    """
    try:
        import o_voxel

        # O-Voxel provides its own extraction with sharp feature preservation
        if hasattr(ovoxel_mesh, "vertices") and hasattr(ovoxel_mesh, "faces"):
            return trimesh.Trimesh(
                vertices=ovoxel_mesh.vertices.cpu().numpy()
                if torch.is_tensor(ovoxel_mesh.vertices)
                else ovoxel_mesh.vertices,
                faces=ovoxel_mesh.faces.cpu().numpy()
                if torch.is_tensor(ovoxel_mesh.faces)
                else ovoxel_mesh.faces,
            )
    except ImportError:
        pass

    # Generic fallback
    if hasattr(ovoxel_mesh, "vertices"):
        verts = ovoxel_mesh.vertices
        faces = ovoxel_mesh.faces
        if torch.is_tensor(verts):
            verts = verts.cpu().numpy()
            faces = faces.cpu().numpy()
        return trimesh.Trimesh(vertices=verts, faces=faces)

    raise ValueError("Cannot extract mesh from O-Voxel output")


def _dual_contouring_fallback(sdf_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Simple Dual Contouring approximation when NDC model isn't available.

    Not as sharp as Neural Dual Contouring but better than Marching Cubes.
    """
    # Use Marching Cubes as baseline, then sharpen edges
    mesh = extract_marching_cubes(sdf_grid)

    # Simple edge sharpening via Laplacian
    vertices = mesh.vertices.copy()
    faces = mesh.faces

    # Compute vertex normals
    mesh_t = trimesh.Trimesh(vertices=vertices, faces=faces)
    normals = mesh_t.vertex_normals

    # Push vertices along normals to sharpen (simple heuristic)
    # This is a rough approximation; real NDC uses a trained network
    laplacian = trimesh.smoothing.laplacian_calculation(mesh_t)
    displacement = laplacian.dot(vertices) - vertices
    sharpness = np.linalg.norm(displacement, axis=1, keepdims=True)
    sharpness = np.clip(sharpness / (sharpness.max() + 1e-8), 0, 1)

    # Sharpen high-curvature vertices
    vertices += normals * sharpness * 0.01

    return vertices, faces
