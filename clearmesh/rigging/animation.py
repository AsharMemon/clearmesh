"""Video-guided animation via Puppeteer's differentiable optimization pipeline.

Given a rigged mesh and a reference video, Puppeteer generates animation
keyframes through differentiable rendering and optimization.

This is an optional post-rigging step for character meshes.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh


class Animator:
    """Video-guided animation using Puppeteer.

    Takes a rigged mesh (skeleton + skinning weights) and a reference video,
    produces animation keyframes via differentiable optimization.

    Usage:
        animator = Animator()
        animation = animator.animate(rigged_fbx_path, video_path)
        animator.export(animation, 'animated.fbx')
    """

    def __init__(self, puppeteer_dir: str = "/mnt/data/Puppeteer"):
        self.puppeteer_dir = puppeteer_dir

    def animate(
        self,
        rigged_mesh_path: str,
        reference_video: str,
        output_path: str | None = None,
        num_frames: int | None = None,
    ) -> str:
        """Generate animation from video reference.

        Puppeteer's animation pipeline:
          1. Extract 2D pose sequences from reference video
          2. Differentiable optimization to fit 3D skeleton motion
          3. Apply motion to rigged mesh via skinning weights
          4. Export animated FBX

        Args:
            rigged_mesh_path: Path to rigged FBX/GLB
            reference_video: Path to reference video file
            output_path: Output animated FBX path (auto-generated if None)
            num_frames: Override number of frames (defaults to video length)

        Returns:
            Path to animated FBX file
        """
        if output_path is None:
            stem = Path(rigged_mesh_path).stem
            output_path = str(Path(rigged_mesh_path).parent / f"{stem}_animated.fbx")

        cmd = [
            sys.executable,
            "run_animation.py",
            "--input",
            rigged_mesh_path,
            "--video",
            reference_video,
            "--output",
            output_path,
        ]

        if num_frames is not None:
            cmd.extend(["--num_frames", str(num_frames)])

        subprocess.run(cmd, cwd=self.puppeteer_dir, check=True, capture_output=True)

        return os.path.abspath(output_path)

    def animate_from_poses(
        self,
        rigged_mesh_path: str,
        pose_sequence: np.ndarray,
        output_path: str,
    ) -> str:
        """Generate animation from a sequence of joint poses.

        For programmatic animation without a reference video.

        Args:
            rigged_mesh_path: Path to rigged FBX/GLB
            pose_sequence: (T, J, 3) array of joint positions per frame
            output_path: Output animated FBX path

        Returns:
            Path to animated FBX file
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            poses_path = os.path.join(tmpdir, "poses.npy")
            np.save(poses_path, pose_sequence)

            cmd = [
                sys.executable,
                "run_animation.py",
                "--input",
                rigged_mesh_path,
                "--poses",
                poses_path,
                "--output",
                output_path,
            ]

            subprocess.run(cmd, cwd=self.puppeteer_dir, check=True, capture_output=True)

        return os.path.abspath(output_path)
