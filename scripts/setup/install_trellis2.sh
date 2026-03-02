#!/usr/bin/env bash
# Install TRELLIS.2 and download pre-trained 4B weights.
# Run inside the 'clearmesh' conda environment on GCP VM.
#
# Usage: conda activate clearmesh && ./install_trellis2.sh

set -euo pipefail

DATA_DIR="${1:-/mnt/data}"

echo "=== Installing TRELLIS.2 ==="

cd "${DATA_DIR}"

# Clone repository
if [ ! -d "TRELLIS.2" ]; then
    git clone https://github.com/microsoft/TRELLIS.2.git
fi
cd TRELLIS.2

# Run TRELLIS.2 setup (installs custom CUDA kernels, dependencies)
echo "Running TRELLIS.2 setup script..."
. ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm

# Download pre-trained weights (~20GB from HuggingFace)
echo "Downloading pre-trained TRELLIS.2-4B weights..."
python -c "
from trellis2.pipelines import Trellis2ImageTo3DPipeline
pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
print('TRELLIS.2-4B weights downloaded and loaded successfully')
"

# Quick smoke test
echo "Running smoke test..."
python -c "
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline
pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
pipeline.cuda()
image = Image.open('assets/example_image/T.png')
mesh = pipeline.run(image)[0]
print(f'Generated mesh: {mesh.vertices.shape[0]} verts, {mesh.faces.shape[0]} faces')
import o_voxel
o_voxel.postprocess.to_glb(mesh, '${DATA_DIR}/test_output.glb')
print('Exported to ${DATA_DIR}/test_output.glb')
"

echo ""
echo "=== TRELLIS.2 installed and verified ==="
echo "Test output: ${DATA_DIR}/test_output.glb"
