#!/usr/bin/env bash
# Install NDC (Neural Dual Contouring) and FlexiCubes (via NVIDIA Kaolin).
#
# Usage: conda activate clearmesh && ./install_ndc.sh

set -euo pipefail

DATA_DIR="${1:-/mnt/data}"

echo "=== Installing NDC ==="

cd "${DATA_DIR}"

# Clone NDC repository
if [ ! -d "NDC" ]; then
    git clone https://github.com/czq142857/NDC.git
fi
cd NDC

# Build CUDA extensions
echo "Building NDC CUDA extensions..."
python setup.py build_ext --inplace 2>/dev/null || {
    echo "NDC CUDA build failed - this is expected if not on GPU machine."
    echo "Build will succeed on the GCP VM with CUDA toolkit."
}

# Install FlexiCubes via NVIDIA Kaolin
echo ""
echo "=== Installing FlexiCubes (NVIDIA Kaolin) ==="
pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.6.0_cu124.html 2>/dev/null || {
    echo "Kaolin install may need adjustment for your exact PyTorch/CUDA versions."
    echo "Check: https://github.com/NVIDIAGameWorks/kaolin#installation"
}

# Verify FlexiCubes
python -c "
try:
    import kaolin
    print(f'Kaolin {kaolin.__version__} installed')
    from kaolin.non_commercial import FlexiCubes
    print('FlexiCubes available')
except ImportError as e:
    print(f'Kaolin not yet available: {e}')
    print('Will be available after installing on GPU VM with matching CUDA')
"

echo ""
echo "=== NDC and FlexiCubes setup complete ==="
