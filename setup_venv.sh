#!/usr/bin/env bash
# Reinstall packages that are NOT in the base /venv/main image.
# Run this after any instance recycle (torch/numpy/tqdm are already there).
/venv/main/bin/pip install \
    torchaudio --index-url https://download.pytorch.org/whl/cu126 \
    scipy scikit-learn h5py pandas matplotlib osfclient transformers \
    EMD-signal \
    --quiet
echo "Done. Verifying..."
/venv/main/bin/python3 -c "
import torch, torchaudio, scipy, sklearn, h5py, pandas, matplotlib
from PyEMD import CEEMDAN
print('torch:', torch.__version__)
print('torchaudio:', torchaudio.__version__)
print('scipy:', scipy.__version__)
print('pandas:', pandas.__version__)
print('PyEMD (CEEMDAN): OK')
print('CUDA:', torch.cuda.is_available())
"
