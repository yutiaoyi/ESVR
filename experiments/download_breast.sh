#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p data/public .tools/vvv/bin

curl -L --fail --retry 3 \
  http://volume.open-terrain.org/Tumor-DCIS.pvm \
  -o data/public/Tumor-DCIS_448x448x160_uint8.pvm
curl -L --fail --retry 3 \
  https://downloads.sourceforge.net/project/volren/VIEWER-5.2.zip \
  -o /tmp/VIEWER-5.2.zip
unzip -q -o /tmp/VIEWER-5.2.zip -d .tools/vvv

g++ -O2 -I.tools/vvv/viewer/volren \
  .tools/vvv/viewer/tools/pvm2raw.cpp \
  .tools/vvv/viewer/volren/ddsbase.cpp \
  experiments/vrerrorhandler.cpp \
  -o .tools/vvv/bin/pvm2raw
.tools/vvv/bin/pvm2raw \
  data/public/Tumor-DCIS_448x448x160_uint8.pvm \
  data/public/breast_448x448x160_uint8.raw

.venv/bin/python - <<'PY'
import numpy as np
from skimage.io import imsave

source = "data/public/breast_448x448x160_uint8.raw"
target = "data/public/breast_448x448x160_uint8.tif"
volume = np.memmap(source, dtype=np.uint8, mode="r", shape=(160, 448, 448))
sparse = np.asarray(volume).copy()
sparse[sparse < 60] = 0
imsave(target, sparse, check_contrast=False)
print("nonzero ratio:", np.count_nonzero(sparse) / sparse.size)
PY

sha256sum \
  data/public/Tumor-DCIS_448x448x160_uint8.pvm \
  data/public/breast_448x448x160_uint8.raw \
  data/public/breast_448x448x160_uint8.tif
