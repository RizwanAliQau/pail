<div align="center">
  <img src="misc/logo.jpg" alt="PAI-Lib Logo" width="200" height="200" />
</div>

# Pseudo‑Anomaly Insertion Library (PAI‑Lib)

PAI‑Lib is a Python library for **synthetic anomaly generation** to support **self‑supervised anomaly detection and segmentation** in industrial vision. It provides a unified API to generate:
- a **pseudo‑anomalous image** `i_a`, and
- its corresponding **ground‑truth segmentation mask** `m_a`

from a **normal image** `i_n` (and optionally an **anomaly source image** `i_s`).

---

## Overview

In many industrial settings, real defect data is scarce or expensive to label. PAI‑Lib improves robustness by generating diverse synthetic defects using two families of methods:

### 1) Source‑Free Insertion (SFI)
Modify a normal image directly (no external source image required). Included techniques:
- CutPaste
- CutPaste Scar
- Simplex Noise
- Random Perturbation
- Hard‑Augment CutPaste

### 2) Source‑Based Insertion (SBI)
Insert anomalies using an external anomaly source image for more realistic defect appearance. Included techniques:
- Perlin‑Noise Pattern
- Superpixel Anomaly
- Perlin with ROI selection
- Random Augment CutPaste
- Fractal Anomaly Generation (FAG)

### 3) Hybrid (SFI + SBI)
Methods that can work with or without a source image:
- Affined Anomaly (AA / AAC / AAS)

---

## How it works (high level)

1. Load a **normal image** `i_n`
2. Optionally load an **anomaly source** `i_s` (required for SBI methods)
3. Choose an insertion method
4. Get:
   - `i_a`: pseudo‑anomalous image
   - `m_a`: binary/label mask of inserted region

---

## Installation

You can run PAI‑Lib using **Docker** or a **Conda environment**.

### Option A — Docker

```bash
docker build -t pail_test .
docker run -it -d pail_test
```

To run anomaly generation interactively, attach the container to VS Code (Dev Containers / Docker extension).

### Option B — Conda

```bash
conda create -n pai_lib --file requirement.txt
conda activate pai_lib
```

> If you also have a `requirement_pip.txt`, install it with:
> `pip install -r requirement_pip.txt`

---

## Quickstart

### 1) Import and initialize

```python
from pai import pai
import numpy as np
import os
import cv2

normal_img_path = "example_img/example_0.jpg"
anom_source_path = "anom_source_imgs"
height, width = 256, 256

# Load anomaly source image (optional; required for SBI)
try:
    anom_source_files = [os.path.join(anom_source_path, f) for f in os.listdir(anom_source_path)]
    anom_source_index = np.random.choice(len(anom_source_files), 1)[0]
    anom_source_img = cv2.imread(anom_source_files[anom_source_index])
    anom_source_img = cv2.resize(anom_source_img, (width, height))
except Exception:
    print("Anomaly source data not found; using a dummy image.")
    anom_source_img = np.zeros((height, width, 3), dtype=np.uint8)

# Load normal image
try:
    normal_image = cv2.imread(normal_img_path)
    normal_image = cv2.resize(normal_image, (width, height))
except Exception:
    print("Normal image not found; using a dummy image.")
    normal_image = np.ones((height, width, 3), dtype=np.uint8) * 255

anom_insertion = pai.Anomaly_Insertion()
```

> Tip: Most methods return `(aug_img, mask)`.

---

## Usage examples

## Source‑Free Methods (SFI)

### CutPaste Scar
**Paper:** CutPaste: Self‑Supervised Learning for Anomaly Detection and Localization  
https://arxiv.org/pdf/2104.04015.pdf

```python
aug_img, msk = anom_insertion.cutpaste_scar(normal_image)
```

### Simplex Noise Anomaly
**Paper:** Revisiting Reverse Distillation for Anomaly Detection (CVPR 2023)  
https://openaccess.thecvf.com/content/CVPR2023/papers/Tien_Revisiting_Reverse_Distillation_for_Anomaly_Detection_CVPR_2023_paper.pdf

```python
aug_img, msk = anom_insertion.simplex_noise_anomlay(normal_image)
```

### Random Perturbation
**Paper:** Collaborative Discrepancy Optimization for Reliable Image Anomaly Localization  
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10034849

```python
aug_img, msk = anom_insertion.random_perturbation(normal_image)
```

### Hard‑Augment CutPaste
**Paper:** ANOSEG: Anomaly Segmentation Network using Self‑Supervised Learning  
https://openreview.net/pdf?id=35-QqyfmjfP

```python
aug_img, msk = anom_insertion.hard_aug_cutpaste(normal_image)
```

### CutPaste
**Paper:** CutPaste: Self‑Supervised Learning for Anomaly Detection and Localization  
https://arxiv.org/pdf/2104.04015.pdf

```python
aug_img, msk = anom_insertion.cutpaste(normal_image)
```

---

## Source‑Based Methods (SBI)

> These methods require `anom_source_img`.

### Perlin Noise Pattern
**Paper:** DRÆM – A discriminatively trained reconstruction embedding for surface anomaly detection  
https://arxiv.org/pdf/2108.07610.pdf

```python
aug_img, msk = anom_insertion.perlin_noise_pattern(normal_image, anom_source_img=anom_source_img)
```

### Superpixel Anomaly
**Paper:** Two‑Stage Coarse‑to‑Fine Image Anomaly Segmentation and Detection Model  
https://www.sciencedirect.com/science/article/abs/pii/S0262885623001919

```python
aug_img, msk = anom_insertion.superpixel_anomaly(normal_image, anom_source_img=anom_source_img)
```

### Perlin with ROI selection
**Paper:** MemSeg: A semi‑supervised method for image surface defect detection using differences and commonalities  
https://arxiv.org/pdf/2205.00908.pdf

```python
aug_img, msk = anom_insertion.perlin_with_roi_anomaly(normal_image, anom_source_img=anom_source_img)
```

### Random Augmented CutPaste
**Paper:** Explicit Boundary Guided Semi‑Push‑Pull Contrastive Learning for Supervised Anomaly Detection  
https://arxiv.org/pdf/2207.01463.pdf

```python
aug_img, msk = anom_insertion.rand_augmented_cut_paste(normal_image, anom_source_img=anom_source_img)
```

### Fractal Anomaly Generation (FAG)
**Paper:** FractalAD: A simple industrial anomaly detection method using fractal anomaly generation  
https://arxiv.org/pdf/2301.12739.pdf

```python
aug_img, msk = anom_insertion.fract_aug(normal_image, anom_source_img=anom_source_img)
```

---

## Hybrid (SFI + SBI)

### Affined Anomaly
- **AA / AAC**: affined anomaly (with optional color changes)
- **AAS**: affined anomaly using anomaly source

```python
# AA / AAC
aug_img, msk = anom_insertion.affined_anomlay(normal_image)

# AAS
aug_img, msk = anom_insertion.affined_anomlay(normal_image, anom_source_img=anom_source_img)
```

---

## Examples

![examples_pai](https://github.com/user-attachments/assets/57257328-65af-4d42-a173-eab9f3be277b)

---

## Notes / Troubleshooting

- If your **anomaly source directory is empty**, SBI methods will fall back only if you provide a dummy `anom_source_img`.
- Ensure images are loaded as `H×W×3` (BGR if using OpenCV).

---

## Citation

If this repository is used in academic work, please consider citing the relevant papers corresponding to the methods you use.

---

## License

Add your license here (e.g., MIT, Apache‑2.0), or link to `LICENSE`.
