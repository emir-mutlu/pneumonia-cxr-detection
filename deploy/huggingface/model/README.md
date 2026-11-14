---
license: other
license_name: research-and-education-only
license_link: https://github.com/emir-mutlu/pneumonia-cxr-detection/blob/main/LICENSE
library_name: onnx
pipeline_tag: image-classification
datasets:
  - hf-vision/chest-xray-pneumonia
metrics:
  - roc_auc
tags:
  - medical-imaging
  - chest-x-ray
  - pneumonia
  - grad-cam
  - explainable-ai
  - convnext
  - onnx
---

# Pneumonia detection from chest X-rays (ConvNeXt-Tiny, ONNX)

Binary classifier (pneumonia vs. no pneumonia) for frontal chest radiographs. The ONNX graph
includes input normalisation, temperature scaling and a closed-form **Grad-CAM**, so a single
forward pass returns both the calibrated probability and the explanation map. It runs in
ONNX Runtime on a server or in the browser.

**Live demo (runs in your browser):** [emir-mutlu/pneumonia-cxr-detection](https://huggingface.co/spaces/emir-mutlu/pneumonia-cxr-detection)
**Code, training and evaluation:** [github.com/emir-mutlu/pneumonia-cxr-detection](https://github.com/emir-mutlu/pneumonia-cxr-detection)

> **Not a medical device.** Research and education only; not for diagnosis, triage or treatment.

## Files

| File | Content |
|---|---|
| `model_browser.onnx` | inputs `image` float32 (N, 1, 224, 224) in [0, 1]; outputs `probability` (N,) and `cam` (N, 224, 224) in [0, 1] |
| `model_meta.json` | decision threshold (0.576), temperature, pre-processing sizes, test metrics |

Weights are stored in float16 and cast to float32 when the session is created. All computation
runs in float32, and test AUROC is identical to the float32 model.

## Pre-processing

Convert to 8-bit grayscale, then resize the whole image (no crop) to 256 × 256 and to
224 × 224 with anti-aliased bilinear interpolation (Pillow `BILINEAR`), and scale to [0, 1].
Predict pneumonia when `probability >= threshold` from `model_meta.json`.

```python
import json, numpy as np, onnxruntime as ort
from PIL import Image

meta = json.load(open("model_meta.json"))
img = Image.open("cxr.png").convert("L").resize((256, 256), Image.BILINEAR).resize((224, 224), Image.BILINEAR)
x = (np.asarray(img, dtype=np.float32) / 255)[None, None]
prob, cam = ort.InferenceSession("model_browser.onnx").run(None, {"image": x})
print(prob[0], prob[0] >= meta["threshold"])
```

## Training data and evaluation

Fine-tuned from ConvNeXt-Tiny (ImageNet-22k) on 24,928 images. Two independent sources were
combined, with patient-level splits:

* Kermany et al. pediatric chest X-rays (Guangzhou, CC BY 4.0).
* RSNA Pneumonia Detection Challenge 2018 adult chest X-rays (NIH Clinical Center). This set's
  images map to only 12,233 patients; the splits use the true NIH patient id.

| Held-out test set | Images (patients) | AUROC (95 % CI) | Sensitivity | Specificity | ECE |
|---|---:|---|---:|---:|---:|
| Kermany, pediatric | 624 (427) | 0.990 (0.982–0.995) | 0.995 | 0.838 | 0.068 |
| RSNA, adult | 6,133 (2,446) | 0.890 (0.877–0.903) | 0.691 | 0.878 | 0.068 |

CIs come from a patient-clustered bootstrap. Grad-CAM peaks fall inside radiologist boxes for
76.5 % of RSNA test pneumonia images (random point: 12.4 %).

## Limitations

* Two hospitals only. Performance elsewhere is unknown; models trained on one hospital lost
  0.15–0.21 AUROC on the other.
* One global threshold: sensitivity is much lower in adults (0.69) than in children (0.995).
  Lower still in patients aged 60+ and on PA views.
* AP (bedside) views carry more disease; part of the adult AUROC comes from view-related cues.
* Labels differ between sources (clinical pneumonia vs. radiographic lung opacity).

Full model card: [MODEL_CARD.md](https://github.com/emir-mutlu/pneumonia-cxr-detection/blob/main/MODEL_CARD.md).

## Citations

Kermany et al., Cell 2018 (doi:10.17632/rscbjbr9sj.2); Wang et al., ChestX-ray8, CVPR 2017;
Shih et al., Radiology: AI 2019 (doi:10.1148/ryai.2019180041); Liu et al., ConvNeXt, CVPR 2022.
