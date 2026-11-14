---
title: Pneumonia CXR Assistant
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Chest X-ray pneumonia AI with Grad-CAM, runs in your browser
models:
  - emir-mutlu/pneumonia-cxr-convnext
tags:
  - medical-imaging
  - chest-x-ray
  - pneumonia
  - grad-cam
  - explainable-ai
  - onnx
---

# Chest X-ray Pneumonia Assistant

Research prototype of an AI decision-support viewer for pneumonia on frontal chest X-rays.
**Everything runs in your browser** with ONNX Runtime Web: images are never uploaded to a server.

Upload a PNG or JPEG image, or pick one of the sample studies, to get:

* a **calibrated probability** of pneumonia with the decision threshold,
* a **Grad-CAM heatmap** and the lung zone with the strongest evidence. Grad-CAM is computed
  in closed form inside the ONNX graph, so no backward pass is needed in the browser,
* the model's **validated performance** on two independent hospitals.

| Held-out test set | AUROC (95 % CI) | Sensitivity | Specificity |
|---|---|---:|---:|
| Kermany, pediatric (Guangzhou) | 0.990 (0.982–0.995) | 0.995 | 0.838 |
| RSNA, adult (NIH) | 0.890 (0.877–0.903) | 0.691 | 0.878 |

Model: [emir-mutlu/pneumonia-cxr-convnext](https://huggingface.co/emir-mutlu/pneumonia-cxr-convnext),
ConvNeXt-Tiny fine-tuned on 24,928 chest X-rays with patient-level splits. The first visit
downloads the model once (~57 MB).

**Not a medical device.** Do not use for diagnosis, triage or treatment decisions.

Code, training pipeline and evaluation:
[github.com/emir-mutlu/pneumonia-cxr-detection](https://github.com/emir-mutlu/pneumonia-cxr-detection)

Sample images: Kermany et al. 2018 (CC BY 4.0); RSNA-STR Pneumonia Detection Challenge 2018,
images provided by the NIH Clinical Center. See `examples/README.md` for attribution.
