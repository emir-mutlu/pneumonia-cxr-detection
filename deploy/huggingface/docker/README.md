---
title: Pneumonia CXR Assistant
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Chest X-ray pneumonia AI with Grad-CAM (research only)
tags:
  - medical-imaging
  - chest-x-ray
  - pneumonia
  - grad-cam
  - explainable-ai
  - pytorch
---

# Chest X-ray Pneumonia Assistant

Research prototype of an AI decision-support viewer for pneumonia on frontal chest X-rays.
Upload a PNG, JPEG or DICOM image, or pick one of the sample studies, to get:

* a **calibrated probability** of pneumonia with the decision threshold,
* a **Grad-CAM heatmap** and the lung zone with the strongest evidence,
* the model's **validated performance** on two independent hospitals.

| Held-out test set | AUROC (95 % CI) | Sensitivity | Specificity |
|---|---|---:|---:|
| Kermany, pediatric (Guangzhou) | 0.990 (0.982–0.995) | 0.995 | 0.838 |
| RSNA, adult (NIH) | 0.890 (0.877–0.903) | 0.691 | 0.878 |

Model: ConvNeXt-Tiny (ImageNet-22k pre-training) fine-tuned on 24,928 chest X-rays with
patient-level splits. Runs on the free CPU hardware; the first analysis after the Space wakes
up takes a few seconds.

**Not a medical device.** Do not use for diagnosis, triage or treatment decisions.

Code, training pipeline, evaluation and model card:
[github.com/emir-mutlu/pneumonia-cxr-detection](https://github.com/emir-mutlu/pneumonia-cxr-detection)

Data: Kermany et al. 2018 (CC BY 4.0); RSNA-STR Pneumonia Detection Challenge 2018, images
provided by the NIH Clinical Center. See the repository's `docs/DATASETS.md` for full citations.
