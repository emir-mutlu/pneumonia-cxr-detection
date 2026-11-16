# Model card – pneumonia-cxr (released model in `models/`)

Also published on Hugging Face as [emir-mutlu/pneumonia-cxr-convnext](https://huggingface.co/emir-mutlu/pneumonia-cxr-convnext)
(ONNX with a closed-form Grad-CAM output) and runnable in the browser in the
[live demo](https://huggingface.co/spaces/emir-mutlu/pneumonia-cxr-detection).

| | |
|---|---|
| Model | ConvNeXt-Tiny (`convnext_tiny.fb_in22k_ft_in1k`, ImageNet-22k pre-training), single-logit head |
| Parameters | 27.8 M |
| Training data | Kermany pediatric (4,482 train images) + RSNA adult (20,446 train images), patient-level splits |
| Training | AdamW (lr 1e-4, wd 0.05), warm-up + cosine, bf16, drop-path 0.1, batch 32, 224 px; best epoch 5 of 8 (early stopping on mean per-source validation AUROC) |
| Calibration | temperature scaling, T = 1.061 (fitted on validation) |
| Decision threshold | 0.576 (Youden's J on pooled validation data) |
| Files | `model.onnx` (56.7 MB, opset 18, dynamic batch, calibration included), `model.pt` (PyTorch checkpoint for Grad-CAM), `model_meta.json` |
| Weight storage | float16 on disk, cast to float32 at load; test AUROC identical to the float32 model (0.9900 / 0.8902 vs 0.9899 / 0.8901) |
| Latency | ≈30 ms per image, ONNX Runtime, CPU, batch 1 (laptop) |
| Version | 2.0.0 (2026). Version 1 (2020) was a Keras CNN trained on Kermany only |

## Performance (held-out test sets)

| Test set | Images (patients) | AUROC (95 % CI) | Sensitivity | Specificity | PPV | NPV | ECE |
|---|---:|---|---:|---:|---:|---:|---:|
| Kermany, pediatric | 624 (427) | 0.990 (0.982–0.995) | 0.995 | 0.838 | 0.911 | 0.990 | 0.068 |
| RSNA, adult | 6,133 (2,446) | 0.890 (0.877–0.903) | 0.691 | 0.878 | 0.656 | 0.894 | 0.068 |
| VinDr-PCXR, pediatric, Vietnam (external) | 1,397 (n/a) | 0.746 (0.702–0.789) | 0.071 | 0.993 | 0.571 | 0.885 | 0.083 |

The third row is a hospital the model never saw: at the stored threshold it finds only 7 % of
pneumonia cases there, and reaching 80 % sensitivity would need a threshold of 0.012
(specificity 0.50). CIs are patient-clustered bootstrap intervals (1,000 resamples; image-level
for VinDr-PCXR, which has no patient ids).

A variant trained on all three hospitals with a threshold per site reaches 0.990 / 0.886 / 0.833
(VinDr-PCXR sensitivity 0.735, specificity 0.759). It is not released, because its weights are
derived from PhysioNet restricted data; see the README. Subgroup results (sex, age, view)
and the comparison with seven other models are in the [README](README.md) and
[reports/results.md](reports/results.md).

Grad-CAM localisation on the 1,548 RSNA test pneumonia images: the heatmap peak falls inside a
radiologist box in 76.5 % of images (random point: 12.4 %).

## Intended use

* **Intended**: research, teaching and demonstrating a rigorous medical-imaging ML workflow.
* **Not intended**: diagnosis, triage, screening or any clinical decision. The model is not a
  medical device and has not been prospectively validated or reviewed by a regulator.

## Inputs and outputs

* Input: one frontal (PA or AP) chest radiograph, PNG / JPEG / DICOM. It is converted to 8-bit
  grayscale, resized to 256 px and then 224 px, and scaled to [0, 1].
* Output: calibrated probability of pneumonia (Kermany) / lung opacity suggestive of pneumonia
  (RSNA), and a binary decision at the stored threshold.
* Not handled: lateral views, CT, non-chest images. There is no out-of-distribution detector,
  so the model will return a probability for any image.

## Training data

Kermany et al. pediatric chest X-rays (Guangzhou, China; ages 1–5) and RSNA Pneumonia Detection
Challenge 2018 (NIH Clinical Center, USA; adults). Patient-level splits; see
[docs/DATASETS.md](docs/DATASETS.md).

## Ethical considerations and limitations

* **Population shift**: both training sources are from two hospitals, and performance elsewhere
  can be much lower. The Kermany-only models lose about 0.15–0.21 AUROC on RSNA, and adult-only
  models score 0.07–0.11 lower on children than models trained on children. On a third hospital
  (VinDr-PCXR, Vietnamese children, median age 1 year) this model reaches AUROC 0.746, and only
  0.681 for infants under one year.
* **Threshold transfer**: a threshold tuned on one hospital can give very different sensitivity
  / specificity on another (see the external results in `reports/results.md`). Any new site
  needs local validation and its own operating point: `pneumo calibrate` measures the model on a
  site's own labelled X-rays and writes a site profile (Platt recalibration + local threshold)
  that `pneumo predict --site-profile` applies. It does not replace a clinical study.
* **Label definitions differ** between sources (clinical pneumonia vs. radiographic lung
  opacity) and contain reader noise.
* **Shortcut risk**: AP / portable acquisitions are more common in sick patients; subgroup
  metrics by view are reported for this reason.
* **Explanations**: Grad-CAM is a coarse, post-hoc saliency map. Its localisation was measured
  against radiologist boxes, but it does not prove the model reasons like a radiologist.

## Reproducibility

`pneumo download all && pneumo prepare all && pneumo train configs/<config>.yaml`, seed 42,
PyTorch 2.11, timm 1.0, single RTX 3060 Laptop GPU. All metrics in `reports/` are generated by
`pneumo evaluate` / `pneumo report` from the saved runs.
