# Datasets

Two public chest X-ray datasets from different hospitals, countries and age groups are used
for training. A third, access-controlled dataset from a third country (VinDr-PCXR, see
[below](#third-hospital-vindr-pcxr-external-test-only)) is used only as an external test set.
None is redistributed in this repository; `pneumo download` fetches the public ones and
`pneumo prepare` builds the cached arrays and splits.

| | Kermany (Guangzhou) | RSNA Pneumonia Challenge 2018 |
|---|---|---|
| Population | Children aged 1–5, Guangzhou Women and Children's Medical Center, China | Adults (median age 49), NIH Clinical Center, USA |
| Images | 5,856 JPEG | 29,684 DICOM (1024 × 1024) |
| Patients | 3,288 (from file names) | 12,233 (recovered via the NIH mapping) |
| Labels | Normal / Pneumonia (bacterial, viral) | Lung Opacity (with boxes) / No Lung Opacity–Not Normal / Normal |
| Positive class here | Pneumonia (73 %) | Lung Opacity (24 %) |
| Extra metadata | – | sex, age, view position (PA/AP), radiologist bounding boxes |
| Size on disk | 1.2 GB | 4.0 GB |
| License | CC BY 4.0 | RSNA terms of use (research, education, commercial use with attribution) |
| Source | [Mendeley Data V2](https://data.mendeley.com/datasets/rscbjbr9sj/2), mirrored at [hf-vision/chest-xray-pneumonia](https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia) | [RSNA download site](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018) |

## Splits (patient-level, stratified, seed 42)

**Kermany** – the official 16-image validation set is too small to select a model or a
threshold, so the official train + validation images are pooled and re-split *by patient*
(85 / 15). The official test set is kept unchanged.

| split | images | patients | pneumonia | normal |
|---|---:|---:|---:|---:|
| train | 4,482 | 2,432 | 3,333 | 1,149 |
| val | 750 | 429 | 550 | 200 |
| test (official) | 624 | 427 | 390 | 234 |

**RSNA** – 70 / 10 / 20 split on the NIH patient id.

| split | images | patients | lung opacity | not normal | normal |
|---|---:|---:|---:|---:|---:|
| train | 20,446 | 8,563 | 4,796 | 8,732 | 6,918 |
| val | 3,105 | 1,224 | 762 | 1,363 | 980 |
| test | 6,133 | 2,446 | 1,548 | 2,693 | 1,892 |

## Data-quality findings

1. **RSNA "patientId" is an image id, not a patient id.** Joining the challenge data with
   RSNA's official mapping to the NIH ChestX-ray14 file names (`00012773_000.png` →
   patient `00012773`) shows that the 29,684 images come from only **12,233 patients**;
   4,781 patients have more than one image (up to 74), and **75 % of all images belong to
   such patients**. A split on the Kaggle `patientId` therefore puts the same person in
   train and test. All RSNA splits here use the NIH patient id.
2. **Kermany has 2.6 images per pneumonia patient on average** (up to 31). Image-level
   random splits, which are common in public notebooks, leak patients across splits.
3. **Kermany official train/test share 170 `personN` ids**, but images with the same id are
   no more similar to each other than random pairs (mean normalised cross-correlation 0.520
   vs 0.517), so the numbering restarts per folder rather than indicating real leakage. The
   official test set is therefore kept as is.
4. **Many Hugging Face "pneumonia" datasets are copies of Kermany** under new names (for
   example sets advertised as African or multimodal reuse the `IM-xxxx` / `personN` file
   names). Combining such sets with Kermany would silently duplicate test images into
   training, so only the two independent sources above are used.
5. **RSNA age has impossible values** (max 412 years); ages above 110 are treated as unknown
   in subgroup analysis.

## Third hospital: VinDr-PCXR (external test only)

| | VinDr-PCXR v1.0.0 |
|---|---|
| Population | Children under 10 (median age 1 year), Phu Tho Obstetric & Pediatric Hospital, Vietnam, 2020–2021 |
| Images | 9,125 DICOM (official split: 7,728 train, 1,397 test); no patient identifiers |
| Labels | One pediatric radiologist per image; 15 image-level columns (No finding + 14 diagnoses) |
| Positive class here | Pneumonia, Broncho-pneumonia or Pleuro-pneumonia (12 % of the test set) |
| Negatives | No finding, or only other diagnoses (bronchitis, bronchiolitis, …) as hard negatives |
| X-ray systems | Manufacturer tags removed. 8,824 images (96.7 %) come from a main system storing inverted pixels (MONOCHROME1, 12-bit, mostly uncompressed or JPEG Lossless); 301 (3.3 %) come from other systems (MONOCHROME2, 8–14-bit, several pixel spacings). Pneumonia share: 11 % vs 44 %. Recorded per image as `photometric`, `transfer_syntax`, `bits_stored`. |
| Size on disk | 23.4 GB ZIP, 34.1 GB extracted |
| License | PhysioNet Restricted Health Data License 1.5.0: sign the data use agreement; research use only; never redistributed |
| Source | [physionet.org/content/vindr-pcxr](https://physionet.org/content/vindr-pcxr/1.0.0/) |

Used **only as an external test set**: no model was trained, selected, calibrated or
thresholded on it. The official test split is kept; 15 % of the official training split is held
out as validation for possible future training. Without patient ids, every image is treated as
its own patient (image-level CIs).

| split | images | pneumonia | no finding | other diagnoses only |
|---|---:|---:|---:|---:|
| train | 6,569 | 782 | 4,376 | 1,411 |
| val | 1,159 | 138 | 767 | 254 |
| test (official) | 1,397 | 170 | 907 | 320 |

How to get it: sign the DUA on the project page, use "Download the ZIP file" in its Files
section (PhysioNet's server delivers ≈0.6 MB/s, so ≈11 h), then run
`pneumo physionet-import <zip> --project vindr-pcxr`, which extracts the archive safely and checks
all 9,130 files against `SHA256SUMS.txt`, and `pneumo prepare vindr_pcxr`. Only aggregate metrics
from this dataset appear in this repository.

## Preprocessing

Every image is converted to 8-bit grayscale, resized to 256 × 256 with anti-aliased bilinear
interpolation and stored in a memory-mapped `uint8` array (`data/processed/<source>_images.npy`).
For DICOM, the modality and stored display window (VOI LUT) are applied when present, as a
viewer would show the image, and MONOCHROME1 images are inverted. Other 16-bit data is min-max
scaled. VinDr-PCXR's 12-bit images carry such windows; part of them are JPEG-Lossless compressed
and need the `pylibjpeg` plugins (`pip install -e ".[data]"`). Models see 224 × 224 inputs in
[0, 1]; normalisation for each backbone happens inside the model, so the exported ONNX graph and
the API use exactly the same input.

## Required attribution

* Kermany D., Zhang K., Goldbaum M. (2018). *Labeled Optical Coherence Tomography (OCT) and
  Chest X-Ray Images for Classification.* Mendeley Data, V2. doi:10.17632/rscbjbr9sj.2 —
  and Kermany D. S. et al. (2018). *Identifying Medical Diagnoses and Treatable Diseases by
  Image-Based Deep Learning.* Cell 172(5), 1122–1131.
* NIH Clinical Center is the data provider of the RSNA images
  ([NIH download site](https://nihcc.app.box.com/v/ChestXray-NIHCC)). Wang X. et al. (2017).
  *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised
  Classification and Localization of Common Thorax Diseases.* IEEE CVPR, 3462–3471.
* Shih G. et al. (2019). *Augmenting the National Institutes of Health Chest Radiograph
  Dataset with Expert Annotations of Possible Pneumonia.* Radiology: Artificial
  Intelligence 1(1). doi:10.1148/ryai.2019180041 —
  [RSNA download site](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018).
* Pham H. H., Tran T. T., Nguyen H. Q. (2022). *VinDr-PCXR: An open, large-scale pediatric chest
  X-ray dataset for interpretation of common thoracic diseases* (version 1.0.0). PhysioNet.
  doi:10.13026/k8qc-na36 — and the standard PhysioNet citation: Pollard T. et al. (2026).
  *PhysioNet as a global platform for biomedical research.* Nature Health.
  doi:10.1038/s44360-026-00096-z.
* Chest X-ray pre-trained weights: Cohen J. P. et al. (2022). *TorchXRayVision: A library of
  chest X-ray datasets and models.* MIDL. CheXpert weights: Irvin J. et al. (2019). AAAI.
