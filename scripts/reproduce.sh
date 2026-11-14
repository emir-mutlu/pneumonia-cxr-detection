#!/usr/bin/env bash
# Reproduce every number and figure in the README (about 2.5 h on an RTX 3060 laptop GPU).
set -euo pipefail

pneumo download all          # ~5.2 GB: Kermany (Hugging Face mirror) + RSNA (official RSNA site)
pneumo prepare all           # decode, resize to 256 px, patient-level splits

for cfg in configs/*.yaml; do
  pneumo train "$cfg"
done

for run in runs/*/; do
  pneumo evaluate "$run"
  pneumo explain "$run" --figure "${run%/}/gradcam_examples.png"
done

FEATURED=multi_convnext_tiny
pneumo explain "runs/$FEATURED" --figure reports/figures/gradcam_examples.png
pneumo compare-cam runs/kermany_simple_cnn runs/kermany_convnext_tiny "runs/$FEATURED" \
  --labels "Simple CNN (2020 baseline),ConvNeXt-T · Kermany,ConvNeXt-T · Kermany+RSNA"
pneumo report --featured "$FEATURED"
pneumo export "runs/$FEATURED" --out-dir models
