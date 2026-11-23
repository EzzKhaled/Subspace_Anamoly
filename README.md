# SubspaceAD: Training-Free Few-Shot Industrial Anomaly Detection

A re-implementation and evaluation of **SubspaceAD** on **MVTec-AD** and **VisA**.
SubspaceAD detects defects without training any model: it extracts frozen DINOv2 patch
features, fits a PCA subspace on normal patches, and scores anything the subspace cannot
reconstruct as anomalous.

This repository is part of our graduation project on zero/few-shot industrial anomaly
detection (Computer and Communication Engineering, Alexandria University).

## How it works

1. **Backbone:** frozen DINOv2-giant with registers. A 672x672 input gives a 48x48 grid of
   patches, each a 1536-dim feature.
2. **Layers:** average the hidden states of layers -12 to -18; drop the CLS and register tokens.
3. **PCA:** fit on normal patches only. Keep the fewest components that explain 99% of the
   variance (k is about 900 to 1040 of 1536).
4. **Residual score:** for a patch feature `x`, `s(x) = ||(I - C C^T)(x - mu)||^2`, where `mu`
   is the mean normal feature and `C` the top-k principal directions.
5. **Maps and image score:** upsample the patch scores to 672x672, apply a Gaussian blur
   (3x3, sigma 4). The image score is the mean of the top 1% pixels.

There is no memory bank, no nearest-neighbour search and no gradient training. The fitted
model is just `mu` and `C`.

### Two evaluation regimes

| Regime | PCA is fitted on | Notes |
|---|---|---|
| **R** (few-shot) | k = 1, 2 or 4 normal training images, plus 30 random rotations each (none for `transistor`) | 5 seeds, mean and std reported |
| **T** (batched 0-shot) | the whole unlabeled test set of the category | no normal reference images, but the model sees the test batch |

## Results

Image AUROC (%). Mine is a 5-seed mean +- std. Official values are read from the official
README and come from a single seed.

| Dataset | | 1-shot | 2-shot | 4-shot | 0-shot (T) |
|---|---|---|---|---|---|
| MVTec-AD | Mine | 95.84 +- 0.60 | 96.55 +- 0.79 | 97.99 +- 0.32 | 96.55 |
| MVTec-AD | Official | 97.1 | 97.5 | 98.0 | 96.6 |
| VisA | Mine | 92.74 +- 0.70 | 93.85 +- 0.54 | 94.75 +- 0.21 | 94.08 |
| VisA | Official | 93.2 | 93.8 | 94.7 | 94.1 |

Other metrics:

| Setting | Img AP | TPR@1%FPR | TPR@5%FPR | Pixel AUROC | Pixel AP | AUPRO@30 |
|---|---|---|---|---|---|---|
| MVTec R 4-shot | 98.97 | 85.07 | 91.24 | 97.90 | 61.82 | 95.13 |
| MVTec T | 98.11 | 67.40 | 85.11 | 95.84 | 52.60 | 93.38 |
| VisA R 4-shot | 94.30 | 49.05 | 74.83 | 98.46 | 41.37 | 96.17 |
| VisA T | 94.60 | 46.67 | 71.17 | 98.45 | 39.13 | 96.10 |

Key observations:

- 4-shot and 0-shot results match the official numbers within 0.05 points. The gap is
  largest at 1-shot (-1.26 on MVTec, -0.46 on VisA).
- More shots help and stabilise runs: from 1 to 4 shots, +2.15 (MVTec) and +2.01 (VisA)
  image AUROC, and the seed std drops from 0.60 to 0.32 and from 0.70 to 0.21.
- AUROC is high but the operating point is weak on VisA: only 49.05% of defects are caught
  at 1% false positives.
- Weakest categories: MVTec `screw` (91.7), `cable` (93.3), `capsule` (93.6) and
  `transistor` (AUPRO@30 only 72.5); VisA `macaroni2` (85.5) and `pcb3` (88.8).
- Raw score scales differ more than 5x between categories, so each category needs its own
  threshold (see `figures/fig4_regime_scale_mismatch.png`).
- Re-evaluating pixel metrics at 672x672 instead of 256x256 changes pixel AUROC by at most
  0.02 points.

## Differences from the official implementation

- The maths is the same: backbone, layers, PCA (float64 covariance, `eigh`, 99% variance),
  score and map.
- fp32 weights with fp16 autocast (an fp32 option is available).
- Five seeds instead of one.
- PCA is fitted from summed statistics (sum x, sum x x^T, n), which can be cached.
- Transformer blocks after the deepest layer used are removed (same features, checked).
- Pixel evaluation is done at 256x256 first, then re-checked at 672x672.
- Extras: checkpointed runs, calibration statistics and VLM ROI proposals.

## Triage for a VLM verifier

On all 3,887 test images (MVTec + VisA, R 4-shot, seed 0), robust z-scores split the images into:

| Band | Images | Anomalous | Normal |
|---|---|---|---|
| high_conf_anomaly | 1,711 | 1,684 | 27 |
| borderline_send_to_vlm | 645 | 520 | 125 |
| auto_normal | 1,531 | 254 | 1,277 |

Only the borderline band (16.6% of images) would be sent to a vision-language model.
The 254 anomalies in `auto_normal` are the cost: subtle defects the band misses.
This is a single-seed result.

## Repository contents

```
notebooks/
  subspacead_final.ipynb                                   main implementation and experiments
  subspacead_official_visual_eval672_and_impl.ipynb        official-style 672 evaluation and visuals
results/
  all_runs.csv                                             every run (dataset, regime, shots, seed, category, metrics)
  eval672_per_run.csv, eval256_vs_eval672_comparison.csv   pixel evaluation at 672 vs 256
  mvtec_regimeR_*.csv, mvtec_regimeT_*.csv                 MVTec summaries and per-category results
  visa_regimeR_*.csv, visa_regimeT_*.csv                   VisA summaries and per-category results
  per_item_all.parquet                                     per-image scores and labels
  subspacead_calibration_stats.csv                         normal quantiles, TPR at fixed FPR, separation gaps
  subspacead_vlm_roi_proposals.csv                         peak ROI boxes and triage bands
figures/
  fig1_fewshot_scaling.png
  fig2_mvtec_per_category_R4_vs_T.png, fig2_visa_per_category_R4_vs_T.png
  fig3_score_distributions_calibration.png, fig4_regime_scale_mismatch.png
  fig5_vlm_roi_crop_preview.png
heatmaps/
  heatmap_mvtec_{bottle,capsule,screw,transistor}.png
  heatmap_visa_{cashew,macaroni2,pcb3}.png
```

## Reading a heatmap sheet

Columns: original image, ground-truth defect, Regime R map (4-shot, seed 0), Regime T map
(0-shot). Blue means the normal subspace explains the patch well; red means a large residual.
Colours are scaled per category. Rows 1 to 3 are example defects, row 4 is the hardest defect
(lowest score), row 5 is the hardest normal image (highest score). R and T scores are on
different scales, so compare within one column only.

![Few-shot scaling](figures/fig1_fewshot_scaling.png)
![Transistor heatmaps](heatmaps/heatmap_mvtec_transistor.png)

## Running the code

1. Download the datasets:
   - MVTec-AD: https://www.mvtec.com/company/research/datasets/mvtec-ad
   - VisA: https://github.com/amazon-science/spot-diff
2. Open `notebooks/subspacead_final.ipynb` on a GPU runtime (the notebooks were run on Kaggle).
3. Set the dataset paths in the first cells, then run the cells in order. Package imports are
   listed in the first code cell of each notebook.

## Limitations

- Official numbers are one seed, read from the official README, so small gaps are not conclusive.
- Regime T sees the unlabeled test set, so it is not zero-shot in the strict sense.
- The residual is local: it misses logical defects such as a misplaced part (`transistor`).
- Very small defects (screw tip, cashew specks) are often missed.
- Triage and calibration numbers come from one seed.

## Planned improvements

Per-category conformal thresholds; hybrid R + T fitting; per-category augmentation and rank;
context features for logical defects; tiled high-resolution scoring; other backbones
(DINOv3, RADIO); robust PCA for Regime T; a VLM verifier on borderline ROI crops.



## Author

- Ezz Eldien Khaled - Ahmed Ayman
- Computer and Communication Engineering, Alexandria University.
