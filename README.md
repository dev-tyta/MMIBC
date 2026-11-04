# MMIBC v2: explainable multimodal breast-cancer imaging

Code for **"MMIBC: An Explainable Multimodal Imaging Framework for Breast Cancer Diagnosis Using Mammography and Ultrasonography"**.
This branch (`revision/v2`) is a ground-up rebuild made in response to the reviews of the first submission. The old code is kept untouched in [`legacy/`](legacy/).

> **Status.** The pipeline, the audits and the test suite run end to end on synthetic data (`mmibc smoke`), and the first real-data runs on Colab exist (see *What the first real runs showed*). The numbers in the rejected submission (e.g. 84 % accuracy) came from the old code, whose problems are listed below, and should not be quoted as results of this repository. **The mitigations added after the first real runs (calibration, EMA, layer-wise lr decay, higher-resolution mammograms, unimodal baselines on the pool, seeds) are covered by unit and end-to-end tests on synthetic data only: their effect on the real datasets has not been measured yet. The CMMD path was run end to end on fake DICOMs in the real layout and its DICOM reader on 4 real CMMD files (one study); `prepare cmmd` has not yet run on the full 1,775-patient download.**

![Architecture of MMIBC v2 with the corrections marked](paper/figures/pdf/fig_architecture.png)

## Quick start (Google Colab)

Open [`notebooks/colab_quickstart.ipynb`](notebooks/colab_quickstart.ipynb) and run it top to bottom, or from a terminal:

```bash
git clone -b revision/v2 https://github.com/dev-tyta/MMIBC.git && cd MMIBC
pip install -e ".[download,dicom]"   # kagglehub + requests for downloads; pydicom + openpyxl for CMMD
mmibc smoke                       # whole pipeline on synthetic data, CPU, under a minute
mmibc experiments --list              # EVERYTHING at once: shows what would run on the datasets you have, what is finished, what is blocked
mmibc download vindr_images vindr_annotations busi
mmibc prepare all -c mammo_us_synthetic
mmibc audit   -c mammo_us_synthetic          # prints every data check; trains nothing
mmibc run     -c vindr_multiview             # native CC<->MLO pairs: prepare -> train -> evaluate -> explain -> figures
mmibc run     -c mammo_us_synthetic --set task.pair_mode=label_matched
mmibc suite ladder                           # fusion ablation on identical frozen test pairs
mmibc suite pool_ladder                      # same, on the mammogram + ultrasound pool, WITH both unimodal baselines
mmibc suite recipe                           # training mitigations, one at a time, on one VinDr split
mmibc suite recipe --seeds 42 43 44          # repeat on the same partition: mean +- sd and the seed-ensemble AUROC
mmibc prepare cmmd -c cmmd_multiview         # CMMD: DICOM -> PNG, biopsy labels, patient-level split (see "CMMD" below)
mmibc run -c cmmd_multiview --set data.cmmd.subsets=[D1] run_name=cmmd_mv_d1
mmibc suite backbones                        # DINOv2 / +registers / DINOv3 / CNNs; DINOv3 rows are skipped until HF_TOKEN exists
mmibc suite backbones_cmmd --set data.cmmd.subsets=[D1]
mmibc figures --template lncs
mmibc experiments                            # all of the above on every available dataset in one go (resumable); see "Everything at once"
```

Persistent files (processed images, splits, checkpoints, weights, figures) go to Google Drive under `MyDrive/MMIBC`; raw downloads are saved there too (`raw/`; set `paths.raw_on_drive=false` for Colab's temporary disk if your Drive is small). Everything is resumable: re-run a command (or a whole suite) after a disconnect. Code is cloned from GitHub into `/content` each session; do not run it from a copy kept on Drive.

## Everything at once: `mmibc experiments`

```bash
mmibc experiments --list             # the plan: datasets found, runs already finished, tracks blocked and why (prepares and trains nothing)
mmibc experiments                    # prepare what is downloaded, then train + evaluate every track; summary in runs/experiments_all/summary.md
mmibc experiments --only cmmd        # a group (vindr | robust | cmmd | external | suites) or named tracks
mmibc experiments --suites           # also the fusion ladders and the backbone comparison (many more runs)
mmibc experiments --xai              # also the explanation metrics for every run
mmibc experiments --set training.epochs=1 --only vindr    # a quick end-to-end check of the machinery before the long run
```

The programme is `configs/programs/all.yaml`: a list of *tracks*, each a config plus overrides (a suite is a track too). Nothing in it says which datasets you own: each track's needs are read from its configuration (mammogram source, and every ultrasound source when it pairs mammograms with ultrasound), checked against `raw/` and `processed/`, and prepared once up front (resumable; the ultrasound pool is built from the union of the sources the selected tracks need).

* A track whose dataset is **not downloaded is SKIPPED with the reason**; one that **fails is FAILED** with its traceback in `FAILED_<track>.txt`; neither stops the others. Download or fix the cause and call the same command again: **finished tracks are reused** (same configuration fingerprint), only the missing ones train. A Colab disconnect costs at most the run in progress, which resumes from its checkpoint.
* `runs/experiments_all/` holds `plan.csv` (before), `status.csv` (rewritten after every track), `summary.csv` / `summary.md` (one row per track: test AUROC with its DeLong interval, external-test AUROC, AUROC per CMMD subset, minutes) and the failure files. The exit code is 1 if a track failed or nothing finished, 2 if interrupted.
* The tracks keep the run names already on Drive (`multiview_xattn_s42_dinov2_base`, `pool_busi_busbra_*`), so those results are picked up, not retrained.
* **VinDr and CMMD are never pooled into one training set.** The labels mean different things (BI-RADS suspicion with an official split, versus biopsy with a split carved by patient), so their numbers are separate results.
* A pool or external-test run now trains only on the ultrasound sources its configuration lists (`data.us_sources`); before, the processed pool was shared and a pool run made after `prepare us -c mammo_us_external` would silently have trained on BrEaST as well. A listed source that was never prepared is an error that names the fix.
* Verified here on fake VinDr, CMMD (DICOM) and ultrasound data with a tiny backbone (`tests/test_experiments.py`): preparation, skip, failure isolation, reuse and the summary. Not verified: the real runtime of 13 DINOv2-base trainings on your GPU, and the real CMMD preparation throughput.

## What was wrong, and what changed

Statements about v1 are limited to what was verified in the submitted code or in the submission text.

| | Component | Rejected submission (v1) | This work (v2) |
|---|---|---|---|
| C1 | Splits | sklearn train\_test\_split on files / images (not group-aware). | Group-stratified by study, patient and near-duplicate cluster; official VinDr test untouched; assert\_no\_leakage. |
| C2 | Ultrasound data | 'normal' mapped to benign (label 0); duplicate and conflicting-label copies kept. | Default binary\_no\_normal; dHash de-duplication, conflicts dropped; Doppler / caliper overlays inpainted; masks merged. |
| C3 | Pairing | Random pairing of mammogram and ultrasound images that share a label (as described in the submission). | Tiers: native CC-MLO; real mammogram-US (evaluation only); synthetic label-matched / size-conditioned; random and shuffled controls. Within-split; frozen 1-to-1 val / test pairs; train pairs re-drawn each epoch. |
| C4 | Merge audit | No integrity checks. | Schema, label mapping, cross-split duplicates, external-source isolation, class counts, file integrity, source-label confounding; errors stop the run. |
| C5 | Cross-attention | Attention over a length-1 sequence: softmax = 1, output = v\_proj(context), query ignored. | Bidirectional multi-head cross-attention over patch-token sequences (pre-norm, FFN); query-dependence assertion. |
| C6 | Backbone | DINOv2 only; CLS vector only. | Registry: DINOv2, DINOv2 with registers, DINOv3 (patch 16, 4 registers excluded from the grid), CNN baselines; dead mask\_token frozen. |
| C7 | Fusion | Single fusion variant. | Ladder: unimodal, early, concat, GMU, x-attn, x-attn + GMU; late fusion computed at evaluation. |
| C8 | Optimiser | Multimodal trainer optimised fusion\_classifier only. | AdamW groups (head / backbone lr, no decay on norms and tokens); every trainable parameter must be in the optimiser. |
| C9 | Checkpoints | Not resumable. | Atomic checkpoints; exact resume (RNG restored); config and frozen-weight fingerprints. |
| C10 | Loss / sampling | Focal alpha was one scalar multiplier (no class weighting). | Class-indexed alpha; balanced sampler; warning when both correct imbalance. |
| C11 | Evaluation | Point estimates on a 117-image test set; 95% CIs for the unimodal rows only; checkpoint chosen by lowest validation loss. | Threshold chosen on validation only; stratified bootstrap CIs; DeLong, exact McNemar, paired bootstrap; drop-a / drop-b / shuffled-partner controls. |
| C12 | Explanation | Qualitative Grad-CAM overlays on each stream; no quantitative localisation or faithfulness evaluation. | Grad-CAM on patch tokens through the fused logit; rollout; cross-attention maps; pointing, energy, IoU, deletion / insertion vs random. |
| C13 | Benchmarks | Four-row comparison with works on other datasets and modalities (some private); no protocol-matched baselines. | Literature table with protocol flags (no unread numbers); protocol-matched config; baselines on identical splits. |
| C14 | Figures | Raster PNG figures in the default plotting style (e.g. the Grad-CAM panel). | Vector PDF / 600-dpi PNG, embedded fonts, journal column widths, colour-blind-safe palette. |

The corrections are drawn in `paper/figures/` (TikZ, vector) and tested; see *Figures* below.

## What the first real runs showed

First Colab runs (L4 GPU, `dinov2_base`, last 2 blocks unfrozen, seed 42, one run each). Intervals are bootstrap/DeLong 95 % CIs on the held-out test split; with test sets of a few hundred images, differences below ~0.03 AUROC are not effects.

| Run | Test AUROC | Controls (change in AUROC) |
|---|---|---|
| VinDr native CC<->MLO pairs, x-attn (n = 2000, 4.95 % positive) | 0.830 [0.787-0.876] | drop CC -0.053, drop MLO -0.118, shuffled MLO -0.143 |
| Pool BUSI + BUS-BRA, `label_matched` pairs (n = 353, 32.9 % positive) | 0.933 [0.906-0.958] | drop mammogram -0.030, drop ultrasound -0.171, shuffled ultrasound -0.346 |
| Pool, `random` pairs (label = mammogram label) | 0.755 [0.628-0.866] | drop mammogram -0.257, shuffled ultrasound about 0 |
| Pool, `shuffled_partner` pairs | 0.769 [0.709-0.815] | drop mammogram -0.249, drop ultrasound -0.011 |

What this does and does not show, and what the code now does about it:

| Finding | Consequence | Added in this branch |
|---|---|---|
| Dropping the mammogram costs only 0.03 AUROC on the pool, and the ultrasound alone carries most of the signal; there is no unimodal baseline on the pool | the 0.93 cannot be attributed to the fusion or to the mammogram | `pool_ladder` suite (unimodal a / b, concat, gated, x-attn, x-attn + modality dropout) and `fusion_gain.csv`: paired ΔAUROC (DeLong p, paired-bootstrap CI) of each fused model against both unimodal models and against the best one chosen on validation |
| VinDr test Brier 0.0488 vs 0.0471 for always predicting the prevalence; the sensitivity-target threshold landed at p = 0.0004 (specificity 0.47) | the probabilities are miscalibrated by the balanced sampler (about 50 % positives in training vs 5 % in the data) | `eval.calibration=platt` (default; fitted on validation, applied to every split and control, AUROC unchanged), Brier skill score, `spec_target` threshold and sensitivity/specificity operating points (`sens90/95`, `spec90/95`, Youden) in `metrics.json` |
| Best epoch 4 of 12; validation AUROC jitters by 0.02 between epochs | small, noisy validation split, backbone fine-tuned at one learning rate | EMA of the trainable weights (`training.ema_decay`), layer-wise lr decay, stochastic depth, label smoothing; all off by default so old runs keep their fingerprints |
| Mammogram explanations at chance (pointing game 0/91 and 3/97 vs about 1-2 % for random maps); ultrasound x-attn pointing 0.38 [0.28-0.48] vs 0.07 random | a 1-2 cm lesion is about one 14-px patch at 224 x 224 | rectangular, per-stream input sizes (`input.image_size_a=[448,266]`): the stored 912 x 1520 PNGs are already cropped, so this is a resolution problem, not a cropping bug |
| The first ladder run died with the runtime and left `suite_ladder` empty | an hour of work lost | suites are resumable: finished runs are skipped, `status.csv` after every run, `--force` to redo |
| `prepare us` stopped at BUS-BRA ("no images found") until a manifest was hand-written in the notebook | manual step on every fresh Drive | `source_manifest.csv` for `bus_bra` and `breast_lesions_usg` is built from their spreadsheets automatically (`mmibc manifest <key>`) |

These are hypotheses until the suites are run on the real data (`mmibc suite pool_ladder`, `mmibc suite recipe`, then `--seeds 42 43 44` on the best configuration). The `*_robust` configs bundle the mitigations; they are untested on real data.

## Design in brief

* **Pairing is tiered, and says what it is.** 1a native CC<->MLO views of one breast (genuine correspondence); 1b real same-patient mammogram + ultrasound pairs (KAU-BCMD, evaluation only); 2 synthetic label-matched or size-conditioned pairs; 3 controls (random, shuffled partner). Pairs never cross a split, validation/test pairs are frozen 1-to-1, training partners are re-drawn every epoch. A model that scores the same on the controls is not using the pairing, and the evaluation reports exactly that (`drop_a`, `drop_b`, `shuffle_b`).
* **Cross-attention that attends.** Pre-norm multi-head cross-attention over *token sequences* in both directions. The legacy module attended over a length-1 sequence (softmax = 1), so its output ignored the query; a test pins this down.
* **Backbones.** `dinov2_base` is the default (verified model id, Apache-2.0, reproducible). The registry also holds DINOv2 small/large, DINOv2 with registers (ids confirmed), DINOv3 (ViT-S/B/L/H+; gated on Hugging Face, custom licence, needs `HF_TOKEN`; the ViT-L/16 and ViT-S/16 ids were checked against model card / docs, ViT-B/16 and H+/16 follow the naming scheme and are flagged unverified in the error message if they fail to load) and torchvision CNNs. `mmibc suite backbones` (VinDr) and `mmibc suite backbones_cmmd` (CMMD) decide on your data; do not assume the newest model is the best one. **Without a token the two DINOv3 rows are `SKIPPED` before anything is built (the other four run), and a second call after adding `HF_TOKEN` trains only those two:** finished rows are kept. The token is found in `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`, a saved `huggingface-cli login`, or the Colab secret (`export_colab_secrets()` copies it into the environment for `!mmibc` commands). The gated weights themselves could not be downloaded in the build sandbox: this machinery is tested with tiny stand-ins, the real DINOv3 rows are untested.
* **Merge integrity.** Before anything trains: schema, uid/path collisions, label mapping, leakage and cross-split near-duplicates (dHash), external-source isolation, class counts, file integrity, source-label confounding, pair integrity. Errors stop the run (`audit.stop_on_error`).
* **Evaluation.** Calibration (Platt by default) and threshold chosen on validation only; DeLong AUROC intervals, stratified bootstrap, exact McNemar, paired bootstrap; late fusion computed from the unimodal runs; sensitivity/specificity operating points; fusion-gain tests against the unimodal baselines.
* **Explanations are measured, not just displayed.** Grad-CAM on patch tokens through the *fused* logit, attention rollout, cross-attention maps; pointing game, energy in lesion, IoU, deletion/insertion AUC, each against a random-map baseline, scored on VinDr boxes and ultrasound masks.
* **Literature comparison with protocol flags** (`benchmarks/literature.csv`). A published number is shown as comparable only if the run satisfies the protocol it was obtained under (labels, split). Papers that could not be read carry no numbers.

## Data

| key | what | status |
|---|---|---|
| `vindr_images`, `vindr_annotations` | VinDr-Mammo (PNG conversion, annotations) | Kaggle slugs from the legacy code |
| `busi` | BUSI | Kaggle mirror `sabahesaraki/breast-ultrasound-images-dataset` (chosen by the maintainer; layout unverified, inspect first). The original Cairo link is dead |
| `bus_bra` | BUS-BRA ultrasound (Zenodo, CC BY 4.0) | URL verified against the record page and checked by MD5; if Zenodo refuses the download, put `BUSBRA.zip` into `raw/bus_bra/` by hand. Layout confirmed on a real run (`BUSBRA/{Images,Masks,bus_data.csv}`, 1,875 images, 1,064 patients); `source_manifest.csv` is built automatically |
| `kau_bcmd` | KAU-BCMD, real mammogram + ultrasound pairs | Kaggle slug; layout unknown, inspect first |
| `bus_uclm` | BUS-UCLM ultrasound (lesion segmentation) | Kaggle slug from the page title; layout unknown, inspect first |
| `cmmd` | CMMD, **mammography** (biopsy-confirmed), not ultrasound | TCIA DICOM download as found on the maintainer's Drive (`raw/cmmd/**/CMMD/D*-*/**.dcm`, `CMMD_clinicaldata_revision.xlsx`, `metadata.csv`); the Kaggle mirror in the registry has an unverified layout. Used by `mmibc prepare cmmd` and `data.mammo_source=cmmd` (see *CMMD* below) |
| `breast_lesions_usg` | BrEaST-Lesions-USG, intended external ultrasound test set | TCIA: images-and-masks zip **and** clinical spreadsheet. 256 cases (154 benign / 98 malignant / 4 normal; 55 cases confirmed by follow-up rather than biopsy), flat `caseNNN.png` / `caseNNN_tumor.png`; `source_manifest.csv` is built from the spreadsheet automatically. Use it via `mmibc run -c mammo_us_external` (held out entirely, reported as `ext_test`) |

`mmibc download` lists them, `mmibc download --all` fetches every one (continues past failures, prints a summary, skips what already arrived); `mmibc inspect <key>` shows what actually arrived. Sources that do not follow a class-folder layout need a `source_manifest.csv` (`uid, raw_path, class_name[, mask_paths, patient_id]`): `prepare us` builds it for `bus_bra` and `breast_lesions_usg` (`mmibc manifest <key>` does only that step and never overwrites an existing file without `--force`); for `bus_uclm` write it by hand after `mmibc inspect bus_uclm`. The VinDr label is a BI-RADS *suspicion* grouping (`birads_4plus` or `birads_3plus`), not pathology. Known caveat for anyone using foundation models pre-trained on VinDr: a validation split carved from VinDr-train may overlap their pre-training data.

## CMMD: biopsy-labelled mammograms

CMMD (Chinese Mammography Database, TCIA) labels are **pathology**, not BI-RADS, so a "label-matched" mammogram + ultrasound pair means the same thing on both sides (with VinDr the mammogram label is a suspicion grouping). It is wired as an alternative mammogram source: `data.mammo_source=cmmd` in `cmmd_multiview.yaml` (native CC<->MLO pairs), `cmmd_us_pool.yaml` (paired with the BUSI + BUS-BRA pool) and the suites `cmmd_pool_ladder`, `backbones_cmmd`.

What was read from the real files and sheets (not guessed), and what the code does with it:

| Fact | Handling |
|---|---|
| DICOM, 8-bit, ~2294 x 1914; `ImageLaterality` is present, `Laterality` / `ViewPosition` are not; the view is only in `ViewCodeSequence` (cranio-caudal / medio-lateral oblique) | side from `ImageLaterality`, view from the code sequence; a missing view falls back to the instance number (1 = CC, 2 = MLO, ...) and the use is logged; unreadable files are dropped, and refused above 5 % |
| The labels are not in the DICOM files: `CMMD_clinicaldata_revision.xlsx` has one row per biopsied breast (1,872 rows for 1,775 patients; 1,316 malignant / 556 benign) | joined on patient and breast side; several lesions in one breast: malignant wins |
| A study has 2 images (the biopsied breast) or 4 (both breasts); the other breast of a 4-image study has **no recorded finding** | it is left out, never labelled benign |
| Subset **D1 is 51 % malignant, D2 98 %** (D1: 1,107 breasts, D2: 765). A model can score well by recognising the subset | the audit warns (`subset_shortcut`, `metadata_shortcut`), `data.cmmd.subsets=[D1]` removes it, and `metrics.json` reports AUROC per subset (`by_stratum`). Use the D1-only number as the headline |
| No official split, no lesion boxes | split by patient (stratified by label and subset); no localisation metrics for CMMD, and `size_conditioned` pairing is refused for it |
| TCIA's `metadata.csv` lists the files of each patient | an incomplete download (> 5 % of patients short) is refused |

Throughput is untested: the first `prepare cmmd` reads about 16 GB of DICOM through the Drive mount. It is resumable (`progress.jsonl`); `data.num_prepare_workers=16` may help since the work is mostly I/O.

## Repository map

```
configs/        base.yaml, vindr_multiview[_robust], cmmd_{multiview,us_pool}, mammo_us_{synthetic,pool,external,robust}, protocol_petrini, smoke,
                suites/{ladder,pool_ladder,recipe,backbones,backbones_cmmd,cmmd_pool_ladder,pairing}, programs/all (mmibc experiments)
src/mmibc/
  data/         registry, download, preprocess, ultrasound, manifests, vindr, cmmd, splits, pairing, merge_audit
  models/       backbones (registry), attention, fusion (ladder), factory
  training/     losses, optim (layer-wise lr decay), ema, checkpoint (atomic, resumable), trainer
  evaluation/   metrics (calibration, operating points), stats (DeLong, McNemar, bootstrap), evaluate, compare (fusion gain, ensembles)
  xai/          ground_truth, explain, metrics, report
  benchmarks/   literature.csv, compare
  figures/      style, plots, make, tex (TikZ build), corrections (single source of the key)
  utils/        config, paths, seed, hf (Hugging Face token, gated-backbone pre-flight)
  pipeline.py, suite.py, experiments.py (the whole programme), cli.py
paper/figures/  TikZ diagrams (fig_architecture, fig_algorithm), corrections table, pre-built PDF / SVG / 600-dpi PNG
tests/          every claim in the table above has a test
legacy/         the first-submission code, unchanged
```

## Figures for the paper

* Data plots (ROC with DeLong intervals, fusion ladder, controls, calibration, explanation summary, split composition, explanation overlays) come from `mmibc figures`: vector PDF + SVG + 600-dpi PNG, embedded fonts, colour-blind-safe palette, widths per venue (`--template lncs|ieee|elsevier|generic`).
* The architecture and algorithm diagrams are TikZ, drawn for a 12.2 cm text width (Springer LNCS / MICCAI). In the paper: `\input{figures/mmibc_tikz}` once in the preamble, then `\input{figures/fig_architecture}` inside a `figure`; for another width wrap it in `\resizebox{\linewidth}{!}{...}`. `tab_corrections.tex` is the table above. Rebuild with `mmibc figures --tex` (needs `texlive-latex-extra` and `poppler-utils`); the committed PDFs work without TeX.

![Algorithm of MMIBC v2](paper/figures/pdf/fig_algorithm.png)

## Tests

```bash
pip install -e ".[dev]" && pytest -q
```

CPU only, no downloads (OpenCV is optional: inpainting has a fallback, CLAHE needs it; tiny random backbones, synthetic images with planted defects to prove that the audits catch them).

## Citation

To be added after the revised paper is accepted. Please cite the datasets you use (VinDr-Mammo, CMMD, BUSI, BUS-BRA, BrEaST-Lesions-USG, KAU-BCMD).
