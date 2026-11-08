# KDD Cup 2015 MOOC Dropout Prediction

This project predicts whether a learner will drop out of a MOOC from their event history in the KDD Cup 2015 dataset. The central baseline represents every learner as an ordered sequence of event tokens plus time gaps and trains a bidirectional LSTM classifier. The repository also contains learning-style features, a Transformer, probabilistic sequence clustering, three GRU representation learners, and a sequential graph neural network.

The code is organized for Google Colab. The checked-in notebooks retain their recorded outputs, but the project has **not** been rerun end-to-end after reorganization. Treat stored scores as evidence from earlier runs, not as a newly reproduced benchmark.

## Project layout

```text
.
├── PROJECT.md
├── dataset/
│   ├── train/train/
│   └── test/test/
├── shared/metadata/
│   ├── object.csv
│   └── date.csv
├── models/
│   ├── baseline_bilstm/
│   ├── fslsm_bilstm/
│   ├── transformer/
│   ├── feature_engineering/
│   └── sequence_representation/
├── archive/notebooks/
├── references/
│   ├── papers/
│   ├── reports/
│   └── figures/
├── tools/
└── docs/
```

The complete original-to-new path mapping and integrity hashes are in [the original file inventory](docs/original-file-inventory.md).

## Dataset

The current repository contains the official train/test-style split plus the provided truth files. The notebooks interpret label `1` as dropout and label `0` as retained.

| File | Data rows | Purpose |
|---|---:|---|
| `dataset/train/train/enrollment_train.csv` | 120,542 | Enrollment, anonymized user, and course identifiers |
| `dataset/train/train/log_train.csv` | 8,157,277 | Timestamped learner events used for training features |
| `dataset/train/train/truth_train.csv` | 120,542 | Training labels; no header |
| `dataset/test/test/enrollment_test.csv` | 80,362 | Held-out enrollment records |
| `dataset/test/test/log_test.csv` | 5,387,847 | Held-out learner events |
| `dataset/test/test/truth_test.csv` | 80,362 | Held-out labels; no header |
| `shared/metadata/object.csv` | 27,249 | Object/module category and course metadata |
| `shared/metadata/date.csv` | 39 | Course date ranges |

The label distribution is strongly skewed toward dropout:

- Train: 95,581 dropout and 24,961 retained (79.29% dropout).
- Test: 63,642 dropout and 16,720 retained (79.19% dropout).

This imbalance makes accuracy alone misleading. ROC AUC is the current headline metric, while precision, recall, F1, accuracy, and preferably PR AUC should be retained for context.

## Baseline pipeline

The baseline consists of three notebooks in [models/baseline_bilstm](models/baseline_bilstm/README.md):

1. Optional event-count extraction and temporal-slice clustering.
2. Sequential preprocessing.
3. BiLSTM training and evaluation.

The sequence preprocessing performs the following operations:

1. Parse and sort timestamps per `enrollment_id`.
2. Keep the first 14 days of each learner's observed activity.
3. Join each logged object to its metadata category.
4. Create a composite token: `source + "_" + event + "_" + category`.
5. Compute `log1p` of the elapsed seconds since the learner's preceding event.
6. Build categorical vocabularies from the training split only; unseen values map to index `0`.
7. Split sequences into windows of 150 events with stride 75 and zero-pad shorter windows.
8. Save encoded arrays and labels for model training.

The BiLSTM embeds source into 4 dimensions, event into 8 dimensions, and category into 8 dimensions. The embeddings are concatenated with the scalar time gap, giving 21 dimensions per event. A 64-unit bidirectional LSTM produces 128-dimensional contextual states; learned attention aggregates these states before a sigmoid dropout classifier. Window predictions are averaged by enrollment before metrics are calculated.

## Model catalog

| Model family | Main idea | Entry point | Evaluation type |
|---|---|---|---|
| Baseline BiLSTM | Composite event tokens + time gaps + attention | [Baseline guide](models/baseline_bilstm/README.md) | End-to-end dropout prediction |
| FSLSM BiLSTM | Fuse four learning-style scores with the sequence representation | [FSLSM guide](models/fslsm_bilstm/README.md) | End-to-end baseline/augmentation comparison |
| FSLSM Transformer | Self-attention with no, late, or token-level FSLSM fusion | [Transformer guide](models/transformer/README.md) | End-to-end dropout prediction |
| Markov mixture | Soft mixture of first-order transition models | [Sequence representation guide](models/sequence_representation/README.md) | Learned features evaluated by a downstream classifier |
| Next-token GRU | Causal language-model pretraining over event/gap tokens | [Sequence representation guide](models/sequence_representation/README.md) | Downstream classifier |
| GRU autoencoder | Bidirectional encoder with a normalized bottleneck | [Sequence representation guide](models/sequence_representation/README.md) | Downstream classifier |
| Contrastive GRU | Mask/crop augmentations with an InfoNCE objective | [Sequence representation guide](models/sequence_representation/README.md) | Downstream classifier |
| Sequential GNN | Session graph with gated graph message passing | [Sequence representation guide](models/sequence_representation/README.md) | Downstream classifier |
| Feature engineering | FSLSM scoring, proportional events, and clustering | [Feature guide](models/feature_engineering/README.md) | Feature/cluster diagnostics |

## Running in Google Colab

### Repository-root workflows

Upload or synchronize this whole project to:

```text
/content/drive/MyDrive/kdd_dropout_project
```

The normalized baseline, two-stage FSLSM, and feature-engineering notebooks use that location by default. If the folder has another name, set the environment variable before running the configuration cell:

```python
import os
os.environ["KDD_PROJECT_ROOT"] = "/content/drive/MyDrive/your_folder"
```

Keep `dataset/`, `shared/metadata/`, and `artifacts/` under the chosen project root. Each notebook mounts Drive and prints the resolved project and artifact paths. Use a GPU runtime for neural models; preprocessing and scikit-learn clustering can run on CPU but require substantial RAM.

### Kaggle-download workflows

The all-dimensions FSLSM BiLSTM and Transformer notebooks contain their own Kaggle download cells. They expect a valid `kaggle.json` in the user's Drive configuration folder. Never place credentials in this repository.

### Sequence-representation workflow

The sequence suite preserves its original Drive contract:

```text
/content/drive/MyDrive/kdd2015   # train/, test/, object.csv
/content/drive/MyDrive/kdd_seq   # shared artifacts and results
```

To use this repository's current tree without duplicating data, set the shared-setup variables to:

```python
DATA_ROOT = "/content/drive/MyDrive/kdd_dropout_project/dataset"
OBJECT_PATH = "/content/drive/MyDrive/kdd_dropout_project/shared/metadata/object.csv"
WORK = "/content/drive/MyDrive/kdd_seq"
```

`DATA_ROOT` and `OBJECT_PATH` are separate because the large train/test logs remain under `dataset/` while object metadata lives under `shared/metadata/`. Copy `kdd_seq_common.py` into the configured `WORK` folder when requested. If you change `WORK` from `/content/drive/MyDrive/kdd_seq`, make the same edit at the top of notebooks `01` through `06`; each is designed to run independently.

## Recommended run order

| Order | Notebook | Runtime | Produces/uses |
|---:|---|---|---|
| 1 | `models/baseline_bilstm/02_sequence_preprocessing.ipynb` | High-RAM CPU | Produces `artifacts/baseline_bilstm/processed_data.pt` |
| 2 | `models/baseline_bilstm/03_bilstm_training_and_evaluation.ipynb` | GPU | Uses the baseline processed artifact |
| 3 | `models/fslsm_bilstm/01_fslsm_sequence_preprocessing.ipynb` | High-RAM CPU | Produces `processed_data_fslsm.pt` |
| 4 | `models/fslsm_bilstm/02_fslsm_bilstm_training_and_evaluation.ipynb` | GPU | Fair within-notebook baseline vs FSLSM comparison |
| 5 | `models/transformer/fslsm_transformer.ipynb` | GPU | Transformer fusion variants; downloads data independently |
| 6 | `models/sequence_representation/00_shared_setup.ipynb` | High-RAM CPU | Builds shared vocabulary, sparse transitions, and evaluation space |
| 7 | `01_markov_mixture.ipynb` through `05_sequential_gnn.ipynb` | CPU/GPU by model | Independent learned representations and downstream results |
| 8 | `models/sequence_representation/06_model_comparison.ipynb` | GPU recommended | Consolidates stored model results and fusion tests |

`01_event_extraction_and_scaling.ipynb` and the feature-engineering notebooks are optional exploratory/feature-analysis stages, not prerequisites for the baseline sequence model.

## Metrics

- **Accuracy:** fraction of correct thresholded predictions; sensitive to class imbalance.
- **Precision:** fraction of predicted dropouts that are actual dropouts.
- **Recall:** fraction of actual dropouts detected.
- **F1:** harmonic mean of precision and recall at the chosen threshold.
- **ROC AUC:** ranking quality across thresholds; the main metric used in the current notebooks.
- **PR AUC / average precision:** recommended addition because dropout is the majority class and threshold behavior matters.

Threshold-based metrics currently use `0.5` in the baseline. Any future comparison must report how its threshold was selected.

## Recorded results and status

These values are copied from stored notebook outputs. They were **not rerun during reorganization**.

| Status | Experiment | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---:|---:|---:|---:|---:|
| Recorded active output | Current three-stage BiLSTM | 0.8443 | 0.8789 | 0.9317 | 0.9045 | 0.8333 |
| Recorded archived output | Earlier `Project_v2.ipynb` run | 0.8581 | 0.8959 | 0.9287 | 0.9120 | 0.8545 |
| User-reported summary | Baseline family | — | — | — | — | approximately 0.85 |
| Recorded active output | Event-level BiLSTM, sequential only | 0.8447 | 0.8748 | 0.9381 | 0.9053 | 0.8323 |
| Recorded active output | Event-level BiLSTM + four FSLSM dimensions | 0.8423 | 0.8673 | 0.9456 | 0.9048 | 0.8320 |

The active FSLSM notebook reports an AUC change of roughly `-0.0004`, so that particular feature fusion did not improve the stored baseline run.

The sequence-representation comparison uses a different downstream-classifier protocol. Its stored cross-validation baseline is `0.85318 ± 0.00205` ROC AUC. The strongest displayed Markov configuration reaches `0.85357 ± 0.00198` in that table. On its stored repeated held-out evaluation, the downstream baseline is `0.85305 ± 0.00006`, while `markov_soft` is `0.85386 ± 0.00010`. These small differences must not be compared directly with the end-to-end BiLSTM table until splits, windows, labels, and baseline features are aligned.

Results not explicitly listed above should be treated as **pending rerun and consolidation**.

## Methodological risks

1. **Test-set reuse:** multiple notebooks inspect and optimize against the supplied test truth. Repeated selection on this split can make it an informal validation set rather than an unbiased final test.
2. **Protocol mismatch:** notebook families use different windows, sequence truncation, baseline features, and downstream classifiers.
3. **Window leakage/weighting:** multiple windows from the same enrollment share a label. Any random window-level split would leak learner information; splits must be enrollment-level.
4. **Class imbalance:** the roughly 79% dropout rate inflates accuracy and can make a high recall value easy to obtain.
5. **Threshold dependence:** precision, recall, and F1 depend on the fixed `0.5` threshold.
6. **Seed variance:** some earlier notebooks use one run or do not fix every source of randomness.
7. **Activity-volume confounding:** learned clusters may encode how much a learner clicked rather than behavior type; the sequence suite attempts volume residualization, which should be retained and audited.
8. **FSLSM construct validity:** learning-style dimensions are operationalized from click behavior with assumptions that require empirical validation and ablation.
9. **Competition semantics:** having `truth_test.csv` locally enables evaluation, but it is no longer an unseen competition test once consulted during development.

## Improvement roadmap

The most likely path past the current baseline is better experimental control first, then targeted modeling changes:

1. **Create one authoritative enrollment-level validation protocol.** Keep the supplied test labels untouched until final comparison; use stratified train/validation splits or grouped cross-validation with fixed seeds.
2. **Add leakage and consistency checks.** Assert train-only vocabularies/scalers, disjoint enrollment IDs, identical observation windows, and identical label aggregation across models.
3. **Establish strong tabular baselines.** Aggregate event counts, recency, active days, session statistics, time-gap summaries, course ID, and early trajectory features; compare logistic regression, random forest, and gradient boosting/XGBoost/LightGBM.
4. **Report PR AUC, calibration, and confidence intervals.** Tune the decision threshold only on validation data and report bootstrap uncertainty at the enrollment level.
5. **Run controlled ablations.** Remove time gaps, source, category, attention, FSLSM dimensions, and sliding-window overlap one at a time.
6. **Align observation windows.** Compare 1-day, 3-day, 7-day, and 14-day early-warning settings using exactly the same enrollment splits.
7. **Improve temporal representation.** Try learned time-gap embeddings or Time2Vec and explicit session boundaries before increasing network depth.
8. **Use course-aware modeling.** Add course embeddings or course-stratified evaluation to measure generalization to unseen courses.
9. **Calibrated ensembling.** Combine a strong tabular model with the best sequence model only after each adds statistically reliable validation lift.
10. **Promote only reproduced gains.** Record configuration, seed, split IDs, artifact hashes, and result CSVs for every claimed improvement.

## Historical work and references

- [Archive guide](archive/README.md) explains why older notebooks remain.
- `references/papers/` contains the research sources collected with the project.
- `references/reports/` contains project reports, presentation exports, and formatting material.
- `references/figures/` contains pipeline and generated images.

Run `node tools/validate_project.js` from the project root after moving or editing notebooks. It checks active notebook JSON, Colab-safe paths, likely embedded credentials, required guides, and local documentation links.
