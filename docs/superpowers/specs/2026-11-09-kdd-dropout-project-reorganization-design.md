# KDD Cup 2015 Dropout Project Reorganization Design

## Intent

Turn the current collection of KDD Cup 2015 dropout-prediction notebooks, datasets, papers, and duplicate experiments into a project that a student or reviewer can understand and run in Google Colab. The reorganization must preserve the existing experimental work while making the baseline, model variants, comparison workflow, data requirements, and next improvement steps explicit.

The immediate purpose is not to claim a new best score. It is to establish a trustworthy, reproducible project structure from which the current event-level BiLSTM baseline (reported at roughly 0.85 AUC) can be compared with the existing FSLSM, Transformer, Markov, GRU, contrastive, and graph-based experiments.

## Current State

- The checked-in dataset is split between `dataset/train/train/` and `dataset/test/test/` and contains enrollment, log, and truth CSV files.
- The original object metadata used by several notebooks is stored at `code/archive/object.csv`; `date.csv` is also archived there.
- The baseline is a three-stage pipeline: raw event extraction/scaling, token-and-time-gap preprocessing, and a PyTorch BiLSTM evaluation notebook.
- Event tokens combine `source`, `event`, and object `category`; the sequential input adds a log-scaled time-gap value. The BiLSTM embeds the three categorical components and concatenates the scalar gap.
- FSLSM experiments derive learning-style features and fuse them with a BiLSTM or Transformer.
- The `kdd_seq_*` notebooks form a separate unsupervised sequence-representation suite: shared setup, Markov mixture, causal next-token GRU, GRU autoencoder, contrastive GRU, sequential GNN, and cross-model comparison.
- Data paths are inconsistent. Some notebooks use local `data/raw/...` paths, some download the Kaggle dataset into Colab, and the sequence suite expects `/content/drive/MyDrive/kdd2015` plus `/content/drive/MyDrive/kdd_seq`.
- Several notebooks are unnamed, duplicated, or earlier versions. Results are mostly embedded in notebook outputs rather than stored in a single authoritative results table.

## Goals

1. Organize active notebooks by model family and give them stable, descriptive filenames.
2. Preserve superseded, duplicate, and exploratory notebooks in an archive rather than deleting them.
3. Keep the checked-in dataset in place and avoid duplicating the large log files.
4. Make every active workflow understandable as a Google Colab workflow, including Drive mounting, expected dataset placement, GPU expectations, and notebook execution order.
5. Create a root `PROJECT.md` that documents the research question, dataset, feature construction, model catalog, evaluation metrics, known limitations, execution instructions, and a prioritized improvement roadmap.
6. Provide model-level README files where a directory contains a multi-notebook pipeline or important dependencies.
7. Remove misleading filename suffixes such as `(1)`, `(2)`, and `(3)` from active copies.
8. Ensure internal documentation points to the new locations and clearly distinguishes verified code behavior from reported or unverified metric values.

## Non-goals

- Retrain all models or promise a higher AUC during this reorganization.
- Rewrite every notebook into a Python package.
- Delete research papers, datasets, old notebooks, or notebook outputs.
- Change model mathematics, hyperparameters, feature definitions, or train/test semantics merely to make the folders look consistent.
- Treat the official test truth files as an unseen competition test set; the documentation must describe them as the repository's held-out evaluation split when they are used for scoring.
- Make large data files importable through relative paths in hosted Colab without Drive upload or Kaggle download.

## Anti-goals

- A polished structure that hides broken paths or unsupported claims.
- Multiple copies of the 1 GB-scale logs.
- Mixing active model notebooks with historical variants without explaining which one should be run.
- Renaming notebooks without preserving a mapping from old names to new names.
- Reporting notebook output as a fair comparison when preprocessing windows, seeds, or evaluation protocols differ.

## Constraints

- The primary execution environment is Google Colab.
- Existing datasets remain under `dataset/`; no large CSV is moved or copied.
- Existing user work is preserved. Files classified as historical are moved into `archive/`, not removed.
- Notebook JSON must remain valid after relocation or edits.
- Colab-specific shell commands and Drive mounting are allowed, but credentials must not be checked in.
- The project currently has no detectable Git repository, so the work cannot rely on commits for recovery.
- Metric values are documented only when supported by notebook outputs or explicitly identified as the user's reported baseline.

## Approaches Considered

### 1. Index-only organization

Leave all notebooks where they are and add a root index that links to each file. This has the lowest risk of breaking paths, but it leaves duplicate names, mixed model families, and unclear active-versus-archive status. It does not adequately satisfy the request to organize the project into different models.

### 2. Conservative model-centered reorganization — selected

Create a model-centered hierarchy, move active notebooks into descriptive model directories, move duplicates and superseded work into an archive, and update documentation and path/configuration cells only where needed for Colab clarity. Preserve datasets and research sources. This gives the project a usable shape without changing the experimental logic.

### 3. Full notebook consolidation

Rewrite the work into a small set of standardized end-to-end Colab notebooks backed by shared Python modules. This would ultimately be cleaner and reduce duplication, but it would also be a behavior-changing refactor, require extensive retraining, and risk invalidating existing outputs. It is better treated as a later improvement project.

## Selected Structure

```text
project/
├── PROJECT.md
├── dataset/
│   ├── train/train/
│   └── test/test/
├── models/
│   ├── baseline_bilstm/
│   │   ├── 01_event_extraction_and_scaling.ipynb
│   │   ├── 02_sequence_preprocessing.ipynb
│   │   ├── 03_bilstm_training_and_evaluation.ipynb
│   │   └── README.md
│   ├── fslsm_bilstm/
│   │   ├── 01_fslsm_sequence_preprocessing.ipynb
│   │   ├── 02_fslsm_bilstm_training_and_evaluation.ipynb
│   │   ├── 03_all_dimensions_event_level_bilstm.ipynb
│   │   └── README.md
│   ├── transformer/
│   │   ├── fslsm_transformer.ipynb
│   │   └── README.md
│   ├── sequence_representation/
│   │   ├── 00_shared_setup.ipynb
│   │   ├── 01_markov_mixture.ipynb
│   │   ├── 02_next_token_gru.ipynb
│   │   ├── 03_gru_autoencoder.ipynb
│   │   ├── 04_contrastive_gru.ipynb
│   │   ├── 05_sequential_gnn.ipynb
│   │   ├── 06_model_comparison.ipynb
│   │   ├── kdd_seq_common.py
│   │   └── README.md
│   └── feature_engineering/
│       ├── fslsm_all_dimensions_scoring.ipynb
│       ├── proportional_features.ipynb
│       └── README.md
├── archive/
│   ├── notebooks/
│   └── legacy_metadata/
├── references/
│   ├── papers/
│   ├── reports/
│   └── figures/
└── docs/superpowers/
```

The precise classification of ambiguous notebooks is left to the builder after comparing their code. The builder may keep an ambiguous experiment in `archive/notebooks/` and document it rather than promoting it to an active model.

## File Classification Rules

- Promote a notebook to `models/` only when it has a distinct, understandable role in an active pipeline.
- When two notebooks implement substantially the same experiment, keep the clearest or latest active version and archive the other with its original filename.
- Put intermediate feature-generation notebooks under `models/feature_engineering/` when they are reusable across models.
- Put superseded EDA, unnamed experiments, earlier baseline versions, and duplicate copies under `archive/notebooks/`.
- Put `object.csv` and `date.csv` under `archive/legacy_metadata/` only if active notebooks are updated to point there or the files are deliberately promoted to a small shared metadata directory. Their provenance and necessity must be documented.
- Group academic PDFs and project reports under `references/` without rewriting their contents.

## Colab Compatibility Design

Each active model README will explain one of two supported data-loading modes:

1. **Google Drive mode:** mount Drive and point a clearly named configuration variable to a folder containing the KDD files.
2. **Kaggle download mode:** install/configure the Kaggle client at runtime and download the public dataset without storing credentials in the repository.

Active notebooks should expose configuration paths near the top rather than scattering hard-coded paths throughout the notebook. The reorganization should not force all model families onto one artifact directory when their existing pipelines use different artifact formats. Instead, documentation will state the expected input and output location per model family.

Relocation edits must be minimal. When a notebook already has a coherent Colab setup, update names and documentation but do not rewrite its training code. When a notebook only contains local relative paths, add or revise a top configuration cell so the notebook can locate Drive data in Colab while retaining a documented repository-relative option.

## Documentation Design

`PROJECT.md` will be the main handoff document and will include:

- Problem statement and definition of dropout.
- Actual repository dataset tree and the role of every CSV.
- Baseline feature construction: first-two-week filtering where used, object-category join, composite event tokens, train-only vocabularies, time-gap transform, sequence/window building, embeddings, and BiLSTM classifier.
- Model catalog with each model's purpose, inputs, outputs, and notebook path.
- A run-order table with Colab/GPU requirements and upstream artifacts.
- Evaluation definitions for accuracy, precision, recall, F1, and ROC AUC.
- A results section that separates the user-reported baseline AUC (~0.85), notebook-recorded values, and results that still need rerunning.
- Methodological risks: potential preprocessing inconsistency, differing observation windows, test-set reuse, threshold dependence, class imbalance, random seed variation, event-count confounding, and lack of a unified validation protocol.
- A prioritized roadmap for improving the baseline, beginning with a reproducible stratified validation protocol and strong tabular baselines before architectural complexity.
- A legacy-name mapping so references in reports remain traceable.

Model READMEs will remain short and operational: objective, files, required inputs, generated artifacts, Colab run order, and caveats.

## Evaluation and Scientific Integrity

The reorganization will not present accuracy as the primary comparison metric because dropout labels can be imbalanced. ROC AUC will remain the headline metric to match the current project, with F1, precision, recall, and accuracy reported alongside it. Future experiments should also consider PR AUC and confidence intervals.

Comparisons are valid only when models share the same observation window, enrollment split, feature availability, and label handling. The documentation must flag any notebook family that violates or does not establish this condition. The sequence-representation suite evaluates learned features through downstream classifiers and therefore is not directly identical to the end-to-end BiLSTM evaluation unless the protocol is aligned.

## Verification

After reorganization:

1. Every original file must be accounted for in its new location or deliberately left in place.
2. All moved `.ipynb` files must parse as JSON.
3. Documentation links and listed paths must exist.
4. No notebook may reference an old repository path without the documentation explaining it as a Colab/Drive path.
5. No Kaggle credentials or secrets may be present.
6. Dataset file sizes and row counts must remain unchanged.
7. The project document must distinguish measured, reported, and pending results.

Full model execution is outside this reorganization because of dataset scale and Colab/GPU requirements. Static validation and lightweight preprocessing checks are required locally where practical.

## Builder Decisions

- Select the clearest duplicate notebook as active based on code completeness and explanatory content.
- Choose whether `object.csv` belongs in a small shared metadata directory or remains in legacy metadata, provided active dependencies and provenance are explicit.
- Make only mechanical notebook path edits required by relocation and Colab clarity.
- Create a concise legacy filename map rather than renaming mentions inside historical reports or PDFs.
