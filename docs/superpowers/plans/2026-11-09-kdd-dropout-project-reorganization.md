# KDD Dropout Project Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the KDD Cup 2015 dropout-prediction repository into documented, model-centered, Google Colab-compatible workflows without losing datasets, experiments, or research material.

**Architecture:** Preserve the large dataset tree in place, move active notebooks into model-family directories, place duplicates and superseded work in a traceable archive, and collect papers/reports separately. Use a root project guide plus focused model READMEs as the stable interface; make only mechanical Colab path/configuration edits to active notebooks.

**Tech Stack:** Jupyter Notebook, Google Colab, Python, PyTorch, pandas, scikit-learn, NumPy, Google Drive, Kaggle CLI

**Spec:** `docs/superpowers/specs/2026-11-09-kdd-dropout-project-reorganization-design.md`

## Global Constraints

- The primary execution environment is Google Colab.
- Keep `dataset/` and every large CSV at its current path and byte content.
- Preserve every original notebook; archive superseded or duplicate notebooks instead of deleting them.
- Do not change model mathematics, feature definitions, hyperparameters, or evaluation semantics during relocation.
- Never add Kaggle credentials or other secrets.
- Mark the approximately 0.85 baseline AUC as user-reported unless a stored notebook output independently supports it.
- A model comparison is not described as fair unless its split, observation window, and label handling match.

## Review Focus

- Active notebooks with local-only `data/raw/...` paths must gain an explicit Colab/Drive configuration path while retaining understandable local fallback behavior; test by scanning active notebooks for unresolved legacy paths.
- Three identical FSLSM event-level notebooks must not all appear active; test source hashes and preserve two copies in the archive.
- `object.csv` and `date.csv` must remain available to active preprocessing without being duplicated; test checksums after moving them to `shared/metadata/`.
- Dataset files must remain byte-identical and unmoved; test the six recorded SHA-256 checksums after all moves.
- Documentation must not turn incomparable or unrerun experiments into a leaderboard; test project text for explicit reported/measured/pending labels and protocol caveats.

---

### Task 1: Create the model-centered structure and move active workflows

**Files:**
- Move: `code/01_Raw_Event_Extraction_and_Scaling (3).ipynb` -> `models/baseline_bilstm/01_event_extraction_and_scaling.ipynb`
- Move: `code/02_Data_Preprocessing.ipynb` -> `models/baseline_bilstm/02_sequence_preprocessing.ipynb`
- Move: `code/03_Model_Training_Evaluation.ipynb` -> `models/baseline_bilstm/03_bilstm_training_and_evaluation.ipynb`
- Move: `project/02_Data_Preprocessing_FSLSM.ipynb` -> `models/fslsm_bilstm/01_fslsm_sequence_preprocessing.ipynb`
- Move: `project/03_Model_Training_Evaluation_FSLSM.ipynb` -> `models/fslsm_bilstm/02_fslsm_bilstm_training_and_evaluation.ipynb`
- Move: `code/exp1/fslsm+eventlvl.ipynb` -> `models/fslsm_bilstm/03_all_dimensions_event_level_bilstm.ipynb`
- Move: `code/exp1/fslsm_transformer.ipynb` -> `models/transformer/fslsm_transformer.ipynb`
- Move: `code/kdd_seq_00_setup (2).ipynb` -> `models/sequence_representation/00_shared_setup.ipynb`
- Move: `code/kdd_seq_01_markov_mixture (2).ipynb` -> `models/sequence_representation/01_markov_mixture.ipynb`
- Move: `code/kdd_seq_02_next_token_gru.ipynb` -> `models/sequence_representation/02_next_token_gru.ipynb`
- Move: `code/kdd_seq_03_gru_autoencoder (1).ipynb` -> `models/sequence_representation/03_gru_autoencoder.ipynb`
- Move: `code/kdd_seq_04_contrastive_gru (1).ipynb` -> `models/sequence_representation/04_contrastive_gru.ipynb`
- Move: `code/kdd_seq_06_sequential_gnn (1).ipynb` -> `models/sequence_representation/05_sequential_gnn.ipynb`
- Move: `code/kdd_seq_05_all_in_one_neural_and_comparison (1).ipynb` -> `models/sequence_representation/06_model_comparison.ipynb`
- Move: `code/kdd_seq_common (1).py` -> `models/sequence_representation/kdd_seq_common.py`
- Move: `code/exp1/FSLSM_All_Dimensions_Scoring.ipynb` -> `models/feature_engineering/fslsm_all_dimensions_scoring.ipynb`
- Move: `code/exp1/proportional_features_1 (2).ipynb` -> `models/feature_engineering/proportional_features.ipynb`
- Move: `code/KDD_Cup_2015_Colab_Drive.ipynb` -> `models/feature_engineering/event_scaling_and_kmeans.ipynb`
- Move: `code/archive/object.csv` -> `shared/metadata/object.csv`
- Move: `code/archive/date.csv` -> `shared/metadata/date.csv`

**Interfaces:**
- Consumes: Existing notebook files and metadata with unchanged contents.
- Produces: Stable model-family paths referenced by all later documentation and checks.

- [ ] **Step 1: Record the pre-move inventory and checksums**

Write `docs/original-file-inventory.md` with every original notebook path, its destination, active/archive status, and the six dataset plus two metadata SHA-256 values captured during analysis.

- [ ] **Step 2: Create directories and move active files**

Use explicit source and destination paths. Do not use broad globs for moves. Preserve file contents during this step.

- [ ] **Step 3: Verify active notebook inventory and JSON validity**

Run a JSON parser across `models/**/*.ipynb` and assert that 17 active notebooks parse successfully. Confirm each expected destination exists.

- [ ] **Step 4: Verify shared metadata checksums**

Run: `sha256sum shared/metadata/object.csv shared/metadata/date.csv`

Expected prefixes: `80d7f475a681614e` and `1d9805e0ac7cbfa4` respectively.

### Task 2: Archive historical work and organize supporting material

**Files:**
- Create: `archive/README.md`
- Move: Remaining notebooks under `code/`, `code/archive/`, and `code/exp1/` -> `archive/notebooks/` using descriptive collision-free subdirectories.
- Move: `code/GEMINI.md` -> `archive/notes/GEMINI.md`
- Move: academic source PDFs -> `references/papers/`
- Move: project reports, presentations, and formatting documents -> `references/reports/`
- Move: PNG, JPG, and JPEG diagrams -> `references/figures/`

**Interfaces:**
- Consumes: The remainder after Task 1's active-file moves.
- Produces: A clean archive and reference collection with legacy traceability.

- [ ] **Step 1: Classify all remaining notebooks**

Place early baseline notebooks in `archive/notebooks/baseline_history/`, FSLSM variants and identical copies in `archive/notebooks/fslsm_history/`, and earlier proportional-feature work in `archive/notebooks/feature_history/`. Keep original basenames unless a collision requires a parent-category distinction.

- [ ] **Step 2: Move remaining notebooks and notes**

Verify that the identical source hash `7c79e09c71086549` remains represented by the active `03_all_dimensions_event_level_bilstm.ipynb` and both historical copies in the archive.

- [ ] **Step 3: Classify and move supporting files**

Use paper titles and content-oriented filenames for `references/papers/`; place `projectppt*`, `ppt2.pdf`, `EXP1.pdf`, `abstract.pdf`, `htmlCode.pdf`, and `PROJECT PART-1 FORMAT.docx` in `references/reports/`; place generated pipeline images in `references/figures/`. Ambiguous numbered PDFs may remain under `references/reports/` with their original names.

- [ ] **Step 4: Verify no unclassified user files remain in obsolete containers**

Run: `find code project -type f -print`

Expected: no files. Remove the empty `code/` and `project/` directories only after this check.

### Task 3: Make active notebook entry points Colab-compatible

**Files:**
- Modify: `models/baseline_bilstm/*.ipynb`
- Modify: `models/fslsm_bilstm/01_fslsm_sequence_preprocessing.ipynb`
- Modify: `models/fslsm_bilstm/02_fslsm_bilstm_training_and_evaluation.ipynb`
- Modify: `models/feature_engineering/fslsm_all_dimensions_scoring.ipynb`
- Create: `tools/validate_project.js`

**Interfaces:**
- Consumes: Stable paths from Task 1.
- Produces: Active notebooks with an explicit first-section configuration for Colab/Drive and a reusable static validator.

- [ ] **Step 1: Write the static validator before notebook edits**

`tools/validate_project.js` must exit nonzero when an active notebook is invalid JSON, an expected active path is missing, a documentation link target is missing, a credential filename is embedded as secret content, or a local-only `data/raw/` path remains unexplained in an active notebook.

- [ ] **Step 2: Run the validator and confirm it detects the legacy path problem**

Run: `node tools/validate_project.js`

Expected: FAIL identifying baseline/FSLSM notebooks that still assume `data/raw/...`.

- [ ] **Step 3: Add minimal Colab configuration cells and centralize paths**

For affected notebooks, add a top markdown section explaining Colab runtime setup and a code cell that mounts Drive when `google.colab` is available, defines `PROJECT_ROOT`, `DATA_ROOT`, `METADATA_ROOT`, and `ARTIFACT_ROOT`, and creates output directories. Replace only hard-coded repository data/artifact strings with those variables. Do not alter feature or model code.

- [ ] **Step 4: Preserve already coherent Colab entry points**

For the sequence-representation suite and Kaggle-downloading notebooks, retain their existing Colab setup and document their distinct Drive/Kaggle expectations in model READMEs rather than rewriting their artifact format.

- [ ] **Step 5: Run the validator and notebook parser**

Run: `node tools/validate_project.js`

Expected: PASS with 17 active notebooks validated and no unexplained local-only paths.

### Task 4: Write model guides and the project document

**Files:**
- Create: `PROJECT.md`
- Create: `models/baseline_bilstm/README.md`
- Create: `models/fslsm_bilstm/README.md`
- Create: `models/transformer/README.md`
- Create: `models/sequence_representation/README.md`
- Create: `models/feature_engineering/README.md`
- Create: `archive/README.md`

**Interfaces:**
- Consumes: Final model paths, actual notebook code, and inventory from Tasks 1–3.
- Produces: The primary project handoff and model-specific run instructions.

- [ ] **Step 1: Write focused model READMEs**

Each README must state the model objective, notebook order, required data/artifacts, outputs, Colab runtime requirements, and caveats. The sequence-representation README must explain that `00_shared_setup.ipynb` runs first and that `06_model_comparison.ipynb` consumes prior results.

- [ ] **Step 2: Write `PROJECT.md`**

Cover the research question, actual dataset tree, baseline token/time-gap design, all model families, run order, metrics, result-status vocabulary, known limitations, and improvement roadmap. Include tables for the model catalog and legacy filename mapping. Label ~0.85 AUC as user-reported and do not invent other scores.

- [ ] **Step 3: Document the improvement roadmap**

Prioritize: unified stratified validation and fixed seeds; leakage checks; strong logistic/gradient-boosted tabular baselines; class/threshold analysis with PR AUC; fair observation-window alignment; ablations for time gaps/token components/FSLSM; then attention, pretrained sequence encoders, calibrated ensembles, and uncertainty estimates.

- [ ] **Step 4: Validate documentation paths and scientific caveats**

Run: `node tools/validate_project.js`

Expected: PASS. Manually confirm that measured, user-reported, and pending results are distinct and that downstream unsupervised-feature evaluation is not equated with the end-to-end BiLSTM protocol.

### Task 5: Perform final integrity verification

**Files:**
- Modify: `docs/original-file-inventory.md` only if verification finds a mapping omission.
- Modify: `PROJECT.md` only if verification finds a broken or inaccurate path.

**Interfaces:**
- Consumes: Entire reorganized project.
- Produces: Evidence that the reorganization is complete without data loss.

- [ ] **Step 1: Recompute dataset checksums**

Run `sha256sum` on all six files below and compare exactly:

- `dataset/train/train/enrollment_train.csv`: `333ad96b2dfa0aa3662e99e37f8ea90d38bed32cb41174438076166cb3b2fd9f`
- `dataset/train/train/log_train.csv`: `8e732d01a488177bc7b663092f1000fd693969169c8503d483c1f80bd6e63354`
- `dataset/train/train/truth_train.csv`: `9a048bdbcb02827bf28dc621577c2c8d9df400d5ddfc7e5de3796e03f39ff0d6`
- `dataset/test/test/enrollment_test.csv`: `375d63f88d2da7c37c65322c61d6307e47f0971ff7924f0a59608ff104b8c4eb`
- `dataset/test/test/log_test.csv`: `cea2881663927131e1c6c16605e2f908a50f5a8bf33df8cf7de5a44ba8a164b3`
- `dataset/test/test/truth_test.csv`: `9f2ce06d0d1541527080b779f0548e98b6b4b1019f0e163b761cd1a3b7165e0f`

- [ ] **Step 2: Verify every original notebook is accounted for**

Compare the inventory mapping against all active and archived `.ipynb` files. Expected: 31 original notebooks accounted for, with no deletion; identical copies remain separate archived files where applicable.

- [ ] **Step 3: Run full static validation**

Run: `node tools/validate_project.js`

Expected: PASS, including JSON parse, expected-path, documentation-link, secret-scan, and legacy-path checks.

- [ ] **Step 4: Review the final tree**

Run: `find models archive shared references docs -maxdepth 4 -type f | sort`

Expected: all files are in an intentional model, archive, metadata, reference, or documentation location; `dataset/` remains unchanged.

- [ ] **Step 5: Record execution limits**

In the final handoff, state that static validation and integrity checks passed but full training was not run locally because the notebooks target Colab and the raw logs/GPU training are large.
