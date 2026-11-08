# Project Contribution Rules

These rules keep the project reproducible, synchronized with GitHub, and easy to review.

## 1. GitHub synchronization

- Every completed change must be committed and pushed to GitHub before the task is considered complete.
- Start work by checking the current branch and pulling the latest remote changes:

  ```bash
  git status
  git pull --rebase origin main
  ```

- Keep commits focused. One commit should represent one logical change.
- Use descriptive commit messages such as `feat: add time-gap ablation` or `docs: update Colab run instructions`.
- Prefer a feature branch and pull request for model/code changes. Direct pushes to `main` are acceptable only for small documentation or maintenance changes.
- Never rewrite published history or force-push unless explicitly approved.
- Finish with a clean working tree:

  ```bash
  git status --short
  git log -1 --oneline
  git ls-remote origin refs/heads/main
  ```

## 2. Data and security

- Never commit `dataset/`, Kaggle credentials, API keys, tokens, private environment files, or local secrets.
- Do not commit generated model weights, large NumPy artifacts, checkpoints, or temporary results unless the project explicitly promotes them as a release artifact.
- Keep small reusable metadata such as `shared/metadata/object.csv` under version control.
- Run a credential and large-file check before pushing.

## 3. Notebook rules

- Active notebooks belong under the appropriate `models/` directory; historical experiments belong under `archive/`.
- Keep the Colab configuration near the top of every active notebook.
- Use `KDD_PROJECT_ROOT`, `DATA_ROOT`, `METADATA_ROOT`, and `MODEL_ARTIFACT_ROOT`; do not scatter machine-specific absolute paths through model code.
- Run notebooks from the first cell in order. Do not rely on hidden state from an earlier session.
- Keep preprocessing, model training, and evaluation stages separate when they produce reusable artifacts.
- Do not overwrite a previous experiment's outputs without recording the configuration and seed.

## 4. Reproducibility rules

Every new experiment must record:

- observation window and enrollment-level split;
- random seed(s);
- feature/token definition;
- model and optimizer settings;
- threshold-selection method;
- accuracy, precision, recall, F1, ROC AUC, and preferably PR AUC;
- output artifact paths and the commit SHA used to produce them.

Use train-only fitting for vocabularies, scalers, dimensionality reduction, and feature selection. Never split windows from the same enrollment across train and validation folds.

## 5. Scientific reporting

- Update `PROJECT.md` whenever a model, metric, preprocessing rule, or run procedure changes.
- Label results as `recorded`, `reproduced`, `user-reported`, or `pending`.
- Do not compare model families unless their observation window, enrollment split, label handling, and evaluation protocol match.
- Treat the supplied test truth as a held-out development evaluation once it has been inspected; reserve a genuinely untouched split for final claims where possible.
- Include a short note explaining any result that improves or worsens the baseline.

## 6. Required checks before pushing

Run from the project root:

```bash
node tools/validate_project.js
git diff --check
git status --short
```

For notebook or preprocessing changes, also verify JSON parsing and compile the changed Python cells. For model changes, record the exact Colab runtime and whether a GPU was used.

## 7. Recommended pull-request checklist

- [ ] The change is documented in `PROJECT.md` or the relevant model README.
- [ ] No dataset, credential, checkpoint, or generated artifact is accidentally staged.
- [ ] Validation checks pass.
- [ ] Notebook paths and artifact paths work from a fresh Colab session.
- [ ] Seeds, splits, metrics, and output locations are recorded.
- [ ] The PR explains how the change compares with the current baseline.
- [ ] The branch is pushed and the GitHub PR or commit URL is included in the handoff.

