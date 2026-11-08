# MOOC Learning Style and Dropout Prediction

KDD Cup 2015 MOOC dropout-prediction research project built for Google Colab. The project compares an event-sequence BiLSTM baseline with FSLSM learning-style fusion, a Transformer, Markov mixtures, GRU representation learners, and a sequential graph neural network.

Start with [PROJECT.md](PROJECT.md) for:

- the dataset structure and Colab setup;
- the event-token and time-gap baseline design;
- model-by-model notebook order;
- recorded accuracy, precision, recall, F1, and ROC AUC results;
- methodological limitations and the improvement roadmap.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the GitHub synchronization, Colab, reproducibility, security, and review rules.

## Repository structure

```text
models/       Active model and feature-engineering notebooks
archive/      Historical and duplicate experiments
shared/       Small object/date metadata required by preprocessing
references/   Papers, reports, presentations, and figures
tools/        Static project validation and notebook normalization
docs/         Design, implementation plan, and original file mapping
```

The large raw dataset is intentionally excluded from Git. Download KDD Cup 2015 separately or place it in the local `dataset/` layout described in [PROJECT.md](PROJECT.md).

`Project.ipynb` is retained as the repository's original legacy notebook. Its identical organized copy is stored under `archive/notebooks/baseline_history/`.

## Quick validation

```bash
node tools/validate_project.js
```

The notebooks are designed primarily for Google Colab; neural-model training should use a GPU runtime.
