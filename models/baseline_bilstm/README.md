# Baseline Event-Sequence BiLSTM

This is the main end-to-end dropout baseline. It encodes each event as source, event type, object category, and a scalar log time gap, then trains an attention-based bidirectional LSTM.

## Files and order

1. `01_event_extraction_and_scaling.ipynb` — optional temporal event-count extraction and K-means diagnostics.
2. `02_sequence_preprocessing.ipynb` — required sequence construction; saves `artifacts/baseline_bilstm/processed_data.pt`.
3. `03_bilstm_training_and_evaluation.ipynb` — required model training and enrollment-level evaluation.

## Colab setup

Store the repository at `/content/drive/MyDrive/kdd_dropout_project`, or set `KDD_PROJECT_ROOT` before running the configuration cell. The project root must contain `dataset/` and `shared/metadata/`. Use a high-RAM runtime for preprocessing and a GPU for training.

## Important behavior

- Observation window: first 14 days of each learner's recorded activity.
- Token: `source_event_category`, separated internally into three vocabularies.
- Time feature: `log1p` elapsed seconds between consecutive events.
- Window length/stride: 150/75.
- Training vocabulary is reused on test; unknown components map to zero.
- Window probabilities are averaged per enrollment before metrics are calculated.

The stored active output reports ROC AUC `0.8333`. Earlier archived pipelines contain values near `0.85`; they are not interchangeable until their preprocessing is reproduced under one split.

The training notebook loads `processed_data.pt` with `weights_only=False` because the artifact contains NumPy arrays and a pandas DataFrame in addition to tensors. Only load the file created by this project's preprocessing notebook; do not use that setting with an untrusted `.pt` file.
