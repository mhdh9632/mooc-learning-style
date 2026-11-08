# FSLSM-Augmented BiLSTM

These notebooks test whether clickstream-derived Felder–Silverman learning-style scores improve the event-sequence BiLSTM.

## Files and order

1. `01_fslsm_sequence_preprocessing.ipynb` — builds the same event windows as the baseline and adds four window-level FSLSM scores.
2. `02_fslsm_bilstm_training_and_evaluation.ipynb` — trains sequence-only and sequence-plus-FSLSM variants with the same windows and seed.
3. `03_all_dimensions_event_level_bilstm.ipynb` — a self-contained Colab/Kaggle experiment using eight static features (scores plus related dimension values).
4. `04_fslsm_static_feature_comparison.ipynb` — imported Colab comparison of sequence-only versus sequence-plus-four-static-FSLSM-score BiLSTMs.

The first two notebooks share `artifacts/fslsm_bilstm/processed_data_fslsm.pt`. Store the repository at `/content/drive/MyDrive/kdd_dropout_project` or set `KDD_PROJECT_ROOT`. The third notebook keeps its original Kaggle download workflow and requires the user's Drive-hosted Kaggle configuration.

The FSLSM mappings are behavioral proxies, not questionnaire measurements. Dwell time, retries, navigation jumps, and event-category rules should be treated as assumptions requiring ablation and construct-validity checks.

The self-contained notebook's stored outputs show ROC AUC `0.8323` without FSLSM and `0.8320` with all four dimensions; this run does not demonstrate improvement.

The two-stage training notebook loads `processed_data_fslsm.pt` with `weights_only=False` because the trusted local artifact contains NumPy arrays and a pandas DataFrame. Do not load an untrusted `.pt` file through this path.
