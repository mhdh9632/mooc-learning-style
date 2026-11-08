# Original File Inventory and Move Map

This inventory records the project layout before the 2026-11-09 reorganization. Every original notebook is preserved at an active or archived destination.

## Active notebooks

| Original path | New path |
|---|---|
| `code/01_Raw_Event_Extraction_and_Scaling (3).ipynb` | `models/baseline_bilstm/01_event_extraction_and_scaling.ipynb` |
| `code/02_Data_Preprocessing.ipynb` | `models/baseline_bilstm/02_sequence_preprocessing.ipynb` |
| `code/03_Model_Training_Evaluation.ipynb` | `models/baseline_bilstm/03_bilstm_training_and_evaluation.ipynb` |
| `project/02_Data_Preprocessing_FSLSM.ipynb` | `models/fslsm_bilstm/01_fslsm_sequence_preprocessing.ipynb` |
| `project/03_Model_Training_Evaluation_FSLSM.ipynb` | `models/fslsm_bilstm/02_fslsm_bilstm_training_and_evaluation.ipynb` |
| `code/exp1/fslsm+eventlvl.ipynb` | `models/fslsm_bilstm/03_all_dimensions_event_level_bilstm.ipynb` |
| `code/exp1/fslsm_transformer.ipynb` | `models/transformer/fslsm_transformer.ipynb` |
| `code/kdd_seq_00_setup (2).ipynb` | `models/sequence_representation/00_shared_setup.ipynb` |
| `code/kdd_seq_01_markov_mixture (2).ipynb` | `models/sequence_representation/01_markov_mixture.ipynb` |
| `code/kdd_seq_02_next_token_gru.ipynb` | `models/sequence_representation/02_next_token_gru.ipynb` |
| `code/kdd_seq_03_gru_autoencoder (1).ipynb` | `models/sequence_representation/03_gru_autoencoder.ipynb` |
| `code/kdd_seq_04_contrastive_gru (1).ipynb` | `models/sequence_representation/04_contrastive_gru.ipynb` |
| `code/kdd_seq_06_sequential_gnn (1).ipynb` | `models/sequence_representation/05_sequential_gnn.ipynb` |
| `code/kdd_seq_05_all_in_one_neural_and_comparison (1).ipynb` | `models/sequence_representation/06_model_comparison.ipynb` |
| `code/KDD_Cup_2015_Colab_Drive.ipynb` | `models/feature_engineering/event_scaling_and_kmeans.ipynb` |
| `code/exp1/FSLSM_All_Dimensions_Scoring.ipynb` | `models/feature_engineering/fslsm_all_dimensions_scoring.ipynb` |
| `code/exp1/proportional_features_1 (2).ipynb` | `models/feature_engineering/proportional_features.ipynb` |

The companion module `code/kdd_seq_common (1).py` moves to `models/sequence_representation/kdd_seq_common.py`.

## Archived notebooks

| Original path | New path |
|---|---|
| `code/Project.ipynb` | `archive/notebooks/baseline_history/Project.ipynb` |
| `code/Project_v2.ipynb` | `archive/notebooks/baseline_history/Project_v2.ipynb` |
| `code/Projectpart1.ipynb` | `archive/notebooks/baseline_history/Projectpart1.ipynb` |
| `code/archive/01_Exploratory_Data_Analysis.ipynb` | `archive/notebooks/baseline_history/01_Exploratory_Data_Analysis.ipynb` |
| `code/archive/02_Data_Preprocessing.ipynb` | `archive/notebooks/baseline_history/02_Data_Preprocessing.ipynb` |
| `code/archive/03_Model_Training_Evaluation.ipynb` | `archive/notebooks/baseline_history/03_Model_Training_Evaluation.ipynb` |
| `code/exp1/old_project.ipynb` | `archive/notebooks/baseline_history/old_project.ipynb` |
| `code/exp1/04_Student1_LearningStyle_LSTM.ipynb` | `archive/notebooks/fslsm_history/04_Student1_LearningStyle_LSTM.ipynb` |
| `code/exp1/Untitled5_fixed (1).ipynb` | `archive/notebooks/fslsm_history/Untitled5_fixed (1).ipynb` |
| `code/exp1/Untitled5_fixed.ipynb` | `archive/notebooks/fslsm_history/Untitled5_fixed.ipynb` |
| `code/exp1/alldimscor1.ipynb` | `archive/notebooks/fslsm_history/alldimscor1.ipynb` |
| `code/exp1/collabaldim.ipynb` | `archive/notebooks/fslsm_history/collabaldim.ipynb` |
| `code/archive/proportional_features.ipynb` | `archive/notebooks/feature_history/proportional_features.ipynb` |
| `code/exp1/proportional_cluster.ipynb` | `archive/notebooks/feature_history/proportional_cluster.ipynb` |

The three original files `fslsm+eventlvl.ipynb`, `Untitled5_fixed.ipynb`, and `Untitled5_fixed (1).ipynb` have the same normalized cell-source hash (`7c79e09c71086549`). One is retained as the active, descriptively named notebook and two remain in the archive.

## Additional Colab imports (2026-11-09)

The following notebooks were added to the project root from Google Colab and have now been sorted without deleting any copies:

| Imported path | Destination | Classification |
|---|---|---|
| `Project_v2.ipynb` | `archive/notebooks/baseline_history/imported_colab/Project_v2.ipynb` | Archived baseline/sequence implementation |
| `old_project.ipynb` | `archive/notebooks/baseline_history/imported_colab/old_project.ipynb` | Archived baseline/sequence implementation; duplicate of the imported `Project_v2.ipynb` |
| `old_1.ipynb` | `archive/notebooks/baseline_history/imported_colab/old_1.ipynb` | Archived BiLSTM variant with first-week evaluation |
| `Untitled2.ipynb` | `models/fslsm_bilstm/04_fslsm_static_feature_comparison.ipynb` | Active FSLSM comparison: sequence-only versus sequence + four static FSLSM scores |

The imported `Project_v2.ipynb` and `old_project.ipynb` are retained separately for provenance even though their normalized notebook contents are identical.

## Dataset integrity baseline

| File | SHA-256 |
|---|---|
| `dataset/train/train/enrollment_train.csv` | `333ad96b2dfa0aa3662e99e37f8ea90d38bed32cb41174438076166cb3b2fd9f` |
| `dataset/train/train/log_train.csv` | `8e732d01a488177bc7b663092f1000fd693969169c8503d483c1f80bd6e63354` |
| `dataset/train/train/truth_train.csv` | `9a048bdbcb02827bf28dc621577c2c8d9df400d5ddfc7e5de3796e03f39ff0d6` |
| `dataset/test/test/enrollment_test.csv` | `375d63f88d2da7c37c65322c61d6307e47f0971ff7924f0a59608ff104b8c4eb` |
| `dataset/test/test/log_test.csv` | `cea2881663927131e1c6c16605e2f908a50f5a8bf33df8cf7de5a44ba8a164b3` |
| `dataset/test/test/truth_test.csv` | `9f2ce06d0d1541527080b779f0548e98b6b4b1019f0e163b761cd1a3b7165e0f` |

## Shared metadata integrity baseline

| Original file | New file | SHA-256 |
|---|---|---|
| `code/archive/object.csv` | `shared/metadata/object.csv` | `80d7f475a681614e901005d358629d33d330020a7a2cad1b689601c0ff65eaff` |
| `code/archive/date.csv` | `shared/metadata/date.csv` | `1d9805e0ac7cbfa444fcf5970257108b68c0e54940793d84f92574414ac4f332` |
