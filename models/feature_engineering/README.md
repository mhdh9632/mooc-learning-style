# Feature Engineering and Clustering

This directory contains optional analyses that create interpretable behavior features rather than the main sequence classifier.

- `event_scaling_and_kmeans.ipynb` — Colab/Drive event-count extraction, scaling, temporal slices, and K-means diagnostics.
- `fslsm_all_dimensions_scoring.ipynb` — four clickstream-derived FSLSM dimensions saved under `artifacts/feature_engineering/`.
- `proportional_features.ipynb` — raw and proportional event representations with K-means and deep embedded clustering diagnostics.

The latter two notebooks use the repository-wide Colab configuration and default to `/content/drive/MyDrive/kdd_dropout_project`. Set `KDD_PROJECT_ROOT` when necessary. The event-scaling notebook preserves its original `/content/drive/MyDrive/KDD_Cup_2015` contract.

These notebooks are exploratory. Cluster labels are not ground truth learning styles, and cluster-quality metrics do not establish dropout-prediction improvement.
