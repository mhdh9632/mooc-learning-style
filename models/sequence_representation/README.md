# Sequence Representation and Clustering Suite

This suite learns unsupervised representations of learner event sequences, clusters those representations, and tests whether cluster/embedding features add dropout-prediction value beyond a shared tabular baseline.

## Drive contract

The original Colab paths are intentionally preserved:

```text
/content/drive/MyDrive/kdd2015   # train/, test/, object.csv
/content/drive/MyDrive/kdd_seq   # sequence artifacts, kdd_seq_common.py, results/
```

Edit `DATA_ROOT`, `OBJECT_PATH`, and `WORK` when using another location. Place `kdd_seq_common.py` in the configured working directory when required.

For this repository's reorganized Drive tree, use these exact settings in `00_shared_setup.ipynb`:

```python
DATA_ROOT = "/content/drive/MyDrive/kdd_dropout_project/dataset"
OBJECT_PATH = "/content/drive/MyDrive/kdd_dropout_project/shared/metadata/object.csv"
WORK = "/content/drive/MyDrive/kdd_seq"
```

The independent `OBJECT_PATH` is required because `object.csv` is not inside `dataset/`. If you choose a different `WORK`, update the `WORK` value in notebooks `01` through `06` as well; those notebooks do not automatically inherit variables from a separate Colab runtime.

## Run order

1. `00_shared_setup.ipynb` — build train-only vocabulary, gap tokens, compact sequences, sparse transition features, labels, baseline features, and a shared SVD evaluation space.
2. `01_markov_mixture.ipynb` — mixture of first-order Markov transition models.
3. `02_next_token_gru.ipynb` — causal next-token GRU language model.
4. `03_gru_autoencoder.ipynb` — bidirectional GRU encoder/decoder with normalized bottleneck.
5. `04_contrastive_gru.ipynb` — masked/cropped sequence augmentations with contrastive learning.
6. `05_sequential_gnn.ipynb` — SR-GNN-style session graph with contrastive training.
7. `06_model_comparison.ipynb` — consolidate models, compare clusters, and evaluate feature fusion.

The model notebooks are independent after shared setup, but the final comparison expects their result files. Neural notebooks retain the 128 most recent tokens and test cluster counts across a grid. Volume residualization and random-feature controls are included to test whether representations encode more than activity quantity.

## Interpretation caution

These are not direct replacements for the end-to-end BiLSTM. They add unsupervised representations to a downstream classifier. The stored comparison shows very small AUC differences around a `0.85318 ± 0.00205` cross-validation baseline; the strongest displayed gains are on the order of `10^-4`. Treat these as hypotheses until rerun with fixed splits, confidence intervals, and an untouched final test.
