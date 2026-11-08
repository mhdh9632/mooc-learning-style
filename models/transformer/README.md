# FSLSM Transformer

`fslsm_transformer.ipynb` replaces the BiLSTM backbone with a Transformer encoder while keeping the source/event/category embeddings and time-gap input.

It evaluates three fusion modes:

- `none`: sequential features only.
- `concat`: late fusion of the pooled sequence representation and FSLSM vector.
- `token`: early fusion by inserting a projected FSLSM token into the sequence.

The notebook is self-contained for Colab, mounts Drive, and downloads the KDD dataset through Kaggle. Supply credentials through the user's private Kaggle configuration folder; do not add them to the repository. Use a GPU runtime. The notebook truncates long sequences for Transformer memory control, so comparisons with the BiLSTM require an aligned preprocessing protocol.

Stored outputs have not been promoted to the main results table because this notebook needs a clean rerun under the authoritative split and observation window.
