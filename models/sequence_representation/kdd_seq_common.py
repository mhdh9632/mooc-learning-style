"""Shared helpers for the KDD Cup 2015 sequence-clustering experiments.

Every method notebook (Markov mixture, embeddings, session archetypes, GNN) uses the same
artifacts, the same quality / stability metrics and the same downstream protocol, so the
numbers are comparable across methods.
"""
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score, silhouette_score,
                             davies_bouldin_score, calinski_harabasz_score, roc_auc_score,
                             average_precision_score, f1_score, precision_recall_curve)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

HGB_PARAMS = dict(max_iter=250, learning_rate=0.06, max_leaf_nodes=31,
                  l2_regularization=1.0, early_stopping=False)


# ----------------------------------------------------------------------------- artifacts
def load_artifacts(work):
    d = {}
    for s in ("train", "test"):
        z = np.load(f"{work}/seq_{s}.npz")
        d[f"ids_{s}"] = z["ids"]
        d[f"tok_{s}"] = z["tokens"]
        d[f"off_{s}"] = z["offsets"]
        d[f"F_{s}"] = sp.load_npz(f"{work}/F_{s}.npz").tocsr()
        d[f"base_{s}"] = pd.read_pickle(f"{work}/base_{s}.pkl")
        d[f"y_{s}"] = np.load(f"{work}/y_{s}.npy")
        d[f"eval_{s}"] = np.load(f"{work}/eval_{s}.npy")
    d["vocab"] = json.load(open(f"{work}/vocab.json"))
    return d


def onehot(labels, K):
    out = np.zeros((len(labels), K), dtype=np.float32)
    out[np.arange(len(labels)), labels] = 1.0
    return out


# ----------------------------------------------------------------------------- clustering quality
def cramers_v(labels, y):
    ct = pd.crosstab(np.asarray(labels), np.asarray(y)).values
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(ct, correction=False)[0]
    return float(np.sqrt(chi2 / (ct.sum() * (min(ct.shape) - 1))))


def cluster_quality(Z, labels, sample=20000, seed=0):
    """Silhouette / DB / CH computed on a COMMON embedding Z (and a random sample for speed)."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(labels), min(sample, len(labels)), replace=False)
    l = labels[idx]
    shares = np.bincount(labels) / len(labels)
    out = dict(n_clusters=int((shares > 0).sum()), min_cluster_share=float(shares[shares > 0].min()))
    if len(np.unique(l)) < 2:
        out.update(silhouette=np.nan, davies_bouldin=np.nan, calinski_harabasz=np.nan)
        return out
    out.update(silhouette=float(silhouette_score(Z[idx], l)),
               davies_bouldin=float(davies_bouldin_score(Z[idx], l)),
               calinski_harabasz=float(calinski_harabasz_score(Z[idx], l)))
    return out


def volume_agreement(labels, n_events, seed=0):
    """How much do the clusters just re-encode activity volume?
    ARI/NMI against a K-Means on log1p(n_events) with the same K, plus eta^2 of log-volume."""
    labels = np.asarray(labels)
    K = len(np.unique(labels))
    lv = np.log1p(np.asarray(n_events, dtype=float))
    vol = KMeans(K, n_init=5, random_state=seed).fit_predict(lv.reshape(-1, 1))
    g = pd.Series(lv).groupby(labels)
    between = (g.count() * (g.mean() - lv.mean()) ** 2).sum()
    total = ((lv - lv.mean()) ** 2).sum()
    return dict(ari_vs_volume=float(adjusted_rand_score(vol, labels)),
                nmi_vs_volume=float(normalized_mutual_info_score(vol, labels)),
                eta2_logvolume=float(between / total))


def cluster_profile(labels, y, n_events):
    df = pd.DataFrame(dict(c=np.asarray(labels), y=np.asarray(y), n=np.asarray(n_events)))
    g = df.groupby("c")
    out = pd.DataFrame(dict(size=g.size(), share=g.size() / len(df), dropout=g.y.mean(),
                            mean_events=g.n.mean(), median_events=g.n.median()))
    return out.round(3)


# ----------------------------------------------------------------------------- stability
def seed_stability(label_list):
    aris = [adjusted_rand_score(a, b) for i, a in enumerate(label_list) for b in label_list[i + 1:]]
    return dict(seed_ari_mean=float(np.mean(aris)), seed_ari_min=float(np.min(aris)))


def bootstrap_stability(fit, predict, n_rows, ref_labels, n_boot=10, frac=0.8, seed=0):
    """fit(idx, seed) -> model fitted on a random 80% subsample; predict(model, np.arange(n)) -> labels.
    Reports ARI between each subsample model's labelling of ALL rows and the reference labelling."""
    rng = np.random.default_rng(seed)
    aris = []
    for b in range(n_boot):
        idx = rng.choice(n_rows, int(frac * n_rows), replace=False)
        m = fit(idx, seed + b)
        aris.append(adjusted_rand_score(ref_labels, predict(m, np.arange(n_rows))))
    return dict(boot_ari_mean=float(np.mean(aris)), boot_ari_std=float(np.std(aris)))


def transfer_consistency(lab_tr, y_tr, lab_te, y_te, K):
    """Train -> test: same cluster proportions? same dropout rate per cluster?"""
    p_tr = np.bincount(lab_tr, minlength=K) / len(lab_tr)
    p_te = np.bincount(lab_te, minlength=K) / len(lab_te)
    d_tr = pd.Series(y_tr).groupby(lab_tr).mean().reindex(range(K))
    d_te = pd.Series(y_te).groupby(lab_te).mean().reindex(range(K))
    ok = d_tr.notna() & d_te.notna()
    return dict(share_L1_train_vs_test=float(np.abs(p_tr - p_te).sum()),
                dropout_corr_train_vs_test=float(np.corrcoef(d_tr[ok], d_te[ok])[0, 1]) if ok.sum() > 2 else np.nan,
                cramers_v_train=cramers_v(lab_tr, y_tr), cramers_v_test=cramers_v(lab_te, y_te))


# ----------------------------------------------------------------------------- downstream
def _hgb(seed):
    return HistGradientBoostingClassifier(random_state=seed, **HGB_PARAMS)


def downstream_cv(base, y, variants, n_sample=60000, n_splits=3, n_repeats=3, seed=0):
    """Repeated stratified CV on a fixed stratified sample of TRAIN.
    `variants`: {name: array (N_train x d)} of extra features added to the baseline.
    Embeddings must be unsupervised (no dropout label used) so fitting them on all of train is not leaky.
    Returns (summary, per_fold_auc)."""
    N = len(y)
    idx = (train_test_split(np.arange(N), train_size=n_sample, stratify=y, random_state=seed)[0]
           if n_sample and n_sample < N else np.arange(N))
    Xb = np.asarray(base.values, dtype=np.float32)[idx]
    yy = np.asarray(y)[idx]
    Xs = {"baseline": Xb}
    for k, v in variants.items():
        Xs[k] = np.hstack([Xb, np.asarray(v, dtype=np.float32)[idx]])
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    rows = []
    for f, (tr, va) in enumerate(rskf.split(Xb, yy)):
        for nm, X in Xs.items():
            p = _hgb(seed).fit(X[tr], yy[tr]).predict_proba(X[va])[:, 1]
            rows.append((f, nm, roc_auc_score(yy[va], p)))
    wide = pd.DataFrame(rows, columns=["fold", "variant", "auc"]).pivot(index="fold", columns="variant", values="auc")
    wide = wide[list(Xs)]
    summ = pd.DataFrame(dict(auc_mean=wide.mean(), auc_std=wide.std()))
    delta = wide.sub(wide["baseline"], axis=0)
    summ["delta_vs_base"] = delta.mean()
    summ["delta_std"] = delta.std()
    summ["folds_better"] = (delta > 0).sum()
    return summ.round(5), wide


def final_test_eval(base_tr, y_tr, base_te, y_te, var_tr, var_te, seeds=(0, 1, 2)):
    """Train on (80% of) full train, choose the F1 threshold on the other 20%, score the real TEST set."""
    Xtr_all = {"baseline": np.asarray(base_tr.values, dtype=np.float32)}
    Xte_all = {"baseline": np.asarray(base_te.values, dtype=np.float32)}
    for k in var_tr:
        Xtr_all[k] = np.hstack([Xtr_all["baseline"], np.asarray(var_tr[k], dtype=np.float32)])
        Xte_all[k] = np.hstack([Xte_all["baseline"], np.asarray(var_te[k], dtype=np.float32)])
    rows = []
    for nm in Xtr_all:
        for s in seeds:
            Xa, Xv, ya, yv = train_test_split(Xtr_all[nm], y_tr, test_size=0.2, stratify=y_tr, random_state=s)
            m = _hgb(s).fit(Xa, ya)
            pv = m.predict_proba(Xv)[:, 1]
            p, r, t = precision_recall_curve(yv, pv)
            f = 2 * p * r / np.maximum(p + r, 1e-12)
            thr = t[np.argmax(f[:-1])]
            pt = m.predict_proba(Xte_all[nm])[:, 1]
            rows.append(dict(variant=nm, seed=s, auc=roc_auc_score(y_te, pt),
                             ap=average_precision_score(y_te, pt), f1=f1_score(y_te, pt >= thr)))
    df = pd.DataFrame(rows)
    out = df.groupby("variant", sort=False)[["auc", "ap", "f1"]].agg(["mean", "std"]).round(5)
    out["auc_delta_vs_base"] = (out[("auc", "mean")] - out.loc["baseline", ("auc", "mean")]).round(5)
    return out
