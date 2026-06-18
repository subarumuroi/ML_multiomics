"""
wgcna.py
========
Weighted Gene Co-expression Network Analysis on the BaseMethod interface, with a
``reduce()`` method that uses WGCNA as **dimensionality reduction**: correlated
features collapse into module eigengenes (+ optional hub features), giving a
small samples x module matrix that can feed a supervised method with no leakage
(modules are built without the labels). Reference: Langfelder & Horvath (2008).

Ported from multiomics_integration.wgcna (pure numpy/scipy). Changes: the module
pipeline runs without a target (trait correlation is optional), and the
scale-free-fit messages are ASCII (`R^2`) and go through logging, not print.

handles_missing = False (correlations need complete data; impute just-in-time).
WGCNA was designed for n >= 15-20; at small n treat modules as exploratory.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr

from ..base import BaseMethod

logger = logging.getLogger(__name__)

VALID_CORR_METHODS = {"pearson", "spearman"}
VALID_NETWORK_TYPES = {"unsigned", "signed", "signed_hybrid"}


def _validate_corr_method(method):
    if method not in VALID_CORR_METHODS:
        raise ValueError(f"Unknown correlation method '{method}'. Expected {sorted(VALID_CORR_METHODS)}.")


def _validate_network_type(network_type):
    if network_type not in VALID_NETWORK_TYPES:
        raise ValueError(f"Unknown network_type '{network_type}'. Expected {sorted(VALID_NETWORK_TYPES)}.")


def compute_correlation_matrix(X, method="pearson"):
    if method == "spearman":
        corr = np.corrcoef(pd.DataFrame(X).rank().values.T)
    else:
        corr = np.corrcoef(X.T)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1, 1)


def adjacency_from_correlation(corr, power=6, network_type="unsigned"):
    _validate_network_type(network_type)
    corr = np.clip(np.asarray(corr, dtype=float), -1.0, 1.0)
    if network_type == "unsigned":
        adj = np.abs(corr) ** power
    elif network_type == "signed":
        adj = ((1.0 + corr) / 2.0) ** power
    else:
        adj = np.where(corr > 0, corr ** power, 0.0)
    adj = (adj + adj.T) / 2.0
    np.fill_diagonal(adj, 0.0)
    return np.clip(adj, 0.0, 1.0)


def _compute_module_eigengene(X_mod):
    X_mod = np.asarray(X_mod, dtype=float)
    Xc = X_mod - X_mod.mean(axis=0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    eig = U[:, 0] * S[0]
    avg = Xc.mean(axis=1)
    if np.std(avg) > 0 and np.std(eig) > 0:
        r = np.corrcoef(eig, avg)[0, 1]
        if np.isfinite(r) and r < 0:
            eig = -eig
    return eig


def compute_module_eigengenes(X, module_assignments):
    eigengenes = {}
    for mod in sorted(module_assignments["Module"].unique()):
        if mod == 0:
            continue
        idx = np.where((module_assignments["Module"] == mod).values)[0]
        if len(idx) < 2:
            continue
        try:
            eigengenes[mod] = _compute_module_eigengene(X[:, idx])
        except np.linalg.LinAlgError:
            continue
    return eigengenes


def pick_soft_threshold(X, powers=None, method="pearson", network_type="unsigned", target_r2=0.8):
    _validate_corr_method(method)
    _validate_network_type(network_type)
    if powers is None:
        powers = list(range(2, 21))
    corr = compute_correlation_matrix(X, method=method)
    results = []
    for beta in powers:
        adj = adjacency_from_correlation(corr, power=beta, network_type=network_type)
        k = adj.sum(axis=0)
        k_pos = k[k > 0]
        if len(k_pos) < 5:
            results.append({"power": beta, "r_squared": 0.0, "mean_connectivity": k.mean()})
            continue
        n_bins = max(5, min(20, len(k_pos) // 3))
        hist, edges = np.histogram(k_pos, bins=n_bins)
        centers = (edges[:-1] + edges[1:]) / 2
        mask = hist > 0
        if mask.sum() < 3:
            results.append({"power": beta, "r_squared": 0.0, "mean_connectivity": k.mean()})
            continue
        log_k = np.log10(centers[mask])
        log_pk = np.log10(hist[mask].astype(float))
        slope, intercept = np.polyfit(log_k, log_pk, 1)
        pred = slope * log_k + intercept
        ss_res = np.sum((log_pk - pred) ** 2)
        ss_tot = np.sum((log_pk - log_pk.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        results.append({"power": beta, "r_squared": -np.sign(slope) * r2, "mean_connectivity": k.mean()})
    df = pd.DataFrame(results)
    good = df[df["r_squared"] > target_r2]
    if len(good) > 0:
        best, rule = good.iloc[0], "first_above_target"
    else:
        best, rule = df.loc[df["r_squared"].idxmax()], "best_available"
    return {"power": int(best["power"]), "r_squared": best["r_squared"], "target_r2": float(target_r2),
            "selection_rule": rule, "threshold_met": bool(len(good) > 0), "all_results": df}


def compute_adjacency(X, power=6, method="pearson", network_type="unsigned"):
    _validate_corr_method(method)
    _validate_network_type(network_type)
    return adjacency_from_correlation(compute_correlation_matrix(X, method=method),
                                      power=power, network_type=network_type)


def compute_tom(adjacency):
    adjacency = np.clip((np.asarray(adjacency, float) + np.asarray(adjacency, float).T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(adjacency, 0.0)
    k = adjacency.sum(axis=0)
    L = adjacency @ adjacency
    k_min = np.minimum(k[np.newaxis, :], k[:, np.newaxis])
    num = L + adjacency
    den = k_min + 1 - adjacency
    tom = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    tom = np.clip((tom + tom.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(tom, 1.0)
    return tom


def detect_modules(tom, feature_names=None, min_module_size=3, cut_height=None):
    p = tom.shape[0]
    names = feature_names or [f"f{i}" for i in range(p)]
    dist = np.clip(1 - tom, 0, 1)
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist, checks=False), method="average")
    if cut_height is None:
        cut_height = 0.40 if p <= 25 else (0.35 if p <= 100 else 0.25)
    labels = fcluster(Z, t=cut_height, criterion="distance")
    counts = pd.Series(labels).value_counts()
    small = counts[counts < min_module_size].index
    labels = np.array([0 if l in small else l for l in labels])
    uniq = sorted(set(labels) - {0})
    remap = {0: 0}
    for i, m in enumerate(uniq, 1):
        remap[m] = i
    labels = np.array([remap[l] for l in labels])
    return pd.DataFrame({"Feature": names, "Module": labels}), Z


def merge_modules_by_eigengene(X, module_assignments, threshold=0.25):
    merged = module_assignments.copy()
    eig = compute_module_eigengenes(X, merged)
    if len(eig) < 2:
        return merged
    modules = sorted(eig)
    me = np.column_stack([eig[m] for m in modules])
    me_corr = np.nan_to_num(np.corrcoef(me.T), nan=0.0)
    np.fill_diagonal(me_corr, 1.0)
    diss = np.clip((1.0 - np.abs(me_corr)), 0.0, 1.0)
    diss = (diss + diss.T) / 2.0
    np.fill_diagonal(diss, 0.0)
    Z = linkage(squareform(diss, checks=False), method="average")
    lab = fcluster(Z, t=threshold, criterion="distance")
    mod_to_cl = {m: l for m, l in zip(modules, lab)}
    cl_to_new = {c: i + 1 for i, c in enumerate(sorted(set(lab)))}
    merged["Module"] = merged["Module"].map(lambda m: 0 if m == 0 else cl_to_new[mod_to_cl[m]])
    return merged


def module_trait_correlation(X, y_encoded, module_assignments, method="pearson"):
    _validate_corr_method(method)
    eig = compute_module_eigengenes(X, module_assignments)
    results = []
    for mod in sorted(module_assignments["Module"].unique()):
        if mod == 0 or mod not in eig:
            continue
        idx = np.where((module_assignments["Module"] == mod).values)[0]
        r, p = (spearmanr if method == "spearman" else pearsonr)(eig[mod], y_encoded)
        results.append({"Module": mod, "N_Features": len(idx), "Correlation": r,
                        "P_Value": p, "Abs_Correlation": abs(r)})
    if not results:
        return pd.DataFrame(columns=["Module", "N_Features", "Correlation", "P_Value", "Abs_Correlation"]), eig
    return pd.DataFrame(results).sort_values("Abs_Correlation", ascending=False).reset_index(drop=True), eig


def identify_hub_features(X, module_assignments, adjacency, y_encoded=None, top_n=5, method="pearson"):
    _validate_corr_method(method)
    all_features = module_assignments["Feature"].tolist()
    records = []
    for mod in sorted(module_assignments["Module"].unique()):
        if mod == 0:
            continue
        mask = module_assignments["Module"] == mod
        idx = np.where(mask.values)[0]
        if len(idx) < 2:
            continue
        adj_sub = adjacency[np.ix_(idx, idx)]
        k_in = adj_sub.sum(axis=0)
        k_norm = k_in / k_in.max() if k_in.max() > 0 else k_in
        for i, fi in enumerate(idx):
            rec = {"Feature": all_features[fi], "Module": mod,
                   "Intramodular_Connectivity": k_in[i], "Normalized_Connectivity": k_norm[i]}
            if y_encoded is not None:
                r, p = (spearmanr if method == "spearman" else pearsonr)(X[:, fi], y_encoded)
                rec["Trait_Correlation"] = r
                rec["Trait_P_Value"] = p
                rec["Gene_Significance"] = abs(r)
            records.append(rec)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["Hub_Score"] = (df["Normalized_Connectivity"] * df["Gene_Significance"]
                       if "Gene_Significance" in df.columns else df["Normalized_Connectivity"])
    df["Is_Hub"] = False
    for mod in df["Module"].unique():
        top = df.loc[df["Module"] == mod].nlargest(top_n, "Hub_Score")
        df.loc[top.index, "Is_Hub"] = True
    return df.sort_values(["Module", "Hub_Score"], ascending=[True, False]).reset_index(drop=True)


def run_wgcna(X, y_encoded=None, feature_names=None, power=None, corr_method="spearman",
              network_type="unsigned", min_module_size=None, module_cut_height=None,
              merge_cut_height=0.25, top_n_hubs=5):
    """Full WGCNA pipeline. y_encoded is OPTIONAL (None -> skip trait correlation)."""
    _validate_corr_method(corr_method)
    _validate_network_type(network_type)
    names = feature_names or [f"f{i}" for i in range(X.shape[1])]
    p = X.shape[1]
    if min_module_size is None:
        min_module_size = min(20, max(3, p // 10))

    threshold_result = None
    if power is None:
        threshold_result = pick_soft_threshold(X, method=corr_method, network_type=network_type)
        power = threshold_result["power"]
        reason = "threshold met" if threshold_result["threshold_met"] else "best available"
        logger.info("WGCNA soft-threshold power: %d (R^2=%.3f; %s)",
                    power, threshold_result["r_squared"], reason)

    adj = compute_adjacency(X, power=power, method=corr_method, network_type=network_type)
    tom = compute_tom(adj)

    cut = module_cut_height
    if cut is None:
        cut = 0.40 if p <= 25 else (0.35 if p <= 100 else 0.25)
    modules_df, Z = detect_modules(tom, names, min_module_size, cut)
    if module_cut_height is None and len(set(modules_df["Module"]) - {0}) == 0:
        for fb in (0.40, 0.50, 0.60, 0.70):
            modules_df, Z = detect_modules(tom, names, min_module_size, fb)
            if len(set(modules_df["Module"]) - {0}) > 0:
                cut = fb
                break
    modules_df = merge_modules_by_eigengene(X, modules_df, threshold=merge_cut_height)
    logger.info("WGCNA detected %d modules (%d unassigned features)",
                len(set(modules_df["Module"]) - {0}), int((modules_df["Module"] == 0).sum()))

    if y_encoded is not None:
        mod_trait_df, eigengenes = module_trait_correlation(X, y_encoded, modules_df, corr_method)
    else:
        eigengenes = compute_module_eigengenes(X, modules_df)
        mod_trait_df = pd.DataFrame(columns=["Module", "N_Features", "Correlation", "P_Value", "Abs_Correlation"])
    hubs_df = identify_hub_features(X, modules_df, adj, y_encoded=y_encoded, top_n=top_n_hubs, method=corr_method)

    return {"power": power, "network_type": network_type, "module_cut_height": cut,
            "merge_cut_height": merge_cut_height, "modules": modules_df, "module_trait": mod_trait_df,
            "hubs": hubs_df, "adjacency": adj, "tom": tom, "linkage": Z,
            "scale_free_fit": None if threshold_result is None else threshold_result["all_results"],
            "eigengenes": eigengenes}


class NativeWGCNA(BaseMethod):
    """EXPERIMENTAL native-Python WGCNA (teaching / exploration).

    NOT validated against the R WGCNA package — uses a static tree cut + eigengene
    merge, NOT WGCNA's dynamic tree cut. For analysis, use the R-backed `WGCNA`
    (wgcna_r.WGCNA). Kept for portability / understanding the pipeline.
    """
    handles_missing = False
    requires_target = False
    supported_targets = ("nominal", "ordinal", "continuous", "none")

    def __init__(self, power=None, corr_method="spearman", network_type="unsigned",
                 min_module_size=None, merge_cut_height=0.25, top_n_hubs=5,
                 impute="metaboanalyst"):
        super().__init__(impute=impute)
        self.power = power
        self.corr_method = corr_method
        self.network_type = network_type
        self.min_module_size = min_module_size
        self.merge_cut_height = merge_cut_height
        self.top_n_hubs = top_n_hubs
        self.result_ = None
        self.feature_names_ = None
        self.index_ = None
        self._X = None

    _PARAM_KEYS = ("power", "corr_method", "network_type", "min_module_size", "merge_cut_height")

    def describe(self) -> str:
        return (
            "EXPERIMENTAL native-Python WGCNA. Co-abundance modules + eigengenes like the R-backed "
            "WGCNA, but uses a STATIC tree cut (not WGCNA's dynamic tree cut) and is UNVALIDATED. "
            "Use the R-backed `WGCNA` for analysis; kept for portability/understanding."
        )

    def assumptions(self) -> list[str]:
        return super().assumptions() + [
            "Co-abundance structure reflects biology; adequate sample size.",
            "Static cut differs from WGCNA's dynamic tree cut -- module boundaries are not equivalent.",
        ]

    def divergences(self, context=None) -> list[str]:
        return super().divergences(context) + [
            "Experimental native port -- NOT validated against the R WGCNA package; prefer R-backed WGCNA."
        ]

    @staticmethod
    def _encode_trait(y):
        if y is None:
            return None
        s = pd.Series(np.asarray(y))
        if s.dtype.kind in "biufc":
            return s.to_numpy(dtype=float)
        return pd.factorize(s)[0].astype(float)  # nominal/ordinal -> codes

    def fit(self, X, y=None, feature_names=None, target_type=None) -> "WGCNA":
        Xp = self._prepare_X(X)
        if isinstance(Xp, pd.DataFrame):
            self.feature_names_ = list(Xp.columns)
            self.index_ = list(Xp.index)
            arr = Xp.to_numpy()
        else:
            arr = np.asarray(Xp)
            self.feature_names_ = feature_names or [f"f{i}" for i in range(arr.shape[1])]
            self.index_ = list(range(arr.shape[0]))
        self._X = arr
        self.result_ = run_wgcna(
            arr, self._encode_trait(y), self.feature_names_,
            power=self.power, corr_method=self.corr_method, network_type=self.network_type,
            min_module_size=self.min_module_size, merge_cut_height=self.merge_cut_height,
            top_n_hubs=self.top_n_hubs,
        )
        self._fitted = True
        return self

    # -- views -------------------------------------------------------------
    def modules(self) -> pd.DataFrame:
        return self.result_["modules"]

    def module_trait(self) -> pd.DataFrame:
        return self.result_["module_trait"]

    def hubs(self) -> pd.DataFrame:
        return self.result_["hubs"]

    def eigengenes(self) -> pd.DataFrame:
        """Samples x module eigengene matrix (the reduced representation)."""
        eig = self.result_["eigengenes"]
        cols = {f"ME_{m}": eig[m] for m in sorted(eig)}
        return pd.DataFrame(cols, index=self.index_)

    # -- dimensionality reduction -----------------------------------------
    def reduce(self, strategy: str = "eigengenes_and_hubs") -> pd.DataFrame:
        """Collapse features to module eigengenes (+ optional hub features).

        Returns a samples x reduced-feature DataFrame (eigengene columns prefixed
        ME_), ready to feed a supervised method or add back as an OmicsDataset
        block. No leakage: modules are built without labels.
        """
        valid = {"eigengenes_only", "hubs_only", "eigengenes_and_hubs"}
        if strategy not in valid:
            raise ValueError(f"strategy must be one of {sorted(valid)}")
        res = self.result_
        modules_df = res["modules"]
        eig = res["eigengenes"]
        hubs_df = res["hubs"]
        all_features = modules_df["Feature"].tolist()

        cols, names = {}, []
        if strategy in ("eigengenes_only", "eigengenes_and_hubs"):
            for m in sorted(eig):
                cols[f"ME_{m}"] = eig[m]
                names.append(f"ME_{m}")
        if strategy in ("hubs_only", "eigengenes_and_hubs") and hubs_df is not None and not hubs_df.empty:
            for feat in hubs_df.loc[hubs_df["Is_Hub"], "Feature"].tolist():
                if feat in all_features:
                    cols[feat] = self._X[:, all_features.index(feat)]
        return pd.DataFrame(cols, index=self.index_)
