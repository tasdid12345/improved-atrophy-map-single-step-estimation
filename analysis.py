"""Synthetic atrophy mapping: MLP patch regressor (baseline) vs. CV-tuned LightGBM with smoothed multi-scale features (improved)."""
import json, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless plotting on CI
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GroupKFold
import lightgbm as lgb

SEED = 42
SEEDS = [42, 43, 44]          # model seeds; data generation and split use SEED
N_SUBJ, N = 60, 64            # synthetic subjects and slice size
PR = 3                        # patch radius -> 7x7 patches
MAX_TRAIN_ROWS = 60000        # subsample training pixels to fit the CPU budget
T0 = time.time()
GY, GX = np.mgrid[0:N, 0:N].astype(float)
C = (N - 1) / 2


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def ellipse(cy, cx, ry, rx):
    return ((GY - cy) / ry) ** 2 + ((GX - cx) / rx) ** 2 <= 1.0


def render(brain, vent, rng):
    """Brain-like slice: grey-matter rim, white matter, dark CSF ventricles, bias field and noise."""
    gm = brain & (ndimage.distance_transform_edt(brain) <= 6)
    img = np.where(brain, np.where(gm, 0.8, 0.6), 0.0)
    img[vent] = 0.15
    img = img * (1 + rng.uniform(-0.2, 0.2) * (GX - C) / N) * rng.uniform(0.95, 1.05)
    return img + rng.normal(0, 0.04, img.shape)


def make_subject(rng):
    """Simulate baseline/follow-up slices and the ground-truth atrophy map."""
    brain0 = ellipse(C, C, 27 + rng.uniform(-2, 2), 23 + rng.uniform(-2, 2))
    d0 = ndimage.distance_transform_edt(brain0)
    vy, vx = C + rng.uniform(-2, 2), rng.uniform(-1, 1)
    vent0 = ellipse(vy, C - 4 + vx, 5, 3.5) | ellipse(vy, C + 4 + vx, 5, 3.5)
    k = rng.uniform(0, 3)     # cortical shrinkage (pixels eroded from outer rim)
    dv = rng.uniform(0, 4)    # ventricular enlargement (pixels of dilation)
    brain1 = brain0 & (d0 > k)
    vent1 = vent0 | (ndimage.distance_transform_edt(~vent0) <= dv)
    loss = ((brain0 & ~brain1) | (vent1 & ~vent0)).astype(float)
    atrophy = ndimage.gaussian_filter(loss, 1.0)  # ground-truth atrophy map
    I0 = render(brain0, vent0, rng)
    I1 = render(brain1, vent1, rng)
    mask = ndimage.binary_dilation(brain0 | brain1, iterations=2)  # evaluation region
    return I0, I1, atrophy, mask


def features(I0, I1):
    """Raw 7x7 patches from both time points, plus smoothed / multi-scale change features."""
    D = I1 - I0
    def win(A):  # flattened 7x7 neighbourhood of every pixel
        return np.lib.stride_tricks.sliding_window_view(
            np.pad(A, PR, mode="reflect"), (2 * PR + 1, 2 * PR + 1)).reshape(N * N, -1)
    patch = np.hstack([win(I0), win(I1)])  # 98 features
    eng = [D, ndimage.gaussian_filter(D, 1), ndimage.gaussian_filter(D, 2),
           ndimage.gaussian_filter(D, 4), ndimage.uniform_filter(D, 5),
           ndimage.uniform_filter(D, 11), ndimage.gaussian_filter(I0, 2),
           ndimage.gaussian_filter(I1, 2)]
    return patch, np.stack([e.ravel() for e in eng], axis=1)


# ---- build per-subject pixel tables (same code path for both models) ----
subs = []
for s in range(N_SUBJ):
    I0, I1, A, m = make_subject(np.random.default_rng(SEED + s))
    P, E = features(I0, I1)
    idx = np.flatnonzero(m.ravel())
    subs.append(dict(P=P[idx], E=E[idx], y=A.ravel()[idx]))
log(f"simulated {N_SUBJ} subject pairs, {sum(len(s['y']) for s in subs)} pixels")


def gather(ids):
    P = np.vstack([subs[i]["P"] for i in ids])
    E = np.vstack([subs[i]["E"] for i in ids])
    Y = np.concatenate([subs[i]["y"] for i in ids])
    G = np.concatenate([np.full(len(subs[i]["y"]), i) for i in ids])
    return P, E, Y, G


# ---- subject-level split, identical for both models ----
tr_ids, te_ids = train_test_split(np.arange(N_SUBJ), test_size=0.3, random_state=SEED)
Ptr, Etr, Ytr, Gtr = gather(tr_ids)
Pte, Ete, Yte, Gte = gather(te_ids)
sub = np.random.default_rng(SEED).choice(len(Ytr), min(MAX_TRAIN_ROWS, len(Ytr)), replace=False)
Ptr, Etr, Ytr, Gtr = Ptr[sub], Etr[sub], Ytr[sub], Gtr[sub]
Xtr, Xte = np.hstack([Ptr, Etr]), np.hstack([Pte, Ete])
log(f"train rows {len(Ytr)} (subjects {len(tr_ids)}), test rows {len(Yte)} (subjects {len(te_ids)})")


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def lgb_params(nl, mcs, seed):
    return dict(n_estimators=300, learning_rate=0.05, num_leaves=nl, min_child_samples=mcs,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)


# ---- CV tuning of LightGBM on training subjects only ----
log("CV tuning LightGBM (GroupKFold=3 on training subjects)")
best = None
for nl in (15, 63):
    for mcs in (20, 100):
        rs = []
        for a, b in GroupKFold(3).split(Xtr, Ytr, Gtr):
            m = lgb.LGBMRegressor(**lgb_params(nl, mcs, SEED)).fit(Xtr[a], Ytr[a])
            rs.append(pearson(m.predict(Xtr[b]), Ytr[b]))
        cv = float(np.mean(rs))
        log(f"  num_leaves={nl} min_child_samples={mcs}: CV Pearson={cv:.4f}")
        if best is None or cv > best[0]:
            best = (cv, nl, mcs)
cv_best, NL, MCS = best
log(f"selected num_leaves={NL}, min_child_samples={MCS}")


def score(pred, y, g):
    """Pearson r on pooled test pixels, MAE, and mean absolute error of per-scan atrophied volume."""
    vol = np.mean([abs(pred[g == s].sum() - y[g == s].sum()) for s in np.unique(g)])
    return pearson(pred, y), float(np.mean(np.abs(pred - y))), float(vol)


R = {"base": [], "imp": [], "base_mae": [], "imp_mae": [], "base_vol": [], "imp_vol": []}
for sd in SEEDS:
    log(f"seed {sd}: fitting baseline MLP patch regressor")
    mlp = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=40,
                                                       early_stopping=True, random_state=sd))
    mlp.fit(Ptr, Ytr)
    pb = mlp.predict(Pte)
    log(f"seed {sd}: fitting improved LightGBM")
    pi = lgb.LGBMRegressor(**lgb_params(NL, MCS, sd)).fit(Xtr, Ytr).predict(Xte)
    for key, p in (("base", pb), ("imp", pi)):
        r, mae, vol = score(p, Yte, Gte)
        R[key].append(r); R[key + "_mae"].append(mae); R[key + "_vol"].append(vol)
    log(f"seed {sd}: baseline r={R['base'][-1]:.4f}  improved r={R['imp'][-1]:.4f}")

bs, im = np.array(R["base"]), np.array(R["imp"])
if not (np.all(np.isfinite(bs)) and np.all(np.isfinite(im)) and len(Yte) > 0):
    log("could not produce meaningful metrics")
    sys.exit(1)

# ---- chart ----
import os
os.makedirs("figures", exist_ok=True)
plt.figure(figsize=(5, 4))
plt.bar(["baseline MLP", "improved LGBM"], [bs.mean(), im.mean()],
        yerr=[bs.std(), im.std()], capsize=6, color=["grey", "steelblue"])
plt.ylabel("Pearson r (test, mean of 3 seeds)")
plt.title("Held-out atrophy-map correlation")
plt.tight_layout()
plt.savefig("figures/pearson_bar.png", dpi=120)

results = {
    "question": "Can a cross-validated LightGBM model with smoothed multi-scale difference features estimate voxel-wise atrophy maps from paired scans more accurately than a single-step MLP regressor on patches, in a simulated brain-slice setting?",
    "method": "Synthetic 64x64 brain-like slice pairs with known atrophy maps; subject-level 70/30 split; MLP patch regressor vs. CV-tuned LightGBM on patches plus smoothed/multi-scale change features; Pearson r on held-out pixels.",
    "dataset": {
        "name": "Synthetic 2D brain-like slice pairs (generated in code)",
        "source_url": "https://scikit-learn.org",
        "n_rows": int(sum(len(s["y"]) for s in subs)),
        "n_train": int(len(Ytr)),
        "n_test": int(len(Yte)),
        "same_as_paper": False,
        "baseline_is_papers_method": False,
    },
    "metric_name": "Pearson correlation between predicted and true atrophy (held-out pixels)",
    "higher_is_better": True,
    "baseline": "sklearn MLPRegressor (128-64, early stopping) on flattened 7x7 patches from both time points, predicting centre-pixel atrophy; stand-in for a single-step mapping network.",
    "improved": "LightGBM regressor (num_leaves and min_child_samples chosen by 3-fold group CV on training subjects) on the same patches plus Gaussian-smoothed difference and multi-scale change features.",
    "metrics": {
        "baseline_score": float(bs.mean()),
        "improved_score": float(im.mean()),
        "improvement": float(im.mean() - bs.mean()),
        "baseline_score_std": float(bs.std()),
        "improved_score_std": float(im.std()),
        "baseline_mae": float(np.mean(R["base_mae"])),
        "improved_mae": float(np.mean(R["imp_mae"])),
        "baseline_volume_abs_error": float(np.mean(R["base_vol"])),
        "improved_volume_abs_error": float(np.mean(R["imp_vol"])),
        "lgb_cv_pearson": float(cv_best),
        "lgb_num_leaves": NL,
        "lgb_min_child_samples": MCS,
    },
    "notes": "Synthetic data only, not the paper's dataset; the paper's single-step U-Net is not reproduced, and the MLP is a stand-in. Hyperparameters were selected by CV on training subjects (seed 42) only; the test set was used once per model and seed. Model seeds 42-44 share one simulated dataset and one split, so the std reflects model randomness only. Volume error is in pixel units per scan. Results may not transfer to real MRI.",
}
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
log(f"done: baseline r={bs.mean():.4f}, improved r={im.mean():.4f}; wrote results.json")
