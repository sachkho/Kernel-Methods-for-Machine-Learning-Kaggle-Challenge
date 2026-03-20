import numpy as np
import os
import csv
import time

from features   import (load_images, load_labels,
                         extract_features_grouped, normalize_groups)
from kernel_svm import MKLRidgeClassifier, rbf_kernel, chi2_kernel
from augment    import augment_dataset

DATA_DIR    = "./data/"
CACHE_DIR   = "./cache"
N_FOLDS     = 5
RANDOM_SEED = 42

DO_AUGMENT   = True
AUG_FLIPS    = True
AUG_SHIFTS   = True
AUG_SHIFT_PX = 2

NYSTROM_M = 1000

os.makedirs(CACHE_DIR, exist_ok=True)

LAMBDAS = [1e-5, 1e-4, 1e-3]
GAMMA_MULTS = [0.1, 0.5, 1.0, 2.0, 5.0]
KERNEL_CONFIGS = [
    {'hog': 'rbf',  'lbp': 'rbf',  'color': 'rbf'},
    {'hog': 'rbf',  'lbp': 'chi2', 'color': 'chi2'},
    {'hog': 'rbf',  'lbp': 'chi2', 'color': 'rbf'},
]

WEIGHT_CONFIGS = [
    {'hog': 0.5,  'lbp': 0.35, 'color': 0.15},
    {'hog': 0.6,  'lbp': 0.30, 'color': 0.10},
    {'hog': 0.4,  'lbp': 0.45, 'color': 0.15},
    {'hog': 1/3,  'lbp': 1/3,  'color': 1/3},
]

#we load the data and the feature and we added a cache to avoid recomputing everytime since it took a lot of time
def cache_path(name):
    return os.path.join(CACHE_DIR, name + ".npy")

def maybe_cache_groups(prefix, fn):
    names = ['hog', 'lbp', 'color']
    paths = {n: cache_path(f"{prefix}_{n}") for n in names}
    if all(os.path.exists(p) for p in paths.values()):
        print(f"  [cache] Loading {prefix}")
        return {n: np.load(paths[n]) for n in names}
    groups = fn()
    for n in names:
        np.save(paths[n], groups[n])
    return groups


print("Loading images...")
imgs = load_images(os.path.join(DATA_DIR, "Xtr.csv"))
y    = load_labels(os.path.join(DATA_DIR, "Ytr.csv"))
N    = imgs.shape[0]
print(f"Loaded {N} images, {len(np.unique(y))} classes.")

print("Extracting features (no augmentation for base features)...")
groups_raw = maybe_cache_groups("cv_base", lambda: extract_features_grouped(imgs))
print("Features ready.")
for k, v in groups_raw.items():
    print(f"  '{k}': {v.shape}  var={v.var():.4f}")

#K-fold cross-val

def kfold_indices(N, n_folds, seed):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(N)
    fold_size = N // n_folds
    folds = []
    for k in range(n_folds):
        val_idx   = idx[k*fold_size:(k+1)*fold_size]
        train_idx = np.concatenate([idx[:k*fold_size], idx[(k+1)*fold_size:]])
        folds.append((train_idx, val_idx))
    return folds


def make_kernel_fns(kernel_config, gamma_mult, groups_train_n):
    fns = {}
    for name, ktype in kernel_config.items():
        X = groups_train_n[name]
        gamma = gamma_mult / (X.shape[1] * float(X.var()) + 1e-12)
        if ktype == 'rbf':
            fns[name] = (lambda A, B, g=gamma: rbf_kernel(A, B, gamma=g))
        elif ktype == 'chi2':
            fns[name] = (lambda A, B, g=gamma: chi2_kernel(A, B, gamma=g))
    return fns


def run_cv(lam, gamma_mult, kernel_config, weights, n_folds=N_FOLDS):
    folds = kfold_indices(N, n_folds, RANDOM_SEED)
    accs  = []

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        groups_tr  = {k: v[train_idx] for k, v in groups_raw.items()}
        groups_val = {k: v[val_idx]   for k, v in groups_raw.items()}
        y_tr       = y[train_idx]
        y_val      = y[val_idx]

        if DO_AUGMENT:
            imgs_fold = imgs[train_idx]
            imgs_aug, y_aug = augment_dataset(
                imgs_fold, y_tr,
                flips=AUG_FLIPS,
                translations=AUG_SHIFTS,
                shift=AUG_SHIFT_PX)
            n_orig = len(train_idx)
            groups_extra = extract_features_grouped(imgs_aug[n_orig:], verbose=False)
            groups_tr = {k: np.concatenate([groups_tr[k], groups_extra[k]], axis=0)
                         for k in groups_tr}
            y_tr = y_aug
        groups_tr_n, groups_val_n = normalize_groups(groups_tr, groups_val)

        kernel_fns = make_kernel_fns(kernel_config, gamma_mult, groups_tr_n)

        # Train
        model = MKLRidgeClassifier(
            lam=lam,
            kernel_fns=kernel_fns,
            weights=weights,
            nystrom_m=NYSTROM_M,
            seed=RANDOM_SEED)
        model.fit(groups_tr_n, y_tr)

        acc = model.score(groups_val_n, y_val)
        accs.append(acc)
        print(f"    fold {fold_i+1}/{n_folds}: {acc*100:.2f}%")

    return float(np.mean(accs)), float(np.std(accs))


#Now the grid search

results = []
total = len(LAMBDAS) * len(GAMMA_MULTS) * len(KERNEL_CONFIGS) * len(WEIGHT_CONFIGS)
run   = 0

print(f"\n{'='*60}")
print(f"Starting grid search: {total} configurations x {N_FOLDS} folds")
print(f"{'='*60}\n")

for kernel_config in KERNEL_CONFIGS:
    for weights in WEIGHT_CONFIGS:
        for gamma_mult in GAMMA_MULTS:
            for lam in LAMBDAS:
                run += 1
                kstr = "/".join(kernel_config[k] for k in ['hog','lbp','color'])
                wstr = f"({weights['hog']:.2f},{weights['lbp']:.2f},{weights['color']:.2f})"
                print(f"[{run}/{total}] kernels={kstr}  weights={wstr}  "
                      f"gamma_mult={gamma_mult}  lam={lam:.0e}")

                t0 = time.time()
                try:
                    mean_acc, std_acc = run_cv(lam, gamma_mult, kernel_config, weights)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    mean_acc, std_acc = 0.0, 0.0

                elapsed = time.time() - t0
                print(f"  >>> mean={mean_acc*100:.2f}%  std={std_acc*100:.2f}%  "
                      f"({elapsed:.0f}s)\n")

                results.append({
                    'mean_acc':    mean_acc,
                    'std_acc':     std_acc,
                    'lam':         lam,
                    'gamma_mult':  gamma_mult,
                    'kernel_hog':  kernel_config['hog'],
                    'kernel_lbp':  kernel_config['lbp'],
                    'kernel_color':kernel_config['color'],
                    'w_hog':       weights['hog'],
                    'w_lbp':       weights['lbp'],
                    'w_color':     weights['color'],
                    'elapsed_s':   elapsed,
                })


results.sort(key=lambda r: r['mean_acc'], reverse=True)

print("\n" + "="*60)
print("TOP 10 CONFIGURATIONS")
print("="*60)
for i, r in enumerate(results[:10]):
    kstr = f"{r['kernel_hog']}/{r['kernel_lbp']}/{r['kernel_color']}"
    wstr = f"({r['w_hog']:.2f},{r['w_lbp']:.2f},{r['w_color']:.2f})"
    print(f"#{i+1:2d}  acc={r['mean_acc']*100:.2f}% ± {r['std_acc']*100:.2f}%  "
          f"lam={r['lam']:.0e}  gamma_mult={r['gamma_mult']}  "
          f"kernels={kstr}  weights={wstr}")

# Save to CSV
csv_path = "cv_results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print(f"\nAll results saved to {csv_path}")

# Print best config ready to paste into main.py
best = results[0]
print("\n" + "="*60)
print("BEST CONFIG — paste into main.py:")
print("="*60)
print(f"LAMBDA       = {best['lam']}")
print(f"GAMMA_MULT   = {best['gamma_mult']}  # multiply auto-gamma by this")
print(f"kernel_fns['hog']   → {best['kernel_hog']}")
print(f"kernel_fns['lbp']   → {best['kernel_lbp']}")
print(f"kernel_fns['color'] → {best['kernel_color']}")
print(f"weights = {{'hog': {best['w_hog']}, 'lbp': {best['w_lbp']}, 'color': {best['w_color']}}}")
