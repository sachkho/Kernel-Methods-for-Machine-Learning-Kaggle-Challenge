"""
Main script — kernel methods image classification challenge.
Pipeline:
  1. Load data
  2. Data augmentation (flip + translations) on training set only
  3. Extract features grouped by type (HOG / LBP / colour)
  4. Normalise each group independently
  5. Train MKL-KRR (one kernel per feature group, combined)
  6. Evaluate on validation set + confusion matrix
  7. Retrain on full data, write submission

Usage:
    python main.py

IMPORTANT: delete the cache/ folder whenever you change features.py
"""

import numpy as np
import os

from features   import (load_images, load_labels,
                         extract_features_grouped, normalize_groups)
from kernel_svm import (MKLRidgeClassifier, KernelRidgeClassifier,
                         MulticlassKernelSVM,
                         rbf_kernel, chi2_kernel)
from augment    import augment_dataset

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────
DATA_DIR    = "./data/"
CACHE_DIR   = "./cache"
VAL_RATIO   = 0.1
RANDOM_SEED = 42

# Augmentation
DO_AUGMENT   = True    # set False to disable
AUG_FLIPS    = True
AUG_SHIFTS   = True
AUG_SHIFT_PX = 2

# Regularisation
LAMBDA = 1e-4

# Nystrom (set None for exact solve — very slow on augmented data)
NYSTROM_M = 2000

os.makedirs(CACHE_DIR, exist_ok=True)

def cache_path(name):
    return os.path.join(CACHE_DIR, name + ".npy")

def maybe_cache_groups(prefix, fn):
    """Cache/load a dict of feature group arrays."""
    names = ['hog', 'lbp', 'color']
    paths = {n: cache_path(f"{prefix}_{n}") for n in names}
    if all(os.path.exists(p) for p in paths.values()):
        print(f"  [cache] Loading {prefix}")
        return {n: np.load(paths[n]) for n in names}
    groups = fn()
    for n in names:
        np.save(paths[n], groups[n])
    return groups


# ═══════════════════════════════════════════════════════
print("="*60); print("Step 1: Loading data"); print("="*60)
imgs_train = load_images(os.path.join(DATA_DIR, "Xtr.csv"))
imgs_test  = load_images(os.path.join(DATA_DIR, "Xte.csv"))
y_all      = load_labels(os.path.join(DATA_DIR, "Ytr.csv"))
print(f"Train: {imgs_train.shape}  Test: {imgs_test.shape}")


# ═══════════════════════════════════════════════════════
print("\n"+"="*60); print("Step 2: Train/val split"); print("="*60)
rng = np.random.RandomState(RANDOM_SEED)
N   = imgs_train.shape[0]
idx = rng.permutation(N)
val_size  = int(N * VAL_RATIO)
val_idx   = idx[:val_size]
train_idx = idx[val_size:]

imgs_tr = imgs_train[train_idx]; y_tr = y_all[train_idx]
imgs_val= imgs_train[val_idx];   y_val= y_all[val_idx]
print(f"Train: {imgs_tr.shape[0]}   Val: {imgs_val.shape[0]}")


# ═══════════════════════════════════════════════════════
print("\n"+"="*60); print("Step 3: Data augmentation (train only)"); print("="*60)
if DO_AUGMENT:
    imgs_tr_aug, y_tr_aug = augment_dataset(
        imgs_tr, y_tr,
        flips=AUG_FLIPS,
        translations=AUG_SHIFTS,
        shift=AUG_SHIFT_PX)
else:
    imgs_tr_aug, y_tr_aug = imgs_tr, y_tr
    print("  Augmentation disabled.")


# ═══════════════════════════════════════════════════════
print("\n"+"="*60); print("Step 4: Feature extraction"); print("="*60)
# Augmented train features
groups_tr  = maybe_cache_groups("train_aug",
                 lambda: extract_features_grouped(imgs_tr_aug))
# Validation features (no augmentation)
groups_val = maybe_cache_groups("val",
                 lambda: extract_features_grouped(imgs_val))
# Full train features (for final submission, no aug split)
def _get_all_aug_imgs():
    if DO_AUGMENT:
        aug_imgs, _ = augment_dataset(imgs_train, y_all,
                          flips=AUG_FLIPS, translations=AUG_SHIFTS,
                          shift=AUG_SHIFT_PX)
        return extract_features_grouped(aug_imgs)
    return extract_features_grouped(imgs_train)

groups_all = maybe_cache_groups("all_aug", _get_all_aug_imgs)
groups_test= maybe_cache_groups("test",
                 lambda: extract_features_grouped(imgs_test))


# ═══════════════════════════════════════════════════════
print("\n"+"="*60); print("Step 5: Normalisation (per group)"); print("="*60)
# Normalise val/test using train statistics
groups_tr_n, groups_val_n   = normalize_groups(groups_tr,  groups_val)
groups_all_n, groups_test_n = normalize_groups(groups_all, groups_test)
print("  Normalised HOG, LBP, color independently.")

# Also compute full-train y for submission
if DO_AUGMENT:
    _aug_info = augment_dataset(imgs_train, y_all,
                    flips=AUG_FLIPS, translations=AUG_SHIFTS,
                    shift=AUG_SHIFT_PX)
    y_all_aug = _aug_info[1]
else:
    y_all_aug = y_all


# ═══════════════════════════════════════════════════════
print("\n"+"="*60); print("Step 6: Build MKL kernels"); print("="*60)
# Auto-gamma per group: 1 / (D * var)
def auto_gamma(X):
    return 1.0 / (X.shape[1] * float(X.var()))

gamma_hog   = auto_gamma(groups_tr_n['hog'])
gamma_lbp   = auto_gamma(groups_tr_n['lbp'])
gamma_color = auto_gamma(groups_tr_n['color'])
print(f"  gamma_hog={gamma_hog:.2e}  gamma_lbp={gamma_lbp:.2e}  gamma_color={gamma_color:.2e}")

kernel_fns = {
    'hog':   lambda A, B, g=gamma_hog:   rbf_kernel(A, B, gamma=g),
    'lbp':   lambda A, B, g=gamma_lbp:   rbf_kernel(A, B, gamma=g),
    'color': lambda A, B, g=gamma_color: rbf_kernel(A, B, gamma=g),
}

# Optional: use chi2 for LBP/color (often better for histograms)
# kernel_fns['lbp']   = lambda A, B: chi2_kernel(A, B, gamma=1.0)
# kernel_fns['color'] = lambda A, B: chi2_kernel(A, B, gamma=1.0)

# Weights: HOG usually most discriminative, LBP helps texture, color secondary
weights = {'hog': 0.5, 'lbp': 0.35, 'color': 0.15}


# ═══════════════════════════════════════════════════════
print("\n"+"="*60); print("Step 7: Training MKL-KRR"); print("="*60)
model = MKLRidgeClassifier(
    lam=LAMBDA,
    kernel_fns=kernel_fns,
    weights=weights,
    nystrom_m=NYSTROM_M,
    seed=RANDOM_SEED)
model.fit(groups_tr_n, y_tr_aug)


# ═══════════════════════════════════════════════════════
print("\n"+"="*60); print("Step 8: Validation"); print("="*60)
val_acc = model.score(groups_val_n, y_val)
print(f"Validation accuracy: {val_acc*100:.2f}%")

preds_val = model.predict(groups_val_n)
for cls in np.unique(y_all):
    mask = (y_val == cls)
    if mask.sum() > 0:
        print(f"  Class {cls}: {np.mean(preds_val[mask]==cls)*100:.1f}%  ({mask.sum()} samples)")

# Confusion matrix
import matplotlib.pyplot as plt
classes = np.unique(y_all)
cm = np.zeros((len(classes), len(classes)), dtype=int)
for t, p in zip(y_val, preds_val):
    cm[t, p] += 1
fig, ax = plt.subplots(figsize=(8,6))
im = ax.imshow(cm, cmap="Blues")
plt.colorbar(im)
ax.set_xticks(classes); ax.set_yticks(classes)
ax.set_xticklabels(classes); ax.set_yticklabels(classes)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"Confusion Matrix — val acc {val_acc*100:.1f}%")
for i in classes:
    for j in classes:
        ax.text(j, i, cm[i,j], ha="center", va="center",
                color="white" if cm[i,j] > cm.max()/2 else "black")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Confusion matrix saved.")


# ═══════════════════════════════════════════════════════
print("\n"+"="*60); print("Step 9: Retrain on full data + submit"); print("="*60)
final_model = MKLRidgeClassifier(
    lam=LAMBDA, kernel_fns=kernel_fns, weights=weights,
    nystrom_m=NYSTROM_M, seed=RANDOM_SEED)
final_model.fit(groups_all_n, y_all_aug)
test_preds = final_model.predict(groups_test_n)

out_file = "Yte.csv"
with open(out_file, "w") as f:
    f.write("Id,Prediction\n")
    for i, pred in enumerate(test_preds):
        f.write(f"{i+1},{pred}\n")
print(f"Done! {out_file} written.")
print(f"Label distribution: {dict(zip(*np.unique(test_preds, return_counts=True)))}")
