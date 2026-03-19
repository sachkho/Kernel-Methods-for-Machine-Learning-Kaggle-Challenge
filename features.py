"""
Feature extraction for image classification.
Implements HOG, LBP, and opponent-color histograms from scratch.

Key change vs previous version:
  extract_features_grouped() returns a dict of feature groups
  so MKL can apply a separate kernel + gamma to each group.
"""

import numpy as np


# ─────────────────────────────────────────────
#  Image loading
# ─────────────────────────────────────────────

def load_images(path):
    raw = np.loadtxt(path, delimiter=',', usecols=range(3072))
    return raw.reshape(raw.shape[0], 3, 32, 32).astype(np.float32)

def load_labels(path):
    data = np.loadtxt(path, delimiter=',', skiprows=1, dtype=int)
    return data[:, 1]


# ─────────────────────────────────────────────
#  HOG
# ─────────────────────────────────────────────

def _rgb_to_gray(img_chw):
    return 0.2989*img_chw[0] + 0.5870*img_chw[1] + 0.1140*img_chw[2]

def _compute_gradients(gray):
    gx = np.zeros_like(gray); gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    return np.sqrt(gx**2 + gy**2), np.arctan2(gy, gx)*180/np.pi % 180

def hog_descriptor(img_chw, pixels_per_cell=(4,4), cells_per_block=(2,2),
                   n_bins=9, use_color_gradient=True):
    _, H, W = img_chw.shape
    cy, cx = pixels_per_cell
    by, bx = cells_per_block

    if use_color_gradient:
        mags, angs = zip(*[_compute_gradients(img_chw[c]) for c in range(3)])
        mags = np.stack(mags); angs = np.stack(angs)
        best = np.argmax(mags, axis=0)
        magnitude = mags[best, np.arange(H)[:,None], np.arange(W)]
        angle     = angs[best, np.arange(H)[:,None], np.arange(W)]
    else:
        magnitude, angle = _compute_gradients(_rgb_to_gray(img_chw))

    n_cy, n_cx = H//cy, W//cx
    cell_hists = np.zeros((n_cy, n_cx, n_bins), dtype=np.float32)
    bw = 180.0 / n_bins
    for i in range(n_cy):
        for j in range(n_cx):
            pm = magnitude[i*cy:(i+1)*cy, j*cx:(j+1)*cx]
            pa = angle[i*cy:(i+1)*cy, j*cx:(j+1)*cx]
            for b in range(n_bins):
                cell_hists[i,j,b] = pm[(pa >= b*bw) & (pa < (b+1)*bw)].sum()

    eps = 1e-6; blocks = []
    for i in range(n_cy - by + 1):
        for j in range(n_cx - bx + 1):
            blk = cell_hists[i:i+by, j:j+bx, :].ravel()
            blk = blk / np.sqrt(np.dot(blk,blk) + eps**2)
            blk = np.clip(blk, 0, 0.2)
            blk = blk / np.sqrt(np.dot(blk,blk) + eps**2)
            blocks.append(blk)
    return np.concatenate(blocks).astype(np.float32)

def spatial_pyramid_hog(img_chw, levels=(1,2), **kw):
    _, H, W = img_chw.shape
    parts = []
    for L in levels:
        sy, sx = H//L, W//L
        for i in range(L):
            for j in range(L):
                parts.append(hog_descriptor(
                    img_chw[:, i*sy:(i+1)*sy, j*sx:(j+1)*sx], **kw))
    return np.concatenate(parts).astype(np.float32)


# ─────────────────────────────────────────────
#  LBP
# ─────────────────────────────────────────────

def lbp_descriptor(img_chw, n_points=8, radius=1, n_bins=256):
    gray = _rgb_to_gray(img_chw)
    H, W = gray.shape
    lbp_img = np.zeros((H, W), dtype=np.uint8)
    for p in range(n_points):
        a = 2*np.pi*p/n_points
        dy, dx = radius*np.sin(a), radius*np.cos(a)
        y_n = np.arange(H, dtype=np.float32) + dy
        x_n = np.arange(W, dtype=np.float32) + dx
        y0 = np.clip(np.floor(y_n).astype(int), 0, H-1)
        y1 = np.clip(y0+1, 0, H-1)
        x0 = np.clip(np.floor(x_n).astype(int), 0, W-1)
        x1 = np.clip(x0+1, 0, W-1)
        fy = (y_n - np.floor(y_n))[:,None]
        fx = (x_n - np.floor(x_n))[None,:]
        nb = ((1-fy)*(1-fx)*gray[y0[:,None],x0[None,:]] +
              (1-fy)*   fx *gray[y0[:,None],x1[None,:]] +
                 fy *(1-fx)*gray[y1[:,None],x0[None,:]] +
                 fy *   fx *gray[y1[:,None],x1[None,:]])
        lbp_img = (lbp_img << 1) | (nb >= gray).astype(np.uint8)
    hist, _ = np.histogram(lbp_img.ravel(), bins=n_bins, range=(0,256))
    return (hist.astype(np.float32) / (hist.sum() + 1e-6))

def spatial_pyramid_lbp(img_chw, levels=(1,2), **kw):
    _, H, W = img_chw.shape
    parts = []
    for L in levels:
        sy, sx = H//L, W//L
        for i in range(L):
            for j in range(L):
                parts.append(lbp_descriptor(
                    img_chw[:, i*sy:(i+1)*sy, j*sx:(j+1)*sx], **kw))
    return np.concatenate(parts).astype(np.float32)


# ─────────────────────────────────────────────
#  Opponent color channels
# ─────────────────────────────────────────────

def _opponent_channels(img_chw):
    """
    Convert RGB to opponent color space.
    O1 = (R-G)/sqrt(2)          red vs green
    O2 = (R+G-2B)/sqrt(6)       yellow vs blue
    O3 = (R+G+B)/sqrt(3)        luminance

    More discriminative than raw RGB because opponent channels
    decorrelate colour from luminance — important for dark images.
    """
    R, G, B = img_chw[0], img_chw[1], img_chw[2]
    O1 = (R - G)       / np.sqrt(2)
    O2 = (R + G - 2*B) / np.sqrt(6)
    O3 = (R + G + B)   / np.sqrt(3)
    return np.stack([O1, O2, O3], axis=0)

def opponent_color_histogram(img_chw, n_bins=32):
    """Histogram of each opponent channel."""
    opp = _opponent_channels(img_chw)
    hists = []
    for c in range(3):
        ch = opp[c].ravel()
        lo, hi = ch.min(), ch.max()
        if hi == lo:
            hists.append(np.zeros(n_bins, dtype=np.float32))
        else:
            h, _ = np.histogram(ch, bins=n_bins, range=(lo, hi))
            hists.append(h.astype(np.float32) / (h.sum() + 1e-6))
    return np.concatenate(hists).astype(np.float32)

def opponent_hog(img_chw, **hog_kwargs):
    """
    HOG on opponent channels O1 and O2 only (pure colour edges).
    Captures edges that exist in colour space but not luminance —
    e.g. a red cat on a green background.
    """
    opp = _opponent_channels(img_chw)
    parts = []
    for c in [0, 1]:   # O1=R-G, O2=yellow-blue; skip O3=luminance (normal HOG covers it)
        ch3 = np.stack([opp[c]]*3, axis=0)
        parts.append(hog_descriptor(ch3, use_color_gradient=False, **hog_kwargs))
    return np.concatenate(parts).astype(np.float32)


# ─────────────────────────────────────────────
#  Master extractor — grouped output for MKL
# ─────────────────────────────────────────────

def extract_features_grouped(imgs, verbose=True):
    """
    Extract features and return as a dict of groups.
    Each group gets its own kernel in MKL (Multiple Kernel Learning).

    Groups
    ------
    'hog'   HOG full image + HOG 2x2 pyramid + opponent HOG
    'lbp'   LBP radius=1 full + 2x2 pyramid
            + LBP radius=2 full + 2x2 pyramid   (multi-scale texture)
    'color' Opponent colour histograms

    Returns
    -------
    {'hog': (N,D1), 'lbp': (N,D2), 'color': (N,D3)}
    """
    N = imgs.shape[0]
    hog_list, lbp_list, color_list = [], [], []

    for i in range(N):
        if verbose and i % 500 == 0:
            print(f"  Extracting features: {i}/{N}")
        img = imgs[i]

        # HOG group
        hog_list.append(np.concatenate([
            hog_descriptor(img, pixels_per_cell=(4,4), cells_per_block=(2,2), n_bins=9),
            spatial_pyramid_hog(img, levels=[2], pixels_per_cell=(4,4),
                                cells_per_block=(2,2), n_bins=9),
            opponent_hog(img, pixels_per_cell=(4,4), cells_per_block=(2,2), n_bins=9),
        ]))

        # LBP group — multi-scale (r=1 fine texture, r=2 coarser texture)
        lbp_list.append(np.concatenate([
            lbp_descriptor(img, n_points=8, radius=1, n_bins=256),
            spatial_pyramid_lbp(img, levels=[2], n_points=8, radius=1, n_bins=256),
            lbp_descriptor(img, n_points=8, radius=2, n_bins=256),
            spatial_pyramid_lbp(img, levels=[2], n_points=8, radius=2, n_bins=256),
        ]))

        # Color group
        color_list.append(opponent_color_histogram(img, n_bins=32))

    groups = {
        'hog':   np.stack(hog_list,   axis=0).astype(np.float32),
        'lbp':   np.stack(lbp_list,   axis=0).astype(np.float32),
        'color': np.stack(color_list, axis=0).astype(np.float32),
    }
    for k, v in groups.items():
        print(f"  Group '{k}': {v.shape}")
    return groups


def normalize_group(X_train, X_test=None):
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0) + 1e-8
    Xtr = (X_train - mean) / std
    if X_test is not None:
        return Xtr, (X_test - mean) / std
    return Xtr

def normalize_groups(groups_tr, groups_te=None):
    """Normalise each group independently — critical for MKL."""
    ntr, nte = {}, {}
    for k in groups_tr:
        if groups_te is not None:
            ntr[k], nte[k] = normalize_group(groups_tr[k], groups_te[k])
        else:
            ntr[k] = normalize_group(groups_tr[k])
    return (ntr, nte) if groups_te is not None else ntr

# Backward-compat
def extract_features(imgs, verbose=True):
    g = extract_features_grouped(imgs, verbose)
    X = np.concatenate([g['hog'], g['lbp'], g['color']], axis=1)
    print(f"Feature matrix shape: {X.shape}")
    return X

def normalize_features(X_train, X_test=None):
    return normalize_group(X_train, X_test)
