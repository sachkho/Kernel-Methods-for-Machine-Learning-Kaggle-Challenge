"""
Data augmentation for kernel methods image classification.

With kernel methods, augmentation = adding augmented copies to the training set.
We keep it simple and fast: only augmentations that are guaranteed to preserve
the class label.

Recommended for CIFAR-10:
  - Horizontal flip (symmetric classes: car, ship, plane, horse, deer, truck)
  - Translation ±2px (small enough to not cut the object)

NOT recommended here:
  - Rotation (objects in CIFAR-10 are mostly upright)
  - Color jitter (images are already pre-processed / dark)
  - Vertical flip (would make horses upside-down)
"""

import numpy as np


def horizontal_flip(imgs):
    """
    Flip images left-right along the W axis.
    imgs: (N, 3, H, W) → (N, 3, H, W)
    """
    return imgs[:, :, :, ::-1].copy()


def translate(imgs, dy=0, dx=0):
    """
    Shift image by (dy, dx) pixels with edge padding.
    imgs: (N, 3, H, W) → (N, 3, H, W)
    dy > 0 = shift down, dx > 0 = shift right.
    """
    _, _, H, W = imgs.shape
    out = imgs.copy()

    # Vertical shift
    if dy > 0:
        out[:, :, dy:, :] = imgs[:, :, :H-dy, :]
        out[:, :, :dy, :] = imgs[:, :, :1, :]       # pad with top edge
    elif dy < 0:
        out[:, :, :H+dy, :] = imgs[:, :, -dy:, :]
        out[:, :, H+dy:, :] = imgs[:, :, -1:, :]    # pad with bottom edge

    # Horizontal shift
    if dx > 0:
        out[:, :, :, dx:] = out[:, :, :, :W-dx]
        out[:, :, :, :dx] = out[:, :, :, :1]        # pad with left edge
    elif dx < 0:
        out[:, :, :, :W+dx] = out[:, :, :, -dx:]
        out[:, :, :, W+dx:] = out[:, :, :, -1:]     # pad with right edge

    return out


def augment_dataset(imgs, y, flips=True, translations=True, shift=2):
    """
    Generate augmented training data and append to originals.

    Parameters
    ----------
    imgs         : (N, 3, H, W) original images
    y            : (N,) labels
    flips        : bool — add horizontal flips
    translations : bool — add 4 translated versions (up/down/left/right)
    shift        : int  — translation magnitude in pixels (default 2)

    Returns
    -------
    imgs_aug : (N * multiplier, 3, H, W)
    y_aug    : (N * multiplier,)

    Multiplier = 1 (orig) + 1 (flip) + 4 (translations) = 6 by default
    """
    imgs_list = [imgs]
    y_list    = [y]

    if flips:
        imgs_list.append(horizontal_flip(imgs))
        y_list.append(y)
        print(f"  + horizontal flips: {imgs.shape[0]} images")

    if translations:
        for dy, dx in [(shift, 0), (-shift, 0), (0, shift), (0, -shift)]:
            imgs_list.append(translate(imgs, dy=dy, dx=dx))
            y_list.append(y)
        print(f"  + translations (shift={shift}px, 4 directions): {4*imgs.shape[0]} images")

    imgs_aug = np.concatenate(imgs_list, axis=0)
    y_aug    = np.concatenate(y_list,    axis=0)

    print(f"  Dataset size: {imgs.shape[0]} → {imgs_aug.shape[0]} images "
          f"(x{imgs_aug.shape[0]//imgs.shape[0]})")
    return imgs_aug, y_aug
