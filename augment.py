import numpy as np

def horizontal_flip(imgs):
    """
    Flip images left-right along the W axis.
    """

    return imgs[:, :, :, ::-1].copy()


def translate(imgs, dy=0, dx=0):
    """
    Shift image by (dy, dx) pixels with edge padding.
    """
    _, _, H, W = imgs.shape
    out = imgs.copy()

    # Vertical shift
    if dy > 0:
        out[:, :, dy:, :] = imgs[:, :, :H-dy, :]
        out[:, :, :dy, :] = imgs[:, :, :1, :] 
    elif dy < 0:
        out[:, :, :H+dy, :] = imgs[:, :, -dy:, :]
        out[:, :, H+dy:, :] = imgs[:, :, -1:, :]

    # Horizontal shift
    if dx > 0:
        out[:, :, :, dx:] = out[:, :, :, :W-dx]
        out[:, :, :, :dx] = out[:, :, :, :1] 
    elif dx < 0:
        out[:, :, :, :W+dx] = out[:, :, :, -dx:]
        out[:, :, :, W+dx:] = out[:, :, :, -1:]

    return out


def augment_dataset(imgs, y, flips=True, translations=True, shift=2):
    """
    Generate augmented training data and append to originals.
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
