import numpy as np
from .utils import get_extreme_k

def get_dimension_progression(num_conditions, data, tol, dims, k=10):
    slice_dim, progress_dim = dims
    middle_lev_slice = (np.max(data[:,slice_dim]) + np.min(data[:,slice_dim]))/2
    slice_dim_usable_idx = np.where(np.logical_and(
        data[:, slice_dim] > middle_lev_slice - tol,
        data[:, slice_dim] < middle_lev_slice + tol
    ))[0]
    progress_dim_usable = np.squeeze(data[slice_dim_usable_idx, progress_dim])
    progress_dim_usable_range = np.max(progress_dim_usable) - np.min(progress_dim_usable)
    progress_dim_levels = np.linspace(np.max(progress_dim_usable), np.min(progress_dim_usable), num_conditions)
    print("\nmax :", np.max(progress_dim_usable), "\nmin :", np.min(progress_dim_usable))
    print(f"levels {progress_dim_levels}")
    img_levels_usable = [np.where(np.logical_and(
        progress_dim_usable > progress_dim_levels[lev] - tol,
        progress_dim_usable < progress_dim_levels[lev] + tol
    ))[0] for lev in range(num_conditions)]
    img_levels = [slice_dim_usable_idx[idx_lvl] for idx_lvl in img_levels_usable]
    if len(img_levels[0]) < k:
        print(f"only {len(img_levels[0])} images for the top extreme, getting top {k}")
        extreme = "top"
        img_levels_top_idx = get_extreme_k(progress_dim_usable, progress_dim_usable, k, 0, extreme)
        img_levels[0] = slice_dim_usable_idx[img_levels_top_idx]
    if len(img_levels[-1]) < k:
        print(f"only {len(img_levels[-1])} images for the bottom extreme, getting bottom {k}")
        extreme = "bottom"
        img_levels_bot_idx = get_extreme_k(progress_dim_usable, progress_dim_usable, k, 0, extreme)
        img_levels[-1] = slice_dim_usable_idx[img_levels_bot_idx]
    print(len(img_levels[0]), len(img_levels[-1]))
    return img_levels
