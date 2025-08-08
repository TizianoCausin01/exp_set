import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def get_extreme_k(data, dataset, k, dim, extreme):
    sorted_indices = np.argsort(data)  # ascending order
    if extreme == "top":
        topk_indices = sorted_indices[-k:]  
        return topk_indices
    elif extreme == "bottom":
        bottomk_indices = sorted_indices[:k] 
        return bottomk_indices
#EOF


def plot_imgs(imgs, title=None, square_size=2, multi=False):
    """
    imgs: list of images, or list of list of images if multi=True
    title: optional title string
    square_size: size per image
    multi: if True, imgs is a list of image groups; each will be shown in its own square grid
    """
    if not multi:
        n_imgs = len(imgs)
        n_cols = math.ceil(math.sqrt(n_imgs))
        n_rows = math.ceil(n_imgs / n_cols)
    
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(square_size, square_size),gridspec_kw=dict(wspace=0, hspace=0))
    
        # Flatten axes for easy indexing
        axes = axes.flatten()
    
        for i in range(n_imgs):
            axes[i].imshow(imgs[i])  # Replace with your image array
            axes[i].axis('off')
        # end for i in range(n_imgs):
    
        # Hide any extra axes (if grid is larger than number of images)
        for i in range(n_imgs, len(axes)):
            axes[i].axis('off')
        # end for i in range(n_imgs, len(axes)):
    
        plt.subplots_adjust(wspace=0, hspace=0)  # Remove spacing between subplots
        fig.suptitle(title, fontsize=16, y=0.95)
        plt.show()
        return fig    
    else:
        n_groups = len(imgs)
        # Compute grid shape and size for each group
        fig_list = []
        grids = []
        for group in imgs:
            n_imgs = len(group)
            n_cols = math.ceil(math.sqrt(n_imgs))
            n_rows = math.ceil(n_imgs / n_cols)
            grids.append((group, n_rows, n_cols))
    
        # Determine total figure size
        max_rows = max(nr for _, nr, _ in grids)
        total_cols = sum(nc for _, _, nc in grids)
    
        fig, axes = plt.subplots(
            max_rows,
            total_cols,
            figsize=(square_size * total_cols, square_size * max_rows),
            gridspec_kw=dict(wspace=0, hspace=0)
        )
    
        # Handle single-row or single-column edge cases
        if max_rows == 1:
            axes = [axes]
        if isinstance(axes[0], plt.Axes):
            axes = [axes]
    
        col_offset = 0
        for group, n_rows, n_cols in grids:
            for idx, img in enumerate(group):
                row = idx // n_cols
                col = idx % n_cols
                ax = axes[row][col + col_offset]
                ax.imshow(img)
                ax.axis("off")
    
            # Turn off unused axes in this group
            for idx in range(len(group), n_rows * n_cols):
                row = idx // n_cols
                col = idx % n_cols
                axes[row][col + col_offset].axis("off")
    
            col_offset += n_cols
    
        if title:
            fig.suptitle(title, fontsize=25, y=0.95)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()
        return fig
