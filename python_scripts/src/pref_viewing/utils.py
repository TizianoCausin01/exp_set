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



def plot_imgs(imgs, title=None, square_size=2):
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

