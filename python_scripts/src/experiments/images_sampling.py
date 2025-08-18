import os, sys
from pref_viewing.utils import plot_imgs
from parallel.parallel_funcs import print_wise
from .utils import map_on_savenames, get_k_imgs

from torchvision.transforms import ToPILImage
import pandas as pd

def save_imgs_PCs(data, model_name, layer_name, loader,pooling, num_dim, k, paths): # TODO add ponce lab to paths
    to_pil = ToPILImage()
    counter = 0
    model_save_name, layer_save_name = map_on_savenames(model_name, layer_name) 
    if pooling == "PC_pool":
        final_dir = f"{model_save_name}_{layer_save_name}"
    else:
        final_dir = f"{model_save_name}_{layer_save_name}_{pooling}"
    # end if pooling == "PC_pool":
    root_dir = f"{paths['PonceLab_path']}/2025_diverseset/{final_dir}"
    win_root_dir = fr"{paths['win_PonceLab_path']}\2025_diverseset\{final_dir}"
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    provenance = []

    for d in range(num_dim):
        for extreme in ["top", "bottom"]:
            img_idx, img_list = get_k_imgs(data, loader, k, d, extreme, show_opt=False)
            for idx in range(len(img_list)):
                path2save = f"{root_dir}/net_{model_save_name}_layer_{layer_save_name}_pc{d}_{extreme}_{idx}_{counter}.png"
                win_path2save = fr"{win_root_dir}\net_{model_save_name}_layer_{layer_save_name}_pc{d}_{extreme}_{idx}_{counter}.png"
                print(path2save)
                counter +=1
                to_pil(img_list[idx]).save(path2save)
                imagenet_file_path = loader.imgs[img_idx[idx]][0]
                class_dir, file_name = imagenet_file_path.split(os.sep)[-2:]
                win_imagenet_file_path = fr"{paths['win_PonceLab_path']}\imagenet\val\{class_dir}\{file_name}"
                
                provenance.append({
                "output_image_path": win_path2save, 
                "dataset_index": img_idx[idx], 
                "total_image_index": counter,
                "rank_in_pc": idx,
                "pc_axis": d + 1,
                "top_or_bottom": extreme,
                "original_filepath": win_imagenet_file_path 
                })
    prov_path = f"{root_dir}/provenance_{model_save_name}_layer_{layer_save_name}.csv"
    pd.DataFrame(provenance).to_csv(prov_path, index=False)



def save_imgs_CCs(data, model_name1, model_name2, layer_name1, layer_name2, loader, pooling, num_dim, k, paths): # TODO add ponce lab to paths
    to_pil = ToPILImage()
    counter = 0
    model_save_name1, layer_save_name1 = map_on_savenames(model_name1, layer_name1)
    model_save_name2, layer_save_name2 = map_on_savenames(model_name2, layer_name2)
    if pooling == "PC_pool":
        final_dir = f"{model_save_name1}+{model_save_name2}_{layer_save_name1}+{layer_save_name2}"
    else:
        final_dir = f"{model_save_name1}+{model_save_name2}_{layer_save_name1}+{layer_save_name2}_{pooling}"
    # end if pooling == "PC_pool":
    root_dir = f"{paths['PonceLab_path']}/2025_diverseset/{final_dir}"
    win_root_dir = fr"{paths['win_PonceLab_path']}\2025_diverseset\{final_dir}"
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    # end if side == 1:
    
    provenance = []
    for d in range(num_dim):
        for extreme in ["top", "bottom"]:
            img_idx, img_list = get_k_imgs(data, loader, k, d, extreme, show_opt=False)
            for idx in range(len(img_list)):
                path2save = f"{root_dir}/net_{model_save_name1}+{model_save_name2}_layer_{layer_save_name1}+{layer_save_name2}_pc{d}_{extreme}_{idx}_{counter}.png"
                win_path2save = fr"{win_root_dir}\net_{model_save_name1}+{model_save_name2}_layer_{layer_save_name1}+{layer_save_name2}_pc{d}_{extreme}_{idx}_{counter}.png"
                print(path2save)
                counter +=1
                to_pil(img_list[idx]).save(path2save)
                imagenet_file_path = loader.imgs[img_idx[idx]][0]
                class_dir, file_name = imagenet_file_path.split(os.sep)[-2:]
                win_imagenet_file_path = fr"{paths['win_PonceLab_path']}\imagenet\val\{class_dir}\{file_name}"
                provenance.append({
                "output_image_path": win_path2save, 
                "dataset_index": img_idx[idx], 
                "total_image_index": counter,
                "rank_in_pc": idx,
                "pc_axis": d + 1,
                "top_or_bottom": extreme,
                "original_filepath": win_imagenet_file_path 
                })
    prov_path = f"{root_dir}/provenance_{model_save_name1}+{model_save_name2}_layer_{layer_save_name1}+{layer_save_name2}.csv"
    pd.DataFrame(provenance).to_csv(prov_path, index=False)
