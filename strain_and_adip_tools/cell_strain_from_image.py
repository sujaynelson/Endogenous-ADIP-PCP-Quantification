"""
Cell Strain from Image Analysis

This script performs strain analysis on cell images, extracting protein,
corner, and actin information from microscopy images with corresponding masks.
"""

import numpy as np
from PIL import Image
from tifffile import imread
import cv2

import matplotlib.pyplot as plt
import importlib
import os

from strain_and_adip_tools.strain_inference import StrainInference
from aggregate_analysis_tools import import_data


# =============================================================================
# Utility Functions
# =============================================================================

def if_possible_make_path(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def read_image(path: str):
    # if tif file
    if path[-4:] == ".tif":
        img = imread(path)

        # Some of Sergei's images had channels on first axis. Take this into account.
        channels_index = np.argmin(img.shape)
        img = np.moveaxis(img, [channels_index], [-1])

        return img
    elif path[-4:] == ".png":
        return np.array(imread(path))

    raise ValueError("File type not supported")


# =============================================================================
# Setup and Configuration
# =============================================================================

super_path = "all_data_structured/"

# Okay, I am getting tired of this, so from now on:
# We will loop through all subfolders in the path.
# then chek if any of them have subfolders.
# If they do, we will loop through those as well.
# for each (sub)folder we will check if there is a .txt
# file giving us the protein, cell and shroom channels and
# a subfolder titled "masks" with the masks.

# In parallel, a copy of the structure will be created in the
# output folder, where we will save the arrow results and readables
# in the same structure as the input.

all_folder_names = []

first_level = os.listdir(super_path)
for folder in first_level:
    if not os.path.isdir(super_path + folder) or folder == "masks":
        continue

    all_folder_names.append(folder)

for folder in all_folder_names:
    second_level = os.listdir(super_path + folder)
    for subfolder in second_level:
        if subfolder == "masks" or subfolder == "results":
            continue

        if not os.path.isdir(super_path + folder + "/" + subfolder) or subfolder == "masks":
            continue

        all_folder_names.append(folder + "/" + subfolder)

        third_level = os.listdir(super_path + folder + "/" + subfolder)
        for subsubfolder in third_level:
            if not os.path.isdir(super_path + folder + "/" + subfolder + "/" + subsubfolder) or subsubfolder == "masks":
                continue

            all_folder_names.append(folder + "/" + subfolder + "/" + subsubfolder)


def check_if_valid_path(path):
    if not os.path.isdir(super_path + path + "/masks"):
        print("No masks folder in " + path)
        return False

    # check if there is a txt file with the channels
    if not os.path.isfile(super_path + path + "/channels.txt"):
        print("No channels.txt file in " + path)
        return False

    return True


all_folders = {}

for fold in all_folder_names:
    all_folders[fold] = {"is_valid": check_if_valid_path(fold)}


for fold in all_folders.keys():
    if not all_folders[fold]["is_valid"]:
        continue

    with open(super_path + fold + "/channels.txt", "r") as f:
        lines = f.readlines()

        # protein, cell, shroom
        cs = [int(x.strip()) for x in lines]

        all_folders[fold]["channels"] = cs


# =============================================================================
# Analysis Functions
# =============================================================================

PROTEIN_SENSITIVITY = 0.2  # sensitivity percentage between 0 and 1


def preprocess_image(img_nolog):
    img_nolog_is_zero = np.logical_or(img_nolog == 0., img_nolog == 1.)
    img_nolog[img_nolog_is_zero] = img_nolog[~img_nolog_is_zero].min()
    print(img_nolog.min(), img_nolog.max())
    img = np.log(img_nolog)

    img[img < 4.5] = 4.5  # catch the -inf and 0 values

    return img


def do_single_image_analysis(fold, img, img_name, show_images=False, do_corner_analysis=False,
                              do_protein_analysis=True, do_actin_analysis=False, do_actin_null_model=False):
    assert do_protein_analysis or do_corner_analysis or do_actin_analysis or do_actin_null_model, \
        "You must do either protein, corner or actin analysis or actin null model"

    path = super_path + fold + "/"
    mask_path = path + "masks/"
    result_path = path + "results/"
    if not os.path.isdir(result_path):
        print("Creating result path")
        os.mkdir(result_path)
    pc, cc = all_folders[fold]["channels"][:2]

    ac = -1

    # remove all channels except the cell channel and the protein channel
    img_protein = img[:, :, pc]
    img_cell = img[:, :, cc]

    if (len(all_folders[fold]["channels"]) == 3):
        ac = all_folders[fold]["channels"][2]

    print("AC", ac)

    print("Doing analysis of " + img_name)

    if show_images:
        # split the image into its channels
        fig, ax = plt.subplots(1, img.shape[2], figsize=(img.shape[2]*7, 7))
        for i in range(img.shape[2]):
            ax[i].imshow(img[:, :, i])

            if i == pc:
                ax[i].set_title("Protein Channel")
            elif i == cc:
                ax[i].set_title("Cell Channel")
            elif i == ac:
                ax[i].set_title("Actin Channel")

        fig.suptitle(img_name)
        fig.tight_layout()
        plt.show()

    # find the masks
    mask_path_individual = mask_path + "mask_" + img_name + '.npy'

    # check if the mask exists
    shouldreturn = False
    if not os.path.isfile(mask_path_individual):
        print("No mask found at " + mask_path_individual)
        shouldreturn = True

    img_name = img_name.replace("+", "")

    if shouldreturn and not os.path.isfile(mask_path_individual):
        mask_path_individual = mask_path + "mask_" + img_name + '.npy'
        print("No mask found at " + mask_path_individual, "either")
        shouldreturn = True
    else:
        shouldreturn = False

    if shouldreturn:
        print("Aborting! Skipping " + img_name)
        return

    # make image savepath
    img_path = result_path + "images/"
    if not os.path.isdir(img_path):
        os.mkdir(img_path)

    img_actin = None if ac == -1 else img[:, :, ac]

    strain_inferer = StrainInference(img_cell, img_protein, mask_path_individual,
                                      protein_sensitivity=PROTEIN_SENSITIVITY,
                                      show_images=show_images, name=img_name,
                                      img_savepath=img_path, img_actin=img_actin)

    def save_csv(name, arr):
        np.savetxt(name + ".csv", arr, delimiter=",")

    # do the protein analysis
    if do_protein_analysis:
        protein_arrows, protein_positions, protein_ids, protein_magnitudes, protein_sizes = strain_inferer.do_protein_part()

        protpath = result_path + "protein/"
        if not os.path.isdir(protpath):
            os.mkdir(protpath)
        save_csv(protpath + "protein_arrows_" + img_name, protein_arrows)
        save_csv(protpath + "protein_positions_" + img_name, protein_positions)
        save_csv(protpath + "protein_ids_" + img_name, protein_ids)
        save_csv(protpath + "protein_magnitudes_" + img_name, protein_magnitudes)
        save_csv(protpath + "protein_sizes_" + img_name, protein_sizes)

    if do_corner_analysis:
        protein_positions, protein_ids, protein_sizes, corner_positions, corner_ids, border = strain_inferer.do_corner_analysis(count_borders=True)
        protpath = result_path + "corner/"
        if not os.path.isdir(protpath):
            os.mkdir(protpath)

        save_csv(protpath + "protein_positions_" + img_name, protein_positions)
        save_csv(protpath + "corner_positions_" + img_name, corner_positions)
        save_csv(protpath + "protein_sizes_" + img_name, protein_sizes)
        save_csv(protpath + "corner_ids_" + img_name, corner_ids)
        save_csv(protpath + "protein_ids_" + img_name, protein_ids)
        save_csv(protpath + "on_border_" + img_name, border)

    if do_actin_analysis:
        actin_arrows, actin_positions, actin_ids, actin_magnitudes = strain_inferer.do_actin_part()

        actpath = result_path + "actin/"
        if not os.path.isdir(actpath):
            os.mkdir(actpath)
        save_csv(actpath + "actin_arrows_" + img_name, actin_arrows)
        save_csv(actpath + "actin_positions_" + img_name, actin_positions)
        save_csv(actpath + "actin_ids_" + img_name, actin_ids)
        save_csv(actpath + "actin_magnitudes_" + img_name, actin_magnitudes)

        print("Finished actin analysis of " + img_name)

    if do_actin_null_model:
        null_arrows, null_positions, null_ids, null_magnitudes = strain_inferer.do_multiple_actin_null_models(n_repeats=100)

        actpath = result_path + "actin_null/"
        if not os.path.isdir(actpath):
            os.mkdir(actpath)
        save_csv(actpath + "null_arrows_" + img_name, null_arrows)
        save_csv(actpath + "null_positions_" + img_name, null_positions)
        save_csv(actpath + "null_ids_" + img_name, null_ids)
        save_csv(actpath + "null_magnitudes_" + img_name, null_magnitudes)

        print("Finished actin null_model analysis of " + img_name)


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    show_images = True

    do_protein_analysis = True
    do_corner_analysis = True
    do_actin_analysis = True
    do_actin_null_model = True

    iiii = 0
    # for every folder in the path
    for fold in all_folders.keys():
        if not all_folders[fold]["is_valid"]:
            continue

        path = super_path + fold + "/"

        channels = all_folders[fold]["channels"]
        print("WORKING ON FOLDER: ", fold)

        for img_dir in os.listdir(path + "/"):
            # extract the name
            img_name = img_dir.replace(".png", "").replace(".tiff", "").replace(".tif", "")

            # read in the image
            try:
                img_nolog = read_image(path + "/" + img_dir)
                img = preprocess_image(img_nolog)
            except Exception as e:
                print("ERROR: ", e, "for", img_dir)
                continue

            isvideo = len(img.shape) == 4

            img_analysis = lambda ximg, ximg_name: do_single_image_analysis(
                fold, ximg, ximg_name,
                show_images=show_images,
                do_corner_analysis=do_corner_analysis,
                do_protein_analysis=do_protein_analysis,
                do_actin_analysis=do_actin_analysis,
                do_actin_null_model=do_actin_null_model
            )

            # if image is video (ie. if it has a 4th dimension)
            if not isvideo:
                img_analysis(img, img_name)
            else:
                print("Video detected")
                for i in range(img.shape[0]):
                    if iiii < -1:
                        iiii += 1
                        continue
                    img_single = img[i, :, :, :]
                    frame_img_name = "frame_" + str(i) + "_" + img_name
                    print(iiii)
                    img_analysis(img_single, frame_img_name)
                    iiii += 1
