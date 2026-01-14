"""
Python code extracted from creating_figures-vs.ipynb
Generated on November 19, 2025
"""

import numpy as np
import matplotlib.pyplot as plt

from aggregate_analysis_tools import import_data, get_angles, make_rose_plot, get_resolution, get_shortest_distances

path = "myexperiment/"

# Quantify protein only (exclude actin & actin_null)
data_protein = import_data(path + "Fig1", "protein", False)

all_prot_arrows = []
all_prot_magnitudes = []
all_prot_sizes    = []

for exp in data_protein["protein_arrows"].keys():
    arrows     = data_protein["protein_arrows"][exp]
    magnitudes = data_protein["protein_magnitudes"][exp]
    sizes      = data_protein["protein_sizes"][exp]
    all_prot_arrows.extend(arrows)
    all_prot_magnitudes.extend(magnitudes)
    all_prot_sizes.extend(sizes)

# convert to arrays
all_prot_arrows     = np.array(all_prot_arrows)
all_prot_magnitudes = np.array(all_prot_magnitudes)
all_prot_sizes      = np.array(all_prot_sizes)

# basic summaries
print(f"Total protein vectors : {len(all_prot_arrows)}")
print(f"Mean magnitude       : {np.mean(all_prot_magnitudes):.2f} ± {np.std(all_prot_magnitudes):.2f}")

#data_actin = import_data(path + "Fig1", "actin", False)
data_adip = import_data(path + "Fig1", "protein", False)
#data_actin_null = import_data(path + "Fig1", "actin_null", False)

all_adip_arrows = []
all_adip_magnitudes = []
all_adip_sizes = []

for experiment in data_adip["protein_arrows"].keys():
    arrows = data_adip["protein_arrows"][experiment]
    magnitudes = data_adip["protein_magnitudes"][experiment]
    sizes = data_adip["protein_sizes"][experiment]
    all_adip_arrows.extend(arrows)
    all_adip_magnitudes.extend(magnitudes)
    all_adip_sizes.extend(sizes)

all_adip_arrows = np.array(all_adip_arrows)
all_adip_magnitudes = np.array(all_adip_magnitudes)
all_adip_sizes = np.array(all_adip_sizes)

fig, axs = plt.subplots(2, 2, figsize=(8, 10), dpi=300, subplot_kw=dict(polar=True))
for tf, ax in zip([True, False], axs):
    make_rose_plot(all_adip_arrows, all_adip_sizes, all_adip_magnitudes, exclude_empty=tf, fit = True, title = "Fig 1_emb1 Right", ax = ax[1], relative=np.pi,)

plt.tight_layout()

# Load protein data from all experiment folders
data_all_protein = import_data("myexperiment", "protein", subfolders=True)

# Combine all protein vectors, magnitudes, and sizes from all experiments
all_combined_arrows = []
all_combined_magnitudes = []
all_combined_sizes = []

for exp_name in data_all_protein.keys():
    exp_data = data_all_protein[exp_name]
    
    # Iterate through each sub-experiment in this folder
    for sub_exp in exp_data["protein_arrows"].keys():
        all_combined_arrows.extend(exp_data["protein_arrows"][sub_exp])
        all_combined_magnitudes.extend(exp_data["protein_magnitudes"][sub_exp])
        all_combined_sizes.extend(exp_data["protein_sizes"][sub_exp])

# Convert to numpy arrays
all_combined_arrows = np.array(all_combined_arrows)
all_combined_magnitudes = np.array(all_combined_magnitudes)
all_combined_sizes = np.array(all_combined_sizes)

print(f"Total protein vectors combined: {len(all_combined_arrows)}")
print(f"Mean magnitude: {np.mean(all_combined_magnitudes):.2f} ± {np.std(all_combined_magnitudes):.2f}")

# Generate rose plot with combined protein data
fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300, subplot_kw=dict(polar=True))

make_rose_plot(
    all_combined_arrows, 
    all_combined_sizes, 
    all_combined_magnitudes, 
    exclude_empty=False, 
    fit=True, 
    title="Combined Protein Data Rose Plot", 
    ax=ax, 
    relative=np.pi
)

plt.tight_layout()
plt.show()
