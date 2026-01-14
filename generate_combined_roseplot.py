"""
Generate combined rose plot from all protein data
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from aggregate_analysis_tools import make_rose_plot

# Paths to all experiment folders
experiment_paths = [
    "all_data_structured/myexperiment/Fig1_opt1/results/protein/",
    "all_data_structured/myexperiment/Fig1_opt2/results/protein/",
    "all_data_structured/myexperiment/Fig1_opt3/results/protein/"
]

# Initialize combined arrays
all_combined_arrows = []
all_combined_magnitudes = []
all_combined_sizes = []

# Load data from each experiment
for i, path in enumerate(experiment_paths, 1):
    print(f"\nLoading experiment {i} from: {path}")
    
    # Load CSV files
    arrows_df = pd.read_csv(path + "protein_arrows_div.csv", header=None)
    magnitudes_df = pd.read_csv(path + "protein_magnitudes_div.csv", header=None)
    sizes_df = pd.read_csv(path + "protein_sizes_div.csv", header=None)
    
    # Convert to arrays (assuming 2 columns for arrows: x, y)
    arrows = arrows_df.values
    magnitudes = magnitudes_df.values.flatten()
    sizes = sizes_df.values.flatten()
    
    print(f"  Loaded {len(arrows)} protein vectors")
    print(f"  Mean magnitude: {np.mean(magnitudes):.2f} ± {np.std(magnitudes):.2f}")
    
    # Append to combined arrays
    all_combined_arrows.extend(arrows)
    all_combined_magnitudes.extend(magnitudes)
    all_combined_sizes.extend(sizes)

# Convert to numpy arrays
all_combined_arrows = np.array(all_combined_arrows)
all_combined_magnitudes = np.array(all_combined_magnitudes)
all_combined_sizes = np.array(all_combined_sizes)

print(f"\n{'='*60}")
print(f"COMBINED DATA SUMMARY:")
print(f"Total protein vectors: {len(all_combined_arrows)}")
print(f"Mean magnitude: {np.mean(all_combined_magnitudes):.2f} ± {np.std(all_combined_magnitudes):.2f}")
print(f"{'='*60}\n")

# Generate rose plot with combined protein data
fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300, subplot_kw=dict(polar=True))

make_rose_plot(
    all_combined_arrows, 
    all_combined_sizes, 
    all_combined_magnitudes, 
    exclude_empty=False, 
    fit=True, 
    title="Combined Protein Data - All Datasets", 
    ax=ax, 
    relative=np.pi
)

plt.tight_layout()

# Save the figure as compact PNG
output_path = "combined_roseplot_all_datasets.png"
plt.savefig(output_path, bbox_inches='tight', dpi=300, format='png')
print(f"Rose plot saved to: {output_path}")

plt.show()
