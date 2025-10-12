import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import numpy as np

# Your existing code
dict_of_paths = {
    "SMB_Def_easy": "WSDOutputs/easy.csv",
    "SMB_Def_medium": "WSDOutputs/medium.csv",
    "SMB_Def_hard": "WSDOutputs/hard.csv",
    "SMB_Def_random": "WSDOutputs/random.csv",
    "SMB_Exp_easy": "WSDEOutputs/easy.csv",
    "SMB_Exp_Emedium": "WSDEOutputs/medium.csv",
    "SMB_Exp_Ehard": "WSDEOutputs/hard.csv",
    "SMB_Exp_Erandom": "WSDEOutputs/random.csv",
    "WiC": "WiCOutputs/definition.csv",
    "WiCthr05": "WiCOutputs/definition_thr05.csv"
}
all_dfs = {name: pd.read_csv(path).set_index("Shot") for name, path in dict_of_paths.items()}
all_combined = pd.concat(all_dfs, axis=1, join="inner")


# Function to compute Spearman correlation matrix
def compute_spearman_correlation(df, index):
    datasets = list(dict_of_paths.keys())
    corr_matrix = pd.DataFrame(index=datasets, columns=datasets, dtype=float)
    
    for dataset1 in datasets:
        for dataset2 in datasets:
            # Get all columns for each dataset
            data1 = df[dataset1].loc[index].rank(ascending=False).astype(int)
            data1.name = "dataset1"
            data2 = df[dataset2].loc[index].rank(ascending=False).astype(int)
            data2.name = "dataset2"
            data_concat = pd.concat([data1, data2], axis=1).dropna()
            corr, _ = spearmanr(data_concat["dataset1"], data_concat["dataset2"])
            corr_matrix.loc[dataset1, dataset2] = corr
            
    
    return corr_matrix.astype(float)


# Compute correlation matrices
corr_0shot = compute_spearman_correlation(all_combined, 0)
corr_5shot = compute_spearman_correlation(all_combined, 5)

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Plot 0-shot correlation
sns.heatmap(corr_0shot, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0.75, vmin=0.50, vmax=1, square=True, 
            cbar_kws={'label': 'Spearman Correlation'},
            ax=axes[0], linewidths=0.5)
axes[0].set_title('0-Shot Spearman Correlation Between Datasets', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Dataset', fontsize=12)
axes[0].set_ylabel('Dataset', fontsize=12)

# Plot 5-shot correlation
sns.heatmap(corr_5shot, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0.75, vmin=0.50, vmax=1, square=True,
            cbar_kws={'label': 'Spearman Correlation'},
            ax=axes[1], linewidths=0.5)
axes[1].set_title('5-Shot Spearman Correlation Between Datasets', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Dataset', fontsize=12)
axes[1].set_ylabel('Dataset', fontsize=12)

plt.tight_layout()
plt.savefig('spearman_correlation_heatmaps.png', dpi=300, bbox_inches='tight')
plt.show()

# Print correlation matrices
print("0-Shot Spearman Correlation Matrix:")
print(corr_0shot)
print("\n" + "="*80 + "\n")
print("5-Shot Spearman Correlation Matrix:")
print(corr_5shot)