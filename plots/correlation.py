import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import numpy as np
import argparse


# Argument parser
parser = argparse.ArgumentParser(description='Compute and plot Spearman correlation heatmaps.')
#TODO: Add language to the parser:
parser.add_argument('--language', type=str, help='The languages that will be evaluated: "EN" for English, "ES" for Spanish')
parser.add_argument("--big-axis", action="store_true", help="Use bigger axes for the plots")
parser.set_defaults(big_axis=False)
parser.set_defaults(language="EN")
args = parser.parse_args()
language = args.language
big_axis = args.big_axis
sns.set_theme(style="whitegrid",context="paper")
if language == "EN":
# Your existing code
    dict_of_paths = {
        "Exp EASY": "WSDOutputs/easy.csv",
        "Exp MED": "WSDOutputs/medium.csv",
        "Exp HARD": "WSDOutputs/hard.csv",
        "Exp RAND": "WSDOutputs/random.csv",
        "Def EASY": "WSDEOutputs/easy.csv",
        "Def MED": "WSDEOutputs/medium.csv",
        "Def HARD": "WSDEOutputs/hard.csv",
        "Def RAND": "WSDEOutputs/random.csv",
        #"WiC": "WiCOutputs/definition.csv",
        "WiC": "WiCOutputs/definition_thr05.csv"
    }
    shots = [0, 5]
else:
    dict_of_paths = {
        "Def RAND": f"WSDEOutputs{language}/random.csv",
        "WiC": f"WiCOutputs{language}/definition.csv",
        "WiC": f"WiCOutputs{language}/definition_thr05.csv"
    }
    shots = [5]
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
if language == "EN":
    for shot in shots:
        corr_matrices = {}
        for shot in shots:
            corr_matrices[shot] = compute_spearman_correlation(all_combined, shot)

        # ===== PLOT 1: Full heatmap with all datasets =====
        # Create figure with subplots based on number of shots
        fig, axes = plt.subplots(1, len(shots), figsize=(8.5 * len(shots), 7))

        # Handle case where there's only one shot (axes won't be an array)
        if len(shots) == 1:
            axes = [axes]

        for idx, shot in enumerate(shots):
            if idx == 0:
                cbar =False
            else:
                cbar = True
            sns.heatmap(corr_matrices[shot], annot=True, fmt='.2f', cmap='RdYlBu_r', 
                    center=0.5, cbar=cbar,vmin=0.0, vmax=1, square=True, 
                    cbar_kws={'label': "Spearman's ρ", 'shrink': 0.8},
                    ax=axes[idx], linewidths=1.5, linecolor='white',
                    annot_kws={'size': 10})
            # Make colorbar label bigger and move it to the right
            
            
            axes[idx].set_title(f'{shot}-Shot', fontsize=16, pad=15)
            #axes[idx].set_xlabel('Dataset', fontsize=13, fontweight='bold', labelpad=10)
            #axes[idx].set_ylabel('Dataset', fontsize=13, fontweight='bold', labelpad=10)
            axes[idx].tick_params(labelsize=10)
            # Rotate labels for better readability
            axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45, ha='right')
            axes[idx].set_yticklabels(axes[idx].get_yticklabels(), rotation=0)
        cbar = axes[1].collections[0].colorbar
        cbar.ax.set_ylabel("ρ", fontsize=11, labelpad=20)
        cbar.ax.tick_params(labelsize=10)

        plt.tight_layout()
        plt.savefig('SpearmanCorrelations/'+language+str(shot)+'Shot.pdf', dpi=300, bbox_inches='tight')
        plt.show()

        # ===== PLOT 2: Bar plots comparing 0-Shot vs 5-Shot for Exp and Def =====
        # Extract WiC correlations for all shots
        wic_corr_data = []
        for shot in shots:
            wic_col = corr_matrices[shot]['WiC']
            wic_corr_data.append(wic_col)
        
        # Create a DataFrame with WiC correlations
        wic_df = pd.DataFrame(wic_corr_data, columns=wic_col.index).T
        wic_df.columns = [f"{shot}" for shot in shots]
        # Remove the WiC row (self-correlation)
        wic_df = wic_df.drop('WiC', axis=0)
        
        # Separate Exp and Def datasets
        exp_datasets = [name for name in wic_df.index if name.startswith('Exp')]
        def_datasets = [name for name in wic_df.index if name.startswith('Def')]
        
        exp_df = wic_df.loc[exp_datasets]
        def_df = wic_df.loc[def_datasets]
        
        # Clean up labels for x-axis
        exp_labels = [label.replace('Exp ', '') for label in exp_datasets]
        def_labels = [label.replace('Def ', '') for label in def_datasets]
        
        # Colors from scatter plots
        color_0shot = "#AA2E2E"  # Red color (similar to line color)
        color_5shot = '#2E86AB'  # Blue color (similar to dot color)
        
        # ===== Create Combined Bar Plot (Exp and Def side by side) =====
        fig_combined, (ax_def, ax_exp) = plt.subplots(1, 2, figsize=(12, 6.5))
        
        x = np.arange(len(exp_labels))
        width = 0.35

        # Def subplot
        bars3 = ax_def.bar(x - width/2, def_df['0'], width, label='0-Shot', 
                          color=color_0shot, alpha=0.8, edgecolor='gray', linewidth=1)
        bars4 = ax_def.bar(x + width/2, def_df['5'], width, label='5-Shot', 
                          color=color_5shot, alpha=0.8, edgecolor='gray', linewidth=1)
        
        ax_def.set_xlabel('$SemBench_{Def}$', fontsize=25)
        ax_def.set_ylabel('$ρ_{WiC}$', fontsize=25)
        ax_def.set_xticks(x)
        ax_def.set_xticklabels(def_labels, fontsize=23)
        ax_def.tick_params(labelsize=23)
        ax_def.set_ylim(0, 1)
        ax_def.grid(True, alpha=0.35, axis='y', linestyle='--', linewidth=1.5, color="gray")
        
        # Exp subplot
        bars1 = ax_exp.bar(x - width/2, exp_df['0'], width, label='0-Shot', 
                          color=color_0shot, alpha=0.8, edgecolor='gray', linewidth=1)
        bars2 = ax_exp.bar(x + width/2, exp_df['5'], width, label='5-Shot', 
                          color=color_5shot, alpha=0.8, edgecolor='gray', linewidth=1)
        
        ax_exp.set_xlabel('$SemBench_{Ex}$', fontsize=25)
        #ax_exp.set_ylabel('$ρ_{WiC}$', fontsize=14)
        ax_exp.set_xticks(x)
        ax_exp.set_xticklabels(exp_labels, fontsize=23)
        ax_exp.tick_params(labelsize=23)
        ax_exp.set_ylim(0, 1)
        ax_exp.grid(True, alpha=0.3, axis='y', linestyle='--', color="gray", linewidth=0.7)
        
       
        
        # Add a single legend for the entire figure
        handles, labels = ax_def.get_legend_handles_labels()
        fig_combined.legend(handles, labels, 
                           fontsize=23,
                           framealpha=0.9,
                           loc='upper center',
                           bbox_to_anchor=(0.5, 1.03),
                           ncol=2,
                           columnspacing=1.2)
        
        plt.tight_layout()
        fig_combined.subplots_adjust(top=0.88)
        fig_combined.savefig('SpearmanCorrelations/'+language+'_0shot_vs_5shot_COMBINED.pdf', dpi=300, bbox_inches='tight')
        plt.close(fig_combined)
# Add correlation info
# Create scatter plot for SMB_Def_random vs WiCthr05
color_dot = '#2E86AB'
color_line = "#AA2E2E"
fig_scatter, ax_scatter = plt.subplots(figsize=(8, 7))

shot = 5
smb_data = all_combined[("Def RAND",)].loc[shot]
wic_data = all_combined[("WiC",)].loc[shot]

# Combine and drop NaN values
scatter_data = pd.concat([smb_data, wic_data], axis=1).dropna()

ax_scatter.set_xlabel('$SemBench$ $Acc$', fontsize=27)
ax_scatter.set_ylabel('$WiC$ $Acc$', fontsize=27)
if not big_axis:
    max_val_x = scatter_data.iloc[:,0].max() + 1.0
    min_val_x = scatter_data.iloc[:,0].min() - 1.0
    max_val_y = scatter_data.iloc[:,1].max() + 1.0
    min_val_y = scatter_data.iloc[:,1].min() - 1.0
else:
    max_val_x = 100.0
    min_val_x = 48.0
    max_val_y = 100.0
    min_val_y = 48.0
ax_scatter.set_xlim(min_val_x , max_val_x)
ax_scatter.set_ylim(min_val_y, max_val_y)
ax_scatter.set_aspect('equal', adjustable='box')
ax_scatter.grid(True, alpha=0.35, linestyle='--', linewidth=1.0, color="gray")
ax_scatter.tick_params(labelsize=23)
ax_scatter.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)

ax_scatter.scatter(scatter_data.iloc[:, 0], scatter_data.iloc[:, 1], 
                    alpha=0.7, s=120, label=f'{shot}-shot', 
                    edgecolors='black', linewidth=0.5, color=color_dot)

smb_data = all_combined[("Def RAND",)].loc[shot]
wic_data = all_combined[("WiC",)].loc[shot]
scatter_data = pd.concat([smb_data, wic_data], axis=1).dropna()
corr, p_value = spearmanr(scatter_data.iloc[:, 0], scatter_data.iloc[:, 1])
print(f"Spearman correlation (Def RAND vs WiC) for {shot}-shot: {corr:.3f}, p-value: {p_value}")

# Add regression/trend line
z = np.polyfit(scatter_data.iloc[:, 0], scatter_data.iloc[:, 1], 1)
p = np.poly1d(z)
# Use axis limits for the line
xlim = ax_scatter.get_xlim()
x_line = np.linspace(xlim[0], xlim[1], 100)

ax_scatter.plot(x_line, p(x_line), color=color_line, linewidth=2.5, 
                linestyle='--', alpha=0.8, clip_on=True)

ax_scatter.text(0.05, 0.95, f"ρ = {corr:.3f}\np-value = {p_value:.3e}",
                transform=ax_scatter.transAxes, verticalalignment='top',
                fontsize=23,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5, edgecolor='black', linewidth=1.5))

# Save the limits from the first scatter plot to use in the second one
first_plot_xlim = ax_scatter.get_xlim()
first_plot_ylim = ax_scatter.get_ylim()

plt.tight_layout()
if not big_axis:
    plt.savefig('SpearmanCorrelations/scatter_' + language + str(shot) + 'Shot.pdf', dpi=300, bbox_inches='tight')
else:
    plt.savefig('SpearmanCorrelations/scatter_' + language + str(shot) + 'Shot_bigAxis.pdf', dpi=300, bbox_inches='tight')
plt.show()

if language == "EN":
    # Add correlation info
    # Create scatter plot for SMB_Def_random vs WiCthr05
    color_dot = '#2E86AB'
    color_line = "#AA2E2E"
    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 7))

    smb_data = all_combined[("Exp RAND",)].loc[shot]
    wic_data = all_combined[("WiC",)].loc[shot]

    # Combine and drop NaN values
    scatter_data = pd.concat([smb_data, wic_data], axis=1).dropna()

    ax_scatter.set_xlabel('$SemBench$ $Acc$', fontsize=27)
    ax_scatter.set_ylabel('$WiC$ $Acc$', fontsize=27)
    # Use the same scale as the first scatter plot (Def RAND)
    ax_scatter.set_xlim(first_plot_xlim)
    ax_scatter.set_ylim(first_plot_ylim)
    ax_scatter.set_aspect('equal', adjustable='box')
    ax_scatter.grid(True, alpha=0.35, linestyle='--', linewidth=1.0, color="gray")
    ax_scatter.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax_scatter.tick_params(labelsize=23)

    ax_scatter.scatter(scatter_data.iloc[:, 0], scatter_data.iloc[:, 1], 
                        alpha=0.7, s=120, label=f'{shot}-shot', 
                        edgecolors='black', linewidth=0.5, color=color_dot)

    smb_data = all_combined[("Exp RAND",)].loc[shot]
    wic_data = all_combined[("WiC",)].loc[shot]
    scatter_data = pd.concat([smb_data, wic_data], axis=1).dropna()
    corr, p_value = spearmanr(scatter_data.iloc[:, 0], scatter_data.iloc[:, 1])
    print(f"Spearman correlation (Exp RAND vs WiC) for {shot}-shot: {corr:.3f}, p-value: {p_value}")

    # Add regression/trend line
    z = np.polyfit(scatter_data.iloc[:, 0], scatter_data.iloc[:, 1], 1)
    p = np.poly1d(z)
    # Use axis limits for the line
    xlim = ax_scatter.get_xlim()
    x_line = np.linspace(xlim[0], xlim[1], 100)

    ax_scatter.plot(x_line, p(x_line), color=color_line, linewidth=2.5, 
                    linestyle='--', alpha=0.8, clip_on=True)

    ax_scatter.text(0.05, 0.95, f"ρ = {corr:.3f}\np-value = {p_value:.3e}",
                    transform=ax_scatter.transAxes, verticalalignment='top',
                    fontsize=23,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5, edgecolor='black', linewidth=1.5))



    plt.tight_layout()
    if not big_axis:
        plt.savefig('SpearmanCorrelations/scatter_' + language + str(shot) + 'Shot_F_EXP.pdf', dpi=300, bbox_inches='tight')
    else:
        plt.savefig('SpearmanCorrelations/scatter_' + language + str(shot) + 'Shot_F_EXP_bigAxis.pdf', dpi=300, bbox_inches='tight')
    plt.show()

