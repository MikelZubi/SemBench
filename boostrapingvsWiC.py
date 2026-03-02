import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from collections import defaultdict

class BenchmarkBootstrapAnalyzer:
    def __init__(self, data_path: str, gold_path: str, model_names: List[str], dataset_name: str = "Sembench",
                 difficulties: List[str] = ['random', 'easy', 'medium', 'hard'],
                 lengths: List[int] = [50, 100, 200, 500],
                 n_bootstrap: int = 1000,
                 random_seed: int = 42,
                 ):
        """
        Initialize the bootstrap analyzer for LLM benchmarks.
        
        Args:
            data_path: Base path where JSONL files are stored
            model_names: List of model names to analyze
            difficulties: List of difficulty levels
            lengths: List of sample sizes to test
            n_bootstrap: Number of bootstrap iterations
            random_seed: Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.gold_path = Path(gold_path)
        self.model_names = model_names
        self.difficulties = difficulties
        self.lengths = lengths
        self.dataset_name = dataset_name
        self.n_bootstrap = n_bootstrap
        self.random_seed = random_seed
        self.max_length = 1000  # Assuming max benchmark length is 1000
        np.random.seed(random_seed)
        
        self.data = {}
        self.results = defaultdict(dict)
        
    def load_data(self):
        """Load all JSONL files for each model and difficulty."""
        print("Loading data...")
        for model in self.model_names:
            self.data[model] = {}
            for difficulty in self.difficulties:
                file_path = self.data_path / f"{model}_{difficulty}.json"
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        self.data[model][difficulty] = [json.loads(line) for line in f]
                    print(f"Loaded {len(self.data[model][difficulty])} examples for {model} - {difficulty}")
                else:
                    print(f"Warning: File not found: {file_path}")
                    self.data[model][difficulty] = []
    
    def extract_scores(self, examples: List[Dict]) -> np.ndarray:
        """
        Extract scores from examples. Modify this based on your JSONL structure.
        Assumes each example has a 'score' or 'correct' field.
        """
        scores = []
        for ex in examples:
            # Adjust these keys based on your actual JSONL structure
            if ex["pred_label"] == ex["label"]:
                scores.append(1.0)
            else:
                scores.append(0.0)
        return np.array(scores)
    
    def bootstrap_ranking(self, difficulty: str, length: int) -> np.ndarray:
        """
        Generate one bootstrap sample and return model rankings.
        
        Args:
            difficulty: Difficulty level to analyze
            length: Sample size for bootstrap
            
        Returns:
            Array of rankings (1 = best, n = worst) for each model
        """
        model_scores = {}
        indices = np.random.choice(self.max_length, size=length, replace=True)
        for model in self.model_names:
            scores = self.extract_scores(self.data[model][difficulty])
            if len(scores) >= length:
                # Bootstrap sample
                model_scores[model] = np.mean(scores[indices])
            else:
                model_scores[model] = np.nan
        
        # Convert scores to rankings (higher score = better rank)
        valid_models = {k: v for k, v in model_scores.items() if not np.isnan(v)}
        sorted_models = sorted(valid_models.items(), key=lambda x: x[1], reverse=True)
        
        rankings = {}
        for rank, (model, score) in enumerate(sorted_models, 1):
            rankings[model] = rank
        
        # Return rankings in original model order
        return np.array([rankings.get(model, np.nan) for model in self.model_names])
    
    def calculate_ranking_correlation(self, rankings1: np.ndarray, rankings2: np.ndarray) -> float:
        """
        Calculate Spearman correlation between two ranking arrays.
        Handles NaN values by excluding them from correlation calculation.
        """
        # Remove NaN values
        mask = ~(np.isnan(rankings1) | np.isnan(rankings2))
        if np.sum(mask) < 2:
            return np.nan
        
        valid_rankings1 = rankings1[mask]
        valid_rankings2 = rankings2[mask]
        
        # If all rankings are the same, correlation is undefined
        if len(np.unique(valid_rankings1)) == 1 or len(np.unique(valid_rankings2)) == 1:
            return np.nan
        
        corr, _ = stats.spearmanr(valid_rankings1, valid_rankings2)
        return corr
    
    def get_gold_ranking(self) -> np.ndarray:
        """
        Get the ranking based on the WiC dataset data. 
        """
        model_scores = {}
        
        for model in self.model_names:
            with open(self.gold_path / f"{model}_thr05.txt", 'r') as f:
                lines = f.readlines()
                score = float(lines[0].replace("Definition: ", ""))
            model_scores[model] = score
        
        # Convert scores to rankings (higher score = better rank)
        valid_models = {k: v for k, v in model_scores.items() if not np.isnan(v)}
        sorted_models = sorted(valid_models.items(), key=lambda x: x[1], reverse=True)
        
        rankings = {}
        for rank, (model, score) in enumerate(sorted_models, 1):
            rankings[model] = rank
        
        # Return rankings in original model order
        return np.array([rankings.get(model, np.nan) for model in self.model_names])
    
    def analyze_ranking_stability_by_length(self):
        """
        Analyze how model rankings correlate across different benchmark lengths.
        Compares rankings at each length against the full benchmark (using all 1000 examples).
        """
        print("\nAnalyzing ranking stability across lengths...")
        full_ranking = self.get_gold_ranking()
        for difficulty in self.difficulties:
            print(f"\nProcessing difficulty: {difficulty}")
            self.results[difficulty] = {
                'ranking_stability': {},
                'bootstrap_distributions': {},
                'full_ranking': None
            }
            
            # Get the TRUE baseline ranking using ALL available data (no bootstrap)
            #full_ranking = self.get_full_ranking(difficulty)
            self.results[difficulty]['full_ranking'] = full_ranking
            
            print(f"  Full benchmark ranking: {full_ranking}")
            
            # Analyze correlation at each length vs FULL RANKING (1000)
            for length in self.lengths:
                print(f"  Analyzing length: {length}")
                
                correlations = []
                length_rankings = []
                
                for _ in range(self.n_bootstrap):
                    # Get rankings at current length
                    rankings = self.bootstrap_ranking(difficulty, length)
                    length_rankings.append(rankings)
                    
                    # Calculate correlation with FULL ranking (1000)
                    corr = self.calculate_ranking_correlation(rankings, full_ranking)
                    if not np.isnan(corr):
                        correlations.append(corr)
                
                length_rankings = np.array(length_rankings)
                
                self.results[difficulty]['ranking_stability'][length] = {
                    'mean_correlation': np.mean(correlations) if correlations else np.nan,
                    'std_correlation': np.std(correlations) if correlations else np.nan,
                    'ci_lower': np.percentile(correlations, 2.5) if correlations else np.nan,
                    'ci_upper': np.percentile(correlations, 97.5) if correlations else np.nan,
                    'mean_rankings': np.nanmean(length_rankings, axis=0),
                    'std_rankings': np.nanstd(length_rankings, axis=0)
                }
                
                self.results[difficulty]['bootstrap_distributions'][length] = length_rankings
            '''
            # Also add the comparison with 1000 (full benchmark)
            self.results[difficulty]['ranking_stability'][1000] = {
                'mean_correlation': 1.0,  # Perfect correlation with itself
                'std_correlation': 0.0,
                'ci_lower': 1.0,
                'ci_upper': 1.0,
                'mean_rankings': full_ranking,
                'std_rankings': np.zeros_like(full_ranking)
            }
            '''
    
    def analyze_ranking_correlation_between_lengths(self):
        """
        Analyze pairwise correlations between rankings at different lengths.
        Also includes comparisons with the full benchmark (1000).
        """
        print("\nAnalyzing pairwise ranking correlations between lengths...")
        
        for difficulty in self.difficulties:
            print(f"\nProcessing difficulty: {difficulty}")
            self.results[difficulty]['pairwise_correlations'] = {}
            
            # Get full ranking
            full_ranking = self.results[difficulty]['full_ranking']
            
            # Compare each length with 1000 (full benchmark)
            for length in self.lengths:
                pair_key = f"{length}_vs_1000"
                print(f"  Comparing {pair_key}")
                
                correlations = []
                
                for _ in range(self.n_bootstrap):
                    rankings = self.bootstrap_ranking(difficulty, length)
                    
                    corr = self.calculate_ranking_correlation(rankings, full_ranking)
                    if not np.isnan(corr):
                        correlations.append(corr)
                
                self.results[difficulty]['pairwise_correlations'][pair_key] = {
                    'mean_correlation': np.mean(correlations) if correlations else np.nan,
                    'std_correlation': np.std(correlations) if correlations else np.nan,
                    'ci_lower': np.percentile(correlations, 2.5) if correlations else np.nan,
                    'ci_upper': np.percentile(correlations, 97.5) if correlations else np.nan,
                    'correlations': correlations
                }
            
            # Compare each pair of lengths (excluding 1000)
            for i, length1 in enumerate(self.lengths):
                for length2 in self.lengths[i+1:]:
                    pair_key = f"{length1}_vs_{length2}"
                    print(f"  Comparing {pair_key}")
                    
                    correlations = []
                    
                    for _ in range(self.n_bootstrap):
                        rankings1 = self.bootstrap_ranking(difficulty, length1)
                        rankings2 = self.bootstrap_ranking(difficulty, length2)
                        
                        corr = self.calculate_ranking_correlation(rankings1, rankings2)
                        if not np.isnan(corr):
                            correlations.append(corr)
                    
                    self.results[difficulty]['pairwise_correlations'][pair_key] = {
                        'mean_correlation': np.mean(correlations) if correlations else np.nan,
                        'std_correlation': np.std(correlations) if correlations else np.nan,
                        'ci_lower': np.percentile(correlations, 2.5) if correlations else np.nan,
                        'ci_upper': np.percentile(correlations, 97.5) if correlations else np.nan,
                        'correlations': correlations
                    }
    
    def analyze_full_ranking_correlation_between_difficulties(self):
        """
        Analyze correlations of full benchmark rankings between different difficulties.
        """
        print("\nAnalyzing full ranking correlations between difficulties...")
        
        for i, diff1 in enumerate(self.difficulties):
            for diff2 in self.difficulties[i+1:]:
                pair_key = f"{diff1}_vs_{diff2}"
                print(f"  Comparing {pair_key}")
                
                full_ranking1 = self.results[diff1]['full_ranking']
                full_ranking2 = self.results[diff2]['full_ranking']
                
                corr = self.calculate_ranking_correlation(full_ranking1, full_ranking2)
                
                self.results.setdefault('difficulty_correlations', {})
                self.results['difficulty_correlations'][pair_key] = {
                    'correlation': corr
                }
    def plot_full_ranking_correlation_heatmap(self, save_path: str = None):
        """Plot heatmap of full ranking correlations between difficulties."""
        difficulties = self.difficulties
        n = len(difficulties)
        corr_matrix = np.ones((n, n))
        
        for i, diff1 in enumerate(difficulties):
            for j, diff2 in enumerate(difficulties):
                if i < j:
                    pair_key = f"{diff1}_vs_{diff2}"
                    if 'difficulty_correlations' in self.results and pair_key in self.results['difficulty_correlations']:
                        corr = self.results['difficulty_correlations'][pair_key]['correlation']
                        corr_matrix[i, j] = corr
                        corr_matrix[j, i] = corr
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', 
                    xticklabels=difficulties, yticklabels=difficulties,
                    cmap='RdYlGn', center=0.8, vmin=0, vmax=1)
        plt.title('Full Benchmark Ranking Correlations Between Difficulties', fontsize=16)
        plt.xlabel('Difficulty', fontsize=12)
        plt.ylabel('Difficulty', fontsize=12)
        
        if save_path:
            plt.savefig(f"{save_path}/full_ranking_difficulty_correlations.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_ranking_stability(self, save_path: str = None):
        """Plot how ranking correlation changes with benchmark length."""
        # Store results for later combined plotting
        self.stability_plot_data = {
            'dataset_name': self.dataset_name,
            'difficulties': self.difficulties,
            'results': self.results
        }
        return self.stability_plot_data
    
    @staticmethod
    def plot_combined_ranking_stability(data_def, data_exp, save_path: str = None):
        """
        Create a combined plot with (2,4) layout for Definition and Example analyses.
        
        Args:
            data_def: Plot data from Definition analysis
            data_exp: Plot data from Example analysis
            save_path: Path to save the figure
        """
        # Set style for publication quality
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("colorblind")
        sns.set_theme(style="whitegrid",context="paper")
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # Main title
        #fig.suptitle('Ranking Stability: Sembench vs WiC (Threshold 0.5, 5-Shot)', 
        #             fontsize=18, fontweight='bold', y=0.98)
        
        # Row titles
        #axes[0, 0].text(-0.3, 0.5, 'From Definition', fontsize=14,
        #              rotation=90, transform=axes[0, 0].transAxes, 
        #               verticalalignment='center', horizontalalignment='center')
        #axes[1, 0].text(-0.3, 0.5, 'From Example', fontsize=14,
        #               rotation=90, transform=axes[1, 0].transAxes, 
        #               verticalalignment='center', horizontalalignment='center')
        
        # Color scheme for better aesthetics
        color_mean = '#2E86AB'
        #color_ci = '#A23B72'
        difficulty_mapping = {'random':'RAND', 'easy': 'EASY', 'medium': 'MED', 'hard': 'HARD'}
        # Process each difficulty level
        for idx, difficulty in enumerate(data_def['difficulties']):
            # Definition row (top) - column for each difficulty
            ax_def = axes[0, idx]
            
            if 'ranking_stability' in data_def['results'][difficulty]:
                lengths = []
                means = []
                ci_lower = []
                ci_upper = []
                
                all_lengths = sorted(list(data_def['results'][difficulty]['ranking_stability'].keys()))
                
                for length in all_lengths:
                    data = data_def['results'][difficulty]['ranking_stability'][length]
                    lengths.append(length)
                    means.append(data['mean_correlation'])
                    ci_lower.append(data['ci_lower'])
                    ci_upper.append(data['ci_upper'])
                
                # Plot with improved styling
                ax_def.plot(lengths, means, 'o-', linewidth=2.5, markersize=9, 
                           color=color_mean, label='Mean Correlation', markeredgewidth=1.5, 
                           markeredgecolor='white')
                ax_def.fill_between(lengths, ci_lower, ci_upper, alpha=0.25, 
                                   color=color_mean, label='95% CI')
                
                # Styling
                plot_difficulty = difficulty_mapping[difficulty]
                ax_def.set_title(f'{plot_difficulty}', fontsize=21,  pad=10)
                if idx == 0:
                    ax_def.set_ylabel("From Definition ρ", fontsize=19)
                ax_def.tick_params(labelsize=17)
                ax_def.grid(True, alpha=0.5, linestyle='--', linewidth=1.2)
                ax_def.set_ylim([0, 1.05])
                ax_def.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
            
            # Example row (bottom) - column for each difficulty
            ax_exp = axes[1, idx]
            
            if 'ranking_stability' in data_exp['results'][difficulty]:
                lengths = []
                means = []
                ci_lower = []
                ci_upper = []
                
                all_lengths = sorted(list(data_exp['results'][difficulty]['ranking_stability'].keys()))
                
                for length in all_lengths:
                    data = data_exp['results'][difficulty]['ranking_stability'][length]
                    lengths.append(length)
                    means.append(data['mean_correlation'])
                    ci_lower.append(data['ci_lower'])
                    ci_upper.append(data['ci_upper'])
                
                # Plot with improved styling
                ax_exp.plot(lengths, means, 'o-', linewidth=2.5, markersize=9, 
                           color=color_mean, label='Mean Correlation', markeredgewidth=1.5,
                           markeredgecolor='white')
                ax_exp.fill_between(lengths, ci_lower, ci_upper, alpha=0.25, 
                                   color=color_mean, label='95% CI')
                
                # Styling
                if idx == 0:
                    ax_exp.set_ylabel("From Example ρ", fontsize=19)
                ax_exp.set_xlabel('Benchmark Size', fontsize=19)
                ax_exp.tick_params(labelsize=17)
                ax_exp.grid(True, alpha=0.5, linestyle='--', linewidth=1.2)
                ax_exp.set_ylim([0, 1.05])
                ax_exp.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        
        # Create a single legend for the entire figure, placed at the top center
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=len(labels), 
                  frameon=True, shadow=False, fontsize=18, 
                  bbox_to_anchor=(0.5, 1.015))

        # Adjust layout to make room for the legend
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_path:
            plt.savefig(f"{save_path}/combined_ranking_stability.png", 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.savefig(f"{save_path}/combined_ranking_stability.pdf", 
                       dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def plot_pairwise_correlation_heatmap(self, save_path: str = None):
        """Plot heatmap of correlations between different lengths including 1000."""
        fig, axes = plt.subplots(2, 2, figsize=(18, 16))
        fig.suptitle('Pairwise Ranking Correlations Between Lengths (including 1000)', fontsize=16)
        
        for idx, difficulty in enumerate(self.difficulties):
            ax = axes[idx // 2, idx % 2]
            
            if 'pairwise_correlations' not in self.results[difficulty]:
                continue
            
            # Include 1000 in the lengths for the heatmap
            all_lengths = self.lengths + [1000]
            n = len(all_lengths)
            corr_matrix = np.ones((n, n))
            
            for i, length1 in enumerate(all_lengths):
                for j, length2 in enumerate(all_lengths):
                    if i == j:
                        corr_matrix[i, j] = 1.0
                    elif i < j:
                        # Check for both orderings of the pair
                        pair_key1 = f"{length1}_vs_{length2}"
                        pair_key2 = f"{length2}_vs_{length1}"
                        
                        if pair_key1 in self.results[difficulty]['pairwise_correlations']:
                            corr = self.results[difficulty]['pairwise_correlations'][pair_key1]['mean_correlation']
                            corr_matrix[i, j] = corr
                            corr_matrix[j, i] = corr
                        elif pair_key2 in self.results[difficulty]['pairwise_correlations']:
                            corr = self.results[difficulty]['pairwise_correlations'][pair_key2]['mean_correlation']
                            corr_matrix[i, j] = corr
                            corr_matrix[j, i] = corr
            
            sns.heatmap(corr_matrix, annot=True, fmt='.3f',
                       xticklabels=all_lengths, yticklabels=all_lengths,
                       cmap='RdYlGn', center=0.8, vmin=0, vmax=1, ax=ax)
            ax.set_title(f'Difficulty: {difficulty}', fontsize=14)
            ax.set_xlabel('Benchmark Length', fontsize=12)
            ax.set_ylabel('Benchmark Length', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/pairwise_correlations.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_ranking_distributions(self, save_path: str = None):
        """Plot ranking distributions for each model at different lengths."""
        for difficulty in self.difficulties:
            if 'bootstrap_distributions' not in self.results[difficulty]:
                continue
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Model Ranking Distributions - {difficulty}', fontsize=16)
            
            for idx, length in enumerate(self.lengths[:4]):  # Plot first 4 lengths
                if idx >= 4:
                    break
                
                ax = axes[idx // 2, idx % 2]
                
                if length not in self.results[difficulty]['bootstrap_distributions']:
                    continue
                
                rankings = self.results[difficulty]['bootstrap_distributions'][length]
                
                # Plot distribution for each model
                positions = np.arange(len(self.model_names))
                for i, model in enumerate(self.model_names):
                    model_rankings = rankings[:, i]
                    model_rankings = model_rankings[~np.isnan(model_rankings)]
                    
                    if len(model_rankings) > 0:
                        ax.violinplot([model_rankings], positions=[positions[i]], 
                                     showmeans=True, showmedians=True, widths=0.7)
                
                ax.set_xticks(positions)
                ax.set_xticklabels(self.model_names, rotation=45, ha='right')
                ax.set_ylabel('Rank (1=best)', fontsize=12)
                ax.set_title(f'Length = {length}', fontsize=14)
                ax.set_ylim([0, len(self.model_names) + 1])
                ax.invert_yaxis()  # Lower rank number at top
                ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(f"{save_path}/ranking_distributions_{difficulty}.png", 
                           dpi=300, bbox_inches='tight')
            plt.show()
    
    def generate_report(self, output_file: str = "ranking_correlation_report.txt"):
        """Generate a text report of the ranking correlation analysis."""
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("BENCHMARK RANKING CORRELATION ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Models analyzed: {', '.join(self.model_names)}\n")
            f.write(f"Bootstrap iterations: {self.n_bootstrap}\n")
            f.write(f"Lengths tested: {', '.join(map(str, self.lengths))}\n")
            f.write(f"Baseline: Full benchmark (1000 examples, no bootstrap)\n\n")
            
            for difficulty in self.difficulties:
                f.write(f"\n{'='*80}\n")
                f.write(f"DIFFICULTY: {difficulty.upper()}\n")
                f.write(f"{'='*80}\n")
                
                # Full benchmark ranking
                if 'full_ranking' in self.results[difficulty]:
                    f.write("\nFull Benchmark Ranking (1000 examples):\n")
                    f.write("-" * 80 + "\n")
                    full_ranking = self.results[difficulty]['full_ranking']
                    for i, model in enumerate(self.model_names):
                        if not np.isnan(full_ranking[i]):
                            f.write(f"  {model}: Rank {int(full_ranking[i])}\n")
                
                # Ranking stability vs 1000
                f.write("\n\nRanking Stability (Correlation with Full 1000-example Benchmark):\n")
                f.write("-" * 80 + "\n")
                if 'ranking_stability' in self.results[difficulty]:
                    for length in sorted(self.results[difficulty]['ranking_stability'].keys()):
                        if length == 1000:
                            continue  # Skip 1000 vs 1000 (always 1.0)
                        
                        data = self.results[difficulty]['ranking_stability'][length]
                        f.write(f"\nLength {length} vs 1000:\n")
                        f.write(f"  Correlation: {data['mean_correlation']:.4f} ± {data['std_correlation']:.4f}\n")
                        f.write(f"  95% CI: [{data['ci_lower']:.4f}, {data['ci_upper']:.4f}]\n")
                        f.write(f"  Mean Rankings: ")
                        for i, model in enumerate(self.model_names):
                            if not np.isnan(data['mean_rankings'][i]):
                                f.write(f"{model}={data['mean_rankings'][i]:.2f} ")
                        f.write("\n")
                
                # Pairwise correlations (including vs 1000)
                f.write("\n\nPairwise Correlations Between Lengths:\n")
                f.write("-" * 80 + "\n")
                if 'pairwise_correlations' in self.results[difficulty]:
                    # First show comparisons with 1000
                    f.write("\nComparisons with Full Benchmark (1000):\n")
                    for length in self.lengths:
                        pair_key = f"{length}_vs_1000"
                        if pair_key in self.results[difficulty]['pairwise_correlations']:
                            data = self.results[difficulty]['pairwise_correlations'][pair_key]
                            f.write(f"  {pair_key}: {data['mean_correlation']:.4f} ± {data['std_correlation']:.4f}")
                            f.write(f" [{data['ci_lower']:.4f}, {data['ci_upper']:.4f}]\n")
                    
                    # Then show other pairwise comparisons
                    f.write("\nOther Pairwise Comparisons:\n")
                    for pair_key, data in sorted(self.results[difficulty]['pairwise_correlations'].items()):
                        if '_vs_1000' not in pair_key:
                            f.write(f"  {pair_key}: {data['mean_correlation']:.4f} ± {data['std_correlation']:.4f}")
                            f.write(f" [{data['ci_lower']:.4f}, {data['ci_upper']:.4f}]\n")
                
                f.write("\n")
        
        print(f"\nReport saved to {output_file}")
    
    def run_full_analysis(self, save_plots: bool = True, plot_path: str = "./plots", 
                          generate_individual_plots: bool = False):
        """Run the complete ranking correlation analysis pipeline."""
        if save_plots:
            Path(plot_path).mkdir(parents=True, exist_ok=True)
        
        self.load_data()
        self.analyze_ranking_stability_by_length()
        self.analyze_ranking_correlation_between_lengths()
        
        print("\nGenerating visualizations...")
        plot_data = self.plot_ranking_stability(save_path=plot_path if save_plots else None)
        
        if generate_individual_plots:
            self.plot_pairwise_correlation_heatmap(save_path=plot_path if save_plots else None)
            self.plot_ranking_distributions(save_path=plot_path if save_plots else None)
        
        self.generate_report()
        
        print("\nAnalysis complete!")
        return self.results, plot_data
        

    
    


# Example usage
if __name__ == "__main__":
    # Configure your analysis for Examples
    analyzer_exp = BenchmarkBootstrapAnalyzer(
        data_path="WSDOutputs/5Shot/",
        gold_path="WiCOutputs/5Shot/",
        model_names=["Llama2", "Llama3_8B", "Llama3_70B", "Gemma3_4B", "Gemma3_12B", 
                     "Gemma3_27B", "Qwen3_4B","Qwen3_8B", "Qwen3_14B", "Qwen3_32B", 
                     "Latxa_8B", "Latxa_70B"],
        lengths=[50, 100, 200, 500, 1000],
        n_bootstrap=100,
        random_seed=42,
        dataset_name="Sembench from Example"
    )
    
    # Run the analysis for Examples
    print("Running analysis for Sembench from Example...")
    results_exp, plot_data_exp = analyzer_exp.run_full_analysis(
        save_plots=True, 
        plot_path="./bootstrap_plots_SMB_Exp",
        generate_individual_plots=False
    )

    # Configure your analysis for Definitions
    analyzer_def = BenchmarkBootstrapAnalyzer(
        data_path="WSDEOutputs/5Shot/",
        gold_path="WiCOutputs/5Shot/",
        model_names=["Llama2", "Llama3_8B", "Llama3_70B", "Gemma3_4B", "Gemma3_12B", 
                     "Gemma3_27B", "Qwen3_4B","Qwen3_8B", "Qwen3_14B", "Qwen3_32B", 
                     "Latxa_8B", "Latxa_70B"],
        lengths=[50, 100, 200, 500, 1000],
        n_bootstrap=100,
        random_seed=42,
        dataset_name="Sembench from Definition"
    )
    
    # Run the analysis for Definitions
    print("\nRunning analysis for Sembench from Definition...")
    results_def, plot_data_def = analyzer_def.run_full_analysis(
        save_plots=True, 
        plot_path="./bootstrap_plots_SMB_Def",
        generate_individual_plots=False
    )
    
    # Create combined plot
    print("\nGenerating combined plot...")
    Path("./bootstrap_plots_combined").mkdir(parents=True, exist_ok=True)
    BenchmarkBootstrapAnalyzer.plot_combined_ranking_stability(
        plot_data_def, 
        plot_data_exp, 
        save_path="./bootstrap_plots_combined"
    )
    
    print("\nAll analyses complete! Combined plot saved to ./bootstrap_plots_combined/")
    
    # Access specific results if needed
    # For example: results['easy']['ranking_stability'][100]



