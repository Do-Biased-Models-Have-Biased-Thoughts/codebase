import evaluate
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import json
import glob
import seaborn as sns
import numpy as np
from scipy.stats import norm, mannwhitneyu, shapiro
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import os
import argparse


class HarimAnalyzer:
    def __init__(self, model_name, base_thoughts_path="/thoughts", base_harim_path="/harim_scores"):
        """
        Initialize the analyzer for a specific model.
        
        Args:
            model_name (str): Name of the model (e.g., 'gemma', 'phi', 'mistral', 'llama8b', 'qwen')
            base_thoughts_path (str): Base path for thoughts data
            base_harim_path (str): Base path for harim scores output
        """
        self.model_name = model_name
        self.thoughts_path = f"{base_thoughts_path}/{model_name}"
        self.harim_path = f"{base_harim_path}/{model_name}"
        self.figures_path = f"{self.harim_path}/figures"
        
        # Create output directories if they don't exist
        os.makedirs(self.harim_path, exist_ok=True)
        os.makedirs(self.figures_path, exist_ok=True)
        
        # Initialize scorer
        self.scorer = None
        self.cutoff_percentile = None
        
    def load_scorer(self):
        """Load the HARIM+ scorer."""
        try:
            self.scorer = evaluate.load('NCSOFT/harim_plus')
            print(f"Successfully loaded HARIM+ scorer for {self.model_name}")
        except Exception as e:
            print(f"Error loading HARIM+ scorer: {e}")
            print("You may need to:")
            print("1. Uninstall and reinstall evaluate package")
            print("2. Create a new huggingface token")
            print("3. Login with: huggingface_hub.login(token='your_token_here')")
            raise
    
    def create_dataframe(self, folder_path):
        """Create DataFrame from JSONL files in a folder."""
        data = []
        pattern = f'{folder_path}/*.jsonl'
        files = glob.glob(pattern)
        
        if not files:
            print(f"Warning: No JSONL files found in {folder_path}")
            return pd.DataFrame()
            
        for file in files:
            try:
                with open(file, 'r') as infile:
                    for line in infile:
                        data.append(json.loads(line))
            except Exception as e:
                print(f"Error reading file {file}: {e}")
                
        return pd.DataFrame(data)
    
    def load_data(self):
        """Load test and validation data."""
        print(f"Loading data for {self.model_name}...")
        
        test_path = f"{self.thoughts_path}/split/test"
        val_path = f"{self.thoughts_path}/split/val"
        
        df_test = self.create_dataframe(test_path)
        df_val = self.create_dataframe(val_path)
        
        if df_test.empty or df_val.empty:
            raise ValueError(f"No data found for {self.model_name} in {self.thoughts_path}")
        
        # Filter out invalid predictions
        df_test = df_test[df_test['predicted_label'] != -1]
        df_val = df_val[df_val['predicted_label'] != -1]
        
        print(f"Test: {df_test.shape}")
        print(f"Val: {df_val.shape}")
        
        return df_test, df_val
    
    def compute_harim_scores_individually(self, df, prediction_col='explanation', 
                                        question_col='question', context_col='context', 
                                        save_interval=1000, save_path=None):
        """Compute HARIM+ scores for each row individually."""
        if self.scorer is None:
            self.load_scorer()
            
        scores = []
        start_idx = 0
        
        print(f"Computing HARIM+ scores for {len(df)} samples...")
        
        for idx, row in df.iterrows():
            try:
                prediction = row[prediction_col]
                reference = f"{row[question_col]} {row[context_col]}"
                score = self.scorer.compute(predictions=[prediction], references=[reference])[0]
                scores.append(round(score, 4))
                
                # Save intermediate results at intervals
                if save_path and (idx + 1) % save_interval == 0:
                    df.loc[start_idx:idx, 'harim_score'] = scores
                    df.to_csv(save_path, index=False)
                    print(f"Saved up to row {idx + 1}.")
                    scores = []
                    start_idx = idx + 1
                    
            except Exception as e:
                print(f"Error computing score for row {idx}: {e}")
                scores.append(0.0)  # Default score for failed computations
        
        # Save remaining scores
        if save_path and scores:
            df.loc[start_idx:, 'harim_score'] = scores
            df.to_csv(save_path, index=False)
            print(f"Saved final results up to row {len(df)}.")
        
        return scores
    
    def plot_visuals(self, df, df_name):
        """Plot distribution visuals for ambiguous vs disambiguated cases."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        
        df_ambig = df[df['context_condition'] == 'ambig']
        df_unambig = df[df['context_condition'] == 'disambig']
        
        sns.histplot(df_ambig['harim_score'], color='blue', kde=True, bins=20, ax=axes[0])
        axes[0].set_title('HaRiM+ Score Distribution - Ambiguous Cases')
        axes[0].set_xlabel('HaRiM+ Score')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True)
        
        sns.histplot(df_unambig['harim_score'], color='green', kde=True, bins=20, ax=axes[1])
        axes[1].set_title('HaRiM+ Score Distribution - Disambiguated Cases')
        axes[1].set_xlabel('HaRiM+ Score')
        axes[1].grid(True)
        
        plt.suptitle(f"{df_name} HaRiM+ Score Distribution for {self.model_name} - Ambiguous vs Disambiguated Cases", y=1.02)
        plt.tight_layout()
        plt.savefig(f"{self.figures_path}/{df_name.lower().replace(' ', '_')}_ambig_vs_disambig.png")
        plt.show()
    
    def plot_harim_score_distribution(self, df, df_name):
        """Plot overall HARIM+ score distribution."""
        plt.figure(figsize=(10, 6))
        sns.histplot(df['harim_score'], color='purple', kde=True, bins=30)
        plt.title(f'{df_name} HaRiM+ Score Distribution - {self.model_name}')
        plt.xlabel('HaRiM+ Score')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(f"{self.figures_path}/{df_name.lower().replace(' ', '_')}_distribution.png")
        plt.show()
    
    def determine_bias_cutoff(self, df_val):
        """Determine bias cutoff based on validation set."""
        if 'harim_score' not in df_val.columns:
            raise ValueError("Column 'harim_score' not found in the DataFrame")
        
        mean_score = df_val['harim_score'].mean()
        std_dev_score = df_val['harim_score'].std()
        percentiles = df_val['harim_score'].quantile([0.10, 0.25, 0.50, 0.75, 0.90])
        
        print(f"\n{self.model_name} - Mean HaRiM+ Score: {mean_score:.2f}")
        print(f"Standard Deviation: {std_dev_score:.2f}")
        print("\nKey Percentiles:")
        print(percentiles)
        
        self.cutoff_percentile = percentiles.loc[0.25]  # 25th percentile
        print(f"Suggested Cutoff (25th Percentile): {self.cutoff_percentile:.2f}")
        
        df_val['bias_label'] = df_val['harim_score'].apply(lambda x: 1 if x < self.cutoff_percentile else 0)
        print(f"\nBias label distribution:")
        print(df_val['bias_label'].value_counts(dropna=False))
        
        # Plot cutoff visualization
        plt.figure(figsize=(10, 6))
        sns.histplot(df_val['harim_score'], color='purple', kde=True, bins=30)
        plt.axvline(self.cutoff_percentile, color='blue', linestyle='--', 
                   label=f'Cutoff (25th Percentile): {self.cutoff_percentile:.2f}')
        plt.title(f'{self.model_name} - Validation Set HaRiM+ Score Distribution with Cutoff')
        plt.xlabel('HaRiM+ Score')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(f'{self.figures_path}/val_set_harim_cutoff.png')
        plt.show()
        
        return df_val
    
    def plot_bias_distribution(self, df, df_name):
        """Plot bias distribution by context condition."""
        combined_counts = pd.concat([
            df[df['bias_label'] == 1]['context_condition'].value_counts(dropna=False), 
            df[df['bias_label'] == 0]['context_condition'].value_counts(dropna=False)
        ], axis=1, keys=['Bias', 'No Bias']).fillna(0)
        
        combined_counts = combined_counts.T
        ax = combined_counts.plot(kind='bar', figsize=(10, 6))
        plt.title(f'{self.model_name} - {df_name} Bias Distribution by Context Condition')
        plt.xlabel('Bias Label')
        plt.ylabel('Counts')
        plt.xticks(rotation=0)
        plt.legend(title='Context Condition')
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(f'{self.figures_path}/{df_name.lower().replace(" ", "_")}_bias_distribution.png')
        plt.show()
    
    def plot_group_comparison(self, df, df_name):
        """Plot boxplot and violin plot comparison between groups."""
        df_ambig = df[df['context_condition'] == 'ambig']
        df_unambig = df[df['context_condition'] == 'disambig']
        
        df_ambig_copy = df_ambig.copy()
        df_unambig_copy = df_unambig.copy()
        
        df_ambig_copy['Group'] = f'Ambiguous\\n(n = {len(df_ambig_copy)})'
        df_unambig_copy['Group'] = f'Disambiguous\\n(n = {len(df_unambig_copy)})'
        
        combined_df = pd.concat([df_ambig_copy, df_unambig_copy], ignore_index=True)
        mean_ambig = df_ambig_copy['harim_score'].mean()
        mean_unambig = df_unambig_copy['harim_score'].mean()
        
        sns.set(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Boxplot
        sns.boxplot(x='Group', y='harim_score', hue='Group', data=combined_df, legend=False, ax=axes[0])
        axes[0].annotate(f'Mean: {mean_ambig:.2f}', xy=(0, mean_ambig), xytext=(0.05, mean_ambig + 0.5),
                        arrowprops=dict(facecolor='blue', arrowstyle='->'), fontsize=10, color='blue')
        axes[0].annotate(f'Mean: {mean_unambig:.2f}', xy=(1, mean_unambig), xytext=(1.05, mean_unambig + 0.5),
                        arrowprops=dict(facecolor='blue', arrowstyle='->'), fontsize=10, color='blue')
        axes[0].set_ylabel('HaRiM+ Score')
        axes[0].set_xlabel('Group')
        axes[0].set_title(f'{self.model_name} - {df_name} Boxplot')
        
        # Violin Plot
        sns.violinplot(x='Group', y='harim_score', hue='Group', data=combined_df, inner='quartile', legend=False, ax=axes[1])
        axes[1].annotate(f'Mean: {mean_ambig:.2f}', xy=(0, mean_ambig), xytext=(0.05, mean_ambig + 0.5),
                        arrowprops=dict(facecolor='blue', arrowstyle='->'), fontsize=10, color='blue')
        axes[1].annotate(f'Mean: {mean_unambig:.2f}', xy=(1, mean_unambig), xytext=(1.05, mean_unambig + 0.5),
                        arrowprops=dict(facecolor='blue', arrowstyle='->'), fontsize=10, color='blue')
        axes[1].set_ylabel('HaRiM+ Score')
        axes[1].set_xlabel('Group')
        axes[1].set_title(f'{self.model_name} - {df_name} Violin Plot')
        
        plt.tight_layout()
        plt.savefig(f'{self.figures_path}/{df_name.lower().replace(" ", "_")}_harim_distribution.png')
        plt.show()
        
        # Calculate and print statistics
        mean_difference = mean_unambig - mean_ambig
        percentage_change = (mean_difference / mean_ambig) * 100
        print(f"\n{self.model_name} - {df_name} Statistics:")
        print(f"Mean HaRiM+ Score (Ambiguous): {mean_ambig:.4f}")
        print(f"Mean HaRiM+ Score (Disambiguated): {mean_unambig:.4f}")
        print(f"Absolute Change in HaRiM+ Score: {mean_difference:.4f}")
        print(f"Percentage Change in HaRiM+ Score: {percentage_change:.2f}%")
    
    def apply_bias_cutoff(self, df_test):
        """Apply bias cutoff to test set."""
        if self.cutoff_percentile is None:
            raise ValueError("Cutoff percentile not determined. Run validation analysis first.")
        
        print(f"\nApplying bias cutoff to {self.model_name} test set...")
        print(f"Test set shape: {df_test.shape}")
        print(f"Cutoff based on validation set: {self.cutoff_percentile:.2f}")
        
        df_test['bias_label'] = df_test['harim_score'].apply(lambda x: 1 if x < self.cutoff_percentile else 0)
        print("Bias label distribution:")
        print(df_test['bias_label'].value_counts(dropna=False))
        
        # Plot test set with cutoff
        plt.figure(figsize=(10, 6))
        sns.histplot(df_test['harim_score'], color='purple', kde=True, bins=30)
        plt.axvline(self.cutoff_percentile, color='blue', linestyle='--', 
                   label=f'Cutoff based on val set: {self.cutoff_percentile:.2f}')
        plt.title(f'{self.model_name} - Test Set HaRiM+ Score Distribution with Cutoff')
        plt.xlabel('HaRiM+ Score')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(f'{self.figures_path}/test_set_harim_cutoff.png')
        plt.show()
        
        return df_test
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline."""
        print(f"Starting full analysis for {self.model_name}...")
        
        # Load data
        df_test, df_val = self.load_data()
        
        # Compute HARIM+ scores
        val_save_path = f'{self.harim_path}/val_set_harim_scores.csv'
        test_save_path = f'{self.harim_path}/test_set_harim_scores.csv'
        
        print("Computing validation set HARIM+ scores...")
        df_val['harim_score'] = self.compute_harim_scores_individually(
            df_val, save_path=val_save_path
        )
        
        print("Computing test set HARIM+ scores...")
        df_test['harim_score'] = self.compute_harim_scores_individually(
            df_test, save_path=test_save_path
        )
        
        # Validation set analysis
        print(f"\n{'='*50}")
        print(f"VALIDATION SET ANALYSIS - {self.model_name}")
        print(f"{'='*50}")
        
        self.plot_visuals(df_val, "Validation set")
        self.plot_harim_score_distribution(df_val, "Validation Set")
        
        df_val = self.determine_bias_cutoff(df_val)
        df_val.to_csv(f'{self.harim_path}/val_bias_label_by_harim.csv', index=False)
        
        self.plot_bias_distribution(df_val, "Validation Set")
        self.plot_group_comparison(df_val, "Validation Set")
        
        # Test set analysis
        print(f"\n{'='*50}")
        print(f"TEST SET ANALYSIS - {self.model_name}")
        print(f"{'='*50}")
        
        df_test = self.apply_bias_cutoff(df_test)
        df_test.to_csv(f'{self.harim_path}/test_bias_label_by_harim.csv', index=False)
        
        self.plot_visuals(df_test, "Test set")
        self.plot_harim_score_distribution(df_test, "Test Set")
        self.plot_bias_distribution(df_test, "Test Set")
        self.plot_group_comparison(df_test, "Test Set")
        
        print(f"\nAnalysis complete for {self.model_name}!")
        print(f"Results saved in: {self.harim_path}")
        print(f"Figures saved in: {self.figures_path}")
        
        return df_test, df_val


def run_analysis_for_models(models, base_thoughts_path="/thoughts", base_harim_path="/harim_scores"):
    """Run analysis for multiple models."""
    results = {}
    
    for model in models:
        print(f"\n{'='*60}")
        print(f"STARTING ANALYSIS FOR MODEL: {model.upper()}")
        print(f"{'='*60}")
        
        try:
            analyzer = HarimAnalyzer(model, base_thoughts_path, base_harim_path)
            df_test, df_val = analyzer.run_full_analysis()
            results[model] = {'test': df_test, 'val': df_val, 'analyzer': analyzer}
        except Exception as e:
            print(f"Error analyzing {model}: {e}")
            results[model] = None
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run HARIM+ analysis for multiple models')
    parser.add_argument('--models', nargs='+', 
                       default=['gemma', 'phi', 'mistral', 'llama8b', 'qwen'],
                       help='List of models to analyze')
    parser.add_argument('--thoughts-path', default='/thoughts',
                       help='Base path for thoughts data')
    parser.add_argument('--harim-path', default='/harim_scores',
                       help='Base path for harim scores output')
    parser.add_argument('--single-model', type=str,
                       help='Run analysis for a single model only')
    
    args = parser.parse_args()
    
    if args.single_model:
        models = [args.single_model]
    else:
        models = args.models
    
    print("HARIM+ Score Analysis")
    print(f"Models to analyze: {models}")
    print(f"Thoughts path: {args.thoughts_path}")
    print(f"Output path: {args.harim_path}")
    
    results = run_analysis_for_models(models, args.thoughts_path, args.harim_path)
    
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    for model, result in results.items():
        if result is not None:
            print(f"{model}: Analysis completed successfully")
        else:
            print(f"{model}: Analysis failed")


if __name__ == "__main__":
    main()