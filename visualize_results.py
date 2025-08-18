#!/usr/bin/env python3
"""
Advanced EEG Analysis Results Visualization Suite
=================================================

A comprehensive visualization toolkit for EEG consciousness analysis results.
Features modern styling, statistical analysis, and interactive visualizations.

Key Features:
- Advanced statistical visualizations (violin plots, ridge plots, heatmaps)
- Statistical significance testing and effect size calculations
- Interactive plots with detailed tooltips
- Modern, publication-ready styling
- Comprehensive data validation and error handling
- Modular design for easy customization
- Automated report generation

Author: EEG Analysis Team
Version: 2.0
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import argparse
import warnings
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Configuration ---

@dataclass
class PlotConfig:
    """Configuration settings for plot styling and behavior."""
    
    # Figure dimensions
    fig_width: int = 16
    fig_height: int = 10
    dpi: int = 300
    
    # Color palettes
    primary_palette: str = 'husl'
    secondary_palette: str = 'viridis'
    categorical_palette: str = 'Set2'
    
    # Font settings
    title_size: int = 20
    label_size: int = 14
    tick_size: int = 12
    legend_size: int = 12
    
    # Style settings
    style: str = 'whitegrid'
    context: str = 'talk'
    
    # Statistical settings
    confidence_level: float = 0.95
    alpha: float = 0.05
    
    # Interactive settings
    save_interactive: bool = True
    save_static: bool = True

# Global configuration
config = PlotConfig()

# Set matplotlib and seaborn styling
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    plt.style.use('seaborn')
sns.set_context(config.context)
sns.set_style(config.style)
plt.rcParams.update({
    'figure.figsize': (config.fig_width, config.fig_height),
    'font.size': config.label_size,
    'axes.titlesize': config.title_size,
    'axes.labelsize': config.label_size,
    'xtick.labelsize': config.tick_size,
    'ytick.labelsize': config.tick_size,
    'legend.fontsize': config.legend_size,
    'figure.dpi': config.dpi,
    'savefig.dpi': config.dpi,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'axes.axisbelow': True,
    'grid.alpha': 0.3
})

# --- Data Processing Classes ---

class ResultsProcessor:
    """Advanced data processing and validation for EEG analysis results."""
    
    def __init__(self):
        self.data_quality_report = {}
        
    def parse_filename(self, filepath: Path) -> Dict[str, str]:
        """Enhanced filename parsing with better error handling."""
        filename = filepath.name
        
        patterns = [
            # Pattern 1: 'sub-1010_mib-ksg_results.json'
            (r'(sub-\d+)_(\w+)-(\w+)_results\.json', 
             lambda m: {'subject': m.group(1), 'metric': m.group(2).upper(), 'estimator': m.group(3).upper()}),
            
            # Pattern 2: 'sub-1010_results.json'
            (r'(sub-\d+)_results\.json', 
             lambda m: self._infer_from_path(filepath, m.group(1))),
            
            # Pattern 3: Aggregate files
            (r'.*spectral_summary\.csv', 
             lambda m: self._parse_aggregate_file(filename))
        ]
        
        for pattern, parser in patterns:
            match = re.match(pattern, filename)
            if match:
                return parser(match)
        
        return {'subject': 'unknown', 'metric': 'unknown', 'estimator': 'unknown'}

    def _infer_from_path(self, filepath: Path, subject: str) -> Dict[str, str]:
        """Infer metric and estimator from file path."""
        path_str = str(filepath).lower()
        
        # Determine metric
        if 'mib' in path_str:
            metric = 'MIB'
        elif 'complexity' in path_str:
            metric = 'COMPLEXITY'
        else:
            metric = 'UNKNOWN'
        
        # Determine estimator
        if 'ksg' in path_str:
            estimator = 'KSG'
        elif 'binning' in path_str:
            estimator = 'BINNING'
        elif 'gaussian' in path_str:
            estimator = 'GAUSSIAN'
        else:
            estimator = 'UNKNOWN'
        
        return {'subject': subject, 'metric': metric, 'estimator': estimator}

    def _parse_aggregate_file(self, filename: str) -> Dict[str, str]:
        """Parse aggregate summary files."""
        estimator = 'UNKNOWN'
        if 'ksg' in filename.lower():
            estimator = 'KSG'
        elif 'binning' in filename.lower():
            estimator = 'BINNING'
        elif 'gaussian' in filename.lower():
            estimator = 'GAUSSIAN'
        
        return {'subject': 'aggregate', 'metric': 'COMPLEXITY', 'estimator': estimator}
    
    def load_and_validate_results(self, base_dirs: List[str]) -> pd.DataFrame:
        """Load results with comprehensive validation and quality assessment."""
        all_data = []
        file_count = 0
        error_count = 0
        
        for base_dir in base_dirs:
            if not os.path.isdir(base_dir):
                continue
                
            for root, _, files in os.walk(base_dir):
                for file in files:
                    if file.endswith('.json'):
                        file_count += 1
                        filepath = Path(root) / file
                        
                        try:
                            data = self._load_json_file(filepath)
                            if data:
                                all_data.extend(data)
                        except Exception as e:
                            error_count += 1
                            print(f"Error processing {filepath}: {e}")
        
        # Quality assessment
        self.data_quality_report = {
            'total_files': file_count,
            'errors': error_count,
            'success_rate': (file_count - error_count) / file_count if file_count > 0 else 0,
            'total_records': len(all_data)
        }
        
        if not all_data:
            print("Warning: No valid result files found.")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        return self._validate_and_clean_data(df)
    
    def _load_json_file(self, filepath: Path) -> List[Dict[str, Any]]:
        """Load and process a single JSON file."""
        file_info = self.parse_filename(filepath)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        records = []
        for condition, condition_data in data.items():
            for band, band_data in condition_data.items():
                # Support traditional metrics
                values = band_data.get('metric_values', [])
                if isinstance(values, list) and values:
                    for value in values:
                        if isinstance(value, (int, float)) and not np.isnan(value):
                            records.append({
                                'subject': file_info['subject'],
                                'metric': file_info['metric'],
                                'estimator': file_info['estimator'],
                                'condition': condition,
                                'band': band,
                                'value': float(value)
                            })

                # Support complexes: list per-epoch of (subset, phi_value)
                complexes_epochs = band_data.get('complexes_per_epoch', [])
                selected_channels = band_data.get('selected_channels', [])
                if isinstance(complexes_epochs, list) and complexes_epochs:
                    for epoch_idx, epoch_complexes in enumerate(complexes_epochs):
                        if isinstance(epoch_complexes, list) and epoch_complexes:
                            # Find the complex with maximum Φ in this epoch (the main complex)
                            max_phi_complex = None
                            max_phi_val = -np.inf
                            
                            for complex_entry in epoch_complexes:
                                try:
                                    phi_val = float(complex_entry.get('phi_value', np.nan))
                                    if not np.isnan(phi_val) and phi_val > max_phi_val:
                                        max_phi_val = phi_val
                                        max_phi_complex = complex_entry
                                except Exception:
                                    continue
                            
                            # Only record the main complex (highest Φ) from this epoch
                            if max_phi_complex is not None:
                                try:
                                    phi_val = float(max_phi_complex.get('phi_value', np.nan))
                                    subset_indices = max_phi_complex.get('subset', [])
                                    # Map indices to channel names if available
                                    if selected_channels and all(isinstance(i, int) and 0 <= i < len(selected_channels) for i in subset_indices):
                                        subset_channels = [selected_channels[i] for i in subset_indices]
                                    else:
                                        subset_channels = [f"Ch{i}" for i in subset_indices]
                                except Exception:
                                    continue
                                
                                if not np.isnan(phi_val):
                                    records.append({
                                        'subject': file_info['subject'],
                                        'metric': 'COMPLEXES',
                                        'estimator': file_info['estimator'],
                                        'condition': condition,
                                        'band': band,
                                        'value': phi_val,  # This is now the MAX Φ per epoch
                                        'subset_indices': subset_indices,
                                        'subset_channels': subset_channels,
                                        'subset': ','.join(subset_channels),
                                        'subset_size': len(subset_indices),
                                        'epoch_idx': epoch_idx,
                                        'is_main_complex': True  # Flag to indicate this is the dominant complex
                                    })
        
        return records
    
    def _validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the loaded data."""
        if df.empty:
            return df
        
        # Remove outliers using IQR method
        df_cleaned = df.copy()
        for metric in df['metric'].unique():
            for condition in df['condition'].unique():
                mask = (df['metric'] == metric) & (df['condition'] == condition)
                values = df.loc[mask, 'value']
                
                if len(values) > 4:  # Need at least 5 values for meaningful outlier detection
                    Q1 = values.quantile(0.25)
                    Q3 = values.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outlier_mask = (values < lower_bound) | (values > upper_bound)
                    df_cleaned = df_cleaned[~(mask & outlier_mask)]
        
        # Add derived columns
        df_cleaned['metric_estimator'] = df_cleaned['metric'] + '-' + df_cleaned['estimator']
        df_cleaned['log_value'] = np.log1p(df_cleaned['value'])  # log(1+x) for stability
        
        return df_cleaned

# --- Advanced Plotting Functions ---

class AdvancedPlotter:
    """Advanced plotting class with modern styling and statistical analysis."""
    
    def __init__(self, config: PlotConfig):
        self.config = config
        self.stats_results = {}
    
    def create_comprehensive_overview(self, df: pd.DataFrame, output_dir: str):
        """Create a comprehensive overview dashboard."""
        if df.empty:
            return
        
        # Check if we have complexes data
        is_complexes = 'COMPLEXES' in df['metric'].unique()
        
        if is_complexes:
            self._create_complexes_overview(df, output_dir)
        else:
            self._create_traditional_overview(df, output_dir)

    def make_balanced(self, df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
        """Downsample each condition to equal epochs per subject (and band if present)."""
        if df.empty or 'condition' not in df.columns or 'subject' not in df.columns:
            return df

        rng = np.random.default_rng(random_state)
        group_keys = ['subject'] + (['band'] if 'band' in df.columns else [])

        balanced_records = []
        # Determine min count per group across conditions
        counts = df.groupby(group_keys + ['condition']).size().reset_index(name='n')
        min_counts = counts.groupby(group_keys)['n'].min().reset_index().rename(columns={'n': 'min_n'})
        # Merge back to know the target per group
        merged = counts.merge(min_counts, on=group_keys, how='left')

        for _, row in merged.iterrows():
            group_filter = (df['subject'] == row['subject'])
            if 'band' in df.columns:
                group_filter &= (df['band'] == row['band'])
            group_filter &= (df['condition'] == row['condition'])

            group_df = df[group_filter]
            target_n = int(row['min_n'])
            if len(group_df) == 0 or target_n == 0:
                continue
            if len(group_df) <= target_n:
                balanced_records.append(group_df)
            else:
                # Random sample without replacement
                sampled_idx = rng.choice(group_df.index.values, size=target_n, replace=False)
                balanced_records.append(group_df.loc[sampled_idx])

        return pd.concat(balanced_records, axis=0).reset_index(drop=True) if balanced_records else df
    
    def _create_complexes_overview(self, df: pd.DataFrame, output_dir: str):
        """Create specialized overview for complexes analysis."""
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Phi distribution by condition
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_phi_distributions(df, ax1)
        
        # 2. Complex size distribution
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_complex_size_distribution(df, ax2)
        
        # 3. Channel participation frequency
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_channel_participation(df, ax3)
        
        # 4. Size vs Phi relationship
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_size_phi_relationship(df, ax4)
        
        # 5. Spectral complexes analysis
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_spectral_complexes(df, ax5)
        
        plt.suptitle('EEG Complexes Analysis: Comprehensive Overview', 
                    fontsize=28, y=0.98, fontweight='bold')
        
        save_path = os.path.join(output_dir, 'comprehensive_overview.png')
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved complexes overview: {save_path}")
    
    def _create_traditional_overview(self, df: pd.DataFrame, output_dir: str):
        """Create traditional overview for complexity/MIB metrics."""
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Distribution overview
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_distribution_overview(df, ax1)
        
        # 2. Statistical summary
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_statistical_summary(df, ax2)
        
        # 3. Correlation heatmap
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_correlation_heatmap(df, ax3)
        
        # 4. Effect sizes
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_effect_sizes(df, ax4)
        
        # 5. Spectral analysis
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_spectral_analysis(df, ax5)
        
        plt.suptitle('EEG Analysis Results: Comprehensive Overview', 
                    fontsize=28, y=0.98, fontweight='bold')
        
        save_path = os.path.join(output_dir, 'comprehensive_overview.png')
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved comprehensive overview: {save_path}")

    def _plot_distribution_overview(self, df: pd.DataFrame, ax):
        """Plot distribution overview with violin plots."""
        sns.violinplot(data=df, x='condition', y='value', hue='metric_estimator', 
                      ax=ax, palette=self.config.primary_palette, inner='box')
        ax.set_title('Distribution Overview by Condition', fontsize=16, fontweight='bold')
        ax.set_ylabel('Metric Value (bits)')
        ax.set_xlabel('Condition')
        
        # Add statistical annotations
        self._add_significance_annotations(df, ax)
    
    def _plot_statistical_summary(self, df: pd.DataFrame, ax):
        """Plot statistical summary with confidence intervals."""
        summary_stats = df.groupby(['condition', 'metric_estimator'])['value'].agg([
            'mean', 'std', 'count', 'sem'
        ]).reset_index()
        
        # Calculate confidence intervals
        summary_stats['ci_lower'] = summary_stats['mean'] - 1.96 * summary_stats['sem']
        summary_stats['ci_upper'] = summary_stats['mean'] + 1.96 * summary_stats['sem']
        
        # Create bar plot with error bars
        try:
            # Try with newer seaborn syntax
            sns.barplot(data=df, x='condition', y='value', hue='metric_estimator',
                       ax=ax, palette=self.config.secondary_palette, 
                       errorbar='ci', capsize=0.05, errwidth=2)
        except:
            # Fallback to older seaborn syntax
            sns.barplot(data=df, x='condition', y='value', hue='metric_estimator',
                       ax=ax, palette=self.config.secondary_palette)
        
        ax.set_title('Statistical Summary with 95% CI', fontsize=16, fontweight='bold')
        ax.set_ylabel('Mean Value (bits)')
        ax.set_xlabel('Condition')
    
    def _plot_correlation_heatmap(self, df: pd.DataFrame, ax):
        """Create advanced correlation heatmap."""
        # Pivot data for correlation analysis
        pivot_df = df.pivot_table(
            index=['subject', 'condition', 'band'], 
            columns='metric_estimator', 
            values='value'
        ).reset_index()
        
        # Calculate correlation matrix
        numeric_cols = pivot_df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            ax.text(0.5, 0.5, 'Need at least 2 numeric columns\nfor correlation analysis', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=14, alpha=0.7)
            ax.set_title('Metric Correlation Matrix', fontsize=16, fontweight='bold')
            return

        corr_matrix = pivot_df[numeric_cols].corr()
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)
        
        ax.set_title('Metric Correlation Matrix', fontsize=16, fontweight='bold')
    
    def _plot_effect_sizes(self, df: pd.DataFrame, ax):
        """Calculate and plot effect sizes between conditions."""
        effect_sizes = []
        
        for metric in df['metric_estimator'].unique():
            metric_data = df[df['metric_estimator'] == metric]
            conditions = metric_data['condition'].unique()
            
            if len(conditions) >= 2:
                for i in range(len(conditions)):
                    for j in range(i+1, len(conditions)):
                        cond1_data = metric_data[metric_data['condition'] == conditions[i]]['value']
                        cond2_data = metric_data[metric_data['condition'] == conditions[j]]['value']
                        
                        if len(cond1_data) > 0 and len(cond2_data) > 0:
                            # Cohen's d
                            pooled_std = np.sqrt((cond1_data.var() + cond2_data.var()) / 2)
                            cohens_d = (cond1_data.mean() - cond2_data.mean()) / pooled_std
                            
                            effect_sizes.append({
                                'metric': metric,
                                'comparison': f"{conditions[i]} vs {conditions[j]}",
                                'effect_size': cohens_d,
                                'magnitude': self._interpret_effect_size(abs(cohens_d))
                            })
        
        if effect_sizes:
            effect_df = pd.DataFrame(effect_sizes)
            sns.barplot(data=effect_df, x='comparison', y='effect_size', hue='metric',
                       ax=ax, palette=self.config.categorical_palette)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.set_title('Effect Sizes (Cohen\'s d)', fontsize=16, fontweight='bold')
            ax.set_ylabel('Effect Size')
            ax.set_xlabel('Comparison')
            plt.setp(ax.get_xticklabels(), rotation=45)
        else:
            ax.text(0.5, 0.5, 'No effect sizes calculated\n(need multiple conditions)', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=14, alpha=0.7)
            ax.set_title('Effect Sizes (Cohen\'s d)', fontsize=16, fontweight='bold')
    
    def _plot_spectral_analysis(self, df: pd.DataFrame, ax):
        """Advanced spectral analysis visualization."""
        spectral_df = df[df['band'] != 'broadband']
        
        if spectral_df.empty:
            ax.text(0.5, 0.5, 'No spectral data available', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=14, alpha=0.7)
            ax.set_title('Spectral Analysis', fontsize=16, fontweight='bold')
            return

        # Create spectral profile with confidence intervals
        sns.lineplot(data=spectral_df, x='band', y='value', 
                    hue='metric_estimator', style='condition',
                    marker='o', markersize=8, linewidth=2.5, ax=ax,
                    palette=self.config.primary_palette)
        
        ax.set_title('Spectral Profile Analysis', fontsize=16, fontweight='bold')
        ax.set_ylabel('Metric Value (bits)')
        ax.set_xlabel('Frequency Band')
        ax.grid(True, alpha=0.3)
        
        # Enhance legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    def _add_significance_annotations(self, df: pd.DataFrame, ax):
        """Add statistical significance annotations to plots."""
        # This is a simplified version - in practice, you'd want more sophisticated testing
        conditions = df['condition'].unique()
        if len(conditions) >= 2:
            for metric in df['metric_estimator'].unique():
                metric_data = df[df['metric_estimator'] == metric]
                
                # Perform Mann-Whitney U test between first two conditions
                if len(conditions) >= 2:
                    group1 = metric_data[metric_data['condition'] == conditions[0]]['value']
                    group2 = metric_data[metric_data['condition'] == conditions[1]]['value']
                    
                    if len(group1) > 0 and len(group2) > 0:
                        _, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
                        if p_value < 0.05:
                            # Add significance annotation
                            y_max = max(ax.get_ylim()[1], group1.max(), group2.max())
                            ax.annotate('*' if p_value < 0.05 else 'ns',
                                      xy=(0.5, y_max * 1.02), xytext=(0.5, y_max * 1.05),
                                      ha='center', va='bottom', fontsize=12,
                                      arrowprops=dict(arrowstyle='-', color='black'))
    
    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        if d < 0.2:
            return 'Small'
        elif d < 0.5:
            return 'Medium'
        elif d < 0.8:
            return 'Large'
        else:
            return 'Very Large'
    
    def create_advanced_distributions(self, df: pd.DataFrame, output_dir: str):
        """Create advanced distribution plots."""
        if df.empty:
            return
        
        # Ridge plot
        self._create_ridge_plot(df, output_dir)
        
        # Raincloud plot
        self._create_raincloud_plot(df, output_dir)
        
        # Interactive violin plot
        self._create_interactive_distributions(df, output_dir)
    
    def _create_ridge_plot(self, df: pd.DataFrame, output_dir: str):
        """Create a ridge plot for distribution comparison."""
        fig, axes = plt.subplots(
            len(df['metric_estimator'].unique()), 1, 
            figsize=(14, len(df['metric_estimator'].unique()) * 3),
            sharex=True
        )
        
        if len(df['metric_estimator'].unique()) == 1:
            axes = [axes]
        
        colors = sns.color_palette(self.config.primary_palette, 
                                 len(df['condition'].unique()))
        
        for i, metric in enumerate(df['metric_estimator'].unique()):
            ax = axes[i]
            metric_data = df[df['metric_estimator'] == metric]
            
            for j, condition in enumerate(df['condition'].unique()):
                condition_data = metric_data[metric_data['condition'] == condition]
                if not condition_data.empty:
                    # Create density plot
                    sns.kdeplot(data=condition_data, x='value', ax=ax, 
                              fill=True, alpha=0.7, color=colors[j],
                              label=condition)
            
            ax.set_title(f'{metric} Distribution', fontweight='bold')
            ax.set_ylabel('Density')
            ax.legend()
        
        plt.suptitle('Ridge Plot: Distribution Comparison', fontsize=20, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, 'ridge_distribution_plot.png')
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved ridge plot: {save_path}")

    def _create_raincloud_plot(self, df: pd.DataFrame, output_dir: str):
        """Create raincloud plots combining distributions and raw data."""
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Create a combined plot with violin + strip + box
        sns.violinplot(data=df, x='condition', y='value', hue='metric_estimator',
                      ax=ax, palette=self.config.primary_palette, inner=None, alpha=0.7)
        
        # Add strip plot for individual points
        sns.stripplot(data=df, x='condition', y='value', hue='metric_estimator',
                     ax=ax, dodge=True, size=3, alpha=0.8, jitter=True)
        
        # Add box plot for summary statistics
        sns.boxplot(data=df, x='condition', y='value', hue='metric_estimator',
                   ax=ax, palette=self.config.primary_palette, width=0.3,
                   boxprops=dict(alpha=0.3), showfliers=False)
        
        ax.set_title('Raincloud Plot: Comprehensive Distribution View', 
                    fontsize=18, fontweight='bold')
        ax.set_ylabel('Metric Value (bits)')
        ax.set_xlabel('Condition')
        
        # Clean up legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:len(df['metric_estimator'].unique())], 
                 labels[:len(df['metric_estimator'].unique())],
                 title='Metric-Estimator', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, 'raincloud_plot.png')
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved raincloud plot: {save_path}")
    
    def _create_interactive_distributions(self, df: pd.DataFrame, output_dir: str):
        """Create interactive distribution plots using Plotly."""
        if not self.config.save_interactive:
            return

        fig = px.violin(df, x='condition', y='value', color='metric_estimator',
                       box=True, points='all', hover_data=['subject', 'band'],
                       title='Interactive Distribution Analysis')
        
        fig.update_layout(
            title_font_size=20,
            xaxis_title='Condition',
            yaxis_title='Metric Value (bits)',
            legend_title='Metric-Estimator',
            height=600,
            width=1000
        )
        
        save_path = os.path.join(output_dir, 'interactive_distributions.html')
        fig.write_html(save_path)
        print(f"✓ Saved interactive distributions: {save_path}")
    
    def create_dimensionality_reduction_plots(self, df: pd.DataFrame, output_dir: str):
        """Create PCA and t-SNE plots for pattern exploration."""
        if df.empty or len(df['metric_estimator'].unique()) < 2:
            return
        
        # Prepare data for dimensionality reduction
        pivot_df = df.pivot_table(
            index=['subject', 'condition'], 
            columns='metric_estimator', 
            values='value',
            aggfunc='mean'
        ).reset_index()
        
        # Remove rows with NaN values
        pivot_df = pivot_df.dropna()
        
        if pivot_df.empty or len(pivot_df) < 4:
            return
        
        # Prepare features
        feature_cols = [col for col in pivot_df.columns if col not in ['subject', 'condition']]
        X = pivot_df[feature_cols].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # t-SNE
        if len(pivot_df) >= 4:  # t-SNE requires at least 4 samples
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(pivot_df)-1))
            X_tsne = tsne.fit_transform(X_scaled)
        else:
            X_tsne = X_pca  # Fall back to PCA if not enough samples
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # PCA plot
        scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], 
                            c=pivot_df['condition'].astype('category').cat.codes,
                            cmap=self.config.primary_palette, s=100, alpha=0.7)
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax1.set_title('PCA: Pattern Exploration', fontsize=16, fontweight='bold')
        
        # Add condition labels
        for i, condition in enumerate(pivot_df['condition'].unique()):
            mask = pivot_df['condition'] == condition
            ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       label=condition, s=100, alpha=0.7)
        ax1.legend()
        
        # t-SNE plot
        for i, condition in enumerate(pivot_df['condition'].unique()):
            mask = pivot_df['condition'] == condition
            ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                       label=condition, s=100, alpha=0.7)
        ax2.set_xlabel('t-SNE Component 1')
        ax2.set_ylabel('t-SNE Component 2')
        ax2.set_title('t-SNE: Non-linear Pattern Exploration', fontsize=16, fontweight='bold')
        ax2.legend()
        
        plt.tight_layout()
    
        save_path = os.path.join(output_dir, 'dimensionality_reduction.png')
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved dimensionality reduction plots: {save_path}")
    
    def create_statistical_report(self, df: pd.DataFrame, output_dir: str):
        """Generate a comprehensive statistical report."""
        if df.empty:
            return
        
        report = []
        report.append("# EEG Analysis Statistical Report\n")
        report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Data summary
        report.append("## Data Summary\n")
        report.append(f"- Total subjects: {df['subject'].nunique()}\n")
        report.append(f"- Total conditions: {df['condition'].nunique()}\n")
        report.append(f"- Total metrics: {df['metric_estimator'].nunique()}\n")
        report.append(f"- Total data points: {len(df)}\n\n")
        
        # Descriptive statistics
        report.append("## Descriptive Statistics\n")
        desc_stats = df.groupby(['condition', 'metric_estimator'])['value'].describe()
        report.append(desc_stats.to_string())
        report.append("\n\n")
        
        # Statistical tests
        report.append("## Statistical Tests\n")
        
        # Perform tests between conditions for each metric
        for metric in df['metric_estimator'].unique():
            metric_data = df[df['metric_estimator'] == metric]
            conditions = metric_data['condition'].unique()
            
            if len(conditions) >= 2:
                report.append(f"### {metric}\n")
                
                for i in range(len(conditions)):
                    for j in range(i+1, len(conditions)):
                        group1 = metric_data[metric_data['condition'] == conditions[i]]['value']
                        group2 = metric_data[metric_data['condition'] == conditions[j]]['value']
                        
                        if len(group1) > 0 and len(group2) > 0:
                            # Mann-Whitney U test
                            statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
                            
                            # Effect size (Cohen's d)
                            pooled_std = np.sqrt((group1.var() + group2.var()) / 2)
                            cohens_d = (group1.mean() - group2.mean()) / pooled_std
                            
                            report.append(f"**{conditions[i]} vs {conditions[j]}:**\n")
                            report.append(f"- Mann-Whitney U: {statistic:.3f}, p = {p_value:.6f}\n")
                            report.append(f"- Effect size (Cohen's d): {cohens_d:.3f}\n")
                            report.append(f"- Interpretation: {self._interpret_effect_size(abs(cohens_d))}\n\n")
        
        # Save report
        save_path = os.path.join(output_dir, 'statistical_report.md')
        with open(save_path, 'w') as f:
            f.writelines(report)
        
        print(f"✓ Saved statistical report: {save_path}")

    # --- Complexes-Specific Plotting Functions ---
    
    def _plot_phi_distributions(self, df: pd.DataFrame, ax):
        """Plot Φ (phi) value distributions by condition."""
        sns.violinplot(data=df, x='condition', y='value', ax=ax, 
                      palette=self.config.primary_palette, inner='box')
        ax.set_title('Main Complex Φ Distribution by Condition', fontsize=16, fontweight='bold')
        ax.set_ylabel('Max Φ per Epoch (bits)')
        ax.set_xlabel('Condition')
        ax.grid(True, alpha=0.3)

    def _plot_complex_size_distribution(self, df: pd.DataFrame, ax):
        """Plot distribution of complex sizes."""
        sns.histplot(data=df, x='subset_size', hue='condition', ax=ax,
                    palette=self.config.primary_palette, bins=range(2, 9), 
                    alpha=0.7, kde=True)
        ax.set_title('Main Complex Size Distribution', fontsize=16, fontweight='bold')
        ax.set_xlabel('Main Complex Size (Number of Channels)')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)

    def _plot_channel_participation(self, df: pd.DataFrame, ax):
        """Plot frequency of each channel's participation in complexes."""
        if 'subset_channels' not in df.columns:
            ax.text(0.5, 0.5, 'Channel information not available', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=14, alpha=0.7)
            ax.set_title('Channel Participation Frequency', fontsize=16, fontweight='bold')
            return
        
        # Count channel participation
        channel_counts = {}
        for _, row in df.iterrows():
            for channel in row['subset_channels']:
                condition = row['condition']
                if condition not in channel_counts:
                    channel_counts[condition] = {}
                if channel not in channel_counts[condition]:
                    channel_counts[condition][channel] = 0
                channel_counts[condition][channel] += 1
        
        # Convert to DataFrame for plotting
        plot_data = []
        for condition, channels in channel_counts.items():
            for channel, count in channels.items():
                plot_data.append({'condition': condition, 'channel': channel, 'count': count})
        
        if plot_data:
            plot_df = pd.DataFrame(plot_data)
            sns.barplot(data=plot_df, x='channel', y='count', hue='condition', ax=ax,
                       palette=self.config.primary_palette)
            ax.set_title('Main Complex Channel Participation', fontsize=16, fontweight='bold')
            ax.set_xlabel('EEG Channel')
            ax.set_ylabel('Times in Main Complex')
            plt.setp(ax.get_xticklabels(), rotation=45)
        else:
            ax.text(0.5, 0.5, 'No channel data available', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=14, alpha=0.7)

    def _plot_size_phi_relationship(self, df: pd.DataFrame, ax):
        """Plot relationship between complex size and Φ value."""
        sns.scatterplot(data=df, x='subset_size', y='value', hue='condition', 
                       ax=ax, palette=self.config.primary_palette, alpha=0.6, s=50)
        
        # Add trend lines
        for condition in df['condition'].unique():
            condition_data = df[df['condition'] == condition]
            if len(condition_data) > 1:
                sns.regplot(data=condition_data, x='subset_size', y='value', ax=ax,
                           scatter=False, lowess=True, line_kws={'linewidth': 2})
        
        ax.set_title('Complex Size vs Φ Value', fontsize=16, fontweight='bold')
        ax.set_xlabel('Complex Size (Number of Channels)')
        ax.set_ylabel('Φ Value (bits)')
        ax.grid(True, alpha=0.3)

    def _plot_spectral_complexes(self, df: pd.DataFrame, ax):
        """Plot spectral analysis of complexes."""
        spectral_df = df[df['band'] != 'broadband'] if 'band' in df.columns else df
        
        if spectral_df.empty:
            ax.text(0.5, 0.5, 'No spectral data available', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=14, alpha=0.7)
            ax.set_title('Spectral Complexes Analysis', fontsize=16, fontweight='bold')
            return

        # Plot mean Φ by frequency band and condition
        spectral_summary = spectral_df.groupby(['band', 'condition'])['value'].agg(['mean', 'sem']).reset_index()
        
        sns.lineplot(data=spectral_summary, x='band', y='mean', hue='condition',
                    marker='o', markersize=8, linewidth=2.5, ax=ax,
                    palette=self.config.primary_palette)
        
        # Add error bars
        for condition in spectral_summary['condition'].unique():
            condition_data = spectral_summary[spectral_summary['condition'] == condition]
            ax.errorbar(range(len(condition_data)), condition_data['mean'], 
                       yerr=condition_data['sem'], fmt='none', capsize=5, alpha=0.7)
        
        ax.set_title('Mean Φ by Frequency Band', fontsize=16, fontweight='bold')
        ax.set_ylabel('Mean Φ Value (bits)')
        ax.set_xlabel('Frequency Band')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45)

    def create_gamma_analysis(self, gamma_df: pd.DataFrame, output_dir: str, min_phi: float = None):
        """Create comprehensive gamma-only analysis.
        Optionally filter to values >= min_phi and include a reference line at that threshold.
        """
        if gamma_df.empty:
            print("⚠️ No gamma band data available for analysis.")
            return
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Apply Φ threshold if provided
        if min_phi is not None:
            gamma_df = gamma_df[gamma_df['value'] >= min_phi].copy()
        
        print(f"   📊 Gamma analysis: {len(gamma_df)} records across {gamma_df['condition'].nunique()} conditions" + (f" (Φ ≥ {min_phi} bits)" if min_phi is not None else ""))
        
        # 1. Gamma-specific comprehensive overview
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Main complex Φ distribution for gamma
        ax1 = fig.add_subplot(gs[0, 0])
        sns.violinplot(data=gamma_df, x='condition', y='value', ax=ax1, 
                      palette=self.config.primary_palette, inner='box')
        ax1.set_title('Gamma Band: Main Complex Φ Distribution', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Max Φ per Epoch (bits)')
        ax1.set_xlabel('Condition')
        ax1.grid(True, alpha=0.3)
        if min_phi is not None:
            ax1.axhline(min_phi, color='red', linestyle='--', linewidth=1, alpha=0.7)
            ax1.text(0.98, min_phi, f"Φ = {min_phi} bit", va='bottom', ha='right', transform=ax1.get_yaxis_transform(), fontsize=9, color='red')
        
        # Complex size distribution for gamma
        ax2 = fig.add_subplot(gs[0, 1])
        sns.histplot(data=gamma_df, x='subset_size', hue='condition', ax=ax2,
                    palette=self.config.primary_palette, bins=range(2, 9), 
                    alpha=0.7, kde=False, discrete=True)
        ax2.set_title('Gamma Band: Main Complex Size', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Complex Size (Channels)')
        ax2.set_ylabel('Count')
        ax2.grid(True, alpha=0.3)
        
        # Channel participation for gamma
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_gamma_channel_participation(gamma_df, ax3)
        
        # Size vs Phi relationship for gamma
        ax4 = fig.add_subplot(gs[1, 0])
        sns.scatterplot(data=gamma_df, x='subset_size', y='value', hue='condition', 
                       ax=ax4, palette=self.config.primary_palette, alpha=0.7, s=60)
        ax4.set_title('Gamma Band: Size vs Φ', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Complex Size')
        ax4.set_ylabel('Φ Value (bits)')
        ax4.grid(True, alpha=0.3)
        if min_phi is not None:
            ax4.axhline(min_phi, color='red', linestyle='--', linewidth=1, alpha=0.7)
        
        # Statistical comparison
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_gamma_statistics(gamma_df, ax5, min_phi=min_phi)
        
        # Time series of main complex Φ (if epoch info available)
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_gamma_time_series(gamma_df, ax6, min_phi=min_phi)
        
        plt.suptitle('Gamma Band Analysis: Main Complex Integration', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        save_path = os.path.join(output_dir, 'gamma_comprehensive_analysis.png')
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved gamma comprehensive analysis: {save_path}")
        
        # 2. Detailed gamma statistical report
        self._create_gamma_statistical_report(gamma_df, output_dir)
        
        # 3. Gamma-specific raincloud plot
        self._create_gamma_raincloud(gamma_df, output_dir)

    def _plot_gamma_channel_participation(self, gamma_df: pd.DataFrame, ax):
        """Plot channel participation specifically for gamma band."""
        if 'subset_channels' not in gamma_df.columns:
            ax.text(0.5, 0.5, 'Channel info not available', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Gamma: Channel Participation', fontsize=14, fontweight='bold')
            return
        
        # Count participation
        channel_counts = {}
        for _, row in gamma_df.iterrows():
            for channel in row['subset_channels']:
                condition = row['condition']
                if condition not in channel_counts:
                    channel_counts[condition] = {}
                if channel not in channel_counts[condition]:
                    channel_counts[condition][channel] = 0
                channel_counts[condition][channel] += 1
        
        # Convert to plotting format
        plot_data = []
        for condition, channels in channel_counts.items():
            for channel, count in channels.items():
                plot_data.append({'condition': condition, 'channel': channel, 'count': count})
        
        if plot_data:
            plot_df = pd.DataFrame(plot_data)
            sns.barplot(data=plot_df, x='channel', y='count', hue='condition', ax=ax,
                       palette=self.config.primary_palette)
            ax.set_title('Gamma: Channel Participation', fontsize=14, fontweight='bold')
            ax.set_xlabel('EEG Channel')
            ax.set_ylabel('Times in Main Complex')
            plt.setp(ax.get_xticklabels(), rotation=45)
        else:
            ax.text(0.5, 0.5, 'No channel data', 
                   transform=ax.transAxes, ha='center', va='center')

    def _plot_gamma_statistics(self, gamma_df: pd.DataFrame, ax, min_phi: float = None):
        """Plot statistical summary for gamma band."""
        # Calculate summary statistics
        stats_data = gamma_df.groupby('condition')['value'].agg(['mean', 'std', 'count', 'sem']).reset_index()
        
        # Bar plot with error bars
        bars = ax.bar(stats_data['condition'], stats_data['mean'], 
                     yerr=stats_data['sem'], capsize=5, 
                     color=sns.color_palette(self.config.primary_palette, len(stats_data)),
                     alpha=0.8)
        
        ax.set_title('Gamma: Mean Φ by Condition', fontsize=14, fontweight='bold')
        ax.set_ylabel('Mean Φ (bits)')
        ax.set_xlabel('Condition')
        ax.grid(True, alpha=0.3)
        if min_phi is not None:
            ax.axhline(min_phi, color='red', linestyle='--', linewidth=1, alpha=0.7)
        
        # Add value labels on bars
        for bar, mean_val in zip(bars, stats_data['mean']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{mean_val:.3f}', ha='center', va='bottom', fontsize=10)

    def _plot_gamma_time_series(self, gamma_df: pd.DataFrame, ax, min_phi: float = None):
        """Plot time series of gamma main complex Φ if epoch info available."""
        if 'epoch_idx' in gamma_df.columns:
            # Plot time series by condition
            for condition in gamma_df['condition'].unique():
                condition_data = gamma_df[gamma_df['condition'] == condition].sort_values('epoch_idx')
                ax.plot(condition_data['epoch_idx'], condition_data['value'], 
                       label=condition, alpha=0.7, linewidth=1.5)
            
            ax.set_title('Gamma: Φ Time Series', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch Index')
            ax.set_ylabel('Φ Value (bits)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            if min_phi is not None:
                ax.axhline(min_phi, color='red', linestyle='--', linewidth=1, alpha=0.7)
        else:
            ax.text(0.5, 0.5, 'Epoch information\nnot available', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, alpha=0.7)
            ax.set_title('Gamma: Time Series', fontsize=14, fontweight='bold')

    def _create_gamma_statistical_report(self, gamma_df: pd.DataFrame, output_dir: str):
        """Create detailed statistical report for gamma band."""
        report = []
        report.append("# Gamma Band Analysis Statistical Report\n")
        report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Basic statistics
        report.append("## Gamma Band Summary\n")
        report.append(f"- Total epochs analyzed: {len(gamma_df)}\n")
        report.append(f"- Conditions: {gamma_df['condition'].nunique()}\n")
        report.append(f"- Subjects: {gamma_df['subject'].nunique()}\n\n")
        
        # Descriptive statistics by condition
        report.append("## Descriptive Statistics by Condition\n")
        desc_stats = gamma_df.groupby('condition')['value'].describe()
        report.append(desc_stats.to_string())
        report.append("\n\n")
        
        # Statistical tests between conditions
        report.append("## Statistical Comparisons\n")
        conditions = gamma_df['condition'].unique()
        
        if len(conditions) >= 2:
            from scipy.stats import mannwhitneyu
            for i in range(len(conditions)):
                for j in range(i+1, len(conditions)):
                    group1 = gamma_df[gamma_df['condition'] == conditions[i]]['value']
                    group2 = gamma_df[gamma_df['condition'] == conditions[j]]['value']
                    
                    if len(group1) > 0 and len(group2) > 0:
                        # Statistical test
                        statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
                        
                        # Effect size
                        pooled_std = np.sqrt((group1.var() + group2.var()) / 2)
                        cohens_d = (group1.mean() - group2.mean()) / pooled_std
                        
                        report.append(f"### {conditions[i]} vs {conditions[j]}\n")
                        report.append(f"- Sample sizes: {len(group1)} vs {len(group2)}\n")
                        report.append(f"- Means: {group1.mean():.4f} vs {group2.mean():.4f}\n")
                        report.append(f"- Mann-Whitney U: {statistic:.3f}, p = {p_value:.6f}\n")
                        report.append(f"- Effect size (Cohen's d): {cohens_d:.3f}\n")
                        report.append(f"- Interpretation: {self._interpret_effect_size(abs(cohens_d))}\n\n")
        
        # Save report
        save_path = os.path.join(output_dir, 'gamma_statistical_report.md')
        with open(save_path, 'w') as f:
            f.writelines(report)
        print(f"✓ Saved gamma statistical report: {save_path}")

    def _create_gamma_raincloud(self, gamma_df: pd.DataFrame, output_dir: str):
        """Create gamma-specific raincloud plot."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Violin + strip + box plot combination
        sns.violinplot(data=gamma_df, x='condition', y='value', ax=ax,
                      palette=self.config.primary_palette, inner=None, alpha=0.6)
        sns.stripplot(data=gamma_df, x='condition', y='value', ax=ax,
                     size=3, alpha=0.7, jitter=True, color='black')
        sns.boxplot(data=gamma_df, x='condition', y='value', ax=ax,
                   width=0.3, boxprops=dict(alpha=0.3), showfliers=False)
        
        ax.set_title('Gamma Band: Main Complex Φ Distribution (Raincloud)', 
                    fontsize=16, fontweight='bold')
        ax.set_ylabel('Max Φ per Epoch (bits)')
        ax.set_xlabel('Condition')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, 'gamma_raincloud_plot.png')
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved gamma raincloud plot: {save_path}")

# --- Main Execution ---

def main():
    """Enhanced main function with comprehensive analysis pipeline."""
    parser = argparse.ArgumentParser(
        description='Advanced EEG Analysis Results Visualization Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_results.py --path results
  python visualize_results.py --path results --no-interactive
  python visualize_results.py --path results --config-style darkgrid
        """
    )
    
    parser.add_argument('--path', default='results',
                        help='Path to the results directory to analyze')
    parser.add_argument('--no-interactive', action='store_true',
                        help='Skip interactive plots generation')
    parser.add_argument('--config-style', default='whitegrid',
                        choices=['whitegrid', 'darkgrid', 'white', 'dark', 'ticks'],
                        help='Seaborn style configuration')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for saved figures')
    
    args = parser.parse_args()
    
    # Update configuration
    global config
    config.style = args.config_style
    config.dpi = args.dpi
    config.save_interactive = not args.no_interactive
    
    # Update plot styling
    sns.set_style(config.style)
    plt.rcParams['savefig.dpi'] = config.dpi

    results_dir = args.path
    summary_output_dir = os.path.join(results_dir, 'summary_plots')

    print("=" * 70)
    print("🧠 EEG Analysis Results Visualization Suite v2.0")
    print("=" * 70)
    print(f"📁 Results directory: {results_dir}")
    print(f"💾 Output directory: {summary_output_dir}")
    print(f"🎨 Plot style: {config.style}")
    print(f"📊 Interactive plots: {'Enabled' if config.save_interactive else 'Disabled'}")
    print("-" * 70)
    
    # Create output directory
    os.makedirs(summary_output_dir, exist_ok=True)
    
    # Initialize processors
    processor = ResultsProcessor()
    plotter = AdvancedPlotter(config)
    
    # Load and process data
    print("🔍 Scanning for result files...")
    results_df = processor.load_and_validate_results([results_dir])
    
    if results_df.empty:
        print("❌ No results found to visualize. Exiting.")
        return
        
    # Print data quality report
    report = processor.data_quality_report
    print(f"📊 Data Quality Report:")
    print(f"   • Files processed: {report['total_files']}")
    print(f"   • Success rate: {report['success_rate']:.1%}")
    print(f"   • Total records: {report['total_records']}")
    print(f"   • Subjects: {results_df['subject'].nunique()}")
    print(f"   • Conditions: {results_df['condition'].nunique()}")
    print(f"   • Metrics: {results_df['metric_estimator'].nunique()}")
    print("-" * 70)

    # Generate visualizations
    print("🎨 Generating comprehensive visualizations...")
    
    # 1. Comprehensive overview
    print("   Creating comprehensive overview...")
    plotter.create_comprehensive_overview(results_df, summary_output_dir)
    
    # 2. Advanced distributions
    print("   Creating advanced distribution plots...")
    plotter.create_advanced_distributions(results_df, summary_output_dir)
    
    # 3. Dimensionality reduction
    print("   Creating dimensionality reduction plots...")
    plotter.create_dimensionality_reduction_plots(results_df, summary_output_dir)
    
    # 4. Statistical report
    print("   Generating statistical report...")
    plotter.create_statistical_report(results_df, summary_output_dir)
    
    # 5. Gamma-only analysis if we have spectral data
    if 'band' in results_df.columns and 'gamma' in results_df['band'].unique():
        print("   Creating gamma-only analysis...")
        gamma_df = results_df[results_df['band'] == 'gamma'].copy()
        gamma_output_dir = os.path.join(summary_output_dir, 'gamma_only')
        os.makedirs(gamma_output_dir, exist_ok=True)
        
        # Generate gamma-specific plots
        plotter.create_gamma_analysis(gamma_df, gamma_output_dir)
        # Filtered: ignore unintegrated (Φ < 3 bits) and mark threshold
        plotter.create_gamma_analysis(gamma_df, os.path.join(gamma_output_dir, 'phi_ge_3bit'), min_phi=3.0)

        # Balanced gamma analysis
        print("   Creating balanced gamma-only analysis (matched epochs per condition)...")
        gamma_balanced = plotter.make_balanced(gamma_df)
        gamma_balanced_dir = os.path.join(gamma_output_dir, 'balanced')
        os.makedirs(gamma_balanced_dir, exist_ok=True)
        plotter.create_gamma_analysis(gamma_balanced, gamma_balanced_dir)
        plotter.create_gamma_analysis(gamma_balanced, os.path.join(gamma_balanced_dir, 'phi_ge_3bit'), min_phi=3.0)
    
    print("-" * 70)
    print("✅ Visualization suite completed successfully!")
    print(f"📁 All outputs saved to: {summary_output_dir}")
    print("=" * 70)

if __name__ == '__main__':
    main()
