#!/usr/bin/env python3
"""
Complexes Results Analysis
==========================

This script analyzes the IIT complexes results to find the most integrated
subsets in each condition and frequency band, and compares them across readings.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def load_complexes_data(file_path):
    """Load complexes data from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def find_most_integrated_subsets(data):
    """
    Find the most integrated subset (highest phi_value) for each condition and frequency band.
    
    Returns:
    --------
    results : dict
        Dictionary with condition -> band -> most_integrated_info
    """
    results = {}
    
    for condition, bands in data.items():
        results[condition] = {}
        
        for band, band_data in bands.items():
            if 'complexes_per_epoch' not in band_data:
                continue
                
            max_phi = -np.inf
            best_subset = None
            best_epoch = None
            all_phi_values = []
            
            # Go through all epochs and find the maximum phi_value
            for epoch_idx, epoch_complexes in enumerate(band_data['complexes_per_epoch']):
                for complex_info in epoch_complexes:
                    phi_value = complex_info['phi_value']
                    all_phi_values.append(phi_value)
                    
                    if phi_value > max_phi:
                        max_phi = phi_value
                        best_subset = complex_info['subset']
                        best_epoch = epoch_idx
            
            results[condition][band] = {
                'most_integrated_subset': best_subset,
                'max_phi_value': max_phi,
                'epoch_found': best_epoch,
                'mean_phi': band_data.get('mean_phi', 0),
                'total_complexes': band_data.get('total_complexes', 0),
                'n_epochs': band_data.get('n_epochs', 0),
                'all_phi_values': all_phi_values
            }
    
    return results

def analyze_subset_patterns(results):
    """Analyze patterns in the most integrated subsets."""
    subset_patterns = defaultdict(list)
    subset_sizes = defaultdict(list)
    
    for condition in results:
        for band in results[condition]:
            subset = results[condition][band]['most_integrated_subset']
            phi_value = results[condition][band]['max_phi_value']
            
            if subset is not None:
                subset_key = tuple(sorted(subset))
                subset_patterns[subset_key].append((condition, band, phi_value))
                subset_sizes[len(subset)].append((condition, band, phi_value))
    
    return subset_patterns, subset_sizes

def create_comparison_table(results):
    """Create a comparison table of most integrated subsets."""
    rows = []
    
    for condition in results:
        for band in results[condition]:
            info = results[condition][band]
            subset = info['most_integrated_subset']
            
            if subset is not None:
                rows.append({
                    'Condition': condition,
                    'Frequency Band': band,
                    'Most Integrated Subset': f"Channels {subset}",
                    'Subset Size': len(subset),
                    'Max Φ Value': f"{info['max_phi_value']:.3f}",
                    'Mean Φ Value': f"{info['mean_phi']:.3f}",
                    'Total Complexes': info['total_complexes'],
                    'Epochs': info['n_epochs']
                })
    
    return pd.DataFrame(rows)

def plot_phi_distributions(results):
    """Plot phi value distributions across conditions and bands."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'broadband']
    conditions = list(results.keys())
    
    for i, band in enumerate(bands):
        ax = axes[i]
        
        for condition in conditions:
            if band in results[condition]:
                phi_values = results[condition][band]['all_phi_values']
                if phi_values:
                    ax.hist(phi_values, alpha=0.6, label=condition, bins=30)
        
        ax.set_title(f'{band.capitalize()} Band')
        ax.set_xlabel('Φ Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('phi_distributions_by_band.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_max_phi_comparison(results):
    """Plot comparison of maximum phi values across conditions and bands."""
    data_for_plot = []
    
    for condition in results:
        for band in results[condition]:
            info = results[condition][band]
            if info['most_integrated_subset'] is not None:
                data_for_plot.append({
                    'Condition': condition,
                    'Band': band,
                    'Max Φ': info['max_phi_value'],
                    'Mean Φ': info['mean_phi'],
                    'Subset Size': len(info['most_integrated_subset'])
                })
    
    df = pd.DataFrame(data_for_plot)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Max phi by condition and band
    pivot_max = df.pivot(index='Band', columns='Condition', values='Max Φ')
    sns.heatmap(pivot_max, annot=True, fmt='.2f', cmap='viridis', ax=axes[0])
    axes[0].set_title('Maximum Φ Values by Condition and Band')
    
    # Mean phi by condition and band
    pivot_mean = df.pivot(index='Band', columns='Condition', values='Mean Φ')
    sns.heatmap(pivot_mean, annot=True, fmt='.2f', cmap='viridis', ax=axes[1])
    axes[1].set_title('Mean Φ Values by Condition and Band')
    
    # Subset size by condition and band
    pivot_size = df.pivot(index='Band', columns='Condition', values='Subset Size')
    sns.heatmap(pivot_size, annot=True, fmt='.0f', cmap='plasma', ax=axes[2])
    axes[2].set_title('Most Integrated Subset Size by Condition and Band')
    
    plt.tight_layout()
    plt.savefig('phi_comparison_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main analysis function."""
    # Load data
    print("Loading complexes data...")
    data = load_complexes_data('results/batch_complexes-ksg_sample/subjects/sub-1010_complexes-ksg_results.json')
    
    # Find most integrated subsets
    print("Finding most integrated subsets...")
    results = find_most_integrated_subsets(data)
    
    # Create comparison table
    print("Creating comparison table...")
    comparison_df = create_comparison_table(results)
    
    # Display results
    print("\n" + "="*80)
    print("MOST INTEGRATED SUBSETS ANALYSIS - SUBJECT 1010")
    print("="*80)
    print(comparison_df.to_string(index=False))
    
    # Analyze patterns
    print("\n" + "="*80)
    print("SUBSET PATTERN ANALYSIS")
    print("="*80)
    
    subset_patterns, subset_sizes = analyze_subset_patterns(results)
    
    print("\nMost common subset patterns:")
    for subset, occurrences in sorted(subset_patterns.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        print(f"  Channels {list(subset)}: {len(occurrences)} occurrences")
        for condition, band, phi in occurrences:
            print(f"    - {condition} {band}: Φ = {phi:.3f}")
    
    print("\nSubset size distribution:")
    for size in sorted(subset_sizes.keys()):
        occurrences = subset_sizes[size]
        avg_phi = np.mean([phi for _, _, phi in occurrences])
        print(f"  Size {size}: {len(occurrences)} occurrences, avg Φ = {avg_phi:.3f}")
    
    # Condition comparisons
    print("\n" + "="*80)
    print("CONDITION COMPARISONS")
    print("="*80)
    
    for band in ['delta', 'theta', 'alpha', 'beta', 'gamma', 'broadband']:
        print(f"\n{band.upper()} BAND:")
        band_results = []
        for condition in results:
            if band in results[condition]:
                info = results[condition][band]
                subset = info['most_integrated_subset']
                if subset is not None:
                    band_results.append((condition, subset, info['max_phi_value']))
        
        # Sort by phi value
        band_results.sort(key=lambda x: x[2], reverse=True)
        
        for condition, subset, phi in band_results:
            print(f"  {condition:20s}: Channels {subset} (Φ = {phi:.3f})")
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_phi_distributions(results)
    plot_max_phi_comparison(results)
    
    # Save detailed results
    comparison_df.to_csv('most_integrated_subsets_analysis.csv', index=False)
    print("\nDetailed results saved to 'most_integrated_subsets_analysis.csv'")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 