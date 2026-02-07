#!/usr/bin/env python3
"""
kNN Purity Analysis Script
Analyzes neighborhood preservation across different transformation models.

kNN purity measures how well the k-nearest neighbors in the transformed space
match the k-nearest neighbors in the original space.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import argparse
from typing import Dict, List, Tuple


def compute_knn_purity(
    embeddings_original: torch.Tensor,
    embeddings_transformed: torch.Tensor,
    k_values: List[int]
) -> Dict[int, float]:
    """
    Compute kNN purity for different k values.
    
    Purity = (number of neighbors preserved) / k
    
    Args:
        embeddings_original: Original embeddings [N, D]
        embeddings_transformed: Transformed embeddings [N, D']
        k_values: List of k values to evaluate
        
    Returns:
        Dictionary mapping k -> purity score
    """
    embeddings_original = embeddings_original.cpu().numpy()
    embeddings_transformed = embeddings_transformed.cpu().numpy()
    
    # Find k-nearest neighbors in original space
    max_k = max(k_values)
    nbrs_original = NearestNeighbors(n_neighbors=max_k + 1, metric='euclidean')
    nbrs_original.fit(embeddings_original)
    _, indices_original = nbrs_original.kneighbors(embeddings_original)
    indices_original = indices_original[:, 1:]  # Exclude self
    
    # Find k-nearest neighbors in transformed space
    nbrs_transformed = NearestNeighbors(n_neighbors=max_k + 1, metric='euclidean')
    nbrs_transformed.fit(embeddings_transformed)
    _, indices_transformed = nbrs_transformed.kneighbors(embeddings_transformed)
    indices_transformed = indices_transformed[:, 1:]  # Exclude self
    
    purity_scores = {}
    
    for k in k_values:
        # For each point, compute overlap between original and transformed neighbors
        overlap_counts = []
        for i in range(len(embeddings_original)):
            original_neighbors = set(indices_original[i, :k])
            transformed_neighbors = set(indices_transformed[i, :k])
            overlap = len(original_neighbors.intersection(transformed_neighbors))
            overlap_counts.append(overlap)
        
        # Purity is the average overlap ratio
        purity = np.mean(overlap_counts) / k
        purity_scores[k] = purity
    
    return purity_scores


def load_embeddings(base_path: Path, roundtrip: bool = False) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Load all embeddings for analysis.
    
    Returns:
        Dict with structure:
        {
            'clip': {
                'original': tensor,
                'mlp': tensor,
                'ddpm': tensor,
                'ddim': tensor,
                'flow': tensor
            },
            'dino': {...}
        }
    """
    embeddings = {
        'clip': {},
        'dino': {}
    }
    
    # Load original embeddings
    embeddings['clip']['original'] = torch.load(base_path / 'clip.pt')
    embeddings['dino']['original'] = torch.load(base_path / 'dino.pt')
    
    # Model names and file patterns
    if roundtrip:
        models = {
            'mlp': 'clip_by_roundtrip_mlp.pt',
            'ddpm': 'clip_by_roundtrip_diffusion.pt',
            'ddim': 'clip_by_roundtrip_ddim.pt',
            'flow': 'clip_by_roundtrip_flow.pt'
        }
        dino_models = {
            'mlp': 'dino_by_roundtrip_mlp.pt',
            'ddpm': 'dino_by_roundtrip_diffusion.pt',
            'ddim': 'dino_by_roundtrip_ddim.pt',
            'flow': 'dino_by_roundtrip_flow.pt'
        }
    else:
        models = {
            'mlp': 'clip_by_dino_mlp.pt',
            'ddpm': 'clip_by_dino_diffusion.pt',
            'ddim': 'clip_by_dino_ddim.pt',
            'flow': 'clip_by_dino_flow.pt'
        }
        dino_models = {
            'mlp': 'dino_by_clip_mlp.pt',
            'ddpm': 'dino_by_clip_diffusion.pt',
            'ddim': 'dino_by_clip_ddim.pt',
            'flow': 'dino_by_clip_flow.pt'
        }
    
    # Load CLIP-based embeddings (DINO->CLIP transformations)
    for model_name, filename in models.items():
        filepath = base_path / filename
        if filepath.exists():
            embeddings['clip'][model_name] = torch.load(filepath)
        else:
            print(f"Warning: {filepath} not found, skipping {model_name}")
    
    # Load DINO-based embeddings (CLIP->DINO transformations)
    for model_name, filename in dino_models.items():
        filepath = base_path / filename
        if filepath.exists():
            embeddings['dino'][model_name] = torch.load(filepath)
        else:
            print(f"Warning: {filepath} not found, skipping {model_name}")
    
    return embeddings


def create_purity_table(
    embeddings: Dict[str, Dict[str, torch.Tensor]],
    k_values: List[int]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create purity tables for CLIP and DINO embeddings.
    
    Returns:
        (clip_table, dino_table) as DataFrames
    """
    model_order = ['mlp', 'ddpm', 'ddim', 'flow']
    model_names = {
        'mlp': 'MLP',
        'ddpm': 'DDPM',
        'ddim': 'DDIM',
        'flow': 'Flow'
    }
    
    results = {'clip': {}, 'dino': {}}
    
    # Compute purity for each embedding type
    for embed_type in ['clip', 'dino']:
        original = embeddings[embed_type]['original']
        
        for model_key in model_order:
            if model_key in embeddings[embed_type]:
                transformed = embeddings[embed_type][model_key]
                purity_scores = compute_knn_purity(original, transformed, k_values)
                results[embed_type][model_names[model_key]] = purity_scores
    
    # Create DataFrames
    clip_df = pd.DataFrame(results['clip']).T
    clip_df.columns = [f'k={k}' for k in k_values]
    
    dino_df = pd.DataFrame(results['dino']).T
    dino_df.columns = [f'k={k}' for k in k_values]
    
    return clip_df, dino_df


def plot_purity_comparison(
    clip_df: pd.DataFrame,
    dino_df: pd.DataFrame,
    output_path: Path,
    roundtrip: bool = False
):
    """
    Create a publication-quality plot comparing purity across models and k values.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extract k values from column names
    k_values = [int(col.split('=')[1]) for col in clip_df.columns]
    
    # Define colors and markers for better visibility
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    markers = ['o', 's', '^', 'D']  # Circle, Square, Triangle, Diamond
    linestyles = ['-', '--', '-.', ':']
    
    # Plot CLIP
    ax = axes[0]
    for idx, model_name in enumerate(clip_df.index):
        purities = clip_df.loc[model_name].values
        ax.plot(k_values, purities, 
                marker=markers[idx], 
                linewidth=2.5, 
                markersize=9,
                linestyle=linestyles[idx],
                color=colors[idx],
                label=model_name,
                alpha=0.9)
    
    ax.set_xlabel('k (number of neighbors)', fontsize=12)
    ax.set_ylabel('kNN Purity', fontsize=12)
    title = 'CLIP Space' + (' (Roundtrip)' if roundtrip else ' (DINO→CLIP)')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=10, loc='best', framealpha=0.95)
    
    # Auto-scale y-axis based on data range with padding
    all_values = clip_df.values.flatten()
    y_min = max(0, np.min(all_values) - 0.05)
    y_max = min(1, np.max(all_values) + 0.05)
    ax.set_ylim([y_min, y_max])
    
    # Plot DINO
    ax = axes[1]
    for idx, model_name in enumerate(dino_df.index):
        purities = dino_df.loc[model_name].values
        ax.plot(k_values, purities,
                marker=markers[idx],
                linewidth=2.5,
                markersize=9,
                linestyle=linestyles[idx],
                color=colors[idx],
                label=model_name,
                alpha=0.9)
    
    ax.set_xlabel('k (number of neighbors)', fontsize=12)
    ax.set_ylabel('kNN Purity', fontsize=12)
    title = 'DINO Space' + (' (Roundtrip)' if roundtrip else ' (CLIP→DINO)')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=10, loc='best', framealpha=0.95)
    
    # Auto-scale y-axis based on data range with padding
    all_values = dino_df.values.flatten()
    y_min = max(0, np.min(all_values) - 0.05)
    y_max = min(1, np.max(all_values) + 0.05)
    ax.set_ylim([y_min, y_max])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {output_path}")
    plt.close()


def print_tables(clip_df: pd.DataFrame, dino_df: pd.DataFrame, dataset: str, roundtrip: bool = False):
    """Print formatted tables to terminal."""
    mode = "Roundtrip" if roundtrip else "Forward"
    
    print("\n" + "="*80)
    print(f"kNN PURITY ANALYSIS - {dataset} - {mode} Mode")
    print("="*80)
    
    print(f"\n{'CLIP Space (DINO→CLIP transformation)':^80}")
    print("-"*80)
    print(clip_df.to_string(float_format=lambda x: f'{x:.4f}'))
    
    print(f"\n\n{'DINO Space (CLIP→DINO transformation)':^80}")
    print("-"*80)
    print(dino_df.to_string(float_format=lambda x: f'{x:.4f}'))
    
    print("\n" + "="*80)


def process_dataset(dataset: str, k_values: List[int], roundtrip: bool, output_dir: Path):
    """Process a single dataset."""
    print(f"\n{'='*80}")
    print(f"Processing dataset: {dataset} ({'roundtrip' if roundtrip else 'forward'})")
    print(f"{'='*80}")
    
    base_path = Path('data/embeddings') / dataset
    
    if not base_path.exists():
        print(f"⚠ Dataset path {base_path} not found, skipping...")
        return None
    
    mode_suffix = '_roundtrip' if roundtrip else '_forward'
    csv_path = output_dir / f'knn_purity_{dataset}{mode_suffix}.csv'
    plot_path = output_dir / f'knn_purity_{dataset}{mode_suffix}.pdf'
    
    print(f"Loading embeddings from {base_path}...")
    embeddings = load_embeddings(base_path, roundtrip=roundtrip)
    
    print(f"Computing kNN purity for k values: {k_values}...")
    clip_df, dino_df = create_purity_table(embeddings, k_values)
    
    # Save CSV files
    print(f"Saving results to {csv_path}...")
    with open(csv_path, 'w') as f:
        f.write(f"# CLIP Space (DINO→CLIP)\n")
        clip_df.to_csv(f, float_format='%.4f')
        f.write(f"\n# DINO Space (CLIP→DINO)\n")
        dino_df.to_csv(f, float_format='%.4f')
    
    print(f"✓ CSV saved to {csv_path}")
    
    # Create plot
    plot_purity_comparison(clip_df, dino_df, plot_path, roundtrip=roundtrip)
    
    # Print to terminal
    print_tables(clip_df, dino_df, dataset, roundtrip=roundtrip)
    
    # Print summary statistics
    print(f"\n{'SUMMARY STATISTICS':^80}")
    print("-"*80)
    print("\nCLIP Space - Average purity across all k values:")
    for model in clip_df.index:
        avg_purity = clip_df.loc[model].mean()
        print(f"  {model:8s}: {avg_purity:.4f}")
    
    print("\nDINO Space - Average purity across all k values:")
    for model in dino_df.index:
        avg_purity = dino_df.loc[model].mean()
        print(f"  {model:8s}: {avg_purity:.4f}")
    print("="*80)
    
    return (clip_df, dino_df)


def main():
    parser = argparse.ArgumentParser(
        description='Perform kNN purity analysis on embedding transformations'
    )
    parser.add_argument(
        '--roundtrip',
        action='store_true',
        help='Analyze roundtrip embeddings instead of forward transformations'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=['CIFAR_5k', 'CIFAR_50k', 'CIFAR_5k_test'],
        help='Datasets to analyze (default: all three)'
    )
    parser.add_argument(
        '--k-values',
        type=int,
        nargs='+',
        default=[5, 10, 20, 50, 100, 200, 500],
        help='K values for kNN analysis'
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path('logs/eval/knn_purity')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("kNN PURITY ANALYSIS")
    print("="*80)
    print(f"Mode: {'Roundtrip' if args.roundtrip else 'Forward'}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"K values: {args.k_values}")
    print("="*80)
    
    # Process each dataset
    all_results = {}
    for dataset in args.datasets:
        result = process_dataset(dataset, args.k_values, args.roundtrip, output_dir)
        if result is not None:
            all_results[dataset] = result
    
    # Final summary
    if all_results:
        print("\n\n" + "="*80)
        print("FINAL SUMMARY - AVERAGE PURITY ACROSS ALL K VALUES")
        print("="*80)
        
        for dataset, (clip_df, dino_df) in all_results.items():
            print(f"\n{dataset}:")
            print("  CLIP Space:")
            for model in clip_df.index:
                avg = clip_df.loc[model].mean()
                print(f"    {model:8s}: {avg:.4f}")
            print("  DINO Space:")
            for model in dino_df.index:
                avg = dino_df.loc[model].mean()
                print(f"    {model:8s}: {avg:.4f}")
        
        print("\n" + "="*80)
        print("INTERPRETATION:")
        print("  - Purity = 1.0: Perfect neighborhood preservation")
        print("  - Purity = 0.0: No neighborhood preservation")
        print("  - Higher purity → better preservation of local structure")
        print("  - Flow models typically show highest purity (topology preservation)")
        print("  - MLP models may show lower purity (allow geometric rearrangement)")
        print("="*80 + "\n")
    else:
        print("\n⚠ No datasets were successfully processed.")


if __name__ == '__main__':
    main()
