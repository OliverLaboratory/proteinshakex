#!/usr/bin/env python3
"""
Reproduce Conformation Discovery Benchmark figures and tables.

Downloads ground truth data from ProteinShake (Zenodo) and experiment
outputs (TM-score matrices) from the benchmark Zenodo deposit, then
computes all metrics and generates paper figures.

Usage:
    python scripts/reproduce_benchmark.py
    python scripts/reproduce_benchmark.py --output-dir figures --datasets CFOLD DANCE UNICORNE
"""
import os
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from proteinshake.tasks import UNICORNETask, DANCETask, CFOLDTask
from proteinshake.tasks.ensemble_prediction import (
    discovery_rate, coverage_rate, ensemble_utility, earth_movers_distance,
    hits_at_k, precision_at_k,
    HITS_TM_THRESHOLD, EMD_TM_FLOOR,
)
from proteinshake.utils import download_url

# ─── Configuration ───────────────────────────────────────────────────────────

BENCHMARK_ZENODO_RECORD = 19132538  # experiment outputs
METHODS = ['alphaflow', 'bioemu', 'esmdiff']
METHOD_LABELS = {'alphaflow': 'AlphaFlow', 'bioemu': 'BioEmu', 'esmdiff': 'ESMDiff'}
METHOD_COLORS = {'alphaflow': '#1f77b4', 'bioemu': '#ff7f0e', 'esmdiff': '#2ca02c'}
DATASET_TASKS = {
    'UNICORNE': UNICORNETask,
    'DANCE': DANCETask,
    'CFOLD': CFOLDTask,
}
MAX_K = 250


# ─── Download helpers ────────────────────────────────────────────────────────

def download_experiment_outputs(data_dir, datasets):
    """Download TM-score matrices for benchmark ensembles from Zenodo."""
    os.makedirs(data_dir, exist_ok=True)
    base_url = f'https://zenodo.org/record/{BENCHMARK_ZENODO_RECORD}/files'
    for dataset in datasets:
        for method in METHODS:
            fname = f'{dataset}_{method}.npz'
            fpath = os.path.join(data_dir, fname)
            if not os.path.exists(fpath):
                url = f'{base_url}/{fname}'
                print(f'Downloading {fname}...')
                try:
                    download_url(url, data_dir, verbosity=2)
                except Exception as e:
                    print(f'  Warning: could not download {fname}: {e}')


def load_experiment_matrices(data_dir, dataset, method):
    """Load TM-score matrices for one dataset × method.

    Returns dict: ensemble_id -> {'sim_xy': (n_gt, n_pred), 'sim_xx': (n_gt, n_gt)}
    """
    fpath = os.path.join(data_dir, f'{dataset}_{method}.npz')
    if not os.path.exists(fpath):
        return {}
    data = np.load(fpath, allow_pickle=False)
    ensembles = {}
    for key in data.files:
        eid, mat_type = key.rsplit('__', 1)
        if eid not in ensembles:
            ensembles[eid] = {}
        ensembles[eid][mat_type] = data[key]
    return ensembles


# ─── Metric computation ─────────────────────────────────────────────────────

def compute_all_metrics(task, experiment_data):
    """Compute per-ensemble metrics for one dataset × method.

    Returns list of dicts with metrics + metadata per ensemble.
    """
    results = []
    eid_to_idx = {task.proteins[i]['protein']['ID']: i
                  for i in task.benchmark_index}

    for eid, matrices in experiment_data.items():
        if eid not in eid_to_idx:
            continue
        sim_xy = matrices.get('sim_xy')
        sim_xx = matrices.get('sim_xx')
        if sim_xy is None:
            continue

        idx = eid_to_idx[eid]
        protein = task.proteins[idx]
        ref_len = protein['protein']['num_global_residues']
        n_gt = sim_xy.shape[0]
        n_pred = sim_xy.shape[1]

        # Core metrics
        dr = discovery_rate(sim_xy)
        cr = coverage_rate(sim_xy)
        eu = ensemble_utility(sim_xy)
        try:
            emd = earth_movers_distance(sim_xy)
        except Exception:
            emd = float('nan')

        # Hits@K and Precision@K curves
        h_at_k = hits_at_k(sim_xy)
        p_at_k = precision_at_k(sim_xy)

        # GT diversity (from sim_xx)
        if sim_xx is not None and sim_xx.shape[0] > 1:
            mask = ~np.eye(sim_xx.shape[0], dtype=bool)
            avg_pairwise_tm = float(sim_xx[mask].mean())
        else:
            avg_pairwise_tm = 1.0

        results.append({
            'ensemble_id': eid,
            'DR': dr,
            'CR': cr,
            'EU': eu,
            'EMD': emd,
            'hits_at_k': h_at_k,
            'precision_at_k': p_at_k,
            'n_gt': n_gt,
            'n_pred': n_pred,
            'ref_seq_len': ref_len,
            'avg_pairwise_tm': avg_pairwise_tm,
        })

    return results


# ─── Plotting functions ──────────────────────────────────────────────────────

def _pad_curve(curve, length=MAX_K):
    if len(curve) >= length:
        return curve[:length]
    return np.pad(curve, (0, length - len(curve)), constant_values=curve[-1])


def plot_hits_at_k(all_results, dataset, output_dir):
    """Hits@K (recall) curves averaged across ensembles, one line per method."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in METHODS:
        results = all_results.get(method, [])
        if not results:
            continue
        curves = np.array([_pad_curve(r['hits_at_k']) for r in results])
        mean = curves.mean(axis=0)
        ax.plot(np.arange(1, MAX_K + 1), mean,
                label=METHOD_LABELS[method], color=METHOD_COLORS[method], lw=2)
    ax.set_xlabel('Number of predictions (k)')
    ax.set_ylabel('Discovery Rate')
    ax.set_title(f'{dataset} — Hits@K')
    ax.legend()
    ax.set_xlim(1, MAX_K)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{dataset}_hits_at_k.png'), dpi=150)
    plt.close(fig)


def plot_precision_at_k(all_results, dataset, output_dir):
    """Precision@K curves averaged across ensembles."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in METHODS:
        results = all_results.get(method, [])
        if not results:
            continue
        curves = np.array([_pad_curve(r['precision_at_k']) for r in results])
        mean = curves.mean(axis=0)
        ax.plot(np.arange(1, MAX_K + 1), mean,
                label=METHOD_LABELS[method], color=METHOD_COLORS[method], lw=2)
    ax.set_xlabel('Number of predictions (k)')
    ax.set_ylabel('Precision')
    ax.set_title(f'{dataset} — Precision@K')
    ax.legend()
    ax.set_xlim(1, MAX_K)
    ax.set_ylim(0, None)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{dataset}_precision_curve.png'), dpi=150)
    plt.close(fig)


def plot_dr_distribution(all_results, dataset, output_dir):
    """Histogram of discovery rates per method."""
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 21)
    for method in METHODS:
        results = all_results.get(method, [])
        if not results:
            continue
        drs = [r['DR'] for r in results]
        ax.hist(drs, bins=bins, alpha=0.5, label=METHOD_LABELS[method],
                color=METHOD_COLORS[method], edgecolor='white')
    ax.set_xlabel('Discovery Rate')
    ax.set_ylabel('Count')
    ax.set_title(f'{dataset} — DR Distribution')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{dataset}_dr_distribution.png'), dpi=150)
    plt.close(fig)


def plot_metric_by_diversity(all_results, dataset, metric, output_dir):
    """Scatter plot of a metric vs GT diversity (avg pairwise TM)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in METHODS:
        results = all_results.get(method, [])
        if not results:
            continue
        x = [r['avg_pairwise_tm'] for r in results]
        y = [r[metric] for r in results]
        ax.scatter(x, y, alpha=0.3, s=10, color=METHOD_COLORS[method],
                   label=METHOD_LABELS[method])
    ax.set_xlabel('Avg Pairwise TM (GT diversity)')
    ax.set_ylabel(metric)
    ax.set_title(f'{dataset} — {metric} vs Diversity')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{dataset}_{metric.lower()}_by_diversity.png'), dpi=150)
    plt.close(fig)


def plot_overall_barplot(dataset_results, metric, output_dir):
    """Grouped bar plot: metric across datasets × methods."""
    datasets = list(dataset_results.keys())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(datasets))
    width = 0.25

    for i, method in enumerate(METHODS):
        vals = []
        for ds in datasets:
            results = dataset_results[ds].get(method, [])
            if results:
                vals.append(np.mean([r[metric] for r in results]))
            else:
                vals.append(0)
        ax.bar(x + i * width, vals, width, label=METHOD_LABELS[method],
               color=METHOD_COLORS[method])

    ax.set_xticks(x + width)
    ax.set_xticklabels(datasets)
    ax.set_ylabel(metric)
    ax.set_title(f'Overall — {metric}')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'overall_{metric.lower()}.png'), dpi=150)
    plt.close(fig)


def generate_summary_table(dataset_results, output_dir):
    """Print and save LaTeX summary table."""
    lines = []
    lines.append(r'\begin{tabular}{ll' + 'c' * 4 + '}')
    lines.append(r'\toprule')
    lines.append(r'Dataset & Method & DR $\uparrow$ & CR $\uparrow$ & EU $\uparrow$ & EMD $\downarrow$ \\')
    lines.append(r'\midrule')

    for ds in dataset_results:
        first = True
        for method in METHODS:
            results = dataset_results[ds].get(method, [])
            if not results:
                continue
            dr = np.mean([r['DR'] for r in results])
            cr = np.mean([r['CR'] for r in results])
            eu = np.mean([r['EU'] for r in results])
            emd = np.nanmean([r['EMD'] for r in results])
            ds_label = ds if first else ''
            first = False
            lines.append(
                f'{ds_label} & {METHOD_LABELS[method]} & '
                f'{dr:.3f} & {cr:.3f} & {eu:.3f} & {emd:.3f} \\\\'
            )
        lines.append(r'\midrule')

    lines[-1] = r'\bottomrule'
    lines.append(r'\end{tabular}')

    table = '\n'.join(lines)
    print('\n' + table + '\n')
    with open(os.path.join(output_dir, 'tables.tex'), 'w') as f:
        f.write(table)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Reproduce benchmark figures')
    parser.add_argument('--data-dir', default='data/benchmark', help='Data directory')
    parser.add_argument('--output-dir', default='figures', help='Output directory for figures')
    parser.add_argument('--datasets', nargs='+', default=['CFOLD', 'DANCE', 'UNICORNE'])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Download experiment outputs
    print('=== Downloading experiment outputs ===')
    download_experiment_outputs(args.data_dir, args.datasets)

    # 2. Load tasks (downloads GT data from proteinshakex Zenodo)
    print('\n=== Loading benchmark tasks ===')
    tasks = {}
    for ds in args.datasets:
        TaskClass = DATASET_TASKS[ds]
        print(f'Loading {ds}...')
        task = TaskClass(root=args.data_dir, split='benchmark',
                         use_precomputed=True, verbosity=1)
        tasks[ds] = task
        print(f'  {len(task.benchmark_index)} benchmark ensembles')

    # 3. Compute metrics
    print('\n=== Computing metrics ===')
    dataset_results = {}  # dataset -> method -> list of result dicts
    for ds in args.datasets:
        dataset_results[ds] = {}
        for method in METHODS:
            print(f'  {ds}/{method}...', end=' ')
            experiment_data = load_experiment_matrices(args.data_dir, ds, method)
            if not experiment_data:
                print('no data')
                continue
            results = compute_all_metrics(tasks[ds], experiment_data)
            dataset_results[ds][method] = results
            if results:
                dr = np.mean([r['DR'] for r in results])
                eu = np.mean([r['EU'] for r in results])
                print(f'{len(results)} ensembles, DR={dr:.3f}, EU={eu:.3f}')
            else:
                print('no matching ensembles')

    # 4. Generate figures
    print('\n=== Generating figures ===')

    # Per-dataset figures
    for ds in args.datasets:
        print(f'  {ds}...')
        plot_hits_at_k(dataset_results[ds], ds, args.output_dir)
        plot_precision_at_k(dataset_results[ds], ds, args.output_dir)
        plot_dr_distribution(dataset_results[ds], ds, args.output_dir)
        for metric in ['DR', 'EU', 'EMD']:
            plot_metric_by_diversity(dataset_results[ds], ds, metric, args.output_dir)

    # Overall comparison figures
    print('  Overall...')
    for metric in ['DR', 'CR', 'EU', 'EMD']:
        plot_overall_barplot(dataset_results, metric, args.output_dir)

    # Summary table
    print('\n=== Summary Table ===')
    generate_summary_table(dataset_results, args.output_dir)

    print(f'\nFigures saved to {args.output_dir}/')
    print('Done!')


if __name__ == '__main__':
    main()
