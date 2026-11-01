"""Render the English report figures from unchanged historical CSVs."""

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .plot_readme_figures import reviewed_scores, sha256

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    reviewed_scores()
    rows = read_csv(ROOT / 'results/test_runs.csv')
    scores = {(r['model'], int(r['seed'])): 100*float(r['test_accuracy']) for r in rows}
    inputs = ['results/test_runs.csv', 'results/test_summary.json',
              'results/paired_differences.csv', 'results/metrics/gcn_seed00.csv',
              'results/metrics/mlp_seed00.csv', 'tools/plot_report_en.py',
              'tools/plot_readme_figures.py']
    with tempfile.TemporaryDirectory(prefix='gcn_report_en_') as cache:
        os.environ['MPLCONFIGDIR'] = cache
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt

        plt.rcParams.update({
            'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12,
            'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
            'axes.spines.top': False, 'axes.spines.right': False,
            'svg.fonttype': 'path', 'svg.hashsalt': 'gcn-cora-report-en-v1',
            'savefig.facecolor': 'white', 'figure.facecolor': 'white',
            'axes.edgecolor': '#66717E', 'axes.labelcolor': '#243447',
            'text.color': '#243447',
        })
        blue, orange, gray = '#245C91', '#C57B22', '#58616E'
        figures = []

        def finish(fig, name, sources, note):
            outputs = {}
            for extension in ('png', 'svg'):
                path = out / f'{name}.{extension}'
                metadata = {'Date': None} if extension == 'svg' else {}
                fig.savefig(path, dpi=200, bbox_inches='tight', metadata=metadata)
                if extension == 'svg':
                    path.write_bytes(b'\n'.join(line.rstrip() for line in
                                               path.read_bytes().splitlines()) + b'\n')
                outputs[path.name] = sha256(path)
            plt.close(fig)
            figures.append({'figure': name, 'sources': sources,
                            'note': note, 'outputs_sha256': outputs})

        def grid(ax):
            ax.grid(axis='y', color='#DCE2E8', linewidth=.65)
            ax.set_axisbelow(True)

        fig, ax = plt.subplots(figsize=(9.2, 4.5), layout='constrained')
        for x, (kind, color, ref) in enumerate([('gcn', blue, 81.5), ('mlp', orange, 55.1)]):
            values = np.array([scores[kind, seed] for seed in range(10)])
            ax.scatter(x+np.linspace(-.10, .14, 10), values, c=color, s=35,
                       marker='o' if kind == 'gcn' else 's', zorder=4,
                       label='Individual run' if x == 0 else None)
            mean, std = values.mean(), values.std(ddof=1)
            ax.errorbar(x-.32, mean, yerr=std, fmt='D', color=color,
                        markersize=6, capsize=6, elinewidth=1.8, zorder=5)
            ax.scatter([x+.35], [ref], marker='^', facecolors='none',
                       edgecolors=gray, s=85, zorder=5,
                       label='Paper fixed-split mean (100 runs)' if x == 0 else None)
            ax.annotate(f'{mean:.2f}% ± {std:.2f} pp', (x-.32, mean+std),
                        xytext=(0, 10), textcoords='offset points', ha='center', fontsize=10)
            ax.annotate(f'Paper {ref:.1f}%', (x+.35, ref), xytext=(8, -12),
                        textcoords='offset points', fontsize=9, color=gray)
        ax.annotate('seed=0: 21.6%', (.9, 21.6), xytext=(.47, 13),
                    arrowprops={'arrowstyle': '-', 'color': gray}, fontsize=10)
        ax.set(xlim=(-.75, 1.8), ylim=(0, 100), xticks=[0, 1],
               xticklabels=['GCN (n=10)', 'MLP (n=10)'], ylabel='Test accuracy (%)',
               title='Fixed public split: all seeds and mean ± sample standard deviation')
        ax.legend(loc='lower left', frameon=False, fontsize=9)
        grid(ax)
        finish(fig, '01_test_accuracy', ['results/test_runs.csv',
               'arXiv:1609.02907v4, p7, Tables 2/3'],
               'Sample std., ddof=1; not a confidence interval. All runs retained.')

        seeds = np.arange(10)
        fig, axes = plt.subplots(2, 1, figsize=(9.2, 5.6), layout='constrained', sharex=True)
        for kind, color, marker, offset in [('gcn', blue, 'o', -.09), ('mlp', orange, 's', .09)]:
            axes[0].scatter(seeds+offset, [scores[kind, s] for s in seeds],
                            color=color, marker=marker, s=40, label=kind.upper())
        axes[0].set(ylabel='Test accuracy (%)', ylim=(0, 100),
                    title='Test accuracy and differences for matched seeds')
        axes[0].legend(frameon=False, ncol=2, loc='lower right')
        differences = np.array([scores['gcn', s]-scores['mlp', s] for s in seeds])
        bars = axes[1].bar(seeds, differences, color=blue, width=.65)
        bars[0].set(facecolor='white', edgecolor=blue, hatch='///')
        axes[1].axhline(differences.mean(), color=gray, ls='--', lw=1,
                        label=f'Mean paired difference: {differences.mean():.2f} pp')
        for x, value in enumerate(differences):
            axes[1].text(x, value+1, f'{value:.1f}', ha='center', fontsize=9,
                         bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': .35}, zorder=5)
        axes[1].set(ylabel='GCN − MLP (pp)', ylim=(0, 66), xlabel='Predeclared random seed',
                    xticks=seeds, xlim=(-.65, 9.65))
        axes[1].legend(frameon=False, loc='upper right', fontsize=9)
        for ax in axes:
            grid(ax)
        finish(fig, '02_paired_seeds', ['results/test_runs.csv', 'results/paired_differences.csv'],
               'Seeds are discrete identifiers; no connecting lines. All 10 pairs included.')

        metrics = {kind: [{k: float(v) for k, v in r.items()} for r in
                         read_csv(ROOT/f'results/metrics/{kind}_seed00.csv')]
                   for kind in ('gcn', 'mlp')}
        fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.6), layout='constrained', sharex=True)
        for row, kind in enumerate(('gcn', 'mlp')):
            data = metrics[kind]
            epochs = [r['epoch'] for r in data]
            for key, color, style, label in [('eval_train', blue, '-', 'Training (eval mode)'),
                                             ('val', orange, '--', 'Validation')]:
                axes[row, 0].plot(epochs, [r[key+'_objective'] for r in data],
                                  c=color, ls=style, lw=1.7, label=label)
                axes[row, 1].plot(epochs, [100*r[key+'_accuracy'] for r in data],
                                  c=color, ls=style, lw=1.7, label=label)
            for col in range(2):
                axes[row, col].axvline(epochs[-1], color=gray, ls=':', lw=1)
                axes[row, col].set(title=f'{kind.upper()} · seed=0 · stopped at {int(epochs[-1])}',
                                   xlim=(0, 205))
                grid(axes[row, col])
            axes[row, 0].set(ylabel='CE + first-layer L2', ylim=(0, 2.1))
            axes[row, 1].set(ylabel='Accuracy (%)', ylim=(0, 100))
        axes[0, 0].legend(frameon=False, fontsize=9)
        for ax in axes[1]:
            ax.set_xlabel('Epoch')
        finish(fig, '03_seed0_training', ['results/metrics/gcn_seed00.csv',
               'results/metrics/mlp_seed00.csv'],
               'Predeclared seed=0; common 0–200 epoch scale; no MLP extension or padding.')

        data = metrics['mlp']
        subset = [r for r in data if r['epoch'] >= 17]
        x = [r['epoch'] for r in subset]
        previous = [np.mean([r['val_objective'] for r in data[int(e)-11:int(e)-1]]) for e in x]
        fig, axes = plt.subplots(3, 1, figsize=(9.2, 6.2), layout='constrained', sharex=True)
        axes[0].plot(x, [r['val_ce'] for r in subset], color=blue, marker='o', lw=1.7)
        axes[0].set(ylabel='Validation CE', title='MLP seed=0: early-stopping components in the final 6 epochs')
        axes[1].plot(x, [r['val_l2'] for r in subset], color=orange, marker='s', lw=1.7)
        axes[1].set_ylabel('First-layer L2')
        axes[2].plot(x, [r['val_objective'] for r in subset], color=blue, marker='o',
                     label='Current validation CE + L2')
        axes[2].plot(x, previous, color=orange, ls='--', marker='s', label='Previous 10 epochs: mean')
        axes[2].set(ylabel='Val. objective\n(detail)', xlabel='Epoch', xticks=x, xlim=(16.65, 22.35))
        axes[2].legend(frameon=False, fontsize=9, loc='upper right')
        for ax in axes:
            grid(ax)
            ax.ticklabel_format(style='plain', axis='y', useOffset=False)
        finish(fig, '04_mlp0_early_stop', ['results/metrics/mlp_seed00.csv'],
               'Independent vertical scales. Epoch 22: 1.93950760 > previous-ten mean 1.93944826.')

        fig, ax = plt.subplots(figsize=(9.2, 3.4), layout='constrained')
        for offset, kind, color, hatch in [(-.18, 'gcn', blue, None), (.18, 'mlp', orange, '//')]:
            values = [int(next(r['selected_epoch'] for r in rows
                               if r['model'] == kind and int(r['seed']) == s)) for s in seeds]
            ax.bar(seeds+offset, values, width=.34, color=color, hatch=hatch, label=kind.upper())
        ax.set(xlabel='Random seed', ylabel='Stopping epoch', xticks=seeds, ylim=(0, 225),
               title='Stopping positions for all 20 runs')
        ax.legend(ncol=2, frameon=False, loc='upper right')
        grid(ax)
        finish(fig, '05_stop_epochs', ['results/test_runs.csv'],
               'Limit 200; GCN seed=9 stops at 192; MLP seeds 0/1/3 stop at 22/194/193.')

    manifest = {'edition': 'English report v1.1',
                'inputs_sha256': {name: sha256(ROOT/name) for name in inputs},
                'figures': figures,
                'paper_references': {'gcn': 81.5, 'mlp': 55.1,
                                     'source': 'arXiv:1609.02907v4, p7, Tables 2/3; 100 initializations'},
                'renderer': {'matplotlib': matplotlib.__version__, 'font': 'DejaVu Sans'},
                'all_20_predeclared_runs_included': True, 'training_performed': False,
                'test_predictions_recomputed': False}
    (out/'figure_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n',
                                           encoding='utf-8', newline='\n')
    print('Generated 5 English PNG/SVG figure pairs from unchanged historical results.')


if __name__ == '__main__':
    main()
