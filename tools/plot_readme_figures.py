"""Export English README diagrams without training or changing historical results."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, INK, GRAY = '#245C91', '#B56A19', '#243447', '#66717E'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reviewed_scores():
    with (ROOT / 'results/test_runs.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    expected = {(model, seed) for model in ('gcn', 'mlp') for seed in range(10)}
    assert len(rows) == 20
    assert {(row['model'], int(row['seed'])) for row in rows} == expected
    assert all(row['protocol'] == 'official_code_last' for row in rows)
    assert all(row['selected_checkpoint'].endswith('/last.pt') for row in rows)
    scores = {(r['model'], int(r['seed'])): float(r['test_accuracy']) for r in rows}
    assert all(int(r['test_nodes']) == 1000 for r in rows)
    assert all(math.isclose(float(r['test_accuracy']),
                            int(r['correct_test_nodes']) / 1000) for r in rows)
    summary = json.loads((ROOT / 'results/test_summary.json').read_text(encoding='utf-8'))
    stats = {}
    for model in ('gcn', 'mlp'):
        values = [scores[model, seed] for seed in range(10)]
        mean, std = statistics.mean(values), statistics.stdev(values)
        assert math.isclose(mean, summary['models'][model]['test_mean'], abs_tol=1e-12)
        assert math.isclose(std, summary['models'][model]['test_sample_std_ddof1'], abs_tol=1e-12)
        stats[model] = {'n': 10, 'mean_percent': mean * 100, 'sample_std_pp': std * 100}
    differences = [scores['gcn', seed] - scores['mlp', seed] for seed in range(10)]
    assert all(value > 0 for value in differences)
    assert math.isclose(statistics.mean(differences),
                        summary['paired_difference']['mean_gcn_minus_mlp'], abs_tol=1e-12)
    stats['paired_mean_difference_pp'] = statistics.mean(differences) * 100
    return scores, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    scores, stats = reviewed_scores()
    # Use a disposable font cache rather than adding machine-specific cache files.
    with tempfile.TemporaryDirectory(prefix='gcn_readme_mpl_') as cache:
        os.environ['MPLCONFIGDIR'] = cache
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        from matplotlib.patches import FancyBboxPatch
        from matplotlib.lines import Line2D

        plt.rcParams.update({
            'font.family': 'DejaVu Sans', 'font.size': 11,
            'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': GRAY,
            'ytick.color': GRAY, 'axes.edgecolor': GRAY,
            'axes.spines.top': False, 'axes.spines.right': False,
            'figure.facecolor': 'white', 'savefig.facecolor': 'white',
            'svg.fonttype': 'path', 'svg.hashsalt': 'gcn-cora-readme-v1',
        })
        outputs = {}

        def save(fig, name):
            for extension in ('svg', 'png'):
                path = out / f'{name}.{extension}'
                metadata = {'Date': None} if extension == 'svg' else {}
                fig.savefig(path, dpi=180, metadata=metadata)
                if extension == 'svg':
                    path.write_bytes(b'\n'.join(line.rstrip() for line in
                                               path.read_bytes().splitlines()) + b'\n')
                outputs[path.name] = sha256(path)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(11.6, 3.35))
        fig.subplots_adjust(left=.015, right=.985, bottom=.02, top=.98)
        ax.set(xlim=(0, 12), ylim=(0, 3.5))
        ax.axis('off')
        ax.text(.12, 3.24, 'Two graph convolutions, one shared graph',
                fontsize=15, fontweight='bold')

        def box(x, width, title, formula, shape, color=GRAY, fill='#F4F6F8'):
            ax.add_patch(FancyBboxPatch((x, 1.17), width, 1.24,
                         boxstyle='round,pad=0.04,rounding_size=0.07',
                         facecolor=fill, edgecolor=color, linewidth=1.3))
            center = x + width / 2
            ax.text(center, 2.13, title, ha='center', fontsize=11, fontweight='bold')
            ax.text(center, 1.74, formula, ha='center', fontsize=14)
            ax.text(center, 1.38, shape, ha='center', fontsize=10, color=GRAY)

        box(.15, 2.05, 'Node features', r'$X$', '2,708 × 1,433')
        box(2.95, 2.65, 'GCN layer 1 + ReLU',
            r'$H=\mathrm{ReLU}(\hat{A}X_dW_0)$', '1,433 → 16', BLUE, '#EEF4FA')
        box(6.4, 2.65, 'GCN layer 2', r'$Z=\hat{A}H_dW_1$', '16 → 7', BLUE, '#EEF4FA')
        box(9.8, 2.05, 'Class logits', r'$Z$', '2,708 × 7')
        for left, right, dropout in [(2.2, 2.95, True), (5.6, 6.4, True), (9.05, 9.8, False)]:
            ax.annotate('', xy=(right-.08, 1.8), xytext=(left+.07, 1.8),
                        arrowprops={'arrowstyle': '->', 'color': INK, 'lw': 1.3})
            if dropout:
                ax.text((left+right)/2, .91, 'Dropout\np = 0.5', ha='center', fontsize=9)
        ax.text(6, 2.75, 'Same normalized adjacency in both layers (self-loops included)',
                ha='center', fontsize=11, color=BLUE)
        for center in (4.275, 7.725):
            ax.annotate('', xy=(center, 2.44), xytext=(center, 2.65),
                        arrowprops={'arrowstyle': '->', 'color': BLUE, 'lw': 1.1})
        ax.text(.15, .42, r'$\hat{A}=\tilde{D}^{-1/2}(A+I)\tilde{D}^{-1/2}$', fontsize=13)
        ax.text(5.4, .43, 'Matched MLP: same widths and dropout, without graph propagation.', fontsize=10)
        ax.text(.15, .07, 'Row-normalized X · no bias · 23,040 parameters · dropout disabled at evaluation',
                fontsize=10, color=GRAY)
        save(fig, 'readme_gcn_architecture')

        fig, ax = plt.subplots(figsize=(9.6, 4.9))
        fig.subplots_adjust(left=.105, right=.965, top=.77, bottom=.18)
        fig.text(.105, .93, 'GCN vs. feature-only MLP on Cora', fontsize=17, fontweight='bold')
        fig.text(.105, .865, 'Fixed public split · 140 training labels · 10 predeclared seeds per model',
                 fontsize=11, color=GRAY)
        for x, (model, color, marker) in enumerate([('gcn', BLUE, 'o'), ('mlp', ORANGE, 's')]):
            values = [100*scores[model, seed] for seed in range(10)]
            offsets = [-.11 + seed*.024 for seed in range(10)]
            ax.scatter([x + offset for offset in offsets], values, marker=marker,
                       color=color, s=38, edgecolors='white', linewidths=.55, zorder=4)
            mean, std = stats[model]['mean_percent'], stats[model]['sample_std_pp']
            ax.errorbar(x-.29, mean, yerr=std, fmt='D', color=color,
                        markersize=7, capsize=6, elinewidth=1.7, zorder=5)
            ax.text(x, 94, f'{mean:.2f}% ± {std:.2f} pp', ha='center',
                    fontsize=12, color=color, fontweight='bold')
        ax.annotate('seed 0: 21.6%\nretained in summary',
                    xy=(1-.11, 21.6), xytext=(.36, 10), fontsize=10,
                    arrowprops={'arrowstyle': '-', 'color': GRAY, 'lw': 1}, color=GRAY)
        ax.set(xlim=(-.6, 1.6), ylim=(0, 100), xticks=[0, 1],
               xticklabels=['GCN', 'MLP'], yticks=range(0, 101, 20), ylabel='Test accuracy (%)')
        ax.tick_params(axis='x', length=0, labelsize=13, pad=10)
        ax.grid(axis='y', color='#DCE2E8', linewidth=.65)
        ax.set_axisbelow(True)
        legend = [Line2D([], [], color=GRAY, marker='o', ls='none', markersize=5,
                         label='Individual runs (all 20)'),
                  Line2D([], [], color=GRAY, marker='D', ls='-', markersize=5,
                         label='Mean ± sample std. (ddof=1)')]
        fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(.54, .005),
                   frameon=False, ncol=2, fontsize=10)
        fig.text(.105, .085, 'Error bars show seed variability on this split, not confidence intervals.',
                 fontsize=10, color=GRAY)
        save(fig, 'readme_gcn_vs_mlp')

    sources = ['results/test_runs.csv', 'results/test_summary.json',
               'src/gcn_model.py', 'src/gcn_layer.py', 'src/mlp_model.py',
               'configs/cora_official_last.json', 'tools/plot_readme_figures.py']
    manifest = {
        'purpose': 'English README figures; no new training or test evaluation',
        'inputs_sha256': {name: sha256(ROOT/name) for name in sources},
        'outputs_sha256': outputs,
        'statistics': stats,
        'all_20_predeclared_runs_included': True,
        'error_bars': 'sample standard deviation, ddof=1; not confidence intervals',
        'architecture': 'GCN.forward; dropout before each graph convolution; eval returns logits',
        'renderer': {'matplotlib': matplotlib.__version__, 'font': 'DejaVu Sans'},
    }
    (out/'readme_figure_manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n', encoding='utf-8', newline='\n')
    print('Verified 20 source rows and summaries; exported 2 SVG + 2 PNG figures and manifest.')


if __name__ == '__main__':
    main()
