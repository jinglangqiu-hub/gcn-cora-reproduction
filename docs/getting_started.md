# Getting started

[Back to the project overview](../README.md)

## CPU setup and a single run

Use Python 3.11 and an isolated environment. These instructions are for a new installation; public-package checks and their platform scope are recorded in [publication_validation.json](publication_validation.json).

```bash
git clone https://github.com/jinglangqiu-hub/gcn-cora-reproduction.git
cd gcn-cora-reproduction
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m tools.prepare_data
python -m checks.verify_models
python -m tools.train --model gcn --seed 0 --run-dir runs/gcn_seed00
python -m tools.evaluate --run-dir runs/gcn_seed00 --output-dir evaluations/gcn_seed00
```

Every run and evaluation needs a new output directory; existing outputs are not overwritten. Training uses training labels and validation metrics. Evaluation separately scores the stopping-time `last.pt` on the test set. New runs do not replace the historical results. Bitwise equality is not promised on new platforms; see [reproducibility scope](reproducibility.md).

To run both models for all 10 seeds:

```bash
python -m tools.run_all --output-dir runs/new_batch
```

[The frozen configuration](../configs/cora_official_last.json) defines the experiment. The CLI allows model, seed, data location, and a 1–200 epoch limit; model hyperparameters remain fixed. Short runs check the workflow and do not replace full formal experiments. See [protocol.md](protocol.md) for the exact early-stopping and checkpoint rules.

## Training visualization

The [online viewer](https://jinglangqiu-hub.github.io/gcn-cora-reproduction/) displays all 20 completed historical runs. Select a run, move the slider, or hover over a curve to inspect training/validation losses and accuracy, stopping reason, selected epoch, and CPU time. It reads a static snapshot, does not execute training, and does not read your computer. Curve accuracy is validation accuracy; final test results are in [test_runs.csv](../results/test_runs.csv).

For local live monitoring, start this command, then train in another terminal:

```bash
python -m tools.monitor_training --runs-root runs --port 8765
```

Open `http://127.0.0.1:8765/` in your browser. The viewer reads newly written local metrics once per second. The website uses no third-party scripts or analytics.

## Recompute statistics and figures

No weights or retraining are needed to recompute the historical summaries:

```bash
python -m tools.summarize
python -m tools.plot_results --output-dir artifacts/plots
```

The report plotter reads `results/`, retains every seed, and exports five PNG/SVG figure pairs plus source hashes. Its Chinese labels use Microsoft YaHei on Windows or Noto CJK on Linux; install a system Noto CJK font if necessary. Existing figures are already readable in the repository, and plotting fonts do not affect training. Report PDFs are available in the [release](https://github.com/jinglangqiu-hub/gcn-cora-reproduction/releases/tag/v0.1.0); report generation is not a training dependency.

To recreate the English homepage figures in a separate directory:

```bash
python -m tools.plot_readme_figures --output-dir artifacts/readme_figures
```

The output includes PNG/SVG pairs and a manifest of source, script, and image hashes. The committed versions are in `docs/figures/`.

To recreate the five figures in the [English report](report_en.md):

```bash
python -m tools.plot_report_en --output-dir artifacts/plots/report_en
```

These use the same historical CSVs as the Chinese report. The committed English PNG/SVG pairs and their source hashes are in `docs/figures/en/`; the [Chinese report](report.md) and its original figures remain available.

To typeset the English report as PDF:

```bash
python -m pip install -r requirements-report.txt
python -m tools.build_report_en_pdf --output artifacts/plots/report_pdf/GCN_Cora_Report_v1.1_public_en.pdf
```

The PDF builder reads `docs/report_en.md` and its English PNG figures, embeds the bundled ReportLab Vera fonts, preserves report links, and writes a companion source/hash manifest. The published [English PDF](https://github.com/jinglangqiu-hub/gcn-cora-reproduction/releases/download/v0.1.0/GCN_Cora_Report_v1.1_public_en.pdf) and [checksum](https://github.com/jinglangqiu-hub/gcn-cora-reproduction/releases/download/v0.1.0/SHA256SUMS_en.txt) are available in Release v0.1.0.

## Repository map

| Directory | Contents |
| --- | --- |
| `src/` | Two-layer GCN/MLP and tensor loading |
| `tools/` | Data preparation, training, evaluation, summaries, plotting, and live monitoring |
| `results/` | Historical test results and all 20 per-epoch metric CSVs |
| `site/` | GitHub Pages viewer and static historical data |
| `docs/` | Report, figures, protocol, provenance, rerun scope, and validation |
| `artifacts/` | [Download and verification instructions](../artifacts/README.md) for selected weights, predictions, and logs |

Data comes from the [fixed reference commit](https://github.com/tkipf/gcn/tree/39a4089fe72ad9f055ed6fdb9746abdcfebc4d81); the downloader verifies the recorded raw hashes before loading. See [data notes](../data/README.md) and [source mapping](provenance.md).
