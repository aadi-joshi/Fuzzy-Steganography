# LaTeX Paper Source

Elsevier `elsarticle` manuscript for the journal submission. This is separate from the executable research code; all experiment figures are shared with the README via `figures/`.

## Build

From the repository root, ensure figures exist:

```bash
python scripts/regenerate_figures.py
```

Compile the paper:

```bash
cd docs/paper
tectonic main.tex
# or: pdflatex main.tex
```

Output PDF is written locally as `docs/paper/main.pdf` (gitignored).

## Figure assets

`main.tex` loads figures from `../../figures/`. No duplicate copies are stored under `Steg/`.

| Paper figure | Source in `figures/` |
|--------------|----------------------|
| `architecture.jpeg` | Paper-specific diagram (committed) |
| `smooth_00000.png`, etc. | Sample synthetic covers (`regenerate_figures.py`) |
| `psnr_vs_bpp.png`, `ssim_vs_bpp.png` | V2 pipeline plots or `regenerate_figures.py` |
| `detection_rates.png` | V2 steganalysis plots |
| `roc_bpp_0p05.png`, `roc_bpp_0p2.png` | ROC plots at selected bpp levels |
| `ablation.png`, `complexity.png` | Ablation and complexity analysis |
| `boss_psnr.png`, `bows2_psnr.png`, `mirflickr_psnr.png` | Real-dataset validation |
| `real_datasets_rs.png`, `srm_auc_tpr.png` | Cross-dataset steganalysis |

## Legacy `Steg/` directory

The old `Steg/` folder contained duplicate PNGs, Word/PDF exports, and a copy of `main.tex` for local authoring. It is gitignored. You may delete it locally; use `docs/paper/` and `figures/` instead.
