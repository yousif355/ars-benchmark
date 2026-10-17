# Figures 2 to 4

`scripts/make_table3_figs.py` writes each figure as a TikZ picture:

| File | Figure |
|---|---|
| `fig2_acc_bars.tex` | Fig. 2, accuracy by condition and severity |
| `fig3_acc_heat.tex` | Fig. 3, accuracy heatmap |
| `fig4_ece_heat.tex` | Fig. 4, ECE15 after T* heatmap (red: above the 0.20 collapse threshold) |

To compile one, put it in a LaTeX document with this preamble and
`\input{fig2_acc_bars.tex}` in the body:

```
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\definecolor{collgood}{RGB}{27,120,55}
\definecolor{collmid}{RGB}{255,191,0}
\definecolor{collbad}{RGB}{178,24,43}
```

The values behind the figures are in `results/figs_2_4_values.csv`. Fig. 5 is
drawn from the bins in `results/fig5_reliability_bins.json`.
