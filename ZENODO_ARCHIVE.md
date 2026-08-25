# Zenodo reproduction archive

The Zenodo archive provides the complete reproduction files.

## Archive contents

The archive follows the repository layout and provides:

- the complete formal experiment outputs under `output/pkls/` and
  `output/csvs/`;
- the prepared county-level mobility graph under `graphs/graph1/`;
- the bundled toy-model PKL outputs and SVG products under `toyoutput/`;
- the Fig. 4h and Fig. 5b-e JPG, SVG and plotted-data CSV products;
- the code, environment specification and Bash entry points.

The formal plotting inputs comprise 168 prevalence arrays and 168 matching
`S2E_ratio` arrays across the parameter curves, together with seven
repeated-run prevalence arrays, seven repeated-run `S2E_ratio` arrays and 350
per-repeat flow summaries.

## Directory layout

After extraction, the relevant paths are:

```text
peak-shaving/
  output/
    pkls/us_preCovid_fit_empirical_P0p632_a2_fit_f1/...
    csvs/us_preCovid_fit_empirical_P0p632_a2_fit_f1/...
  toyoutput/
    scalars/PCFoverFlow_log_ED-1_RD200/...
  graphs/
    graph1/average_graph_full_daily_preCovid_workday.pkl
    graph_start_end/artifacts/...
```

Reproduce the bundled figures with:

```bash
bash bash/verify_submission.sh
bash bash/plot_toy_outputs.sh
bash bash/plot_fig4h.sh
bash bash/plot_fig5c.sh
bash bash/plot_fig5_bde.sh
```
