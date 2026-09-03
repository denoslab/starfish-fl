# Federated non-inferiority analysis of the AcT trial

This directory holds the entry-point script behind Table 1, Table 2 and Table 3
of the accompanying manuscript, together with the archived artifacts of the run
that produced them.

`run_analysis.R` partitions the trial extract by site, executes the
`r_noninferiority_meta` task scripts across the emulated 21-site network, pools
the site-level estimates, and writes the three tables and the execution log.
The task scripts themselves live at
`controller/starfish/controller/tasks/r_noninferiority_meta/scripts` and are
resolved from the location of `run_analysis.R`, so the run behaves the same way
whatever working directory it is started from.

## Data availability

The individual patient data from the Alteplase Compared to Tenecteplase (AcT)
trial are restricted and are not included in this repository or in the Zenodo
deposit. The script takes the extract as an argument and never records its path
or its file name. Provenance of the input is established inside the run, by
checking the derived per-arm counts for all four outcomes against those
published by the parent trial. The run halts if any of them disagree, so a
stale input cannot reach a table.

The per-site partitions are written to a temporary directory and removed when
the run ends, which means the archived output carries only the three tables and
the log.

## Running it

```
Rscript analysis/act-noninferiority/run_analysis.R <act_extract.csv> <output_dir>
```

The archived run used R 4.1.2 and metafor 4.6.0, both recorded in the first
lines of `output/execution-log.txt`. `jsonlite` is also required.

## Archived output

`output/` holds the artifacts of the single run behind the manuscript, not
scratch files.

- `execution-log.txt` is the raw log of that run, covering every pass, the
  provenance check against the published counts, and the pooled estimate and
  heterogeneity diagnostics for each pass.
- `Table1.csv` gives per-site results for all four outcomes, each as a
  proportion difference, its standard error, the per-arm event counts and
  denominators, and a flag recording whether the continuity correction applied.
- `Table2.csv` gives the pooled results for all four outcomes under the crude,
  fixed-effect site-stratified, and random-effects estimators, with the
  heterogeneity diagnostics.
- `Table3.csv` gives the primary outcome under the ten specifications of the
  sensitivity grid.

Re-running the script on the same extract reproduces these files, apart from
the timestamps in the log.
