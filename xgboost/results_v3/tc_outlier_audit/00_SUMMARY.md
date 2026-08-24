# Tc Outlier Audit — Summary (Frozen Protocol v3)

Read-only diagnostic stage. No retraining, resplitting, retuning, or prediction regeneration — all values reused from `results_v3/variant_test_predictions.csv` and `artifacts/benchmark_split_v3.csv`. Fingerprints are recomputed from saved canonical SMILES (same Morgan r2/2048 protocol as Scripts 01/03) purely for similarity diagnostics.

**Single frozen split.** Descriptive only — no causal claims, no statistical-significance claims.

## Cohorts
- Primary: top 50 Tc test molecules by |error| under XGB-D (50 molecules)
- Secondary union: top 20 Tc test molecules by |error| under EACH variant, unioned (33 molecules)
- Master (union of both): 54 unique molecules — full diagnostics in `02_master_outlier_table.csv`
- Focused report: top 20 of the primary cohort — `03_top20_focused_report.md`

## Cross-variant consistency (master cohort)
'Elevated' threshold = each variant's own P75 of |error| over its full Tc test set (n=209). See `01_context_error_percentiles.csv` for P50/75/90/95/99/Max per variant.
- SYSTEMATIC: 38 / 54
- MODEL-SENSITIVE: 16 / 54
- NONE: 0 / 54

## Domain of applicability (master cohort, all-train fingerprint similarity)
- LOW: 13 / 54
- MEDIUM: 28 / 54
- HIGH: 13 / 54

## Near-neighbor label consistency
- Mean |Tc label gap| to nearest Tc-labeled-train neighbor: 98.18 K (median 58.74 K, max 817.87 K)
- Flagged 'close neighbor, large label gap' (sim>=0.6 to a Tc-labeled train molecule AND gap > XGB-D's P75 |error|): 10 / 54 — these are candidates for a data/labeling review rather than a model-capacity issue.

## Cluster hotspots (Tc test clusters with >=2 members, top 5 by mean XGB-D |error|)
- cluster 143: n_test_tc=3, bench_total=3, mean |error|=65.8388 K, max |error|=113.041 K (full table: `04_cluster_hotspots.csv`)
- cluster 102: n_test_tc=6, bench_total=6, mean |error|=27.5264 K, max |error|=57.2785 K (full table: `04_cluster_hotspots.csv`)
- cluster 106: n_test_tc=9, bench_total=9, mean |error|=26.4442 K, max |error|=89.2393 K (full table: `04_cluster_hotspots.csv`)
- cluster 307: n_test_tc=2, bench_total=2, mean |error|=25.9247 K, max |error|=31.7068 K (full table: `04_cluster_hotspots.csv`)
- cluster 224: n_test_tc=3, bench_total=3, mean |error|=23.7129 K, max |error|=51.5296 K (full table: `04_cluster_hotspots.csv`)

## Context: |error| percentiles per variant (full Tc test set)

variant   n     P50     P75     P90     P95      P99      Max
  XGB-A 209 11.0042 23.9259 50.0774 70.1041 113.1373 133.0551
  XGB-B 209 11.3592 22.6085 41.4174 56.2782 100.6677 144.9265
  XGB-C 209 10.9718 19.6279 36.5046 53.8862  99.0431 123.7222
  XGB-D 209 10.7506 19.0537 34.2306 49.3610 104.0187 126.2973

_See `plots/tc_outlier_diagnostic.png` for DOA vs error, colored by consistency class._