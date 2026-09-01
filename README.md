# TabPFN for TDC ADMET datasets

Benchmarking [TabPFN](https://doi.org/10.1038/s41586-024-08328-6) — a
transformer-based foundation model for tabular data — on the
[Therapeutics Data Commons](https://tdcommons.ai/) (TDC) ADMET benchmark group.

TabPFN (Tabular Prior-data Fitted Network) is pre-trained on millions of
synthetic datasets and predicts on an entire dataset in a single forward pass
via in-context learning. Gradient-boosted decision trees usually beat deep
learning on tabular chemical data; this repo tests whether TabPFN closes that
gap on molecular property prediction.

**Write-up:** [TabPFN for chemical datasets](https://jonswain.github.io/tabpfn-for-chemical-datasets/)
(the post is also included here as `2025-01-22-TabPFN-for-chemical-datasets.md`).

## Approach

- **Datasets** — the datasets in the TDC `admet_group` benchmark.
- **Features** — the 210 RDKit descriptors, computed per molecule and clipped to
  the fp16 range. MACCS keys and 500-bit ECFP were also tried and performed
  worse.
- **Models** — `TabPFNClassifier` for binary endpoints, `TabPFNRegressor` for
  regression; 5 seeds per dataset, scored with the TDC group's
  `evaluate_many`.
- **Limits** — run on a laptop, so datasets with more than ~1,800 training rows
  were skipped for memory. TabPFN itself caps training at 10,000 rows / 500
  features.

## Results

With RDKit descriptors as features, TabPFN lands in the TDC leaderboard top 10
for every dataset run except `Clearance_Hepatocyte_Az`, and is **1st on
`Clearance_Microsome_Az`**, 2nd on `Caco2_Wang`, `Pgp_Broccatelli` and
`Bbb_Martins`, and 3rd on `Vdss_Lombardo`.

| Dataset | Size | Task | Metric | TabPFN | TDC best (Jan 2025) | Rank |
| --- | --- | --- | --- | --- | --- | --- |
| Caco2_Wang | 906 | Regression | MAE | 0.282 ± 0.005 | 0.276 ± 0.005 | 2nd |
| HIA_Hou | 578 | Classification | AUROC | 0.987 ± 0.001 | 0.990 ± 0.002 | 5th |
| Pgp_Broccatelli | 1218 | Classification | AUROC | 0.936 ± 0.004 | 0.938 ± 0.002 | 2nd |
| Bioavailability_Ma | 640 | Classification | AUROC | 0.735 ± 0.016 | 0.753 ± 0.000 | 5th |
| Bbb_Martins | 2030 | Classification | AUROC | 0.917 ± 0.003 | 0.920 ± 0.006 | 2nd |
| Vdss_Lombardo | 1130 | Regression | Spearman | 0.693 ± 0.004 | 0.713 ± 0.007 | 3rd |
| Cyp2D6_Substrate_Carbonmangels | 667 | Classification | AUPRC | 0.714 ± 0.009 | 0.736 | 6th |
| Cyp3A4_Substrate_Carbonmangels | 670 | Classification | AUROC | 0.641 ± 0.004 | 0.667 ± 0.019 | 7th |
| Cyp2C9_Substrate_Carbonmangels | 669 | Classification | AUPRC | 0.400 ± 0.013 | 0.441 ± 0.033 | 10th |
| Half_Life_Obach | 667 | Regression | Spearman | 0.546 ± 0.013 | 0.576 ± 0.025 | 6th |
| Clearance_Microsome_Az | 1102 | Regression | Spearman | 0.632 ± 0.006 | 0.630 ± 0.010 | 1st |
| Clearance_Hepatocyte_Az | 1213 | Regression | Spearman | 0.391 ± 0.004 | 0.536 ± 0.02 | >10th |
| Herg | 655 | Classification | AUROC | 0.850 ± 0.002 | 0.880 ± 0.002 | 6th |
| Dili | 475 | Classification | AUROC | 0.910 ± 0.005 | 0.925 ± 0.005 | 6th |

Leaderboard ranks are as of January 2025; larger datasets are still to be run.
See the write-up for discussion.

## Usage

Requires [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html).

```bash
git clone https://github.com/jonswain/tabpfn-tdc.git
cd tabpfn-tdc
conda env create -f environment.yml
conda activate tabpfn-tdc
python submission.py 2>&1 | tee -a log.txt
```

Runs on CUDA, Apple MPS, or CPU (auto-detected).

## References

- Hollmann et al., *Accurate predictions on small data with a tabular foundation
  model*, Nature (2024). <https://doi.org/10.1038/s41586-024-08328-6>
- Therapeutics Data Commons — ADMET benchmark group. <https://tdcommons.ai/>

## License

[MIT](LICENSE)
