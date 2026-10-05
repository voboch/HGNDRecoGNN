# n/p diagnostic scripts

These small programs are retained for focused historical diagnostics. Set
`NP_PRIMARY_DIR` to the directory containing `*_prim.csv` before running one:

```bash
NP_PRIMARY_DIR=/path/to/npsample python results/np_primaries/spec.py
```

They no longer hard-code a private temporary path or require pandas' optional
`pyarrow` CSV engine. Cross-file event counts treat `Row` as file-local.

For reproducible plot production and event-clustered uncertainties, use the
authoritative workflow:

```bash
python scripts/np_ratio_smash_check.py \
  --input-dir /path/to/npsample \
  --output-dir results/np_ratio_smash_check \
  --pdf-output output/pdf/np_smash_check_reproduction.pdf
```

The checked-in text files predate the event-clustered analysis and remain as
historical records; do not use their raw Poisson "sigma" values as final
physics significances.
