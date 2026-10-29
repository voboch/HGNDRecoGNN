# Progress report, 21–29 October 2026

The report is a dated synthesis, not a new experiment. `data.json` pins 31
numerical and documentary sources by SHA256. It separates manuscript MC-truth
results, earlier frozen transfer and the later cross-fitted training campaign.
`slurm_status.json` records the read-only 29 October accounting query. The
purity-scan NO-GO is included as a preflight finding, not a completed scan.
The subsequent background-correction draft is archived from commit 020e766;
ongoing edits to that other task are not consumed or overwritten.

Rebuild the figures with a Python environment containing NumPy and Matplotlib,
then the document with ReportLab and pypdf:

```bash
MPLCONFIGDIR=/private/tmp/hgnd-matplotlib XDG_CACHE_HOME=/private/tmp/hgnd-cache \
  /opt/homebrew/Caskroom/miniconda/base/envs/pyg/bin/python \
  reports/progress_20261021_20261029/plot.py
/Users/vovvy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  reports/progress_20261021_20261029/build.py
```

The scripts resolve the repository from their own path. The document builder
refuses changed source files; use the recorded source revision for a historical
rebuild. The paper sources are revision e6afb98. `build.py` supplies both the
Markdown text in the parent reports directory and the PDF, so their numerical
content is shared. Figures are embedded at their natural print size as vector
PDF content; PNGs are review copies. Figures 3 and 4 are unmodified, hash-bound
exports of the verified manuscript build.

The local paths to fonts and runtimes are environment details. No raw files,
prediction scores, thresholds, training, fitting or inference are regenerated.
`validation.json` records the final output, source and render checks. The
manuscript and other agents' purity-scan work are outside this report's edits.
