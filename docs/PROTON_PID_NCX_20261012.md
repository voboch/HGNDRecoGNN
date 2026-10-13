# Proton reconstruction and PID for the SMASH samples on ncx

Status: 12 October 2026. Selection code and synthetic checks are implemented.
The installed ncx dictionaries and one real SMASH DST still need a batch smoke.
The supplied route reaches the JINR gateway, which refused non-interactive
authentication with `Permission denied (keyboard-interactive)`.

## Inputs and the required data level

Proton PID starts from a reconstructed **DST**, with tracks, primary vertices,
TOF matches and calibrated energy-loss estimates. HGND `_hits.csv`, `_vacs.csv`
and `_prim.csv` exports contain neither the central charged tracks nor their
TOF/PID response and cannot be used to reconstruct the proton arm.

| Input content | Next step |
|---|---|
| `bmndata` + `BmnGlobalTrack` + `StsVector` | inspect, extract and select |
| transport/digitisation ROOT file with tracker digits and TOF MC points | installed `run_reco_bmn.C`, then inspect the DST |
| generator-only SMASH particles | full BM@N transport/digitisation is required first |
| HGND CSV exports | return to the original ROOT production |

The October 2026 [Start Guide](https://bmn.jinr.int/bmnroot_start_guide/)
describes the chain in §§4.3–4.6. The [DIGI guide](https://bmn.jinr.int/bmnroot_digi_format/)
states that simulated TOF-400/700 hits are made from TOF MC points; the absence
of experimental-style TOF digit branches does not imply missing simulated TOF.
The [DST guide](https://bmn.jinr.int/bmnroot_dst_format/), dated 9 October 2026,
describes `dev` commit `027633c4`. An older installed build must be inspected;
the reader exits if its track/vertex interface differs from this adapter.

## PID from Parfenov slides 22–23

Source: `/Users/vovvy/Desktop/Nucleus2025_Parfenov.pdf`, two supplied pages labelled
22 and 23. The fit constants are not given on the slides and must be measured
in this production. The curves cannot be calibrated by copying the plot axes.

For species hypothesis $s\in\{p,\pi\}$,

$$
f_s(\beta\gamma)=\frac{a_1}{\beta^{a_4}}
\left[a_2-\beta^{a_4}-\ln\left(a_3+(\beta\gamma)^{-a_5}\right)\right],
\qquad \beta\gamma=\frac{p}{m_s},\quad
\beta^2=\frac{p^2}{p^2+m_s^2}.
$$

Fit Gaussian widths in momentum slices of the relative energy-loss residual
and of the TOF mass-squared distribution:

$$
x_s=\frac{(dE/dx)_{\rm meas}-f_s}{f_s\,\sigma_s^{\rm rel}(dE/dx)},
\qquad
y_s=\frac{m^2-m_s^2}{\sigma_s(m^2)},
\qquad r_s=\sqrt{x_s^2+y_s^2}.
$$

The slide proton selection is positive charge, $r_p<2$ and $r_\pi>3$.
These inherited cut values trade proton efficiency against pion rejection; the
code writes a two-dimensional purity/efficiency scan around them before they
are adopted for the SMASH analysis. In an ideal independent two-dimensional
Gaussian, $r_p<2$ keeps $1-\exp(-2)=86.5\%$, before the pion veto and other
losses. It is not the one-dimensional Gaussian “two sigma” fraction.

The implementation fits one **MC-labelled development calibration**, then
freezes it across the three symmetry-potential samples. MC identity is used
only to fit templates and evaluate PID/efficiency. No MC label or mother ID
enters the applied proton selection. This is a simulation workflow; a
truth-free experimental calibration would require clean control bands or a
joint species-mixture fit.

## DST getters and checks

| Quantity | Implementation | Reason / failure check |
|---|---|---|
| signed rigidity | `GetParamFirst()->GetQp()` | sign gives reconstructed charge; $p/|q|=1/|q/p|$ |
| $dE/dx$ | `BmnGlobalTrack::GetDedx()` | guide §5.2: filled by `BmnDedx` only for Run 8, zero otherwise |
| primary association | `StsVector[GetInnerTrackIndex()/GetGemTrackIndex()]->FromVertex()` | reconstructed track used in the `MpdVertex.` fit |
| TOF match | `GetTof1HitIndex()` / `GetTof2HitIndex()` | common matching hypothesis, not a proton-specific match |
| $m^2$ | `GetMass2(1/2)` | requires a valid matched hit; sentinel `-1000` is rejected |
| older $m^2$ fallback | $p^2(1/\beta^2-1)$, $\beta=L/(ct)$ | use **matched TOF hit** `GetLength()` and `GetTimeStamp()`; global-track length ends at the tracker |
| MC association for QA | inner `GetTrkID()` + `BmnMCInfo.GetMCTracks()` | global `GetRefIndex()` is always `-1` in the documented format |
| event denominator | every `bmndata` entry | includes failed vertex and zero-track events |
| HGND join | `(source_id, entry)` and `Row == entry`, zero-based | matches `Convert2Csv.C`, which writes `Row = ev` from a loop starting at 0; `entry+1` would join each event to the next one's hits |

The current [BmnGlobalTrack header](https://git.jinr.ru/nica/bmnroot/-/blob/dev/base/data/BmnGlobalTrack.h)
confirms the no-argument TOF getters are the common hypothesis, and TOF IDs
for `GetMass2` are 1 and 2. Do not use `GetQ()` for the charge **sign**: it is
the energy-loss charge-magnitude estimate. Do not count a TOF hit with
`IsUsed()` as a match, or substitute the global track's `GetNHits()` for the
inner hit count; the guide identifies both as unsafe for that purpose.

The first smoke must show nonzero calibrated $dE/dx$ and matched TOF counts.
If $dE/dx$ is unfilled, the combined slide PID is unavailable until the
production's digitisation and `BmnDedx` task are validated. The code does not
silently substitute a TOF-only selection. Simulated strip amplitudes must also
be checked for a physical species dependence before fitting a Bethe curve.

## Code and outputs

| File | Role |
|---|---|
| `scripts/run_proton_pid.py` | inspect, manifest, extract, calibrate, select (renamed from `proton_pid.py`: a module of that name shadowed the `proton_pid/` package whenever `scripts/` preceded the repo root on `sys.path`, which broke collection under pytest and would have broken imports on the cluster) |
| `proton_pid/root_io.py` | installed BmnRoot dictionary adapter and event/MC ledgers |
| `proton_pid/core.py` | Bethe fit, measured slice widths and reconstructed-only PID |
| `proton_pid/plots.py` | fitted means/widths, reconstructed PID bands, cut tradeoff |
| `scripts/reconstruct_smash_bmn.py` | optional transport/DIGI → DST stage |
| `slurm/proton_pid.ncx.sbatch` | one input file per extraction/reconstruction task |
| `slurm/proton_pid_analysis.ncx.sbatch` | calibration or per-file selection on compute nodes |
| `tests/test_proton_pid.py` | slide equations, truth independence, missing-data checks and zero-track denominator |

Extraction writes `_tracks.csv.gz`, `_events.csv.gz`, `_truth.csv.gz` and
`_meta.json`. The tracks preserve both TOF arms, but each calibration/selection
uses **one explicit arm** (`400` or `700`). A union or combined estimator
requires a separately validated rule; counting one track twice is forbidden.
The truth ledger contains **all primary protons and neutrons**, including
unreconstructed particles, for later reconstruction-efficiency denominators.

Selection writes `protons.csv.gz`, all scored tracks, an event ledger including
zero selected protons, the effective cut/calibration record, and PID QA plots.
The selected yield is a candidate yield. It is not efficiency-corrected or
background-subtracted. MC PID efficiency is explicitly conditional on an
already reconstructed, primary-associated, TOF-matched track inside calibration
support. Full reconstruction efficiency still needs the generated-primary ledger
and a common truth fiducial definition. MC clones must be deduplicated by
`(source_id,entry,mc_index)` for a true-proton efficiency numerator.

## ncx deployment and smoke

The supplied connection is:

```bash
ssh -J boch@sshgate.jinr.ru boch@ncx102.jinr.ru
```

The following historical paths come from the repository's previous converter;
they have **not** been verified through the new login:

```text
/scratch2/shabanov/Simulation/Smash_zeroSpot_NoAbsorber/zeroSpot_0_200
/scratch2/shabanov/Simulation/Smash_defaultSpot_NoAbsorber/defaultSpot_0_200
/scratch2/shabanov/Simulation/Smash_bigSpot_NoAbsorber/bigSpot_10_200
/nica/bmn1/karpushkin/bmn/bmnrootVersions/bmnroot_dev/build/config.sh
```

Confirm the production is the intended SMASH generation before preparing the
manifest. Retain original ROOT files, existing CSVs and their row mapping. Use
a fresh output directory owned by the current ncx account; historical
`/scratch2/vboch` ownership is not assumed for login `boch`.

Copy the new code to an HGNDRecoGNN checkout on ncx after authenticating:

```bash
rsync -a -e 'ssh -J boch@sshgate.jinr.ru' proton_pid \
  boch@ncx102.jinr.ru:/path/to/HGNDRecoGNN/
rsync -a -e 'ssh -J boch@sshgate.jinr.ru' \
  scripts/run_proton_pid.py scripts/reconstruct_smash_bmn.py \
  scripts/plot_b_full.py \
  boch@ncx102.jinr.ru:/path/to/HGNDRecoGNN/scripts/
rsync -a -e 'ssh -J boch@sshgate.jinr.ru' slurm/proton_pid*.ncx.sbatch \
  boch@ncx102.jinr.ru:/path/to/HGNDRecoGNN/slurm/
```

From that checkout on ncx, make a stable file manifest with the actual DST
filename pattern. This reads file metadata only:

```bash
python3 scripts/run_proton_pid.py manifest \
  --input-dir /verified/path/to/SMASH/DSTs --pattern '*dst*.root' \
  --sample 'SMASH U_sym=18 MeV; production tag; geometry tag' \
  --output /new/output/path/defaultSpot.manifest.txt

mkdir -p out
export PROJECT_DIR="$PWD"
export MANIFEST=/new/output/path/defaultSpot.manifest.txt
export OUTPUT_DIR=/new/output/path/proton_smoke
export BMNROOT_CONFIG=/verified/path/to/bmnroot/build/config.sh
export ROOT_PYTHON=/path/to/python/compatible/with/this/ROOT
export STAGE=inspect MAX_EVENTS=100
sbatch --time=00:10:00 --export=ALL slurm/proton_pid.ncx.sbatch
```

Use ncx's verified default partition/account, or pass them explicitly. HSE
`rocky`, `type_d` and `proj_1855` settings do not apply to ncx. Source the actual
BmnRoot build's `config.sh`; PyROOT must match that ROOT build. Statistical
analysis needs numpy, pandas, scipy and matplotlib and can use a separate
`ANALYSIS_PYTHON`. No job installs software or modifies the BmnRoot environment.

After the inspection passes, smoke-test extraction of the same file in a short
allocation, then fit/calibrate and select on exported development/closure
tables. Only then submit full extraction arrays. For example, after the smoke
job has completed successfully:

```bash
export STAGE=extract MAX_EVENTS=0 OUTPUT_DIR=/new/output/path/defaultSpot_tracks
sbatch --array=0-199%8 --export=ALL slurm/proton_pid.ncx.sbatch
```

The array upper bound must equal **manifest line count minus one**, not a
historical count. The `%8` throttle is an example to limit shared-disk readers;
use the site's allowed concurrency. A file is never silently overwritten.
Incomplete `.part` outputs identify failed extraction; retry with a fresh
prefix after diagnosing the error. Existing complete products retain their
original calibration and input provenance.

## Calibration and selection

Freeze file-level development and test manifests before fitting. The
calibrator uses reconstructed, positive, vertex-associated, MC-labelled proton
and positive-pion tracks. It fits the Bethe curve and Gaussian widths, saves
mean residuals and the $dE/dx$–$m^2$ correlation, and plots the fitted quantities.
Choose momentum bins from observed occupancy; at least five populated slices
are needed for the five-parameter fit. Internal gaps cause an error. PID rejects
momenta outside the common measured calibration centres.

Commands **inside a Slurm allocation**:

```bash
python scripts/run_proton_pid.py calibrate \
  --tracks /development/export/00000_tracks.csv.gz /development/export/00001_tracks.csv.gz \
  --tof 700 --p-edges 0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5 \
  --min-slice-tracks 100 \
  --sample-description 'development files only; production/geometry/build recorded here' \
  --output /new/output/path/calibration700/calibration.json

python scripts/run_proton_pid.py select \
  --tracks /test/export/00002_tracks.csv.gz \
  --events /test/export/00002_events.csv.gz \
  --calibration /new/output/path/calibration700/calibration.json \
  --output-dir /new/output/path/protons700/00002
```

The 0.5 GeV/$c$ bins and 100 entries above are **example conventions**, not tuned
SMASH values. Wider bins improve template statistics but wash out momentum
dependence; the calibration slice-count/width tables and plots show that
tradeoff. Calibrate TOF-400 separately if it is used. The supplied analysis
batch script accepts `STAGE=calibrate/select` and a `TABLE_MANIFEST`.

Applying a calibration to its own source files is rejected unless
`--allow-calibration-overlap` explicitly marks a development closure. Across
0/18/90 MeV use the same frozen curve and widths. Refitting each Spot sample
can remove the response the hybrid observable is intended to measure.

No extra numerical track-quality or DCA cuts are presumed from the slides.
`FromVertex()` supplies the documented primary association; optional
`--max-chi2-ndf` and `--max-dca-cm` remain analysis choices and require their own
distribution/cut scans before a physics result.

## Physics checks before replacing MC-truth protons

1. Compare generated-primary, reconstructable, TOF-matched and PID-selected
   proton spectra on identical source/event keys. Separate primary efficiency,
   secondary contamination, fakes and clones.
2. Measure PID purity and efficiency versus momentum, angle and centrality,
   including kaons, deuterons and heavier charged fragments. A pion veto alone
   does not establish proton purity.
3. Check fitted residual means and correlations. The circular normalised cut
   reproduces the slides' independent-width convention; appreciable correlation
   needs a separately validated covariance-aware selection.
4. Define the common neutron/proton phase space at the primary vertex. A
   straight-line HGND rectangle is a neutron fiducial proxy, **not** the proton
   magnetic acceptance. Keep reconstructed $p_T$, rapidity and lab angles for a
   measured common fiducial region or a forward-folded comparison.
5. Keep production-file correlation in uncertainties, freeze cuts on development
   files, and use event-paired neutron/proton quantities where possible.

The proton kinetic-energy hypothesis is
$E_{\rm kin,p}=\sqrt{p^2+m_p^2}-m_p$. The neutron 1–2 GeV kinetic-energy study
corresponds to proton momenta about 1.70–2.78 GeV/$c$ at the same kinetic energy.
PID support and efficiency must cover both sides of any chosen proton hardness
window before the truth term in $R_n/R_p$ is replaced.

## Validation completed locally

- Five tests pass: exact PID normalisation, MC-label independence, missing
  $dE/dx$/TOF and unsupported momentum rejection, TOF match semantics, zero-track
  event denominators, and recovery of widths in synthetic development templates.
- Python compilation and both batch-script syntax checks pass.
- The complete calibration → held-out selection CLI smoke passes on synthetic
  tables, including 400 extra zero-track events. Calibration, detector-band and
  cut-scan PDFs were rendered and visually checked; they are test fixtures, not
  SMASH results.
- No real ncx PID efficiency, purity or proton yield is claimed from these tests.
- Remaining gate: authenticate ncx, inspect one real DST and run its short
  extraction/calibration smoke with the installed dictionaries.
