# Proton reconstruction and PID — runbook

Supersedes the PyROOT production reader in `docs/PROTON_PID_NCX_20261012.md`
for the **extraction** stage. Track and event export now happen inside the
existing ROOT C++ converter, in the same DST event loop that writes the HGND
CSVs. Python keeps the fitting, plotting and regression tests.

---

## 1. What is verified locally, and what is not

**No part of this has executed against a real DST.** `ssh boch@ncx104.jinr.ru`
times out from here, so there is no execution on ncx; the SSHFS mount at
`/Users/vovvy/ncx` (= `/scratch2/vboch` on the cluster) gives filesystem access
only.

What the mount did make possible, on 2026-10-13, is reading the **installed
BmnRoot source** instead of the published header. That moved the API from
assumed to verified and corrected three errors that would each have been a
hard failure:

| Previously assumed | Installed build (`/scratch2/vboch/bmnroot`) |
|---|---|
| `GetDedx()` | does not exist. dE/dx is `GetdQdNLower()`, written by `BmnGlobalTracking::CalcdQdn`, in calibrated GEM cluster charge — not MeV/cm |
| `GetQ()` charge magnitude | does not exist. `GetZ()` is assigned from a hardcoded He/Z=1 cut on the same dE/dx, so it is derived from it |
| inner branch `StsVector` | the branch is `StsTrack`, class `CbmStsTrack`: `GetNStsHits()` not `GetNHits()`, and no `FromVertex()` — the primary flag is `BmnGlobalTrack::IsPrimary()` |

Two further findings are physics-relevant rather than cosmetic:

- `BmnTrack::GetP()` returns `1.0/GetQp()` and is therefore **signed**. The
  exporter takes the sign from `q/p` and the magnitude from `1/|q/p|`.
- `CalcdQdn` returns early when `GetP() < 0`, so **negative tracks never carry
  dE/dx**. The π⁻ reference band the `r_π > 3` veto was to be built from is not
  available from this branch; it has to come from π⁺ on the positive side. The
  deep probe reports the negative-track dE/dx count so this is measured on the
  production rather than taken on trust.

A DST still has to be inspected: no reconstructed DST exists anywhere under
the mount (the `smash_xecs_2.87gev_hardSkyrme_*` directories hold
already-converted CSVs, and `data/bmnsim_p_99.root` is a simulation file), and
`run_convert.sh` defaults `BMNROOT_CONFIG` to a **different** build
(`/nica/bmn1/karpushkin/.../bmnroot_dev`) than the one read above. Every gate
is therefore re-checked at run time by `inspect_tracks.C`.

### Verified locally, by tests that run in this repo

| Behaviour | Test |
|---|---|
| charge sign comes from the sign of `q/p`, not from `GetZ()` (there is no `GetQ()`) | `tests/cpp/test_track_export.cc` |
| momentum components reconstruct from slopes; `\|p\|` closes | same |
| `dE/dx = 0` is recorded as missing, never as a measurement | same |
| an unmatched TOF arm, and the DST's `m2 = -1000` sentinel, are rejected | same |
| a null inner-track array leaves vertex/MC association unknown | same |
| a zero-track event still writes an event-ledger row | same |
| `source_id` is basename-stable and separates files | same |
| `source_id` agrees between C++ and the Python analysis twin | `tests/test_proton_event_keys.py` |
| `Row = entry`, zero-based, and a one-event shift changes the join | same |
| calibration refuses to load if incomplete or unattributable | `tests/cpp/test_proton_pid.cc` |
| tracks outside the fitted momentum range are rejected, not extrapolated | same |
| missing `dE/dx` or TOF yields an explicit status, never a silent pass | same |
| negative and unknown-sign tracks are never selected | same |
| the slide equations: `x`, `y`, `r = hypot(x, y)` | same |
| `r_p < 2` is a 2-D radius: 1.5σ in both coordinates fails | same |
| the pion veto rejects a pion-like track | same |
| no MC column appears in the PID schema | same |
| the `-100` too-few-GEM-hits sentinel stays distinct from the `0.0` never-ran case | `tests/cpp/test_track_export.cc` |
| a negative track is never given a dE/dx value | same |
| the primary flag survives a missing inner-track array; the inner columns do not | same |
| the smoke validator accepts a well-formed CSV pair and rejects a one-based `Row`, a missing or duplicated event ledger, a dangling track `Row`, a dropped `dedx_status` and an unprefixed truth column | `ncx_refactor/smoke_tracks.sh --self-test` (7 cases, no ROOT needed) |

Run them:

```bash
g++ -std=c++17 -o /tmp/t1 tests/cpp/test_track_export.cc && /tmp/t1
g++ -std=c++17 -o /tmp/t2 tests/cpp/test_proton_pid.cc   && /tmp/t2
python3 tests/test_proton_event_keys.py
./ncx_refactor/smoke_tracks.sh --self-test
python3 tests/test_proton_pid.py
```

### NOT verified — requires ncx

- **That the DST getters exist and behave as assumed.** The names come from the
  published `BmnGlobalTrack` header and the October 2026 DST guide, not from the
  installed build. `BmnTrackExport.h` lists them in a block at the top. A
  renamed getter fails at ROOT's interpretation step, which is intended.
- **That the available files are reconstructed at all.** Generator-only SMASH
  output has no `BmnGlobalTrack` branch. The converter stops with a message
  rather than writing an empty table.
- **That `dE/dx` is filled.** `BmnDedx` runs only for some productions; the DST
  guide says the field is zero otherwise. Without it the combined PID is
  unavailable and the code does not fall back to a TOF-only selection.
- **Whether simulated strip amplitudes carry a physical species dependence**, a
  precondition for fitting a Bethe curve at all.
- Every number in the calibration file. There is no fitted calibration yet.

---

## 2. Extraction

One event loop, all outputs:

```bash
./run_convert.sh --dataset defaultSpot --start 1 --stop 200 --tracks --dry-run
./run_convert.sh --dataset defaultSpot --start 1 --stop 200 --tracks --submit
```

`--tracks` adds `NNNN_tracks.csv` and `NNNN_events.csv` beside the existing
`_hits`, `_vacs` and `_prim`. Without it the conversion is unchanged.

Outputs are written to a `.part` prefix and renamed only when every requested
file is present and non-empty, then a `NNNN.done` marker records which outputs
were requested. A resume skips a file only if its marker matches the current
request, so adding `--tracks` to an earlier primaries-only run re-converts
rather than skipping. A failed file makes the script **exit non-zero**.

### Join to the HGND arm

`(source_id, Row)`. `Row` is the zero-based TTree entry in every table; it
repeats across files, which is why `source_id` exists. The Python twin of
`source_id` is in `tests/test_proton_event_keys.py` and is asserted equal to the
C++ one.

### The event ledger

One row per processed entry, including entries with no tracks, no primaries and
no hits. An efficiency denominator built from the track table alone silently
drops the events where reconstruction failed, which are exactly the ones that
set the efficiency.

---

## 3. Calibration (Python, frozen, versioned)

Fit on an **explicitly declared** development sample, freeze, and apply the same
file to all three U_sym samples. The file is flat text so ROOT's C++ and Python
read it without a library:

```
version            smash-dev-2026-10-13
fitted_on          SMASH defaultSpot, files 0-99, development half
units              dedx arb; m2 GeV^2; p GeV/c
p_range            0.5 4.0
tof_arm            400
cuts               2.0 3.0
proton_bethe       a1 a2 a3 a4 a5
pion_bethe         a1 a2 a3 a4 a5
proton_sigma_rel_dedx   c0 c1 c2
proton_sigma_m2         c0 c1 c2
pion_sigma_rel_dedx     c0 c1 c2
pion_sigma_m2           c0 c1 c2
```

`version` and `fitted_on` are **required**: the loader refuses a calibration it
cannot attribute. `p_range` is the fitted domain and is enforced — a track
outside it is rejected with status `kOutOfDomain`, never extrapolated.
`tof_arm` is 400 or 700 and never a union; combining the arms would count a
track matched in both twice.

---

## 4. Selection

Positive charge, primary-associated, `r_p < 2`, `r_pi > 3`, inside the
calibrated domain, with valid `dE/dx` and a valid TOF match in the declared arm.
Cut values are in the calibration file, not the code.

`r_p < 2` keeps `1 - exp(-2) = 86.5 %` of an ideal two-dimensional Gaussian
before any other loss. It is not the one-dimensional "two sigma" figure, and the
test asserts the difference.

No MC quantity enters. The track table carries `mc_index`, `mc_pdg` and
`mc_mother` for validation, all prefixed, and the PID table carries none.

---

## 5. First run on ncx, in order

Steps 1-5 are wrapped by `smoke_tracks.sh`, deployed to
`/scratch2/vboch/HGND/` with `SMOKE_README.md`. It refuses to convert without
`SLURM_JOB_ID`, so it cannot reconstruct on a login node by accident, writes
only into a fresh `smoke_<timestamp>/`, and leaves `out/` untouched. The
originals it replaced are in `/scratch2/vboch/HGND/backup_20261013_153415/`.

0. `./smoke_tracks.sh --self-test` — validates the validator. No ROOT, no DST,
   no allocation. Passes on the laptop; a failure here means the deployed copy
   is damaged.
1. **Locate a reconstructed DST.** There is none under `/scratch2/vboch`.
2. `./smoke_tracks.sh --allow-login <dst>` — runs the two read-only probes and
   stops before any conversion.
   - `inspect_tracks.C` is the API gate. It touches no BmnRoot type, reaching
     everything through `TClass` reflection, so a renamed getter is reported
     with its signature instead of killing the interpreter at parse time. It
     also reports if a `GetQ()` or `GetDedx()` has appeared on this build,
     which is a reason to revisit the exporter's notes, not to start reading
     them.
   - `inspect_tracks_deep.C` measures feasibility: dE/dx occupancy split by
     charge sign and by the two invalid cases, TOF match rate per arm, m²
     validity given a match, the dE/dx+m² overlap that is the real combined-PID
     denominator, and the dE/dx **percentiles** that fix the units the Bethe
     fit must be done in. If dE/dx is empty on every track, it stops: `CalcdQdn`
     did not run in that production and the combined PID is unavailable there.
     That is an input problem, and the answer is not to fall back to m² alone
     without saying so, nor to substitute MC truth.
3. `srun -p <partition> -n1 -t 00:30:00 --pty bash`, then
   `./smoke_tracks.sh --events 200 <dst>` — the full smoke, including the
   conversion and the CSV validation.
4. Only then fit the calibration, and only then run the array.

Efficiency is measured against **all truth-primary protons in the declared
fiducial region**, from `_prim.csv`, not against reconstructed tracks. Report
purity, secondary and clone contamination, and tracking / TOF / PID losses
separately; a single efficiency number hides which stage lost what. Deduplicate
MC clones on `(source_id, Row, mc_index)`. Use held-out files for performance.
