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


## 1a. First real DST: the combined PID is not available on this production

Run on ncx102, 2026-10-13, against
`/scratch2/shabanov/Simulation/Smash_defaultSpot_NoAbsorber/defaultSpot_0_200/bmndst_smash_xecs_2.87gev_hardSkyrme_defaultSpot_1573086_1.mcini.root`
(1.70 GB, 4126 entries), BmnRoot `bmnroot_dev` at
`/lhep/users/karpushkin/nica130T/...`, ROOT 6.30/08.

**The API gate passed with zero failures.** Every getter the exporter calls
exists with the assumed signature, which settles the question item 2 of the
task was about. `GetQ`, `GetDedx` and `GetCharge` are confirmed absent.

**The feasibility probe failed, and the cause is a reconstruction
configuration, not missing input.** On 2000 entries, 14 537 tracks:

| measurement | value |
|---|---|
| tracks per event | 7.27; 32.6 % of events have none |
| positive / negative | 12 079 / 2458 |
| dE/dx usable | **0 (0.00 %)** — all 14 537 are the `0.0` ctor default |
| `GetGemTrackIndex() >= 0` | **0** |
| `IsPrimary()` | **100.00 %** |
| TOF-400 matched / valid m² | 2079 (14.3 %) / 100 % |
| TOF-700 matched / valid m² | 2754 (18.9 %) / 100 % |

`GetGemTrackIndex() >= 0` on **100 %** of tracks, so `SetGemTrackIndex` ran
and `BmnGlobalTracking`'s `if (fStsTracks)` block was **not** skipped. An
earlier version of this note blamed the `StsVector` / `StsTrack` branch-name
default for the missing dE/dx; that measurement disproves it, and the claim is
withdrawn. `CalcdQdn` sits inside the same block that demonstrably ran.

What the numbers do establish, from `CalcdQdn`'s structure alone: its only
write is `SetdQdNLower(dedx)`, with `dedx = -100` when it finds fewer than
three usable GEM hits and a positive mean otherwise. **Not one track carries
-100.** So this is not "ran and found no hits" — the write never executed, for
all 12 079 positive tracks. (For the 2458 negative tracks `0.0` is expected:
`CalcdQdn` returns early on `GetP() < 0`.)

**Localising it without the production source.** `CalcdQdn(&globTr)` is the
*last* statement of that block. Immediately before it come two Kalman
propagations that `continue` on error, the DCA computation and
`SetChi2InVertex`. So `GetDCAInVertex()` and `GetChi2InVertex()` say whether
the stored tracks reached the block's end:

- **both constant across every track** → nothing wrote them, so the tracks left
  the block early (a `continue` on a propagation, or a null `fVertexL1`) and
  `CalcdQdn` was never reached. The missing dE/dx is then a *symptom*, and the
  defect is in the vertex or the propagation, not in `CalcdQdn`.
- **either varies** → the tracks did reach past `SetChi2InVertex`, so the
  `CalcdQdn` call one line further on is itself what did not happen, and the
  `grep` below is the confirmation.

The probe now measures and reports this. Note the predicate treats
non-finite values explicitly: `NaN != NaN`, so a naive all-equal test would
report an all-NaN column as "varies" and invert the verdict. That case is
covered by an isolated check of the predicate.

The remaining question, if the second branch holds, cannot be answered from the
source read for these notes: the DST was produced by
`/lhep/users/karpushkin/nica130T/bmnrootVersions/bmnroot_dev`, which is not on
the SSHFS mount, whereas the notes were read from `/scratch2/vboch/bmnroot`.
The API matches between them — the gate passed — but the *implementation* need
not. One command settles it:

```bash
grep -n 'CalcdQdn' \
  /lhep/users/karpushkin/nica130T/bmnrootVersions/bmnroot_dev/reconstruction/globaltracking/BmnGlobalTracking.cxx
```

A commented-out or absent call is the simplest explanation consistent with
every number above — and note that the copy read for these notes already has
one commented-out `CalcdQdn` call site alongside the live one.

Two further traps this exposed:

- **`IsPrimary()` is useless on this production.** `BmnGlobalTrack`'s
  constructor sets `fIsPrimary(kTRUE)`, and nothing in the path that ran sets
  it false, so the 100 % rate means "never set" rather than "all primary". It
  must not be used as a primary cut here. The earlier claim in
  `docs/PROTON_PID_NCX_20261012.md` that it "supplies the primary association"
  is withdrawn for this production.
- **The inner-track branch name is `StsVector` here**, not the `StsTrack`
  default. The gate and converter now try both and report which is in use; the
  element class is `CbmStsTrack` either way, so the template type was never
  wrong. This matters for reading the DST — but it is *not* the cause of the
  missing dE/dx, as the 100 % inner-track link shows.

**What is available instead:** m² from TOF on ~33 % of tracks, with m² valid on
100 % of matched tracks. That supports a TOF-only PID, which is a *different
and weaker* observable than the specified combined one — no dE/dx means no
`r_p` radius and no pion veto as defined. Adopting it is a change of method
and has to be declared, not substituted quietly. No calibration has been
fitted, and none should be until the input question is settled.

### Independent of that decision

Two things were wrong regardless of how the dE/dx question resolves, and are
fixed:

- **`mc_pdg` and `mc_mother` were never filled on track rows.** `FillTrackRow`
  set `mc_index` and nothing resolved it, so every track row carried
  `mc_pdg = 0`. "Is this a truth proton" was therefore unanswerable and the
  efficiency and purity denominators item 5 requires could not have been
  computed at all. A reusable `bmnx::FillMcTruth<McTrack>()` now resolves the
  link, kept separate from `FillTrackRow` so the reconstruction path never
  touches the MC array, and an out-of-range link is recorded as unassociated
  rather than read past. Tested, including that `motherId == -1` is *primary*
  while `0` is a valid mother *index* — inverting those would count every
  daughter of track 0 as a primary proton.
- **Whether the MC link works at all on this production was unmeasured.** The
  deep probe now reports `BmnMCInfo.fMCTracks` occupancy, the truth-primary
  proton count (the efficiency denominator), how many reconstructed tracks
  resolve to a truth particle, and how many of those are protons. If
  `GetTrkID()` turns out unpopulated, `BmnGlobalTrackMatch` and `StsTrackMatch`
  are both present in the DST as the alternative route, and the probe says so.

The conversion and CSV validation (smoke steps 5-6) do **not** depend on dE/dx
and can be run now in an allocation; they exercise the export path, the event
ledger and the `(source_id, Row)` join on real data.

**Decision needed** before any further PID work, once the `grep` above has
identified whether `CalcdQdn` is called in the production build: either
re-reconstruct with dE/dx actually enabled, or accept a declared TOF-only PID
with its weaker separation. This is the user's call.

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

Four defects were found by the first real invocation on ncx102 and by
exercising the macros against a local ROOT 6.36, before any DST was read:

- `config.sh` appends to `DYLD_LIBRARY_PATH` without defining it. Under
  `set -u` an unbound variable exits the **whole shell**, not just the sourced
  file, so the run died with no message from the script at all. `run_convert.sh`
  already dropped `-eu` around this; the smoke now does the same.
- The gate reported every getter missing when the BmnRoot libraries were simply
  not loaded — indistinguishable from a renamed API, and it would send the
  reader to fix the wrong thing. It now locates the owning library by searching
  the build's own `.rootmap` files for the class and loads it, so the library
  name comes from the installed build rather than a guess.
- The "getters the exporter deliberately does not use" check printed
  `GetQ absent, as documented` when the class itself had no dictionary. That is
  false reassurance; it now says the check was skipped.
- `dynamic_cast<TTree*>(f->Get("bmndata"))` **segfaults** inside cling when the
  key is not a `TTree` (an RNTuple, in the test fixture) instead of returning
  null. Both macros use `TFile::GetObject` now and exit cleanly. Note
  `inspect_branches.C` still has the original `dynamic_cast`.

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
