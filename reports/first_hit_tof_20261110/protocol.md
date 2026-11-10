# First-hit ToF energy on matched primary neutrons

Registered 10 November 2026 before calculating outcomes. Use the accepted 286
SMASH Xe+Cs files at 2.87 GeV per nucleon, the nominal HGND window and processed
event span. The 142,912 isolated primary–cluster pairs are the exact prior
native direct edges of degree one at both ends of the complete observed direct
graph. Pair by content, Row, side and local cluster ID. The energy comparison
uses the intersection of the two original held-out score selections per seed.

Within each accepted cluster select the earliest retained hit by original
exported time. Resolve ties by Instance. The hit choice uses neither truth nor
scores. Replay raw hit/vacs hashes, observations with the original noise key,
native pack SHA, candidate keys and native labels/targets. Compare original
and deterministically jittered time on that same hit. Use the nominal distance
from the origin to the cell centre. The CSV provides neither separate event
start nor true path, so this calculation estimates a straight-line ToF proxy.

Use beta = L/(c*t) and Ekin = m_n*(1/sqrt(1-beta^2)-1) with the inherited
939.55 MeV mass. Mark nonpositive t or beta outside (0,1) as undefined. Count
invalid cases before any energy comparison. The shared valid subset requires
physical beta for both timestamps. On that subset compare both ToF estimates
and saved raw/calibrated predictions from both pipelines with the initial
MC primary energy. Do not clip finite energies. Record the count above the
production feature ceiling of 10 GeV. Report bias, MAE and median absolute
error, with paired whole-file SD for means and differences. Keep inherited
0.3–1 and ≥2 GeV hardness bands and compare contrast to the same selected
primary truth, using the existing 400 paired file draws. There is no new cut,
fit, model call, odd report tuning or energy-window scan.

One 45-minute rocky/normal CPU job (4 CPUs, 0 GPUs, mem=0, 2 workers,
no-requeue) extracts first-hit times and lengths. It uses the previous accepted
native-pack archive and raw hits/vacs only. Analysis of stored predictions
runs locally. The user's waiver excludes extra unit tests and the test queue.
