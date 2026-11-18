# Local diagnostics review

18 November 2026. Read-only GPT-6 Astra / medium reviewers, separate
methodology and technical roles. Sources: protocol.md, reduce.py, accepted
energy/rapidity/joint and odd Rn sources, project AGENTS and thesis-kb
§§10,20,25,30. Last scientific results: 18 November local no-output and joint
migration. Last documented model scoring: 16 November. No prolonged
infrastructure-only interval.

## Preflight

Methodology **GO** after clarifying geometric primary-neutron denominators,
per-seed quantiles before equal-seed averaging, undefined-seed propagation
and separate even/odd file coordinates. These clarifications were applied
to the protocol before execution.

Technical review initially **NO-GO** for missing full no-output/event-matrix
replay and missing binding of the supplied sidecar acceptance to the accepted
rapidity scan. Both bounded checks were added before execution. Imported
helper source hashes and exact energy-bin count marginal checks were also
added. The review's conditional **GO** applies after those fixes. Actual y,
joint and odd Rn dimensions/keys were inspected by the reviewer. The source
passes AST parsing and focused Ruff F checks. No additional tests, model
calls or jobs were used.

## Final acceptance

Technical critic **GO**. Independently verified current reducer/protocol/data/
assembly/helper hashes; replayed saved per-file truth/direct/miss/split/
candidate/merge/mixed-merge counts and both energy-moment ledgers against
accepted even.npz. Independently checked unique primary keys and the same
58,199-key intersection in all seeds, and recalculated pooled energy-bias
centers, paired differences, file SD and Q16/Q84 within 1e-12. Source review
confirmed the rapidity/joint/Rn indexing and unchanged analysis roles. No
hidden model, ROOT, fit or job invocation was found.

Methodology critic **GO**. Reviewed actual protocol, note, tables, assembly/
result JSON and all three PNGs. Accepted the counter-direction of miss/split
and merge changes, descriptive seed stability, common-selection conditioning,
separate seed/file spreads and the explicit low-pT and sparse-tail exceptions.
Physical-efficiency/background/independent-closure claims remain open.

Independent report critic **GO** for the internal report and PNG renders.
All panels have distinct purposes: multiplicity rates and paired changes;
energy MAE, bias, central width and paired MAE; same-primary seed control
and downstream frozen hardness example. Markers, external legends, neutral
lines and uncertainty bounds were readable without overlap or clipping.
Retrospective samples and MC/reconstructed roles were explicit.

Two nonblocking presentation suggestions were applied after review: table
rounding now preserves nonzero small SDs, and the standalone seed figure
now includes the system/beam energy, energy-cohort bounds and Rn/contrast
definitions. Parent checked the regenerated seed PNG and the affected table
rows; numerical results were unchanged. No scientific rerun was needed.
PDF versions were exported from the same plotting calls; independent visual
review covered PNGs, not manuscript/slide insertion sizes or PDF rasterization.

Accepted numerical result SHA256:
`141e67854e5b51d07b849bc016d73197e149c960eee0ed47e69dcc91cbebc656`.
Assembly/data/source hashes are in `results/assembly.json` and final figure,
table and environment hashes in `results/validation.json`. Runtime: assembly
4.2 s, summary 7.6 s, one thread; no extra tests or HPC allocation.

Thesis impact: neutron-reconstruction performance diagnostics now include
multiplicity-conditioned direct associations and energy errors in true
(y,pT), with descriptive seed variation. The interpretation is retrospective
and geometric. These outputs supplement the earlier global comparison and
event no-output table; no previous accepted result is superseded. ROOT-access
and physical-response gaps remain unchanged.
