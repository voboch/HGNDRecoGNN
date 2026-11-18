# Event multiplicity diagnostic while ROOT access is unavailable

18 November 2026. This is the multiplicity-dependence check in §2 of the
[neutron-yield plan](../neutron_yield_rapidity_plan_20261117.md). It informs
the thesis section on neutron reconstruction and the limits of a linear
yield response. Audience: the HGND analysis group. Budget: one local CPU
reduction of the accepted 176 kB ledger, one figure and a short result note;
no fits, inference, new jobs, threshold scans or additional tests.

Question: how often do the two frozen procedures return no selected candidate
in events containing zero, one, two or at least three geometric MC neutrons?
The sign and size of the conditional differences are unknown before this
reduction. Changes with multiplicity can motivate a later response study;
they do not establish shower overlap as their cause.

Read `even.npz` only after verifying the SHA in accepted `even.json`.
Keep the 286 even files, processed spans, all zero-output events, three
seeds, raw DynamicEdgeConv at the accepted 70% native-label validation-purity
target, rules + ToF, and the common reconstructed energy window
0.3 < T < 4 GeV. Truth multiplicity is the existing geometric-primary count
over all true energies. It is neither entrance multiplicity nor observed
hit occupancy. Its four categories (0, 1, 2, >=3) are inherited from the
accepted matrix, not optimized. The last category pools high multiplicities
and cannot resolve their internal variation.

For each truth category, calculate P(N_selected = 0 | N_geometric), with
the number of events in that category as denominator. Also retain the full
conditional candidate-multiplicity distribution (0, 1, 2, >=3); its tail
cannot be used to compute an untruncated mean. Selected candidates can come
from other particles or primaries outside the geometric reference. Therefore
no-output probabilities are event-level response diagnostics, not per-neutron
miss rates; nonzero output in the zero-truth category is not a proven physical
background fraction. No conclusion about linearity or n/p closure is licensed.

Use the same 400 paired whole-file bootstrap draws. Calculate each seed's
ratio within each draw, then average seeds equally. Export the paired
GNN-minus-rules difference, file SD (ddof=1), both Q16/Q84 bounds, invalid
draw counts and per-seed centers. Do not interpret file SD as training
uncertainty. Give pooled and all three potential settings separately.
Verify event counts, common truth marginals for all seeds/methods, matrix
conservation and exact replay of the accepted unconditional no-output rate.
If a category denominator is zero, leave its ratio undefined.

The figure shows the no-output probability and its paired difference against
the four categorical truth multiplicities. These bins expose single versus
multiple-neutron events without inventing finer resolution than the saved
matrix. Display event counts. This is a retrospective diagnostic on previously
inspected files; it selects no model or operating point. Existing results are
preserved. Do not insert a new claim into the manuscript in this task.

Rules read: thesis-kb conventions index (12 November revision), §10,
§20, §25 and §30; project AGENTS.md. The standalone no-ai-slop skill named
by AGENTS.md was not installed in the available skill locations; its inline
writing checklist applies. Scientific preflight and focused final report
review use GPT-6 Astra / medium under project §10c. Review findings and
source/figure hashes are recorded beside the result.
