# Scoped scientific and report review

18 November 2026. Reviewer: read-only GPT-6 Astra / medium, scoped preflight
followed by final technical/methodological/report review. Rules: project
AGENTS.md and thesis-kb §§10, 20, 25, 30. Verdict: **GO** for this event-level
diagnostic. No manuscript/publication claim was added.

Preflight checked the protocol, accepted matrix definitions in the rule
reducer and summary, and `even.json`. The last scientific result was the
18 November joint rapidity/pT reduction; last documented model scoring ended
16 November. Work included scientific reductions and same-day ROOT export
preparation, with no established infrastructure-only drift. Approved scope:
one local reduction, frozen selections and 400 existing shared file draws.

Final source review confirmed matrix axes and denominators, per-seed ratios
before equal-seed means, paired differences, potential-specific subsetting
of unchanged weights and all four current source hashes. All 48 unconditional
center/SD/quantile fields replay the accepted summary within 1e-12, and all
32 conditional mean distributions sum to 100%. Counts and rounded result-note
values match JSON. The difference signs hold in all three potential settings.

The reviewer inspected the actual PNG. Labels, markers, neutral line, event
counts and uncertainty definition were readable; the initial footer overlap
was corrected before final review. PDF was not independently rendered.
One nonblocking precision suggestion remains: the figure's shorthand
“70% purity target” could spell out “native-label validation-purity target”
on future integration. The adjacent README already gives that definition.

Accepted files:

- `results.json`, SHA256 `28bc2c8f4c5741588c7f04e9fa2de9a4ce40f86128a854a68a552fe9a2ab4911`.
- `no_output_by_multiplicity.png`, SHA256 `5a3ebee44450390efb724475e1b4828dd61b528b944393ab84392945ac186a9a`.
- `no_output_by_multiplicity.pdf`, SHA256 `6f92ac77e27fef41a141be9adf2fe460f2d4104f2e59157bf0b165d1dbc2220b`.
- Parent checked AST and every source SHA; no additional tests, model calls
  or cluster jobs were run. A layout-only revision replayed every numerical
  result exactly before updating the source binding.

Scope: reconstructed candidate response conditional on geometric MC
multiplicity. This does not determine per-neutron efficiency, physical
background, a shower-overlap mechanism or yield/n/p closure.
