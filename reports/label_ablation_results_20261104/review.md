# Read-only review of the finite no-floor closure

04.11.2026. Final verdict: **GO**.

Progress gate: the registered selected-validation result was obtained on
04.11. The last model task ended on 04.11 at 08:38:30 MSK. Work since then
accepted existing exports and compared saved predictions. No new training,
inference, test jobs or additional unit tests were run for this review.

| Review | Evidence and conclusion | Final verdict |
| --- | --- | --- |
| Technical | Nine complete 30-epoch outcomes; same 15-epoch prefix, coherent resume, frozen cuts. Actual export/member SHA acceptance and strict candidate-key binding checked. An independent arithmetic check of 42 macro/paired reductions matched. | GO |
| Statistics | Equal three-fold macro within seed, then three paired training seeds, df=2. Both common references retained; no pooled cross-fold AUC or nine-independent-cases test. Conditional selected-validation limitations explicit. | GO |
| Physics | All 290 validation groups including four unbound remain. Soft targets differ from the baseline regression training population. Same numerical cut does not guarantee the same purity. No physical Rn, spectrum or detector-efficiency claim. Raw response panel visually reviewed. | GO |
| Text | Initially required correction of excess uncertainty precision. Formatter now uses at most two significant SD/SE digits with matched central precision; exact JSON values retained. Soft MAE has explicit x10^3 scale. | GO after correction |
| Manager | All nine models and 18 segments terminal; 117330 allocated GPU seconds, peak concurrency two. No additional model run or epoch extension needed to close the finite registered experience. Finish visual QA, commit and stop the obsolete heartbeat. | GO |

The initial collector lookup assumed every prepare entry had every fold role.
Its KeyError traceback remains in the local closure log. An explicit `.get`
now handles entries outside that fold; the exact expected validation set is
still asserted against the pinned baseline identity and new exports. This
was a local analysis correction, with no repeat of a scientific job.

The two presentation findings were resolved: uncertainty precision and legend
overlap. Final ROC and prevalence guides have different styles and explicit
legend entries. The response axis states the maximum initial energy of the
qualifying MC tracks. All final panels were visually inspected by the parent.

Figures, report and results are bound by `validation.json`. The review does
not approve F scoring, new cut/calibration fits, an energy-window scan or a
new production campaign.
