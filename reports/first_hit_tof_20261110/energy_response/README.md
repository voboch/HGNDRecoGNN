# Neutron energy linearity and resolution

The same matched SMASH neutrons show two different energy-response effects.
Original first-hit ToF gives a smaller central-width point estimate in
every plotted bin. The uncalibrated SMASH-trained estimate has a smaller
central width than ToF with the network-input time jitter in every bin
from 0.7 to 5 GeV. Calibration reduces the high-energy scale shift and
widens the central residual distribution there.

## Common sample

The figures use the accepted [first-hit comparison](../README.md) on 286
SMASH Xe+Cs files at 2.87 GeV per nucleon, with the nominal HGND geometric
window, no impact-parameter selection and the same three training repeats.
The 80,515, 81,097 and 77,384 isolated direct primary-neutron links pass
both original score cuts and give physical speed for both timestamps.
All estimates use the initial MC primary kinetic energy as the reference.
Native MC matching uses a ToF condition. The first-hit estimate assumes
a straight path from the origin and uses the exported hit time without
a separate event start. The second ToF estimate uses Gaussian time jitter
with SD 0.15 ns on that same hit. The core widths apply to this selected
simulation cohort.

## Bin definitions and precision

For each truth-energy bin, define the relative residual
`r = (E_reco − E_true)/E_true`. Linearity is `100 Q50(r)` and resolution is
`50 [Q84(r) − Q16(r)]`, both in percent. The upper linearity panel also
plots the median reconstructed energy at the median true energy in
each bin. Its lower panel resolves scale shifts near the identity line.
The lower resolution panel shows sample support and exposes the sparse
tail.

The bins and 25-object minimum are inherited from
`scripts/reco_performance.py`. Fine bins below 1 GeV resolve the soft
response, and wider bins in the tail trade energy granularity for
precision. The plots start at the inherited 0.3 GeV native label floor
and cover all matched true energies (0.302–6.320 GeV). The bin from 5 to
6.5 GeV contains 30–42 objects per repeat. The bin width there shows the
range of the aggregate, with its point at the median rather than the
midpoint.

Compute each statistic separately for each training repeat and give
the repeats equal weight. The vertical bars show SD over the same 400
paired whole-file bootstrap draws used in the accepted comparison. The
same draw weights all estimators and repeats. Training-repeat values and
counts appear in the tables. These error bars quantify finite-file
precision on the selected sample.

## Response by energy

- In 0.7–5 GeV, original ToF core resolution is 1.15–4.20%, the
  uncalibrated SMASH-trained estimate gives 2.62–8.45%, and jittered ToF
  gives 3.03–20.00%. Original ToF has the narrower core and the larger
  overall mean absolute error (159 versus 86 MeV for the uncalibrated
  network). The accepted comparison finds 231–262 original ToF estimates
  above 10 GeV per repeat. The quantile width describes the core, while
  MAE also weighs these large errors.
- At 4–5 GeV, calibration moves the network median residual from −8.31%
  to +2.62% and increases the central half-width from 8.45% to 15.31%.
  In the 5–6.5 GeV bin the width difference between the uncalibrated
  network and original ToF is +2.26 ± 2.45 percentage points (paired
  file SD), so the difference there is unresolved.

[Linearity PDF](energy_linearity.pdf) · [Resolution PDF](energy_resolution.pdf) ·
[Per-bin CSV](bins.csv) · [Per-bin JSON with paired differences](bins.json)

The table also contains the raw and calibrated DCM-trained estimates.
The plots use the two timing variants and the two SMASH-trained estimates
to show the timing and calibration effects separately.

Reproduce from the project environment with
`python reports/first_hit_tof_20261110/energy_response/analyze.py` followed
by `python reports/first_hit_tof_20261110/energy_response/plot.py`.
The reduction requires exact replay of the accepted population counts,
energy moments and hardness totals before calculating the new bins.
