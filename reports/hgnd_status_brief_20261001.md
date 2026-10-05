# HGND neutron reconstruction and symmetry-potential sensitivity

**Status draft — 1 October 2026, 12:54 MSK**  
**Scope:** SMASH Xe+CsI at $\sqrt{s_{NN}}=2.87$ GeV, $U_{\rm sym}=0,18,90$ MeV; final shared-scaler GNN; MC study.

---

## Slide 1 — Executive result

- The final GNN now reconstructs neutron energy with **5.9–9.1% resolution** and
  **$|$bias$|<9.3\%$ over 0.7–5 GeV**.
- Neutron yield and truth $n/p$ yield are not sensitive observables.
- Reconstructed neutron hardness has the correct positive endpoint response,
  but the held-out result is only **$+3.89\pm2.85\%$ ($1.37\sigma$)** and the
  0/18/90 central values are not ordered.
- The hybrid $R_n/R_p$ reaches **$+11.54\pm2.23\%$ ($5.18\sigma$)** on the
  held-out split and is ordered. This is promising, but its proton term is still
  **MC truth**, not a reconstructed BM@N proton measurement.
- Yield extraction still requires energy- and multiplicity-dependent response
  correction; one inclusive efficiency is insufficient.

**Decision:** retain spectral hardness as the physics observable; treat the
hybrid result as a feasibility target for HGND plus a charged-particle arm, not
as a completed measurement.

---

## Slide 2 — Definitions and data scope

### Observables

For species $s\in\{n,p\}$ and two energy boundaries,

$$
R_s(E_L,E_H)=
\frac{N_s(E_{\rm kin}\ge E_H)}{N_s(E_{\rm kin}<E_L)},
\qquad
D_{np}=\frac{R_n}{R_p}.
$$

The per-event differential neutron yield and endpoint response are

$$
\left.\frac{dY_n}{dE}\right|_i=
\frac{1}{N_{\rm evt}}\frac{N_{n,i}}{\Delta E_i},
\qquad
\Delta_O=\frac{O_{90}}{O_0}-1,
\qquad
Z_O=\frac{|\Delta_O|}{\sigma_{\Delta_O}}.
$$

### What was checked

| Check | Statistical scope | Treatment | Result |
|---|---:|---|---|
| Truth feasibility | 568 production files; 2,345,993 events | generation-file bootstrap; event-level $b$ reweighting | hardness responds; yield ratios do not |
| Final SMASH reconstruction | 614,400 represented events per Spot; 2,746,581 cluster rows | one frozen GNN; one shared scaler; $q>0.60$ | reconstruction target largely met |
| Window choice | 65% development / 35% held out | choose $(E_L,E_H)$ on development only | quote held-out response below |
| DCM reference control | 283,398 graphs | SMASH-to-DCM transfer and native DCM training | no large generator-domain failure |
| Multiplicity response | about 450,000 events with a reconstructable neutron across the three samples | rank-conditional Wilson intervals | higher-rank losses are large and consistent across Spot samples |

For the fixed conventional $E_L=1$ GeV, $E_H=2$ GeV truth observable in the
exact HGND front-face acceptance:

| Full-production truth observable | $\Delta_{90/0}$ | Separation | Conclusion |
|---|---:|---:|---|
| $R_n$ | $+4.52\pm0.45\%$ | $10.15\sigma$ | neutron spectral shape is sensitive |
| $R_p$ | $-5.39\pm0.51\%$ | $10.48\sigma$ | opposite proton response |
| $R_n/R_p$ | $+10.48\pm0.76\%$ | $13.82\sigma$ | strongest truth-level target |
| $Y_n/Y_p$ | $+0.06\pm0.28\%$ | $0.22\sigma$ | reject as a Spot observable |

---

## Slide 3 — Final reconstruction performance

![Final HGND reconstruction performance.](../results/status_brief_20261001/fig01_final_reconstruction_performance.png)

- ROC AUC: **0.958**.
- At $q>0.60$: energy purity **87.3%**, energy efficiency **71.5%**; count
  purity **85.5%**, count efficiency **63.7%**.
- From 0.7 to 5 GeV: maximum $|$median bias$|=9.2\%$ and maximum robust
  resolution $=9.1\%$.
- The energy estimator now satisfies the nominal 10% bias/resolution target in
  the energy range used by the physics windows.

---

## Slide 4 — Which observable survives reconstruction?

![Held-out observable response and separation.](../results/status_brief_20261001/fig02_heldout_observable_separation.png)

The displayed errors and significances are from the held-out 35%; spectral
windows were selected on the development 65%. Filled markers indicate ordered
0/18/90 MeV central values.

| Observable | Development-selected window | Held-out $\Delta_{90/0}$ | $Z$ | Ordered? | Experimental requirement |
|---|---|---:|---:|:---:|---|
| truth $Y_n/Y_p$ | inclusive | $-0.12\pm0.45\%$ | 0.26 | no | MC truth only |
| reco $\widehat Y_n$ | inclusive | $+0.43\pm0.65\%$ | 0.67 | no | HGND only |
| truth $R_n$ | $E<0.8$ / $E\ge2.2$ GeV | $+4.83\pm1.02\%$ | 4.75 | yes | MC truth only |
| reco $R_n$ | $E<2.2$ / $E\ge2.8$ GeV | $+3.89\pm2.85\%$ | 1.37 | no | HGND only |
| truth $R_n/R_p$ | $E<1.7$ / $E\ge1.9$ GeV | $+6.34\pm1.07\%$ | 5.92 | yes | MC truth only |
| reco $R_n$/truth $R_p$ | $E<1.6$ / $E\ge2.4$ GeV | $+11.54\pm2.23\%$ | 5.18 | yes | HGND + proton arm |

Short conclusion:

- Retraining restored a physically useful reconstructed spectral response.
- Neutron hardness alone is still statistics limited on the held-out split.
- The hybrid result is the only reconstructed-level row above $3\sigma$, but
  it is not yet an experimental observable because $R_p$ is truth-level.
- A final claim requires reconstructed protons, common phase space, systematic
  uncertainties, and a pre-frozen window on an independent sample.

---

## Slide 5 — Efficiency and multiplicity corrections

![Energy-dependent and multiplicity-dependent neutron response.](../results/status_brief_20261001/fig03_efficiency_and_multiplicity.png)

The two efficiency factors answer different questions:

$$
\epsilon_{\rm det}(E)=
P(\text{selected cluster}\mid n\text{ reaches HGND}),
\qquad
\epsilon_{\rm sel}(E)=
P(\text{selected}\mid\text{truth-matched cluster exists}).
$$

For neutron rank $k$ in an event,

$$
\varepsilon_k=P(k\text{th neutron selected}\mid N_n^{\rm true}\ge k),
\qquad
C_k=P(N_n^{\rm selected}\ge k\mid N_n^{\rm true}\ge k).
$$

- Detection is only **2% below 0.5 GeV** and rises to about **80% near 2–3
  GeV**; a sub-GeV hardness denominator is therefore not measurable by a simple
  efficiency correction.
- At $q>0.60$, $\varepsilon_1\simeq66\%$, $\varepsilon_2\simeq43\%$, and
  $\varepsilon_3\simeq35\%$.
- The probability of retaining all first $k$ neutrons falls from
  $C_1\simeq67\%$ to $C_2\simeq33\%$ and $C_3\simeq15\%$.
- The narrow bands show that these losses are consistent across the three Spot
  samples. They are response corrections, not sensitive observables.
- Use a joint energy–multiplicity response matrix or forward fold:

$$
m_i=\sum_j A_{ij}t_j+b_i,
$$

where $A_{ij}$ contains efficiency and migration and $b_i$ contains fake
clusters.

---

## Slide 6 — Generator-domain control and live status

| Evaluation | ROC AUC | Energy purity | Energy efficiency | Max $|$bias$|$, 0.7–5 GeV | Max resolution, 0.7–5 GeV |
|---|---:|---:|---:|---:|---:|
| Final SMASH GNN on SMASH | 0.9582 | 87.3% | 71.5% | 9.2% | 9.1% |
| Same SMASH GNN transferred to DCM | 0.9525 | 86.6% | 68.8% | 9.0% | 10.5% |
| Native DCM GNN on DCM | 0.9515 | 86.9% | 66.3% | 4.5% | 9.1% |

- The SMASH-trained estimator transfers to DCM without a large collapse.
- Native DCM training improves energy linearity but not classification AUC;
  the remaining difference from the published $\sim0.97$ AUC is therefore not
  explained by generator transfer alone.
- The comparison is diagnostic, not exact: the transfer cache uses the SMASH
  scaler and 1.5 ns graph window; native DCM uses its own scaler and the
  reference-like 1.0 ns window.

### Live cHARISMa snapshot

| Job | State at 12:46 MSK | Purpose | Current reading |
|---|---|---|---|
| 4367897 | running on rocky/A100 | energy-flattened native DCM training | epoch 14 completed; training progressing |

The energy-flattened loss uses inverse-frequency weights up to 570, so its loss
scale is not comparable with the unweighted run. An eight-sample GPU check saw
only **3–7% utilisation** and about 1.5 GB memory use; this is a throughput and
idle-reaper risk that should be checked again outside a validation/loading
interval. No result from this running job is used above.

---

## Slide 7 — Final decision and next checks

### Established

- Truth spectral hardness is sensitive to $U_{\rm sym}$ after centrality
  control and exact front-face acceptance.
- The final GNN meets the relevant energy-response target.
- Yield ratios remain unsuitable; spectral observables are required.
- Multiplicity losses are large enough to require a response model.

### Not yet established

- A standalone HGND neutron-hardness measurement above $3\sigma$.
- An experimental hybrid measurement with reconstructed protons.
- Stability across model families, seeds, calibration choices, and detector
  systematics.

### Minimum next actions

1. Finish and evaluate the energy-flattened DCM control; stop it if sustained
   GPU utilisation remains low and the cluster idle policy is at risk.
2. Freeze the reconstructed hardness and hybrid windows before another test.
3. Implement the proton-arm reconstruction and common neutron/proton fiducial
   phase space.
4. Forward-fold the joint energy–multiplicity response and repeat the held-out
   0/18/90 test.
5. Compare the frozen baseline with at least one challenger architecture and
   propagate calibration, efficiency, centrality and model-family systematics.

---

## Reproducibility pointers

- Final SMASH performance: `results/final_eval/observables/performance_q060.json`
- Held-out sensitivity: `results/final_eval/figures/final_sensitivity.json`
- Efficiency and multiplicity: `results/final_eval/figures/`
- Full-production truth response: `results/acceptance_full/b_reweight.json`
- Figure builder: `scripts/make_status_brief_figures.py`
