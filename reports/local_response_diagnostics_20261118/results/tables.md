# Tables from the frozen local diagnostic JSON

SMASH Xe+Cs, 2.87 GeV/nucleon. 286 previously inspected even files, 1,180,595 processed events; raw energies, 70% native-label validation-purity target; 0.3 < T(reco) < 4 GeV. Mean of three seed estimates. ± denotes sample SD of 400 paired whole-file bootstrap draws at fixed models. Seed SD is reported separately.

## Geometric MC-neutron multiplicity

MC includes all true energies. Miss/split denominators are geometric MC neutrons; merge denominator is selected candidates. Direct associations do not resolve descendants. No physical efficiency or background fraction is inferred.

| MC multiplicity | Metric (%) | Rules + ToF | DynamicEdgeConv | Paired GNN − rules (pp) |
|---|---|---:|---:|---:|
| 0 | miss | undefined ± undefined | undefined ± undefined | undefined ± undefined |
| 0 | split | undefined ± undefined | undefined ± undefined | undefined ± undefined |
| 0 | merge | 0.0060 ± 0.0027 | 0.0201 ± 0.0046 | 0.0142 ± 0.0045 |
| 1 | miss | 87.91 ± 0.06 | 82.93 ± 0.07 | -4.98 ± 0.05 |
| 1 | split | 1.31 ± 0.02 | 0.94 ± 0.02 | -0.38 ± 0.02 |
| 1 | merge | 0.13 ± 0.01 | 0.27 ± 0.02 | 0.14 ± 0.01 |
| 2 | miss | 86.65 ± 0.07 | 80.92 ± 0.08 | -5.72 ± 0.05 |
| 2 | split | 1.48 ± 0.02 | 1.06 ± 0.02 | -0.42 ± 0.03 |
| 2 | merge | 0.72 ± 0.03 | 1.54 ± 0.05 | 0.82 ± 0.05 |
| 3_or_more | miss | 85.80 ± 0.06 | 79.46 ± 0.07 | -6.35 ± 0.05 |
| 3_or_more | split | 1.61 ± 0.02 | 1.16 ± 0.02 | -0.45 ± 0.02 |
| 3_or_more | merge | 2.17 ± 0.05 | 4.41 ± 0.06 | 2.24 ± 0.06 |

## Seed stability on the common 58,199 primary keys

Both methods select each key for every seed. True energy is in [0.3,4) GeV. Seed variation is descriptive for three fixed trained pipelines; it is not a measured training-confidence interval.

| Energy metric | Rules + ToF, mean ± file SD | GNN, mean ± file SD | GNN seed centers 42 / 123 / 456 | GNN seed SD |
|---|---:|---:|---|---:|
| bias_MeV | 30.72 ± 0.72 | -10.41 ± 0.47 | -11.50 / -10.26 / -9.48 | 1.01 |
| mae_MeV | 101.81 ± 0.65 | 73.86 ± 0.40 | 74.33 / 73.47 / 73.77 | 0.44 |
| median_relative_percent | 0.47 ± 0.02 | -1.10 ± 0.02 | -1.21 / -1.03 / -1.05 | 0.10 |
| central_width_percent | 5.68 ± 0.03 | 4.60 ± 0.02 | 4.64 / 4.54 / 4.61 | 0.05 |

## Energy by true transverse momentum

Common isolated neutrons selected by both procedures within each seed. Counts are per seed; MC y and pT define all bins. A shared sample across methods does not remove common-selection bias.

| True pT (GeV/c) | N by seed 42/123/456 | Metric | Rules + ToF | GNN | Paired GNN − rules |
|---|---|---|---:|---:|---:|
| <0.5 | [39279, 39197, 39240] | bias_MeV | 7.17 ± 0.56 | -12.34 ± 0.40 | -19.51 ± 0.46 |
| <0.5 | [39279, 39197, 39240] | mae_MeV | 56.80 ± 0.53 | 49.11 ± 0.31 | -7.68 ± 0.45 |
| <0.5 | [39279, 39197, 39240] | median_relative_percent | -0.07 ± 0.02 | -1.55 ± 0.02 | -1.49 ± 0.02 |
| <0.5 | [39279, 39197, 39240] | central_width_percent | 4.28 ± 0.03 | 3.91 ± 0.02 | -0.37 ± 0.02 |
| >=0.5 | [19719, 19712, 19706] | bias_MeV | 77.04 ± 1.77 | -6.67 ± 1.28 | -83.71 ± 1.30 |
| >=0.5 | [19719, 19712, 19706] | mae_MeV | 190.62 ± 1.38 | 122.78 ± 0.97 | -67.84 ± 1.09 |
| >=0.5 | [19719, 19712, 19706] | median_relative_percent | 2.65 ± 0.07 | 0.25 ± 0.05 | -2.40 ± 0.07 |
| >=0.5 | [19719, 19712, 19706] | central_width_percent | 8.78 ± 0.07 | 5.87 ± 0.04 | -2.91 ± 0.06 |

## Accepted odd-file Rn results, now separated by seed

282 previously inspected odd files; frozen 70% benchmark and 75% example policies. Open hard band T >= 2 GeV. C90/0 = 100[Rn(90)/Rn(0)−1]. Contrast is in percent; SD and paired MC shift are in percentage points. All other contrasts/windows remain in JSON.

| Purity target | Seed | Reconstructed contrast ± file SD | Paired shift from geometric MC ± file SD |
|---|---|---:|---:|
| 70% | 42 | 2.56 ± 1.16 | -2.39 ± 1.13 |
| 70% | 123 | 3.19 ± 1.15 | -1.77 ± 1.11 |
| 70% | 456 | 2.74 ± 1.16 | -2.22 ± 1.14 |
| 75% frozen | 42 | 3.51 ± 1.30 | -1.44 ± 1.28 |
| 75% frozen | 123 | 3.33 ± 1.30 | -1.63 ± 1.26 |
| 75% frozen | 456 | 3.81 ± 1.33 | -1.14 ± 1.33 |

## All joint true-y/pT energy cells

All registered bins retained. Mean MAE/bias in MeV; paired MAE is GNN minus rules. Central relative width is in percent and requires at least 25 pairs in every seed. Undefined means are not replaced by zero. Any undefined required draw suppresses its interval. The populated y=[1.0,1.1), pT>=0.5 cell contains only 14–15 pairs per seed.

| MC y | MC pT (GeV/c) | N by seed | Paired MAE ± SD (MeV) | Rules bias (MeV) | GNN bias (MeV) | Rules width (%) | GNN width (%) |
|---|---|---|---:|---:|---:|---:|---:|
| [-1.1,-1) | <0.5 | [0, 0, 0] | undefined ± undefined | undefined | undefined | undefined | undefined |
| [-1.1,-1) | >=0.5 | [0, 0, 0] | undefined ± undefined | undefined | undefined | undefined | undefined |
| [-1,-0.9) | <0.5 | [0, 0, 0] | undefined ± undefined | undefined | undefined | undefined | undefined |
| [-1,-0.9) | >=0.5 | [0, 0, 0] | undefined ± undefined | undefined | undefined | undefined | undefined |
| [-0.9,-0.8) | <0.5 | [0, 0, 0] | undefined ± undefined | undefined | undefined | undefined | undefined |
| [-0.9,-0.8) | >=0.5 | [0, 0, 0] | undefined ± undefined | undefined | undefined | undefined | undefined |
| [-0.8,-0.7) | <0.5 | [0, 0, 0] | undefined ± undefined | undefined | undefined | undefined | undefined |
| [-0.8,-0.7) | >=0.5 | [0, 0, 0] | undefined ± undefined | undefined | undefined | undefined | undefined |
| [-0.7,-0.6) | <0.5 | [0, 0, 0] | undefined ± undefined | undefined | undefined | undefined | undefined |
| [-0.7,-0.6) | >=0.5 | [0, 0, 0] | undefined ± undefined | undefined | undefined | undefined | undefined |
| [-0.6,-0.5) | <0.5 | [0, 0, 0] | undefined ± undefined | undefined | undefined | undefined | undefined |
| [-0.6,-0.5) | >=0.5 | [0, 0, 0] | undefined ± undefined | undefined | undefined | undefined | undefined |
| [-0.5,-0.4) | <0.5 | [0, 0, 0] | undefined ± undefined | undefined | undefined | undefined | undefined |
| [-0.5,-0.4) | >=0.5 | [0, 0, 0] | undefined ± undefined | undefined | undefined | undefined | undefined |
| [-0.4,-0.3) | <0.5 | [0, 0, 0] | undefined ± undefined | undefined | undefined | undefined | undefined |
| [-0.4,-0.3) | >=0.5 | [0, 0, 0] | undefined ± undefined | undefined | undefined | undefined | undefined |
| [-0.3,-0.2) | <0.5 | [218, 216, 219] | 2.72 ± 8.33 | 10.93 | 16.69 | 1.21 | 2.99 |
| [-0.3,-0.2) | >=0.5 | [0, 0, 0] | undefined ± undefined | undefined | undefined | undefined | undefined |
| [-0.2,-0.1) | <0.5 | [737, 708, 718] | -17.79 ± 7.17 | 20.42 | -0.35 | 1.55 | 2.83 |
| [-0.2,-0.1) | >=0.5 | [0, 0, 0] | undefined ± undefined | undefined | undefined | undefined | undefined |
| [-0.1,0) | <0.5 | [1437, 1429, 1415] | 0.11 ± 3.03 | 3.48 | -13.57 | 1.91 | 2.62 |
| [-0.1,0) | >=0.5 | [0, 0, 0] | undefined ± undefined | undefined | undefined | undefined | undefined |
| [0,0.1) | <0.5 | [2512, 2506, 2493] | 0.67 ± 2.04 | 1.66 | -19.95 | 2.43 | 2.73 |
| [0,0.1) | >=0.5 | [0, 0, 0] | undefined ± undefined | undefined | undefined | undefined | undefined |
| [0.1,0.2) | <0.5 | [3918, 3897, 3901] | 2.17 ± 1.74 | 0.28 | -25.49 | 2.96 | 2.82 |
| [0.1,0.2) | >=0.5 | [0, 0, 0] | undefined ± undefined | undefined | undefined | undefined | undefined |
| [0.2,0.3) | <0.5 | [5716, 5720, 5727] | 0.75 ± 1.09 | -1.77 | -25.92 | 3.58 | 3.09 |
| [0.2,0.3) | >=0.5 | [0, 0, 0] | undefined ± undefined | undefined | undefined | undefined | undefined |
| [0.3,0.4) | <0.5 | [7309, 7298, 7316] | -3.57 ± 1.03 | 2.19 | -22.66 | 4.20 | 3.51 |
| [0.3,0.4) | >=0.5 | [0, 0, 0] | undefined ± undefined | undefined | undefined | undefined | undefined |
| [0.4,0.5) | <0.5 | [8334, 8318, 8346] | -9.31 ± 0.77 | 4.27 | -11.25 | 5.04 | 4.18 |
| [0.4,0.5) | >=0.5 | [419, 417, 419] | -16.82 ± 2.19 | 10.47 | -1.29 | 5.52 | 4.20 |
| [0.5,0.6) | <0.5 | [6251, 6262, 6259] | -17.77 ± 1.15 | 17.98 | 5.73 | 5.98 | 4.77 |
| [0.5,0.6) | >=0.5 | [3395, 3396, 3394] | -27.25 ± 1.45 | 27.27 | 12.61 | 6.54 | 4.84 |
| [0.6,0.7) | <0.5 | [2594, 2591, 2595] | -28.33 ± 1.62 | 31.79 | 18.26 | 7.00 | 5.23 |
| [0.6,0.7) | >=0.5 | [6180, 6180, 6189] | -46.15 ± 1.25 | 53.20 | 17.86 | 8.04 | 5.44 |
| [0.7,0.8) | <0.5 | [253, 252, 251] | -61.53 ± 9.19 | 70.00 | 22.40 | 9.27 | 5.91 |
| [0.7,0.8) | >=0.5 | [6209, 6213, 6206] | -95.59 ± 2.03 | 114.44 | 7.46 | 9.83 | 6.01 |
| [0.8,0.9) | <0.5 | [0, 0, 0] | undefined ± undefined | undefined | undefined | undefined | undefined |
| [0.8,0.9) | >=0.5 | [2936, 2928, 2918] | -121.49 ± 4.02 | 130.37 | -65.56 | 11.77 | 6.80 |
| [0.9,1) | <0.5 | [0, 0, 0] | undefined ± undefined | undefined | undefined | undefined | undefined |
| [0.9,1) | >=0.5 | [565, 564, 566] | -9.77 ± 10.49 | 6.45 | -234.63 | 10.40 | 6.78 |
| [1,1.1) | <0.5 | [0, 0, 0] | undefined ± undefined | undefined | undefined | undefined | undefined |
| [1,1.1) | >=0.5 | [15, 14, 14] | 160.58 ± 51.61 | -222.56 | -424.18 | undefined | undefined |
| [1.1,1.2) | <0.5 | [0, 0, 0] | undefined ± undefined | undefined | undefined | undefined | undefined |
| [1.1,1.2) | >=0.5 | [0, 0, 0] | undefined ± undefined | undefined | undefined | undefined | undefined |
| [1.2,1.3) | <0.5 | [0, 0, 0] | undefined ± undefined | undefined | undefined | undefined | undefined |
| [1.2,1.3) | >=0.5 | [0, 0, 0] | undefined ± undefined | undefined | undefined | undefined | undefined |
| [1.3,1.4) | <0.5 | [0, 0, 0] | undefined ± undefined | undefined | undefined | undefined | undefined |
| [1.3,1.4) | >=0.5 | [0, 0, 0] | undefined ± undefined | undefined | undefined | undefined | undefined |

Source results.json SHA256 `141e67854e5b51d07b849bc016d73197e149c960eee0ed47e69dcc91cbebc656`.
