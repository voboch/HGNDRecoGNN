# Detector-only replay of neutron reconstruction on SMASH

Slurm job 4506482 completed on 12 November 2026 at 01:01:16 Moscow time in 5 min 17 s, exit 0:0. The archive and its three members passed SHA-256 checks. The file inventory, frozen checkpoint identities and counts reconcile to the registered plan. [Protocol](../protocol.md) · [Submission](../submission_4506482.json) · [Acceptance](acceptance.json)

On 286 historical even Xe+CsI SMASH files at 2.87 GeV/nucleon, reconstruction from detector hits alone reproduced the ordered bytes of 5,833,940 retained hits and every observable graph tensor for 1,798,225 GNN candidates. Rules + ToF gave identical components, candidate energies and selection. The legacy MC-identity veto removed zero hits in these files. The comparison covered 1,180,595 events in the pinned processed spans, including 306,885 without retained hits inferred from those spans.

On one file from each held-out fold, CPU inference with the original classifier and energy checkpoints gave bit-identical logits and raw energies in all nine fold–seed comparisons. Adding MC columns to detector input left the observations unchanged. Injecting a PDG-zero hit into the separate legacy reference removed the predicted 9, 1 and 2 hits in folds 0, 1 and 2. [Full replay ledger](files.json) · [Witness ledger](witnesses.json)

This control supports detector-only inference in the thesis Methods section. It resolves the implementation question for the historical population. The previously reported energy and spectral-hardness results keep their original selections and scope. An independent compatible simulation is still needed for efficiency, migration and background transfer. A contrast closure test also needs at least two matched potential settings.
