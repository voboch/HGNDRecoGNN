# Student notebook reproducibility audit — 19 October 2026

The question is whether the six UFC notebooks in
`../GNN_for_neutron_reconstruction/` reproduce their saved ROC-AUC values and
whether their models can enter the HGND reconstruction pipeline.

This audit reads the original notebooks without modifying them. It checks Python
syntax separately from Colab installation commands; records saved validation
and test outputs separately; reproduces preprocessing and event splits from the
local `NeutronRecoGNN/data/Jan26/hits_Jan_26.pkl`; and exercises the six model
definitions on CPU. Where syntax repairs are needed for a compatibility probe,
the exact repairs are recorded and do not count as an unmodified notebook run.

Expected outcomes before the probes: preprocessing counts should match the
saved outputs if the local pickle is the student's input; observable features
and graph topology should survive a change to MC labels; architecture forwards
should work on the small connected components that preprocessing retains.
Neither successful forwards nor matching counts reproduce a trained ROC-AUC.

The event-metric probe calls the original inference function on a constructed
event, with different batch sizes. A valid event efficiency should not depend
on how that event's candidate graphs are divided between batches. No performance
claim will be made from this constructed example.

No training, cluster submission, old-campaign recovery, evaluation of source
988976, manuscript edit or replacement of the production model is part of this
audit. Complete model-weight/CSV reproduction requires the missing student's
checkpoint and prediction artifacts and their data/source manifests.
