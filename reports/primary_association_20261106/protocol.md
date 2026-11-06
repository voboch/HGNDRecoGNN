# Uncensored primary denominator and direct associations

Registered 6 November 2026, before execution. This phase localises the gap
between an incident primary-neutron population and retained candidates. It
uses S286, accepted truth-bound even parts 0/2/4, frozen observations and native
clusters. No model, fitted correction, score choice or window scan is run.

The primary key is content/Row/Id. T0 includes every exported primary Row;
T1 includes exported Row < the pinned historical processed span. Events with
zero accepted particles remain rows. Absent exported events cannot be declared
zero-primary collisions; B/NPrim=-1 remain unknown. “Uncensored” describes the
energy selection, not a guarantee of complete generator-event production.

Use the unchanged per-vertex front-face rectangle in cm, inclusive edges,
forward rays only, without energy or time cuts. Save exact accepted neutron
energies, including zero and overflow, for subsequent registered reductions.
The fixed display bins [0,.1,.2,.3,.4,.5,1,2,4,8] GeV resolve the earlier floor
scan and inherited low/high boundaries; >=8 is a separate overflow. No bin is
optimised. Reproduce certified full per-event keys, multiplicities, uncensored n/p counts and low/high counters under an exact int64-array SHA, then historical T1 high >=2 and low [.3,1) n/p counters exactly.

Raw hits join vacs one-to-one on Row/Instance. Direct nucleons require Mother=-1,
Side5/6, and exact Row/Id plus PDG,Ekin,Px,Py,Pz,vX,vY,vZ against the pinned
primary file. Refuse conflicts, duplicates, missing or ambiguous links. Neutrons
are PDG2112; antineutrons are not pooled. Bind before the rectangle selection
and preserve out-of-window direct neutrons separately. Mother>=0 is nonprimary;
Mother0 is not primary and nonprimary does not imply material background.

Replay the accepted deterministic observation noise key and observation SHA,
then use accepted cluster membership and exact native labels/float32 targets.
No new graphs or input scalers. An object's stages are raw linked hit, retained
observed hit, no-floor qualifying hit and native qualifying hit. Each stage
counts distinct primary objects; cluster edges are unique primary/cluster
pairs. Split means >1 cluster per primary; merge means >1 direct primary per
cluster within the reported rectangle cohort. Mixed in/out-of-window merges
are recovered separately from the exported common edge table. Maximum bipartite cardinality is reported separately; it does not
assign a unique energy to a merged cluster. No threshold or selection is tuned.

Expectation before measurement: T0>=T1 for counts, and raw>=observed>=qualifying
for distinct directly linked objects. Size of the losses is unknown. Direct
associations omit descendants and unexported genealogy and therefore cannot
establish full neutron detection efficiency. Numerical encoding differences
from the previous phase are not a causal explanation of these losses.

One CPU job: rocky/normal/type_d,4CPU,0GPU,mem0,45min,no-requeue,2 workers,
one numerical thread each. Stage each pinned file node-local and delete only
its local input copy after use. Reuse immutable stage-A readers under their
full source checksum. No test queue/additional tests by direct human waiver;
all scientific input and replay assertions remain. Publish compact per-file
events, unbinned accepted/direct neutrons, association edges, summary and SHAs.
Technical and methodological preflight must approve the concrete diff before
submission. Luna/low monitors only this job; parent accepts the science.
