# Focused exposure mechanism diagnostic: preregistration

17 October 2026. This document and
[the complete configuration](../../configs/exposure_mechanism_v1.json) are frozen
and committed before drawing any datasets for this grid. The earlier four-case
pilot is observed evidence, not part of this new seed-level evaluation.

## Evidence audit and question

The saved pilot manifest hash agrees with the reviewed aggregate evidence.
The separated case generated 12 budgeted U children versus 30 unconstrained;
all used endpoints had correct hidden labels. Count, threshold and weighted-PL
controls matched its MCC. The threshold control had lower estimated exposure
and lower oracle expected child error. Ecoli2 used one unique endpoint across
rounds. Overlap/Ecoli4 were inactive. These data support the prior conclusion:
software feasibility passed, but no reduction of realized lineage error or
unique predictive advantage was demonstrated. The single-dataset seed and
low endpoint diversity prevent a useful mechanism conclusion.

Question: under fixed eligibility and the existing supervised trace, does the
budget reduce descendant-weighted endpoint risk beyond simpler matched controls,
and does any such reduction translate into better child risk or classification?
No adaptive algorithm will be implemented in this stage.

## Frozen design

Eight cells cross separation **1 or 3**, labeled minority count **8 or 32**, and
U minority prevalence **.02 or .20**. Each has dataset seeds **42–51**: 80
independently generated cases, 560 method jobs including NORES. Algorithm seed
stays 42, separating algorithm randomness from dataset replication. Each cell's
streams use `SeedSequence([dataset_seed,cell_index,stream])`, with independent
L, U and test streams. No data are redrawn for unfavorable outcomes or missing
eligibility. The per-cell pairing unit is the independent dataset seed.

Two-dimensional identity-covariance Gaussians have majority mean `(-d/2,0)` and
equally weighted minority modes `(d/2,-1.5)` and `(d/2,1.5)`. L contains exactly
200 majority and 8/32 minority observations, with iid minority-mode membership.
U contains 400 iid draws at its cell prevalence, so its realized minority count
can differ from the expectation. Test has 4,000 independent draws at fixed
minority prevalence .20. Varying U prevalence at fixed test prevalence isolates
pool composition; varying labeled minority count deliberately changes the
observed labeled prior and calibration support. Oracle probabilities explicitly
use the U prior for endpoint/child mechanism diagnostics and the test prior for
test-mode expectations. These are declared simulation assumptions, not dataset
facts. No oracle prior correction is supplied to a method.

The unchanged methods are original UGO-SMOTE (`parent`), budget, unconstrained,
count-matched, threshold-matched and weighted PL. NORES is an additional required
supervised baseline. All receive identical L. The U controls share exactly the
same frozen guide and candidate bank, `.9` eligibility threshold, `.025` exposure
rate, cap five, L-only preprocessing and fixed candidate check. The threshold
control's existing selection within the common eligible bank is retained.
No threshold search, seed search, grid extension or parameter correction follows
inspection of this grid. Default UGO safeguards and fixed .5 final classification
threshold remain unchanged. The frozen source hashes are in the configuration.

## Measurements and denominators

- Eligibility: union of eligible endpoint IDs across rounds, union with at least
  one surviving candidate, selected union, selected/usable fraction, and effective
  endpoint count `(sum m)^2/sum(m^2)`. Record each round's counts as well.
- Constraint activity: among nonempty usable minority-generation banks, report
  continuous binding and restrictive frequencies separately. A loose-exposure
  solution with the same log utility establishes whether the constraint actually
  restricts allocation. Separately report conservative integer-floor loss and
  count reduction; these are not mathematical binding certificates.
- Endpoint error: ordinary eligible/selected error and descendant-weighted
  realized hidden-label error; oracle expected endpoint error is
  `sum m_j*(1-eta_U(u_j))/sum m_j`. Oracle values enter only evaluation.
- Child error: mean `1-eta_U(z)` for actual U-informed children; weighted PL and
  parent have no such children and receive null. Separately report expected error
  across all retained added minority training mass, including parent fallback
  and weighted endpoints, to make the complete training interventions comparable.
- Minority modes: expected joint minority-component mass per mode in retained
  added minority training weight; coverage is the fraction of two modes with
  at least one expected unit. Report U-child mass separately. Also report test
  recall per true minority mode, including its test denominator. This does not
  treat a generated vector's mode as known ground truth.
- Classification: MCC, G-mean and sklearn AP, all at the same untouched test
  distribution. Runtime separates shared preparation from policy/fitting/
  prediction cost. Shared-cost timings are not standalone performance benchmarks.

Inactive cases stay in unconditional classification comparisons. Zero descendants
means undefined lineage/child error, never zero error. Reasons distinguish no
minority round, no eligible endpoints, no accepted candidates and no budgeted
integer allocation. Failed jobs remain in coverage; paired summaries use only
shared successful seeds and give their exact IDs. No failed seed is replaced.

## Paired effects and decision

Report budget minus each control by cell across ten seeds: means, sample SDs,
SEs, ranges and wins/ties/losses (numerical tie tolerance `1e-10`). Mechanism
effects use paired active cases with explicit denominators; unconditional
classification effects retain inactive runs. Report active-subset classification
effects separately. Pooled effects first average across the eight cells within
each dataset-seed replicate, avoiding 80 pseudo-replicates for that summary.
No confirmatory significance claim follows from this initial grid.

Five paired active seeds per cell are required to call that cell adequately
active. Practical margins are .01 for endpoint/child error and .005 for each
classification metric. If threshold-matched or weighted PL matches or improves
primary error and all three classification outcomes within those margins in
every adequately active cell, reject the present added-value claim on this grid.
Insufficient activity, error opportunity or diversity makes the mechanism
under-identified; recommend revising or stopping this hypothesis without changing
the completed grid. Favorable effects would still require independent confirmation
and literature clearance. None automatically authorizes an adaptive extension.

Preregistration is a prospective scope record, not a claim that these power or
practical-margin choices establish definitive scientific rejection or equivalence.
