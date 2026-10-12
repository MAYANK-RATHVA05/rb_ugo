# EC-EBO method decisions before implementation

Source: L. Tao, F. Yu, F. Wang, and Q. Wang, “EC-EBO: Entropy-balanced
oversampling via externally constrained clustering for imbalanced data,”
*Knowledge-Based Systems* 338 (2026) 115544,
DOI: 10.1016/j.knosys.2026.115544. This document audits the full 14-page
article. It specifies a planned implementation; **no EC-EBO sampler has been
implemented or evaluated in this repository yet**.

The method targets intra-class imbalance: the minority class contains
subclusters with different sizes and learning difficulty. The paper proposes
externally constrained minority clustering (Section 3.1), logistic-probability
entropy and entropy-weighted sample allocation (Section 3.2), and coordinate
search for synthetic points (Algorithm 1). It reports 12 parameter-study
datasets plus 48 comparison datasets, with five-fold cross-validation and
KNN, CART, NB, and SVM. Its selected settings are 60 maximum iterations,
decay `alpha=1`, seven nearby majority candidates, and stop probability 0.2.

## Mathematical audit and decisions

The following choices make the algorithm executable without claiming
equivalence to author code, which was not supplied with this PDF. “Paper
rule” identifies a directly stated equation or operation; “declared
interpretation” identifies our resolution of
missing or conflicting details.

| Issue | Evidence in paper | Decision for the planned default |
| --- | --- | --- |
| Cluster entropy | Section 3.2 calls it the **average**, and Eq. 8 writes mean entropy, but Eq. 7 and Algorithm 1 print a **sum**. | Use the arithmetic **mean** of member entropies. Keep the printed-sum rule as a separately labeled sensitivity variant if needed. |
| Balanced-region sign | Eq. 8 subtracts the majority influence from minority influence. Algorithm 1 accepts a move when `psi < 0`, although that can select the majority side. The prose says move toward `psi = 0` while avoiding majority regions. | Start from a minority-side seed (`psi > 0`). Accept a proposed point only when `0 < psi(proposal) < psi(current)`. The positive side and movement toward zero are an interpretation, not a guarantee of no overlap. |
| Point evaluated | Algorithm 1 creates `translated_point` but writes `psi(chi)` without identifying `chi`. | Evaluate `psi` at the **proposed** point, using the same majority reference for its current/proposed comparison. |
| Coordinate step | Algorithm 1 multiplies by `step` but never defines it or gives a value. | After training-fold scaling, set `step = 0.1 × median distance to the nearest minority row at a different location` within the fitting fold. The 0.1 multiplier is an engineering default, not a published setting. If no positive distance exists, report a no-op. |
| Early stopping | Algorithm 1 checks the 0.2 stop probability **before** the first move, allowing exact seed copies. | Check stopping **after at least one accepted move**. After at most 60 proposals, a candidate with no accepted move is recorded as ungenerated; do not silently count a copy as a new geometric sample. This departs from the printed pseudocode. |

The sign conflict is concrete. In one dimension, let a minority centroid be
0, a majority reference be 2, both entropy weights be 0.5, and `alpha=1`.
Equation 8 gives `psi(0)=+0.4323`, `psi(1)=0`, and `psi(2)=-0.4323`.
Thus the printed `psi < 0` test accepts the majority side in this example.
This correction does **not** prove the chosen positive region is free of
majority observations; entropy weights can shift the zero surface. Actual
majority-neighbor intrusion must be measured separately.

The sum/mean conflict also matters for intra-class imbalance. If clusters of
20 and 2 points have the same mean member entropy, Eq. 7's printed sum
allocates synthetic samples in a 10:1 ratio to the *larger* cluster. The
prose's mean allocates equally before considering other difficulty factors,
giving the small cluster more augmentation per original point. This is why
the mean is the planned default.

## Remaining operational rules, fixed in advance

- **Input and leakage:** Accept finite numerical binary fitting data. Use the
  existing fold-local preprocessing; clustering, logistic regression, entropy,
  and synthesis see only the fitting partition. Mixed/categorical features
  require separate decisions before use. No held-out row guides generation.
- **External clustering:** Begin with singleton minority clusters. For each
  cluster, compute the minimum original-member-to-majority Euclidean
  distance, as Eq. 3 states. Among all pairs satisfying Eq. 4's strict
  inequality, merge the pair with the smallest centroid distance. Break ties
  by original row index, update the centroid as the member mean, recompute
  eligibility, and stop when no pair qualifies. The merge order is our
  deterministic interpretation of Algorithm 1.
- **Entropy:** Fit unweighted logistic regression on the fitting partition;
  use its minority probability for **both** minority and majority rows in
  Eq. 6. Clip probabilities only for numerical stability in the logarithm.
  The paper does not specify logistic settings or how majority entropy is
  obtained; record estimator parameters and the clipping constant.
- **Allocation:** Request `max(0, n_majority - n_minority)` new rows. Divide
  this integer budget among clusters in proportion to their **mean** entropy.
  Apply largest-remainder rounding, with original cluster index resolving
  ties, so the allocated counts sum exactly to the budget. If every entropy
  weight is zero, divide equally and record the fallback. If a cluster
  cannot generate its quota, report the shortfall; do not transfer it
  silently to another cluster.
- **Majority reference:** For each candidate, select one of the `min(7,
  n_majority)` majority rows nearest to its cluster centroid, uniformly and
  with a seeded RNG. Use that row's Eq. 6 entropy in Eq. 8. Keep the same
  reference throughout that candidate's coordinate search. These selection
  details are not fixed by the paper.
- **Coordinate search:** Select a seed uniformly from cluster members with
  positive `psi` for the chosen majority reference. Propose a random
  coordinate basis direction and sign, using the declared step. Accept only
  the strictly positive, decreasing `psi` condition above. Stop after 60
  proposals or a post-acceptance 0.2 random stop. If no safe-side seed or
  accepted move exists, record that allocated sample as ungenerated.
  Random seeds, accepted steps, duplicate counts, and generation shortfalls
  must be exposed in diagnostics.

The resulting sampler should be described as an **audited EC-EBO
interpretation**, not an exact reproduction. The paper's Eq. 4 limits
cluster merges; it does not ensure the space between cluster members is free
of majority samples. Likewise, the entropy score is a local proxy, not a
theorem that synthetic samples cannot overlap the majority distribution.

## Implementation gate and evaluation

Before using real data, tests must cover hand-calculated merge decisions,
the mean-vs-sum allocation example, the one-dimensional sign example,
exact integer allocation, deterministic randomness, generation shortfall,
and unchanged original rows. A two-dimensional synthetic case should
visualize unequal minority subclusters separated by majority points.

For a small real-data comparison, start with shared numerical inputs and
use identical outer folds for NORES, SMOTE, UGO-SMOTE, BFCH-SMOTE, and
EC-EBO. Vowel must retain the declared acoustic-feature and speaker-group
protocol if compared to our existing BFCH study; that is not the paper's
13-feature protocol. Report per-cluster original and generated counts,
entropy, unmet quotas, and a training-only majority-neighbor intrusion
proxy, alongside held-out F1, G-mean, MCC, AUC, and false-positive rate.
One or two initial datasets can verify the pipeline; several datasets with
different minority-subcluster structures are needed for an empirical claim.

The article gives no code link, numeric step size, preprocessing recipe, or
exact folds; its data-availability statement says data are available on
request. Author code or clarification would be needed to claim exact
author-implementation fidelity.
