# EC-EBO method decisions and implementation status

Source: L. Tao, F. Yu, F. Wang, and Q. Wang, “EC-EBO: Entropy-balanced
oversampling via externally constrained clustering for imbalanced data,”
*Knowledge-Based Systems* 338 (2026) 115544,
DOI: 10.1016/j.knosys.2026.115544. This document audits the full 14-page
article. It specifies the implementation choices; the resulting code is an
**audited interpretation**, not a verified copy of the authors' implementation.

Part 1 is now implemented in `src/ec_ebo_clustering.py`: it fits externally
constrained minority clusters and exposes member row indices, centroids,
nearest-majority distances, and merge history. Its synthetic, hand-calculated
tests are in `tests/test_ec_ebo_clustering.py`. Part 2 is implemented in
`src/ec_ebo_entropy.py`: unweighted logistic-regression probabilities give
binary sample entropy, cluster weights use the **mean** member entropy, and
largest-remainder allocation returns an exact integer quota. Part 3 is
implemented in `src/ec_ebo.py`: it uses the allocated quotas,
seeded majority references and coordinate proposals, accepts only positive
decreasing Eq. 8 scores, and reports unmet quotas without copying seeds.
Its synthetic generation tests are in `tests/test_ec_ebo.py`. Part 4 registers
`ec_ebo` as an opt-in method in the existing paired evaluation runner;
fold-local preprocessing, entropy fitting, generation, and final classifier
fitting use only training rows. Part 5 is a two-dataset, same-fold descriptive
comparison in [EC_EBO_SMALL_STUDY.md](EC_EBO_SMALL_STUDY.md); it does not
establish general superiority or exact paper fidelity.

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

| Issue | Evidence in paper | Decision for the implemented default |
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
the mean is the implemented default.

## Operational rules, fixed in advance

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

The resulting sampler must be described as an **audited EC-EBO
interpretation**, not an exact reproduction. The paper's Eq. 4 limits
cluster merges; it does not ensure the space between cluster members is free
of majority samples. Likewise, the entropy score is a local proxy, not a
theorem that synthetic samples cannot overlap the majority distribution.

## Implementation gate and evaluation

Synthetic tests cover hand-calculated merge decisions, the mean-vs-sum
allocation example, the one-dimensional sign example, exact integer
allocation, deterministic randomness, generation shortfall, unchanged
original rows, and two-dimensional coordinate generation. Visualizing an
unequal-subcluster synthetic case remains part of the evaluation work.

The runner accepts `ec_ebo` through `--methods`. It keeps the original
default methods unchanged and stores per-job generation diagnostics and
shortfalls in `resampler_diagnostics`. It requires original numerical features
and fold-local scaling; categorical or boolean source columns and `--no-scale`
are rejected before jobs start. A local binary input with an explicit
source-domain positive label can be checked against the baselines on identical
outer folds, for example:

```powershell
.\.venv\Scripts\python.exe main_stage4r5.py --dataset data\raw\my_binary.csv --target class --positive-label positive --output results\ec-ebo-my-binary --classifiers LogReg --methods nores smote ugo_smote bfch_smote ec_ebo
```

The command is a template, not a completed experiment. Use a new output
directory for each input or configuration. The generator is an audited
interpretation of the paper, and the runner's ten-fold protocol is a declared
approximation unless compliant folds are supplied with provenance.

The [small study](EC_EBO_SMALL_STUDY.md) uses shared numerical inputs and
identical outer folds for NORES, SMOTE, UGO-SMOTE, BFCH-SMOTE, and EC-EBO.
Vowel retains the declared acoustic-feature and speaker-group protocol, which
differs from the paper's 13-feature input. Saved diagnostics include per-cluster
original and generated counts, entropy, unmet quotas, and a training-only
majority-neighbor intrusion proxy, alongside held-out F1, G-mean, MCC, AUC,
and false-positive rate. These two datasets verify the pipeline; several
datasets with different minority-subcluster structures would be needed for a
broader empirical claim.

The article gives no code link, numeric step size, preprocessing recipe, or
exact folds; its data-availability statement says data are available on
request. Author code or clarification would be needed to claim exact
author-implementation fidelity.
