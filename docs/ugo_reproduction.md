# Original UGO implementation and paper alignment

Stage 4R-2 recovers the source-audited batch and cap corrections from
[0866dc9](https://github.com/MAYANK-RATHVA05/rb_ugo/commit/0866dc936f231feef9df5e1538b8df206741735d).
The full uploaded article has been inspected. This remains a **provisional
reproduction with explicit paper ambiguities**, not reproduced benchmark evidence.

See the [reproduction specification](reproduction/SPECIFICATION.md),
[source register](reproduction/SOURCE_REGISTER.json) and
[current core alignment report](reproduction/CORE_ALIGNMENT.md).
Stage 4R-3 adds a shared generator interface, numerical RWO and MCSMOTE
baselines, UGO-RWO and UGO-MCSMOTE, the LEAO comparator, and the remaining
standalone method constructors. See
[GENERATOR_ALIGNMENT.md](reproduction/GENERATOR_ALIGNMENT.md). All three
original method papers are now available; source conflicts and implementation
choices still prevent unique author-code equivalence.
The historical Stage 3 prototype is preserved at
[c2fa7f3](https://github.com/MAYANK-RATHVA05/rb_ugo/commit/c2fa7f333689df14c5ab4bb542500bf2676ac897).

## Source and status

Sağlam and Cengiz, “Uncertainty guided oversampling: A classifier-specific approach
for effective handling of class imbalance in machine learning,” Knowledge-Based
Systems 342 (2026), 115886, DOI
[10.1016/j.knosys.2026.115886](https://doi.org/10.1016/j.knosys.2026.115886).
Page numbers refer to the uploaded 20-page PDF; its hash is in the source register.
Author-code equivalence remains unverified. The publisher supplement was supplied
later; the [collection record](reproduction/DATASET_COLLECTION.md) and source
register document its 74-entry metadata/results and accepted input approximations.
Earlier absence statements describe the original source-audit snapshot.

Runtime still requires allow_provisional=True and emits a warning. Diagnostics
report source_audited_with_ambiguities, not unique author-code equivalence.
The sampler implements the published UGO framework, with no held-out risk-budget
acceptance extension. Separate runners fit fresh final classifiers and save
held-out scores; these descriptive results do not resolve source fidelity.
The [project scope](PROJECT_SCOPE.md) now covers published-method reproduction.

## Data and classifier contract

UGO.fit_resample(X, y) accepts fitting data only and returns copied numerical
arrays with original rows/labels first and generated rows appended. Inputs and
the supplied estimator are not mutated or fitted in place.

Features must be finite dense real numerical values with meaningful Euclidean
distances. Missing/categorical features require externally fitted training-only
preprocessing. Typed categorical columns are rejected; integer category codes
cannot be recognized reliably from dtype.

The estimator must be sklearn-cloneable and implement fit/predict_proba. Each
iteration clones it, fits accumulated original plus generated fitting rows, and
scores the original pre-clean rows. Explicit estimator/nested random states are
preserved; unset ones are filled with the sampler's seed and recorded.

The returned estimator_ is the last internal scoring model. If its last iteration
added a batch, it has not fitted that final batch. A later evaluation stage must
train a fresh final classifier. No validation/test arguments exist, but the
software cannot identify held-out rows wrongly passed as fitting data.

## MMA and selection

Section 3.2, p6 defines:

    U_MMA(x,y) = max over k != y of P(k|x) - P(y|x)

For binary normalized probabilities this equals 1-2*P(true), ranging from -1
to 1. Larger values mean stronger disagreement with the known fitting label,
not entropy or a formal epistemic/aleatoric decomposition.

Probability columns use classifier.classes_, including reversed/string labels.
Invalid shapes, labels, nonfinite/out-of-range values and probability sums outside
absolute tolerance 1e-8 are rejected.

Sections 3.3–3.4 and Table 4 choose the class with **highest** mean MMA, which can
be the original majority. The introduction's “lowest” wording conflicts with
those detailed rules. Balanced counts are not themselves a stopping condition.

Uncertainty and seed pools remain fixed original pre-clean rows. Within the
selected class, descending MMA and stable original-index ties choose:

    n_select = max(1, ceil(lambda_select * eligible_original_class_count))

The paper writes n^(c)*lambda_select but does not settle whether that class count
includes noisy rows. Eligible count, rounding and score ties are declared
conventions. Synthetic points do not become scoring points or future seeds.

## Fixed original-total batches

Section 3.4, p7 and Table 4 step 2, p8 define n_step=n*lambda_step, initialized
before the loop. The nominal batch now uses original total size:

    n_nominal = max(1, ceil(lambda_step * N_original))
    n_batch = min(B_remaining, n_nominal)

The nominal batch is fixed across iterations and class choices; only final cap
clipping can shrink it. Ceil rounding, minimum one and clipping are declared
engineering conventions. The later conclusion mentions increasing generation
per iteration without specifying a different equation.

The historical prototype used the selected class's current size. On the existing
120-row example the corrected first batch is 12 rather than 3 for a 24-row class.
This changes generation/stopping trajectories; old generated counts are not
expected to remain identical.

## Noise and generation

Tables 2–3, p8 and Section 3.4, p7 support initial neighbor-vote detection,
generated-to-reference final voting, and relocation to a clean same-class point.

Current explicit conventions:

- Pre-detection uses Euclidean neighbors of original rows, excluding self by
  row index. Distinct coincident rows remain valid neighbors.
- Strict opposing majority marks noise; tied votes retain the assigned label.
- k_pre is clamped to N_original-1; k_after to N_original.
- Pre-noise is computed once. Noisy originals remain in classifier fitting data
  but cannot be seeds, uncertainty-pool rows or relocation destinations.
- Final votes use all original fitting rows. Relocation copies the nearest
  original pre-clean same-class row's coordinates. No eligible destination raises
  an error; samples are not silently dropped.
- Equal distances use original reference index order.
- ROS draws informative seeds uniformly with replacement and copies them.
- SMOTE draws informative seeds, selects among their nearest original pre-clean
  same-class neighbors, and generates (1-r)*seed+r*neighbor with uniform r in
  [0,1). k_smote=5 follows Section 4.2, p8; it is clamped to eligible count-1.
  Fewer than two eligible same-class rows stops with a diagnosed reason.
- Numerical RWO adds independent Gaussian perturbations with population standard
  deviation of the original pre-clean selected class divided by sqrt(class size).
  Informative seeds are centers; clean nonseed class rows contribute spread.
  It requires provisional opt-in and reports original-source numerical status.
  Discrete/categorical RWO rules are not implemented.
- MCSMOTE constructs a Gaussian kernel over clean selected-class originals,
  Sinkhorn-normalizes it, samples transition targets and interpolates from
  informative seeds. The published sign and stopping conflicts are resolved
  explicitly as described in the generator alignment report.

The paper does not uniquely settle every pool, vote-tie and draw convention.
Section 3.1 supports augmented retraining, while Table 4 omits the corresponding
in-loop X/Y reassignment. Original scoring/reference pools are explicit choices,
not a claim that all author conventions are now known.

Neighbor calculations allocate a full pairwise distance matrix and sort it.
This is suitable for small conformance checks; benchmark resource suitability
must be measured before full-scale experiments.

## Welch and generation stopping

Section 3.3, p6 uses Welch ANOVA and stops when equal class uncertainty means are
not rejected. For two classes, the two-sided Welch t-test has F=t² and the same
p-value, with F degrees of freedom (1, Welch df). A defined p>=alpha stops.

Non-rejection does not prove equal uncertainty or balanced predictive risk.
Too-small/doubly constant groups, invalid degrees of freedom and invalid
statistics produce an explicit undefined status. The loop stops separately with
undefined_welch_safeguard:<reason>, never a fabricated p=1.
One constant group remains valid when the other varies.

The article contains two distinct generated-sample thresholds:

| cap_interpretation | Source expression | Integer budget B used here |
| --- | --- | --- |
| section_3_3 (default) | nsyn >= phi*K*Nmax-N | max(0, ceil(phi*K*Nmax-N)) |
| table4 | Continue while nsyn < phi*Nmax | max(0, ceil(phi*Nmax)) |

N, Nmax and K=2 are frozen from original fitting data. Ceil produces the minimal
integer meeting a noninteger inequality. For counts 90/10 and phi=2.5, the
budgets are 350 and 225. For counts 3/2 and phi=1.1, the Section 3.3 real threshold
is 1.6 and its integer budget is 2, correcting the historical floor budget of 1.

Every batch is clipped to the remaining integer budget. Table 4's pre-batch test
can overshoot, so clipping remains a disclosed deviation. Cap termination uses
the existing provisional_generation_cap reason for compatibility. max_iter=100
is a separate engineering bound; all-noise/no-neighbor paths are also diagnosed.

## Parameters and diagnostics

| Parameter | Default | Meaning |
| --- | --- | --- |
| variant | ros | ROS, SMOTE, numerical RWO or MCSMOTE; no silent fallback |
| lambda_select | 0.5 | Eligible-original seed fraction; rounding convention above |
| lambda_step | 0.1 | Fixed original-total nominal batch fraction |
| phi | 2.5 | Selected source threshold multiplier |
| cap_interpretation | section_3_3 | Named paper expression; alternative table4 |
| k_pre / k_after | 5 / 1 | Initial and final votes |
| k_smote | 5 | Same-class interpolation neighborhood |
| q_mcsmote | 0.1 | Declared Gaussian bandwidth quantile default; paper tunes per dataset |
| sinkhorn_tolerance | 1e-4 | MCSMOTE marginal convergence threshold |
| max_sinkhorn_iter | 1000 | Engineering safeguard against nonconvergence |
| alpha | 0.05 | Defined Welch failure-to-reject threshold |
| random_state | 42 | Local default, not a verified author seed |
| max_iter | 100 | Engineering bound |
| allow_provisional | False | Explicit opt-in to remaining ambiguities |

Diagnostics record implementation_revision=stage4r3_original_source_generators,
nominal_batch_size, cap policy/budget, effective parameters, original/final
counts, noise/relocation counts, selected classes/seeds, Welch values and status,
drawn seeds/partners/interpolation steps, generated counts and stopping reason.
Generator metadata identifies the original source for RWO and MCSMOTE.
RWO batch records include spread references and scales; MCSMOTE records
transition references, targets, interpolation steps and matrix diagnostics.
They contain no predictive-performance metrics.

    from sklearn.linear_model import LogisticRegression
    from src.ugo import UGO

    sampler = UGO(
        LogisticRegression(max_iter=1000, random_state=42),
        variant="smote",
        cap_interpretation="section_3_3",  # or "table4"
        allow_provisional=True,
    )
    # X_augmented, y_augmented = sampler.fit_resample(X_fit, y_fit)

## Validation and remaining reproduction

Run from the repository root using Python 3.11:

    .venv/bin/python -m pytest -q
    .venv/bin/python main_stage1.py
    .venv/bin/python main_stage2.py --data tests/fixtures/toy.csv --target class
    .venv/bin/python main_stage2.py --data tests/fixtures/toy.dat
    .venv/bin/python main_stage3.py

Existing checks cover MMA, votes, relocation, Welch, seed ranking, ROS copies,
SMOTE reconstruction, input preservation, cloning, random-state behavior and
failure paths. New conformance checks distinguish original-total batches from
both minority and majority counts, verify fixed batches and augmented fits across
iterations for both generators, test both cap policies and fractional/integral
thresholds, check clipped final batches, and reject invalid policy without stale
fitted state.

Stage 4R-3 extends shared UGO checks to RWO and MCSMOTE and adds numerical
generator, LEAO and comparator checks. ROS/SMOTE
samples and generation diagnostics matched four captured pre-refactor cases
exactly. These remain software checks, not reproduced paper results.

The demo remains an artificial 120-row/four-feature make_classification example
with seed 42 and logistic regression. It does not reproduce either paper
simulation or report held-out scores. Its diagnostics are software evidence only.

Stage 4R-4 adds declared simulation reconstructions and classifier/dataset
registries. Stage 4R-5 implements training-only preprocessing, IAM/MCC/F1,
paired ten-fold evaluation, fresh final fitting and atomic resumable results;
see [the evaluation contract](reproduction/EVALUATION.md). All evaluation
software verification uses artificial fixtures. Stage 4R-6 additionally runs
the [measured simulation pilot](reproduction/SIMULATION_PILOT.md), preserving
mixed results, cap termination and unresolved F1 averaging.

Later records include [BFCH](BFCH_SMALL_STUDY.md) and
[EC-EBO](EC_EBO_SMALL_STUDY.md) real-data comparisons,
[all fourteen declared classifier interfaces](reproduction/CLASSIFIER_COMPLETION.md),
the [74-entry input collection](reproduction/DATASET_COLLECTION.md), and an
[interrupted partial benchmark](reproduction/BENCHMARK_EXECUTION.md).
The [SSHR pilot](https://github.com/MAYANK-RATHVA05/rb_ugo/blob/sshr-implementation/docs/SSHR_SMALL_STUDY.md)
is a separate controlled six-method study. These do not constitute full paper
reproduction. Remaining gaps include author code/ambiguous conventions, exact
variant/classifier/fold fidelity, complete benchmark coverage, statistical
analyses and comparison with all published tables/figures. Stop at the requested
scope; future experiments require a separate user instruction.
