# Original UGO reproduction specification

Stage 4R-1: source inventory, paper-to-code specification and recovery review.
Prepared 2026-10-11. This stage changes documentation only; no algorithm repair,
data acquisition, final-model training or benchmark experiment is executed here.

Current implementation update: Stage 4R-2 has since recovered the fixed-batch
and explicit-cap corrections. See [CORE_ALIGNMENT.md](CORE_ALIGNMENT.md) for
that stage's validation evidence. Baseline-status statements below describe
commit `c2fa7f3`, preserved as the specification's starting point.
Partial Stage 4R-3 now provides a shared batch interface and provisional numerical
RWO/UGO-RWO from a verified third-party archive; original RWO verification,
MCSMOTE and LEAO remain pending. See [GENERATOR_ALIGNMENT.md](GENERATOR_ALIGNMENT.md).

## Objective and baseline

Reproduce the original UGO framework and its reported experiments, tables and
figures. RB-UGO, DRA-UGO, held-out batch acceptance and new research methods are
outside this reproduction.

Starting main: `c2fa7f333689df14c5ab4bb542500bf2676ac897`.
Working branch: `ugo-paper-reproduction`. Keep the original Stage 3 history
intact. Read [AGENTS.md](../../AGENTS.md) and validate each separately authorized
stage before committing and pushing; do not silently advance to the next stage.

The baseline has CSV/KEEL loading, descriptive statistics, and provisional binary
UGO-ROS/SMOTE with a logistic-regression artificial demo. The previously verified
suite has 147 tests, including 81 Stage 3 cases. It has no benchmark evaluation,
paper simulation generator, result pipeline or published-result reproduction.

A fresh final classifier must be fitted after resampling. The current
`estimator_` is the last uncertainty-scoring model and can predate the last
generated batch; it is not automatically a final evaluation model.

## Source inventory and access

The source of truth for available artifacts and hashes is
[SOURCE_REGISTER.json](SOURCE_REGISTER.json).

The complete uploaded 20-page article has been read:
Sağlam and Cengiz, *Uncertainty guided oversampling: A classifier-specific approach
for effective handling of class imbalance in machine learning*, Knowledge-Based
Systems 342 (2026), 115886, DOI
[10.1016/j.knosys.2026.115886](https://doi.org/10.1016/j.knosys.2026.115886).
Page references below use the printed/PDF page numbers, which agree.

The current PDF is 6,672,332 bytes, SHA-256
`42a26a8997efc28bc50ef2e58ddf28eae10d3a9bcec19fe31a91756560e30f6f`.
An older branch records a differently hashed PDF of the same article. Do not
claim the two files are byte-identical or text-identical without comparison.
Do not commit uploaded PDFs or real datasets. Retain originals outside Git;
record hashes and portable provenance rather than transient attachment paths.

No supplementary package or author implementation is available here. The paper
explicitly refers to the supplement for the 74-dataset list and detailed results.
It does not establish that the supplement necessarily includes code, seeds,
package versions or fold files; those are requirements to investigate.

On 2026-10-11, native requests to doi.org, www.sciencedirect.com and
api.crossref.org failed at proxy CONNECT with 403. No publisher response body
was obtained. This is an access observation, not evidence that the supplement
does not exist or that the publisher denied article access.

Next source actions:

- Obtain the publisher's actual supplementary artifacts or unchanged user
  downloads; preserve originals, record hashes, then inspect working copies.
- Locate author code and configuration if published; otherwise retain
  `not_located`, not a fabricated link.
- Obtain authoritative RWO, MCSMOTE and LEAO specifications/implementations.
- Record exact data releases, binaries, folds and software versions when found.
- Continue independent software work while materials remain missing, but do not
  claim exact results where required evidence is absent.

## Reusable work on the historical branch

GitHub branch
[stage4ar-paper-data-verification](https://github.com/MAYANK-RATHVA05/rb_ugo/tree/0d78c2d9db22f92cee912c94671705d4cd5ef8b9)
contains reusable work absent from current main.

| Commit/artifact | Inspected content | Recovery decision |
| --- | --- | --- |
| [0866dc9](https://github.com/MAYANK-RATHVA05/rb_ugo/commit/0866dc936f231feef9df5e1538b8df206741735d) | Fixed original-total batch; Section 3.3/Table 4 cap choices; ceil threshold; updated tests, demo wording and audit | Selectively reuse in the next core-repair stage; not applied in Stage 4R-1 |
| [41dfe2f](https://github.com/MAYANK-RATHVA05/rb_ugo/commit/41dfe2f) and branch scripts | Offline imports, KEEL verification and fold-gated smoke scripts | Reassess for ten-fold reproduction requirements; do not inherit the earlier five-fold gate as a universal paper rule |
| Historical data/resource manifests | Eight full-file inspection records and PDF inventories | Preserve as historical evidence; revalidate originals before claiming current availability or provider provenance |
| DRA research documents and project-scope edits | Separate research proposals | Do not import into original-UGO scope |

The core correction's tests check fixed batches over multiple iterations, named
caps and fractional threshold behavior. Review and run those tests after recovery;
inspection alone is not fresh execution of that branch.

The historical manifest names Ecoli2, Glass6, Yeast3, Page-blocks0, Ecoli4, Yeast4,
Segment0 and Abalone19. It records no official folds in their uploaded archives.
No original archives are available in this task. File inspection does not prove
membership in the paper's 74, official provenance or predictive performance.
Ecoli and Yeast variants are related sources, not independent origins.

## Algorithm requirements and declared choices

Let N be the original fitting size, K=2, Nmax the largest original class count,
and nsyn the accumulated generated count. UGO receives fitting data only.

| ID | Paper evidence | Requirement / baseline status | Validation for the implementation stage |
| --- | --- | --- | --- |
| U01 | §3.2, p6 | MMA=max competing probability minus true-label probability; binary form 1-2P(true). Present | Exact hand values, reversed probability columns, invalid distributions |
| U02 | §3.4, p7; Table 2, p8 | Pre-detect neighbor-majority disagreement; default k_pre=5. Present | Self exclusion by identity, coincident rows, same/opposing votes, ties, small N |
| U03 | §3.1, p5 | Refit probability model after augmentation. Present; Table 4 omits in-loop X/Y update | Spy on increasing fitting sets; input preservation and probability ordering |
| U04 | §§3.3–3.4, pp6–7; Table 4 | Compare clean class MMA means and choose argmax, possibly majority. Present | Controlled class scores and majority selection |
| U05 | §3.4, p7 | Rank eligible seeds descending; n_select=n^(c)*lambda_select. Present with a count convention | Known rankings; noisy exclusion; count/rounding examples |
| U06 | §3.4, p7; Table 4 step 2, p8 | Fixed nominal batch=N*lambda_step. Main differs | Recover fixed-total implementation; independent tests distinguish N from class size |
| U07 | §4.2 and Table 4, p8 | ROS and SMOTE anchored at selected seeds; SMOTE k=5. Present | ROS copies, exact interpolation and selected-seed provenance |
| U08 | Table 3 and Table 4, p8 | Final votes query generated points against X; k_after=1. Present | Reference-pool spy, generated labels, opposing votes |
| U09 | §3.4 and Table 4 | Relocate generated noise to nearest clean same-class reference. Present | Actual destination coordinates and no eligible destination |
| U10 | §3.3, p6 | Stop on Welch non-rejection. Present as equivalent binary two-sided t-test | Two-group ANOVA equivalence, both decisions, degenerate groups |
| U11 | §3.3 versus Table 4 | Conflicting cap expressions. Main exposes one only | Recover named policies; validate threshold arithmetic and diagnostics |
| U12 | §4.2, p8 | UGO-RWO and UGO-MCSMOTE. Missing | Source-backed generator equations and common UGO behavioral contracts |

Source conflicts and reproduction policy:

- **Class direction:** introductory p3 says lowest uncertainty; detailed
  §§3.3–3.4 and Table 4 say highest. Follow the detailed argmax rule.
- **Fixed batch:** follow Table 4's pre-loop initialization. Its later narrative
  mentions increasing generation per iteration but supplies no replacement
  equation. Nominal size `max(1, ceil(N*lambda_step))` is a declared rounding
  convention, not a verbatim author rule.
- **Caps:** expose `section_3_3` as the primary declared source policy,
  `max(0, ceil(phi*K*Nmax-N))`, and `table4` as the alternative
  `max(0, ceil(phi*Nmax))`. Freeze original counts. With counts 90/10 and phi=2.5,
  thresholds are 350 and 225. Ceil is the minimal integer meeting a real threshold;
  counts frozen at originals and rounding must still be disclosed.
- **Clipping:** retain last-batch clipping as a finite-budget safeguard;
  Table 4 checks its cap before a full batch and can overshoot. Report this
  deviation and assess sensitivity if needed; do not label clipping author code.
- **Pools:** provisionally retain augmented classifier fits, original clean rows
  for uncertainty/seeds/SMOTE partners, all original rows for final votes, and
  original clean same-class rows for relocation. Whether synthetics enter later
  scoring/selection pools is unresolved; record all pool policies in results.
- **Seed count:** provisionally retain ceil(lambda_select*clean eligible count),
  minimum one. Whether n^(c) means complete class or clean eligible count is
  unresolved; do not silently treat the choice as confirmed.
- **Neighbors:** retain Euclidean distance, self exclusion by row index, stable
  equal-distance ties, assigned-label retention on tied votes and effective-k
  clamping as documented conventions. Tables 2–3 do not uniquely specify all
  these choices; recommended odd k avoids ordinary vote ties.
- **Safeguards:** distinguish undefined Welch, insufficient neighbors and
  max_iter=100 from published statistical/cap stopping. Non-rejection does not
  prove uncertainty equality, balance of risk or improved predictive performance.
- **Interpretation:** MMA and neighbor noise masks are heuristic proxies;
  do not claim an identified epistemic/aleatoric decomposition or noise-free labels.

Paper defaults: lambda_select=.5, lambda_step=.1, phi=2.5, k_pre=5, k_after=1,
SMOTE k=5. Use seed 42 for the local default and record it; author seeds are
unknown. Report engineering conventions separately from paper defaults.

## Dataset and simulation specification

Benchmarks: 74 binary variants from UCI/KEEL, IR>5; p8 §4.1.2 and Table 5 p10.
The aggregate ranges (192–17,898 rows, 2–54 features) corroborate a manifest;
they cannot identify the exact dataset collection.

For each artifact require source/release, checksum, license, exact binary target,
positive outcome, feature types, class counts, missingness, duplicates, supplied
fold identities and any transformation. Do not infer membership or licensing from
a familiar filename. Preserve legitimate duplicate records; flag identity
ambiguity and overlap rather than automatically deleting rows to improve results.

Two-dimensional simulations, §4.1.1 pp7–8 and Figure 7 p10:

| Design | Stated construction | Evidence still needed |
| --- | --- | --- |
| Linear | Majority N(0,3), minority N(6,2), minority-label noise; illustration has nP=65, nN=985 | Whether second normal argument is SD/variance, original counts, independence/covariance, noise denominator and seeds |
| Nonlinear | X1,X2 from N(0,1); Z=sin²(X1)+cos(X2)+sin(X1)cos(X2); label is rounded logistic(Z); majority-to-minority relabeling; illustration nP=85, nN=915 | Exact sampling/imbalance construction, pre-noise versus post-noise counts, rounding boundary and seeds |
| Noise sweep | Twenty datasets move class centers closer, beginning at separation 5; §5.1.1 p11, Figures 9–12 | Movement increments, reuse versus regeneration, seeds and precise noise/overlap definition |

The text describes 30% label noise. Do not equate that without evidence to
flipping 30% of all rows. The current 120-row four-feature make_classification
demo is not a paper simulation.

## Comparator and classifier configurations

Section 4.2 p8 names eleven conditions: NORES, RUS, ROS, SMOTE, RWO, MCSMOTE,
LEAO, UGO-ROS, UGO-SMOTE, UGO-RWO and UGO-MCSMOTE. Tables 9–11/A1 display ten,
omitting RUS. Build method availability toward all eleven, but label RUS as an
additional condition unless the intended author grid is resolved. Never substitute
a different unavailable method under its name.

Tables 7–8 pp11 describe fourteen classifiers:

| Classifier | Settings stated by the article | Outstanding implementation detail |
| --- | --- | --- |
| BART | ntree=20 | Package/version, priors, posterior probability interface |
| BNNET | 2 layers, 20 nodes, dropout .25, warm-up 100, RMSprop, patience 20 | Layer interpretation, priors, batch/epoch settings and inner early stopping |
| C5.0 | CF=.25, earlyStopping=TRUE | Native estimator/package; sklearn trees are not equivalent |
| CART | cp=.01, minsplit=20 | Native pruning semantics; no silent sklearn parameter substitution |
| GKNB | Unbiased cross-validation bandwidth estimation | Kernel, bandwidth implementation and inner training-only fitting |
| GNB | No additional settings given | Package defaults/version and variance handling |
| KNN | k=5 | Scaling, distance, tie and probability conventions |
| LDA | No additional settings given | Solver, covariance assumptions and package defaults |
| LogReg | maxit=1000 | Penalty, solver and other defaults; sklearn default regularization is not automatically equivalent |
| NNET | Same stated architecture/training settings as BNNET | Exact architecture, training and probability outputs |
| QDA | No additional settings given | Covariance regularization and singular-class handling |
| RF | 500 trees, mtry=sqrt(p), unlimited depth, min.node.size=1, min.bucket=1 | Feature-count rounding, package semantics and seed handling |
| SVMLinear | C=.95 | Probability estimation and package/kernel conventions |
| SVMRadial | C=.95, sigma=.5 | Package-specific sigma/gamma mapping and probability estimation |

§4.4 p10 states 100 posterior samples and model averaging for Bayesian models.
Do not assume probabilistic interfaces across languages are interchangeable.
If authoritative packages differ from the chosen implementation, disclose the
substitution and classify affected fidelity accordingly.

## Evaluation and analysis contract

§4.4 pp10–11 specifies 10-fold CV, IAM/MCC/F1, Friedman tests,
Bonferroni-corrected paired Wilcoxon comparisons and significance .05.

For every outer fold:

1. Establish author folds if available. Otherwise use recorded deterministic
   stratified ten-fold assignments as a declared approximate protocol, respecting
   provider grouping/time constraints. Never relabel generated folds as official.
2. Fit all feature transformations on training rows only: missing-value handling,
   categorical representation, scaling and the stated within-class variance filter.
   The exact variance-filter interpretation/order is unresolved and must be fixed
   before runs. Do not drop test rows or use their labels for feature decisions.
3. Fit calibration, tuning or early stopping using inner training partitions only.
4. Resample training rows only. Test rows never enter noise, seed, uncertainty,
   neighbor, generation or stopping calculations.
5. Fit a fresh final classifier matching the uncertainty-classifier family and
   declared settings on augmented training data; evaluate on the untouched test.
6. Persist fold assignments, predictions, metrics, method/estimator configuration,
   seeds, hashes, versions, runtime, sample counts and UGO diagnostics atomically.

Use fixed domain/source label meanings rather than learning the positive outcome
from outer-test frequencies. The Stage 2 file-level minority labels are inspection
metadata, not permission to use test labels for training decisions.

Fail or explicitly mark infeasible ten-fold configurations with very small classes;
do not silently reduce folds, switch methods or replace undefined scores with
favorable values. Preserve failure/skipped counts and comparable paired coverage.

Metric definitions:

- Standard true-row/predicted-column TP/TN/FP/FN; Table 6 p10 swaps FP and FN
  labels. Do not copy that error into FPR or class-specific measures.
- F1=2TP/(2TP+FP+FN), with declared positive outcome and zero-denominator policy.
- MCC=(TP*TN-FP*FN)/sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)), with explicit degenerate policy.
- IAM, §4.3 p9: mean over classes i of
  (Cii-max(sum off-diagonal row i, sum off-diagonal column i)) /
  max(row-total i, column-total i). Define matrix orientation and absent-class
  behavior; test hand-computed matrices and class permutations.
- The article's random-classifier F1=.5 claim is not a universal oracle.

Aggregate using the author unit of comparison where established. Paired folds
and multiple classifiers on one dataset are not independent datasets.
Resolve classifier-group aggregation, tied ranks/wins, missing outcomes and the
Bonferroni family before significance reporting. Keep NORES in every comparison.
Avoid adapting parameters after seeing outer-test performance.

Uncertainty reduction (§5.3, Table 12 p17) must use a defined before/after
population and aggregation. MMA is signed; no naive percentage division by
zero/negative starting MMA. Recompute after the final fit if that is the declared
definition; do not use stale internal estimator state.

Sample-generation analysis (§5.4 p15): fifty UGOSMOTE repetitions, generated
counts and rates versus the amount needed for numerical balance. Define handling
of already-balanced inputs, majority selection and all-noise no-ops. Author seeds,
grouping and aggregation still need clarification.

## Table and figure reproduction map

| Target | Required output / dependency |
| --- | --- |
| Table 1, Figures 1–4 | Method/background illustrations, noise examples, workflow and MMA mechanics; distinguish recreated illustrations from measured experiments |
| Tables 2–4 | Algorithm conformance report and disclosed choices |
| Figure 5 | First-iteration noise, MMA, seeds, generation and relocation using the stated NNET configuration |
| Figure 6 | Final resampled simulation distributions and classifier-specific differences |
| Figure 7, Tables 9–10 | Both simulations, IAM/MCC/F1 for the agreed methods/classifiers |
| Table 5, Figure 8 | Statistics/distributions of the verified 74-dataset collection |
| Tables 6–8 | Correct metric orientation, classifier groups and recorded model settings |
| Figures 9–12 | Twenty-dataset overlap/noise sweep and IAM/MCC/F1 curves |
| Table 11, Figure 13 | Dataset-based average ranks, wins and rank distributions by classifier group |
| Figure 14 | Corrected paired comparisons with complete coverage/multiple-testing records |
| Table 12 | Defined uncertainty reduction by classifier group |
| §5.4 outputs | Fifty-repeat generation counts/rates and fewer-than-full-balance summaries |
| Table A1 | Mean benchmark metrics as secondary evidence |
| Supplementary outputs | Dataset/classifier-level results and count tables once the actual package is obtained |

Several narrative table/figure references are incorrect. Match actual captions
and contents, not a misleading in-text number. Every reported comparison needs
paper value, reproduced value, protocol/source differences and fidelity status.

## Implementation checklist and stage gates

All following stages are planned, not completed by this specification.

| Stage | Work | Exit evidence |
| --- | --- | --- |
| 4R-1 (this stage) | Source inventory, audit of existing work, requirement map and unresolved evidence | Valid hashes/links/refs, explicit source gaps; docs committed and pushed |
| 4R-2 | Selective recovery and completion of binary ROS/SMOTE core | Source-conformance tests, both cap diagnostics, documented deviations; no benchmark claim |
| 4R-3 | Verified RWO/MCSMOTE generators, UGO wrappers and other comparators | Original-source equations, implementation identity, generator and edge-case tests |
| 4R-4 | Simulations, dataset registry and fourteen classifier configurations | Verified available artifacts, declared unavailable ones and train-only probability smoke checks |
| 4R-5 | Preprocessing, metrics, ten-fold runner and resumable result schema | Leakage checks, independent metric examples, repeat/resume correctness and fresh final fitting |
| 4R-6 | Small pilot, then scoped expansion | Representative simulations/data, failure accounting, measured runtime/memory and valid result comparison |
| 4R-7 | Full selected benchmark matrix and additional analyses | Complete declared coverage, no fabricated/imputed successful runs, fifty-repeat/overlap evidence |
| 4R-8 | Automated tables/figures and final report | Reproduction differences, statistical coverage, configs and independently rerunnable artifacts |

A pilot should first use NORES/ROS/SMOTE/UGO-ROS/UGO-SMOTE and simple classifiers,
then expand as verified generators/models become available. It need not wait for
every external implementation to diagnose the evaluation engine.

The proposed eleven-condition benchmark grid has 74*14*11*10=113,960 outer fold
runs; the displayed ten-condition grid has 103,600. Neither is a confirmed author
run count. UGO internal fits and additional analyses increase cost. Set
parallelism and scheduling from a measured pilot, not an unsupported calendar
estimate. Version and invalidate affected runs if code changes.

Completion labels must separately report source availability, implemented
components, tests, executed datasets and reproduced results. No overall percentage
or blanket exactness claim is justified. Missing source details block affected
exact-result claims, not independent software progress.
