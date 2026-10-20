# Bounded pilot, fixed before the first measured run

This is an explicitly authorized research feasibility stage. Existing UGO,
BFCH, EC-EBO, SSHR, AW, NADOS and Soft-NADOS files/results remain intact.
See [source audit](SOURCE_AUDIT.md) before interpreting this candidate.
The machine-readable design is
[original_support_hs_v1.json](../../../configs/original_support_hs_v1.json).

## Question and fixed method

Does replacing augmented parent-node counts with original-fitting-row support
improve predictions over **tuned standard HS** after SMOTE? Only the count
denominator changes. Both B and C below share one already fitted, full CART
with identical structure and pre-shrinkage probabilities, including the root.
Neither method refits the tree. Original support comes from actual original
fitting rows' paths, including every actual duplicate once per observation.
No synthetic ancestry is converted to original support. Zero support and
the uniform-inflation/lambda identity are specified in the source audit.

No claim is made that original counts are a proven effective sample size or
that augmented counts caused AW-SMOTE's published decision-tree weakness.
This is standard SMOTE, not AW-SMOTE or SMOTEWB. Shrinkage cannot undo split
selection or reduce depth/leaf count. A successful pilot would motivate
broader evaluation rather than prove the proposed mechanism or novelty.

## Data and paired seeds

Use three already local, hash-verified numeric KEEL variants from different
source families: Ecoli4 (336 rows, 7 features, 20 positives), Glass6 (214, 9,
29 positives), Yeast4 (1484, 8, 51 positives). Raw SHA-256 values and label
contracts are in the JSON. Author preprocessing/folds are unverified. These
choices were made using availability/counts, without new test performance.
They appeared in prior work, so this is feasibility, not an unseen benchmark.

Five seeds: 42–46. Exact raw-feature-coordinate groups remain within one
partition, preventing duplicates crossing splits. First shuffled stratified
group fold of five is test (about 20%); first group fold of four within the
remainder is validation (about 20% overall); remaining rows fit the models
(about 60%). No redraw if class coverage fails. Original subgroup labels do
not exist for these inputs, so no real-data rare-recall claim is reported.

One controlled two-dimensional Gaussian experiment has 1000 majority, 80
common minority and 20 rare minority fitting examples. Independent validation
has the same counts; test has 5000/400/100. Distribution means/SDs and keyed
partition streams are frozen in JSON. At concentration c=0/.5/1, keep the
originals, validation, test and total synthetic count identical. Interpolate
using three nearest minority partners and uniform fractions, as in SMOTE.
Vary only seed probabilities: mixture of uniform seed weights and Gaussian
weights around the minority row with greatest first standardized feature
(bandwidth .75). Anchor, weights and partners use fitting features/observed
labels only; latent subgroups are joined for evaluation after selection.
This is a declared controlled SMOTE variant, not the stock library generator
for c>0. Even c=0 uses the controlled generator's declared stable RNG/ties.
Same random uniforms across levels pair the experiments. Group labels are
never inputs to the sampler, tree, shrinkage or validation selection.

Total: 15 real cases + 15 synthetic cases = **30 cases**. Concentration levels
share original populations; they are not three independent datasets.

## Arms and tuning budget

| Arm | Model | Validation candidates |
| --- | --- | ---: |
| A | Stock SMOTE + tuned CART | 12 structure settings |
| B | Stock SMOTE + standard HS | 12 lambdas, fixed shared full CART |
| C | Stock SMOTE + original-support HS | Same 12 relative lambdas, same CART as B |
| D | Balanced class-weighted CART, original data | Same 12 structure settings as A |
| E | Unweighted CART, original data | Same 12 structure settings as A |

On synthetic cases the controlled concentration generator replaces stock
SMOTE identically in A/B/C. E is the required no-resampling baseline.
Structural grid: max_depth 2/3/4/5/8/None crossed with min_samples_leaf 1/5.
The HS base is full CART, min_samples_leaf=1, other sklearn defaults, no
sample weights. HS is tuned via lambda rather than structural changes.
This fixes **final selected B/C structure**, not only structure per candidate.
Relative lambdas: 0/.001/.003/.01/.03/.1/.3/1/3/10/30/100. Each scheme uses
lambda=relative*its root count. This gives identical candidate predictions
if support inflation is proportional everywhere, a required confound control.

Scale on original fitting rows only. Stock SMOTE k=3, balance to ratio 1.0,
seed=replicate. No label noise is injected. All candidates are evaluated on
untouched validation observations. Select validation AP, then validation
minority F1, then original grid order. The prediction cutoff is fixed at .5.
Test rows/labels are inaccessible to the tuning API. Do not refit on
validation after selection: selected fitting-only model is evaluated once.

Each arm gets 12 validation evaluations; total **1800 candidates**. Tree
fitting is cached where HS is post-hoc: 12 A + 1 shared B/C + 12 D + 12 E =
37 fits per case, **1110 tree fits**. B and C have equal structure, exact
lambda-grid sizes and data. A/D/E tune structure, so budgets are comparable
selection budgets, not equal numbers of expensive tree fits per arm.
Runtime records report shared B/C fit cost in both per-arm times; those
times must not be summed as unique total wall time.

## Metrics, uncertainty and stop rule

Primary: average precision (sklearn non-interpolated AP). Secondary: minority
F1, threshold .5. Report rare recall only for known synthetic group 2.
Publish case/seed values and C-minus-control mean, sample SD and descriptive
t(4) 95% interval across the five paired seeds. Repeated holdouts overlap;
these intervals describe seed variation, not independent-dataset inference.
No significance claim or fold-as-independent-dataset test.

Before viewing results, advancement requires **all**:

- Full coverage; no discarded/replaced case or candidate failure.
- At least two of three real datasets gain mean AP >= .01 over tuned B.
- Real macro AP gain >= .01 and mean F1 change >= -.01 versus B.
- No real dataset loses mean AP by more than .02 versus B.
- Synthetic mean AP gain >= .005 and mean rare recall change >= -.02 versus B.

Average equally across datasets/concentration levels, then paired seeds.
Comparisons with A/D/E are contextual, never a substitute for beating B.
Thresholds are practical feasibility criteria, not proofs of significance.
If unmet, stop this exact candidate; do not automatically broaden the
benchmark or tune after observing held-out results.

## Evidence and execution

Commit the implementation/config before running. Save exact split indices,
input hashes, source commit, package versions, scaler, candidate validation
scores, selected parameters, shared-tree fingerprints, test probabilities,
HS arrays/counts, runtime and failure messages. Raw inputs stay ignored.
Compact results go in this research directory, with no graphs required.

```powershell
.\.venv\Scripts\python.exe -m scripts.run_original_support_hs_pilot --check-only
.\.venv\Scripts\python.exe -m scripts.run_original_support_hs_pilot --output results\original-support-hs-v1
```

Use a fresh output directory; the runner never overwrites prior evidence.
Required correctness checks include parent-count arithmetic, zero support,
lambda=0 identity, original paths, proportional inflation/lambda rescaling,
probability validity, identical B/C raw tree state, fixed selection budgets,
and independence of selection from changes to final test features/labels.
