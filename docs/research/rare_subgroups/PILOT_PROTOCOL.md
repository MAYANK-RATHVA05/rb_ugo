# Prospective rare-subgroup diagnostic pilot, version 1

Status: prospective design, dated 19 October 2026, now authorized for its first execution. [The JSON configuration](../../../configs/rare_subgroups_pilot_v1.json) drives the frozen runner. Changes after observing pilot results require a new protocol version and must not be presented as preregistered.

Software progress: the baseline and runner have artificial-fixture checks; see [IMPLEMENTATION.md](IMPLEMENTATION.md) and [RUNNER.md](RUNNER.md). The prospective distributions, selection rules, thresholds and decision gates below remain unchanged and are fingerprinted by the runner. Pilot data generation and model evaluation are still pending.

## Question and limits

Does AW-style cleaning discard genuinely labeled rare minority observations, and does that reduce held-out rare-group recall under a validation-selected false-positive-rate target? Does any problem remain after simple baselines and DBSCAN parameter selection?

This is a mechanism check on known synthetic groups, not a new sampler, a paper-score reproduction, or evidence of improvement on real applications. The groups are known because we construct the distribution. Their identities are available only to the evaluator and diagnostic logger. They must never enter features, sample weights, parameter selection, threshold selection, or classifier fitting.

## Controlled distributions

Use two numerical features. Within each independent partition, draw latent components:

| Component | Observed class | Mean | Covariance |
| --- | --- | --- | --- |
| Majority | 0 | (0,0) | I |
| Common minority | 1 | (3,0) | .35² I |
| Rare minority | 1 | (0,3) | sigma_rare² I |

All labels in this first pilot are correct according to their generating component. Overlap of Gaussian distributions is not introduced label corruption; perfect classification is not guaranteed.

Fitting data always contain 1,000 majority and 100 minority rows. Cross `n_rare={4,8,16}` with `sigma_rare={.35,.90}` for **six settings**. Common minority count is `100−n_rare`. This preserves global observed IR=10 while changing how minority observations are allocated; it does not hold the common-group sample size fixed. Larger sigma changes both spread and overlap, so any effect cannot be attributed to density alone.

Use independent population draws for fitting, selection validation, negative-only threshold calibration, and testing:

| Partition | Majority | Common minority | Rare minority | Purpose |
| --- | ---: | ---: | ---: | --- |
| Fitting | 1000 | 100−n_rare | n_rare | Scaling, clustering, resampling, classifier training |
| Selection validation | 2000 | 180 | 20 | Select the tuned AW configuration using aggregate AP only |
| Threshold calibration | 2000 | 0 | 0 | Choose a score threshold using known negative labels only |
| Test | 20000 | 1800 | 200 | One final evaluation after selection |

Validation/test populations keep minority group proportions 90:10 and overall IR=10 across fitting-count settings. Their rare covariance matches the relevant sigma. Counts 4 and 8 underrepresent the rare component relative to its population share; count 16 provides a sensitivity control above that share. This studies variation in subgroup representation in training; it is not an iid random split of one fixed table. Calibration labels and selection-validation labels are additional supervised information and must be reported as such. This pilot makes no label-efficiency claim.

Use seeds **42,43,44,45,46**, giving 30 paired setting/seed cases. Within a seed, share base standard-normal draws across settings: take prefixes for fitting counts and scale the rare draws for sigma. Draw partitions independently. Use `numpy.random.default_rng(SeedSequence([seed,partition_id,component_id]))`, with partition IDs fitting=1, selection=2, calibration=3, test=4 and component IDs majority=0, common=1, rare=2. Use common pools of 100 and rare pools of 16 for fitting, then take the required prefixes. Shuffle assembled rows, carrying evaluator IDs alongside them, using component ID 99 for that partition. Shared draws do not make the six settings independent datasets.

Fit one `StandardScaler` on the **original fitting X only**, then transform all partitions. Do not refit it on synthetic data or on the cleaned fitting subset. No test-derived scaling, epsilon, clusters, or neighbor pools are permitted.

## Methods and classifier

All methods use the same fixed `RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=1.0, bootstrap=True, n_jobs=1, random_state=seed)`. Remaining scikit-learn defaults are recorded with the existing pinned version. This is a diagnostic classifier choice, not one of the paper's four-classifier reproduction claims. Generality across classifiers requires later work.

| ID | Treatment |
| --- | --- |
| NORES | Original fitting rows, no resampling or class weighting. |
| WEIGHTED | Same rows; classifier `class_weight='balanced'`. |
| ROS | Randomly duplicate fitting minority rows to 1:1 using imbalanced-learn. |
| SMOTE | Standard imbalanced-learn SMOTE, k_neighbors=5, to 1:1. Total fitting minority count is 100 even in the rare-count=4 setting. |
| AW_REMOVE | AW-paper-conventions with default parameters; exclude DBSCAN noise originals from final fitting data. |
| AW_SEED_FILTER_RETAIN | Same eligible clusters, quotas, seeds, and synthetic rows as AW_REMOVE; restore excluded originals to final fitting data. This is an ablation, not the paper's KeepNoise. |
| AW_REMOVE_TUNED | Select one AW_REMOVE configuration on aggregate selection-validation AP from MinPts={2,3,5} × epsilon percentile={50,65,80}. No rare-group labels enter selection. |

The tuned control has nine candidate sampler settings; other methods are fixed baselines. State that selection-budget difference in the report. This pilot tests whether simple AW tuning suffices; it does not claim fair exhaustive optimization of all samplers.

All stochastic samplers use the replicate seed in a fresh instance. AW_REMOVE and its retention control must share identical synthetic rows by construction; classifiers also share their seed. Reuse the default AW fit within the tuning grid. Exact repeatability is required within a pinned environment; input row-order effects in DBSCAN ties are recorded, not assumed absent.

### Explicit AW conventions for independent implementation

These resolve gaps identified in [the audit](AW_SMOTE_SOURCE_AUDIT.md). Several are declared choices rather than verified paper rules:

1. Positive/minority label is explicitly 1. Every reordered/filtered feature array carries its aligned y array. Reject nonbinary or nonfinite inputs.
2. Estimate epsilon from the specified percentile (NumPy `method='linear'`) of distances to each minority row's **fifth other minority row**, excluding its own row index. MinPts controls DBSCAN independently. The paper does not fully specify this radius/self convention; it differs from the author release.
3. DBSCAN uses Euclidean distance, `algorithm='brute'`, and the stated epsilon/MinPts. Only surviving clusters of size at least two generate samples. Do not create one all-minority cluster if everything is noise.
4. Boundary counts use each row's five **other** nearest rows in cleaned fitting D; exclude the query's own row index. Use distances then original fitting row ID to break ties. Never use evaluator group IDs. Original fitting row order defines stable row IDs, not hidden subgroup membership.
5. Use maximum cluster boundary count, paper Eqs. (1)–(3). If all cluster weights are zero, use uniform cluster weights, as disclosed in the author release. Within a cluster with zero boundary sum, use uniform boundary weights. Use delta=`1e-8` for density. Use the paper's theta=.2, gamma=5 fusion.
6. Use `floor(normalized_cluster_weight*N + .5)` and correct the difference at the first maximum-weight cluster in DBSCAN label order. Assert nonnegative integer quotas and exact totals; stop and record failure if these conditions cannot be met. Do not silently replace the allocation rule.
7. Use categorical seed allocation. Select any other member of the same cluster uniformly, following the author's neighbor-pool convention; use lambda uniform in [0,1), following the paper's interval rather than the released code's .2–.8 restriction. No cross-cluster interpolation.
8. Set generated count **N=1000−100=900**, using original fitting counts, for all generating treatments in this pilot. If d original minority rows are deleted, AW_REMOVE finishes with `100−d+900` minority rows, not necessarily 1,000. Its retention control, ROS and SMOTE finish 1:1 when successful. Log actual counts; this matched generation count does not match final class totals after deletion. Do not advertise these contrasts as holding every class-ratio effect fixed.
9. If no eligible cluster survives, epsilon is nonpositive, quotas become invalid, or the requested count cannot be generated, mark that AW candidate unsupported with a reason. Do not silently use ROS, preserve an all-noise cluster, or score a missing result as zero. An unsupported tuned candidate is ineligible; if all fail, the tuned method fails for that case. Report coverage alongside every summary.

For selection, maximize aggregate positive-class `average_precision_score` on selection validation. Resolve exact ties by preferring the default (MinPts=2, percentile=65), then ascending MinPts and percentile. Reuse the chosen fitted model; no refitting on validation or calibration rows. Compute test predictions only for the chosen candidate and the fixed comparison methods, after locking selection.

## Thresholds and metrics

For each reported fitted model, choose the **lowest threshold** among the unique calibration-negative positive-class scores plus positive infinity for which empirical calibration FPR is at most **.05**. Predict positive when `score >= threshold`. This handles score ties conservatively and uses no test labels. If no finite threshold qualifies, use positive infinity and report that outcome. A calibration target is not a guarantee that population or test FPR will equal .05.

Primary outcome is held-out **rare-group recall** at that threshold. Also report:

- Common-group recall, overall positive recall, and their denominators, so improvements cannot hide a common-group collapse.
- Actual test FPR, its numerator/denominator, and achieved calibration FPR.
- Positive-class F1 at the chosen threshold and threshold-free average precision (AP, scikit-learn's noninterpolated definition).
- Worst-group recall, the threshold, fitting class totals, actual generation count, and completion status.
- Evaluator-only group retention fractions, cluster membership, noise flags, cluster quotas, and synthetic seed/partner row IDs. Synthetic samples have provenance, not oracle true subgroup labels; do not label their correctness by their seed alone.

Log seed/cell/configuration, selected parameters, input/output hashes, library versions, timings, and structured failure reasons. Diagnostics may read hidden group IDs only after the method has finished; the model and sampler interfaces receive X/y only.

Report all 30 paired cases, then mean and sample SD across five seeds separately for each of the six settings. Compare on the same completed cases and disclose missing coverage before computing any average. Avoid ranking means that use different case sets. Show rare recall, common recall, FPR, F1 and AP separately; never average these unlike metrics into one score. Five seeds and shared synthetic families do not justify a claim of statistical significance or 30 independent real datasets.

## Scope and decision gates

The six fixed methods plus eight additional AW grid fits give at most **14 classifier fits per case, 420 total**. The selected tuned model is reused. Seven report rows per case give 210 planned result rows. Unsupported cases reduce executed fits/rows with status retained. This is a count budget, not a measured runtime promise.

Before any fitting, the next software stage must check aligned boundary labels, self-neighbor exclusion, tied/zero distances, all-noise and zero-weight behavior, integer quota totals, interpolation interval/cluster constraints, repeatable RNG behavior, paired synthetic identity for the retention control, and isolation of preprocessing and evaluator-only IDs. Those are correctness checks, not new algorithms.

Predeclared diagnostic screening rules, chosen for practical effect size rather than statistical proof:

1. A deletion-related effect is worth further investigation only if a group loses original rows **and** AW_SEED_FILTER_RETAIN exceeds AW_REMOVE rare recall by at least .05 on average in at least two settings, with a positive paired difference in at least four of five seeds for each setting. Mean absolute paired test-FPR difference must be at most .01 in those settings; otherwise calibration drift leaves the comparison unresolved. Always show all settings and FPR differences. Retention also changes final training counts, so this is not a pure class-ratio-free causal estimate.
2. Do not claim a residual failure in a qualifying setting where tuned AW, ROS, SMOTE, or the weighted baseline achieves mean rare recall at least `retention_control − .02`, mean common recall at least `retention_control − .02`, and mean absolute paired test-FPR difference at most .01 relative to that control. If simple controls suffice in every qualifying setting, stop this line of method development on the basis of this pilot. Even otherwise, a new algorithm is not yet justified by this pilot alone.
3. Even if a residual loss remains, this clean-label pilot cannot show that a remedy distinguishes rare valid observations from actual label noise. A separate, prospectively specified contamination study and real data with independently meaningful subgroup identities are required before a noise-robustness or novelty claim.
4. A negative or inconclusive outcome is a valid result. Do not add favorable settings, seeds, thresholds, or baselines after seeing test scores and present them as confirmation. Any follow-up is a new version with a new rationale.

Defer the other paper implementations, true-label contamination, additional classifiers, real dataset selection (including a possible multiclass-to-binary subgroup benchmark), new sampling rules, and the previous 30-variant benchmark. They are outside this first diagnostic stage.
