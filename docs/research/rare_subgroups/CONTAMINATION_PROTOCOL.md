# Prospective rare-group and label-contamination follow-up, version 1

Status: **prospective design frozen in protocol commit `bff7ce5`, 19 October 2026**. The first experiment completed on run commit `975020a`; [audited measured results](CONTAMINATION_RESULTS.md) are now available. All 630 method evaluations succeeded, but no base setting met both screening conditions. The [configuration](../../../configs/rare_subgroups_contamination_v1.json) records the complete design separately from the clean-label pilot; the scientific design and decision rules below retain their pre-run values.

In plain language: keep the genuine rare examples, deliberately give a few majority training examples the wrong minority label, and check which existing methods protect the rare group without amplifying those errors. This stage specifies that check; it does not propose a new sampler.

## Why a new study is needed

The [first pilot](PILOT_RESULTS.md) found rare-group losses under AW removal, but only one setting met its predeclared screen and ordinary controls performed well. Its observations were all correctly labeled. It cannot show that a method distinguishes genuine sparse groups from mislabeled examples.

This follow-up is explicitly informed by those results. It is not a successful extension of the first screen, an independent confirmation, or an external preregistration. We freeze its rules before generating its contaminated cases or examining their outcomes. The paper already discusses sparse legitimate observations, label noise and retention alternatives; see the [source audit](AW_SMOTE_SOURCE_AUDIT.md). Another noise table or simple retention rule would not establish novelty.

Questions:

1. Does AW removal discard genuine rare observations, mislabeled majority observations, or both?
2. Does retaining originals protect rare recall but become more vulnerable to label contamination?
3. Do NORES, weighting, ROS, SMOTE or validation-tuned AW already address the observed tradeoff?

## Paired data and exactly what is corrupted

Use the first pilot's two-dimensional Gaussian populations, independent partitions and seeds **42–46**. Majority has mean (0,0) and covariance I; common minority has mean (3,0) and covariance .35² I; rare minority has mean (0,3) and covariance sigma² I. Cross rare fitting counts **4, 8, 16** with sigma **.35, .90**. These six base settings remain related synthetic conditions, not independent real datasets.

Before corruption, each fitting table contains 1,000 true majority observations, 100−r true common minority observations, and r true rare observations. For each base setting and seed, create three versions of the **same feature rows** by flipping exactly **m = 0, 5, 10** true-majority fitting labels from 0 to 1. Never change a genuine minority label, feature value, row identity or row order between these versions.

| Flipped fitting labels m | Observed majority | Observed minority | Wrong labels among all 1,100 rows | Wrong labels among observed minority | Requested synthetic rows |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1000 | 100 | 0% | 0% | 900 |
| 5 | 995 | 105 | 0.4545% | 4.7619% | 890 |
| 10 | 990 | 110 | 0.9091% | 9.0909% | 880 |

The observed imbalance ratios are 10, approximately 9.4762, and 9. These are counts of one-way errors, not 5%/10% corruption of the full dataset. This construction preserves the true rare/common sample sizes and all features. Changes in observed class balance, weights and requested generation count are consequences of label corruption and must be reported; the study does not hold those algorithm responses fixed.

Selection validation stays clean with 2,000 majority, 180 common and 20 rare observations. Calibration stays clean with **2,000 known negatives only**. Test stays clean with 20,000 majority, 1,800 common and 200 rare observations. Within a base setting/seed, these partitions are identical across all noise levels and methods. Rare spread still changes overlap as well as density. This is a supervised synthetic mechanism study, not a claim about learning without clean validation labels.

### Random streams and pairing

Preserve the original pilot's Gaussian stream convention: `default_rng(SeedSequence([seed, partition_id, component_id]))`; partition IDs fitting=1, selection=2, calibration=3, test=4; component IDs majority=0, common=1, rare=2, shuffle=99. Use fitting common/rare pools of 100/16 with prefixes, as before. The noise-zero X, observed y and latent groups must reproduce the first pilot's partition fingerprints.

Define majority stable IDs 0–999 **before shuffling**. Independently generate their corruption ordering with `default_rng(SeedSequence([seed, 5, 100])).permutation(1000)`. Flip the first m IDs and carry their flags through the existing shuffle. Thus the five flipped rows are a subset of the ten, and the same ordering applies across all six base settings for a seed. Selection of corrupted rows cannot depend on distance, density, subgroup geometry or future performance.

Hidden true fitting labels, component IDs and corruption flags belong only to the evaluator. Sampler/classifier interfaces receive **X and observed y**. Neither tuning nor weighting may inspect hidden truth. Fit `StandardScaler` once on original fitting X only; since X is unchanged, its parameters must be identical across noise levels. Transform selection/calibration/test using that scaler.

## Methods, fitting and generation conventions

Keep the same seven treatments and fixed 100-tree random forest from the [first protocol](PILOT_PROTOCOL.md). The complete classifier, AW conventions and nine-setting tuning grid are copied into the new JSON, not inherited from changing defaults.

| Method | Fitting behavior |
| --- | --- |
| NORES | Original X and observed y, no resampling or weighting. |
| WEIGHTED | Same data; `class_weight='balanced'` from observed labels. |
| ROS | Resample observed minority to match observed majority; log duplicated original IDs. |
| SMOTE | Standard SMOTE, k_neighbors=5, from observed minority rows. |
| AW_REMOVE | Existing AW-paper-conventions, default MinPts=2/percentile=65; remove DBSCAN-noise originals. |
| AW_SEED_FILTER_RETAIN | Restore all originals around the exact same generated rows as AW_REMOVE; excluded rows remain ineligible as seeds. |
| AW_REMOVE_TUNED | Select MinPts 2/3/5 × percentile 50/65/80 by aggregate clean selection AP only. |

All stochastic methods use a fresh instance with the replicate seed. Keep the original RF settings, original-fitting scaling, cluster weights, boundary conventions, quotas, interpolation and explicit unsupported conditions. This is still **AW-paper-conventions**, not an exact verified paper reproduction or the paper's KeepNoise.

Request **N = observed n0 − observed n1 = 900−2m** for every generating treatment. If AW removal excludes d observed-minority originals, it finishes with 1000−m−d observed minority and 1000−m observed majority rows. ROS, SMOTE and the retention control finish at observed 1:1 when successful. Log requested/actual generation and final counts. The removal/retention contrast changes final training size and class ratio; it cannot isolate rare-row deletion alone.

Maximize aggregate `average_precision_score` on selection validation for AW tuning. Reuse the default fit in its nine-candidate grid; exact ties prefer default, then ascending MinPts and percentile. Reuse the chosen fitted model without refitting. Unselected candidates never predict test data. Other controls remain fixed, so their selection budgets differ from tuned AW.

## Threshold calibration and score ties

The previous forest's coarse scores often produced calibration FPR far below .05. This study prospectively changes the threshold rule to **independent randomized tie handling**, targeting exactly 100 positive predictions among 2,000 calibration negatives. It is a change in evaluation protocol, not a sampler improvement. Do not compare its thresholded scores directly against the first pilot as if their operating rules were identical.

For calibration and test separately, draw one independent uniform per row with `default_rng(SeedSequence([seed, partition_id, 101])).random(n_rows)`. Reuse each partition's vector across methods, noise levels and base settings within a seed. These uniforms are not features or model inputs and do not enter AP or AW selection.

After model selection locks, order calibration negatives by the lexicographic key `(positive_probability, uniform)`, largest first. Use the **100th largest key (one-based)** as `(t, u_star)`. Predict positive when `p > t` or when `p == t and u >= u_star`. Unique calibration keys give exactly 100/2000=.05 empirical calibration FPR. Record duplicate-key failure explicitly rather than changing the seed or redrawing. Apply the same stored cutoff to test scores and their independent test uniforms.

This rule randomizes only a boundary-score tie; it preserves ordering between different probabilities. Save cutoffs, vectors and their fingerprints. **Actual test FPR can still differ from .05.** Always show it, and retain the .01 mean absolute paired test-FPR comparability gate below. No threshold choice, curve interpolation or test-set operating-point selection may use test labels. AP always uses the original model probabilities.

## Measurements and evidence to save

Primary measurement is rare-group test recall. Also report common/overall positive recall, actual test FPR, calibration FPR, F1, noninterpolated AP, worst-group recall, confusion counts, group denominators, chosen parameters, thresholds and coverage. Show mean and sample SD across five seeds for each of the **18 settings**, with all seed-level rows retained. Compare identical seeds and show every cell, including negative outcomes.

After training/selection finish, use hidden truth to log:

- Retained/removed counts and fractions for valid common originals, valid rare originals and mislabeled majority originals. With m=0, noise-retention fraction is null, not zero.
- AW noise flags, clusters, quotas, seed/partner original IDs, and counts of synthetic rows with clean/clean, clean/corrupt or corrupt/corrupt parents. Record equivalent parent identities for SMOTE and duplication identities for ROS.
- These parent categories indicate exposure to corrupted labels; they do not certify a generated location's true class or subgroup. Gaussian component membership is latent for an interpolated point.

Save **selection labels and all successful candidate probability vectors, calibration labels/probabilities, test labels/groups/probabilities, and the tie uniforms**. The first pilot did not save calibration/selection probabilities, limiting independent audit; this study must support recomputation of AP, tuning order, threshold cutoffs and all test metrics without classifier refits. Keep source/configuration/environment fingerprints, partitions, scaler parameters, timings, actual fit attempts, structured failure reasons and checksummed per-case artifacts.

## Predeclared descriptive screen

The practical cutoffs below are research design choices, not theoretical guarantees or significance levels. They are frozen before contaminated results. Inclusive numeric boundaries use tolerance 1e-12; strict seed-level positive/negative comparisons use >1e-12 and <−1e-12.

Write K for AW retention and D for default AW removal. For each base setting b and contaminated level m in {5,10}, a cell qualifies only if **both** conditions hold on the same five seeds:

**A. A rare-preservation problem remains:** D deletes at least one genuine rare original across the five seeds; mean `rare_recall(K,m) − rare_recall(D,m)` is at least .05; that difference is positive in at least four seeds; mean common-recall difference K−D is at least −.02; and mean **absolute paired** test-FPR difference is at most .01.

**B. Retention has a contamination-specific cost:** D deletes at least one mislabeled original across the five seeds, and the paired AP interaction

`I(seed,m) = [AP(K,m) − AP(K,0)] − [AP(D,m) − AP(D,0)]`

has mean at most −.02 and is negative in at least four seeds. This requires retention to lose at least .02 more AP under contamination than removal, rather than merely having lower AP in a clean table. It does not prove that one type of deleted observation alone caused the difference.

At least **two distinct base settings** must have a qualifying contaminated level. Two noise levels in the same base setting count once. Report A and B separately for all 12 contaminated cells even when the combined screen fails.

For each qualifying base setting, test whether an ordinary control among **NORES, WEIGHTED, ROS, SMOTE, AW_REMOVE_TUNED** covers it. A single control must meet all of these at **each** m=0,5,10:

- Mean rare recall ≥ retention's mean − .02.
- Mean common recall ≥ retention's mean − .02.
- Mean AP ≥ max(retention's mean AP, removal's mean AP) − .01.
- Mean absolute paired test-FPR difference from retention ≤ .01.

Controls may differ between base settings; that coverage is descriptive, not a deployable rule choosing methods from hidden groups or test scores. List every control's outcomes rather than reporting only a favorable one.

Use the first applicable status:

1. `incomplete_coverage`: any of the 630 planned method evaluations is unsuccessful or a clean/noisy seed pair is unavailable. Keep its failure/unsupported status; do not fill metrics with zeros. A failed unselected candidate need not invalidate tuned AW if another candidate succeeds; report all candidate statuses.
2. `tradeoff_trigger_not_met`: fewer than two distinct base settings qualify.
3. `simple_controls_suffice`: the trigger is met, but every qualifying base setting is covered by at least one ordinary control across its three noise levels.
4. `residual_tradeoff_requires_replication`: the trigger is met and some qualifying base setting is uncovered. This supports further diagnosis and independent replication, **not** a novelty, efficacy or noise-robustness claim.

No status authorizes claiming a new method. Keep clean and contaminated outcomes separate, do not pool unlike metrics, and do not describe shared synthetic settings as independent evidence. Post-outcome changes to seeds, grids, cutoffs or comparators require another disclosed protocol version.

## Budget, implementation prerequisites and stopping point

Six base settings × three noise levels × five seeds give **90 cases**. Seven method rows give **630 evaluations**. Six fixed fits plus eight additional AW grid fits give at most **14 fits/case, 1,260 total**, with 810 candidate records. Counts assume one uninterrupted traversal; interrupted unfinished cases can add attempts. No new datasets, classifiers or dependencies are planned. Runtime has not been measured. Noise-zero cases are rerun under the new identity and threshold rule; old checkpoints must not be treated as this study's completed cases.

The new JSON stores a SHA-256 of its `scientific_design` object using UTF-8 canonical JSON (`sort_keys=True`, separators `(',', ':')`, `ensure_ascii=False`, `allow_nan=False`). Its digest is `adfdb2961963ff83b9552e84b06963689f938d853859d9bafc224106a08849db`. The separate contamination loader enforces this design against its frozen constant as well as the file's recorded hash. Progress metadata is separate. The original pilot loader deliberately rejects this new format/design; changing its execution flag or passing this file to the old runner will not implement this experiment.

Software requirements include a separate loader/runner and meaningful checks for label-only pairing, nested flips, hidden-truth isolation, original-only scaling, score ties, exactly 100 calibration false positives, candidate selection reuse, provenance, complete coverage, screen boundaries, and checkpoint integrity. These are covered by the [runner's software checks](CONTAMINATION_RUNNER.md). SMOTE/ROS provenance instrumentation must reproduce the reference library output rows and RNG behavior. Hand-derived fixtures verify the AP interaction and its sign. Metamorphic checks verify that changing hidden truth cannot change fitting or selection, while held-out changes cannot change fitted models or selected AW parameters; invalid positive calibration labels are rejected before fitting. Later, the independent audit must recompute selection/calibration metrics from saved vectors.

The protocol-only stage ended at `bff7ce5`; the software stage ended at `c097d4a`; the first experiment ran from clean enabled commit `975020a` and is complete. Its audit and interpretation are reported separately. New samplers, KeepNoise/SoftNoise adaptations, extra classifiers/seeds/noise modes, real-data collection and the old 30-dataset benchmark remain deferred. A further study needs a new disclosed protocol rather than post-outcome changes to this one.
