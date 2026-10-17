# Frozen Question B diagnostic protocol

Frozen specification: [JSON protocol](../../data/manifests/question_b_diagnostic_protocol.json).
This stage implements the diagnostic authorized after the
[research proposal](QUESTION_B_RESEARCH_PROPOSAL.md). It does not propose or
register a new research sampler. No test outcomes were inspected when fixing
these settings. Runtime manifests additionally freeze implementation/package
hashes, generated fitting-data hashes and every seed/configuration.

## Evidence ledger and attribution boundaries

| Question | Verified evidence | Still requires this pilot |
| --- | --- | --- |
| Does filtering delete classifier observations? | No. `src/ugo.py` fits on all originals plus additions; `pre_noise_mask_` restricts scoring/generation. Runtime checks require every original row and label to remain as the exact output prefix. | Its indirect effect through subsequent augmentation. |
| Can a clean rare mode become seed-ineligible? | Yes: the executed 42/15/3 construction in the proposal loses its three rare seeds at k=5/7 and retains them at k=1/3. | Whether this happens in the frozen population draws and affects active generation. |
| Does a class mean guarantee rare-mode deprivation? | No. UGO ranks individual eligible MMA scores. Class-mean dilution is algebra, not a demonstrated trajectory. UGO paper pp.6–8, Sections 3.2–3.4, Tables 2–4. | Class chosen, positive generation rounds, counterfactual rare-seed ranks, actual anchors and additions. |
| Do few rare-mode additions imply prediction loss? | No. Originals remain in fitting; the learner can generalize without rare-mode augmentation. | Independent recall, achieved FPR and AP under a fixed target distribution. |
| Could capacity or thresholds explain a deficit? | Hamilton Section 5.5/Table 5 reports learner-dependent threshold parity and an MLP exception; this is reported evidence, not a universal theorem. | Equal validation-FPR thresholds for every arm, default-threshold sensitivity, and linear/nonlinear capacity controls. |
| Can existing subgroup-aware rules suffice? | KMeans-SMOTE author/library sparsity rules and SLEO summed-entropy allocation already exist. A-SUWO library code uses cluster-specific LDA errors and a zero-same-label-neighbor isolation rule. Sources and qualifications are in the proposal. | KMeansSMOTE and the repository's EC-EBO are the runnable existing subgroup-aware comparators here. Their generator differences prevent attribution to filtering alone. |
| Are mismatch and sample quantity controlled? | Ma–Zhang pp.9–12/Eqs.2–4 and VTSS p.22 motivate these controls under their stated assumptions. | Common 480-addition caps in the controlled UGO arms; realized-class-count matched ROS and a matching frequency-weight control; outside-minority-support diagnostics. |

Primary sources for ADPCHFO, SD-CSMOTE and PPEO remain unidentified/inaccessible
under the documented search limits. A-SUWO, SLEO and AROSS are not silently
treated as executed comparators. Their source-level evidence and unavailable
primary-paper details retain the qualifications in the proposal. This pilot
cannot establish comprehensive methodological novelty or superiority.

## Population, objective and controls

Use the proposal's compact disks and surrounding majority annuli, fixed class
prior .10 and conditional minority weights 70/25/5. All groups have the same
clean support-separation geometry. F/V/T sizes are 600/2,000/10,000 with exact
class/subgroup strata, seeds 42–51. Feature draws are independent between splits.
No ground-truth group identifier reaches fitting or selection code.

Two frozen regimes: clean, and .20 majority-conditional contamination by the
minority feature law. In the latter, for any decision rule, population
`FPR >= .20 * population-weighted minority recall`; recall at FPR=.05 is
therefore at most .25. This follows from the declared mixture and is a
negative control, not evidence that filtering causes irreducible error.

The common threshold uses validation negatives only, strict score > threshold,
the smallest negative score meeting empirical FPR<=.05. Ties can make it
conservative. Achieved test FPR is measured, never assumed to be .05.
Neighborhood/capacity choices maximize global validation balanced accuracy at
that threshold, with fixed grid-order tie breaking. Test groups cannot tune it.
Report AP, F1, balanced accuracy, per-group/macro/worst recall, denominators,
and default-.5 threshold results as a separate operating-point sensitivity.

Learners: LogReg (C=1, max_iter=1000) and RF100 (100 trees, unrestricted depth,
min_leaf=1, n_jobs=1). The reduced tree count is explicit, shared across arms,
and differs from the historical 500-tree UGO registry setting. The linear
learner is misspecified for concentric supports; it cannot by itself establish
a sampler failure. NORES additionally gets a four-value validation capacity
grid to control the four-choice neighborhood-selection opportunity.

| Arm | Meaning |
| --- | --- |
| `ugo_natural` | Unmodified source-audited UGO-SMOTE, phi=2.5, k_pre=5; original natural-budget reference. |
| `ugo_k5`, `ugo_k3`, `ugo_k1`, `ugo_k7` | Unmodified UGO-SMOTE with those existing k_pre settings and phi=1 (cap=480). Other parameters/generator fixed. |
| `ugo_tuned_k` | Select one of the preceding four using validation only; default k=5 wins ties. No additional model fit. |
| `ugo_prefilter_disabled` | **Diagnostic modification:** temporarily return an all-clean pre-mask. Expands MMA scoring, seed/neighbor pools and relocation references; it cannot isolate seed eligibility alone. Post-generation voting/relocation remains enabled. |
| `ugo_generation_filter_disabled` | **Diagnostic modification:** retain the original pre-mask for scoring, Welch/class choice and final relocation. At generation, rank all original selected-class rows using the same scoring model, retain the original seed-pool size, and allow original same-class interpolation partners. Underlying SMOTE, batch sizes and cap are unchanged. Subsequent trajectories can change as a mediated effect. This tests generation-anchor/partner eligibility together, not anchors separately. |
| `nores`, `weighted` | All original fitting rows; class_weight=None or balanced. |
| `nores_tuned_capacity` | Four validation-only C/min-leaf choices; initial C/leaf settings win ties. |
| `ros`, `smote` | Existing standard comparators, request 480 minority additions; SMOTE k=5. |
| `kmeans_smote` | Existing imbalanced-learn 0.12.4, fixed 12 clusters, k=2, threshold=.01. Smaller within-cluster neighbor requirement is disclosed; this is a full-method comparator, not a controlled UGO ablation. Ceil-based count deviations are retained. |
| `ec_ebo` | Existing audited default implementation, quota 480, actual shortfalls recorded. Full-method comparator with its own generator. |
| `ros_match_*` | Fitting-only ROS matching each UGO configuration's actual final counts in **both** classes. |
| `weights_match_generation` | Original rows weighted by the exact multiplicities of `ros_match_ugo_generation_filter_disabled`; same total weight and C as that duplication control. RF bootstrap behavior is not equivalent to frequency duplication. |

The ablations are internal harness modes, not registered methods. Instrumentation
must reproduce original UGO numerically when no ablation is enabled. Scopes
restore the original filter even on exceptions and are run sequentially.

## Mechanism-first execution and decision

Phase 1 creates F only, runs resampling, and saves arrays, data hashes and
traces. For each original UGO arm, count actual positive-size minority rounds,
rare rows excluded from seed access, and rounds where those rows would enter
the **same-size** seed pool under all-row scoring of the already-fitted model.
This latter probe uses fitting features/labels, not oracle group information;
indices are joined to true groups only by the evaluator. It is not a claim
that the counterfactual full trajectory would be identical.

If no such rounds occur, the seed-access mechanism is inactive for that cell.
Do not turn a hypothetical excluded point into a causal performance finding.
Phase 2 requires complete matching phase-1 records, reuses their checked arrays,
fits models and fixes all validation choices before constructing/evaluating T.
Methods never receive V/T for resampling. Failures are retained, not retuned
after outcomes. No adaptive search for a more favorable simulation is allowed.

Report stage-specific eligibility (including N/A for arms with no generator),
original rows retained, ranked/drawn anchors, minority/majority additions,
relocations and support-region counts. Seed ancestry and generated location
are distinct: membership in a known disk is an evaluator support diagnostic,
not a general truth label for a synthetic point. EC-EBO eligibility is the
union of positive-score anchors over its permitted majority references, with
positive quota/step; KMeans/SMOTE eligibility is observed at actual synthesis
pool calls. No-generation pools can consequently be empty.

Primary contrast: clean RF100 generation-eligibility ablation versus `ugo_k5`,
interpreted only when the original gate is active. Use whole paired seed
replicates for exploratory percentile bootstrap 95% intervals (5,000 resamples,
seed 42). Require >.05 worst-recall benefit with <=.01 FPR increase and achieved
test FPR<=.06 before considering residual-failure promotion; AP/population-risk
tradeoffs must be shown. Secondary intervals are descriptive and unadjusted.
The fixed ten-seed artificial pilot is not confirmatory general evidence.

Select one of the user's four conclusions from the measured evidence. Inactive
generation, strong simple controls, uncertainty or overlap can all block a
positive failure claim. A residual novel-method proposal needs fresh independent
confirmation and a fuller prior-art audit; no new sampler is authorized here.

## Execute and retain evidence

From the repository root, run the fitting-only phase first:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/workspace/.cache/matplotlib \
  .venv/bin/python -m scripts.run_subgroup_diagnostic --phase mechanism \
  --output results/question-b-diagnostic-v1
```

Inspect `mechanism.json` for actual round/ranking opportunities, then run the
prespecified pilot without changing source or protocol:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/workspace/.cache/matplotlib \
  .venv/bin/python -m scripts.run_subgroup_diagnostic --phase pilot \
  --output results/question-b-diagnostic-v1
```

`manifest.json` freezes identities; checked NPZ arrays and individual job records
stay ignored in `results/`. `metrics.csv` and `summary.json` contain evaluator
results. Exact reruns reuse matching saved results; source/protocol changes
require a new output directory. Reviewed aggregate evidence and the scientific
conclusion are saved separately under `docs/research/` after execution.
