# Project scope: provisional DRA-UGO research

Updated 2026-10-11. Working title: **Data-Reducibility-Aware Uncertainty-Guided
Oversampling for Imbalanced Classification (DRA-UGO)**. This is a research
identity, not a completed or validated algorithm.

## Official problem statement

- **A:** How can we generate minority samples close to the decision boundary
  without increasing harmful class overlap?
- **B:** How can we handle intra-class imbalance, where different regions or
  subgroups of the minority class contain unequal numbers of observations?
- **C:** Can unlabelled data be used to support synthetic minority sample generation?

Each requires a specific methodology and independent experimental evidence.
Implementing ordinary uncertainty ranking, subgroup allocation or pseudo-labels
is not sufficient novelty; several ingredients already exist [S3, S5, S6 in the
reference register](PRIOR_ART_MATRIX.md#verified-reference-register-and-audit-limits).

## Core hypothesis

> Can a classifier-specific oversampling method identify local minority regions
> where additional synthetic training information improves genuine discrimination,
> distinguish those regions from threshold-recoverable or irreducibly overlapping
> regions, and use unlabelled feature-space information to improve these decisions?

This is a question, not a demonstrated decomposition. The operational target is
**conditional finite-intervention benefit** beyond matched threshold/weight/count
controls, with observed-harm and inconclusive outputs. It is not an oracle for
intrinsic data limitation or irreducibility. Synthetic points derived from existing
observations provide no new independent labels; utility can arise from weighting,
optimization or inductive bias. See the derivation and counterexamples in
[MATHEMATICAL_GAP_ANALYSIS.md](MATHEMATICAL_GAP_ANALYSIS.md).

Threshold decisions can improve without better ranking [S1]. Uncertainty and
sparsity do not alone diagnose whether augmentation helps. U informs structure,
not class membership without assumptions [S3]. The current MMA is signed
label-aware disagreement, not a formal epistemic/aleatoric decomposition
([source](../../src/uncertainty.py), [baseline record](../ugo_reproduction.md)).

## Objectives and independent evidence

| Question | Objective | Required evidence | Main failure |
| --- | --- | --- | --- |
| A | Beneficial boundary augmentation versus overlap/noise reinforcement | Known-overlap simulations; boundary recall, ranking, balanced risk and FPR against tuned controls | Recall rises while FPR/ranking worsens |
| B | Detect underserved, intervention-responsive minority subgroups | Unequal-subgroup simulations; per-region recall/counts/coverage and confirmed intervention effects | Dense subgroup dominates allocation; rare clean mode rejected |
| C | Measure incremental value of separate fitting U | Matched-label-budget no-U/U/SSL comparisons at 10/25/50/100% labels; shift ablations | Majority bridges or shift create misleading support |

These are study specifications, not outcomes. Simulation truth is diagnostic only;
it must never guide the proposed mechanism's augmentation decisions.

## Naming history and compatibility

**Risk-Budgeted Uncertainty-Guided Oversampling (RB-UGO)** was the earlier working
title, focused on held-out risk/FPR constraints. Those remain potential safeguards,
not a finalized contribution. CSG-UGO and combinations of scores are historical
hypotheses, not approved algorithms. The DRA-UGO name does not turn ordinary
validation selection into novel methodology.

Keep repository `MAYANK-RATHVA05/rb_ugo`, `src.*` imports, working entry points,
tests and the UGO baseline. Fidelity repairs require a separate approved task
and explicitly versioned baseline; no silent comparator changes.

## In scope and out of scope

Stage 4A covers legitimate source verification, explicit source limitations,
mathematical comparison and counterexamples, at most three candidate mechanisms,
README/agent rules, roadmap, dataset/evaluation planning, existing tests, and a
separate review PR. Two mechanisms are discussed, neither established as novel
[CANDIDATE_HYPOTHESES.md](CANDIDATE_HYPOTHESES.md).

No source/test rewrite, new oversampler, benchmark-data download, experiment,
dependency change, repository rename or published-result reproduction occurs
here. No Stage 4B decision is made on the user's behalf. Synthetic observations
are not independent evidence; confident pseudo-labels are not ground truth.

## Open decisions

The six core papers remain inaccessible even at abstract level. Exact titles and
method topics in the brief are user-supplied, not independent verification.
Remedy decomposition and risk/mismatch/VTSS are direct novelty-threat questions,
not verified findings. See [REFERENCE_STATUS.md](REFERENCE_STATUS.md).

KEEL variants, target meanings, counts, schemas, folds and licenses need checking;
[DATASET_PLAN.md](DATASET_PLAN.md) does not invent statistics. Decisions also remain
on region definitions, independent validation budgets, generators, practical
benefit/harm margins, graph smoothness assumptions and matched compute budgets.

Current decision: **no defensible methodological novelty established**.
[NOVELTY_DECISION.md](NOVELTY_DECISION.md) gives the implementation gates.
Stop at Stage 4A for user review before Stage 4B.
