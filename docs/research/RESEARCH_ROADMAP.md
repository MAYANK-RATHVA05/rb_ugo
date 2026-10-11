# Research roadmap

Updated 2026-10-11 for provisional **DRA-UGO**. Repository/import names stay
`rb_ugo` / `src.*`. Roadmap entries are plans, not approval to implement.
The current decision is [no defensible novelty established](NOVELTY_DECISION.md).

## Stages and approval gates

| Stage | Scope | Status / exit evidence |
| --- | --- | --- |
| 1 | Project setup, paths/configuration checks, portable commands and pins | Software implemented; tests establish behavior, not research performance |
| 2 | Binary CSV/KEEL loading and descriptive imbalance analysis | Software implemented; no splitting, preprocessing fitting or experiments |
| 3 | Original UGO implementation, internal scoring and sanity checks | Provisional prompt-based UGO-ROS/UGO-SMOTE exists; full-paper fidelity and published-result reproduction pending |
| 4A | Literature/source verification, mathematical gap audit and documentation update | Current task; evidence-limited audit. Core-paper and KEEL verification remain blocked |
| 4B | Final mathematical method selection and research-design review | Needs user approval, source review, precise competitor differences, assumptions, label/compute budget and preregistration; may conclude no new method is justified |
| 5 | Leakage-free benchmark infrastructure and faithful UGO evaluation | Future approval; verified provenance and paper-consistent protocol. Development/pilot data must be distinct from final outer evaluation |
| 6 | Implement the approved new methodological mechanism | Conditional on Stage 4B approval; preserve unchanged UGO. Revise/stop if no mechanism survives |
| 7 | Investigate unlabelled feature integration | Future approval; matched labelled-only/U/SSL controls and explicit structural assumptions |
| 8 | Evaluate boundary behavior, intra-class imbalance, U benefit and ablations | Future approval; frozen procedure, controlled simulations and verified benchmarks, separate question-specific outcomes |
| 9 | Statistical analysis, robustness, complexity and research writing | Future approval; correct experimental units, multiplicity, negative findings and reproducible artifacts |

All stages after 4A are provisional. Stage 5 must not use final outer-test outcomes
to guide Stage 6 design: use separate development data or fresh, prespecified
outer evaluation for the final study. Software completion is not proof of paper
fidelity, superiority, theoretical safety or novelty.

## Exact three-question coverage

- **A:** How can we generate minority samples close to the decision boundary
  without increasing harmful class overlap?
- **B:** How can we handle intra-class imbalance, where different regions or
  subgroups of the minority class contain unequal numbers of observations?
- **C:** Can unlabelled data be used to support synthetic minority sample generation?

| Question | Stage 4A/4B definition | Implementation if approved | Stage 8 independent evidence |
| --- | --- | --- | --- |
| A | Threshold versus ranking effects, harm margins and known Bayes-overlap diagnostics | Stage 6 conditional intervention mechanism; ordinary distance filtering alone is not novelty | Simulation A; boundary recall, balanced risk, FPR, ranking, synthetic harm and tuned controls |
| B | Regional counts versus learner-dependent benefit; inconclusive small-disjunct outcomes | Stage 6 regional allocation only if justified | Simulation B; sparse/dense recall, coverage, per-region counts and allocation ablations |
| C | What X identifies, anchor/smoothness assumptions and failure cases | Stage 7 separate fitting U, no truth status for pseudo-labels | Simulation D; 10/25/50/100% labels, matched label/U access, no-U/ordinary SSL/shift controls |

Simulation C varies label noise and minority size to stress A and B. C may fail
even if A or B succeeds. Independent coverage means distinct prespecified
interventions, metrics and falsification criteria, not necessarily different
datasets for every question.

## Stage 4B entry requirements

1. Obtain legitimate source evidence for the six core publications and named
   competitors, especially remedy decomposition and augmentation theory/VTSS.
2. Reconcile UGO's unresolved cap, pools, noise, rounding and protocol before
   faithful-baseline claims. Any code repair requires separate approval.
3. Select at most one defensible mechanism or conclude no new method is warranted.
   Define every input/output, region, uncertainty quantity and generator.
4. State model-dependent effects, assumptions and counterexamples; do not equate
   non-benefit with irreducibility.
5. Verify KEEL variants/folds and declare F/H/A/O/U allocations, label budgets
   and compute budgets.
6. Predeclare metrics, tuning, practical margins, confirmation, adaptation
   handling and baselines/ablations in [DATASET_PLAN.md](DATASET_PLAN.md).
7. Receive user approval before subsequent implementation or experiments.

## Evidence discipline

[REFERENCE_STATUS.md](REFERENCE_STATUS.md) records access; author equations and
assumptions are in [PRIOR_ART_MATRIX.md](PRIOR_ART_MATRIX.md). Audit derivations
are reasoning, not claims of newly published theorems. Experiments stay “not run”
until executed with provenance. Roadmap revisions must report consequences for
all three official questions.

## Software validation of this documentation update

Executed on 2026-10-11 with the existing Python 3.11 environment:

- `.venv/bin/python -m pytest -q`: **147 passed in 4.03s**.
- `.venv/bin/python main_stage1.py`: initialization succeeded.
- `.venv/bin/python main_stage2.py --help`: usage displayed successfully.
- `git diff --check`: passed; local documentation links, heading anchors,
  display-math/fence pairs and the eight canonical filenames were checked.

Source, tests, entry points, dependency declarations and ignore rules are unchanged.
These are software/documentation checks, not benchmarks, research experiments,
UGO-paper fidelity verification or Windows runtime validation.
