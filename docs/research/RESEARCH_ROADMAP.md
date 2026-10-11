# Research roadmap

Updated 2026-10-11 for provisional **DRA-UGO**. Repository/import names stay
`rb_ugo` / `src.*`. Roadmap entries are plans, not approval to implement.
The current decision is [no defensible novelty established](NOVELTY_DECISION.md).

## Stages and approval gates

| Stage | Scope | Status / exit evidence |
| --- | --- | --- |
| 1 | Project setup, paths/configuration checks, portable commands and pins | Software implemented; tests establish behavior, not research performance |
| 2 | Binary CSV/KEEL loading and descriptive imbalance analysis | Software implemented; no splitting, preprocessing fitting or experiments |
| 3 | Original UGO implementation, internal scoring and sanity checks | UGO-ROS/SMOTE implemented; Stage 4A-R repairs source batch/cap handling, with disclosed remaining ambiguities; result reproduction pending |
| 4A | Initial mathematical/literature audit | Historical evidence-limited documentation, superseded where Stage 4A-R supplies full evidence |
| 4A-R | Scientific verification/recovery, offline resources, real-file loader checks, source-alignment repairs/tests | Six core full papers and eight full datasets verified; no official folds; real-fold smoke blocked; novelty unestablished |
| 4B | Final mathematical method selection and research-design review | Needs user approval, source review, precise competitor differences, assumptions, label/compute budget and preregistration; may conclude no new method is justified |
| 5 | Leakage-free benchmark infrastructure and faithful UGO evaluation | Future approval; verified provenance and paper-consistent protocol. Development/pilot data must be distinct from final outer evaluation |
| 6 | Implement the approved new methodological mechanism | Conditional on Stage 4B approval; preserve unchanged UGO. Revise/stop if no mechanism survives |
| 7 | Investigate unlabelled feature integration | Future approval; matched labelled-only/U/SSL controls and explicit structural assumptions |
| 8 | Evaluate boundary behavior, intra-class imbalance, U benefit and ablations | Future approval; frozen procedure, controlled simulations and verified benchmarks, separate question-specific outcomes |
| 9 | Statistical analysis, robustness, complexity and research writing | Future approval; correct experimental units, multiplicity, negative findings and reproducible artifacts |

Stages 4B onward remain provisional and need explicit approval. Stage 5 must not use final outer-test outcomes
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

1. Complete access to the oversized related-paper collection and focused direct
   competitor review. Six core full-text methods are now inspected, including B/T.
2. Reconcile UGO's unresolved cap, pools, noise, rounding and protocol before
   faithful-baseline claims. Future repairs require explicit scope and regression evidence; Stage 4A-R
   repairs are recorded rather than silently redefining the baseline.
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

## Verification/recovery evidence

Executed in Python 3.11.16, Linux, on 2026-10-11:

- `.venv/bin/python -m pytest -q`: **175 passed in 3.84s**, none skipped; includes
  all eight real-file integrations. Historical pre-recovery suite had 147 tests.
- Focused UGO modules: **84 passed in 1.44s** after the source repairs.
- Stage 1 initialization and Stage 2 Abalone19 inspection succeeded; numeric /
  nominal schema and counts matched the manifest.
- Offline imports were repeated with identical source/content hashes; verification
  loads eight full files and reports eight MISSING_FOLDS.
- `scripts/ugo_smoke.py`: exit 2, six BLOCKED_MISSING_VERIFIED_FIVE_FOLD rows;
  no real-fold UGO fit or predictive result.
- Installed dependencies are checked with `uv pip check` using a writable cache;
  this uv-created venv has no pip module. No dependency pins were changed.

Source repairs and new utility/regression tests are authorized Stage 4A-R changes.
Stage 2 implementation/public APIs and dependency declarations remain unchanged.
These are software/source checks, not benchmarks, complete paper fidelity or
Windows runtime validation. Stop at this recovery review.
