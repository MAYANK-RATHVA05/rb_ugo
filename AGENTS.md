# Research and development rules

<<<<<<< HEAD
## Current research identity

- Working title: **Data-Reducibility-Aware Uncertainty-Guided Oversampling for Imbalanced Classification (DRA-UGO)**. This is provisional, not an implemented or validated algorithm.
- Risk-Budgeted UGO / RB-UGO is an earlier proposal, not the finalized contribution. Preserve repository name `rb_ugo` and existing `src.*` import paths.
- Investigate the exact three-question problem statement:
  - **A:** How can we generate minority samples close to the decision boundary without increasing harmful class overlap?
  - **B:** How can we handle intra-class imbalance, where different regions or subgroups of the minority class contain unequal numbers of observations?
  - **C:** Can unlabelled data be used to support synthetic minority sample generation?
- Evaluate A, B and C independently; a global metric alone does not answer all three.
- The current hypothesis concerns learner-dependent regional augmentation benefit beyond threshold, weighting and ordinary sampling controls. No mechanism is finalized.
- Read `docs/research/PROJECT_SCOPE.md`, `RESEARCH_ROADMAP.md`, `REFERENCE_STATUS.md` and `NOVELTY_DECISION.md` before further research work.

## Research integrity and baseline protection

- Freeze the explicitly source-audited UGO revision as the comparison baseline. Never silently mix new research mechanisms into `src/ugo.py` or its supporting modules. Historical pre-repair code is preserved in Git at `2c54d3d`.
- Stage 4A-R explicitly authorizes demonstrated source-fidelity repairs, with before/after records and regression tests. Read `docs/research/UGO_PAPER_CODE_AUDIT.md` and `docs/ugo_reproduction.md`: fixed original-total batches are repaired; conflicting caps/pools/rounding remain disclosed. Keep `allow_provisional=True` opt-in and `source_audited_with_ambiguities` diagnostics.
- Consult supplied U/L/H/G/B/T papers using exact filename/hash/page/section references in `REFERENCE_STATUS.md` and `PRIOR_ART_MATRIX.md`. Do not replace evidence with earlier generated descriptions. Full-text availability is not a comprehensive proof audit or reproduction of author results.
- B already provides remedy decomposition and targeted matched-budget interventions; T provides mismatch theory and size search. No DRA-UGO novelty is established. The inaccessible 13-paper collection remains FULL TEXT UNAVAILABLE.
- Distinguish implemented software, faithful reproduction, hypotheses, completed experiments and literature-verified facts. Stages 1–3 software exists; UGO fidelity and published-result reproduction remain pending.
- Audit mathematical novelty and direct competitors before implementation. Clustering, entropy weighting, pseudo-labels, boundary-distance filtering, count tuning or multiplying existing scores is not sufficient novelty by itself.
- Never invent citations, paper access, equations, datasets, counts, results or guarantees. Label missing evidence UNVERIFIED; prompt-supplied titles are not independent metadata verification.
- Never claim uncertainty estimates formally decompose epistemic and aleatoric uncertainty without a justified model and verified assumptions.
- High uncertainty or sparse density does not establish data limitation. Intervention effects depend on learner, fitting data, generator and evaluation target; no observed benefit does not prove irreducibility.
- Unlabelled density does not establish minority membership. Pseudo-labels are model estimates, not observed truth.

## Leakage and evaluation safeguards

- Never use outer-test features or labels for preprocessing, graph construction, uncertainty estimation, tuning, region selection, generation or unlabelled support estimation.
- All resampling and synthetic anchors remain within fitting partitions. Fit preprocessing on fitting partitions only.
- Declare provenance and independent fitting, threshold-selection, augmentation-audit and outer-evaluation allocations. Use nested evaluation or separate selection data where necessary; respect group/time constraints.
- Adaptive candidate comparisons cannot reuse a fixed audit set as if candidates stayed independent of it. Use fresh confirmation or an explicitly justified selection protocol.
- Always include no resampling with default and validation-tuned thresholds, class weighting with tuned thresholds, and validation-tuned synthetic-size controls. Preserve UGO as comparator; match label, U-access and tuning budgets.
- Report threshold-independent ranking, decisions and majority false-positive harm. State whether AUPRC means average precision or trapezoidal PR area.
- Measure minority-region counts, recall and coverage; assess U with labelled-only and structural/SSL controls. Hidden simulated labels are not algorithm inputs.
- Keep the outer test untouched until the complete procedure and reporting choices are frozen. Do not redesign using test results.
- Set and report seeds (default `RANDOM_STATE = 42`); record versions, split identifiers, transformations and computational budgets.

## Approved stages and engineering safeguards

- Implement only the stage explicitly approved. Stop after it and await approval; roadmap entries do not authorize future stages.
- Current Stage 4A-R permits supplied-resource offline imports, provenance/manifests, fold checks, all-file Stage 2 validation, demonstrated UGO source repairs/tests, official-training-fold-only software smoke checks, and research documentation. It authorizes no new DRA-UGO mechanism, pseudo-labelling, generation heuristic or full benchmark.
- All eight supplied archives have full files but no official folds. Do not substitute full data/generated folds for the requested Ecoli2/Yeast3/Ecoli4 smoke checks. Keep blocked outcomes explicit. Authenticate source split provenance before calling future folds official.
- Imports preserve original names/source ZIPs, verify hashes, reject unsafe entries/conflicting contents and run offline. Real raw assets, PDFs and outputs stay ignored; portable scripts/manifests are tracked. Regenerating a derived report must not overwrite a raw resource.
- Fold mappings come from training and are reapplied to held-out partitions; test-frequency reversal must not change the positive class. Identical-valued rows without record IDs cannot prove distinct record identity.
- Stage 4B is mathematical method selection and research-design review. Stop after Stage 4A-R for review; later roadmap entries are unapproved.
- Stage 1 was initialization only; Stage 2 loading/descriptive inspection without splitting, preprocessing fitting, resampling or training; Stage 3 internal UGO scoring and software sanity checks only. Preserve those limits in historical reports.
- Keep working Stages 1–3 unchanged unless a separately approved task requires a specific fix.
- Write meaningful tests for implemented functionality, including failure behavior. Do not weaken assertions or treat software fixtures as research evidence.
- Keep code modular and readable; target Python 3.11, existing pins and portable `pathlib` paths. Do not rename working modules unnecessarily.
- Do not add unrelated dashboards, frontend, API, databases, cloud infrastructure or new ML architectures.
- Never commit real/private raw data, credentials, `.venv`, caches or generated results. Preserve ignore rules and useful historical documentation.
- Codex and Antigravity are separate environments: cloud edits require Git synchronization, and cloud tests do not verify Windows execution.
- After completing and validating an authorized stage, commit and push to the existing GitHub repository. Use small, meaningful commits with accurate messages; do not fabricate work history or rewrite commits to disguise authorship.
- Do not force-push, overwrite existing branches or merge without authorization. Report actual branch, commits, checks and PR status; stop before the next stage.
=======
- Never leak outer-test data into training or model selection.
- Fit preprocessing on training partitions only.
- Generate synthetic examples from fitting data only.
- Use nested evaluation or separate selection data where necessary.
- Always compare against the original no-resampling baseline.
- Distinguish faithfully reproduced UGO from our proposed changes.
- Set and report random seeds (default RANDOM_STATE = 42).
- Never invent datasets, results, citations, or theoretical guarantees.
- Write tests for implemented functionality, including failure behavior.
- Keep code modular and readable. Target Python 3.11 and portable pathlib paths.
- Do not add unnecessary frontend, API, dashboard, database, or cloud infrastructure.
- Stop after the requested stage. Stage 1 is initialization; Stage 2 is data inspection; Stage 3 is original-UGO implementation and software checks. Further stages require user approval.
- Do not implement SMOTE, UGO, RB-UGO, model training, dataset downloading, or experiments in Stage 1.
- Codex and Antigravity are separate tools; never assume cloud changes have synchronized locally.
- After completing and validating each requested stage, commit its changes and push to GitHub. Report the commit and branch; do not proceed to the next stage without approval.
- Use small, meaningful commits for independently validated changes, with accurate messages; do not fabricate work history or rewrite existing commits to disguise authorship.
- Stage 2 must not fit preprocessing, split or resample data, train models, calculate model performance, or implement UGO/RB-UGO. Synthetic fixtures are software tests only; never commit real or private raw datasets.
- Stage 3 may fit internal UGO classifiers on fitting data only. Do not implement Risk-Budgeted UGO, held-out batch selection, FPR constraints, final-model scoring, or benchmark experiments.
- The Stage 3 implementation is provisional until the full paper is checked. Read docs/ugo_reproduction.md; do not call the assumed cap, pools, rounding, or noise conventions verified paper rules. Keep opt-in and diagnostic status explicit.
>>>>>>> origin/main
