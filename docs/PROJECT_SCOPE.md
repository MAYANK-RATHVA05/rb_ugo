# Current project scope

The project is **Imbalanced Learning Reproductions**: audited implementation and
controlled evaluation of published methods. Reproducing existing papers is the
current objective; developing an extension requires a separate request. The
earlier risk-budgeted UGO/CSG direction is inactive and is not an implemented or
claimed contribution. Its historical branches/source records remain in Git.

## Research questions and the published methods

| Question | Current work | What the evidence can establish |
| --- | --- | --- |
| Can minority generation near difficult boundaries avoid unnecessary overlap? | Classifier-specific UGO and separate BFCH-SMOTE | Audited rules and descriptive proxy/performance outcomes; no guarantee of safe generated labels |
| How can differently represented minority subregions be handled? | EC-EBO clustering, entropy allocation and generation | Cluster sizes, quota fulfilment/shortfalls and held-out comparisons under declared interpretations |
| Can unlabelled observations support minority training augmentation? | SSHR/SSHC using observed L plus training-only U | Pseudo-label correctness, SMOTE counts, matched empty-U ablation and held-out outcomes |

SSHR selects **existing** unlabelled observations with minority pseudo-labels.
Only its fallback SMOTE creates new feature vectors. Unlabelled density or
confidence alone does not verify their minority membership.

## Completed records and remaining gaps

The complete working branch is
[`sshr-implementation`](https://github.com/MAYANK-RATHVA05/rb_ugo/tree/sshr-implementation).
It includes all four primary methods. `main` includes UGO/BFCH-SMOTE/EC-EBO;
publishing this documentation on main does not merge the SSHR implementation.

- UGO has four generator variants, declared cap/pool conventions, a measured
  simulation pilot and an interrupted partial real-data benchmark. The historical
  record does not mean its ignored raw/cached files exist in this checkout or
  that the full 74-dataset paper has been reproduced.
- BFCH-SMOTE and EC-EBO have small paired Winequality/Vowel studies with explicit
  feature/grouping choices and mixed outcomes.
- SSHR Stages 0–4 are complete: full-paper audit, separate SSHC, resampling core,
  estimator/controlled runner and Ecoli4/Yeast4 pilot. Its 800 jobs succeeded and
  were independently audited; an exact repeat reused all 800. The last validated
  full suite on that branch passed 738 tests.

See [UGO alignment](ugo_reproduction.md), [BFCH decisions](BFCH_METHOD.md),
[EC-EBO decisions](EC_EBO_METHOD_DECISIONS.md) and
[SSHR decisions](https://github.com/MAYANK-RATHVA05/rb_ugo/blob/sshr-implementation/docs/SSHR_METHOD_DECISIONS.md)
for actual source rules and remaining uncertainty. The
[SSHR measured report](https://github.com/MAYANK-RATHVA05/rb_ugo/blob/sshr-implementation/docs/SSHR_SMALL_STUDY.md)
records reduced mean F1 with U in all four conditions, low retrospective
pseudo-label correctness, metric tradeoffs and quota shortfalls. It does not
establish general superiority or the authors' exact result tables.

Author-code equivalence, several source conventions, author folds/preprocessing,
full benchmark coverage and some printed metric definitions remain unresolved.
Do not replace those gaps with a completion percentage, a novelty claim or new
experiment results. Further training/benchmarks require an explicit new task;
this documentation/identity refresh changes no algorithms or measured evidence.

## Project identity

The display name describes the published-method work. The verified GitHub slug
is `MAYANK-RATHVA05/rb_ugo`; the prepared replacement is
`MAYANK-RATHVA05/imbalanced-learning-reproductions`. Renaming is a hosting setting,
separate from checkout-folder names and source module identifiers. See
[the rename note](REPOSITORY_RENAME.md). Historical audit metadata, immutable
study JSON/CSV files and old commit links retain their original identity/context.
