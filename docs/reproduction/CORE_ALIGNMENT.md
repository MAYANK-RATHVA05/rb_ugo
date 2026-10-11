# Stage 4R-2: binary UGO core alignment

## Scope and provenance

This stage selectively recovers the batch/cap code and tests reviewed in
[0866dc9](https://github.com/MAYANK-RATHVA05/rb_ugo/commit/0866dc936f231feef9df5e1538b8df206741735d),
then extends their conformance coverage. The starting reproduction specification
is preserved in [SPECIFICATION.md](SPECIFICATION.md); its baseline is c2fa7f3.

No other historical research scope, resource-import code or data assets were
merged. The current work implements original UGO-ROS/SMOTE corrections only.
No final evaluation classifier or benchmark performance experiment was run.

## Changes and source support

| Change | Source / reason | Current behavior |
| --- | --- | --- |
| Fixed nominal batch | §3.4 p7; Table 4 step 2 p8 | max(1, ceil(lambda_step*N_original)), computed once; independent of class size and augmented count |
| Named cap expressions | §3.3 p6 versus Table 4 p8 | cap_interpretation=section_3_3 by default, or table4 |
| Integer threshold | Smallest integer meeting selected inequality | Ceil replaces the historical floor; real threshold 1.6 requires two generated rows |
| Budget clipping | Declared finite-budget safeguard | Only the final batch can shrink to the remaining budget; Table 4 can instead overshoot |
| Diagnostics | Expose reproduction choices | Named cap, nominal batch, source_audited_with_ambiguities and stage4r2_fixed_original_total_batch |
| Source wording | Full PDF is now available and reviewed | Warnings remain explicit about ambiguities; no stale “full paper unavailable” runtime claim |

Both generators retain the existing probability/seed/noise logic. Original noisy
rows remain in classifier fitting data; clean original pools supply scores/seeds
and relocation anchors. These and integer/vote/partner conventions remain
disclosed choices where the article is not unique.

The optional constructor parameter is appended, preserving prior positional
argument order. fit_resample(X,y) remains unchanged. Invalid cap policy fails
before fitting and cannot leave prior fitted diagnostics/model state exposed.
The existing provisional_generation_cap stopping reason is retained.

## Conformance evidence

| Requirement | Evidence |
| --- | --- |
| Binary MMA and class-column mapping | Existing exact-value, reverse-order and invalid-distribution tests |
| Initial/final votes and actual relocation | Existing noise, coincident-neighbor, coordinate and integration tests |
| Highest-mean class selection | Existing majority-selection checks plus new minority/majority fixed-batch cases |
| Original-total rather than class-total batch | Six-row fixture, class counts 4/2, lambda_step=.5: batches [3,3] for both class choices and both generators |
| Fixed batches during augmented retraining | The same cases verify classifier fitting sizes [6,9] and twelve final rows |
| Conflicting caps explicitly available | Counts 90/10, phi=2.5: Section 3.3=350 versus Table 4=225 |
| Fractional/integral integer thresholds | Independent expected counts and inequality checks distinguish ceil from floor and avoid adding one to integral limits |
| Last-batch clipping | Both generators and cap policies use a five-row budget with nominal batch three, generating [3,2] |
| Invalid policy behavior | Unsupported policy raises ValueError without fitting or retaining previous fitted state |
| Reproducibility, preservation and safe stopping | Existing seed, input-copy, cloning, no-op, undefined-Welch and termination checks |

## Executed validation

On Python 3.11.17, the complete command:

    .venv/bin/python -m pytest -q

completed with **162 passed in 6.44 seconds**, no skipped or failed cases.
There are 96 Stage 3/core cases: UGO 46, stopping 22, uncertainty 15 and noise
detection 13; 66 other cases cover initialization, loading, statistics and CLI.

The focused UGO/stopping run also completed: 68 passed. The final full run
includes the added diagnostic display in main_stage3.py.

Initialization and the documented CSV/KEEL fixture CLI commands completed
successfully. Both fixtures report six rows, majority/minority counts 5/1 and
imbalance ratio 5. These are artificial software fixtures.

The documented main_stage3.py demonstration completed with seed 42:

| Variant | Original counts | Generated | Final counts | Nominal batch | Stop |
| --- | --- | ---: | --- | ---: | --- |
| ROS | 96 / 24 | 12 | 96 / 36 | 12 | welch_fail_to_reject |
| SMOTE | 96 / 24 | 12 | 96 / 36 | 12 | welch_fail_to_reject |

The default Section 3.3 budget was 360. Both demonstrations emitted the expected
provisional warning and displayed cap choice and implementation revision.
The prototype previously generated three ROS and six SMOTE rows in this example;
the new counts follow the fixed original-total batch and a changed stopping path.

These are software diagnostics, not accuracy, generalization or paper-result
reproduction. No Windows execution was performed.

## Remaining scope

The corrected binary control loop is software-validated under the declared
policies; it is not a uniquely faithful author implementation. Seed-count
denominator, augmented scoring/seed pools, distance/vote ties, last-batch clipping
and author probability settings still require disclosed decisions or clarification.

RWO, MCSMOTE and their UGO variants, LEAO and the full comparator suite remain
unimplemented. The supplement, exact benchmark variants, author folds/seeds,
simulations, fourteen-classifier configurations, evaluation pipeline and reported
results remain outstanding.

The next separately authorized stage is source-backed missing generators and
comparators under the reproduction specification. Do not infer their equations
from names or replace unavailable methods under the same names.
