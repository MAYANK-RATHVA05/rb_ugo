# Original-support-aware HS: bounded pilot results

**The predeclared advancement trigger was not met.** The modified regularizer
had a small real-dataset AP gain, but did not consistently beat tuned standard
HS. The synthetic concentration experiment showed mixed AP effects and a
recall/F1 tradeoff. Do not launch a broader benchmark from this pilot.

## What was implemented and verified

After growing an unweighted single CART on original plus SMOTE-generated
fitting rows, route only actual original fitting rows to count node support.
In Agarwal et al. (2022), Eq. (1), substitute that support for the **parent**
count in each prediction increment. Root probability, raw node probabilities
and the whole tree are identical between standard and modified HS.

For positive lambda, zero-support parents suppress their outgoing increments;
lambda=0 recovers the raw tree everywhere. Synthetic ancestry never counts
as an original node visit. No random forests, bootstrap conventions, new
sampler, probability re-estimation from original labels, or pruning was added.

The standard-HS implementation agreed with the pinned author routine across
60 fixture/lambda comparisons (maximum node-probability error `4.44e-16`).
The proportional-inflation identity is tested: if augmented counts equal c
times original counts throughout a fixed tree, standard HS at c*lambda equals
modified HS at lambda. Relative-to-root lambda grids make that global-scale
confound equivalent in both candidate grids.

The supplied AW paper motivates studying classifier behavior after
oversampling; it does not establish synthetic-count inflation as the cause of
its reported tree weakness. No exact original-support substitution was found
in the supplied HS paper or pinned author code. Broader web/index searches
were blocked with HTTP 403, so **novelty remains unverified**. See the
[source audit](SOURCE_AUDIT.md) and [frozen protocol](PROTOCOL.md).

## Run and selection

- Frozen source commit: `dcd686d5fef29908d085d7e3d13cce427659110b`.
- Design SHA-256: `7330b7ab05d3a7dbe1699fb94c980c6e66ad09dea157b716390f77497cc8ff1f`.
- 30/30 cases, 150/150 selected-model test evaluations, 1800/1800 validation
  candidates, 1110 unique tree fits; **no failures or replacements**.
- Five paired seeds (42–46) on three already available numerical variants:
  Ecoli4, Glass6 and Yeast4. Duplicate coordinates stay in one partition.
- Approximate 60% fitting / 20% validation / 20% test real-data splits.
  Scaling and SMOTE use fitting observations only. Selected models are not
  refitted after validation. Original observations alone reach validation/test.
- Twelve validation candidates for each arm; select validation AP, then
  validation F1, then grid order. Fixed classification threshold .5.
- B/C share exactly one fixed full CART per case and tune only lambda. A/D/E
  tune 12 tree-structure settings. Thus B/C predictions differ only through
  the count rule and selected lambda, with equal search budgets.
- Controlled synthetic data use independently generated fitting/validation/
  test populations with known rare groups. Concentration 0/.5/1 changes only
  fitting-only SMOTE seed probabilities; counts and original populations stay
  fixed. Concentration levels share populations and random generation uniforms.
- Measured loop wall time: **10.94 seconds** in this cloud environment,
  including case reports/prediction writes; audit/testing time is additional.
  Per-arm mean tuning times are in the summary CSV, with shared B/C base-fit
  cost counted in each arm. They must not be summed as unique total runtime.

## Primary comparison: modified C versus tuned standard B

AP is average precision; F1 is minority F1 at .5. Entries average five seeds.
CI columns are descriptive t(4) intervals for paired C-minus-B AP effects.

| Setting | B AP | C AP | AP difference | Descriptive 95% interval | B F1 | C F1 |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Ecoli4 | .5902 | .5902 | .0000 | [.0000, .0000] | .6333 | .6333 |
| Glass6 | .6620 | .6620 | .0000 | [.0000, .0000] | .7273 | .7095 |
| Yeast4 | .3443 | .3543 | +.0100 | [-.0089, +.0289] | .3470 | .3400 |
| Synthetic c=0 | .4580 | .4605 | +.0024 | [-.0001, +.0049] | .4716 | .4292 |
| Synthetic c=.5 | .3682 | .3855 | +.0172 | [-.0123, +.0468] | .4775 | .4430 |
| Synthetic c=1 | .3397 | .3210 | -.0187 | [-.0435, +.0060] | .4354 | .3965 |

Ecoli4 and Glass6 have exactly tied AP in all five pairs. This means equal
ranking scores, not necessarily equal probabilities or classifications.
Yeast4 wins AP in four of five seeds, with paired SD .01526. Its exact mean
gain is **.0099992955**, slightly below the frozen .01 requirement; it is
not rounded into a qualifying dataset. Even a rounded pass there would be
insufficient, since two qualifying real datasets were required.

Real macro AP difference is **+.0033331**; F1 difference **-.0082549**.
Synthetic macro AP difference is **+.0003165**; F1 difference **-.03860**.
At c=.5 AP improves in 4/5 seeds, but at c=1 AP worsens in 4/5. There is
no consistent evidence that original support offsets increasing synthetic
concentration. An AP increase alongside lower F1 can reflect ranking versus
the fixed probability threshold; no threshold was tuned using test outcomes.

These are repeated overlapping holdouts and related synthetic conditions.
Intervals quantify seed variation and are not independent-dataset inference
or statistical significance. Every paired seed value, SD, interval and
win/tie/loss count appears in [paired.csv](results_v1/paired.csv).

## All required controls

These macro means weight the three real datasets equally, or the three
synthetic concentration settings equally. Synthetic settings are not three
independent datasets. Full per-setting means/SDs are in the summary CSV.

| Arm | Real AP | Real F1 | Synthetic AP | Synthetic F1 |
| --- | ---: | ---: | ---: | ---: |
| A: SMOTE + tuned tree | .4928 | .5589 | .3746 | .4527 |
| B: SMOTE + tuned standard HS | .5322 | .5692 | .3887 | .4615 |
| C: SMOTE + tuned original-support HS | .5355 | .5610 | .3890 | .4229 |
| D: Tuned balanced class-weighted tree | .5169 | .5595 | .4120 | .4596 |
| E: Tuned no-resampling tree | .5621 | .6614 | .4026 | .4329 |

Beating A alone would overstate the contribution: standard HS already
improves mean real AP over A. E is stronger than C on real macro AP/F1;
D is stronger on synthetic macro AP/F1. This does not imply a universal
method ranking outside these small selected datasets and splits.

## Known synthetic rare-subgroup recall

| Concentration | Standard B | Modified C | Difference |
| --- | ---: | ---: | ---: |
| 0 | .460 | .694 | +.234 |
| .5 | .560 | .568 | +.008 |
| 1 | .506 | .506 | .000 |

The average rare-recall increase is **+.08067**, largely from c=0, with
substantial seed variation. At c=0 its paired interval is [-.0507, +.5187].
It accompanies worse synthetic F1 and is not proof of an overall improvement.
No real-data rare-recall value is reported because genuine subgroup identities
are unavailable. Hidden synthetic identities never enter training/selection.

## Support diagnostics and audit

Final software verification: **15 focused checks passed; 931 tests passed in
the complete repository suite**. Tests cover arithmetic, failure handling,
probability validity, original routing, paired tree identity, proportional
inflation, fixed tuning budgets, training-only scaling and test isolation.

Selected B/C raw trees were identical in every case. They had 0–15 nodes
with zero original support on real datasets and 13–59 on synthetic settings.
All original/augmented counts were reconstructed by routing actual feature
rows, not counting synthetic parents. The zero-support convention was both
unit-tested and exercised in measured trees.

The read-only audit replays all source input/split hashes, fitting-only
scalers, SMOTE generation, raw augmented node class probabilities, original/
augmented node counts, and final test leaf routes. It independently sums HS
increments leaf-to-root and compares saved probabilities. It recomputes all
150 test metric rows and checks all 150 validation argmax selections from
recorded candidate scores. CSV summaries, paired intervals and the frozen
decision are independently regenerated from case records. **No classifier
was refitted** for this audit. See [audit.json](results_v1/audit.json).

Limitation: validation probability vectors are not saved, so their AP/F1
values are not independently recomputed from vectors; their recorded scores,
selection ordering and selection isolation are checked. Original tree training
is verified by source and tests rather than refitting in the audit. Package
versions and run identity are in [manifest.json](results_v1/manifest.json).

## Frozen advancement decision

Passed: full coverage; real F1 noninferiority cutoff -.01; no real AP loss
greater than .02; synthetic rare-recall change above -.02.

Failed: two real datasets with AP gain >=.01; real macro AP gain >=.01;
synthetic macro AP gain >=.005. Overall status:
**`advancement_trigger_not_met`**.

The bounded pilot is complete. It found some local AP/recall benefits but
insufficient evidence to advance this exact count substitution. No large
benchmark, post-outcome tuning, or claim of verified novelty is authorized
or performed here.

## Files and reproduction

- [Summary with sample SDs and runtime](results_v1/summary.csv)
- [150 seed-level scores and selected parameters](results_v1/scores.csv)
- [1800 validation candidate scores](results_v1/selection.csv)
- [Paired effects and uncertainty](results_v1/paired.csv)
- [Decision](results_v1/decision.json), [audit](results_v1/audit.json),
  [source/artifact fingerprints](results_v1/provenance.json)

Raw case JSON and NPZ predictions stay under ignored
`results/original-support-hs-v1`; no raw datasets are committed. The CSVs
and report are available after pulling the research branch. No graphs were
generated.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_original_support_hs.py tests\test_original_support_hs_study.py tests\test_original_support_hs_audit.py
.\.venv\Scripts\python.exe -m scripts.audit_original_support_hs_pilot --input results\original-support-hs-v1
```

The audit command requires the original raw case/prediction directory and
matching source fingerprints; pulling the compact results alone is sufficient
for viewing, but not for a raw audit. A new experiment needs a fresh output
directory. Do not overwrite the measured evidence.
