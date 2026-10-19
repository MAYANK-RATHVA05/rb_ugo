# Rare minority subgroup research

Status: the first frozen pilot completed on 19 October 2026: **30 cases, 210 successful evaluations, 420 classifier fits**. Its predeclared screen reports `trigger_not_met`; these results do not justify a new sampler. The separate [contamination follow-up protocol](CONTAMINATION_PROTOCOL.md) and [runner/software checks](CONTAMINATION_RUNNER.md) are implemented. Its first frozen experiment is now authorized; measured results remain pending.

The question is whether density-based cleaning and sampling can neglect a genuine but sparsely observed minority subgroup. We first need to measure that behavior and check whether ordinary baselines or parameter selection already address it. A new algorithm is not yet justified.

## Read in this order

1. [Measured pilot results and figures](PILOT_RESULTS.md): subgroup loss, strong ordinary baselines, the complete screening result and limitations.
2. [AW-SMOTE source audit](AW_SMOTE_SOURCE_AUDIT.md): what the supplied paper states and how the public code differs.
3. [Pilot protocol](PILOT_PROTOCOL.md): six controlled synthetic settings, five seeds, common baselines, independent evaluation, and explicit stop criteria.
4. [Implementation and checks](IMPLEMENTATION.md): the independent AW sampler, paired retention control, diagnostics, and failure behavior.
5. [Runner and local checks](RUNNER.md): partition isolation, selection/calibration, output files and checkpoint behavior.
6. [Protocol configuration](../../../configs/rare_subgroups_pilot_v1.json): frozen scientific design and current execution status.
7. [Source register](SOURCE_REGISTER.json): paper fingerprint and pinned author release.
8. [Contamination follow-up](CONTAMINATION_PROTOCOL.md), [runner and checks](CONTAMINATION_RUNNER.md), and [configuration](../../../configs/rare_subgroups_contamination_v1.json): paired training-label flips, independent tie calibration, seven controls and prospective stop rules. The first frozen experiment is authorized.

## Current conclusion

The author release cannot be treated as an exact executable version of the supplied paper. In particular, its final output retains all original observations, and its intra-cluster boundary calculation uses an original label array against reordered feature rows. These are source fidelity issues, not research contributions.

The paper already discusses legitimate sparse observations being removed, includes KeepNoise and SoftNoise experiments, and proposes probability regularization. Simply retaining DBSCAN noise, reducing MinPts, or smoothing sampling weights would not establish novelty.

The pilot confirms that cleaning can discard genuine rare originals, but only one setting met every predeclared comparison criterion and ordinary baselines already worked well. No new method is claimed. The next protocol specifies contamination on unchanged feature rows and addresses score ties while retaining actual test-FPR comparisons. Its design is informed by the clean pilot, and its results remain unknown. Existing UGO/BFCH-SMOTE/EC-EBO/SSHR code and the 30-dataset study remain the historical baseline work; this pilot does not replace their results.

## Windows synchronization

After this stage has been pushed, run in the existing checkout:

```powershell
cd C:\ugo\ugo_base
git switch rare-subgroup-research
git pull --ff-only origin rare-subgroup-research
```

Open `docs\research\rare_subgroups\PILOT_RESULTS.md` and `docs\research\rare_subgroups\pilot_v1\comparison.csv`. These published results are included in Git; viewing them requires no retraining.

Read `docs\research\rare_subgroups\CONTAMINATION_RUNNER.md` for preflight and test commands. The current stage runs the authorized frozen experiment and audits its saved evidence.
