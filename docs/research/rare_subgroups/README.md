# Rare minority subgroup research

Status: AW baseline software and correctness checks completed, 19 October 2026. No pilot training or new research results are included. The source audit and prospective protocol were prepared first; this stage adds their independent baseline implementation.

The question is whether density-based cleaning and sampling can neglect a genuine but sparsely observed minority subgroup. We first need to measure that behavior and check whether ordinary baselines or parameter selection already address it. A new algorithm is not yet justified.

## Read in this order

1. [AW-SMOTE source audit](AW_SMOTE_SOURCE_AUDIT.md): what the supplied paper states, what its public code actually does, and the differences that must be resolved before evaluation.
2. [Pilot protocol](PILOT_PROTOCOL.md): six controlled synthetic settings, five seeds, common baselines, independent evaluation, and explicit stop criteria.
3. [Implementation and checks](IMPLEMENTATION.md): the independent AW sampler, paired retention control, diagnostics, and failure behavior.
4. [Protocol configuration](../../../configs/rare_subgroups_pilot_v1.json): machine-readable design, currently not connected to an experiment runner.
5. [Source register](SOURCE_REGISTER.json): paper fingerprint and pinned author release.

## Current conclusion

The author release cannot be treated as an exact executable version of the supplied paper. In particular, its final output retains all original observations, and its intra-cluster boundary calculation uses an original label array against reordered feature rows. These are source fidelity issues, not research contributions.

The paper already discusses legitimate sparse observations being removed, includes KeepNoise and SoftNoise experiments, and proposes probability regularization. Simply retaining DBSCAN noise, reducing MinPts, or smoothing sampling weights would not establish novelty.

The audited baseline is now implemented as `src/aw_smote.py`, with meaningful checks in `tests/test_aw_smote.py`. This establishes software behavior under the disclosed conventions, not efficacy or novelty. The next stage can build the diagnostic runner. Only a later authorized stage should run the pilot. Existing UGO/BFCH-SMOTE/EC-EBO/SSHR code and the 30-dataset study remain the historical baseline work; this pilot does not replace their results.

## Windows synchronization

After this stage has been pushed, run in the existing checkout:

```powershell
cd C:\ugo\ugo_base
git switch rare-subgroup-research
git pull --ff-only origin rare-subgroup-research
```

Open `docs\research\rare_subgroups\IMPLEMENTATION.md`. There is no AW-SMOTE training command in this stage.
