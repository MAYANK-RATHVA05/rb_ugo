# Rare minority subgroup research

Status: AW baseline and pilot runner software checked, 19 October 2026. The frozen pilot is now authorized for execution; measured coverage and results will be recorded after it finishes.

The question is whether density-based cleaning and sampling can neglect a genuine but sparsely observed minority subgroup. We first need to measure that behavior and check whether ordinary baselines or parameter selection already address it. A new algorithm is not yet justified.

## Read in this order

1. [AW-SMOTE source audit](AW_SMOTE_SOURCE_AUDIT.md): what the supplied paper states, what its public code actually does, and the differences that must be resolved before evaluation.
2. [Pilot protocol](PILOT_PROTOCOL.md): six controlled synthetic settings, five seeds, common baselines, independent evaluation, and explicit stop criteria.
3. [Implementation and checks](IMPLEMENTATION.md): the independent AW sampler, paired retention control, diagnostics, and failure behavior.
4. [Runner and local checks](RUNNER.md): partition isolation, selection/calibration, output files and checkpoint behavior.
5. [Protocol configuration](../../../configs/rare_subgroups_pilot_v1.json): frozen scientific design and current execution status.
6. [Source register](SOURCE_REGISTER.json): paper fingerprint and pinned author release.

## Current conclusion

The author release cannot be treated as an exact executable version of the supplied paper. In particular, its final output retains all original observations, and its intra-cluster boundary calculation uses an original label array against reordered feature rows. These are source fidelity issues, not research contributions.

The paper already discusses legitimate sparse observations being removed, includes KeepNoise and SoftNoise experiments, and proposes probability regularization. Simply retaining DBSCAN noise, reducing MinPts, or smoothing sampling weights would not establish novelty.

The audited baseline and diagnostic runner are implemented with meaningful software checks. This establishes behavior under the disclosed conventions, not efficacy or novelty. The current stage runs the frozen pilot. Existing UGO/BFCH-SMOTE/EC-EBO/SSHR code and the 30-dataset study remain the historical baseline work; this pilot does not replace their results.

## Windows synchronization

After this stage has been pushed, run in the existing checkout:

```powershell
cd C:\ugo\ugo_base
git switch rare-subgroup-research
git pull --ff-only origin rare-subgroup-research
```

Open `docs\research\rare_subgroups\RUNNER.md` for check-only and explicit execution commands.
