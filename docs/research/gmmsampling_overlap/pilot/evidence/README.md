# Saved evidence for the completed bounded pilot

- per_seed_results.csv: 180 selected method results; macro AP/F1, balanced accuracy,
  class precision/recall, all subgroup recalls, raw/normalized confusion matrices,
  minority confusion and separate majority errors, counts and runtime.
- paired_per_seed.csv and paired_summary.csv: C minus A/B/D/NORES/WEIGHTED, all
  five seeds per condition; means, SD and descriptive t(4) intervals.
- condition_summary.csv: five-seed means and SD, without pooling related conditions.
- selected_parameters.csv: validation-selected sampler/classifier settings.
- candidate_validation_scores.csv: all 1080 candidate results/selection flags.
- component_counts.csv: all sampled settings, each class/component, with selected
  flag; requested, generated, accepted, rejected, numerical rejection, duplication,
  realized and intentionally unfilled counts. Repeated sampling_seconds are sampler
  timing, not extra independent measurements; do not sum across components.
- original_component_assignments.csv: fitting row index/ID to fitted hard-MAP component.
  These are estimated components, not the true subgroup identities.
- generation_diagnostics_evaluator_only.csv: analytical density diagnostics for added
  Gaussian vs duplicate rows; not certified synthetic label correctness.
- gmm_development_scores.csv and case_runtime.csv: development selection and fitting
  accounting (254 EM fits, 2540 initializations), runtime per case.
- source_versions.json and resolved_configuration.json: frozen source/configuration.
- decision.json, failures.json, run_completion.json, evidence_audit.json: immutable
  run outcome; advancement failed, execution/audit completed, zero failures.
- RAW_ARTIFACT_MANIFEST.json: relative paths, bytes, SHA256 of all raw saved files.
- DOWNLOAD_BUNDLES.json: six lossless full-evidence bundle names/bytes/SHA256.

Full candidate validation and selected test probabilities, labels, sample IDs and
evaluator-only groups are in the raw NPZ files, NOT reconstituted from rounded CSVs.
The raw files and six bundles are saved in the task workspace outside Git's tracked
results, per repository policy. Bundles are under /workspace/gmmsampling-overlap-evidence;
extract all six into one fresh folder to run the read-only full-evidence audit.
Do not treat sparse partial bundle contents as a complete 30-case audit.
