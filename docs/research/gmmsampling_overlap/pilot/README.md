# Competing-minority gate: bounded implementation

Read [FROZEN_PROTOCOL.md](FROZEN_PROTOCOL.md) and the machine-readable
[configuration](../../../../configs/research/gmmsampling_overlap_pilot.json)
before interpreting results. Author code is archived verbatim for hash and helper
agreement checks, not imported as a project module. Its MIT license is preserved.
The previous audit and previous binary implementations/results are unchanged.

From repository root, Python 3.11 with project requirements:

```sh
python -m pytest -q tests/test_gmmsampling_overlap.py
python -m scripts.run_gmmsampling_overlap_pilot --output results/gmmsampling-overlap-pilot
python -m scripts.audit_gmmsampling_overlap_pilot --output results/gmmsampling-overlap-pilot
```

Use a fresh output directory; the runner refuses overwriting previous artifacts.
The default run is 30 synthetic cases and 1080 small classifier candidates. It is
not a real-data or expanded benchmark. Sampler code accepts only original X/y,
minority labels and settings. True subgroups enter evaluator outputs only.

Raw evidence is under ignored results/gmmsampling-overlap-pilot:
original_partitions.npz stores raw X, labels, globally partition-prefixed row IDs
and evaluator-only groups. Parent indices in samples_*.npz reference the original
fitting row order, NOT true generator parents of Gaussian draws (parent=-1).
kind distinguishes original/gaussian/duplicate; component=-1 for originals.
Proposals include only consumed prefixes with acceptance/numerical masks; unused
pre-generated vectors do not count as proposals. Counts are by label/component.
All candidate validation vectors and selected method test vectors are saved,
along with fitted coefficient/scaler/GMM states and validation selection records.
Summary CSVs and a read-only reconstruction audit require no classifier refits.

Synthetic labels, densities and genuine subgroups are fully controlled here;
results cannot establish real-data usefulness, causal synthetic label correctness,
novelty or statistical certainty from five paired seeds.
