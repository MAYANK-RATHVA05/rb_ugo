# NADOS baseline specification

Status: independently implemented, paper-aligned software baseline. The separate
[90-case diagnostic](NADOS_DIAGNOSTIC_RESULTS.md) is complete and audited; this
baseline specification itself contains no efficacy or novelty claim.

## Source and scope

The supplied source is **NADOS: Reliability-Gated Difficulty-Aware
Oversampling for Noisy Imbalanced Classification**, Ednel Ashraff Misran,
Syahid Anuar and Adam Mohd Khairuddin, *International Journal of Advanced
Computer Science and Applications*, volume 17, number 7, 2026. The supplied
ten-page PDF has SHA-256
`c6ff5e5b3b0f61f972dce9c5934e0f6e780e1026c771fd74235c4e93e854303e`.
The method is specified in Section III and Algorithm 1. Section VIII.C says the
authors retain the implementation and may provide it on reasonable request;
there is no author code available here for output-level verification.

NADOS retains the observed training dataset, computes seed reliability before
generation, and assigns difficulty-based synthesis priority only to eligible
seeds. For minority seed `i`, the implementation follows the paper's equations:

```
support_i = m_i / k
distance_i = d_majority_i / (d_minority_i + d_majority_i)
R_i = lambda * support_i + (1-lambda) * distance_i
B_i = 1 - abs(2*support_i - 1)
D_i = eta*B_i + (1-eta)*(1-support_i)
P_i = R_i*D_i if R_i >= tau else 0
```

Seeds are drawn from normalized `P`; a minority partner is drawn and the new
row is `x_i + alpha*(x_j-x_i)`, with `alpha` uniform on `[0,1)`. The paper's
fixed main configuration is the default: `k=5`, `lambda=.5`, `eta=.5`,
`tau=.35`, target minority/majority ratio `1.0`, and random seed 42.

The paper evaluates 22 binary benchmark datasets, five symmetric training-label
noise levels, four classifier families and eight oversampling baselines. It
reports aggregate F1, G-mean, balanced accuracy and AUPRC. It does not report
known rare-subgroup recall, genuine rare-seed eligibility, corrupted-parent
lineage, or the one-sided majority-to-minority corruption used by this
repository's completed contamination diagnostic. Those measurements motivate a
future baseline evaluation; they do not imply that NADOS will fail it.

## Declared conventions where the paper is incomplete

These choices make the independent implementation deterministic and auditable;
they are not attributed to the authors' unavailable code.

- The `k` neighbors are the nearest rows across both observed classes, excluding
  the query by original row index. Equal distances resolve by original row
  index.
- `m_i`, both mean distances and the partner pool use that same all-class
  neighborhood. A partner is uniformly selected from its minority-labelled
  members.
- With no local minority member, distance evidence is zero and the row is
  ineligible because it has no interpolation partner. With no local majority
  member, distance evidence is one. Coincident local cross-class points that
  make both means zero are explicitly unsupported.
- If generation is required but all priorities are zero, the sampler reports an
  unsupported geometry. It does not silently fall back to SMOTE, ROS or uniform
  seeds.
- The target minority count is `floor(target_ratio * majority_count)`. Every
  original row remains in the output. One fresh NumPy generator draws seed,
  partner and interpolation fraction in that order for each synthetic row.
- The sampler receives already prepared fitting features and observed labels
  only. Scaling and evaluation remain outside it. Hidden true labels, subgroup
  IDs, corruption flags, selection data and held-out data are never inputs.

The implementation records full neighborhood scores, eligibility, priorities,
parent row IDs and interpolation fractions. Hidden subgroup and corruption
diagnostics may be joined to those row IDs only after fitting and selection are
locked.

## Stopping point

This stage provides the standalone baseline and software checks only. The
separate [NADOS diagnostic protocol](NADOS_DIAGNOSTIC_PROTOCOL.md) now
integrates it with the existing 90-case synthetic design, and its first
authorized experiment is complete. Cluster-Safe Oversampling and a
proposed rare-subgroup-aware method are not implemented here.
