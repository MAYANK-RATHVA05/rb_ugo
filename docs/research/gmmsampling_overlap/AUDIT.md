# GMMSampling minority-class overlap: feasibility audit

20 October 2026. **Recommendation: refine before implementation.** The
author-stated limitation is real, and a small conditional candidate is
technically plausible. Its exact novelty is unresolved because required
primary texts, especially the JSM abstract, were inaccessible. No sampler,
GMM, preprocessing model or classifier was fitted or implemented in this audit.
Existing implementations, measured results and failed advancement decisions
are preserved. Source identities/access limits are in
[SOURCE_REGISTER.json](SOURCE_REGISTER.json); the bounded future design is
[PILOT_PROTOCOL.md](PILOT_PROTOCOL.md), **not authorized for execution**.

## Scope and verified limitation

AGENTS.md requires explicit stage authorization and training-only operations;
it contains no prohibition on multiclass research. This request authorizes an
audit. The repository's current **runtime is binary**: README describes that
scope; `src/_validation.py:binary_labels`, `src/data_loader.py` (line 281),
`src/metrics.py` and `src/evaluation.py` enforce two classes. A multiclass
sampler cannot be inserted into those interfaces unchanged. Future work needs
a separate, explicitly authorized multiclass loader/evaluator, preserving
existing binary APIs. This is an implementation-contract conflict, not a ban
on a report or separate research direction.

Naglik and Lango, [GMMSampling](https://doi.org/10.1007/s10994-023-06416-8),
online 2023, *Machine Learning* 113:5183–5202 (2024), Section 5,
**printed p. 5200 / supplied PDF p. 18 of 20**, states:

> “the presented method fits a probabilistic model for each minority class
> separately. Therefore, it does not take into account the distributions of
> the other classes and potentially can fail to model complex scenarios with
> strongly overlapping minority classes.”

This is a stated possibility/future direction, not proof that every overlap
configuration defeats GMMSampling. The authors also say minority–minority
overlap was less harmful than other overlap types in cited experiments.

## What GMMSampling actually does

- Section 3.2, pp. 5191–5192: target class size is the mean of the largest
  minority class and smallest majority class. Minority classes grow and
  majority classes shrink to this intermediate target.
- Fit a separate GMM for each minority class, selecting component count
  through within-class fitting/development likelihood and refitting that
  GMM on original fitting observations. GMM components approximate subconcepts;
  they are not guaranteed to match genuine latent groups.
- Eq. (1), p. 5187: `safe(x)=sum_j n_j*mu(c,j)/k`, with
  `mu(c,j)=min(n_c,n_j)/max(n_c,n_j)` for the stated heuristic.
  Other classes already influence this neighbourhood score. The limitation
  concerns separate density models, not total absence of cross-class information.
- Eqs. (2)–(3), p. 5192: allocate draws using
  `Qtilde_ck=sum_{x in class c}(1-safe(x))*p(x|component k)`, then normalize
  across components. The prose assumes uniform component priors. This is
  **not simply equal synthetic counts per component**, nor ordinary sampling
  according to the fitted mixture weights.
- Draw directly from the selected Gaussian component; no rival-minority
  density condition is specified. Remove least-safe majority observations.

A useful deduction from Eq. (1): **equally sized but different minority
classes have mu=1**, just like same-class neighbours. A neighbour from the
other equally small class therefore does not lower this safe score, whereas
a much larger majority class does. This explains why a separate class-competition
diagnostic is relevant; it is not new empirical evidence.

## Author implementation: verified source, not exact paper equivalence

The paper's footnote 2, p. 5194, links
[damian-horna/multi-imbalance](https://github.com/damian-horna/multi-imbalance)
and specifies **develop**. Inspected develop commit
`7a89dd674a7a10d73ac03548a2ee9b48cd04c654` and its
[`gmm_sampler.py`](https://github.com/damian-horna/multi-imbalance/blob/7a89dd674a7a10d73ac03548a2ee9b48cd04c654/multi_imbalance/resampling/gmm_sampler.py).
The GMMSampler feature branch at `aff0cafe8e6b508e544966dd6a30ffff93eac563`
has identical sampler bytes. The inspected master tip
`e2535f5f7f8e1f4d51f8ffe53a775e92056162a8` lacks this sampler. It is
unsafe to install master and assume GMMSampling is present.

Material findings from static inspection:

| Topic | Paper versus inspected code |
| --- | --- |
| Allocation | Lines 386–392 use fitted `GaussianMixture.predict_proba`, including learned component weights, then add `1e-8`. That is not the stated uniform-prior/raw-component-density expression. Raw likelihood and per-point normalized responsibilities also differ when summed over x. A future baseline must name its convention. |
| New-point filtering | `filter_new=-1` defaults to no filter. Optional filtering (lines 412–445) is Mahalanobis distance from the **same** component's centre, not competition with other minority densities. It is not the proposed cross-minority gate. |
| Exact counts | Each component uses `int(Qk*budget)`. Rounding can lose rows; with default filtering disabled, `_create_samples` can emit at least dimension+1 rows for a smaller positive request. Exact class balance is not guaranteed by this code. |
| GMM selection | Code retains the last accepted model using development log-likelihood change. Algorithm 1 line 10 prints `until l_best < l`, conflicting with prose that stops on a decrease; do not claim a unique paper stopping rule. |
| Neighbours/RNG | Default k=7, excludes the first returned neighbour position rather than checking self ID; duplicate-distance ties need audit. Constructor sets global NumPy seed and output is shuffled. |

These are baseline-fidelity/engineering questions, **not research contributions**.
No author module was imported/executed, and no published scores were reproduced.

## Three different overlap questions

| Type | Meaning and relevance |
| --- | --- |
| Minority–majority | A minority label shares feature regions with a large class. GMMSampling's safe levels and majority removal already address it. The candidate below does not claim to solve it. |
| Different minority classes | Two scarce **different labels** occupy the same region. Generating ambiguous points for either label can increase mutual confusion. This is the Section 5 target. |
| Same-class subgroups | Several components share **one label**. Overlap between those components is not label confusion. Quota preservation may help representation, but component counts do not establish true-subgroup coverage. |

## Prior-art comparison and access limits

Links are primary publication or implementation sources. “Not equivalent” below
means not equivalent **in the inspected material**, not a verified novelty finding.

| Source | Verified mechanism / precise relation |
| --- | --- |
| [SOMM 2025 DOI](https://link.springer.com/article/10.1007/s10115-025-02394-6); [matching R repository](https://github.com/hadi1453/Synthetic-Over-sampling-with-the-Minority-and-Majority-classes-for-imbalance-problems/tree/8c406e8018e1122c551304332bceefe1e7837fa3) | README and `SOMM - Function.R` generate within the target class's coordinate bounds, inspect all-class neighbours, reject when no target neighbour exists, otherwise accept or displace towards a target neighbour. Competing minority labels are included among non-target labels. This already addresses multiclass overlap; generic rival-aware rejection is not a new idea. It does not use the inspected GMM density-ratio/component-quota mechanism. Publisher/PDF/Crossref access was HTTP 403; the 2023 repository cannot be confirmed to match the final 2025 article. |
| [NDESO 2025](https://www.techscience.com/cmc/v83n3/61046/html); [source](https://github.com/goshlive/imbalanced-ndeso/blob/aefd3b4be0966f42bc1b49031d392f31a048f67e/ndeso.py) | README and source select observations with more different-class than same-class neighbours, also select nearby same-class rows, reposition using class centroid/neighbour distances, then default to random oversampling. It changes originals and considers all competing labels; it is not a GMM draw-acceptance rule or a fixed component-quota mechanism. The code's relocation is not a guarantee of moving closer. Publisher/PDF and author-linked [preprint](https://arxiv.org/abs/2501.04099) were inaccessible (403); equivalence to the published revision is unverified. |
| [MCD-regularized GMM oversampling, JSM 2026](https://ww2.amstat.org/meetings/jsm/2026/onlineprogram/abstract.cfm?sid=6951&tid=7350) | **Abstract inaccessible, HTTP 403.** Only the user's description is available. Authors, exact title, meaning of MCD, objective, class scope, generation/acceptance and quota rules remain unverified. Cannot claim it is either equivalent or different; this is the principal unresolved collision risk. |
| [SOUP 2019 reference source](https://github.com/damian-horna/multi-imbalance/blob/7a89dd674a7a10d73ac03548a2ee9b48cd04c654/multi_imbalance/resampling/soup.py) | Class-size-weighted safe levels, majority removal and duplication of safer minority observations. Same-class-size blind spot above applies to this particular similarity rule. No GMM rivalry gate or component-conditioned refill in this source. Original article not read; cited directly by GMMSampling. |
| [SMOM 2017](https://doi.org/10.1016/j.patcog.2017.07.024); [reference implementation](https://github.com/damian-horna/multi-imbalance/blob/7a89dd674a7a10d73ac03548a2ee9b48cd04c654/multi_imbalance/resampling/smom.py) | NBDOS clustering and neighbour/interpolation-pair weighting distinguish target/non-target and minority/majority contexts. Class-aware local overlap handling predates 2021. The inspected reference implementation is not class-GMM posterior rejection with preserved GMM quotas. Original paper full text not accessed; do not infer exact paper equivalence from this implementation. |
| [MC-CCR 2020](https://doi.org/10.1016/j.knosys.2020.106223) | GMMSampling's primary related-work section describes energy-bounded spheres, moving majority observations out, and generating minority observations inside. Relevant multiclass overlap baseline. Its original full text/code were not verified here; exact minority-rival rules remain unchecked. |
| [MDO reference implementation](https://github.com/damian-horna/multi-imbalance/blob/e2535f5f7f8e1f4d51f8ffe53a775e92056162a8/multi_imbalance/resampling/mdo.py), 2015/2016; [difficulty study 2022](https://doi.org/10.1016/j.eswa.2022.116962) | GMMSampling discusses Mahalanobis generation and different pairwise overlap types. These establish relevant earlier sampling/analysis, not a claim that our exact mechanism is absent. Original full texts unverified. |

Additional bounded searches covered GMMSampling extensions, density-ratio
oversampling and MCD-regularized mixtures. Public repository searches enabled
code discovery but are not exhaustive scholarly searches. General search and
publisher/preprint endpoints were blocked; unauthenticated GitHub code search
offered a login shell. Missing search hits do not establish novelty.

## One conditional candidate

**Component-quota-preserving cross-minority rejection**, a descriptive working
name. Keep original GMM fits, allocation, target sizes and majority removal.
For a draw z from component k of minority class c, compute in log space

`a_c(z) = f_c(z) / sum_{j in minority_classes} f_j(z)`

using class-conditional GMM densities with equal class priors. Accept if
`a_c(z)>=tau`. Rejection redraws from **that same component**, not a safer
component. Cap attempts at 20 per requested row. Fill an exhausted component's
remaining quota by duplicating its original hard-MAP-assigned fitting members;
if the component has no such members, record unsupported coverage. Original
minority rows are always retained. Tau=0 must reproduce the baseline exactly.

This couples existing class densities only at generation time. It does not
introduce a joint GMM objective, fit a classifier gate, redefine the safe
level, or claim fitted density scores are calibrated probabilities. Equal
priors avoid interpreting original class imbalance as a reason to reject
the smaller minority label. A zero-quota/missed component is not rescued.

The **specific hypothesis** is that keeping the allocation attached to its
component prevents a class-wise rejection filter from spending the entire
budget in already easy subgroups. Generic posterior rejection alone is not
the candidate contribution: compare the *same gate* with global refill and
without component quotas. Duplication fallback is a coverage policy, not a
guarantee of disambiguating intrinsically overlapping labels.

No exact equivalent was found in the inspected code. The missing JSM/SOMM
full-text checks prevent declaring the combination new. If an existing method
has the same rival-density acceptance, component-conditioned refill and
coverage policy, treat it as prior art or a baseline—not a renamed invention.

## Feasibility, failure cases and decision

- Identical minority distributions cannot be separated from these features.
  High rejection can simply truncate genuine overlap or substitute ambiguous
  duplicates; no method can remove the Bayes-error floor by oversampling.
- Small/anisotropic/high-dimensional groups can yield unstable GMM densities;
  rare modes may be missed entirely. Same-class overlapping components are
  unidentifiable and are not separate labels to repel.
- Fixed component quotas preserve estimated allocation, not true latent-group
  coverage. A rare subgroup can be lost through fitting, zero allocation or
  wrong component assignments. Fallback can replicate label noise and
  overfit; rejected-sample counts/duplication must be reported separately.
- Cross-minority gates ignore majority density, may worsen minority–majority
  errors, depend on priors, and add rejection cost. Score underflow requires
  log-sum-exp; candidate caps must be explicit. Do not relax gates after failure.

The density check itself is small; a reliable project change also needs a
code/paper-convention baseline, multiclass contracts, provenance and tests.
Planning estimate after sources are resolved: roughly **2–4 engineering days**
for baseline/harness/candidate checks, not a performance promise. Compatibility
of old private imbalanced-learn imports/Pydantic APIs is untested.

**Refine**: obtain the JSM abstract and final SOMM/NDESO texts, settle baseline
allocation/stopping/count conventions, and check the *exact* combined mechanism
against those sources. Then seek separate implementation approval for the
small proposed protocol. Stop this novelty path if an equivalent is found;
otherwise run a bounded feasibility test without promising improvement. A
paper-stated limitation alone is insufficient justification for a new method.
