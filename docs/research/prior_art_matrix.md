# Stage 4A: prior-art evidence and mathematical comparison

Audit date: 2026-10-10. This is an evidence-limited research audit, not an
implementation or a benchmark report. All links below are citations, not evidence
that every linked page was accessible. Claim scope and access status are explicit.

## Verification ledger

| Requested publication identifier | User-supplied citation | Full text read? | Abstract verified? | Equation/method status |
| --- | --- | --- | --- | --- |
| UGO | [10.1016/j.knosys.2026.115886](https://doi.org/10.1016/j.knosys.2026.115886) | No | No | **UNVERIFIED**; previous prompt and local code are not the paper |
| SLEO | [10.1016/j.knosys.2025.115009](https://doi.org/10.1016/j.knosys.2025.115009) | No | No | **UNVERIFIED**, including acronym expansion and bibliographic metadata |
| SSHR | [10.1016/j.eswa.2023.119733](https://doi.org/10.1016/j.eswa.2023.119733) | No | No | **UNVERIFIED**, including the precise semi-supervised protocol |
| GDEO | [10.1016/j.neucom.2026.134121](https://doi.org/10.1016/j.neucom.2026.134121) | No | No | **UNVERIFIED** |
| Don't Oversample the Boundary | [10.1016/j.neucom.2026.134831](https://doi.org/10.1016/j.neucom.2026.134831) | No | No | **UNVERIFIED**; title-to-DOI match not independently checked |
| Synthetic Augmentation in Imbalanced Learning | [arXiv:2601.16120](https://arxiv.org/abs/2601.16120) | No | No | **UNVERIFIED**; title-to-identifier match not independently checked |

All six exact supplied landing URLs returned proxy tunnel `403 Forbidden`.
The arXiv PDF and export API also returned that error. Crossref and OpenAlex
lookups for UGO were blocked. These are **network-access failures**, not evidence
that a paper is nonexistent, paywalled, invalid, or methodologically deficient.
No verified abstracts or section previews for these six were available. Inferring
algorithms from their titles would fabricate evidence, so the comparison cells
below deliberately remain unknown. No negative novelty claim about any of the six
is licensed by missing access.

Workspace and merged `main` contained only `docs/ugo_reproduction.md`, not a
related-paper audit. The available Stage 2 branch had no research audit either.
No paper PDFs were attached. The existing UGO document explicitly records
provisional pools, cap, rounding, and neighbor conventions. Its code was preserved
unchanged in this stage; “original UGO” is still an **unverified reproduction
baseline**, not a source-validated original algorithm. See
[the existing reproduction record](../ugo_reproduction.md).

## Required comparison: six requested sources

**U = UNVERIFIED from a primary source.** U never means “absent.”

| Source | Optimized/estimated quantity | Synthetic seed choice | Minority-region difficulty | Inter-class overlap | Intra-class imbalance |
| --- | --- | --- | --- | --- | --- |
| UGO | U | U | U | U | U |
| SLEO | U | U | U | U | U |
| SSHR | U | U | U | U | U |
| GDEO | U | U | U | U | U |
| Don't Oversample the Boundary | U | U | U | U | U |
| Synthetic Augmentation in Imbalanced Learning | U | U | U | U | U |

| Source | Separate unlabelled input X? | Pseudo-labels? | Downstream classifier affects generation? | Marginal benefit of additions? | Threshold-only comparator? | Claimed guarantees and assumptions |
| --- | --- | --- | --- | --- | --- | --- |
| UGO | U | U | U | U | U | U |
| SLEO | U | U | U | U | U | U |
| SSHR | U | U | U | U | U | U |
| GDEO | U | U | U | U | U | U |
| Don't Oversample the Boundary | U | U | U | U | U | U |
| Synthetic Augmentation in Imbalanced Learning | U | U | U | U | U | U |

For each source, the author equations are **not verified**. Before replacing a U,
record the exact page/section/equation and whether it came from full text, an
abstract, or a section preview. Abstract-only evidence cannot establish an absent
threshold baseline, a specific eligible-neighbor set, or an unmentioned theorem.
When a source becomes available, distinguish using genuinely separate unlabelled
observations from temporarily hiding known labels, labelling synthetic points, or
running unsupervised clustering on already-labelled fitting features.

The prompt-derived local UGO implementation computes signed MMA, selects a class
by its mean MMA, uses kNN noise filtering/relocation, and applies Welch stopping.
Those are verified **repository behaviors**, not verified claims about [UGO].
No exact original cap equation is available. Do not use this audit to relabel
the local provisional formula as a published equation.

## Additional publications verified from full text

Three complete 10-page PDFs were obtained with HTTP 200 from **PMLR's public
GitHub repositories**. Their titles/authors match the official repository's
publication metadata. Relevant mathematical and experimental sections were read;
supplementary appendices/proofs were not obtained. PDF hashes are recorded below.
These sources do not fill the six requested evidence gaps, but they materially
limit possible novelty claims.

### [R] Ren et al. (2018), Learning to Reweight Examples for Robust Deep Learning

Full-text source: [PMLR PDF mirror](https://raw.githubusercontent.com/mlresearch/v80/gh-pages/ren18a/ren18a.pdf).
[Official PMLR metadata and abstract](https://raw.githubusercontent.com/mlresearch/v80/gh-pages/_posts/2018-07-03-ren18a.md).

In Section 3.1, Eqs. (1)–(2), their bilevel objective is:

\[
\theta^*(w)=\arg\min_\theta\sum_{i=1}^{N}w_i f_i(\theta),\qquad
w^*=\arg\min_{w\ge0}\frac1M\sum_{j=1}^{M}f_j^v(\theta^*(w)).
\]

Their Eqs. (5), (7), and (8) use a one-step perturbed training update and a
validation meta-gradient:

\[
\widehat\theta_{t+1}(\epsilon)=\theta_t-
\alpha\sum_{i=1}^{n}\epsilon_i\nabla f_i(\theta_t),\quad
u_{i,t}=-\eta\left.\frac{\partial}{\partial\epsilon_i}
\frac1m\sum_j f_j^v(\widehat\theta_{t+1}(\epsilon))\right|_{\epsilon=0},\quad
\widetilde w_{i,t}=\max(u_{i,t},0).
\]

Eq. (9) normalizes these weights, handling an all-zero batch explicitly.
This verifies that **validation-guided marginal training influence is already a
published idea**, including class-imbalance settings. It does not establish that
their algorithm is a synthetic generator or uses our proposed local ranking
contrast. Section 3.2, Eq. (12), relates the meta-gradient to training/validation
gradient alignment. Gradient alignment alone is not a new contribution here.

Section 3.4, Lemma 1 and Theorem 2, claim monotone validation-loss behavior and
a stationary-point rate under a Lipschitz-smooth validation objective, bounded
training-loss gradients, a restricted step size, and the full validation batch
condition discussed there. Specifically the lemma gives
\(\alpha_t\le2n/(L\sigma^2)\), and Eq. (15) bounds
\(\min_{0<t<T}\mathbb E\|\nabla G(\theta_t)\|^2\le C/\sqrt T\).
These are authors' stated **optimization** claims, not guarantees of outer-test
improvement, minority recall, safe synthetic labels, or no false-positive harm.
The setup permits the small validation set to belong to the training set; that
must not be imported as an independence assumption for our proposed audit bounds.
Detailed supplemental proof verification is outstanding.

### [D] Ghorbani & Zou (2019), Data Shapley: Equitable Valuation of Data for Machine Learning

Full-text source: [PMLR PDF mirror](https://raw.githubusercontent.com/mlresearch/v97/gh-pages/ghorbani19c/ghorbani19c.pdf).
[Official PMLR metadata and abstract](https://raw.githubusercontent.com/mlresearch/v97/gh-pages/_posts/2019-05-24-ghorbani19c.md).

Section 2 defines the value function V(S) for a learning algorithm trained on S
and a chosen evaluation metric. Eq. (2) expresses the Shapley value as:

\[
\phi_i=\mathbb E_\pi[V(S_\pi^i\cup\{i\})-V(S_\pi^i)],
\]

where S_pi^i precedes i in a uniformly sampled ordering. Proposition 2.1
characterizes the value up to scale using null contribution, symmetry, and
additivity properties. This is an **axiomatic valuation** result, not a guarantee
that a high-value region benefits from unlimited synthetic interpolation.

Section 3.2 explicitly extends valuation to **groups**. Section 4.1 uses predicted
data value to guide acquisition of new patients; Section 4.5 evaluates group
valuation. Claiming that “local data value,” “subgroup usefulness,” or “marginal
benefit of extra training data” is untouched prior art would contradict these
sections. Actual newly acquired labelled observations are not equivalent to
synthetic examples derived from existing anchors. The acquisition example uses
observables to predict value; this is not a pseudo-labelled semi-supervised
oversampling algorithm.

The exact sum is exponential; Section 3.1 describes Monte Carlo and truncation.
The empirical truncation approximation is not a theorem that all omitted
contributions vanish. Section 4 explicitly separates the data used for V from
another held-out set used for reporting final results. That is a directly
relevant precedent for avoiding valuation/evaluation reuse.

### [I] Koh & Liang (2017), Understanding Black-box Predictions via Influence Functions

Full-text source: [PMLR PDF mirror](https://raw.githubusercontent.com/mlresearch/v70/gh-pages/koh17a/koh17a.pdf).
[Official PMLR metadata and abstract](https://raw.githubusercontent.com/mlresearch/v70/gh-pages/_posts/2017-07-17-koh17a.md).

Section 2.1, Eqs. (1)–(2), gives the influence of upweighting an example z:

\[
I_{up,params}(z)=-H_{\widehat\theta}^{-1}\nabla_\theta L(z,\widehat\theta),\quad
I_{up,loss}(z,z_{eval})=-\nabla_\theta L(z_{eval},\widehat\theta)^\top
H_{\widehat\theta}^{-1}\nabla_\theta L(z,\widehat\theta).
\]

The paper calls the query point z_test; here it is renamed z_eval to avoid
suggesting that outer-test labels can guide our synthesis or selection. The
formula requires an attained risk minimizer, twice-differentiable risk, and a
positive-definite Hessian in the stated convex setup. Removal and finite-batch
effects are approximations, not exact derivatives for a large augmentation.
Sections 4.2–4.3 examine damping/nonconvex and smooth-loss approximations;
empirical usefulness there does not restore the original convex guarantees.

This is verified prior art for estimating training-data impact on another
example's loss. Section 7 itself warns about moving from local perturbations to
larger subgroup changes. A finite, group-specific synthetic intervention still
needs direct confirmation rather than an influence-score guarantee.

## Mathematical comparison of verified additional sources

| Source | Quantity | Seeds | Difficulty | Overlap | Intra-class imbalance |
| --- | --- | --- | --- | --- | --- |
| [R], §3 | Validation loss under learnt example weights | No synthetic seeds; minibatch examples | Gradient contribution to validation objective | No explicit geometric overlap guarantee; noisy examples can receive zero weight | Can reweight individual examples; no explicit small-disjunct coverage theorem |
| [D], §§2–3 | Marginal contributions averaged over coalitions | No synthetic seeds; §4.1 acquires valuable new observations | Model/metric-dependent data value, including groups | Low value need not mean overlap; no overlap-removal guarantee | Group valuation is explicit, not limited to global class frequencies |
| [I], §2 | Infinitesimal parameter/evaluation-loss response | No oversampling seed policy | Curvature-adjusted training influence | Helpful/harmful effects are model-dependent, not a geometric safety certificate | Local influence may be aggregated; finite subgroup effects are not guaranteed |

| Source | Separate unlabelled features | Pseudo-labels | Classifier dependence | Marginal additions | Threshold-only comparison | Guarantee scope |
| --- | --- | --- | --- | --- | --- | --- |
| [R] | Algorithm 1 uses labelled training and clean validation; no distinct U input | Not a pseudo-label generator | Yes, through training/validation gradients | Infinitesimal example-weight effects; no synthetic-batch test | Not identified in reported experiments | Smoothness/gradient/step-size-dependent optimization result |
| [D] | Core valuation is supervised; acquisition pool covariates are used in §4.1 | Not in the core valuation algorithm | Yes, learner A is part of V | Yes; finite data/group contributions; acquisition guidance | Not identified in §§4.1–4.5 | Valuation axioms, not synthetic safety or outer generalization |
| [I] | Labelled training and an evaluation query; not a semi-supervised generator | Not part of its influence formula | Yes, gradients and Hessian | Infinitesimal weight effects; finite changes approximated | Not identified in reported comparisons | Differentiability, optimum and Hessian assumptions |

“Not identified” is limited to the reviewed full-text method/experiment sections,
not a claim about every later implementation or all related work.

## Verified implementation/documentation comparators

These are source-level evidence, **not full-text verification of their cited
original papers**:

- [S1] [scikit-learn 1.5.2: threshold tuning](https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.5.2/doc/modules/classification_threshold.rst).
  Separates prediction scores from decisions; threshold tuning leaves ROC/PR
  curves unchanged. Warns against fitting and tuning on the same observations.
- [S2] [scikit-learn 1.5.2: evaluation metrics](https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.5.2/doc/modules/model_evaluation.rst).
  Verifies balanced accuracy and AP equations. AP is not trapezoidal PR-AUC.
- [S3] [scikit-learn 1.5.2: semi-supervised learning](https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.5.2/doc/modules/semi_supervised.rst).
  Verifies a genuine unlabelled-input API, self-training with model-assigned
  labels, and graph propagation/spreading requiring distributional assumptions.
- [S4] [scikit-learn 1.5.2: cross-validation](https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.5.2/doc/modules/cross_validation.rst).
  Explains separate validation/test roles and overfitting by repeated selection.
- [S5] [imbalanced-learn 0.12.4: oversampling guide](https://raw.githubusercontent.com/scikit-learn-contrib/imbalanced-learn/0.12.4/doc/over_sampling.rst).
  Documents ROS, SMOTE, ADASYN, boundary variants and KMeansSMOTE; warns about
  outlier-driven generation. Thus clustering, boundary focus, and difficult-point
  emphasis are already established ingredients at the implementation level.
- [S6] [imbalanced-learn 0.12.4: KMeansSMOTE source](https://raw.githubusercontent.com/scikit-learn-contrib/imbalanced-learn/0.12.4/imblearn/over_sampling/_smote/cluster.py).
  `_find_cluster_sparsity` computes mean pair distance^exponent / target-class
  count; eligible-cluster sampling weights normalize these sparsities. The
  source cites Last, Douzas and Bacao, [arXiv:1711.00837](https://arxiv.org/abs/1711.00837).
  Their original paper was not read; do not attribute the version's automatic
  exponent or thresholds to an unverified original equation.
- [S7] [scikit-learn 1.5.2: graph source](https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.5.2/sklearn/semi_supervised/_label_propagation.py).
  Cites Zhou et al., “Learning with local and global consistency” (2004) for
  LabelSpreading, and Zhu/Ghahramani for LabelPropagation. The publications
  themselves were not obtained; graph implementation evidence does not verify
  their original theorem statements.

## Verified reference register and audit limits

| ID | Bibliography / primary evidence | Verification level |
| --- | --- | --- |
| R | Ren, Zeng, Yang & Urtasun. ICML 2018, PMLR 80:4334–4343. Links above. | Full primary PDF and publisher abstract/metadata |
| D | Ghorbani & Zou. ICML 2019, PMLR 97:2242–2251. Links above. | Full primary PDF and publisher abstract/metadata |
| I | Koh & Liang. ICML 2017, PMLR 70:1885–1894. Links above. | Full primary PDF and publisher abstract/metadata |
| S1–S7 | Pinned library documents/source linked above. | Complete source pages/files relevant to stated implementation claims |
| Six requested identifiers | Exact supplied links in the first table. | Neither full text nor abstracts verified |

PDF SHA-256 values:

```text
R: 3d46118cfb8575a12ed3ea2e5c90857db865fb9fc01c7f8b6565090de8ddf276
D: 48979f27f610c834ffda28ab48194c1af13c453d32add965fc817bba1e9b4407
I: 715fb9f935b0af32c7ae1f0d834fd5158edb279a575a4ae91196d1830f281f65
```

Downloaded PDFs/text were kept outside the project under `/tmp`, not committed.
No paywall circumvention or private-data access was used. This is a targeted
counterexample search, not an exhaustive systematic literature review. All
novelty conclusions must reflect that limitation.

To unblock the six-paper audit, supply legitimate PDFs/abstracts or allow the
required public destinations in environment settings: `doi.org`, `arxiv.org`,
`export.arxiv.org`, `api.crossref.org`, `api.openalex.org`, and publisher redirect
hosts such as `www.sciencedirect.com` and `linkinghub.elsevier.com`. Reachability
alone does not supply subscription rights or full-text access. No network settings
were changed as part of this documentation-only task.
