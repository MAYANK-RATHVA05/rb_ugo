# AW-SMOTE paper and author-code audit

Audited: 19 October 2026. This is a static source review, not a reproduction result. No author Python code was imported or executed.

## Sources

Shen Yan, Haifeng Guo, and Xiaoming Su, **Adaptive Weighting–Synthetic Minority Oversampling Technique**, *Mathematics* 14 (2026), 3238, [doi:10.3390/math14173238](https://doi.org/10.3390/math14173238). References below use the PDF's printed page numbers, out of 32. The uploaded `mathematics-14-03238-v2.pdf` has SHA-256 `52565ba72dde51afaf7567a5553d9760bceca25fc66701d550b17a8f93a266b6`.

The paper links [guo-source-1/AWSMOTE](https://github.com/guo-source-1/AWSMOTE) on p. 9. Its `main` tip at inspection was [`f0dc14807e02292a096e261ca9af21b0d1300753`](https://github.com/guo-source-1/AWSMOTE/tree/f0dc14807e02292a096e261ca9af21b0d1300753), committed 27 August 2026. The inspected tree contains only `README.md`, `adw.py`, and `results.docx`; it contains no evaluation runner, tests, dependency declaration, or license file. The README is just a repository heading. This audit covers that pinned tree, not every historical or unpublished version. No author files are vendored here.

The sole sampler is called `ADWSMOTE` in [`adw.py`](https://github.com/guo-source-1/AWSMOTE/blob/f0dc14807e02292a096e261ca9af21b0d1300753/adw.py). All line references below refer to that immutable file. [SOURCE_REGISTER.json](SOURCE_REGISTER.json) records file hashes.

## Paper versus released code

| Topic | Supplied paper | Pinned `adw.py` | Consequence |
| --- | --- | --- | --- |
| Structure discovery | Minority-only DBSCAN, pp. 6–7; MinPts=2, 65th percentile of k-distances, default k=5, pp. 24–25. | Lines 50–60 calculate distances with `n_neighbors=min_samples`; the query includes itself. At default 2, the last distance is to the nearest other point, rather than the fifth other point. | Radius estimation is coupled to MinPts. The paper does not fully specify self inclusion or which k is used for this radius. Do not call these equivalent conventions. |
| No clusters | Paper does not specify an all-noise fallback. | Lines 62–65 turn all minority observations into one cluster if no cluster survives. | An extra fallback can reconnect separated groups. It must be disclosed, not inferred as a paper rule. |
| What is removed | Algorithm 1, lines 4–5 on p. 10, replaces minority data and D after removing noise; §4.10 calls AW-Remove removal before oversampling. | Phase 1 filters generation clusters, but lines 208, 217 use original X/y for quota boundaries; lines 221–224 append synthetic rows to the entire original X/y. | Released code removes points from the seed pool, not from the final classifier's original training rows. It is not the paper's AW-Remove treatment. |
| Cluster weights | Eqs. (1)–(3), pp. 7–8: maximum majority-neighbor count within each cluster, using cleaned D. | Lines 84–102 use cluster maxima, but query the original D; zero total weight becomes uniform cluster weights. | Maxima agree in form. Reference data and the zero-weight fallback need explicit treatment. |
| Boundary neighbor convention | k=5; self inclusion is not specified. | Lines 85, 92 and 121, 130 query explicit rows against data containing those rows, so the query itself occupies one neighbor position. | Five returned rows usually mean four other observations, except duplicate/tie cases. |
| Cluster quota rounding | Eq. (4), p. 8: `floor(w*N + 0.5)`, followed by correction at the largest weight. | Lines 103–108 use NumPy round-to-even and correct the first largest rounded quota. | Half-integer and tie cases can differ. The correction has no explicit nonnegative-quota guard. |
| Intra-cluster boundary labels | Eqs. (1), (5), Algorithm 1 require labels to correspond to the rows being queried. | Lines 118–122 build D as `[clean clusters, majority]`, but set the majority mask from the original y. Lines 130–131 index this mask using indices into reordered D. | This is a row/label alignment defect for general input order and after filtering. Boundary weights and resulting probabilities may be wrong. |
| Density and fusion | Eqs. (6)–(11), pp. 8–9: inverse mean within-cluster distance, normalized density/boundary weights, sigmoid with theta=.2, gamma=5. | Lines 134–179 follow this form; delta is `1e-8`, and zero boundary weights become uniform. `alpha_fixed` supports fusion ablations. | Core formula is recognizable; the boundary input defect still affects it. The paper does not specify every numerical fallback. |
| Seed allocation | p. 9 requires per-point counts summing to the cluster quota, without specifying an allocation algorithm. | Lines 181–188 repeatedly draw categorical seeds from P. | This is a useful disclosed convention, not a uniquely specified paper rule. |
| Interpolation partner and interval | Eq. (12), p. 9: a neighboring member of the same cluster and lambda in (0,1); exact neighbor pool not specified. | Lines 192–197 choose any other member of the cluster and lambda uniformly in [0.2,0.8). | The restricted interval changes where synthetic samples can lie. |
| Number generated | Eq. (26), p. 13: N=n_majority−n_minority, with a claimed final 1:1 ratio. | Lines 214–224 use counts before filtering and retain originals, achieving 1:1 if N points are produced. Singleton clusters, if possible under nondefault settings, are skipped without reassignment (lines 125–126). | Removing original rows while holding this N fixed would not give final 1:1. The cleaning/counting convention must be stated. |
| Noise ablations | §4.10, pp. 23–24: Remove, KeepNoise, SoftNoise; Eq. (29) gives `P_new=r*P`. | No KeepNoise/SoftNoise mode or reliability coefficient appears in the release. | Do not claim author implementations of these variants are available. Their normalization, noise-neighbor pool, and reliability choices remain incomplete in the paper. |
| Reproducibility | §4.3, pp. 13–15: fixed classifiers, shuffled stratified fivefold CV. | A seeded `RandomState` is stored at construction (line 46); repeated calls advance it. CV constants are unused by an absent runner. Warnings are globally suppressed (line 6). | Seeded sampler state does not verify evaluation order, preprocessing, fold isolation, or published tables. |

### Why label alignment matters

Suppose original rows are `[majority A, majority B, majority C, minority a, minority b]`, with labels `[0,0,0,1,1]`. With no filtering, phase 3 orders rows as `[minority a, minority b, majority A, majority B, majority C]`. The correct majority mask is now `[False,False,True,True,True]`; the code still uses `[True,True,True,False,False]`. A nearest-neighbor index can therefore receive the wrong class. Filtering minority rows creates another mismatch even if the original data were minority-first. This is a static index example, not an empirical performance result.

## What the reported numbers do and do not establish

`results.docx` is a text log of aggregate experiments, not the full evaluation program or per-fold predictions. It logs resampled class totals and subsequent scores. This does not establish whether resampling happened inside each training fold; the missing runner prevents verification. Do not claim either verified fold isolation or proven leakage from that log alone.

There are additional unresolved reporting differences:

- Table 1, p. 12, lists `Winequality-red-3_vs_5` with n=691 and IR=39.31. The released log lists 681/10 for the same name and n, which implies IR=68.1.
- Tables 4/11 report aggregate AW F1=.9309, whereas Tables 12/13 give AW F1=.5446 at noise level zero. These summaries cannot be equated without the full runner and clarification of their aggregation/protocols.

These are reasons to avoid promising exact published scores. They do not by themselves invalidate the proposed mathematical method.

## Existing ideas we must acknowledge

1. **Sparse points can be genuine.** §4.10 explicitly says DBSCAN noise may be valid small disjuncts or low-density minority patterns. It reports Remove outperforming KeepNoise and SoftNoise *on average*. This does not measure true rare-subgroup recall.
2. **MinPts sensitivity is already studied.** §4.11 evaluates MinPts 2/3/5 and epsilon percentiles 50/60/65/70/80. Lowering MinPts alone is not a new contribution.
3. **Soft reliability is already proposed.** Eq. (29) attenuates noise sampling probability. Our own fully specified implementation would be a declared adaptation, not a new concept.
4. **Probability regularization is already proposed.** §4.14, pp. 28–29, discusses weight clipping, temperature smoothing, and uniform mixing.
5. **Noise is already evaluated.** §4.9 examines feature and label corruption, and the conclusion acknowledges sensitivity to DBSCAN settings. We need evidence beyond another aggregate noise table.

## Decisions for the next stage

Implement an independently written, cited baseline named **AW-paper-conventions**, with the choices fixed in the [pilot protocol](PILOT_PROTOCOL.md). It may be described as paper-based; it must not be called an exact verified reproduction. Resolve row/label alignment as a software correctness issue, use the paper's interpolation interval and quota rounding, and expose cleaning diagnostics.

Keep the frozen author release as a source reference. Do not silently patch it and then call resulting behavior the original author implementation. No performance comparison should attribute changes from these corrections to a novel research method.

KeepNoise/SoftNoise are deferred until their unspecified mechanics can be resolved or documented as adaptations. The first pilot instead includes a narrowly defined **seed-filter-only retention control**: exactly the same eligible seeds and generated samples as AW-Remove, but all original observations are retained. This is not AW-KeepNoise, because excluded points still cannot generate samples.

No novelty or efficacy conclusion is possible yet. A later pilot must first determine whether a repeatable subgroup problem survives ordinary controls.
