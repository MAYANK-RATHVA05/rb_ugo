# Generator implementation: Stage 4R-3, partial

This stage adds a shared batch interface, provisional numerical RWO, UGO-RWO,
and a standalone RWO baseline. ROS/SMOTE behavior is preserved. **Stage 4R-3 is
partially complete:** MCSMOTE, UGO-MCSMOTE and LEAO require their original
specifications or verified implementations. No simulation, benchmark, predictive
score or paper-result comparison is produced.

## Evidence obtained

The uploaded UGO article names RWO in Section 4.2 and cites Zhang and Li,
*RWO-sampling: A random walk over-sampling approach to imbalanced data
classification*, Information Fusion 20 (2014), 99–116,
DOI [10.1016/j.inffus.2013.12.003](https://doi.org/10.1016/j.inffus.2013.12.003).
It does not supply the full RWO algorithm.

We downloaded and inspected the published `smote-variants==1.0.1` source archive
from PyPI without installing or executing it. Its
`smote_variants/oversampling/_rwo_sampling.py` cites that article and uses:

    stds = np.std(X_min, axis=0) / np.sqrt(len(X_min))
    samples = samples + random_state.normal(size=samples.shape) * stds

Its default simplex dimension is one, so its seed-generation configuration
operates on individual input points. We also inspected the corresponding
`base/_simplexsampling.py`. This is **third-party numerical reference evidence,
not the original paper or author code**. Exact archive/file SHA-256 hashes and
retrieval URL are recorded in [SOURCE_REGISTER.json](SOURCE_REGISTER.json).
The MIT notice is preserved in [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).

DOI requests for RWO and MCSMOTE, and the GitHub repository search for MCSMOTE,
returned HTTP 403 in this environment; their bodies were not obtained. We cannot
attribute those denials to the publishers or conclude that author code does not
exist. The exact PyPI `mcsmote` metadata endpoint returned 404. No original
RWO/MCSMOTE full text has become available. Older uploaded SLEO is a different
method from the UGO paper's LEAO comparator and is not substituted.

## Implemented numerical contract

`src.generators.generate_batch` receives original fitting features and labels,
informative seed indices, the selected class, the fixed pre-noise mask, exact
batch size and a local NumPy generator. It rejects noisy, out-of-bounds,
noninteger or wrong-class seeds. It accepts no classifier or held-out data.

ROS and SMOTE retain their previous algorithms and random draws. SMOTE still
returns a diagnosed insufficient-neighbor result, never a fallback.

For RWO, let C be all original pre-clean rows in the selected class:

    s_j = population_std(X[C, j]) / sqrt(len(C))
    x_new[j] = x_seed[j] + Normal(0, 1)[j] * s_j

Seeds are sampled uniformly with replacement from the informative seed set;
perturbations are independent across generated rows/features. `ddof=0` follows
the inspected numerical source. Spread includes clean nonseed class rows and
excludes noisy rows, other classes and generated rows. It is frozen to original
data even when classifiers refit on augmented data. UGO then applies its usual
final voting and relocation, batch size, cap and uncertainty stopping rules.

The UGO seed/spread pools are explicit adaptation choices, **not verified author
UGO-RWO conventions**. Our `default_rng` differs from the third-party package's
RandomState/simplex machinery, so neither random sequences nor complete outputs
are claimed identical. No third-party dependency is added.

Zero-spread features remain constant. A singleton spread pool yields duplicates
with zero perturbation and records that degeneracy; this does not switch methods.
Overflow/nonfinite perturbations raise a clear error. Gaussian samples may lie
outside the seed convex hull; final UGO relocation remains active.

Only dense numerical features are supported. Discrete/categorical RWO rules have
not been verified or implemented. Integer-coded categorical inputs cannot be
recognized from their dtype, so callers must supply meaningful numerical data.

`RandomWalkOversampler` supplies a provisional standalone RWO comparator. It
uses every original minority row as both seed and spread reference, adds the
original class-count difference, and returns copied originals unchanged at the
front. Balanced input returns copies with no generation. It performs no UGO
filtering, uncertainty selection, Welch test or relocation. Both RWO interfaces
require `allow_provisional=True` and expose the distinct status
`third_party_numeric_reference_original_unverified`.

## Usage

```python
from sklearn.linear_model import LogisticRegression
from src.generators import RandomWalkOversampler
from src.ugo import UGO

baseline = RandomWalkOversampler(random_state=42, allow_provisional=True)
guided = UGO(LogisticRegression(max_iter=1000), variant="rwo",
             random_state=42, allow_provisional=True)
# Apply each independently to fitting data, after training-only preprocessing:
# X_base, y_base = baseline.fit_resample(X_fit, y_fit)
# X_guided, y_guided = guided.fit_resample(X_fit, y_fit)
```

`variant="mcsmote"` raises an actionable error before classifier fitting. No
transition probabilities, graph construction, interpolation or default settings
are inferred from the method's name.

## Validation and remaining work

Tests check numerical population spread analytically and empirical Gaussian
spread on a deterministic fixture, exclusion of noisy/other-class rows,
nonseed spread references, constant/singleton features, reproducibility, input
preservation, minority balancing, cloning, invalid inputs, overflow and clearing
stale state. Shared UGO checks now include RWO for majority/minority selection,
fixed batches, augmented fits and both cap policies with clipped final batches.
Real-estimator checks include string labels and all three UGO generators.

Before refactoring, four ROS/SMOTE cases (both selected classes) were captured;
the refactored arrays and generation diagnostics matched exactly. These checks
establish software behavior, not agreement with author results.

Validation on Python 3.11: `pytest -q tests/test_generators.py tests/test_ugo.py`
passed 82 tests; the full `pytest -q` suite passed 198 tests. The source archive's
SHA-256 matched PyPI metadata before inspection. No dependency pins changed.

Remaining in this stage: obtain original RWO specification and validate numerical
and discrete rules; obtain MCSMOTE and LEAO sources; implement their supported
generators/comparators and UGO-MCSMOTE with source-based edge-case checks.
Remaining later stages: dataset/simulation/classifier setup, evaluation,
statistical analysis and comparison with published results. See the
[full specification](SPECIFICATION.md).
