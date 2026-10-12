# Fourteen-classifier completion

All fourteen classifier identities in UGO Tables 7–8 now have executable,
cloneable binary probability interfaces. This closes the earlier software gap;
it does not claim author-package equivalence or reproduce benchmark results.

The five additions are:

| Identity | Implementation | Paper-aligned setting | Declared uncertainty |
| --- | --- | --- | --- |
| BART | `stochtree==0.2.1` probit BART | 20 trees; mean of 100 posterior probability draws | priors, burn-in and package were unstated |
| BNNET | NumPy mean-field Gaussian variational network | two hidden layers of 20, dropout .25, RMSprop, warm-up 100, patience 20; 100 posterior weight draws | activation, prior, learning rate, batch, validation and inference method were unstated |
| C5.0 | pinned CRAN C50 0.2.0 native learner | CF=.25, early stopping enabled | package version and trial count were unstated; adapter uses one tree trial |
| GKNB | Gaussian product-kernel naive Bayes | per-class/per-feature unbiased-CV bandwidth | paper kernel and bandwidth implementation were unstated; large samples use 128 bins |
| NNET | NumPy dense network | two hidden layers of 20, dropout .25, RMSprop, warm-up 100, patience 20 | activation, learning rate, batch, validation and epoch cap were unstated |

The C5.0 setup verifies the pinned upstream source hash before compiling. Its
learner and probability sources are unchanged; a GPL-compatible adapter replaces
R allocation/output glue and runs each fit/prediction in a separate process.
The evaluation run identity includes source files, optional package versions and
the native build manifest, so cached jobs cannot cross backend changes.

Install and build from the repository root:

    .venv/bin/python -m pip install -r requirements-dev.txt -r requirements-classifiers.txt
    .venv/bin/python scripts/setup_c50.py
    MPLCONFIGDIR=/tmp/ugo-matplotlib XDG_CACHE_HOME=/tmp/ugo-cache .venv/bin/python -m pytest -q

Real-data training has not been started in this stage.

Validation completed with **329 passing tests** and a fitting-only smoke run of
all fourteen classifiers on 1,050 generated rows.
