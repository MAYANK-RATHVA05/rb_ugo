# Manual Stage 4R-7 benchmark execution

The complete declared matrix contains 74 datasets, 14 classifiers, 11
conditions and 10 outer folds: **113,960 jobs**. Results are local and ignored
by Git. Each fold job is written atomically, so rerunning the same command reuses
finished records.

The 74 datasets are evaluation benchmarks, not one combined training set. A
smaller run can validate the pipeline and support a clearly scoped exploratory
comparison. A claim about the full paper benchmark requires all 74 datasets and
the additional analyses specified by the paper.

## Current local progress

The initial run was stopped at the user's request after **28,919 saved jobs**.
Of 74 datasets, **18 are complete** (27,720 terminal jobs): 27,610 successes,
110 recorded failures and no unavailable classifiers. The failures comprise
109 UGO-MCSMOTE Sinkhorn non-convergence cases and one QDA invalid-probability
case after random undersampling. Four other datasets have partial records.
These 18 datasets are concentrated among the small Ecoli variants and are not a
representative sample of all 74. Treat any analysis of them as exploratory.

To inspect the saved work without starting training:

    .venv/bin/python main_stage4r7.py --output results/stage4r7-full --status-only

No benchmark worker should be running while reviewing these records. The local
results are ignored by Git; a different machine needs the copied raw inputs and
result directory to resume them.

## Resume in the current cloud workspace

From `/workspace/rb_ugo`:

    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    export MPLCONFIGDIR=/tmp/ugo-matplotlib XDG_CACHE_HOME=/tmp/ugo-cache
    .venv/bin/python main_stage4r7.py --output results/stage4r7-full --workers 4

Leave that command running in a terminal. It performs computation locally; it
does not consume chat tokens. Four workers were chosen for the five-core cloud
environment so the scheduler and operating system retain capacity.

For a detached Linux run, use a process group so the scheduler and workers can
be stopped together:

    setsid env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ugo-matplotlib XDG_CACHE_HOME=/tmp/ugo-cache .venv/bin/python main_stage4r7.py --output results/stage4r7-full --workers 4 > results/stage4r7-run.log 2>&1 &
    echo $! > results/stage4r7-run.pid
    tail -f results/stage4r7-run.log

To stop that detached run safely, signal its process group once and wait for
workers to exit:

    kill -TERM -- -$(cat results/stage4r7-run.pid)

Do not change files in `src/`, `native/`, `requirements*.txt`, or
`scripts/setup_c50.py` while resuming this result directory. They are part of
the saved run identity. A changed implementation must use a new output path.

## Run on another Linux machine

Git does not contain raw datasets or result records. Copy both
`data/raw/ugo74_approx/` and, if resuming current progress,
`results/stage4r7-full/` to the same repository-relative paths. Then run:

    python3.11 -m venv .venv
    .venv/bin/python -m pip install -r requirements-dev.txt -r requirements-classifiers.txt
    .venv/bin/python scripts/setup_c50.py
    .venv/bin/python -m pytest -q
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    .venv/bin/python main_stage4r7.py --output results/stage4r7-full --workers 4

The runner validates all 74 source hashes before training. It refuses a result
directory created by a different dataset, configuration, software version or
backend build.

Failures remain terminal records with empty scores. Current observed failures
include UGO-MCSMOTE Sinkhorn non-convergence and one QDA invalid-probability
case after undersampling. Do not delete, replace or average over these failures;
the final report must show paired coverage explicitly.
