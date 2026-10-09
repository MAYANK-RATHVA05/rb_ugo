# Research and development rules

- Never leak outer-test data into training or model selection.
- Fit preprocessing on training partitions only.
- Generate synthetic examples from fitting data only.
- Use nested evaluation or separate selection data where necessary.
- Always compare against the original no-resampling baseline.
- Distinguish faithfully reproduced UGO from our proposed changes.
- Set and report random seeds (default RANDOM_STATE = 42).
- Never invent datasets, results, citations, or theoretical guarantees.
- Write tests for implemented functionality, including failure behavior.
- Keep code modular and readable. Target Python 3.11 and portable pathlib paths.
- Do not add unnecessary frontend, API, dashboard, database, or cloud infrastructure.
- Stop after the requested stage. Stage 1 is initialization only; Stage 2 requires user approval.
- Do not implement SMOTE, UGO, RB-UGO, model training, dataset downloading, or experiments in Stage 1.
- Codex and Antigravity are separate tools; never assume cloud changes have synchronized locally.
- After completing and validating each requested stage, commit its changes and push to GitHub. Report the commit and branch; do not proceed to the next stage without approval.
