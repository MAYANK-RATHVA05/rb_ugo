# Numerical audit — Task 1

**All specified report values match at four decimal places; no numerical mismatch was found.** The twelve named score/delta values and the two pseudo-label fractions are verified below. Evidence: [executed audit commands, lines 21–25](command_evidence.txt#L21), [recomputed_numbers.json](recomputed_numbers.json).

## What was actually recomputed

`docs/sshr_pilot/` contains aggregate CSV/JSON values, fold IDs and diagnostics, not the 25 underlying score vectors. Those summaries alone cannot independently determine sample SD. We checked their consistency and independently recomputed each held-out score from original saved predictions/probabilities under `results/sshr-small-study/<dataset>/{with-u,without-u}/job-*.json`, then checked saved test truth and pseudo-label correctness against the original local `.dat` files. No classifier or sampler was refitted. Evidence: [command_evidence.txt:21–34](command_evidence.txt#L21), [recompute_numbers.py:27](recompute_numbers.py#L27), [audit_verification.json](audit_verification.json).

For each condition, the mean is `sum(s_i)/25`; sample SD uses divisor 24 (`statistics.stdev`). FPR is `FP/(TN+FP)` and ROC-AUC uses positive-class probabilities. These are fold means, not metrics computed by pooling predictions. Zero-denominator MCC/F1 values follow the saved convention of zero; the verifier includes them rather than dropping folds. Evidence: [recompute_numbers.py:27–43,86](recompute_numbers.py#L27), [sample-SD calculation:86](recompute_numbers.py#L86), [src/metrics.py:23](../../src/metrics.py#L23). SD is descriptive because repeats share observations; it is not a standard error or confidence interval. Evidence: [scripts/run_sshr_small_study.py:96–104](../../scripts/run_sshr_small_study.py#L96).

## All 24 main-study conditions

Each entry is **mean ± sample SD**, using 25 shared successful fold events. The evidence column lists the original CSV rows for F1, G-mean, MCC, probability ROC-AUC and FPR, in that order. Every value was independently recomputed; exact unrounded results are in [recomputed_scores.csv](recomputed_scores.csv). Evidence: [command_evidence.txt:24](command_evidence.txt#L24).

### Ecoli4

| Classifier | Method | F1 | G-mean | MCC | ROC-AUC | FPR | Original CSV rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| LogReg | NORES | 0.8720 ± 0.1085 | 0.9009 ± 0.0929 | 0.8738 ± 0.1056 | 0.9955 ± 0.0071 | 0.0032 ± 0.0065 | [2, 3, 4, 5, 6](../sshr_pilot/ecoli4/scores.csv) |
| LogReg | SMOTE | 0.8194 ± 0.1350 | 0.9381 ± 0.0732 | 0.8158 ± 0.1397 | 0.9951 ± 0.0070 | 0.0210 ± 0.0199 | [12, 13, 14, 15, 16](../sshr_pilot/ecoli4/scores.csv) |
| LogReg | UGO-SMOTE | 0.8890 ± 0.1071 | 0.9287 ± 0.0863 | 0.8885 ± 0.1076 | 0.9951 ± 0.0072 | 0.0051 ± 0.0075 | [22, 23, 24, 25, 26](../sshr_pilot/ecoli4/scores.csv) |
| LogReg | BFCH-SMOTE | 0.7949 ± 0.1398 | 0.9492 ± 0.0579 | 0.7971 ± 0.1369 | 0.9960 ± 0.0058 | 0.0292 ± 0.0270 | [32, 33, 34, 35, 36](../sshr_pilot/ecoli4/scores.csv) |
| LogReg | EC-EBO | 0.7247 ± 0.1218 | 0.9373 ± 0.0586 | 0.7279 ± 0.1182 | 0.9940 ± 0.0100 | 0.0424 ± 0.0282 | [42, 43, 44, 45, 46](../sshr_pilot/ecoli4/scores.csv) |
| LogReg | SSHR | 0.5838 ± 0.1640 | 0.9331 ± 0.0342 | 0.6130 ± 0.1464 | 0.9916 ± 0.0130 | 0.1006 ± 0.0560 | [52, 53, 54, 55, 56](../sshr_pilot/ecoli4/scores.csv) |
| GNB | NORES | 0.7694 ± 0.1663 | 0.8661 ± 0.1214 | 0.7670 ± 0.1713 | 0.9871 ± 0.0160 | 0.0152 ± 0.0168 | [62, 63, 64, 65, 66](../sshr_pilot/ecoli4/scores.csv) |
| GNB | SMOTE | 0.7363 ± 0.1511 | 0.8788 ± 0.1104 | 0.7348 ± 0.1543 | 0.9888 ± 0.0134 | 0.0260 ± 0.0251 | [72, 73, 74, 75, 76](../sshr_pilot/ecoli4/scores.csv) |
| GNB | UGO-SMOTE | 0.7729 ± 0.1679 | 0.8664 ± 0.1218 | 0.7704 ± 0.1728 | 0.9871 ± 0.0160 | 0.0146 ± 0.0165 | [82, 83, 84, 85, 86](../sshr_pilot/ecoli4/scores.csv) |
| GNB | BFCH-SMOTE | 0.7201 ± 0.1471 | 0.8521 ± 0.1165 | 0.7141 ± 0.1517 | 0.9850 ± 0.0145 | 0.0215 ± 0.0188 | [92, 93, 94, 95, 96](../sshr_pilot/ecoli4/scores.csv) |
| GNB | EC-EBO | 0.5871 ± 0.1392 | 0.9093 ± 0.0748 | 0.6016 ± 0.1355 | 0.9838 ± 0.0254 | 0.0822 ± 0.0446 | [102, 103, 104, 105, 106](../sshr_pilot/ecoli4/scores.csv) |
| GNB | SSHR | 0.5714 ± 0.2103 | 0.8580 ± 0.1589 | 0.5824 ± 0.2103 | 0.9757 ± 0.0264 | 0.0779 ± 0.0612 | [112, 113, 114, 115, 116](../sshr_pilot/ecoli4/scores.csv) |

### Yeast4

| Classifier | Method | F1 | G-mean | MCC | ROC-AUC | FPR | Original CSV rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| LogReg | NORES | 0.2018 ± 0.1759 | 0.3050 ± 0.2242 | 0.2331 ± 0.1889 | 0.8718 ± 0.0652 | 0.0046 ± 0.0047 | [2, 3, 4, 5, 6](../sshr_pilot/yeast4/scores.csv) |
| LogReg | SMOTE | 0.2757 ± 0.0811 | 0.7898 ± 0.0757 | 0.3083 ± 0.0798 | 0.8698 ± 0.0574 | 0.1258 ± 0.0290 | [12, 13, 14, 15, 16](../sshr_pilot/yeast4/scores.csv) |
| LogReg | UGO-SMOTE | 0.2519 ± 0.1808 | 0.3961 ± 0.2717 | 0.2717 ± 0.1796 | 0.8737 ± 0.0615 | 0.0106 ± 0.0104 | [22, 23, 24, 25, 26](../sshr_pilot/yeast4/scores.csv) |
| LogReg | BFCH-SMOTE | 0.3025 ± 0.0875 | 0.7739 ± 0.0807 | 0.3234 ± 0.0892 | 0.8545 ± 0.0682 | 0.0995 ± 0.0253 | [32, 33, 34, 35, 36](../sshr_pilot/yeast4/scores.csv) |
| LogReg | EC-EBO | 0.3189 ± 0.0852 | 0.7854 ± 0.0746 | 0.3414 ± 0.0825 | 0.8705 ± 0.0667 | 0.0923 ± 0.0186 | [42, 43, 44, 45, 46](../sshr_pilot/yeast4/scores.csv) |
| LogReg | SSHR | 0.2962 ± 0.0908 | 0.7935 ± 0.1284 | 0.3363 ± 0.0881 | 0.8781 ± 0.0729 | 0.1107 ± 0.0351 | [52, 53, 54, 55, 56](../sshr_pilot/yeast4/scores.csv) |
| GNB | NORES | 0.2768 ± 0.1243 | 0.5895 ± 0.1255 | 0.2580 ± 0.1268 | 0.8357 ± 0.0854 | 0.0488 ± 0.0171 | [62, 63, 64, 65, 66](../sshr_pilot/yeast4/scores.csv) |
| GNB | SMOTE | 0.2301 ± 0.0642 | 0.7598 ± 0.0857 | 0.2599 ± 0.0711 | 0.8187 ± 0.0847 | 0.1535 ± 0.0382 | [72, 73, 74, 75, 76](../sshr_pilot/yeast4/scores.csv) |
| GNB | UGO-SMOTE | 0.2639 ± 0.1220 | 0.5680 ± 0.1313 | 0.2438 ± 0.1254 | 0.8326 ± 0.0836 | 0.0470 ± 0.0177 | [82, 83, 84, 85, 86](../sshr_pilot/yeast4/scores.csv) |
| GNB | BFCH-SMOTE | 0.2239 ± 0.0779 | 0.6872 ± 0.1179 | 0.2312 ± 0.0917 | 0.7782 ± 0.0847 | 0.1246 ± 0.0469 | [92, 93, 94, 95, 96](../sshr_pilot/yeast4/scores.csv) |
| GNB | EC-EBO | 0.2789 ± 0.0807 | 0.7782 ± 0.0918 | 0.3065 ± 0.0881 | 0.8360 ± 0.0969 | 0.1150 ± 0.0246 | [102, 103, 104, 105, 106](../sshr_pilot/yeast4/scores.csv) |
| GNB | SSHR | 0.2722 ± 0.0810 | 0.7032 ± 0.1294 | 0.2825 ± 0.0821 | 0.8356 ± 0.0872 | 0.0933 ± 0.0477 | [112, 113, 114, 115, 116](../sshr_pilot/yeast4/scores.csv) |

## SSHR paired with-U minus empty-U ablation

Deltas are computed on identical repeat/fold pairs, not unmatched cohorts. All 100 duplicated NORES pairs matched predictions, probabilities, metrics, preprocessing, seed and observed/test indices. Evidence: [command_evidence.txt:24](command_evidence.txt#L24), [recompute_numbers.py](recompute_numbers.py).

Each entry is **mean delta ± sample SD of paired deltas**, with 25 pairs. Exact values and original row citations: [recomputed_ablation.csv](recomputed_ablation.csv).

| Dataset | Classifier | ΔF1 | ΔG-mean | ΔMCC | ΔROC-AUC | ΔFPR | Original ablation CSV rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| ecoli4 | LogReg | -0.0777 ± 0.0857 | -0.0160 ± 0.0171 | -0.0678 ± 0.0749 | -0.0028 ± 0.0059 | +0.0303 ± 0.0324 | [2, 3, 4, 5, 6](../sshr_pilot/ecoli4/ablation.csv) |
| ecoli4 | GNB | -0.0922 ± 0.1393 | -0.0054 ± 0.0902 | -0.0810 ± 0.1301 | -0.0078 ± 0.0107 | +0.0361 ± 0.0399 | [7, 8, 9, 10, 11](../sshr_pilot/ecoli4/ablation.csv) |
| yeast4 | LogReg | -0.0096 ± 0.0294 | +0.0172 ± 0.0456 | -0.0018 ± 0.0319 | -0.0071 ± 0.0136 | +0.0145 ± 0.0216 | [2, 3, 4, 5, 6](../sshr_pilot/yeast4/ablation.csv) |
| yeast4 | GNB | -0.0054 ± 0.1099 | +0.0810 ± 0.2233 | +0.0017 ± 0.1266 | +0.0296 ± 0.0506 | +0.0333 ± 0.0568 | [7, 8, 9, 10, 11](../sshr_pilot/yeast4/ablation.csv) |

Mean F1 decreased and mean FPR increased in all four conditions. Yeast4/GNB nevertheless gained ROC-AUC and G-mean; 'U did not help on any metric' is false. Evidence: the four paired rows above and their cited ablation CSV rows.

## Checks against the supplied presentation numbers

| Report claim | Report value | Independently recomputed value | Verdict | Evidence |
| --- | ---: | ---: | --- | --- |
| ecoli4 LogReg ugo_smote f1 | 0.8890 | 0.88904762 | MATCH at 4 d.p. | [docs/sshr_pilot/ecoli4/scores.csv, row 22](../../docs/sshr_pilot/ecoli4/scores.csv#L22) |
| ecoli4 LogReg nores f1 | 0.8720 | 0.87200000 | MATCH at 4 d.p. | [docs/sshr_pilot/ecoli4/scores.csv, row 2](../../docs/sshr_pilot/ecoli4/scores.csv#L2) |
| ecoli4 LogReg sshr f1 | 0.5838 | 0.58376468 | MATCH at 4 d.p. | [docs/sshr_pilot/ecoli4/scores.csv, row 52](../../docs/sshr_pilot/ecoli4/scores.csv#L52) |
| yeast4 LogReg ec_ebo f1 | 0.3189 | 0.31887581 | MATCH at 4 d.p. | [docs/sshr_pilot/yeast4/scores.csv, row 42](../../docs/sshr_pilot/yeast4/scores.csv#L42) |
| yeast4 LogReg nores f1 | 0.2018 | 0.20181969 | MATCH at 4 d.p. | [docs/sshr_pilot/yeast4/scores.csv, row 2](../../docs/sshr_pilot/yeast4/scores.csv#L2) |
| yeast4 LogReg sshr f1 | 0.2962 | 0.29616801 | MATCH at 4 d.p. | [docs/sshr_pilot/yeast4/scores.csv, row 52](../../docs/sshr_pilot/yeast4/scores.csv#L52) |
| yeast4 LogReg nores fpr | 0.0046 | 0.00459544 | MATCH at 4 d.p. | [docs/sshr_pilot/yeast4/scores.csv, row 6](../../docs/sshr_pilot/yeast4/scores.csv#L6) |
| yeast4 LogReg ec_ebo fpr | 0.0923 | 0.09234913 | MATCH at 4 d.p. | [docs/sshr_pilot/yeast4/scores.csv, row 46](../../docs/sshr_pilot/yeast4/scores.csv#L46) |
| ecoli4 LogReg SSHR delta f1 | -0.0777 | -0.07773474 | MATCH at 4 d.p. | [docs/sshr_pilot/ecoli4/ablation.csv, row 2](../../docs/sshr_pilot/ecoli4/ablation.csv#L2) |
| ecoli4 GNB SSHR delta f1 | -0.0922 | -0.09224684 | MATCH at 4 d.p. | [docs/sshr_pilot/ecoli4/ablation.csv, row 7](../../docs/sshr_pilot/ecoli4/ablation.csv#L7) |
| yeast4 LogReg SSHR delta f1 | -0.0096 | -0.00963384 | MATCH at 4 d.p. | [docs/sshr_pilot/yeast4/ablation.csv, row 2](../../docs/sshr_pilot/yeast4/ablation.csv#L2) |
| yeast4 GNB SSHR delta f1 | -0.0054 | -0.00539530 | MATCH at 4 d.p. | [docs/sshr_pilot/yeast4/ablation.csv, row 7](../../docs/sshr_pilot/yeast4/ablation.csv#L7) |

## Pseudo-label correctness

| Dataset | Classifier | Correct / selected events | Event-weighted precision | Evidence |
| --- | --- | ---: | ---: | --- |
| ecoli4 | LogReg | 170/313 | 54.3131% | [diagnostics.csv, row 7](../sshr_pilot/ecoli4/diagnostics.csv#L7); [gold-label recheck](pseudo_label_audit.csv) |
| ecoli4 | GNB | 170/313 | 54.3131% | [diagnostics.csv, row 13](../sshr_pilot/ecoli4/diagnostics.csv#L13); [gold-label recheck](pseudo_label_audit.csv) |
| yeast4 | LogReg | 96/273 | 35.1648% | [diagnostics.csv, row 7](../sshr_pilot/yeast4/diagnostics.csv#L7); [gold-label recheck](pseudo_label_audit.csv) |
| yeast4 | GNB | 96/273 | 35.1648% | [diagnostics.csv, row 13](../sshr_pilot/yeast4/diagnostics.csv#L13); [gold-label recheck](pseudo_label_audit.csv) |

These fractions describe correctness among selected existing U observations, weighted by selection events. They are not accuracy over all U, not truth labels for SMOTE vectors, and not counts of unique individuals. SSHR repeats the same resampling operations for the two downstream classifiers, so adding classifier copies doubles events rather than independent information. Evidence: [src/sshr_evaluation.py:430–437](../../src/sshr_evaluation.py#L430), [src/sshr.py:103–118](../../src/sshr.py#L103), [audit_verification.json](audit_verification.json).

## Audit boundaries

The 800 saved jobs succeeded; there were no numeric failures/missing fold values in this scope. The independent audit checked 84 named assertion families, with 271,321 row/artifact assertions, not 271,321 experiments or pytest tests. Evidence: [command_evidence.txt:22](command_evidence.txt#L22), [audit_verification.json, assertion_counts](audit_verification.json).

The audit verifies saved values and inspected code; it does not refit models, establish author-code equivalence, compute significance tests or certify unlogged historical decisions. Evidence: [audit_verification.json, limits](audit_verification.json). Generated feature coordinates are not saved, so per-generated-row geometric proxies cannot be independently reconstructed from the saved records; their aggregation/code scope can be checked. Evidence: [audit_verification.json, limits](audit_verification.json).

Only new audit files were written. No tracked file changes, commits or pushes were made under the user's read-only instruction. Evidence: [command_evidence.txt:6–20](command_evidence.txt#L6).
