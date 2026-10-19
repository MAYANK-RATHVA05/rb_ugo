# Soft-NADOS feasibility pilot

Design SHA-256: `93f66da35a0b5f65e6b20c961c101921b2f1616b9dd9d75407d028aa21d2ab46`. This is a small feasibility screen, not final evidence. It uses three scikit-learn datasets, five stratified folds, 10% training-only symmetric label noise and one logistic-regression classifier. Values are five-fold means.

| dataset | method | f1 | gmean | average_precision | corrupted_parent_exposure |
| --- | --- | --- | --- | --- | --- |
| breast_cancer | NORES | 0.9257 | 0.9365 | 0.9758 |  |
| breast_cancer | WEIGHTED | 0.9315 | 0.9466 | 0.9764 |  |
| breast_cancer | NADOS | 0.9277 | 0.9448 | 0.9722 | 0.1415 |
| breast_cancer | SOFT_NADOS | 0.9211 | 0.9396 | 0.9684 | 0.1394 |
| wine_class_2_vs_rest | NORES | 0.9587 | 0.9771 | 0.9940 |  |
| wine_class_2_vs_rest | WEIGHTED | 0.8937 | 0.9522 | 0.9856 |  |
| wine_class_2_vs_rest | NADOS | 0.8671 | 0.9330 | 0.9414 | 0.2204 |
| wine_class_2_vs_rest | SOFT_NADOS | 0.8592 | 0.9290 | 0.9567 | 0.2127 |
| digits_1_vs_rest_3_to_1 | NORES | 0.8758 | 0.9301 | 0.9219 |  |
| digits_1_vs_rest_3_to_1 | WEIGHTED | 0.7969 | 0.9018 | 0.9116 |  |
| digits_1_vs_rest_3_to_1 | NADOS | 0.8158 | 0.9094 | 0.9108 | 0.1948 |
| digits_1_vs_rest_3_to_1 | SOFT_NADOS | 0.8029 | 0.9021 | 0.9004 | 0.1964 |

## Frozen decision

Status: **pilot_trigger_not_met**. Soft-NADOS minus NADOS: mean dataset F1 -0.0091, F1 wins 0/3, mean AP +0.0003, and mean corrupted-parent exposure -0.0027. The frozen feasibility trigger was not met, so this exact Soft-NADOS formulation should not advance to the 30-dataset experiment.
