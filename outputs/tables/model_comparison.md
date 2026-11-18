| model | variant | cv_macro_f1 | cv_accuracy | cv_precision_macro | cv_recall_macro | cv_log_loss | cv_obs_top1 | student_macro_f1 | student_log_loss | student_obs_top1 | student_obs_top1_recallable | fit_time_s | infer_us_per_cand |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decision_tree | untuned | 0.84 ± 0.0456 | 0.9078 | 0.7997 | 0.9201 | 0.3053 | 0.7593 | 0.8786 | 0.1999 | 0.8532 | 0.8545 | 17.5478 | 0.1914 |
| hist_gradient_boosting | untuned | 0.8587 ± 0.0294 | 0.9220 | 0.8186 | 0.9277 | 0.2023 | 0.7611 | 0.8859 | 0.1599 | 0.8576 | 0.8589 | 16.6591 | 3.2434 |
| knn | untuned | 0.8102 ± 0.0875 | 0.9215 | 0.8450 | 0.7966 | 1.1707 | 0.6836 | 0.8748 | 0.6674 | 0.8079 | 0.8091 | 0.0424 | 19.0030 |
| logistic_regression | untuned | 0.7875 ± 0.0462 | 0.8685 | 0.7502 | 0.8886 | 0.3104 | 0.7521 | 0.8279 | 0.2427 | 0.8444 | 0.8456 | 4.5341 | 0.2041 |
| random_forest | untuned | 0.8428 ± 0.0536 | 0.9100 | 0.8048 | 0.9169 | 0.2226 | 0.7565 | 0.8837 | 0.1844 | 0.8434 | 0.8447 | 20.0114 | 0.9829 |