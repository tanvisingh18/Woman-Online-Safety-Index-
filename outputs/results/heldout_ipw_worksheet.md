# Held-out IPW worksheet (Faculty Round 3 Step 1)

Created: 2026-08-04T08:59:17.935991+00:00
Sampling seed: 20260712

## Band definitions (from sampling script, NOT appendix illustration)
- low: gendered_harm_proba < 0.35
- mid: 0.35 ≤ proba < 0.55
- high: proba ≥ 0.55

Protocol v2.0 specified confidence-band stratification for IAA but omitted inverse-probability reweighting for corpus metric estimation. Faculty Round 3 identified the omission; this worksheet corrects it.

## Band table
platform band                    band_definition  corpus_N  sample_n  weight_N_over_n
  Reddit  low         gendered_harm_proba < 0.35      2890        21       137.619048
  Reddit  mid 0.35 <= gendered_harm_proba < 0.55       935        21        44.523810
  Reddit high        gendered_harm_proba >= 0.55       646        28        23.071429
Telegram  low         gendered_harm_proba < 0.35      1824        12       152.000000
Telegram  mid 0.35 <= gendered_harm_proba < 0.55       110        12         9.166667
Telegram high        gendered_harm_proba >= 0.55        28        16         1.750000
 Twitter  low         gendered_harm_proba < 0.35        78         6        13.000000
 Twitter  mid 0.35 <= gendered_harm_proba < 0.55        94         6        15.666667
 Twitter high        gendered_harm_proba >= 0.55      2244         8       280.500000
 YouTube  low         gendered_harm_proba < 0.35      7316        21       348.380952
 YouTube  mid 0.35 <= gendered_harm_proba < 0.55      2565        21       122.142857
 YouTube high        gendered_harm_proba >= 0.55      1760        28        62.857143

## Overall metrics
Unweighted: {"TP_w": 28.0, "FP_w": 92.0, "FN_w": 6.0, "TN_w": 74.0, "sensitivity": 0.8235, "specificity": 0.4458, "precision": 0.2333, "recall": 0.8235, "f1": 0.3636, "weighted_flag_rate": 0.6, "sum_weights": 200.0, "n_rows": 200}
Weighted:   {"TP_w": 2196.9762, "FP_w": 5251.0834, "FN_w": 340.9167, "TN_w": 12701.0238, "sensitivity": 0.8657, "specificity": 0.7075, "precision": 0.295, "recall": 0.8657, "f1": 0.44, "weighted_flag_rate": 0.3635, "sum_weights": 20490.0, "n_rows": 200}
Bootstrap CI: {"precision": {"mean": 0.2967, "ci_low": 0.1777, "ci_high": 0.4089}, "recall": {"mean": 0.8681, "ci_low": 0.7086, "ci_high": 0.9889}, "sensitivity": {"mean": 0.8681, "ci_low": 0.7086, "ci_high": 0.9889}, "specificity": {"mean": 0.7082, "ci_low": 0.6628, "ci_high": 0.751}, "f1": {"mean": 0.4393, "ci_low": 0.2926, "ci_high": 0.5627}, "weighted_flag_rate": {"mean": 0.3637, "ci_low": 0.3366, "ci_high": 0.3917}}

Protocol v2 stratified the gold set by classifier-confidence band so that ambiguous mid/high-score cases appear in inter-annotator agreement. That design over-represents high-score rows relative to the corpus, which inflates the apparent false-positive rate. Inverse-probability weights w = N_band/n_band restore each row to the mass of corpus comments it represents. Weighted Se/Sp/precision/F1 are therefore the corpus estimates; unweighted metrics remain valid descriptions of the stratified sample only (and of IAA design), not of platform-wide classifier performance.

## Per-platform sanity (weighted flag ≈ corpus flag)
- Reddit: weighted_flag=0.3095 corpus=0.2943 ok=True Sp_w=0.7278 F1_w=0.2841
- Telegram: weighted_flag=0.0461 corpus=0.0525 ok=True Sp_w=0.9491 F1_w=0.034
- Twitter: weighted_flag=0.9612 corpus=0.9433 ok=True Sp_w=0.0732 F1_w=0.7684
- YouTube: weighted_flag=0.3137 corpus=0.3221 ok=True Sp_w=0.7169 F1_w=0.2391

## Rogan–Gladen (weighted Se/Sp)
- Reddit: {"observed_corpus_flag_rate": 0.2894, "Se_w": 1.0, "Sp_w": 0.7278, "implied_FPR_1_minus_Sp_w": 0.2722, "status": "corrected", "prevalence_corrected_rate": 0.0236}
- Telegram: {"observed_corpus_flag_rate": 0.0639, "Se_w": 0.0217, "Sp_w": 0.9491, "implied_FPR_1_minus_Sp_w": 0.0509, "status": "undefined_youden_Se_plus_Sp_le_1", "prevalence_corrected_rate": null}
- YouTube: {"observed_corpus_flag_rate": 0.3077, "Se_w": 1.0, "Sp_w": 0.7169, "implied_FPR_1_minus_Sp_w": 0.2831, "status": "corrected", "prevalence_corrected_rate": 0.0343}
