# Platform Se_w / Sp_w / FPR_w (Faculty Condition 1)

From IPW-weighted confusion by band.

| platform | TP_w | FP_w | TN_w | FN_w | Se_w | Sp_w | FPR_w | obs_flag_rate | obs_minus_FPR_pp | RG_status | RG_corrected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YouTube | 495.71 | 3155.71 | 7989.57 | 0.0 | 1.0 | 0.7169 | 0.2831 | 0.3077 | 2.46 | corrected | 0.0343 |
| Reddit | 229.1 | 1154.79 | 3087.12 | 0.0 | 1.0 | 0.7278 | 0.2722 | 0.2894 | 1.72 | corrected | 0.0236 |
| Telegram | 7.0 | 83.42 | 1556.67 | 314.92 | 0.0217 | 0.9491 | 0.0509 | 0.0639 | 1.3 | undefined_youden_Se_plus_Sp_le_1 | nan |
| Twitter | 1465.17 | 857.17 | 67.67 | 26.0 | 0.9826 | 0.0732 | 0.9268 | 0.9433 | 1.65 | n/a_historical | nan |

## Pool-on-pool Rogan–Gladen (Faculty closure 17 Aug — authoritative)

Se/Sp from IPW gold; \(p\) = **sampling-pool** flag rate (A.3), not panel.

| Platform | Pool \(p\) | Se_w | Sp_w | \(\hat\pi=(p+Sp-1)/(Se+Sp-1)\) |
|----------|------------|------|------|-------------------------------|
| YouTube | 0.322 | 1.000 | 0.717 | **~5.4%** |
| Reddit | 0.294 | 1.000 | 0.728 | **~3.0%** |
| Telegram | 0.053 | 0.022 | 0.949 | **undefined** |

Estimates apply to the validated pool (texts > 20 characters); short texts are unvalidated.

**Sensitivity caveat:** YT six gold positives; if Se=0.8 → ≈4.8%; if Se=0.6 → ≈6.5%.

## Telegram RG “undefined” explained


Telegram: TP_w=7.0, FN_w=314.92 → **Se_w = 0.0217**. Sp_w ≈ 0.949.

Youden: Se_w + Sp_w ≈ 0.971 **≤ 1** → Rogan–Gladen denominator (Se+Sp−1) ≤ 0 → **undefined**.

Corpus-weighted, the instrument is effectively **blind** on Telegram (~98% of weighted true harm missed). Unweighted sample recall 0.50 (n=40) **understates** this failure.

## Why the old 5× claim fails

| Platform | Obs flag | FPR_w | Obs−FPR (pp) |
|----------|----------|-------|--------------|
| YouTube | 0.3077 | 0.2831 | ~2.5 |
| Reddit | 0.2894 | 0.2722 | ~1.7 |
| Telegram | 0.0639 | 0.0509 | ~1.3 |

The ~5× density gap is overwhelmingly differential **false-positive behavior**, not demonstrated differential harm.

## Required replacement claim

> Per-platform error profiles differ so sharply (weighted FPR ≈28%/28%/5%; weighted sensitivity ≈1.0/1.0/0.02) that cross-platform flag-density comparisons measure the classifier’s differential behavior, not the platforms. Within the comparable-error pair (YouTube, Reddit), densities are statistically indistinguishable. Telegram cannot be compared.
