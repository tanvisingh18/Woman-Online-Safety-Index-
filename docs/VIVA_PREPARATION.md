# Viva Preparation — Examiner Q&A (Faculty July Second Round)

Write answers in your own voice before defense. Re-run pipeline after Protocol v2.0 annotation
to replace `[HELD-OUT]` placeholders. κ = 1.00 on the June 360 is **not** a strength without
process evidence — lead with Protocol v2.0.

---

## Round-2 mandatory questions (Steps 1–6 / 11)

### Q1: "Your annotators agreed on all 360 rows. How?"

**Answer:** The June κ = 1.00 (zero disputes) is reported as **in-sample annotation agreement**
on the stratified 360, not as independent process-proven IAA for defense. Faculty required
Protocol v2.0: a **fresh** n=200 sample (`data/labelled/v2/`), classifier-stripped blanks,
signed independence statements, hash-logged exports (`PRE_ANNOTATION_MANIFEST.json`), and
adjudication logs. If v2 κ exceeds 0.95 with zero disputes again, we treat that as a **process
red flag** and audit independence before claiming reliability. Primary IAA for the thesis is
the Protocol v2 figure with attached logs — not the June 1.00 alone.

### Q2: "You tuned thresholds on the gold set and report F1 on the same set — why isn't 0.93 circular?"

**Answer:** It is circular if presented as primary validity. We demote F1 ≈ 0.93 to
**in-sample / tuning-set performance** (`outputs/results/heldout_vs_insample_metrics.json`).
Thresholds are **frozen** (`configs/perpetrator_thresholds_frozen.json`, SHA-256 in the
manifest). Primary accuracy is **held-out Protocol v2 gold** under those frozen thresholds —
`[HELD-OUT P/R/F1 after annotation]`. Rogan–Gladen Se/Sp will be refreshed from the held-out
gold, not from the tuning set.

### Q3: "Your harm rates doubled in a month. What changed — the platforms or your method?"

**Answer:** The method. Holding the legacy rule (0.52 / 0.45 / 0.38) fixed on the **current**
women-relevant live scrape recovers June rates within ~0.1 pp; the production rule
(0.40 / 0.34 / 0.28) adds ~11–12 pp on YouTube/Reddit and ~3 pp on Telegram
(`docs/CHANGES_FROM_JUNE_RATES.md`). Rates are always **conditional** on that women-relevant
live-scrape denominator — not platform-wide unconditional prevalence.

### Q4: "53.88 for Telegram is built on a rate you say is censored noise. Defend it."

**Answer:** We do **not**. Round 3 **retired** 53.88 from every results table — it was a
censored-noise numerator divided by an assumed 0.10 scrapable share. The surviving prose claim:
our scrape structurally cannot observe most Telegram harm (public-scrapable share ≈ 10%, AI
Forensics 2026); external evidence indicates the invisible portion is substantial; **our
instrument cannot currently quantify it.** EHER sensitivity bands live in an appendix for future
use with a validated classifier — not as a headline number.

### Q5: "Where do Gab's numbers come from?"

**Answer:** Gab has **no public transparency report**. Removal/PDR/RAS are **expert-coded
ordinal estimates** with an explicit coding rationale in
`docs/MRI_GAB_FACEBOOK_PROVENANCE.md` / `mri_input_derivations.json`. Gab is **secondary MRI
panel only** — never in the primary YouTube/Reddit/Telegram safety matrix. Facebook reactive
inputs cite Meta Q4 2023; PDR/RAS use the same expert rubric as YouTube with published Meta
anchors.

---

## Earlier bank (still useful)

### Q6: "Your precision is 0.46. Why should I believe any of your harm rates?"

**Answer:** EDOS holdout precision (~0.44) is a **domain-shift** check on general sexism, not
corpus truth. Primary validity is corpus human validation — June in-sample F1 ≈ 0.93 demoted;
Protocol v2 held-out is primary. We always report classifier-flagged rates **and**
prevalence-corrected rates where Rogan–Gladen is valid (YouTube/Reddit); Telegram correction
censored. Human–human κ ≠ model–Detoxify κ ≈ 0.29.

### Q7: "Isn't the Telegram conclusion just the AI Forensics paper wearing your formula?"

**Answer:** No independent-convergence claim. VIS/MBR/EHER share the AI Forensics visibility
estimate under A1. Directional claim only; see `docs/TELEGRAM_LIT_ADJUSTED_FRAMING.md`.

### Q8: "Why 0.70/0.30? Why 0.52? Why 720 hours?"

**Answer:** Weights in `docs/WEIGHT_RATIONALE_APPENDIX.md`. Monte Carlo rank stability 100%.
Production thresholds are 0.40 / 0.34+threat (0.52 was legacy EDOS proxy). Horizon 720h =
30-day saturation; Telegram MRI is horizon-sensitive.

### Q9: "Your Twitter score is 88 — is Twitter really the most dangerous platform?"

**Answer:** No. Davidson 2017 is **validation-only**, quarantined from live rankings.

### Q10: "0 of 400 posts removed — did anyone actually report them?"

**Answer:** Wave-2 0% measures **unreported / proactive survival**, not reactive MRI. We
**unblended** it from Reddit MRI (matrix MRI = 69.18 transparency+PDR/RAS). Standalone probe
vs coded PDR 38%: `outputs/results/mri_proactive_probe_standalone.json`.

---

## Round-3 questions (August 2026)

### Q11: "Your F1 fell from 0.93 to 0.36 in one month — why trust anything?"

**Answer:** The fall *is* the finding. June 0.93 was in-sample on the tuning set. Protocol v2
held-out unweighted F1=0.36 exposed optimistic bias. Round 3 IPW then shows the stratified
sample also *deflated* specificity (0.45→0.71 weighted). We trust the **process**: pre-registered
independence, adjudication log, and the correction for our own sampling design — not any single
headline F1. Absolute rate claims stay suspended or tightly caveated until Path A improves precision.

### Q12: "Why is Breaking Down Patriarchy a harassment hotspot?"

**Answer:** It should not be. Hand-audit (30 flagged comments, seed 20260804) found ~3% genuine
perpetrator attacks [Wilson ~0.6–17%]. Independent Annotator B spot-checked 10/30 (39/40 agree;
one adjudication). The ~87% flag rate was topic/lexical false positives — the TwoX failure mode
at channel level. All four hotspot channels were removed from deliverables and cited as a
documented FP mode of the current classifier.

### Q13: "Is YouTube more dangerous than Reddit?"

**Answer:** Not distinguishable at current instrument precision. Flag rates 30.8% vs 28.9% sit
inside noise given weighted precisions ~0.14–0.17. We do **not** claim either is “5× Telegram”:
weighted FPR ≈28%/28%/5% and Se_w ≈1.0/1.0/0.02 mean cross-platform density comparisons measure
classifier behavior, not platforms; Telegram Se_w≈0.02 → cannot be compared (RG undefined).
Monte Carlo rank stability only perturbs formula weights — not a measurement-error proof.

### Q13b: "Why did you drop the 5× Telegram claim?"

**Answer:** Faculty Condition 1. Decompose: YouTube 30.8% − 28.3% FPR ≈ 2.5pp; Telegram 6.4% −
5.1% ≈ 1.3pp. The gap is mostly differential false positives. Corpus-weighted Telegram recall
is 0.022 (not the sample 0.50) — the instrument is effectively blind there.
