# Review-2 PPT Content Pack  
**Title (locked):** Women Online Safety Index: A Dual-Index Framework for Measuring Platform Harm and Moderation Accountability  

**Team (Reg. No. sorted):** Naina Jose (23BCB0157) · Tanvi Singh (23BCE2155) · Tejashree Elayaraja (23BDS0294)  
**Guide:** Prof. Ilanthenral K P S K  

**Science lock (post-faculty):** IPW F1 ≈ 0.440 · Path A P_w = 0.402 (gate NOT MET) · MRI 82.28 / **69.18** / 11.27 · *n* = **19,765** · **no** 5× · **no** Telegram lit-adj 53.88 · WTSHI = Women-Targeted Speech Harm Index  

**Figures to insert:**  
- `docs/figures/fig2_system_architecture.png`  
- `docs/figures/fig3_dfd_level0.png`  
- `docs/figures/fig4_use_case.png`  
- `docs/figures/fig5_safety_matrix.png`  

Paste these slides into `reference doc/4 Review 2 PPT Template.pptx` after your existing Aim / Abstract / Lit Review / Gap / Objectives / Outcomes slides.  
**Slide 2 must show guide email approval** (per guidelines).

---

## SLIDE — Guide Approval (Slide 2 — mandatory)

**Title:** Guide Approval — Review 2 Components  

| Component | Status | Guide approval |
|-----------|--------|----------------|
| Attendance Sheet | Ready | ☐ Approved by email |
| PowerPoint / Demonstration | Ready | ☐ Approved by email |
| Initial Draft Project Report | Ready (`docs/REVIEW2_PROJECT_REPORT.md`) | ☐ Approved by email |
| Outcome Report | Patent + Journal (Scopus) planned | ☐ Approved by email |

**Guide:** Prof. Ilanthenral K P S K  
**Note:** Paste screenshot / quote of approval email on this slide before the review.

---

## SLIDE — Aim (keep / lightly update)

To design and validate a dual-index computational framework — the **Women Harassment Severity Index (WHSI)** and the **Moderation Responsiveness Index (MRI)** — that measures platform-level women-targeted harm and platform accountability **separately**, enabling a descriptive Safety Matrix across YouTube, Reddit, and Telegram rather than isolated comment-level toxicity flags.

---

## SLIDE — Abstract (REPLACE old July abstract)

Online harassment of women cannot be captured by comment-level toxicity alone. This project proposes **WHSI** (harm severity) + **MRI** (moderation accountability), with **WTSHI** separating perpetrator attack from victim disclosure, on **19,765** YouTube/Reddit/Telegram comments.

**Method:** EDOS-trained TF-IDF+LR → WTSHI roles → Mamdani fuzzy WHSI (T/Th/F/N) → MRI from transparency evidence → Safety Matrix.

**Validation:** Protocol v2 *n*=200; κ=0.588 (perp) / 0.449 (3-class); IPW P/R/Sp/F1 = **0.295 / 0.866 / 0.708 / 0.440**; Path A best P_w=**0.402** (gate not met).

**Results:** MRI **82.28 / 69.18 / 11.27**; instrument not cross-platform comparable; YT≈Reddit; Telegram not density-comparable; **no** 5× / **no** lit-adj 53.88 / **no** Unsafe–Safe ranking.

**Contribution:** interpretable dual-index + claim-discipline methodology — not a definitive platform league table.

---

## SLIDE — Research Gap (keep structure; update point 4)

1. No platform-level aggregation of women-targeted harm.  
2. Harm and accountability conflated.  
3. Victim disclosure misclassified as harm.  
4. Low-visibility platforms can look “safe” under scrape-only methods — **we document structural unobservability; we do not invent a lit-adj Telegram score in primary results.**

---

## SLIDE — Objectives (update #4 and #6)

1. Gendered-harm classifier + **WTSHI** role layer on human gold.  
2. **WHSI** fuzzy score (T, Th, F, N).  
3. **MRI** accountability score (transparency + standalone probes).  
4. Document visibility limits (esp. Telegram) without primary lit-adj point estimates.  
5. **Safety Matrix** WHSI × MRI (descriptive regions).  
6. Validate via IAA, IPW, Rogan–Gladen where defined, Monte Carlo weights, Path A attempt.

---

## SLIDE — Framework / Architecture / Block Diagram *(NEW)*

**Title:** System Architecture  

Insert **Fig. 2** (`fig2_system_architecture.png`).

**Bullet narration (speak):**
1. Scrape → women-relevant panel (*n*=19,765)  
2. TF-IDF + LR classifier (frozen thresholds)  
3. **WTSHI** role filter  
4. Features T/Th/F/N → Mamdani fuzzy → **WHSI_raw = 0.70×WTSHI + 0.30×fuzzy**  
5. **MRI** from transparency (+ empirical probe standalone)  
6. Validation gate (v2 + IPW) → **Safety Matrix**

**Dotted ASCII (backup if image fails):**

```
Data → Preprocess → Classifier → WTSHI
              ↓
         Features T/Th/F/N
          ↙           ↘
     Fuzzy WHSI      MRI
          ↘           ↙
         Validation → Safety Matrix
```

---

## SLIDE — Functional Requirements *(NEW)*

**Title:** Functional Requirements  

- Ingest YT / Reddit / Telegram (+ EDOS/Davidson)  
- Build women-relevant panel  
- Classify gendered harm (frozen rule)  
- Assign WTSHI roles (perpetrator / victim / neutral)  
- Extract T, Th, F, N  
- Compute WHSI via Mamdani fuzzy engine  
- Compute MRI (removal, speed, consistency, PDR/RAS)  
- Build Safety Matrix  
- Run Protocol v2 dual annotation + IPW evaluation  
- Export faculty/result tables  

**Non-functional (one line):** reproducible configs, interpretable rules, claim discipline, privacy on public data.

---

## SLIDE — Modules *(NEW)*

**Title:** Implemented Modules (~90% technical complete)

| Module | Function | Status |
|--------|----------|--------|
| Data panel | Master women-relevant CSV | Done |
| Classifier | EDOS TF-IDF+LR + frozen thresholds | Done |
| WTSHI | Role separation | Done |
| Fuzzy engine | Mamdani WHSI | Done |
| MRI engine | Accountability score | Done |
| Safety Matrix | WHSI × MRI plot/join | Done |
| Validation | Protocol v2, IPW, Path A | Done |
| Hotspot audit | FP quarantine | Done |
| Docs / Journal / Patent | Packaging & outcomes | In progress |

Remaining for Final Review: full report polish, Scopus journal, patent draft.

---

## SLIDE — Experiments & Results (1/2) — Validation *(NEW)*

**Title:** Experiments — Classifier Validation (PRIMARY)

| Metric | Unweighted sample | **IPW corpus** |
|--------|-------------------|----------------|
| Precision | 0.233 | **0.295** |
| Recall | 0.824 | **0.866** |
| Specificity | 0.446 | **0.708** |
| F1 | 0.364 | **0.440** |

- Held-out *n*=200 · κ = **0.588** · Path A best P_w = **0.402** (**gate not met**)  
- June F1≈0.93 is **in-sample only** — not independent validity  

**Speak:** “We report the honest corpus numbers, not the optimistic June numbers.”

---

## SLIDE — Experiments & Results (2/2) — Indices *(NEW)*

**Title:** Results — WHSI × MRI Safety Matrix  

| Platform | Flag density | WHSI_raw | MRI |
|----------|--------------|----------|-----|
| YouTube | 30.8% | 28.41 | **82.28** |
| Reddit | 28.9% | 24.95 | **69.18** |
| Telegram | 6.4%‡ | 12.69 | **11.27** |

‡ Telegram density **not comparable** (Se_w ≈ 0.022).  

Insert **Fig. 5** Safety Matrix.  

**Claims on this slide:**
- Instrument **not** cross-platform comparable  
- YT ≈ Reddit (not distinguishable)  
- **No** 5× / **no** Telegram lit-adj 53.88  

---

## SLIDE — Conclusion *(NEW)*

**Title:** Conclusion  

- Dual-index **WHSI × MRI** with **WTSHI** is implemented and validated on 19,765 comments.  
- Validation arc: June optimism → held-out → IPW → Path A → claim revision.  
- IPW F1 **0.440**; Path A P_w **0.402** (gate not met) → no absolute danger ranking.  
- MRI scores **82.28 / 69.18 / 11.27** = accountability evidence, not overall “safer platform” verdict.  
- Principal contribution: interpretable dual-index + claim discipline — not a league table.  
- Technical pipeline complete; overall project ~90% (docs / journal / patent remain).

---

## SLIDE — Outcomes (keep)

- Patent (planned)  
- Journal — Scopus (planned)  
- SDG 5 / 16 alignment (keep, but **remove** old Telegram 53.88 / inverted-ranking sentences if still present)  
- TRL 4 — technology validated in lab / offline on real scrapes (not live moderation ops)

---

## SLIDE — Updated References *(NEW — replace old PPT refs)*

**Title:** Key References (sample)

[1] Zadeh (1965) — Fuzzy sets  
[2] Mamdani & Assilian (1975) — Fuzzy controller  
[3] Davidson et al. (2017) — Hate vs offensive  
[4] Fortuna & Nunes (2018) — Hate speech survey  
[5] Kirk et al. (2023) — EDOS SemEval-2023 Task 10  
[6] Guest et al. (2021) — Reddit misogyny dataset  
[7] Liu et al. (2019) — Fuzzy hate-type classification  
[8] Arora et al. (2023) — Platforms need vs research gap  
[9] Trujillo et al. (2025) — DSA Transparency Database audit  
[10] Schneider & Rizoiu (2023) — Moderation effectiveness (PNAS)  
[11] Pamungkas et al. (2020) — Cross-domain misogyny  
[12] La Gatta et al. (2023) — Cross-platform YouTube→Twitter harm  

Full IEEE list: `Literature_Review_IEEE_Survey.docx` (≥50).  
**Do not cite** unverified “Sharma et al. 2023 ~200 fuzzy studies.”

---

## SLIDE — Demo Checklist (for viva)

1. Show master panel row count 19,765  
2. Show one classified comment → WTSHI role → fuzzy inputs  
3. Show platform WHSI_raw + MRI table  
4. Show Safety Matrix figure  
5. Show IPW metrics table (not June 0.93 as primary)  
6. State claim boundaries in one sentence  

---

## What to DELETE / FIX from old “final review ppt 2”

| Old text | Action |
|----------|--------|
| F1≈0.93, κ=1.00 as main result | Demote; show IPW instead |
| Lit-adj invert / Telegram 53.88 | Remove from primary slides |
| MRI Reddit 60.64 | → **69.18** |
| WTSHI = “Trustworthy…” | → **Women-Targeted Speech Harm Index** |
| Sharma et al. 2023 fuzzy survey | Remove |
| “Telegram emerges as highest-risk” | Remove |

---

*End of PPT content pack.*
