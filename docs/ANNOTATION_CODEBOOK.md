# Human Annotation Codebook — WHSI Corpus Validation

**Purpose:** Convert classifier outputs into corpus-specific precision/recall estimates.  
**Sample:** 360 rows in `data/labelled/validation_sample.csv`  
**Required:** Two annotators per row. Document independence in `docs/ANNOTATOR_INDEPENDENCE.md` before submission (κ = 1.00 must be explained). Blindness to classifier labels and to each other is the **intended** protocol; if practice differed, say so factually.

---

## Stratification (fixed)

| Platform | n | Split |
|----------|---|-------|
| YouTube | 120 | 60 classifier-flagged harmful + 60 classifier-flagged safe |
| Reddit | 120 | 60 harmful + 60 safe |
| Telegram | 80 | 40 harmful + 40 safe |
| Twitter (historical) | 40 | 20 harmful + 20 safe — validation corpus only, not live ranking |

---

## Labels

### `human_label_perpetrator` (primary gold label for WTSHI)

| Value | Definition |
|-------|------------|
| **1** | **Perpetrator attack** — speaker directs gendered hostility, degradation, threat, or incitement **at** women or a woman-identified target |
| **0** | Not perpetrator attack (includes victim disclosure, neutral discourse, quoting/reporting without endorsement) |

### `human_label_role` (secondary, for error analysis)

| Role | Definition | Example |
|------|------------|---------|
| `perpetrator_attack` | Active hostility toward women | "Women are stupid and deserve…" |
| `victim_disclosure` | Speaker describes harm **they** suffered | "My ex assaulted me at work…" |
| `neutral_discourse` | Discussion, news, meta-commentary, reclaimed slang without target | "I'm such a bitch sometimes lol" (self-directed) |

### `targets_women` (optional check)

| Value | Definition |
|-------|------------|
| 1 | Comment references or targets women/girls/female-identified persons |
| 0 | No clear women-specific target |

Use **EIGE CVAWG** framing: measure violence **against** women, not all mentions of gender.

---

## Annotation rules

1. Annotate the **comment text only** — not the classifier score.
2. **Sarcasm/reclamation:** Self-directed "bitch" without attacking others → `neutral_discourse`, label 0.
3. **Quoting hate:** Reporting someone else's slur without endorsement → `neutral_discourse`, label 0 unless speaker adds agreement.
4. **Threats:** Physical harm, doxing, KYS directed at a woman → `perpetrator_attack`, label 1.
5. **Ambiguity:** If genuinely unclear after discussion, exclude row from gold set (note in `annotator_notes`).

---

## Dual-annotation workflow

1. **Annotator A** fills `annotator_a_perpetrator` and `annotator_a_role`.
2. **Annotator B** fills `annotator_b_perpetrator` and `annotator_b_role` (same file, different columns).
3. Run adjudication:
   ```bash
   PYTHONPATH=src python src/validate_classifier.py --adjudicate
   ```
4. If Cohen's κ < 0.60 on perpetrator label, revise codebook, re-annotate disputed rows only.
5. Import gold labels:
   ```bash
   PYTHONPATH=src python src/validate_classifier.py --import-annotations data/labelled/validation_sample.csv
   ```

---

## Acceptance criterion (faculty Step 1)

- Per-platform precision, recall, F1 on **perpetrator_attack**
- Inter-annotator κ reported in `outputs/results/classifier_validation.json`
- Every harm-rate claim in thesis references corpus-specific validation table

**Do NOT use `--auto-confirm`** — that copies model labels and produces circular F1=1.0.
