# WOSI Annotation Protocol v2.0
**Mandatory for Faculty Step 1 (July 2026 second-round feedback)**  
**Status:** Binding — follow without deviation.

---

## 0. Purpose

Replace the June 360-row gold set for **evaluation**. The June set remains frozen as the
**in-sample / threshold-tuning** set only. A new, procedurally independent gold set
(n = 200) is the sole source of **primary held-out** classifier metrics and Rogan–Gladen Se/Sp.

---

## 1. Roles (no dual-hatting)

| Role | Duties | Must NOT |
|------|--------|----------|
| **Sampling Lead** | Draw sample, strip classifier columns, compute hashes, open log, distribute files | Annotate any row |
| **Annotator A** | Label in private session; submit raw file + session log | See Annotator B’s labels or classifier columns |
| **Annotator B** | Same as A, separate session | See Annotator A’s labels or classifier columns |
| **Adjudicator** | Resolve disagreements **after** both submissions | Annotate in the primary round |

If the team has only two people: Sampling Lead ≠ Annotator A; Annotator B may be the second person. Adjudicator may be Sampling Lead **only after** both raw files are hashed and committed.

---

## 2. Sample design (n = 200)

Stratify by **platform** and **classifier-confidence band** (using scores known only to Sampling Lead):

| Platform | n | High conf. harm (proba ≥ 0.55) | Mid (0.35–0.55) | Low / safe (< 0.35) |
|----------|---|-------------------------------|-----------------|---------------------|
| YouTube | 70 | 28 | 21 | 21 |
| Reddit | 70 | 28 | 21 | 21 |
| Telegram | 40 | 16 | 12 | 12 |
| Twitter (hist.) | 20 | 8 | 6 | 6 |
| **Total** | **200** | | | |

- Sampling seed: fixed integer recorded in the log (default `20260712`).
- Exclude all `comment_id` / text hashes already in `data/labelled/validation_sample.csv` (June set).
- Draw from `dataset_split == live_scrape` (+ Twitter hist. for the 20 rows only).

---

## 3. Files the Sampling Lead produces (before annotation)

1. `data/labelled/v2/annotation_sample_master.csv` — full row with classifier columns (Sampling Lead only; **not** given to annotators).
2. `data/labelled/v2/annotator_A_blank.csv` — columns: `row_id`, `comment_text`, `platform`, `annotator_a_perpetrator`, `annotator_a_role`, `annotator_notes`.
3. `data/labelled/v2/annotator_B_blank.csv` — same with `annotator_b_*` columns.
4. `data/labelled/v2/PRE_ANNOTATION_MANIFEST.json` — seed, stratum counts, SHA-256 of blank files and master, git commit if available, timestamp.
5. `data/labelled/v2/annotation_log.md` — protocol §7 template, started before any labelling.

**Classifier columns MUST be absent from blank files** (`gendered_harm_proba`, `harm_role`, `classifier_*`, `toxicity_score`, `threat_score`, etc.).

---

## 4. Labelling rules

Use codebook: `docs/ANNOTATION_CODEBOOK.md`.

| Column | Values |
|--------|--------|
| `*_perpetrator` | `0` or `1` only |
| `*_role` | `perpetrator_attack` / `victim_disclosure` / `neutral_discourse` |

- Separate sessions; no joint labelling.
- No discussion of cases until **both** raw files are submitted and hashed.
- Per-session log: start/end time, interruptions, whether classifier columns were visible (must be **no**).

---

## 5. Independence statements (signed, appendix)

Each annotator signs:

> I, [NAME], labelled the WOSI v2 annotation file assigned to me without access to the other annotator’s labels and without access to classifier score columns. I did not discuss individual cases with the other annotator before submission of my raw file.

Store as `data/labelled/v2/independence_A.txt` and `independence_B.txt`. Copy verbatim into the thesis appendix.

---

## 6. Submission & adjudication

1. Annotators submit raw CSVs separately → Sampling Lead hashes them into the log.
2. Merge on `row_id`; compute Cohen’s κ on perpetrator labels.
3. **If κ > 0.95:** STOP. Bring raw files to faculty before any downstream scoring.
4. **If κ < 0.50:** Revise codebook; re-run — legitimate construct ambiguity finding.
5. **If 0.50 ≤ κ ≤ 0.95:** Adjudicate disagreements into `data/labelled/v2/adjudication_table.csv`:
   `row_id, label_A, label_B, final_label, reason`
6. Gold file: `data/labelled/v2/gold_heldout.csv` with `human_label_perpetrator`.

Expected healthy outcome: disagreement on ~10–25% of rows; κ ≈ 0.55–0.85.

---

## 7. Annotation log contents (required evidence)

- Sampling seed and stratum table
- SHA-256 of blank A/B files **before** annotation began
- SHA-256 of submitted raw A/B files
- SHA-256 of frozen threshold config (`configs/perpetrator_thresholds_frozen.json`)
- Session logs A and B
- κ and n_disputed
- Link to adjudication table
- Statement: June 360-row set is **not** used for primary held-out metrics

---

## 8. Commands

```bash
# Sampling Lead — create sample + hashes (does NOT annotate)
PYTHONPATH=src python src/annotation_protocol_v2.py --export-sample

# After both raw files submitted:
PYTHONPATH=src python src/annotation_protocol_v2.py --adjudicate \
  --file-a data/labelled/v2/annotator_A_raw.csv \
  --file-b data/labelled/v2/annotator_B_raw.csv

# Held-out evaluation of FROZEN thresholds:
PYTHONPATH=src python src/heldout_threshold_eval.py
```

---

## 9. Acceptance criteria (faculty)

- [ ] Completed annotation log (§7) with pre-annotation hashes
- [ ] Both annotators’ raw label files, submitted separately
- [ ] Adjudication table
- [ ] Signed independence statements in thesis appendix
- [ ] Primary P/R/F1 from held-out gold; July F1≈0.93 labelled **in-sample only**
