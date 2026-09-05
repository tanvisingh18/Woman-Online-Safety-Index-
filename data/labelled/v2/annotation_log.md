# WOSI v2 Annotation Log

- Created (UTC): 2026-07-12T14:54:41.698465+00:00
- Sampling seed: `20260712`
- n rows: 200
- Frozen threshold config SHA-256: `f501b708268b7e75f9eca9b94ae581cc59a0357a0b1dfa73c4b2263890510b04`
- Blank A SHA-256: `95f5be514b2653c401a4a940d01d6994958d05dc6f5808f7c67ae5a955bcf098`
- Blank B SHA-256: `41b8104af0cc8f7e596b47f3a56b9ed764692fe5a0b54ea47fc8ad9e6673b0ae`
- Master SHA-256: `9a9b0d8795eb89aade381cde71c7beb160a8ccb77ccdb7388e9f4aaac3fff1cb`

## Stratum counts

platform  _band
Reddit    high     28
          low      21
          mid      21
Telegram  high     16
          low      12
          mid      12
Twitter   high      8
          low       6
          mid       6
YouTube   high     28
          low      21
          mid      21

## Session logs (Faculty Condition 3 — fill + sign this week)

### Annotator A — Tanvi
- Start: _______________
- End: _______________
- Classifier columns visible? NO
- Interruptions:
- Independence file signed? `independence_A_SIGNED.txt` — [ ] yes  Date: ____

### Annotator B — Tejashree
- Start: _______________
- End: _______________
- Classifier columns visible? NO
- Interruptions:
- Independence file signed? `independence_B_Tejashree_SIGNED.txt` — [ ] yes  Date: ____

### Annotator B — Naina
- Start: _______________
- End: _______________
- Classifier columns visible? NO
- Interruptions:
- Independence file signed? `independence_B_Naina_SIGNED.txt` — [ ] yes  Date: ____

## Post-submission (Sampling Lead) — completed 2026-08-03

- Raw A SHA-256: `c4aca23d6111a6c6c2aa63169bc818954956f1ab550b2acaecdc936092cea35b`
- Raw B SHA-256: `56a9b4bbd8941ef91d483b5c2616329589a6b784d21b4dad0a0de5bd1cae1fa0` (after victim codebook fix 1→0 on 17 rows)
- Cohen kappa (perpetrator binary, after B codebook fix): **0.5881**
- Cohen kappa (harm_role 3-class): **0.4487**
- n_disputed before adjudication: **23** (11.5%)
- Kappa band: within healthy 0.55–0.85 (not >0.95 STOP)
- Adjudication: codebook adjudication logged in `adjudication_table.csv` (n=23; hard=8)
- Gold file: `gold_heldout.csv`
- Held-out PRIMARY metrics: see `outputs/results/heldout_vs_insample_metrics.json`
- June 360 set NOT used for primary held-out metrics

### Independence statements (Condition 3 — THIS WEEK, not “before viva”)
- Named files ready to sign: `independence_A_SIGNED.txt` (Tanvi), `independence_B_Tejashree_SIGNED.txt`, `independence_B_Naina_SIGNED.txt`
- Sampling lead / adjudicator (logged): Ilanthenral
- After signatures: commit and paste hash into `docs/FACULTY_SIGNOFF_CONDITIONS_COMPLIANCE_AUGUST_2026.md`

### Note on adjudication
Disputed rows were resolved by applying Protocol v2 codebook to comment text with one-line reasons
(Sampling Lead / adjudicator role). Both annotators' raw labels are preserved in the adjudication table.
