# Path A — Instrument fix results

**Primary candidate:** `june_selected`
**IPW-weighted precision:** **0.4021** (target 0.5–0.7: NOT MET)
**IPW-weighted F1:** 0.5176 · Recall 0.7261 · Spec 0.8473

## Protocol
- Tune: June n=360 only (rescored)
- Eval: Protocol v2 + IPW (stratified bootstrap)
- Never tune on v2

## IPW comparison

| Candidate | P_w | R_w | Sp_w | F1_w |
|-----------|-----|-----|------|------|
| Frozen 0.40/0.34/0.28 | 0.2953 | 0.8657 | 0.708 | 0.4404 |
| June-selected (max P, R≥0.5) | 0.4021 | 0.7261 | 0.8473 | 0.5176 |
| Taxonomy topic-veto | 0.4021 | 0.7261 | 0.8473 | 0.5176 |
| DistilBERT | 0.3077 | 0.9595 | 0.6947 | 0.4659 |

**Absolute claims restored?** No — instrument still below Path A gate.

Full JSON: `outputs/results/path_a_instrument_fix.json`
