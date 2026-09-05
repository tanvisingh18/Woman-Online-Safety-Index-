#!/usr/bin/env python3
"""
Run all faculty-mandated thesis improvements (June 2026 feedback).

Phase 1: validation export, prevalence correction, Detoxify analysis, corpus quarantine, A1
Phase 2: MRI sensitivity audit, Monte Carlo, threshold justification, wave-2 protocol
Phase 3: glossary, viva prep (docs only)

Human annotation (Step 1 labels) must be completed by the team — this script prepares infrastructure.

Run: PYTHONPATH=src python src/run_thesis_improvements.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


def main():
    print("=" * 70)
    print("THESIS IMPROVEMENT PIPELINE — Faculty feedback implementation")
    print("=" * 70)

    steps = [
        ("Step 1: Classifier validation JSON", lambda: __import__("validate_classifier", fromlist=["run_validation"]).run_validation(adjudicate=True)),
        ("Step 3: Detoxify disagreement analysis", lambda: __import__("detoxify_disagreement_analysis", fromlist=["run_analysis"]).run_analysis()),
        ("Step 4+5: Integrated WHSI (live only, no WHSI_corrected)", lambda: __import__("integrated_scoring", fromlist=["run_integrated_pipeline"]).run_integrated_pipeline()),
        ("Step 2: Prevalence correction", lambda: __import__("prevalence_correction", fromlist=["compute_prevalence_corrected"]).compute_prevalence_corrected()),
        ("Step 4: Davidson validation corpus", lambda: __import__("validation_corpus_scoring", fromlist=["score_validation_corpus"]).score_validation_corpus()),
        ("Step 9: MRI sensitivity (PDR/RAS + band audit)", lambda: __import__("mri_sensitivity", fromlist=["run_sensitivity"]).run_sensitivity()),
        ("Step 7: Monte Carlo joint sensitivity", lambda: __import__("monte_carlo_sensitivity", fromlist=["run_monte_carlo"]).run_monte_carlo()),
        ("Step 8: Threshold justification", lambda: __import__("threshold_justification", fromlist=["run_threshold_justification"]).run_threshold_justification()),
        ("Rate reconciliation table", lambda: __import__("export_rate_reconciliation", fromlist=["main"]).main()),
        ("Safety matrix refresh", lambda: __import__("safety_matrix", fromlist=["build_safety_matrix"]).build_safety_matrix()),
    ]

    for name, fn in steps:
        print(f"\n--- {name} ---")
        try:
            fn()
        except Exception as exc:
            print(f"  FAILED: {exc}")

    print("\n" + "=" * 70)
    print("ANNOTATION: use validation_sample.numbers or CSV; pipeline auto-imports .numbers if present.")
    print("  PYTHONPATH=src python src/validate_classifier.py --import-numbers")
    print("=" * 70)


if __name__ == "__main__":
    main()
