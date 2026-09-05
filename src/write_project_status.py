"""Write PROJECT_STATUS.json — faculty improvement checklist + blockers."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pandas as pd


def _annotation_status(val_path: str = "data/labelled/validation_sample.csv") -> dict:
    if not os.path.exists(val_path):
        return {"status": "missing_file", "filled": 0, "total": 0}

    val = pd.read_csv(val_path, low_memory=False)
    total = len(val)

    def _filled(col: str) -> int:
        if col not in val.columns:
            return 0
        s = val[col]
        return int((s.notna() & s.isin([0, 1, 0.0, 1.0, "0", "1", "0.0", "1.0"])).sum())

    a = _filled("annotator_a_perpetrator")
    b = _filled("annotator_b_perpetrator")
    gold = _filled("human_label_perpetrator")

    if a >= total and b >= total:
        status = "dual_complete"
    elif a >= total or b >= total or gold >= total:
        status = "single_or_partial"
    elif a + b + gold > 0:
        status = "in_progress"
    else:
        status = "not_started"

    return {
        "status": status,
        "total_rows": total,
        "annotator_a_filled": a,
        "annotator_b_filled": b,
        "human_label_filled": gold,
        "file": val_path,
        "next_command": (
            "PYTHONPATH=src python src/validate_classifier.py --adjudicate && "
            "PYTHONPATH=src python src/validate_classifier.py --import-annotations data/labelled/validation_sample.csv && "
            "PYTHONPATH=src python src/run_thesis_improvements.py"
        ),
    }


def main() -> None:
    os.makedirs("outputs/results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    ann = _annotation_status()
    sched = {}
    if os.path.exists("outputs/results/empirical_mri_schedule.json"):
        with open("outputs/results/empirical_mri_schedule.json") as f:
            sched = json.load(f)

    faculty_checklist = {
        "step_1_human_annotation": {
            "done": ann["status"] == "dual_complete",
            "annotation": ann,
        },
        "step_2_prevalence_correction": {
            "done": os.path.exists("outputs/results/prevalence_corrected_rates.csv"),
            "interim_until_step_1": ann["status"] != "dual_complete",
        },
        "step_3_detoxify_analysis": os.path.exists("outputs/results/detoxify_disagreement_analysis.json"),
        "step_4_davidson_quarantined": (
            os.path.exists("outputs/results/validation_corpus_davidson2017.json")
            and os.path.exists("data/processed/whsi_scores.csv")
            and "Twitter" not in pd.read_csv("data/processed/whsi_scores.csv")["platform"].values
        ),
        "step_5_assumption_A1_no_whsi_corrected": (
            os.path.exists("outputs/results/whsi_literature_sources.json")
            and "WHSI_corrected" not in pd.read_csv("data/processed/whsi_scores.csv").columns
        ),
        "step_6_empirical_mri_wave2": {
            "done": sched.get("wave2_completed_at") is not None,
            "protocol": "docs/EMPIRICAL_MRI_WAVE2_PROTOCOL.md",
            "earliest_at": sched.get("wave2_earliest_at"),
            "command": sched.get("manual_wave2_command"),
        },
        "step_7_monte_carlo": os.path.exists("outputs/results/monte_carlo_sensitivity.json"),
        "step_8_threshold_justification": os.path.exists("outputs/results/threshold_justification.json"),
        "step_9_mri_band_audit": os.path.exists("outputs/results/mri_band_audit.csv"),
        "step_10_glossary": os.path.exists("docs/THESIS_GLOSSARY.md"),
        "step_11_viva_prep": os.path.exists("docs/VIVA_PREPARATION.md"),
    }

    pending = []
    if not faculty_checklist["step_1_human_annotation"]["done"]:
        pending.append("human_annotation_dual_360_rows")
    if faculty_checklist["step_2_prevalence_correction"]["interim_until_step_1"]:
        pending.append("prevalence_correction_needs_corpus_validation")
    if not faculty_checklist["step_6_empirical_mri_wave2"]["done"]:
        pending.append("empirical_mri_wave2")

    optional_accuracy = []
    if not os.path.exists("data/processed/master_with_influence.parquet"):
        optional_accuracy.append("activate_influence_layer")
    tg_channels = "data/scraped/telegram_public_channels.txt"
    if os.path.exists(tg_channels) and open(tg_channels).read().strip().startswith("#"):
        optional_accuracy.append("telegram_public_channel_scrape")

    status = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "faculty_improvement_checklist": faculty_checklist,
        "pending_required": pending,
        "optional_accuracy_improvements": optional_accuracy,
        "annotation": ann,
    }

    if os.path.exists("data/processed/whsi_scores.csv"):
        whsi = pd.read_csv("data/processed/whsi_scores.csv")
        live = whsi[whsi.get("is_historical", 0) != 1] if "is_historical" in whsi.columns else whsi
        status["current_whsi_raw"] = live.set_index("platform")["WHSI_raw"].round(2).to_dict()
        if "WHSI_literature_adjusted" in live.columns:
            status["current_whsi_literature_adjusted"] = live.set_index("platform")[
                "WHSI_literature_adjusted"
            ].round(2).to_dict()
        if "validation_status" in live.columns:
            status["prevalence_validation_status"] = live.set_index("platform")["validation_status"].to_dict()

    if os.path.exists("data/processed/mri_scores.csv"):
        mri = pd.read_csv("data/processed/mri_scores.csv")
        status["current_mri"] = mri.set_index("platform")["MRI_score"].round(2).to_dict()

    if os.path.exists("outputs/results/classifier_validation.json"):
        with open("outputs/results/classifier_validation.json") as f:
            cv = json.load(f)
        status["corpus_validation"] = cv.get("corpus_validation", cv.get("annotation_agreement"))

    with open("outputs/results/PROJECT_STATUS.json", "w") as f:
        json.dump(status, f, indent=2)

    print(f"Wrote outputs/results/PROJECT_STATUS.json")
    print(f"Pending required: {pending or 'none'}")
    print(f"Annotation: {ann['status']} ({ann['annotator_a_filled']}/{ann['total_rows']} A, {ann['annotator_b_filled']}/{ann['total_rows']} B)")


if __name__ == "__main__":
    main()
