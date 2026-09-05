"""
Schedule and run 48-hour empirical MRI two-wave checks.

Wave 1 is frozen now; wave 2 should run >=48h later (same-day = invalid 0% removal).

Usage:
  # Initialize schedule + save wave-1 snapshots (run once):
  PYTHONPATH=src python src/schedule_empirical_mri.py --init

  # Run wave 2 when due (or use cron — see outputs/results/empirical_mri_schedule.json):
  PYTHONPATH=src python src/schedule_empirical_mri.py --run-wave2

  # Force wave 2 immediately (thesis emergency only — label as invalid timing):
  PYTHONPATH=src python src/schedule_empirical_mri.py --run-wave2 --force

Cron example (macOS/Linux, every 6 hours check if due):
  0 */6 * * * cd /path/to/women_safety_index && source ../dataset/.venv/bin/activate && PYTHONPATH=src python src/schedule_empirical_mri.py --run-wave2 >> logs/empirical_mri_cron.log 2>&1
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone

SCHEDULE_PATH = "outputs/results/empirical_mri_schedule.json"
HOURS_REQUIRED = 48.0


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def load_schedule() -> dict:
    if os.path.exists(SCHEDULE_PATH):
        with open(SCHEDULE_PATH) as f:
            return json.load(f)
    return {}


def save_schedule(payload: dict) -> None:
    os.makedirs(os.path.dirname(SCHEDULE_PATH), exist_ok=True)
    with open(SCHEDULE_PATH, "w") as f:
        json.dump(payload, f, indent=2)


def init_schedule(hours_between: float = HOURS_REQUIRED) -> dict:
    sys.path.insert(0, os.path.dirname(__file__))
    from empirical_mri_waves import save_wave1, save_youtube_wave1

    now = _utcnow()
    due = now + timedelta(hours=hours_between)

    reddit_ok = youtube_ok = False
    reddit_err = youtube_err = ""

    try:
        save_wave1()
        reddit_ok = True
    except Exception as exc:
        reddit_err = str(exc)

    try:
        save_youtube_wave1()
        youtube_ok = True
    except Exception as exc:
        youtube_err = str(exc)

    payload = {
        "created_at": now.isoformat(),
        "wave1_saved_at": now.isoformat(),
        "wave2_earliest_at": due.isoformat(),
        "hours_between_waves": hours_between,
        "status": "wave1_frozen",
        "reddit_wave1": reddit_ok,
        "youtube_wave1": youtube_ok,
        "reddit_error": reddit_err or None,
        "youtube_error": youtube_err or None,
        "wave2_completed_at": None,
        "cron_command": (
            "0 */6 * * * cd $(pwd) && source ../dataset/.venv/bin/activate && "
            "PYTHONPATH=src python src/schedule_empirical_mri.py --run-wave2 >> logs/empirical_mri_cron.log 2>&1"
        ),
        "manual_wave2_command": "PYTHONPATH=src python src/schedule_empirical_mri.py --run-wave2",
    }
    save_schedule(payload)
    print(f"\n[EMPIRICAL MRI SCHEDULE]")
    print(f"  Wave 1 frozen at     : {payload['wave1_saved_at']}")
    print(f"  Earliest wave 2 run  : {payload['wave2_earliest_at']}")
    print(f"  Schedule file        : {SCHEDULE_PATH}")
    if reddit_err:
        print(f"  Reddit wave1 warning : {reddit_err}")
    if youtube_err:
        print(f"  YouTube wave1 warning: {youtube_err}")
    return payload


def run_wave2_if_due(force: bool = False, hours_between: float = HOURS_REQUIRED) -> dict:
    sys.path.insert(0, os.path.dirname(__file__))
    from empirical_mri_waves import (
        measure_two_wave_mri,
        measure_youtube_two_wave,
        scrape_wave2,
        scrape_youtube_wave2,
        _write_empirical_all,
        measure_reddit_removal_flags,
    )

    sched = load_schedule()
    if not sched:
        print("No schedule found — run with --init first.")
        return {"error": "no_schedule"}

    due_at = _parse_iso(sched["wave2_earliest_at"])
    now = _utcnow()
    if not force and now < due_at:
        remaining = due_at - now
        print(f"Wave 2 not due yet. Wait {remaining} (due {due_at.isoformat()})")
        print(f"Or run: PYTHONPATH=src python src/schedule_empirical_mri.py --run-wave2 --force")
        return {"status": "not_due", "due_at": due_at.isoformat(), "remaining_hours": remaining.total_seconds() / 3600}

    if force and now < due_at:
        print("WARNING: Running wave 2 before 48h — same-day removal rates are not valid empirical MRI.")

    reddit_tw = {}
    try:
        scrape_wave2(per_sub=180)
        reddit_tw = measure_two_wave_mri(hours_between=hours_between)
    except Exception as exc:
        reddit_tw = {"error": str(exc), "platform": "Reddit"}

    reddit_flags = measure_reddit_removal_flags()
    reddit = reddit_tw if reddit_tw.get("empirical_removal_rate", 0) > reddit_flags.get("empirical_removal_rate", 0) else reddit_flags

    youtube = {}
    if os.environ.get("SKIP_YOUTUBE_EMPIRICAL_MRI", "").lower() in ("1", "true", "yes"):
        youtube = {"skipped": True, "reason": "SKIP_YOUTUBE_EMPIRICAL_MRI=1"}
    else:
        try:
            scrape_youtube_wave2(max_videos=25)
            youtube = measure_youtube_two_wave(hours_between=hours_between)
        except Exception as exc:
            youtube = {"error": str(exc), "platform": "YouTube"}

    _write_empirical_all(reddit, youtube if "error" not in youtube else None)

    sched["status"] = "wave2_complete" if not force else "wave2_complete_forced_early"
    sched["wave2_completed_at"] = now.isoformat()
    sched["wave2_forced_early"] = bool(force and now < due_at)
    sched["results"] = {"Reddit": reddit, "YouTube": youtube}
    save_schedule(sched)

    print(f"\n[EMPIRICAL MRI WAVE 2 COMPLETE]")
    if reddit.get("empirical_removal_rate") is not None:
        print(f"  Reddit removal rate : {reddit['empirical_removal_rate']:.2%}")
    if youtube.get("empirical_removal_rate") is not None:
        print(f"  YouTube removal rate: {youtube['empirical_removal_rate']:.2%}")
    return sched


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Schedule 48h empirical MRI waves")
    parser.add_argument("--init", action="store_true", help="Freeze wave 1 and write schedule")
    parser.add_argument("--run-wave2", action="store_true", help="Run wave 2 if due")
    parser.add_argument("--force", action="store_true", help="Run wave 2 even if <48h (invalid timing)")
    parser.add_argument("--hours", type=float, default=HOURS_REQUIRED)
    args = parser.parse_args()

    if args.init:
        init_schedule(hours_between=args.hours)
    elif args.run_wave2:
        run_wave2_if_due(force=args.force, hours_between=args.hours)
    else:
        sched = load_schedule()
        if sched:
            print(json.dumps(sched, indent=2))
        else:
            print("No schedule. Run: PYTHONPATH=src python src/schedule_empirical_mri.py --init")
