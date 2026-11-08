"""Bounded 12-hour collection session, called once per scheduled wake-up."""

import json
from pathlib import Path

import pandas as pd
from collect_tomtom_live import collect_once

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "data/collected/hourly_session.json"


def main():
    now = pd.Timestamp.now(tz="UTC")
    if not STATE.exists():
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(
            json.dumps(
                {"start": now.isoformat(), "deadline": (now + pd.Timedelta(hours=12)).isoformat()},
                indent=2,
            )
        )
    state = json.loads(STATE.read_text())
    if now >= pd.Timestamp(state["deadline"]):
        print("SESSION_EXPIRED: stop scheduling; never backfill missed observations.")
        return
    collect_once()
    print(f"Session deadline: {state['deadline']}")


if __name__ == "__main__":
    main()
