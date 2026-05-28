from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        print(f"missing: {path}")
        return []
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def main() -> None:
    snapshot = read_rows(Path("data/latest_snapshot.csv"))
    history = read_rows(Path("data/latest_history.csv"))
    dashboard = Path("visualized/sticker_dashboard.html")

    snapshot_ids = {row.get("sticker_id", "") for row in snapshot if row.get("sticker_id")}
    history_ids = {row.get("sticker_id", "") for row in history if row.get("sticker_id")}

    print("Android pipeline check")
    print(f"snapshot rows:     {len(snapshot)}")
    print(f"snapshot ids:      {len(snapshot_ids)}")
    print(f"variant counts:    {dict(Counter(row.get('variant', '') for row in snapshot))}")
    print(f"metadata statuses: {dict(Counter(row.get('metadata_status', '') for row in snapshot))}")
    print(f"history rows:      {len(history)}")
    print(f"history ids:       {len(history_ids)}")
    print(f"missing history:   {len(snapshot_ids - history_ids)}")
    print(f"dashboard exists:  {dashboard.exists()}")


if __name__ == "__main__":
    main()
