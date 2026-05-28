import pandas as pd
from pathlib import Path

SNAPSHOT_PATH = Path("data/latest_snapshot.csv")
SCORES_PATH = Path("data/scores.csv")

if not SNAPSHOT_PATH.exists():
    raise SystemExit("Run collect.py first. data/latest_snapshot.csv not found.")

snap = pd.read_csv(SNAPSHOT_PATH)

# Keep only Holo/Foil
snap = snap[snap["variant"].isin(["Holo", "Foil"])].copy()

# Keep useful columns for manual editing
template = snap[
    [
        "sticker_id",
        "image_formula",
        "name",
        "category",
        "variant",
        "team",
        "price_tokens",
        "usd_price",
        "rank_low_to_high",
        "total_in_group",
        "price_percentile",
        "image_url",
        "item_url",
    ]
].copy()

# Add empty scoring columns
template["visual_score"] = ""
template["craft_score"] = ""
template["demand_score"] = ""
template["color_score"] = ""
template["readability_score"] = ""
template["notes"] = ""

# If scores.csv already exists, preserve your old scores
if SCORES_PATH.exists():
    old = pd.read_csv(SCORES_PATH)

    score_cols = [
        "sticker_id",
        "visual_score",
        "craft_score",
        "demand_score",
        "color_score",
        "readability_score",
        "notes",
    ]

    old = old[[c for c in score_cols if c in old.columns]]

    template = template.drop(
        columns=[
            "visual_score",
            "craft_score",
            "demand_score",
            "color_score",
            "readability_score",
            "notes",
        ]
    ).merge(old, on="sticker_id", how="left")

# Sort useful viewing order
template = template.sort_values(
    ["category", "variant", "price_tokens"],
    ascending=[True, True, True]
)

template.to_csv(SCORES_PATH, index=False, encoding="utf-8-sig")

print(f"Created/updated {SCORES_PATH}")
print("Open it in Google Sheets or Excel and fill the score columns.")