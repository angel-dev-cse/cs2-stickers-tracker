from __future__ import annotations

import csv
from pathlib import Path
import json
import math
import pandas as pd
import numpy as np

DATA_DIR = Path("data")
ANALYZE_DIR = Path("analyze")
OUT_DIR = Path("visualized")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VERDICT_ORDER = {
    "CORE BUY CANDIDATE": 0,
    "SMALL BUY": 1,
    "CHEAP HISTORY PUNT": 2,
    "VISUAL CHECK NOW": 3,
    "SCORE FIRST": 4,
    "WAIT FOR DROP": 5,
    "DO NOT CHASE": 6,
    "FLOOD RISK": 7,
    "SCORE/WAIT": 8,
    "IGNORE": 9,
}

VERDICT_COLORS = {
    "CORE BUY CANDIDATE": "#22c55e",
    "SMALL BUY": "#84cc16",
    "CHEAP HISTORY PUNT": "#facc15",
    "VISUAL CHECK NOW": "#38bdf8",
    "SCORE FIRST": "#a78bfa",
    "WAIT FOR DROP": "#fb923c",
    "DO NOT CHASE": "#ef4444",
    "FLOOD RISK": "#f43f5e",
    "SCORE/WAIT": "#cbd5e1",
    "IGNORE": "#64748b",
}


def candidate_files(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for root in [
        ANALYZE_DIR,
        DATA_DIR,
        Path("."),
        Path("/mnt/data/analyze"),
        Path("/mnt/data/data"),
        Path("/mnt/data"),
    ]:
        if not root.exists():
            continue
        for pattern in patterns:
            files.extend(root.glob(pattern))
    unique = {p.resolve(): p for p in files if p.is_file()}
    return sorted(unique.values(), key=lambda p: p.stat().st_mtime, reverse=True)


def read_csv_best(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def read_csv_loose(path: Path) -> pd.DataFrame:
    """Read CSVs written by old/new collectors without dropping the whole file on ragged rows."""
    try:
        return read_csv_best(path)
    except pd.errors.ParserError:
        rows: list[dict[str, object]] = []
        with open(path, newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            for row in reader:
                row.pop(None, None)
                rows.append(row)
        return pd.DataFrame(rows)


def clean_sticker_name(value: str) -> str:
    name = str(value or "")
    name = name.replace("Sticker | ", "")
    name = name.replace(" | Cologne 2026", "")
    return name


def to_num(series: pd.Series, default=np.nan) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def to_datetime_loose(series: pd.Series) -> pd.Series:
    try:
        parsed = pd.to_datetime(series, errors="coerce", format="mixed", utc=True)
    except TypeError:
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
    try:
        return parsed.dt.tz_convert(None)
    except Exception:
        return parsed


def safe_float(value, default=None):
    try:
        if pd.isna(value):
            return default
        f = float(value)
        if math.isfinite(f):
            return f
        return default
    except Exception:
        return default


def safe_bool(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def short_text(value: str, limit: int = 76) -> str:
    text = str(value or "").strip()
    text = " | ".join([part.strip() for part in text.split("|") if part.strip()])
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def load_analysis() -> pd.DataFrame:
    decision_candidates = [
        ANALYZE_DIR / "decision_board.csv",
        ANALYZE_DIR / "latest_analysis_clean.csv",
        ANALYZE_DIR / "buy_watchlist_clean.csv",
        DATA_DIR / "decision_board.csv",
        DATA_DIR / "latest_analysis_clean.csv",
        DATA_DIR / "buy_watchlist_clean.csv",
    ]
    decision_path = next((path for path in decision_candidates if path.exists()), None)
    if decision_path is None:
        files = candidate_files(["decision_board.csv", "latest_analysis_clean.csv", "buy_watchlist_clean.csv", "latest_analysis*.csv"])
        if not files:
            raise SystemExit("No decision board found. Run analyze.py first.")
        decision_path = files[0]
    df = read_csv_loose(decision_path)

    # Analyzer output is authoritative. Merge debug data only for missing visual/history fields.
    debug_path = next((path for path in [ANALYZE_DIR / "debug_metrics.csv", DATA_DIR / "debug_metrics.csv"] if path.exists()), None)
    if debug_path is not None:
        dbg = read_csv_loose(debug_path)
        merge_key = "sticker_id" if "sticker_id" in df.columns and "sticker_id" in dbg.columns else "sticker"
        if merge_key in df.columns and merge_key in dbg.columns:
            wanted = [
                merge_key, "image_url", "item_url", "recent_return_pct", "hist_last", "hist_min", "hist_max",
                "snapshot_prev_price",
                "hist_points", "last_tooltip_time_raw", "price_slope_recent", "popularity_slope_recent",
                "latest_popularity", "positive_popularity_sum", "absolute_popularity_pressure",
                "discount_from_high_pct", "upside_to_high_pct", "position_in_range",
                "sticker_type", "catalog_type", "player_name", "team_name", "market_hash_name",
                "steam_market_url", "metadata_status",
            ]
            keep = [c for c in wanted if c in dbg.columns and (c == merge_key or c not in df.columns)]
            if len(keep) > 1:
                dbg = dbg[keep].drop_duplicates(subset=[merge_key], keep="first")
                df = df.merge(dbg, on=merge_key, how="left")

    if "sticker" not in df.columns:
        if "name" in df.columns:
            df["sticker"] = df["name"].map(clean_sticker_name)
        else:
            df["sticker"] = ""

    if "priority_score" not in df.columns:
        d = to_num(df.get("decision_score", pd.Series([0] * len(df))), 0)
        h = to_num(df.get("history_score", pd.Series([0] * len(df))), 0)
        t = to_num(df.get("trend_score", pd.Series([0] * len(df))), 0)
        df["priority_score"] = ((0.50 * d + 0.35 * h + 0.15 * t) * 100).round(1)

    if "priority_rank" not in df.columns:
        df["priority_rank"] = np.nan

    if "priority_tier" not in df.columns:
        df["priority_tier"] = pd.cut(
            to_num(df["priority_score"], 0),
            bins=[-1, 52, 62, 72, 101],
            labels=["P4", "P3", "P2", "P1"],
        ).astype(str)

    if "quick_reason" not in df.columns:
        df["quick_reason"] = df.get("reason", "")
    if "risk_note" not in df.columns:
        df["risk_note"] = ""
    if "action_note" not in df.columns:
        df["action_note"] = df.get("suggested_size", "")

    numeric_cols = [
        "priority_rank", "priority_score", "price_tokens", "usd_price", "quality_score",
        "history_score", "decision_score", "discovery_score", "trend_score", "entry_score",
        "flood_risk_score", "discount_from_high_pct", "upside_to_high_pct",
        "position_in_range", "crowding_percentile", "absolute_popularity_pressure",
        "positive_popularity_sum", "hist_last", "hist_min", "hist_max", "hist_points", "recent_return_pct",
        "value_edge_score", "expected_return_pct", "expected_return_score", "robust_reference_price",
        "robust_peak_price", "discount_from_robust_peak_pct", "downside_to_floor_pct",
        "downside_risk_score", "demand_momentum_score", "demand_price_divergence_score",
        "falling_demand_penalty", "prediction_confidence", "score_confidence", "manual_score_count",
        "history_coverage_score", "entry_change_score", "snapshot_prev_price", "snapshot_price_change_pct",
        "snapshot_price_velocity_pct_per_day", "snapshot_price_slope", "snapshot_price_acceleration",
        "rank_change", "rank_percentile_change", "rank_improvement_score",
        "price_drop_opportunity_score", "latest_relative_demand_share",
        "relative_demand_share_change_pct", "relative_demand_share_slope_recent",
        "demand_share_acceleration", "team_exposure_score", "portfolio_group_count",
        "portfolio_variant_count", "launch_gap_pct", "early_avg_gap_pct",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "sticker_id" in df.columns:
        df = df.drop_duplicates(subset=["sticker_id"], keep="first").copy()
    else:
        df = df.drop_duplicates(subset=["sticker"], keep="first").copy()

    df["verdict_rank"] = df.get("verdict", pd.Series("", index=df.index)).map(VERDICT_ORDER).fillna(99)
    if df["priority_rank"].notna().any():
        df["priority_rank"] = df["priority_rank"].fillna(9999)
        df = df.sort_values(["priority_rank", "verdict_rank", "priority_score"], ascending=[True, True, False]).copy()
    else:
        df = df.sort_values(["verdict_rank", "priority_score"], ascending=[True, False]).copy()
        df["priority_rank"] = range(1, len(df) + 1)

    return df


def load_history() -> pd.DataFrame:
    paths = [DATA_DIR / "history_points.csv", DATA_DIR / "latest_history.csv"]
    history_dir = DATA_DIR / "history"
    if history_dir.exists():
        paths.extend(sorted(history_dir.glob("history_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)[:12])
    frames: list[pd.DataFrame] = []
    for path in paths:
        if path.exists():
            try:
                frames.append(read_csv_loose(path))
            except Exception:
                pass
    if not frames:
        files = candidate_files(["history_points*.csv", "latest_history*.csv", "history_*.csv"])
        for path in files:
            try:
                frames.append(read_csv_loose(path))
            except Exception:
                pass
    if not frames:
        return pd.DataFrame()

    hist = pd.concat(frames, ignore_index=True)
    if "token_cost" not in hist.columns and "token_cost_est" in hist.columns:
        hist["token_cost"] = hist["token_cost_est"]
    hist["token_cost"] = pd.to_numeric(hist.get("token_cost"), errors="coerce")
    hist["popularity"] = pd.to_numeric(hist.get("popularity"), errors="coerce")
    hist["point_index"] = pd.to_numeric(hist.get("point_index"), errors="coerce")
    if "fetched_at" in hist.columns:
        hist["point_time"] = to_datetime_loose(hist["fetched_at"])
    else:
        hist["point_time"] = pd.NaT
    if "tooltip_time_raw" in hist.columns:
        tooltip_time = to_datetime_loose(hist["tooltip_time_raw"])
        hist["point_time"] = hist["point_time"].fillna(tooltip_time)
    if "history_scrape_timestamp" in hist.columns:
        scrape_time = to_datetime_loose(hist["history_scrape_timestamp"])
        hist["point_time"] = hist["point_time"].fillna(scrape_time)
    hist = hist.dropna(subset=["sticker_id", "token_cost"])
    hist = hist[hist["token_cost"] > 0].copy()

    dedupe_cols = [c for c in ["sticker_id", "history_range", "point_time", "point_index", "token_cost"] if c in hist.columns]
    hist = hist.drop_duplicates(subset=dedupe_cols, keep="last")
    hist = hist.sort_values(["sticker_id", "point_time", "point_index"], na_position="last")
    return hist


def build_history_series(analysis: pd.DataFrame, hist: pd.DataFrame) -> dict[str, list[dict]]:
    if hist.empty or "sticker_id" not in hist.columns:
        return {}
    wanted_ids = set(analysis["sticker_id"].dropna().astype(str)) if "sticker_id" in analysis.columns else set()
    series: dict[str, list[dict]] = {}
    for sid, g in hist.groupby("sticker_id"):
        sid = str(sid)
        if wanted_ids and sid not in wanted_ids:
            continue
        g = g.sort_values(["point_time", "point_index"], na_position="last")
        points: list[dict] = []
        for i, (_, p) in enumerate(g.iterrows(), start=1):
            token = safe_float(p.get("token_cost"), None)
            if token is None:
                continue
            pop = safe_float(p.get("popularity"), None)
            usd = safe_float(p.get("usd_price"), None)
            point_time = p.get("point_time", "")
            if pd.isna(point_time):
                point_time = p.get("tooltip_time_raw", "")
            points.append({
                "i": i,
                "price": token,
                "usd": usd,
                "popularity": pop,
                "time": "" if pd.isna(point_time) else str(point_time),
            })
        if points:
            first = points[0]["price"]
            if first > 0:
                for point in points:
                    point["norm"] = round((point["price"] / first) * 100, 3)
            series[sid] = points
    return series


def row_to_record(row: pd.Series) -> dict:
    def val(key, default=""):
        value = row.get(key, default)
        if pd.isna(value):
            return default
        return value

    name = str(val("sticker", ""))
    item_url = str(val("item_url", ""))
    variant = str(val("variant", "")).strip()
    sticker_type = str(val("sticker_type", "")).strip()
    catalog_type = str(val("catalog_type", "")).strip()
    display_type = sticker_type or catalog_type or str(val("category", "")).strip()
    team = str(val("team", val("team_name", ""))).strip()
    if not team:
        team = str(val("team_name", "")).strip()
    price_tokens = safe_float(val("price_tokens"), 0) or 0
    hist_min = safe_float(val("hist_min"), None)
    hist_max = safe_float(val("hist_max"), None)
    hist_last = safe_float(val("hist_last"), None)
    snapshot_prev_price = safe_float(val("snapshot_prev_price"), None)
    current_low = bool(hist_min is not None and hist_min > 0 and price_tokens <= hist_min + 0.5)
    low_gap_pct = max(0, ((price_tokens - hist_min) / hist_min * 100)) if hist_min and hist_min > 0 else None

    return {
        "priority_rank": int(safe_float(val("priority_rank"), 9999) or 9999),
        "priority_score": safe_float(val("priority_score"), 0) or 0,
        "priority_tier": str(val("priority_tier", "")),
        "verdict": str(val("verdict", "")),
        "sticker": name,
        "category": str(val("category", "")),
        "variant": variant,
        "sticker_type": sticker_type,
        "catalog_type": catalog_type,
        "display_type": display_type,
        "player_name": str(val("player_name", "")),
        "team_name": str(val("team_name", "")),
        "team": team,
        "price_tokens": price_tokens,
        "usd_price": safe_float(val("usd_price"), 0) or 0,
        "recent_return_pct": safe_float(val("recent_return_pct"), None),
        "hist_min": hist_min,
        "hist_max": hist_max,
        "hist_last": hist_last,
        "snapshot_prev_price": snapshot_prev_price,
        "current_low": current_low,
        "low_gap_pct": low_gap_pct,
        "hist_points": safe_float(val("hist_points"), None),
        "history_span_hours": safe_float(val("history_span_hours"), None),
        "snapshot_points": safe_float(val("snapshot_points"), None),
        "suggested_size": str(val("suggested_size", "")),
        "entry_tier": str(val("entry_tier", "")),
        "flood_risk": str(val("flood_risk", "")),
        "quality_score": safe_float(val("quality_score"), None),
        "history_score": safe_float(val("history_score"), None),
        "decision_score": safe_float(val("decision_score"), None),
        "discovery_score": safe_float(val("discovery_score"), None),
        "value_edge_score": safe_float(val("value_edge_score"), None),
        "expected_return_pct": safe_float(val("expected_return_pct"), None),
        "expected_return_score": safe_float(val("expected_return_score"), None),
        "robust_reference_price": safe_float(val("robust_reference_price"), None),
        "robust_peak_price": safe_float(val("robust_peak_price"), None),
        "discount_from_robust_peak_pct": safe_float(val("discount_from_robust_peak_pct"), None),
        "downside_to_floor_pct": safe_float(val("downside_to_floor_pct"), None),
        "downside_risk_score": safe_float(val("downside_risk_score"), None),
        "demand_momentum_score": safe_float(val("demand_momentum_score"), None),
        "demand_price_divergence_score": safe_float(val("demand_price_divergence_score"), None),
        "falling_demand_penalty": safe_float(val("falling_demand_penalty"), None),
        "prediction_confidence": safe_float(val("prediction_confidence"), None),
        "score_confidence": safe_float(val("score_confidence"), None),
        "manual_score_count": safe_float(val("manual_score_count"), None),
        "history_coverage_score": safe_float(val("history_coverage_score"), None),
        "entry_change_score": safe_float(val("entry_change_score"), None),
        "trend_score": safe_float(val("trend_score"), None),
        "discount_from_high_pct": safe_float(val("discount_from_high_pct"), None),
        "upside_to_high_pct": safe_float(val("upside_to_high_pct"), None),
        "position_in_range": safe_float(val("position_in_range"), None),
        "crowding_percentile": safe_float(val("crowding_percentile"), None),
        "flood_risk_score": safe_float(val("flood_risk_score"), None),
        "snapshot_price_change_pct": safe_float(val("snapshot_price_change_pct"), None),
        "snapshot_price_velocity_pct_per_day": safe_float(val("snapshot_price_velocity_pct_per_day"), None),
        "snapshot_price_slope": safe_float(val("snapshot_price_slope"), None),
        "snapshot_price_acceleration": safe_float(val("snapshot_price_acceleration"), None),
        "rank_change": safe_float(val("rank_change"), None),
        "rank_percentile_change": safe_float(val("rank_percentile_change"), None),
        "rank_improvement_score": safe_float(val("rank_improvement_score"), None),
        "price_drop_opportunity_score": safe_float(val("price_drop_opportunity_score"), None),
        "latest_relative_demand_share": safe_float(val("latest_relative_demand_share"), None),
        "relative_demand_share_change_pct": safe_float(val("relative_demand_share_change_pct"), None),
        "relative_demand_share_slope_recent": safe_float(val("relative_demand_share_slope_recent"), None),
        "demand_share_acceleration": safe_float(val("demand_share_acceleration"), None),
        "team_exposure_score": safe_float(val("team_exposure_score"), None),
        "portfolio_group": str(val("portfolio_group", "")),
        "trend_signal": str(val("trend_signal", "")),
        "quick_reason": short_text(str(val("quick_reason", val("reason", ""))), 240),
        "risk_note": short_text(str(val("risk_note", "")), 180),
        "action_note": short_text(str(val("action_note", val("suggested_size", ""))), 220),
        "notes": short_text(str(val("notes", "")), 220),
        "scored": safe_bool(val("scored", False)),
        "image_url": str(val("image_url", "")),
        "item_url": item_url,
        "market_hash_name": str(val("market_hash_name", "")),
        "metadata_status": str(val("metadata_status", "")),
        "sticker_id": str(val("sticker_id", "")),
    }


def write_priority_csv(df: pd.DataFrame) -> None:
    cols = [
        "priority_rank", "priority_score", "priority_tier", "verdict", "sticker", "category", "variant",
        "sticker_type", "catalog_type", "player_name", "team", "team_name",
        "price_tokens", "usd_price", "suggested_size", "entry_tier", "flood_risk",
        "hist_last", "hist_min", "hist_max", "hist_points", "history_span_hours", "snapshot_points",
        "quality_score", "history_score", "decision_score", "trend_score", "value_edge_score",
        "expected_return_pct", "demand_momentum_score", "demand_price_divergence_score",
        "prediction_confidence", "score_confidence", "quick_reason", "risk_note", "action_note",
        "item_url", "image_url", "market_hash_name", "metadata_status",
    ]
    cols = [c for c in cols if c in df.columns]
    df[cols].to_csv(OUT_DIR / "priority_board_ui.csv", index=False, encoding="utf-8-sig")


def build_html_legacy(records: list[dict], series: dict[str, list[dict]]) -> str:
    data_json = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    series_json = json.dumps(series, ensure_ascii=False).replace("</", "<\\/")

    template = r"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
<title>CS2 Sticker Decision Dashboard</title>
<style>
  :root {
    --bg:#0b0f17;
    --surface:#101722;
    --surface-2:#151f2e;
    --surface-3:#1b2838;
    --line:#2a3648;
    --line-soft:#202b3a;
    --text:#edf2f7;
    --muted:#a8b3c2;
    --faint:#708092;
    --accent:#4f8cff;
    --accent-2:#22c55e;
    --danger:#ef4444;
    --warn:#facc15;
    --blue:#38bdf8;
    --shadow: 0 14px 32px rgba(0,0,0,.22);
    --radius:8px;
  }
  * { box-sizing:border-box; }
  body {
    margin:0;
    background:var(--bg);
    color:var(--text);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size:13px;
  }
  a { color:inherit; text-decoration:none; }
  .app { padding:16px; max-width:1800px; margin:0 auto; }
  .hero {
    display:flex; justify-content:space-between; gap:18px; align-items:flex-start;
    margin-bottom:14px; padding:18px 20px;
    border:1px solid rgba(168,179,194,.18); border-radius:var(--radius);
    background:linear-gradient(145deg, rgba(16,23,34,.98), rgba(21,31,46,.92));
    box-shadow:var(--shadow);
  }
  h1 { margin:0 0 6px; font-size:24px; letter-spacing:0; }
  .sub { color:var(--muted); max-width:920px; line-height:1.5; }
  .stats { display:flex; gap:10px; flex-wrap:wrap; justify-content:flex-end; }
  .stat { min-width:112px; padding:10px 12px; border:1px solid var(--line); border-radius:8px; background:rgba(21,31,46,.68); }
  .stat b { display:block; font-size:18px; }
  .stat span { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em; }
  .toolbar {
    position:sticky; top:0; z-index:30;
    padding:10px; margin-bottom:12px;
    border:1px solid rgba(168,179,194,.18); border-radius:8px;
    background:rgba(11,15,23,.94); backdrop-filter: blur(14px);
    display:grid; grid-template-columns: 1.35fr repeat(6, minmax(130px, .55fr)); gap:10px;
    box-shadow:0 10px 24px rgba(0,0,0,.22);
  }
  input, select, button {
    border:1px solid var(--line); background:#0f1724; color:var(--text); border-radius:8px;
    padding:9px 11px; outline:none; font-size:13px;
  }
  input:focus, select:focus { border-color:var(--accent); box-shadow:0 0 0 3px rgba(79,140,255,.18); }
  button { cursor:pointer; font-weight:700; }
  button:hover { border-color:#4b5563; background:#111827; }
  .grid { display:grid; grid-template-columns: 1fr; gap:18px; }
  .card { border:1px solid rgba(168,179,194,.16); border-radius:var(--radius); background:rgba(16,23,34,.92); box-shadow:var(--shadow); overflow:hidden; }
  .card-head { display:flex; justify-content:space-between; gap:12px; align-items:center; padding:13px 16px; border-bottom:1px solid var(--line-soft); }
  .card-title { font-size:15px; font-weight:800; }
  .hint { color:var(--muted); font-size:12px; }
  .table-wrap { max-height:78vh; overflow-y:auto; overflow-x:hidden; }
  table { width:100%; border-collapse:separate; border-spacing:0; table-layout:fixed; }
  col.rank-col { width:64px; }
  col.sticker-col { width:21%; }
  col.price-col { width:8%; }
  col.decision-col { width:13%; }
  col.market-col { width:19%; }
  col.scores-col { width:10%; }
  col.notes-col { width:25%; }
  thead th {
    position:sticky; top:0; z-index:11; background:#151f2e; color:#d2dbe7;
    font-size:10px; text-transform:uppercase; letter-spacing:.07em; text-align:left;
    padding:8px 9px; border-bottom:1px solid var(--line); white-space:nowrap;
  }
  tbody td { border-bottom:1px solid rgba(148,163,184,.11); padding:14px 12px; vertical-align:top; }
  tbody tr { background:rgba(16,23,34,.5); transition:.12s; }
  tbody tr:hover { background:rgba(25,37,52,.82); }
  .sticky-rank { position:sticky; left:0; z-index:10; background:#151f2e; width:64px; min-width:64px; box-shadow:1px 0 0 var(--line-soft); }
  tbody .sticky-rank { background:#101722; }
  .sticky-sticker { position:sticky; left:64px; z-index:10; background:#151f2e; box-shadow:1px 0 0 var(--line-soft); }
  tbody .sticky-sticker { background:#101722; }
  th.sortable { cursor:pointer; }
  th.sortable:hover { color:white; }
  .rank { font-weight:850; font-size:16px; color:#fff; }
  .tier { display:inline-flex; align-items:center; justify-content:center; min-width:30px; height:22px; padding:0 8px; border-radius:999px; font-weight:850; font-size:10px; background:#d8e6ff; color:#0b1220; }
  .sticker-cell { display:grid; grid-template-columns:156px minmax(0,1fr); align-items:start; gap:12px; min-width:0; }
  .thumb { width:156px; height:156px; object-fit:contain; border-radius:8px; background:linear-gradient(145deg, rgba(255,255,255,.08), rgba(16,23,34,.95)); border:1px solid rgba(168,179,194,.22); }
  .name-wrap { min-width:0; }
  .name { font-size:15px; font-weight:850; color:#fff; line-height:1.25; display:inline-flex; align-items:center; gap:7px; }
  .name:hover { color:#a5b4fc; text-decoration:underline; }
  .meta { margin-top:6px; color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.06em; white-space:normal; line-height:1.35; }
  .verdict { display:inline-flex; align-items:center; gap:6px; padding:5px 8px; border-radius:999px; font-weight:850; color:#020617; font-size:10px; white-space:normal; line-height:1.2; max-width:100%; }
  .price { font-weight:850; font-size:18px; color:var(--muted); font-variant-numeric:tabular-nums; }
  .usd { color:#fff; font-size:40px; font-weight: bold; }
  .metric { font-variant-numeric: tabular-nums; font-weight:750; }
  .pos { color:#4ade80; } .neg { color:#fb7185; } .flat { color:#94a3b8; }
  .chip { display:inline-flex; align-items:center; padding:4px 7px; border-radius:999px; border:1px solid var(--line); background:rgba(15,23,42,.45); font-size:11px; white-space:nowrap; }
  .spark { width:100%; max-width:360px; height:96px; display:block; margin-top:8px; }
  .spark path.line { fill:none; stroke-width:4; stroke-linecap:round; stroke-linejoin:round; }
  .spark path.area { opacity:.18; }
  .spark .dot { stroke:#0b1020; stroke-width:2; }
  .cell-stack { display:grid; gap:8px; min-width:0; }
  .detail-line { display:grid; grid-template-columns:76px minmax(0, 1fr); align-items:baseline; gap:10px; color:var(--muted); font-size:12px; min-width:0; }
  .detail-line b { color:var(--text); font-size:13px; font-weight:800; white-space:normal; text-align:left; }
  .detail-line span { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .market-cell .detail-line { grid-template-columns:76px minmax(0, 1fr); }
  .score-list { display:grid; gap:5px; }
  .notes-block { display:grid; gap:10px; font-size:14px; line-height:1.5; }
  .note-label { display:block; color:var(--muted); font-size:10px; text-transform:uppercase; letter-spacing:.07em; margin-bottom:3px; }
  .reason-small, .action-small { white-space:normal; overflow:visible; display:block; font-size:14px; line-height:1.5; }
  .reason-small { color:#dbeafe; }
  .action-small { color:#f8dfa5; }
  .btns { display:flex; gap:6px; flex-wrap:wrap; margin-top:8px; }
  .mini-btn { padding:6px 8px; border-radius:7px; font-size:11px; background:#0f1724; border:1px solid var(--line); white-space:nowrap; }
  .mini-btn.primary { border-color:rgba(79,140,255,.58); color:#b9d1ff; }
  .mini-btn.steam { border-color:rgba(34,197,94,.52); color:#86efac; }
  .section-grid { display:grid; grid-template-columns: 1fr 1fr; gap:18px; }
  .chartbox { min-height:680px; padding:16px; }
  .chart-controls { display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin-bottom:12px; }
  .plot-scroll { height:620px; overflow:auto; border:1px solid var(--line); border-radius:8px; background:#080d14; }
  svg.main-plot { display:block; min-width:1500px; min-height:820px; }
  .axis text { fill:#94a3b8; font-size:13px; }
  .gridline { stroke:rgba(148,163,184,.14); stroke-width:1; }
  .point-label { fill:#dbeafe; font-size:13px; paint-order:stroke; stroke:#050816; stroke-width:4px; stroke-linejoin:round; }
  .plot-title { font-size:13px; fill:#e5e7eb; font-weight:900; }
  .legend { display:flex; gap:10px; flex-wrap:wrap; color:var(--muted); font-size:12px; }
  .legend-dot { width:10px; height:10px; display:inline-block; border-radius:99px; margin-right:5px; vertical-align:-1px; }
  .empty { padding:40px; color:var(--muted); text-align:center; }
  .footer-note { color:var(--muted); font-size:12px; line-height:1.55; padding:14px 18px; border-top:1px solid var(--line-soft); }
  @media(max-width:1200px) {
    .toolbar { grid-template-columns:1fr 1fr; }
    .section-grid { grid-template-columns:1fr; }
    .hero { flex-direction:column; }
    .sticky-rank, .sticky-sticker { position:static; box-shadow:none; }
    table { min-width:980px; }
    .table-wrap { overflow-x:auto; }
  }
</style>
</head>
<body>
<div class=\"app\">
  <section class=\"hero\">
    <div>
      <h1>CS2 Sticker Decision Dashboard</h1>
      <div class=\"sub\">Priority-first view: start at Rank #1, inspect the image + mini trend, then open CS2Tokens or the sticker's Steam page. Unscored items are discovery targets, not automatic buys.</div>
    </div>
    <div class=\"stats\">
      <div class=\"stat\"><span>Visible</span><b id=\"visibleCount\">0</b></div>
      <div class=\"stat\"><span>Total</span><b id=\"totalCount\">0</b></div>
      <div class=\"stat\"><span>Top verdict</span><b id=\"topVerdict\">—</b></div>
    </div>
  </section>

  <section class=\"toolbar\">
    <input id=\"search\" placeholder=\"Search sticker, team, verdict, reason…\" />
    <select id=\"verdictFilter\"><option value=\"\">All verdicts</option></select>
    <select id=\"categoryFilter\"><option value=\"\">All categories</option></select>
    <select id=\"entryFilter\"><option value=\"\">All entries</option></select>
    <select id=\"floodFilter\"><option value=\"\">All flood risks</option></select>
    <select id=\"scoredFilter\"><option value=\"\">Scored?</option><option value=\"true\">Scored</option><option value=\"false\">Unscored</option></select>
    <button id=\"resetBtn\">Reset</button>
  </section>

  <main class=\"grid\">
    <section class=\"card\">
      <div class=\"card-head\">
        <div><div class=\"card-title\">Priority Decision Table</div><div class=\"hint\">Click column headers to sort. Sticker names open CS2Tokens. Preview opens the sticker's Steam Community page in Steam.</div></div>
        <div class=\"hint\" id=\"sortHint\">Sorted by priority rank</div>
      </div>
      <div class=\"table-wrap\">
        <table id=\"table\">
          <colgroup>
            <col class=\"rank-col\" />
            <col class=\"sticker-col\" />
            <col class=\"price-col\" />
            <col class=\"decision-col\" />
            <col class=\"market-col\" />
            <col class=\"scores-col\" />
            <col class=\"notes-col\" />
          </colgroup>
          <thead>
            <tr>
              <th class=\"sticky-rank sortable\" data-sort=\"priority_rank\">Rank</th>
              <th class=\"sticky-sticker sortable\" data-sort=\"sticker\">Sticker</th>
              <th class=\"sortable\" data-sort=\"price_tokens\">Price</th>
              <th class=\"sortable\" data-sort=\"verdict\">Decision</th>
              <th class=\"sortable\" data-sort=\"recent_return_pct\">Market</th>
              <th class=\"sortable\" data-sort=\"quality_score\">Scores</th>
              <th>Notes</th>
            </tr>
          </thead>
          <tbody id=\"tbody\"></tbody>
        </table>
      </div>
      <div class=\"footer-note\">Steam Preview links open each sticker's Steam Community Market page through Steam and are cached in <code>visualized/steam_preview_cache.json</code>. The dashboard only uses a CS2 inspect command if a real inspect URL exists in source data; it does not generate synthetic inspect payloads.</div>
    </section>

    <section class=\"section-grid\">
      <section class=\"card chartbox\">
        <div class=\"card-head\"><div><div class=\"card-title\">Top Priority Movement Comparator</div><div class=\"hint\">Labels are always visible. Scroll/zoom horizontally if crowded.</div></div></div>
        <div class=\"chart-controls\">
          <label class=\"chip\">Top N <select id=\"topN\"><option>10</option><option selected>16</option><option>24</option><option>32</option></select></label>
          <label class=\"chip\">Scale <select id=\"movementScale\"><option value=\"normalized\" selected>Normalized</option><option value=\"price\">Token price</option></select></label>
          <button id=\"rerenderCharts\">Update charts</button>
        </div>
        <div class=\"plot-scroll\"><svg id=\"movementPlot\" class=\"main-plot\"></svg></div>
      </section>

      <section class=\"card chartbox\">
        <div class=\"card-head\"><div><div class=\"card-title\">Discount vs Flood Risk Map</div><div class=\"hint\">Best speculative zone is far right + lower half: high discount, lower flood risk.</div></div></div>
        <div class=\"chart-controls\"><div class=\"legend\" id=\"legend\"></div></div>
        <div class=\"plot-scroll\"><svg id=\"scatterPlot\" class=\"main-plot\"></svg></div>
      </section>
    </section>
  </main>
</div>

<script id=\"records-json\" type=\"application/json\">__DATA_JSON__</script>
<script id=\"series-json\" type=\"application/json\">__SERIES_JSON__</script>
<script>
const records = JSON.parse(document.getElementById('records-json').textContent);
const historySeries = JSON.parse(document.getElementById('series-json').textContent);
const verdictColors = __VERDICT_COLORS__;
let sortKey = 'priority_rank';
let sortDir = 1;
let filtered = [];
let viewMode = 'list';
const recordById = new Map(records.map(r => [String(r.sticker_id), r]));

const $ = (id) => document.getElementById(id);
const fmt = (v, d=0) => Number.isFinite(Number(v)) ? Number(v).toFixed(d) : '—';
const num = (v) => Number.isFinite(Number(v)) ? Number(v) : null;
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

function uniqueValues(key) { return [...new Set(records.map(r => r[key]).filter(Boolean))].sort(); }
function fillSelect(id, key) {
  const el = $(id);
  for (const v of uniqueValues(key)) { const opt = document.createElement('option'); opt.value = v; opt.textContent = v; el.appendChild(opt); }
}
function colorForVerdict(v) { return verdictColors[v] || '#94a3b8'; }
function pctClass(v) { if (!Number.isFinite(Number(v))) return 'flat'; if (Number(v) > 0.5) return 'pos'; if (Number(v) < -0.5) return 'neg'; return 'flat'; }
function money(v) { return '$' + fmt(v, 2); }
function tokens(v) { return Math.round(Number(v)||0).toLocaleString(); }

function sparkline(points, width=320, height=96) {
  if (!points || points.length < 2) return `<svg class="spark" viewBox="0 0 ${width} ${height}"><text x="8" y="26" fill="#64748b" font-size="11">no chart</text></svg>`;
  const prices = points.map(p => Number(p.price)).filter(Number.isFinite);
  if (prices.length < 2) return `<svg class="spark" viewBox="0 0 ${width} ${height}"><text x="8" y="26" fill="#64748b" font-size="11">no chart</text></svg>`;
  const min = Math.min(...prices), max = Math.max(...prices); const span = Math.max(max-min, 1e-9);
  const coords = points.map((p,i) => {
    const x = 6 + i * ((width-12) / Math.max(points.length-1,1));
    const y = height-7 - ((Number(p.price)-min)/span) * (height-15);
    return [x,y];
  });
  const line = coords.map((p,i)=>`${i?'L':'M'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ');
  const area = `${line} L${coords.at(-1)[0].toFixed(1)},${height-5} L${coords[0][0].toFixed(1)},${height-5} Z`;
  const up = coords.at(-1)[1] < coords[0][1];
  const stroke = up ? '#4ade80' : '#fb7185';
  return `<svg class="spark" viewBox="0 0 ${width} ${height}" aria-label="trend"><path class="area" d="${area}" fill="${stroke}"></path><path class="line" d="${line}" stroke="${stroke}"></path><circle class="dot" cx="${coords.at(-1)[0].toFixed(1)}" cy="${coords.at(-1)[1].toFixed(1)}" r="4" fill="${stroke}"></circle></svg>`;
}

function rowHtml(r) {
  const points = historySeries[r.sticker_id] || [];
  const vcolor = colorForVerdict(r.verdict);
  const ret = Number(r.recent_return_pct);
  const retText = Number.isFinite(ret) ? `${ret.toFixed(1)}%` : '—';
  const link = r.item_url || '#';
  const img = r.image_url || '';
  const previewHref = r.steam_url || '#';
  const previewIsInspect = r.steam_preview_status === 'inspect';
  const previewLabel = 'Preview';
  const previewTitle = previewIsInspect ? 'Open direct sticker inspect in Steam' : 'Open sticker page in Steam';
  return `<tr>
    <td class="sticky-rank"><div class="rank">#${r.priority_rank}</div><div class="tier">${esc(r.priority_tier || '')}</div></td>
    <td class="sticky-sticker">
      <div class="sticker-cell">
        <img class="thumb" src="${esc(img)}" loading="lazy" onerror="this.style.visibility='hidden'" />
        <div class="name-wrap">
          <a class="name" href="${esc(link)}" target="_blank" rel="noopener">${esc(r.sticker)} ↗</a>
          <div class="meta">${esc(r.category)} · ${esc(r.variant)} · ${esc(r.team || '—')}</div>
          <div class="btns"><a class="mini-btn primary" href="${esc(link)}" target="_blank" rel="noopener">CS2Tokens</a><a class="mini-btn steam" href="${esc(previewHref)}" title="${esc(previewTitle)}">${esc(previewLabel)}</a></div>
        </div>
      </div>
    </td>
    <td class="market-cell">
      <div class="price">${tokens(r.price_tokens)}</div>
      <div class="usd">${money(r.usd_price)}</div>
    </td>
    <td>
      <div class="cell-stack">
        <span class="verdict" style="background:${vcolor}">${esc(r.verdict)}</span>
        <div class="detail-line"><span>Priority</span><b>${fmt(r.priority_score,1)}</b></div>
        <div class="detail-line"><span>Size</span><b>${esc(r.suggested_size || '—')}</b></div>
        <div class="detail-line"><span>Entry</span><b>${esc(r.entry_tier || '—')}</b></div>
      </div>
    </td>
    <td>
      <div class="cell-stack">
        <div class="detail-line"><span>24h</span><b class="${pctClass(ret)}">${retText}</b></div>
        <div class="detail-line"><span>Discount</span><b>${fmt(r.discount_from_high_pct,0)}%</b></div>
        <div class="detail-line"><span>Flood</span><b>${esc(r.flood_risk || '—')}</b></div>
        ${sparkline(points)}
      </div>
    </td>
    <td>
      <div class="score-list">
        <div class="detail-line"><span>Quality</span><b>${fmt(r.quality_score,2)}</b></div>
        <div class="detail-line"><span>History</span><b>${fmt(r.history_score,2)}</b></div>
        <div class="detail-line"><span>Decision</span><b>${fmt(r.decision_score,2)}</b></div>
        <div class="detail-line"><span>Trend</span><b>${fmt(r.trend_score,2)}</b></div>
      </div>
    </td>
    <td>
      <div class="notes-block">
        <div><span class="note-label">Reason</span><div class="reason-small">${esc(r.quick_reason || '—')}</div></div>
        <div><span class="note-label">Risk</span><div class="reason-small">${esc(r.risk_note || '—')}</div></div>
        <div><span class="note-label">Action</span><div class="action-small">${esc(r.action_note || r.suggested_size || '—')}</div></div>
      </div>
    </td>
  </tr>`;
}

function applyFilters() {
  const q = $('search').value.trim().toLowerCase();
  const verdict = $('verdictFilter').value;
  const category = $('categoryFilter').value;
  const entry = $('entryFilter').value;
  const flood = $('floodFilter').value;
  const scored = $('scoredFilter').value;
  filtered = records.filter(r => {
    if (verdict && r.verdict !== verdict) return false;
    if (category && r.category !== category) return false;
    if (entry && r.entry_tier !== entry) return false;
    if (flood && r.flood_risk !== flood) return false;
    if (scored && String(r.scored) !== scored) return false;
    if (q) {
      const hay = [
        r.sticker, r.team, r.verdict, r.quick_reason, r.action_note, r.trend_signal,
        r.entry_tier, r.flood_risk, r.price_tokens, money(r.usd_price), tokens(r.price_tokens)
      ].join(' ').toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });
  sortRows();
  renderTable();
}

function sortRows() {
  filtered.sort((a,b)=>{
    const av=a[sortKey], bv=b[sortKey];
    const an=Number(av), bn=Number(bv);
    let cmp;
    if(Number.isFinite(an) && Number.isFinite(bn)) cmp = an-bn;
    else cmp = String(av ?? '').localeCompare(String(bv ?? ''));
    return cmp * sortDir;
  });
}

function renderTable() {
  $('tbody').innerHTML = filtered.map(rowHtml).join('') || `<tr><td colspan="7" class="empty">No stickers match the filters.</td></tr>`;
  $('visibleCount').textContent = filtered.length;
  $('totalCount').textContent = records.length;
  $('topVerdict').textContent = filtered[0]?.verdict || '—';
}

function makeOptions() {
  fillSelect('verdictFilter','verdict'); fillSelect('categoryFilter','category'); fillSelect('entryFilter','entry_tier'); fillSelect('floodFilter','flood_risk');
}

function svg(tag, attrs={}, children='') {
  const attr = Object.entries(attrs).map(([k,v])=>`${k}="${String(v).replace(/"/g,'&quot;')}"`).join(' ');
  return `<${tag} ${attr}>${children}</${tag}>`;
}
function scale(v, a,b, c,d) { if(!Number.isFinite(v)) return c; if (Math.abs(b-a)<1e-9) return (c+d)/2; return c + (v-a)*(d-c)/(b-a); }
function labelSafe(s) { return esc(String(s||'').replace(/\s*\(Holo\)/,'').slice(0,24)); }

function movementChart() {
  const n = Number($('topN').value) || 16;
  const mode = $('movementScale').value;
  const rows = filtered.slice(0,n).filter(r => (historySeries[r.sticker_id]||[]).length >= 2);
  const W = Math.max(1500, 1080 + rows.length*48), H = Math.max(820, 140 + rows.length*38);
  const L=190, R=90, T=70, B=76;
  let values=[];
  rows.forEach(r => { (historySeries[r.sticker_id]||[]).forEach(p => values.push(mode==='price'?p.price:p.norm)); });
  if(!values.length) return `<text x="20" y="40" fill="#94a3b8">No movement data</text>`;
  let min=Math.min(...values), max=Math.max(...values); if(mode==='normalized'){ min=Math.min(50,min); max=Math.max(160,max); }
  let grid='';
  for(let i=0;i<=5;i++){ const y=scale(i,0,5,H-B,T); const val=scale(i,0,5,min,max).toFixed(0); grid += `<line class="gridline" x1="${L}" y1="${y}" x2="${W-R}" y2="${y}"></line><text x="${L-12}" y="${y+4}" text-anchor="end" fill="#94a3b8" font-size="12">${val}</text>`; }
  let body='';
  rows.forEach((r,idx)=>{
    const pts=(historySeries[r.sticker_id]||[]); const vals=pts.map(p=>mode==='price'?p.price:p.norm);
    const xs=vals.map((_,i)=>scale(i,0,Math.max(vals.length-1,1),L,W-R));
    const ys=vals.map(v=>scale(v,min,max,H-B,T));
    const color=colorForVerdict(r.verdict);
    const d=xs.map((x,i)=>`${i?'L':'M'}${x.toFixed(1)} ${ys[i].toFixed(1)}`).join(' ');
    const lastX=xs.at(-1), lastY=ys.at(-1);
    body += `<path d="${d}" fill="none" stroke="${color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" opacity=".9"></path>`;
    body += `<circle cx="${lastX}" cy="${lastY}" r="6" fill="${color}" stroke="#020617" stroke-width="2"><title>${esc(r.sticker)} | ${tokens(r.price_tokens)} tokens</title></circle>`;
    body += `<text class="point-label" x="${lastX+10}" y="${lastY+4}">#${r.priority_rank} ${labelSafe(r.sticker)}</text>`;
  });
  const title=`<text class="plot-title" x="${L}" y="32">Top ${rows.length} priority candidates — ${mode==='price'?'Token price':'Normalized movement, first point = 100'}</text>`;
  return `${title}${grid}${body}<text x="${L}" y="${H-28}" fill="#94a3b8" font-size="13">Left → right = collected tooltip points. Use this to compare bounce/fade behavior, not exact daily candles.</text>`;
}

function scatterChart() {
  const rows = filtered.filter(r => Number.isFinite(Number(r.discount_from_high_pct)) && Number.isFinite(Number(r.flood_risk_score))).slice(0,160);
  const W=1500, H=820, L=90, R=70, T=70, B=90;
  let grid='';
  for(let i=0;i<=5;i++){
    const x=scale(i,0,5,L,W-R); const xv=scale(i,0,5,0,100).toFixed(0); grid+=`<line class="gridline" x1="${x}" y1="${T}" x2="${x}" y2="${H-B}"></line><text x="${x}" y="${H-B+24}" text-anchor="middle" fill="#94a3b8" font-size="12">${xv}%</text>`;
    const y=scale(i,0,5,H-B,T); const yv=scale(i,0,5,0,1).toFixed(1); grid+=`<line class="gridline" x1="${L}" y1="${y}" x2="${W-R}" y2="${y}"></line><text x="${L-12}" y="${y+4}" text-anchor="end" fill="#94a3b8" font-size="12">${yv}</text>`;
  }
  let body='';
  rows.forEach(r=>{
    const x=scale(Number(r.discount_from_high_pct),0,100,L,W-R);
    const y=scale(Number(r.flood_risk_score),0,1,H-B,T);
    const size=5+Math.min(20, Math.max(0, Number(r.priority_score)||0)/6);
    const color=colorForVerdict(r.verdict);
    body+=`<circle cx="${x}" cy="${y}" r="${size.toFixed(1)}" fill="${color}" opacity=".78" stroke="#0f172a" stroke-width="2"><title>#${r.priority_rank} ${esc(r.sticker)}\nDiscount: ${fmt(r.discount_from_high_pct,1)}%\nFlood: ${fmt(r.flood_risk_score,2)}\nPrice: ${tokens(r.price_tokens)} tokens</title></circle>`;
  });
  rows.slice(0,34).forEach(r=>{
    const x=scale(Number(r.discount_from_high_pct),0,100,L,W-R);
    const y=scale(Number(r.flood_risk_score),0,1,H-B,T);
    body+=`<text class="point-label" x="${x+12}" y="${y+4}">#${r.priority_rank} ${labelSafe(r.sticker)}</text>`;
  });
  const title=`<text class="plot-title" x="${L}" y="32">Discount vs Flood Risk — prioritize right side with lower flood risk</text>`;
  const axes=`<text x="${W/2}" y="${H-28}" text-anchor="middle" fill="#94a3b8" font-size="14">Discount from previous high →</text><text x="24" y="${H/2}" transform="rotate(-90 24 ${H/2})" text-anchor="middle" fill="#94a3b8" font-size="14">Flood risk score →</text>`;
  return `${title}${grid}${body}${axes}`;
}

function renderCharts() {
  $('movementPlot').setAttribute('viewBox','0 0 1700 900');
  $('movementPlot').innerHTML = movementChart();
  $('scatterPlot').setAttribute('viewBox','0 0 1500 820');
  $('scatterPlot').innerHTML = scatterChart();
}

function wire() {
  makeOptions();
  for (const id of ['search','verdictFilter','categoryFilter','entryFilter','floodFilter','scoredFilter']) { $(id).addEventListener('input', applyFilters); }
  $('resetBtn').addEventListener('click', () => { ['search','verdictFilter','categoryFilter','entryFilter','floodFilter','scoredFilter'].forEach(id => $(id).value=''); sortKey='priority_rank'; sortDir=1; applyFilters(); });
  $('rerenderCharts').addEventListener('click', renderCharts);
  document.querySelectorAll('th.sortable').forEach(th=>th.addEventListener('click',()=>{
    const key=th.dataset.sort;
    if(sortKey===key) sortDir*=-1; else { sortKey=key; sortDir=1; }
    $('sortHint').textContent=`Sorted by ${key} ${sortDir===1?'ascending':'descending'}`;
    applyFilters();
  }));
  const legend = Object.entries(verdictColors).map(([k,v])=>`<span><span class="legend-dot" style="background:${v}"></span>${k}</span>`).join('');
  $('legend').innerHTML = legend;
  applyFilters();
}
wire();
</script>
</body>
</html>"""
    template = template.replace(r'\"', '"')
    return (template
        .replace("__DATA_JSON__", data_json)
        .replace("__SERIES_JSON__", series_json)
        .replace("__VERDICT_COLORS__", json.dumps(VERDICT_COLORS))
    )


def build_html(records: list[dict], series: dict[str, list[dict]]) -> str:
    data_json = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    series_json = json.dumps(series, ensure_ascii=False).replace("</", "<\\/")

    template = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>CS2 Sticker Decision Dashboard</title>
<style>
  :root {
    --bg:#090d14;
    --panel:#111822;
    --panel-2:#151f2d;
    --panel-3:#1a2635;
    --line:#263244;
    --line-soft:#1e2938;
    --text:#eef3f8;
    --muted:#a9b4c4;
    --faint:#78869a;
    --blue:#5b8cff;
    --green:#34d399;
    --yellow:#f6c945;
    --red:#fb7185;
    --shadow:0 18px 40px rgba(0,0,0,.26);
  }
  * { box-sizing:border-box; }
  html { color-scheme:dark; }
  body {
    margin:0;
    background:var(--bg);
    color:var(--text);
    font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size:14px;
    letter-spacing:0;
  }
  a { color:inherit; text-decoration:none; }
  button, input, select {
    font:inherit;
    color:var(--text);
    background:#0d1420;
    border:1px solid var(--line);
    border-radius:6px;
    min-height:36px;
  }
  input, select { width:100%; padding:8px 10px; }
  button { padding:8px 12px; cursor:pointer; font-weight:750; }
  button:hover, a.action:hover { border-color:#4f6f9e; background:#111d2d; }
  input:focus, select:focus { outline:2px solid rgba(91,140,255,.28); border-color:var(--blue); }
  .app { width:min(1840px, calc(100vw - 28px)); margin:0 auto; padding:16px 0 28px; }
  .topbar {
    display:grid;
    grid-template-columns:minmax(320px,1fr) auto;
    gap:18px;
    align-items:start;
    padding:18px 20px;
    background:linear-gradient(135deg, #111822, #0d1420 68%, #111b2b);
    border:1px solid rgba(169,180,196,.18);
    border-radius:8px;
    box-shadow:var(--shadow);
  }
  h1 { margin:0 0 6px; font-size:24px; line-height:1.1; letter-spacing:0; }
  .sub { margin:0; color:var(--muted); line-height:1.45; max-width:980px; }
  .stats { display:grid; grid-template-columns:repeat(5, minmax(104px, 1fr)); gap:8px; min-width:620px; }
  .stat { padding:10px 12px; border:1px solid var(--line); border-radius:7px; background:rgba(16,24,34,.74); }
  .stat span { display:block; color:var(--muted); font-size:12px; margin-bottom:4px; }
  .stat b { display:block; font-size:20px; line-height:1.05; font-variant-numeric:tabular-nums; }
  .filter-panel { margin:12px 0; }
  .filter-panel > summary { display:none; }
  .filters {
    position:sticky;
    top:0;
    z-index:30;
    display:grid;
    grid-template-columns:minmax(260px,1.25fr) repeat(6, minmax(130px,.6fr)) 120px 112px 118px;
    gap:10px;
    margin:0;
    padding:10px;
    background:rgba(9,13,20,.96);
    border:1px solid rgba(169,180,196,.18);
    border-radius:8px;
    backdrop-filter:blur(12px);
    box-shadow:0 10px 24px rgba(0,0,0,.25);
  }
  .field label { display:block; color:var(--muted); font-size:11px; margin:0 0 4px; }
  .content { display:grid; gap:14px; }
  .panel {
    background:rgba(17,24,34,.94);
    border:1px solid rgba(169,180,196,.16);
    border-radius:8px;
    box-shadow:var(--shadow);
    overflow:hidden;
  }
  .panel-head {
    display:flex;
    justify-content:space-between;
    gap:16px;
    align-items:flex-start;
    padding:14px 16px;
    border-bottom:1px solid var(--line-soft);
  }
  .panel-title { font-size:16px; font-weight:850; }
  .hint { color:var(--muted); font-size:12px; line-height:1.45; }
  .panel-tools { display:flex; flex-wrap:wrap; align-items:flex-start; justify-content:flex-end; gap:9px; min-width:340px; }
  .view-toggle {
    display:inline-flex;
    gap:3px;
    padding:3px;
    border:1px solid var(--line);
    border-radius:8px;
    background:#0b111b;
  }
  .view-btn {
    display:inline-flex;
    align-items:center;
    gap:7px;
    min-height:30px;
    padding:5px 9px;
    border:0;
    border-radius:6px;
    background:transparent;
    color:var(--muted);
    font-size:12px;
    font-weight:900;
  }
  .view-btn.active { background:#1b2b43; color:#edf4ff; box-shadow:inset 0 0 0 1px rgba(91,140,255,.30); }
  .view-icon { width:14px; height:14px; position:relative; display:inline-block; opacity:.95; }
  .view-icon.list-icon::before,
  .view-icon.list-icon::after {
    content:"";
    position:absolute;
    left:0;
    right:0;
    height:2px;
    border-radius:999px;
    background:currentColor;
    box-shadow:0 5px 0 currentColor, 0 10px 0 currentColor;
  }
  .view-icon.list-icon::before { top:1px; }
  .view-icon.grid-icon {
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:3px;
  }
  .view-icon.grid-icon::before {
    content:"";
    grid-column:1 / -1;
    width:100%;
    height:100%;
    border-radius:3px;
    background:currentColor;
    box-shadow:0 0 0 999px currentColor;
    clip-path:polygon(0 0, 42% 0, 42% 42%, 0 42%, 0 58%, 42% 58%, 42% 100%, 0 100%, 58% 0, 100% 0, 100% 42%, 58% 42%, 58% 58%, 100% 58%, 100% 100%, 58% 100%, 58% 0);
  }
  .grid-controls {
    display:none;
    align-items:center;
    gap:7px;
    padding:3px;
    border:1px solid var(--line);
    border-radius:8px;
    background:#0b111b;
  }
  .grid-controls.active { display:flex; }
  .grid-controls label { color:var(--muted); font-size:11px; font-weight:850; padding-left:6px; }
  .grid-controls select,
  .grid-controls input { width:auto; min-width:86px; min-height:30px; padding:5px 7px; font-size:12px; }
  .grid-controls input { width:78px; }
  .chart-grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
  .chart-body { padding:14px; }
  .chart-frame {
    width:100%;
    min-height:390px;
    border:1px solid var(--line);
    border-radius:7px;
    background:#080d14;
    overflow:auto;
  }
  .chart-frame.tall { min-height:470px; }
  svg.chart { display:block; width:100%; min-width:760px; }
  svg.chart.tall { min-width:1180px; }
  .chart-title { fill:#e6edf7; font-size:15px; font-weight:850; }
  .chart-note, .axis text { fill:#98a6b8; font-size:12px; }
  .gridline { stroke:rgba(152,166,184,.15); stroke-width:1; }
  .axis-line { stroke:#36455a; stroke-width:1; }
  .point-label {
    fill:#e6edf7;
    font-size:12px;
    paint-order:stroke;
    stroke:#080d14;
    stroke-width:4px;
    stroke-linejoin:round;
  }
  .legend { display:flex; gap:10px; flex-wrap:wrap; color:var(--muted); font-size:12px; align-items:center; }
  .legend-dot { width:9px; height:9px; display:inline-block; border-radius:99px; margin-right:5px; vertical-align:-1px; }
  .chart-controls { display:flex; flex-wrap:wrap; align-items:center; gap:10px; padding:0 14px 12px; }
  .inline-select { width:auto; min-width:96px; }
  .table-wrap[hidden], .grid-view[hidden] { display:none !important; }
  .table-wrap { max-height:82vh; overflow:auto; }
  .grid-view {
    --grid-cols:5;
    display:grid;
    grid-template-columns:repeat(var(--grid-cols), minmax(0, 1fr));
    gap:12px;
    padding:14px;
  }
  .grid-card {
    position:relative;
    display:grid;
    grid-template-rows:auto minmax(0,1fr) auto;
    gap:8px;
    min-width:0;
    min-height:262px;
    padding:10px;
    border:1px solid rgba(169,180,196,.16);
    border-radius:8px;
    background:
      radial-gradient(circle at 50% 32%, rgba(91,140,255,.13), transparent 46%),
      linear-gradient(180deg, rgba(19,29,44,.96), rgba(11,17,27,.98));
    color:var(--text);
    text-align:left;
    cursor:pointer;
    overflow:hidden;
    box-shadow:0 12px 28px rgba(0,0,0,.20);
    transition:transform .16s ease, border-color .16s ease, box-shadow .16s ease, background-color .16s ease;
  }
  .grid-card:hover,
  .grid-card:focus-visible {
    transform:translateY(-2px);
    border-color:rgba(91,140,255,.42);
    box-shadow:0 16px 34px rgba(0,0,0,.28), inset 0 0 0 1px rgba(91,140,255,.12);
    outline:none;
  }
  .grid-card.release-low-card { border-color:rgba(52,211,153,.34); }
  .grid-rank {
    position:absolute;
    top:9px;
    left:9px;
    z-index:2;
    display:inline-flex;
    align-items:center;
    min-height:24px;
    padding:3px 7px;
    border-radius:999px;
    background:rgba(8,13,20,.76);
    border:1px solid rgba(169,180,196,.20);
    color:#fff;
    font-size:12px;
    font-weight:950;
    backdrop-filter:blur(8px);
  }
  .grid-tier {
    position:absolute;
    top:9px;
    right:9px;
    z-index:2;
    display:inline-flex;
    align-items:center;
    min-height:24px;
    padding:3px 7px;
    border-radius:999px;
    background:#dce8ff;
    color:#0b1220;
    font-size:11px;
    font-weight:950;
  }
  .grid-image {
    display:flex;
    align-items:center;
    justify-content:center;
    aspect-ratio:1;
    min-height:0;
    padding:18px 6px 6px;
  }
  .grid-image img {
    width:100%;
    height:100%;
    object-fit:contain;
    filter:drop-shadow(0 12px 16px rgba(0,0,0,.32));
    transition:transform .16s ease;
  }
  .grid-card:hover .grid-image img { transform:scale(1.035); }
  .grid-title { min-width:0; }
  .grid-name {
    display:block;
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
    color:#fff;
    font-size:13px;
    font-weight:950;
    line-height:1.2;
  }
  .grid-meta {
    display:flex;
    align-items:center;
    gap:6px;
    min-width:0;
    margin-top:5px;
    color:var(--muted);
    font-size:11px;
    line-height:1.2;
  }
  .grid-variant { flex:0 0 auto; padding:2px 5px; border:1px solid var(--line); border-radius:999px; color:#d8e1ee; }
  .grid-team { min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .grid-bottom { display:grid; gap:8px; }
  .grid-verdict {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:7px;
    min-width:0;
  }
  .grid-verdict-pill {
    min-width:0;
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
    padding:5px 7px;
    border-radius:999px;
    color:#07101b;
    font-size:10px;
    font-weight:950;
    line-height:1.05;
    text-transform:uppercase;
  }
  .grid-price { color:#fff; font-size:14px; font-weight:950; font-variant-numeric:tabular-nums; white-space:nowrap; }
  .grid-kpis { display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:5px; }
  .grid-kpi {
    min-width:0;
    padding:5px 5px;
    border:1px solid rgba(169,180,196,.13);
    border-radius:6px;
    background:rgba(8,13,20,.42);
  }
  .grid-kpi small {
    display:block;
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
    color:var(--muted);
    font-size:9px;
    font-weight:850;
    line-height:1.1;
    text-transform:uppercase;
  }
  .grid-kpi b {
    display:flex;
    align-items:center;
    gap:4px;
    min-width:0;
    margin-top:3px;
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
    color:#edf4ff;
    font-size:11px;
    font-weight:950;
    font-variant-numeric:tabular-nums;
  }
  .signal-dot {
    flex:0 0 auto;
    width:7px;
    height:7px;
    border-radius:999px;
    background:#94a3b8;
    box-shadow:0 0 0 0 rgba(148,163,184,.0);
  }
  .signal-dot.up, .signal-dot.low { background:#34d399; }
  .signal-dot.down { background:#fb7185; }
  .signal-dot.watch { background:#f6c945; }
  .grid-low-ribbon {
    position:absolute;
    left:9px;
    bottom:9px;
    width:9px;
    height:9px;
    border-radius:999px;
    background:#34d399;
    box-shadow:0 0 0 4px rgba(52,211,153,.12);
  }
  .grid-view[data-density="dense"] .grid-card { min-height:218px; padding:8px; }
  .grid-view[data-density="dense"] .grid-name { font-size:12px; }
  .grid-view[data-density="dense"] .grid-meta,
  .grid-view[data-density="dense"] .grid-kpi small { display:none; }
  .grid-view[data-density="dense"] .grid-kpis { grid-template-columns:1fr 1fr; }
  .grid-view[data-density="dense"] .grid-kpi:last-child { display:none; }
  .grid-view[data-density="ultra"] .grid-card { min-height:174px; gap:5px; padding:7px; }
  .grid-view[data-density="ultra"] .grid-title { display:none; }
  .grid-view[data-density="ultra"] .grid-kpis { display:none; }
  .grid-view[data-density="ultra"] .grid-verdict { display:block; }
  .grid-view[data-density="ultra"] .grid-verdict-pill { display:block; margin-bottom:5px; }
  .grid-empty { grid-column:1 / -1; padding:38px; text-align:center; color:var(--muted); }
  table { width:100%; border-collapse:separate; border-spacing:0; table-layout:fixed; }
  col.rank-col { width:64px; }
  col.sticker-col { width:28%; }
  col.price-col { width:11%; }
  col.decision-col { width:15%; }
  col.edge-col { width:16%; }
  col.market-col { width:16%; }
  col.notes-col { width:14%; }
  thead th {
    position:sticky;
    top:0;
    z-index:12;
    background:#151f2d;
    color:#cfdae8;
    text-align:left;
    padding:9px 10px;
    border-bottom:1px solid var(--line);
    font-size:12px;
    font-weight:850;
    letter-spacing:0;
  }
  th.sortable { cursor:pointer; }
  th.sortable:hover { color:white; }
  tbody tr { background:#101722; transition:background-color .14s ease, box-shadow .14s ease; }
  tbody tr:nth-child(even) { background:#0e1520; }
  tbody tr:hover { background:#162233; box-shadow:inset 0 0 0 1px rgba(91,140,255,.14); }
  tbody tr.release-low-row td:first-child { box-shadow:inset 3px 0 0 #34d399; }
  tbody td {
    padding:14px 10px;
    border-bottom:1px solid rgba(152,166,184,.12);
    vertical-align:top;
    min-width:0;
  }
  .rank { font-weight:900; font-size:17px; margin-bottom:6px; }
  .tier { display:inline-flex; align-items:center; justify-content:center; min-width:30px; height:24px; padding:0 8px; border-radius:999px; background:#dce8ff; color:#0b1220; font-weight:900; font-size:11px; }
  .sticker-cell { display:grid; grid-template-columns:172px minmax(0,1fr); gap:14px; align-items:start; }
  .thumb {
    width:172px;
    height:172px;
    object-fit:contain;
    border-radius:7px;
    background:linear-gradient(145deg, rgba(255,255,255,.08), rgba(9,13,20,.94));
    border:1px solid rgba(169,180,196,.22);
    transition:transform .14s ease, border-color .14s ease, box-shadow .14s ease;
  }
  tbody tr:hover .thumb { transform:translateY(-1px); border-color:rgba(91,140,255,.38); box-shadow:0 10px 24px rgba(0,0,0,.20); }
  .name { display:inline; font-size:16px; line-height:1.25; font-weight:900; color:#fff; }
  .name:hover { text-decoration:underline; text-decoration-thickness:1px; }
  .meta { margin-top:7px; color:var(--muted); font-size:12px; line-height:1.4; }
  .chips { display:flex; flex-wrap:wrap; gap:6px; margin-top:9px; }
  .chip { display:inline-flex; align-items:center; max-width:100%; padding:4px 7px; border:1px solid var(--line); border-radius:999px; background:rgba(13,20,32,.8); color:#d8e1ee; font-size:12px; line-height:1.2; }
  .actions { display:flex; flex-wrap:wrap; gap:7px; margin-top:10px; }
  .action { display:inline-flex; align-items:center; justify-content:center; min-height:30px; padding:6px 9px; border:1px solid var(--line); border-radius:6px; font-weight:800; font-size:12px; }
  .action.primary { color:#c9dbff; border-color:rgba(91,140,255,.58); }
  .price-main { font-size:30px; line-height:1; font-weight:950; font-variant-numeric:tabular-nums; color:#fff; }
  .price-sub { margin-top:6px; color:var(--muted); font-size:14px; font-weight:800; font-variant-numeric:tabular-nums; }
  .price-range {
    display:grid;
    gap:6px;
    margin-top:12px;
    padding-top:10px;
    border-top:1px solid var(--line-soft);
  }
  .price-range-row {
    display:grid;
    grid-template-columns:34px minmax(0,1fr);
    gap:8px;
    align-items:start;
    font-variant-numeric:tabular-nums;
    border-radius:6px;
  }
  .price-range-row span {
    color:var(--muted);
    font-size:11px;
    font-weight:850;
    text-transform:uppercase;
    letter-spacing:.04em;
  }
  .price-range-row b {
    display:block;
    color:#edf4ff;
    font-size:13px;
    line-height:1.2;
    overflow-wrap:anywhere;
  }
  .price-range-row small {
    display:block;
    margin-top:2px;
    color:var(--muted);
    font-size:11px;
    font-weight:750;
    line-height:1.2;
  }
  .price-range-row.low b { color:#86efac; }
  .price-range-row.high b { color:#f8dfa5; }
  .price-range-row.prev {
    grid-template-columns:38px minmax(0,1fr);
    margin:0 -4px 1px;
    padding:6px 5px;
    border:1px solid rgba(91,140,255,.22);
    background:linear-gradient(90deg, rgba(91,140,255,.15), rgba(91,140,255,.045));
  }
  .price-range-row.prev span { color:#9cc3ff; }
  .price-range-row.prev b { color:#dbeafe; font-size:14px; }
  .price-range-row.prev small { color:#aebeda; }
  .price-range-row.prev.up { border-color:rgba(52,211,153,.28); background:linear-gradient(90deg, rgba(52,211,153,.12), rgba(91,140,255,.04)); }
  .price-range-row.prev.down { border-color:rgba(251,113,133,.30); background:linear-gradient(90deg, rgba(251,113,133,.12), rgba(91,140,255,.04)); }
  .price-range-row.prev:hover { border-color:rgba(147,197,253,.42); }
  .price-delta {
    display:inline-flex;
    align-items:center;
    margin-left:6px;
    padding:1px 5px;
    border-radius:999px;
    font-size:10px;
    font-weight:950;
    line-height:1.25;
    background:rgba(169,180,196,.12);
    color:#d8e1ee;
  }
  .price-delta.up { background:rgba(52,211,153,.14); color:#a7f3d0; }
  .price-delta.down { background:rgba(251,113,133,.14); color:#fecdd3; }
  .release-low-badge {
    display:inline-flex;
    align-items:center;
    margin-top:10px;
    padding:4px 7px;
    border:1px solid rgba(52,211,153,.38);
    border-radius:999px;
    background:rgba(52,211,153,.10);
    color:#a7f3d0;
    font-size:11px;
    font-weight:900;
    line-height:1.15;
    letter-spacing:.02em;
    text-transform:uppercase;
  }
  .low-gap-badge {
    display:inline-flex;
    align-items:center;
    margin-top:8px;
    padding:4px 7px;
    border:1px solid rgba(169,180,196,.24);
    border-radius:999px;
    background:rgba(13,20,32,.76);
    color:#d8e1ee;
    font-size:11px;
    font-weight:900;
    line-height:1.15;
    letter-spacing:.02em;
    text-transform:uppercase;
  }
  .low-gap-badge.near {
    border-color:rgba(52,211,153,.34);
    background:rgba(52,211,153,.08);
    color:#a7f3d0;
  }
  .low-gap-badge.mid {
    border-color:rgba(246,201,69,.34);
    background:rgba(246,201,69,.08);
    color:#f7df9d;
  }
  .verdict { display:inline-flex; max-width:100%; padding:6px 9px; border-radius:999px; color:#031018; font-size:12px; font-weight:950; line-height:1.2; }
  .metric-list { display:grid; gap:7px; margin-top:9px; }
  .metric-row { display:grid; grid-template-columns:minmax(86px,.75fr) minmax(0,1fr); gap:10px; align-items:baseline; }
  .metric-row span { color:var(--muted); font-size:12px; }
  .metric-row b { color:var(--text); font-size:13px; font-weight:850; line-height:1.25; font-variant-numeric:tabular-nums; }
  .pos { color:#5ee592 !important; }
  .neg { color:#fb7185 !important; }
  .flat { color:#bac6d6 !important; }
  .spark { width:100%; height:88px; margin-top:8px; display:block; }
  .spark .line { fill:none; stroke-width:3.2; stroke-linecap:round; stroke-linejoin:round; }
  .spark .area { opacity:.16; }
  .spark-point { cursor:crosshair; }
  .spark-axis { stroke:rgba(152,166,184,.20); stroke-width:1; }
  .spark-tip {
    position:fixed;
    display:none;
    z-index:1000;
    max-width:260px;
    padding:9px 10px;
    border:1px solid #334155;
    border-radius:6px;
    background:#0a1019;
    color:#e6edf7;
    box-shadow:0 14px 30px rgba(0,0,0,.42);
    font-size:12px;
    line-height:1.45;
    white-space:pre-line;
    pointer-events:none;
  }
  @keyframes rowIn {
    from { opacity:.82; transform:translateY(2px); }
    to { opacity:1; transform:translateY(0); }
  }
  @keyframes signalPulse {
    0% { box-shadow:0 0 0 0 rgba(52,211,153,.32); }
    70% { box-shadow:0 0 0 7px rgba(52,211,153,0); }
    100% { box-shadow:0 0 0 0 rgba(52,211,153,0); }
  }
  @media (prefers-reduced-motion:no-preference) {
    tbody tr { animation:rowIn .14s ease-out both; }
    .signal-dot.up, .signal-dot.low { animation:signalPulse 1.8s ease-out infinite; }
  }
  .note-block { display:grid; gap:9px; line-height:1.45; font-size:13px; color:#d8e1ee; }
  .note-block label { display:block; color:var(--muted); font-size:11px; margin-bottom:2px; }
  .note-action { color:#f7df9d; }
  .modal[hidden] { display:none !important; }
  .modal {
    position:fixed;
    inset:0;
    z-index:2000;
    display:grid;
    place-items:center;
    padding:18px;
  }
  .modal-backdrop {
    position:absolute;
    inset:0;
    background:rgba(3,7,12,.72);
    backdrop-filter:blur(10px);
  }
  .modal-dialog {
    position:relative;
    width:min(1080px, calc(100vw - 28px));
    max-height:calc(100vh - 28px);
    overflow:auto;
    border:1px solid rgba(169,180,196,.22);
    border-radius:10px;
    background:#0d1420;
    box-shadow:0 24px 70px rgba(0,0,0,.52);
  }
  .modal-close {
    position:sticky;
    top:10px;
    float:right;
    z-index:3;
    width:34px;
    height:34px;
    min-height:34px;
    margin:10px 10px 0 0;
    padding:0;
    border-radius:999px;
    background:#101a29;
    color:#edf4ff;
    font-size:18px;
    line-height:1;
  }
  .modal-content { padding:18px; }
  .modal-grid { display:grid; grid-template-columns:minmax(270px,.9fr) minmax(0,1.1fr); gap:18px; clear:both; }
  .modal-visual {
    position:relative;
    display:grid;
    gap:12px;
    align-content:start;
    padding:14px;
    border:1px solid rgba(169,180,196,.14);
    border-radius:9px;
    background:radial-gradient(circle at 50% 38%, rgba(91,140,255,.16), transparent 48%), #0a1019;
  }
  .modal-visual img { width:100%; max-height:430px; object-fit:contain; filter:drop-shadow(0 18px 22px rgba(0,0,0,.38)); }
  .modal-rank { position:absolute; top:12px; left:12px; padding:5px 9px; border-radius:999px; background:rgba(8,13,20,.76); border:1px solid rgba(169,180,196,.22); font-weight:950; }
  .modal-main { min-width:0; }
  .modal-title-row { display:flex; flex-wrap:wrap; align-items:center; gap:9px; margin-bottom:7px; }
  .modal-title { margin:0; color:#fff; font-size:26px; line-height:1.12; letter-spacing:0; }
  .modal-meta { color:var(--muted); font-size:13px; margin-bottom:14px; }
  .modal-price { display:flex; flex-wrap:wrap; align-items:baseline; gap:10px; margin:0 0 12px; }
  .modal-price b { color:#fff; font-size:32px; line-height:1; font-weight:950; font-variant-numeric:tabular-nums; }
  .modal-price span { color:var(--muted); font-weight:850; }
  .modal-sections { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
  .modal-section {
    padding:12px;
    border:1px solid rgba(169,180,196,.13);
    border-radius:8px;
    background:rgba(8,13,20,.42);
  }
  .modal-section h3 { margin:0 0 9px; font-size:12px; color:#cfdae8; text-transform:uppercase; letter-spacing:.04em; }
  .modal-note { margin-top:12px; }
  .empty { padding:38px; text-align:center; color:var(--muted); }
  .footer-note { padding:12px 16px; border-top:1px solid var(--line-soft); color:var(--muted); font-size:12px; line-height:1.45; }
  code { color:#d8e1ee; background:#0b111b; border:1px solid var(--line); padding:1px 5px; border-radius:4px; }
  @media (max-width:1450px) {
    .topbar { grid-template-columns:1fr; }
    .stats { min-width:0; grid-template-columns:repeat(5, minmax(110px,1fr)); }
    .filters { grid-template-columns:repeat(4, minmax(150px,1fr)); }
    .chart-grid { grid-template-columns:1fr; }
    table { min-width:1240px; }
  }
  @media (max-width:800px) {
    body { font-size:13px; }
    .app { width:100%; padding:8px 8px 18px; }
    .topbar { display:block; padding:14px; border-radius:8px; }
    h1 { font-size:20px; }
    .sub { font-size:12px; }
    .stats { grid-template-columns:1fr 1fr; gap:6px; min-width:0; margin-top:12px; }
    .stat { padding:8px 9px; }
    .stat span { font-size:11px; }
    .stat b { font-size:17px; }
    .filter-panel {
      margin:8px 0 10px;
      overflow:hidden;
      border:1px solid rgba(169,180,196,.18);
      border-radius:8px;
      background:rgba(9,13,20,.96);
      box-shadow:0 10px 24px rgba(0,0,0,.20);
    }
    .filter-panel > summary {
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:12px;
      min-height:42px;
      padding:10px 12px;
      color:#edf4ff;
      cursor:pointer;
      font-weight:900;
      list-style:none;
    }
    .filter-panel > summary::-webkit-details-marker { display:none; }
    .filter-summary-sub { color:var(--muted); font-size:11px; font-weight:750; }
    .filters {
      position:static;
      z-index:auto;
      grid-template-columns:1fr 1fr;
      gap:8px;
      padding:10px;
      border:0;
      border-top:1px solid rgba(169,180,196,.14);
      border-radius:0;
      background:transparent;
      box-shadow:none;
      backdrop-filter:none;
    }
    .filters .field:first-child { grid-column:1 / -1; }
    .filters .field { min-width:0; }
    button, input, select { min-height:34px; font-size:13px; }
    input, select { padding:7px 8px; }
    .content { gap:10px; }
    .panel-head { display:block; padding:11px 12px; }
    .panel-title { font-size:15px; }
    .hint { font-size:11px; }
    .panel-tools { justify-content:flex-start; min-width:0; margin-top:10px; }
    .view-toggle, .grid-controls { width:100%; }
    .view-btn { flex:1; justify-content:center; }
    .grid-controls { flex-wrap:wrap; }
    .grid-controls select { flex:1; min-width:112px; }
    .grid-controls input { flex:1; min-width:96px; }
    .table-wrap { max-height:none; overflow:visible; }
    .grid-view { gap:8px; padding:10px; }
    .grid-card { min-height:232px; padding:8px; }
    .grid-image { padding-top:16px; }
    .grid-name { font-size:12px; }
    .grid-kpis { grid-template-columns:1fr 1fr; }
    .grid-kpi:last-child { display:none; }
    .grid-view[data-density="dense"] .grid-card,
    .grid-view[data-density="ultra"] .grid-card { min-height:150px; }
    .grid-view[data-density="dense"] .grid-verdict,
    .grid-view[data-density="ultra"] .grid-verdict { display:block; }
    .grid-view[data-density="dense"] .grid-verdict-pill,
    .grid-view[data-density="ultra"] .grid-verdict-pill { display:block; margin-bottom:5px; }
    .grid-view[data-density="ultra"] .grid-price { font-size:12px; }
    table { display:block; width:100%; min-width:0; table-layout:auto; }
    colgroup, thead { display:none; }
    tbody { display:grid; gap:10px; }
    tbody tr {
      display:block;
      overflow:hidden;
      border:1px solid rgba(169,180,196,.16);
      border-radius:10px;
      background:#101722;
      box-shadow:0 12px 26px rgba(0,0,0,.18);
    }
    tbody tr:nth-child(even) { background:#101722; }
    tbody tr:hover { background:#101722; box-shadow:0 12px 26px rgba(0,0,0,.18); }
    tbody tr.release-low-row { border-color:rgba(52,211,153,.35); box-shadow:inset 3px 0 0 #34d399, 0 12px 26px rgba(0,0,0,.18); }
    tbody tr.release-low-row td:first-child { box-shadow:none; }
    tbody td {
      display:block;
      width:100%;
      padding:10px 12px;
      border-bottom:1px solid rgba(152,166,184,.12);
    }
    tbody td:last-child { border-bottom:0; }
    tbody td::before {
      content:attr(data-label);
      display:block;
      margin-bottom:6px;
      color:var(--muted);
      font-size:10px;
      font-weight:900;
      letter-spacing:.04em;
      line-height:1;
      text-transform:uppercase;
    }
    tbody td[data-label="Rank"] {
      display:flex;
      align-items:center;
      gap:8px;
      padding:8px 12px;
      background:#151f2d;
    }
    tbody td[data-label="Rank"]::before { display:none; }
    .rank { margin:0; font-size:15px; }
    .tier { height:22px; min-width:28px; padding:0 7px; }
    .sticker-cell { grid-template-columns:106px minmax(0,1fr); gap:10px; }
    .thumb { width:106px; height:106px; }
    .name { font-size:15px; }
    .meta { margin-top:5px; font-size:11px; }
    .chips { gap:5px; margin-top:7px; }
    .chip { padding:3px 6px; font-size:11px; }
    .actions { margin-top:8px; }
    .action { min-height:28px; padding:5px 8px; font-size:11px; }
    .price-main { font-size:26px; }
    .price-sub { font-size:13px; }
    .price-range { margin-top:10px; padding-top:9px; }
    .price-range-row { grid-template-columns:40px minmax(0,1fr); }
    .price-range-row.prev { margin:0; }
    .metric-list { gap:6px; }
    .metric-row { grid-template-columns:minmax(82px,.7fr) minmax(0,1fr); gap:8px; }
    .metric-row span { font-size:11px; }
    .metric-row b { font-size:13px; }
    .spark { height:80px; }
    .spark-tip { max-width:230px; font-size:11px; }
    .note-block { gap:8px; font-size:12px; }
    .modal { padding:8px; }
    .modal-dialog { width:calc(100vw - 16px); max-height:calc(100vh - 16px); border-radius:9px; }
    .modal-content { padding:12px; }
    .modal-grid { grid-template-columns:1fr; gap:12px; }
    .modal-title { font-size:20px; }
    .modal-price b { font-size:28px; }
    .modal-sections { grid-template-columns:1fr; gap:9px; }
    .footer-note { padding:11px 12px; font-size:11px; }
  }
</style>
</head>
<body>
<div class="app">
  <header class="topbar">
    <div>
      <h1>CS2 Sticker Decision Dashboard</h1>
      <p class="sub">Analyzer output with Paper, Foil, Holo and Gold coverage. Use filters to isolate variants, sticker type, confidence and near-low price setups before judging quality and demand.</p>
    </div>
    <div class="stats">
      <div class="stat"><span>Shown</span><b id="visibleCount">0</b></div>
      <div class="stat"><span>Total</span><b id="totalCount">0</b></div>
      <div class="stat"><span>Avg Expected</span><b id="avgExpected">0%</b></div>
      <div class="stat"><span>Avg Edge</span><b id="avgEdge">0.00</b></div>
      <div class="stat"><span>Scored</span><b id="scoredCount">0</b></div>
    </div>
  </header>

  <details class="filter-panel" id="filterPanel" open>
    <summary>
      <span>Filters & Sort</span>
      <span class="filter-summary-sub" id="mobileFilterSummary">Tap to refine</span>
    </summary>
    <section class="filters" aria-label="Dashboard filters">
      <div class="field"><label for="search">Search</label><input id="search" placeholder="Sticker, team, player, verdict, notes, price" /></div>
      <div class="field"><label for="verdictFilter">Decision</label><select id="verdictFilter"><option value="">All decisions</option></select></div>
      <div class="field"><label for="variantFilter">Variant</label><select id="variantFilter"><option value="">All variants</option></select></div>
      <div class="field"><label for="typeFilter">Type</label><select id="typeFilter"><option value="">All types</option></select></div>
      <div class="field"><label for="categoryFilter">Category</label><select id="categoryFilter"><option value="">All categories</option></select></div>
      <div class="field"><label for="entryFilter">Entry</label><select id="entryFilter"><option value="">All entries</option></select></div>
      <div class="field"><label for="floodFilter">Flood</label><select id="floodFilter"><option value="">All flood levels</option></select></div>
      <div class="field"><label for="confidenceFilter">Confidence</label><select id="confidenceFilter"><option value="">Any</option><option value="0.35">35%+</option><option value="0.50">50%+</option><option value="0.70">70%+</option></select></div>
      <div class="field"><label for="priceMax">Max tokens</label><input id="priceMax" type="number" min="0" step="1" placeholder="Any" /></div>
      <div class="field"><label for="priceStateFilter">Price state</label><select id="priceStateFilter"><option value="">All</option><option value="current_low">Current low</option><option value="above_low">Above low</option></select></div>
      <div class="field"><label for="lowGapMax">Within low %</label><input id="lowGapMax" type="number" min="0" step="0.5" placeholder="5 or 10" /></div>
      <div class="field"><label for="sortPreset">Sort</label><select id="sortPreset"><option value="">Priority rank</option><option value="current_low">Current low first</option><option value="low_gap">Closest to low</option><option value="price_asc">Price low to high</option><option value="price_desc">Price high to low</option></select></div>
      <div class="field"><label for="rowLimit">Rows</label><select id="rowLimit"><option value="80">80 fastest</option><option value="120" selected>120 balanced</option><option value="200">200</option><option value="400">400</option><option value="0">All slower</option></select></div>
      <div class="field"><label for="scoredFilter">Scored</label><select id="scoredFilter"><option value="">All</option><option value="true">Scored</option><option value="false">Unscored</option></select></div>
      <div class="field"><label>&nbsp;</label><button id="resetBtn">Reset</button></div>
    </section>
  </details>

  <main class="content">
    <section class="panel">
      <div class="panel-head">
        <div><div class="panel-title">Priority Table</div><div class="hint">Click headers to sort. Hover over trend points to inspect token price, USD value, popularity and timestamp.</div></div>
        <div class="panel-tools">
          <div class="view-toggle" role="group" aria-label="View mode">
            <button class="view-btn active" id="listViewBtn" type="button" aria-pressed="true"><span class="view-icon list-icon"></span>List</button>
            <button class="view-btn" id="gridViewBtn" type="button" aria-pressed="false"><span class="view-icon grid-icon"></span>Grid</button>
          </div>
          <div class="grid-controls" id="gridControls" aria-label="Grid density">
            <label for="gridCols">Columns</label>
            <select id="gridCols">
              <option value="auto" selected>Auto fit</option>
              <option value="5">5 per row</option>
              <option value="10">10 per row</option>
              <option value="15">15 per row</option>
              <option value="custom">Custom</option>
            </select>
            <input id="gridCustomCols" type="number" min="1" max="24" step="1" placeholder="Custom" hidden />
          </div>
          <div class="hint" id="sortHint">Sorted by priority rank</div>
        </div>
      </div>
      <div class="table-wrap">
        <table id="table">
          <colgroup>
            <col class="rank-col" />
            <col class="sticker-col" />
            <col class="price-col" />
            <col class="decision-col" />
            <col class="edge-col" />
            <col class="market-col" />
            <col class="notes-col" />
          </colgroup>
          <thead>
            <tr>
              <th class="sortable" data-sort="priority_rank">Rank</th>
              <th class="sortable" data-sort="sticker">Sticker</th>
              <th class="sortable" data-sort="price_tokens">Price</th>
              <th class="sortable" data-sort="verdict">Decision</th>
              <th class="sortable" data-sort="expected_return_pct">Edge & Scores</th>
              <th class="sortable" data-sort="flood_risk_score">Market</th>
              <th>Notes</th>
            </tr>
          </thead>
          <tbody id="tbody"></tbody>
        </table>
      </div>
      <div id="gridView" class="grid-view" aria-label="Sticker grid" hidden></div>
      <div class="footer-note"><span id="renderHint">Rows are capped for smooth scrolling; all records remain searchable and sortable.</span> Generated files are written under <code>visualized/</code>.</div>
    </section>
  </main>
  <div id="sparkTip" class="spark-tip" role="tooltip"></div>
  <div id="detailModal" class="modal" hidden>
    <div class="modal-backdrop" data-close-modal></div>
    <div class="modal-dialog" role="dialog" aria-modal="true" aria-labelledby="detailTitle">
      <button class="modal-close" id="modalClose" type="button" aria-label="Close details">&times;</button>
      <div class="modal-content" id="modalContent"></div>
    </div>
  </div>
</div>

<script id="records-json" type="application/json">__DATA_JSON__</script>
<script id="series-json" type="application/json">__SERIES_JSON__</script>
<script>
const records = JSON.parse(document.getElementById('records-json').textContent);
const historySeries = JSON.parse(document.getElementById('series-json').textContent);
const verdictColors = __VERDICT_COLORS__;
const verdictOrder = __VERDICT_ORDER__;
let sortKey = 'priority_rank';
let sortDir = 1;
let filtered = [];
let viewMode = 'list';
const recordById = new Map(records.map(r => [String(r.sticker_id), r]));

const $ = (id) => document.getElementById(id);
const hasNum = (v) => v !== null && v !== undefined && v !== '' && Number.isFinite(Number(v));
const num = (v) => hasNum(v) ? Number(v) : null;
const fmt = (v, d=0) => hasNum(v) ? Number(v).toFixed(d) : '-';
const pct = (v, d=0) => hasNum(v) ? `${Number(v).toFixed(d)}%` : '-';
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const tokens = (v) => hasNum(v) ? Math.round(Number(v)).toLocaleString() : '-';
const money = (v) => hasNum(v) ? '$' + Number(v).toFixed(2) : '-';
function tokenUsdPair(tokenValue, r) {
  const historicalTokens = num(tokenValue);
  if (historicalTokens === null) return {tokens:'-', usd:'-'};
  const currentTokens = num(r.price_tokens);
  const currentUsd = num(r.usd_price);
  if (currentTokens === null || currentTokens <= 0 || currentUsd === null) {
    return {tokens:tokens(historicalTokens), usd:'-'};
  }
  return {
    tokens:tokens(historicalTokens),
    usd:money((historicalTokens / currentTokens) * currentUsd)
  };
}
function isReleaseLow(r) {
  if (r.current_low === true || r.current_low === 'true') return true;
  const currentTokens = num(r.price_tokens);
  const lowTokens = num(r.hist_min);
  return currentTokens !== null && lowTokens !== null && lowTokens > 0 && currentTokens <= lowTokens + 0.5;
}
function previousPriceToken(r, points=[]) {
  const clean = (points || [])
    .map(p => num(p.price))
    .filter(v => v !== null && v > 0);
  const current = num(r.price_tokens);
  if (clean.length >= 2) {
    const latest = clean[clean.length - 1];
    if (current !== null && Math.abs(latest - current) <= 0.5) {
      return clean[clean.length - 2];
    }
    return latest;
  }
  if (num(r.snapshot_prev_price) !== null) return r.snapshot_prev_price;
  return null;
}
function previousDeltaHtml(prevToken, r) {
  const prev = num(prevToken);
  const current = num(r.price_tokens);
  if (prev === null || current === null || prev <= 0) return '';
  const change = ((current - prev) / prev) * 100;
  const cls = change > 0.5 ? 'up' : change < -0.5 ? 'down' : '';
  const sign = change > 0 ? '+' : '';
  return `<span class="price-delta ${cls}">${sign}${fmt(change, 1)}%</span>`;
}
function priceRangeHtml(r, points=[]) {
  const prevToken = previousPriceToken(r, points);
  const previous = tokenUsdPair(prevToken, r);
  const low = tokenUsdPair(r.hist_min, r);
  const high = tokenUsdPair(r.hist_max, r);
  const previousChange = previousDeltaHtml(prevToken, r);
  const previousClass = previousChange.includes(' down') ? ' down' : previousChange.includes(' up') ? ' up' : '';
  return `<div class="price-range">
    <div class="price-range-row prev${previousClass}"><span>Prev</span><div><b>${esc(previous.usd)}${previousChange}</b><small>${esc(previous.tokens)} tokens before current</small></div></div>
    <div class="price-range-row low"><span>Low</span><div><b>${esc(low.usd)}</b><small>${esc(low.tokens)} tokens</small></div></div>
    <div class="price-range-row high"><span>High</span><div><b>${esc(high.usd)}</b><small>${esc(high.tokens)} tokens</small></div></div>
  </div>`;
}
function lowGapHtml(r) {
  if (isReleaseLow(r)) return '<div class="release-low-badge">Current low</div>';
  const gap = num(r.low_gap_pct);
  if (gap === null) return '';
  const cls = gap <= 5 ? ' near' : gap <= 10 ? ' mid' : '';
  const digits = gap < 10 ? 1 : 0;
  return `<div class="low-gap-badge${cls}">+${fmt(gap, digits)}% above low</div>`;
}
const colorForVerdict = (v) => verdictColors[v] || '#94a3b8';
const pctClass = (v) => !hasNum(v) ? 'flat' : Number(v) > 0.5 ? 'pos' : Number(v) < -0.5 ? 'neg' : 'flat';
const shortName = (s, n=25) => String(s || '').replace(/\s*\((Paper|Foil|Holo)\)/i, '').slice(0, n);

function uniqueValues(key) {
  return [...new Set(records.map(r => r[key]).filter(v => v !== null && v !== undefined && String(v).trim() !== ''))].sort((a,b) => String(a).localeCompare(String(b)));
}

function fillSelect(id, key) {
  const el = $(id);
  uniqueValues(key).forEach(v => {
    const opt = document.createElement('option');
    opt.value = v;
    opt.textContent = v;
    el.appendChild(opt);
  });
}

function makeOptions() {
  fillSelect('verdictFilter', 'verdict');
  fillSelect('variantFilter', 'variant');
  fillSelect('typeFilter', 'display_type');
  fillSelect('categoryFilter', 'category');
  fillSelect('entryFilter', 'entry_tier');
  fillSelect('floodFilter', 'flood_risk');
}

function withNorm(points) {
  const clean = (points || [])
    .map(p => ({...p, price:Number(p.price)}))
    .filter(p => Number.isFinite(p.price) && p.price > 0);
  if (!clean.length) return [];
  const first = clean[0].price || 1;
  return clean.map((p, i) => ({...p, i:i + 1, norm:hasNum(p.norm) ? Number(p.norm) : (p.price / first) * 100}));
}

function chartPointsFor(r, rawPoints) {
  const history = withNorm(rawPoints);
  if (history.length >= 2) return {points:history, source:'history', label:'history'};

  const current = num(r.price_tokens);
  if (current === null || current <= 0) return {points:[], source:'none', label:'no data'};

  const snapshotCount = Math.max(0, Math.round(num(r.snapshot_points) || 0));
  if (snapshotCount >= 2) {
    const count = Math.max(2, Math.min(7, snapshotCount));
    const change = num(r.snapshot_price_change_pct);
    let start = current;
    if (change !== null && change > -95 && Math.abs(change) > 0.05) {
      start = current / (1 + change / 100);
    }
    const points = Array.from({length:count}, (_, i) => {
      const t = count === 1 ? 1 : i / (count - 1);
      return {i:i + 1, price:start + (current - start) * t, synthetic:true};
    });
    return {points:withNorm(points), source:'snapshot', label:Math.abs(change || 0) <= 0.05 ? 'snapshot flat' : 'snapshot'};
  }

  const min = num(r.hist_min);
  const max = num(r.hist_max);
  if (min !== null && max !== null && max > 0 && Math.abs(max - min) > 0.001) {
    const points = [
      {i:1, price:max, synthetic:true},
      {i:2, price:Math.max(min, Math.min(max, current)), synthetic:true},
      {i:3, price:current, synthetic:true},
    ];
    return {points:withNorm(points), source:'range', label:'range'};
  }

  const discount = num(r.discount_from_high_pct);
  if (discount !== null && discount > 0.05 && discount < 98) {
    const prior = current / (1 - discount / 100);
    return {points:withNorm([{i:1, price:prior, synthetic:true}, {i:2, price:current, synthetic:true}]), source:'range', label:'range'};
  }

  return {points:withNorm([{i:1, price:current, synthetic:true}, {i:2, price:current, synthetic:true}]), source:'current', label:'current only'};
}

function sparkline(rawPoints, r, width=260, height=88) {
  const chart = chartPointsFor(r, rawPoints);
  const points = chart.points;
  if (!points || points.length < 2) {
    return `<svg class="spark" viewBox="0 0 ${width} ${height}"><text x="8" y="28" fill="#78869a" font-size="12">no chart data</text></svg>`;
  }
  const prices = points.map(p => Number(p.price)).filter(Number.isFinite);
  if (prices.length < 2) return `<svg class="spark" viewBox="0 0 ${width} ${height}"><text x="8" y="28" fill="#78869a" font-size="12">history pending</text></svg>`;
  const min = Math.min(...prices), max = Math.max(...prices), span = Math.max(max - min, 1e-9);
  const topPad = 16;
  const bottomPad = 14;
  const coords = points.map((p, i) => {
    const x = 9 + i * ((width - 18) / Math.max(points.length - 1, 1));
    const y = height - bottomPad - ((Number(p.price) - min)/span) * (height - topPad - bottomPad);
    return [x, y];
  });
  const line = coords.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ');
  const area = `${line} L${coords.at(-1)[0].toFixed(1)},${height - bottomPad} L${coords[0][0].toFixed(1)},${height - bottomPad} Z`;
  const up = coords.at(-1)[1] < coords[0][1];
  const stroke = up ? '#5ee592' : '#fb7185';
  const dash = chart.source === 'history' ? '' : ' stroke-dasharray="5 4" opacity=".82"';
  const label = chart.source === 'history' ? '' : `<text x="8" y="14" fill="#97a6b9" font-size="10">${esc(chart.label)}</text>`;
  const rangeLabels = span > 0.001
    ? `<text x="${width - 4}" y="${topPad - 4}" text-anchor="end" fill="#7f8da1" font-size="10">${tokens(max)}</text><text x="${width - 4}" y="${height - 3}" text-anchor="end" fill="#7f8da1" font-size="10">${tokens(min)}</text>`
    : `<text x="${width - 4}" y="${height - 3}" text-anchor="end" fill="#7f8da1" font-size="10">${tokens(max)}</text>`;
  const pointDots = coords.map(([x, y], i) => {
    const p = points[i];
    const tip = [
      `${r.sticker}`,
      `${tokens(p.price)} tokens${hasNum(p.usd) ? ` (${money(p.usd)})` : ''}`,
      hasNum(p.popularity) ? `Popularity: ${Number(p.popularity).toLocaleString()}` : '',
      p.time ? String(p.time).replace('T', ' ').replace('.000Z', 'Z') : '',
      chart.source === 'history' ? 'Collected history' : `${chart.label} fallback`
    ].filter(Boolean).join('\n');
    return `<circle class="spark-point" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="4.8" fill="${stroke}" stroke="#080d14" stroke-width="2" data-tip="${esc(tip)}"></circle>`;
  }).join('');
  return `<svg class="spark" viewBox="0 0 ${width} ${height}" aria-label="price trend">${label}${rangeLabels}<line class="spark-axis" x1="9" y1="${height - bottomPad}" x2="${width - 9}" y2="${height - bottomPad}"></line><path class="area" d="${area}" fill="${stroke}"></path><path class="line" d="${line}" stroke="${stroke}"${dash}></path>${pointDots}</svg>`;
}

function rowHtml(r) {
  const points = historySeries[r.sticker_id] || [];
  const link = r.item_url || '#';
  const vcolor = colorForVerdict(r.verdict);
  const expectedClass = pctClass(r.expected_return_pct);
  const demandClass = pctClass(r.demand_momentum_score);
  const changeValue = hasNum(r.snapshot_price_change_pct) ? r.snapshot_price_change_pct : r.recent_return_pct;
  const image = r.image_url || '';
  const typeLabel = r.display_type || r.category || '-';
  const atReleaseLow = isReleaseLow(r);
  return `<tr class="${atReleaseLow ? 'release-low-row' : ''}">
    <td data-label="Rank"><div class="rank">#${esc(r.priority_rank)}</div><div class="tier">${esc(r.priority_tier || '')}</div></td>
    <td data-label="Sticker">
      <div class="sticker-cell">
        <img class="thumb" src="${esc(image)}" loading="lazy" decoding="async" fetchpriority="low" onerror="this.style.visibility='hidden'" />
        <div>
          <a class="name" href="${esc(link)}" target="_blank" rel="noopener">${esc(r.sticker)}</a>
          <div class="meta">${esc(r.player_name || r.team_name || r.team || 'No team')} | ${esc(typeLabel)}</div>
          <div class="chips">
            <span class="chip">${esc(r.variant || '-')}</span>
            <span class="chip">${esc(r.category || '-')}</span>
            <span class="chip">${r.scored ? 'Scored' : 'Unscored'}</span>
          </div>
          <div class="actions">
            <a class="action primary" href="${esc(link)}" target="_blank" rel="noopener">CS2Tokens</a>
          </div>
        </div>
      </div>
    </td>
    <td data-label="Price">
      <div class="price-main">${money(r.usd_price)}</div>
      <div class="price-sub">${tokens(r.price_tokens)} tokens</div>
      ${lowGapHtml(r)}
      ${priceRangeHtml(r, points)}
      <div class="metric-list">
        <div class="metric-row"><span>Entry</span><b>${esc(r.entry_tier || '-')}</b></div>
      </div>
    </td>
    <td data-label="Decision">
      <span class="verdict" style="background:${vcolor}">${esc(r.verdict || '-')}</span>
      <div class="metric-list">
        <div class="metric-row"><span>Priority</span><b>${fmt(r.priority_score,1)}</b></div>
        <div class="metric-row"><span>Size</span><b>${esc(r.suggested_size || '-')}</b></div>
        <div class="metric-row"><span>Confidence</span><b>${fmt(r.prediction_confidence,2)}</b></div>
      </div>
    </td>
    <td data-label="Edge & Scores">
      <div class="metric-list" style="margin-top:0">
        <div class="metric-row"><span>Expected</span><b class="${expectedClass}">${pct(r.expected_return_pct,0)}</b></div>
        <div class="metric-row"><span>Value Edge</span><b>${fmt(r.value_edge_score,2)}</b></div>
        <div class="metric-row"><span>Quality</span><b>${fmt(r.quality_score,2)}</b></div>
        <div class="metric-row"><span>Score Conf.</span><b>${fmt(r.score_confidence,2)}</b></div>
        <div class="metric-row"><span>Manual</span><b>${fmt(r.manual_score_count,0)}</b></div>
      </div>
    </td>
    <td data-label="Market">
      <div class="metric-list" style="margin-top:0">
        <div class="metric-row"><span>Flood</span><b>${esc(r.flood_risk || '-')} (${fmt(r.flood_risk_score,2)})</b></div>
        <div class="metric-row"><span>Discount</span><b>${pct(r.discount_from_high_pct,0)}</b></div>
        <div class="metric-row"><span>Demand</span><b class="${demandClass}">${fmt(r.demand_momentum_score,2)}</b></div>
        <div class="metric-row"><span>Change</span><b class="${pctClass(changeValue)}">${pct(changeValue,1)}</b></div>
      </div>
      ${sparkline(points, r)}
    </td>
    <td data-label="Notes">
      <div class="note-block">
        <div><label>Reason</label><div>${esc(r.quick_reason || '-')}</div></div>
        <div><label>Risk</label><div>${esc(r.risk_note || '-')}</div></div>
        <div><label>Action</label><div class="note-action">${esc(r.action_note || '-')}</div></div>
      </div>
    </td>
  </tr>`;
}

function signalFor(value, low=false) {
  if (low) return 'low';
  const n = num(value);
  if (n === null) return 'watch';
  if (n > 0.5) return 'up';
  if (n < -0.5) return 'down';
  return 'watch';
}

function gridCardHtml(r) {
  const vcolor = colorForVerdict(r.verdict);
  const expectedClass = pctClass(r.expected_return_pct);
  const demandClass = pctClass(r.demand_momentum_score);
  const changeValue = hasNum(r.snapshot_price_change_pct) ? r.snapshot_price_change_pct : r.recent_return_pct;
  const changeClass = pctClass(changeValue);
  const atReleaseLow = isReleaseLow(r);
  const typeLabel = r.display_type || r.category || '-';
  const image = r.image_url || '';
  const id = String(r.sticker_id || r.sticker || r.priority_rank);
  return `<button class="grid-card ${atReleaseLow ? 'release-low-card' : ''}" type="button" data-id="${esc(id)}" aria-label="Open details for ${esc(r.sticker)}">
    <span class="grid-rank">#${esc(r.priority_rank)}</span>
    <span class="grid-tier">${esc(r.priority_tier || '')}</span>
    ${atReleaseLow ? '<span class="grid-low-ribbon" title="Current low"></span>' : ''}
    <span class="grid-image"><img src="${esc(image)}" loading="lazy" decoding="async" fetchpriority="low" onerror="this.style.visibility='hidden'" alt="${esc(r.sticker)}" /></span>
    <span class="grid-title">
      <span class="grid-name">${esc(r.sticker)}</span>
      <span class="grid-meta"><span class="grid-variant">${esc(r.variant || '-')}</span><span class="grid-team">${esc(r.player_name || r.team_name || r.team || typeLabel)}</span></span>
    </span>
    <span class="grid-bottom">
      <span class="grid-verdict">
        <span class="grid-verdict-pill" style="background:${vcolor}">${esc(r.verdict || '-')}</span>
        <span class="grid-price">${money(r.usd_price)}</span>
      </span>
      <span class="grid-kpis">
        <span class="grid-kpi"><small>Expected</small><b class="${expectedClass}"><span class="signal-dot ${signalFor(r.expected_return_pct, atReleaseLow)}"></span>${pct(r.expected_return_pct,0)}</b></span>
        <span class="grid-kpi"><small>Demand</small><b class="${demandClass}"><span class="signal-dot ${signalFor(r.demand_momentum_score)}"></span>${fmt(r.demand_momentum_score,2)}</b></span>
        <span class="grid-kpi"><small>Change</small><b class="${changeClass}"><span class="signal-dot ${signalFor(changeValue)}"></span>${pct(changeValue,1)}</b></span>
      </span>
    </span>
  </button>`;
}

function modalMetric(label, value, cls='') {
  return `<div class="metric-row"><span>${esc(label)}</span><b class="${cls}">${value}</b></div>`;
}

function stickerDetailsHtml(r) {
  const points = historySeries[r.sticker_id] || [];
  const vcolor = colorForVerdict(r.verdict);
  const expectedClass = pctClass(r.expected_return_pct);
  const demandClass = pctClass(r.demand_momentum_score);
  const changeValue = hasNum(r.snapshot_price_change_pct) ? r.snapshot_price_change_pct : r.recent_return_pct;
  const changeClass = pctClass(changeValue);
  const typeLabel = r.display_type || r.category || '-';
  const link = r.item_url || '#';
  const image = r.image_url || '';
  return `<div class="modal-grid">
    <div class="modal-visual">
      <span class="modal-rank">#${esc(r.priority_rank)} ${esc(r.priority_tier || '')}</span>
      <img src="${esc(image)}" alt="${esc(r.sticker)}" />
      <div class="actions"><a class="action primary" href="${esc(link)}" target="_blank" rel="noopener">Open CS2Tokens</a></div>
    </div>
    <div class="modal-main">
      <div class="modal-title-row">
        <h2 class="modal-title" id="detailTitle">${esc(r.sticker)}</h2>
        <span class="verdict" style="background:${vcolor}">${esc(r.verdict || '-')}</span>
      </div>
      <div class="modal-meta">${esc(r.player_name || r.team_name || r.team || 'No team')} | ${esc(typeLabel)} | ${esc(r.variant || '-')}</div>
      <div class="modal-price"><b>${money(r.usd_price)}</b><span>${tokens(r.price_tokens)} tokens</span></div>
      ${lowGapHtml(r)}
      ${priceRangeHtml(r, points)}
      <div class="modal-sections">
        <div class="modal-section">
          <h3>Decision</h3>
          <div class="metric-list" style="margin-top:0">
            ${modalMetric('Priority', fmt(r.priority_score,1))}
            ${modalMetric('Size', esc(r.suggested_size || '-'))}
            ${modalMetric('Confidence', fmt(r.prediction_confidence,2))}
            ${modalMetric('Entry', esc(r.entry_tier || '-'))}
          </div>
        </div>
        <div class="modal-section">
          <h3>Edge</h3>
          <div class="metric-list" style="margin-top:0">
            ${modalMetric('Expected', pct(r.expected_return_pct,0), expectedClass)}
            ${modalMetric('Value Edge', fmt(r.value_edge_score,2))}
            ${modalMetric('Quality', fmt(r.quality_score,2))}
            ${modalMetric('Manual Score', fmt(r.manual_score_count,0))}
          </div>
        </div>
        <div class="modal-section">
          <h3>Market</h3>
          <div class="metric-list" style="margin-top:0">
            ${modalMetric('Flood', `${esc(r.flood_risk || '-')} (${fmt(r.flood_risk_score,2)})`)}
            ${modalMetric('Discount', pct(r.discount_from_high_pct,0))}
            ${modalMetric('Demand', fmt(r.demand_momentum_score,2), demandClass)}
            ${modalMetric('Change', pct(changeValue,1), changeClass)}
          </div>
        </div>
        <div class="modal-section">
          <h3>Trend</h3>
          ${sparkline(points, r, 420, 118)}
        </div>
      </div>
      <div class="modal-section modal-note">
        <h3>Notes</h3>
        <div class="note-block">
          <div><label>Reason</label><div>${esc(r.quick_reason || '-')}</div></div>
          <div><label>Risk</label><div>${esc(r.risk_note || '-')}</div></div>
          <div><label>Action</label><div class="note-action">${esc(r.action_note || '-')}</div></div>
        </div>
      </div>
    </div>
  </div>`;
}

function applyFilters() {
  const q = $('search').value.trim().toLowerCase();
  const verdict = $('verdictFilter').value;
  const variant = $('variantFilter').value;
  const type = $('typeFilter').value;
  const category = $('categoryFilter').value;
  const entry = $('entryFilter').value;
  const flood = $('floodFilter').value;
  const scored = $('scoredFilter').value;
  const priceState = $('priceStateFilter').value;
  const sortPreset = $('sortPreset').value;
  const minConfidence = num($('confidenceFilter').value);
  const maxPrice = num($('priceMax').value);
  const maxLowGap = num($('lowGapMax').value);

  filtered = records.filter(r => {
    if (verdict && r.verdict !== verdict) return false;
    if (variant && r.variant !== variant) return false;
    if (type && r.display_type !== type) return false;
    if (category && r.category !== category) return false;
    if (entry && r.entry_tier !== entry) return false;
    if (flood && r.flood_risk !== flood) return false;
    if (scored && String(r.scored) !== scored) return false;
    if (priceState === 'current_low' && !isReleaseLow(r)) return false;
    if (priceState === 'above_low' && isReleaseLow(r)) return false;
    if (minConfidence !== null && (!hasNum(r.prediction_confidence) || Number(r.prediction_confidence) < minConfidence)) return false;
    if (maxPrice !== null && Number(r.price_tokens || 0) > maxPrice) return false;
    if (maxLowGap !== null) {
      const gap = num(r.low_gap_pct);
      if (gap === null) return false;
      if (!isReleaseLow(r) && gap > maxLowGap) return false;
    }
    if (q) {
      const hay = [
        r.sticker, r.team, r.team_name, r.player_name, r.variant, r.display_type, r.category,
        r.verdict, r.quick_reason, r.risk_note, r.action_note, r.notes, r.flood_risk,
        r.entry_tier, r.price_tokens, r.usd_price, r.market_hash_name
      ].join(' ').toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });
  applySortPreset(sortPreset);
  sortRows();
  renderResults();
  renderCharts();
}

function applySortPreset(sortPreset) {
  if (!sortPreset) return;
  if (sortPreset === 'current_low') {
    sortKey = 'current_low';
    sortDir = -1;
  } else if (sortPreset === 'low_gap') {
    sortKey = 'low_gap_pct';
    sortDir = 1;
  } else if (sortPreset === 'price_asc') {
    sortKey = 'price_tokens';
    sortDir = 1;
  } else if (sortPreset === 'price_desc') {
    sortKey = 'price_tokens';
    sortDir = -1;
  }
  $('sortHint').textContent = `Sorted by ${sortPreset.replace(/_/g, ' ')}`;
}

function compareValues(a, b) {
  if (sortKey === 'verdict') {
    return (verdictOrder[a.verdict] ?? 99) - (verdictOrder[b.verdict] ?? 99);
  }
  const av = a[sortKey], bv = b[sortKey];
  const an = num(av), bn = num(bv);
  if (an !== null || bn !== null) {
    if (an === null) return 1;
    if (bn === null) return -1;
    return an - bn;
  }
  return String(av ?? '').localeCompare(String(bv ?? ''));
}

function sortRows() {
  filtered.sort((a, b) => compareValues(a, b) * sortDir);
}

function displayedRows() {
  const limit = num($('rowLimit')?.value);
  if (limit === null || limit <= 0) return filtered;
  return filtered.slice(0, limit);
}

function gridColumnCount() {
  const mode = $('gridCols')?.value || 'auto';
  const custom = $('gridCustomCols');
  let count = mode === 'auto' ? (isMobileLayout() ? 2 : 5) : mode === 'custom' ? num(custom?.value) : num(mode);
  if (count === null) count = isMobileLayout() ? 2 : 5;
  return Math.max(1, Math.min(24, Math.round(count)));
}

function applyGridColumnSetting() {
  const grid = $('gridView');
  const custom = $('gridCustomCols');
  const controls = $('gridControls');
  if (!grid) return;
  const count = gridColumnCount();
  grid.style.setProperty('--grid-cols', String(count));
  grid.dataset.density = count >= 13 ? 'ultra' : count >= 9 ? 'dense' : 'normal';
  if (custom) custom.hidden = $('gridCols')?.value !== 'custom';
  if (controls) controls.classList.toggle('active', viewMode === 'grid');
}

function syncViewMode() {
  const isGrid = viewMode === 'grid';
  const tableWrap = document.querySelector('.table-wrap');
  const grid = $('gridView');
  if (tableWrap) tableWrap.hidden = isGrid;
  if (grid) grid.hidden = !isGrid;
  $('listViewBtn')?.classList.toggle('active', !isGrid);
  $('gridViewBtn')?.classList.toggle('active', isGrid);
  $('listViewBtn')?.setAttribute('aria-pressed', String(!isGrid));
  $('gridViewBtn')?.setAttribute('aria-pressed', String(isGrid));
  applyGridColumnSetting();
}

function renderGrid(rows) {
  const grid = $('gridView');
  if (!grid) return;
  applyGridColumnSetting();
  grid.innerHTML = rows.map(gridCardHtml).join('') || '<div class="grid-empty">No stickers match the active filters.</div>';
}

function renderResults() {
  const rows = displayedRows();
  syncViewMode();
  if (viewMode === 'grid') {
    $('tbody').innerHTML = '';
    renderGrid(rows);
  } else {
    const grid = $('gridView');
    if (grid) grid.innerHTML = '';
    $('tbody').innerHTML = rows.map(rowHtml).join('') || `<tr><td colspan="7" class="empty">No stickers match the active filters.</td></tr>`;
  }
  $('visibleCount').textContent = `${rows.length.toLocaleString()}/${filtered.length.toLocaleString()}`;
  $('totalCount').textContent = records.length.toLocaleString();
  const expected = filtered.map(r => num(r.expected_return_pct)).filter(v => v !== null);
  const edge = filtered.map(r => num(r.value_edge_score)).filter(v => v !== null);
  const scored = filtered.filter(r => r.scored).length;
  $('avgExpected').textContent = expected.length ? pct(expected.reduce((a,b) => a + b, 0) / expected.length, 0) : '-';
  $('avgEdge').textContent = edge.length ? fmt(edge.reduce((a,b) => a + b, 0) / edge.length, 2) : '-';
  $('scoredCount').textContent = `${scored}/${filtered.length}`;
  const hint = $('renderHint');
  if (hint) {
    hint.textContent = rows.length < filtered.length
      ? `Showing ${rows.length.toLocaleString()} of ${filtered.length.toLocaleString()} matched rows for smooth scrolling. Raise Rows or choose All only when needed.`
      : `Showing all ${filtered.length.toLocaleString()} matched rows.`;
  }
  updateMobileFilterSummary();
}

function scale(v, a, b, c, d) {
  if (!Number.isFinite(v)) return c;
  if (Math.abs(b - a) < 1e-9) return (c + d) / 2;
  return c + (v - a) * (d - c) / (b - a);
}

function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }

function opportunityChart() {
  const rows = filtered.filter(r => hasNum(r.expected_return_pct) && hasNum(r.flood_risk_score)).slice(0, 220);
  const W = 920, H = 420, L = 68, R = 26, T = 52, B = 54;
  if (!rows.length) return `<text x="28" y="48" fill="#98a6b8">No opportunity data for the active filters.</text>`;
  const xMin = -50, xMax = 350;
  const yMin = 0, yMax = 1;
  let grid = `<text class="chart-title" x="${L}" y="28">Expected return vs flood risk</text>`;
  for (let i = 0; i <= 5; i++) {
    const x = scale(i, 0, 5, L, W - R);
    const xv = Math.round(scale(i, 0, 5, xMin, xMax));
    grid += `<line class="gridline" x1="${x}" y1="${T}" x2="${x}" y2="${H-B}"></line><text x="${x}" y="${H-B+22}" text-anchor="middle" class="chart-note">${xv}%</text>`;
    const y = scale(i, 0, 5, H - B, T);
    const yv = scale(i, 0, 5, yMin, yMax).toFixed(1);
    grid += `<line class="gridline" x1="${L}" y1="${y}" x2="${W-R}" y2="${y}"></line><text x="${L-10}" y="${y+4}" text-anchor="end" class="chart-note">${yv}</text>`;
  }
  grid += `<line class="axis-line" x1="${L}" y1="${H-B}" x2="${W-R}" y2="${H-B}"></line><line class="axis-line" x1="${L}" y1="${T}" x2="${L}" y2="${H-B}"></line>`;
  grid += `<text x="${(W+L-R)/2}" y="${H-12}" text-anchor="middle" class="chart-note">Expected return, capped at 350%</text>`;
  grid += `<text x="18" y="${(H+T-B)/2}" transform="rotate(-90 18 ${(H+T-B)/2})" text-anchor="middle" class="chart-note">Flood risk score</text>`;
  let body = '';
  rows.forEach(r => {
    const x = scale(clamp(Number(r.expected_return_pct), xMin, xMax), xMin, xMax, L, W - R);
    const y = scale(Number(r.flood_risk_score), yMin, yMax, H - B, T);
    const radius = 5 + Math.min(15, Math.max(0, Number(r.priority_score || 0)) / 8);
    const color = colorForVerdict(r.verdict);
    body += `<circle cx="${x}" cy="${y}" r="${radius.toFixed(1)}" fill="${color}" opacity=".78" stroke="#07101b" stroke-width="2"><title>#${r.priority_rank} ${esc(r.sticker)}\n${pct(r.expected_return_pct,0)} expected\nFlood ${fmt(r.flood_risk_score,2)}\n${tokens(r.price_tokens)} tokens</title></circle>`;
  });
  rows.slice(0, 24).forEach(r => {
    const x = scale(clamp(Number(r.expected_return_pct), xMin, xMax), xMin, xMax, L, W - R);
    const y = scale(Number(r.flood_risk_score), yMin, yMax, H - B, T);
    body += `<text class="point-label" x="${x + 10}" y="${y + 4}">#${r.priority_rank} ${esc(shortName(r.sticker, 19))}</text>`;
  });
  return grid + body;
}

function variantChart() {
  const variants = uniqueValuesFrom(filtered, 'variant');
  const W = 920, H = 420, L = 74, R = 28, T = 48, B = 58;
  if (!variants.length) return `<text x="28" y="48" fill="#98a6b8">No variants match the active filters.</text>`;
  const verdicts = Object.keys(verdictColors).filter(v => filtered.some(r => r.verdict === v));
  const totals = variants.map(v => filtered.filter(r => r.variant === v).length);
  const maxTotal = Math.max(...totals, 1);
  let out = `<text class="chart-title" x="${L}" y="28">Decision mix by variant</text>`;
  for (let i = 0; i <= 4; i++) {
    const x = scale(i, 0, 4, L, W - R);
    const val = Math.round(scale(i, 0, 4, 0, maxTotal));
    out += `<line class="gridline" x1="${x}" y1="${T}" x2="${x}" y2="${H-B}"></line><text x="${x}" y="${H-B+22}" text-anchor="middle" class="chart-note">${val}</text>`;
  }
  const barH = Math.min(62, (H - T - B) / Math.max(variants.length, 1) - 16);
  variants.forEach((variant, idx) => {
    const y = T + idx * ((H - T - B) / variants.length) + 8;
    out += `<text x="${L-12}" y="${y + barH/2 + 4}" text-anchor="end" class="chart-note">${esc(variant)}</text>`;
    let x0 = L;
    verdicts.forEach(verdict => {
      const count = filtered.filter(r => r.variant === variant && r.verdict === verdict).length;
      if (!count) return;
      const w = scale(count, 0, maxTotal, 0, W - L - R);
      out += `<rect x="${x0}" y="${y}" width="${w}" height="${barH}" fill="${colorForVerdict(verdict)}" opacity=".86"><title>${esc(variant)} - ${esc(verdict)}: ${count}</title></rect>`;
      x0 += w;
    });
    out += `<text x="${x0 + 8}" y="${y + barH/2 + 4}" class="chart-note">${totals[idx]}</text>`;
  });
  out += `<text x="${(W+L-R)/2}" y="${H-12}" text-anchor="middle" class="chart-note">Sticker count</text>`;
  return out;
}

function uniqueValuesFrom(rows, key) {
  return [...new Set(rows.map(r => r[key]).filter(v => v !== null && v !== undefined && String(v).trim() !== ''))].sort((a,b) => String(a).localeCompare(String(b)));
}

function movementChart() {
  const n = Number($('topN').value) || 16;
  const mode = $('movementScale').value;
  const rows = filtered.slice(0, n)
    .map(r => ({r, chart:chartPointsFor(r, historySeries[r.sticker_id] || [])}))
    .filter(item => item.chart.points.length >= 2);
  const W = 1320, H = 500, L = 72, R = 300, T = 54, B = 60;
  if (!rows.length) return `<text x="28" y="48" fill="#98a6b8">No chart data for the active filters yet.</text>`;
  const values = [];
  rows.forEach(item => item.chart.points.forEach(p => values.push(mode === 'price' ? Number(p.price) : Number(p.norm))));
  let min = Math.min(...values), max = Math.max(...values);
  if (mode === 'normalized') { min = Math.min(70, min); max = Math.max(145, max); }
  const fallbackCount = rows.filter(item => item.chart.source !== 'history').length;
  let out = `<text class="chart-title" x="${L}" y="30">Top ${rows.length} priority movement - ${mode === 'price' ? 'token price' : 'normalized first point = 100'}</text>`;
  if (fallbackCount) out += `<text x="${L + 520}" y="30" class="chart-note">dashed = snapshot/range fallback (${fallbackCount})</text>`;
  for (let i = 0; i <= 5; i++) {
    const y = scale(i, 0, 5, H - B, T);
    const val = scale(i, 0, 5, min, max).toFixed(0);
    out += `<line class="gridline" x1="${L}" y1="${y}" x2="${W-R}" y2="${y}"></line><text x="${L-10}" y="${y+4}" text-anchor="end" class="chart-note">${val}</text>`;
  }
  out += `<line class="axis-line" x1="${L}" y1="${H-B}" x2="${W-R}" y2="${H-B}"></line><line class="axis-line" x1="${L}" y1="${T}" x2="${L}" y2="${H-B}"></line>`;
  rows.forEach(({r, chart}) => {
    const pts = chart.points;
    const vals = pts.map(p => mode === 'price' ? Number(p.price) : Number(p.norm));
    const xs = vals.map((_, i) => scale(i, 0, Math.max(vals.length - 1, 1), L, W - R));
    const ys = vals.map(v => scale(v, min, max, H - B, T));
    const color = colorForVerdict(r.verdict);
    const d = xs.map((x, i) => `${i ? 'L' : 'M'}${x.toFixed(1)} ${ys[i].toFixed(1)}`).join(' ');
    const dash = chart.source === 'history' ? '' : ' stroke-dasharray="8 6" opacity=".62"';
    out += `<path d="${d}" fill="none" stroke="${color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"${dash}></path>`;
    out += `<circle cx="${xs.at(-1)}" cy="${ys.at(-1)}" r="5.5" fill="${color}" stroke="#07101b" stroke-width="2"><title>#${r.priority_rank} ${esc(r.sticker)}</title></circle>`;
    out += `<text class="point-label" x="${xs.at(-1)+10}" y="${ys.at(-1)+4}">#${r.priority_rank} ${esc(shortName(r.sticker, 27))}</text>`;
  });
  out += `<text x="${(W-R+L)/2}" y="${H-14}" text-anchor="middle" class="chart-note">Solid lines are collected history. Dashed lines are snapshot or range fallback when full history is missing.</text>`;
  return out;
}

function renderCharts() {
  const opportunity = $('opportunityPlot');
  const variant = $('variantPlot');
  const movement = $('movementPlot');
  if (opportunity) opportunity.innerHTML = opportunityChart();
  if (variant) variant.innerHTML = variantChart();
  if (movement) movement.innerHTML = movementChart();
}

function setupSparkTooltip() {
  const tip = $('sparkTip');
  if (!tip) return;

  document.addEventListener('mousemove', event => {
    const target = event.target && event.target.closest ? event.target.closest('.spark-point') : null;
    if (!target) {
      tip.style.display = 'none';
      return;
    }

    tip.textContent = target.dataset.tip || '';
    tip.style.display = 'block';

    const pad = 14;
    const rect = tip.getBoundingClientRect();
    let left = event.clientX + pad;
    let top = event.clientY + pad;

    if (left + rect.width > window.innerWidth - 8) left = event.clientX - rect.width - pad;
    if (top + rect.height > window.innerHeight - 8) top = event.clientY - rect.height - pad;

    tip.style.left = `${Math.max(8, left)}px`;
    tip.style.top = `${Math.max(8, top)}px`;
  });

  document.addEventListener('mouseleave', () => {
    tip.style.display = 'none';
  });
}

function isMobileLayout() {
  return window.matchMedia && window.matchMedia('(max-width: 800px)').matches;
}

function setupFilterPanel() {
  const panel = document.getElementById('filterPanel');
  if (!panel) return;
  if (isMobileLayout() && panel.dataset.mobileReady !== '1') {
    panel.removeAttribute('open');
    panel.dataset.mobileReady = '1';
  }
  window.addEventListener('resize', () => {
    if (!isMobileLayout()) panel.setAttribute('open', '');
  }, {passive:true});
}

function updateMobileFilterSummary() {
  const summary = document.getElementById('mobileFilterSummary');
  if (!summary) return;
  const ids = ['search','verdictFilter','variantFilter','typeFilter','categoryFilter','entryFilter','floodFilter','confidenceFilter','priceMax','priceStateFilter','lowGapMax','sortPreset','scoredFilter'];
  const active = ids.reduce((count, id) => {
    const el = document.getElementById(id);
    return count + (el && String(el.value || '').trim() ? 1 : 0);
  }, 0);
  summary.textContent = active
    ? `${active} active - ${filtered.length.toLocaleString()} matches`
    : 'Tap to refine';
}

function setViewMode(mode) {
  viewMode = mode === 'grid' ? 'grid' : 'list';
  renderResults();
}

function openStickerModal(id) {
  const r = recordById.get(String(id));
  const modal = $('detailModal');
  const content = $('modalContent');
  if (!r || !modal || !content) return;
  content.innerHTML = stickerDetailsHtml(r);
  modal.hidden = false;
  document.body.style.overflow = 'hidden';
  $('modalClose')?.focus({preventScroll:true});
}

function closeStickerModal() {
  const modal = $('detailModal');
  if (!modal || modal.hidden) return;
  modal.hidden = true;
  document.body.style.overflow = '';
  const content = $('modalContent');
  if (content) content.innerHTML = '';
}

function setupDetailModal() {
  const grid = $('gridView');
  if (grid) {
    grid.addEventListener('click', event => {
      const card = event.target && event.target.closest ? event.target.closest('.grid-card') : null;
      if (card) openStickerModal(card.dataset.id);
    });
  }
  $('modalClose')?.addEventListener('click', closeStickerModal);
  document.querySelector('[data-close-modal]')?.addEventListener('click', closeStickerModal);
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape') closeStickerModal();
  });
}

function wire() {
  makeOptions();
  setupFilterPanel();
  ['search','verdictFilter','variantFilter','typeFilter','categoryFilter','entryFilter','floodFilter','confidenceFilter','priceMax','priceStateFilter','lowGapMax','sortPreset','rowLimit','scoredFilter']
    .forEach(id => $(id).addEventListener('input', applyFilters));
  $('listViewBtn')?.addEventListener('click', () => setViewMode('list'));
  $('gridViewBtn')?.addEventListener('click', () => setViewMode('grid'));
  $('gridCols')?.addEventListener('input', () => {
    applyGridColumnSetting();
    if (viewMode === 'grid') renderResults();
  });
  $('gridCustomCols')?.addEventListener('input', () => {
    applyGridColumnSetting();
    if (viewMode === 'grid') renderResults();
  });
  $('resetBtn').addEventListener('click', () => {
    ['search','verdictFilter','variantFilter','typeFilter','categoryFilter','entryFilter','floodFilter','confidenceFilter','priceMax','priceStateFilter','lowGapMax','sortPreset','scoredFilter']
      .forEach(id => $(id).value = '');
    $('rowLimit').value = '120';
    viewMode = 'list';
    $('gridCols').value = 'auto';
    $('gridCustomCols').value = '';
    sortKey = 'priority_rank';
    sortDir = 1;
    $('sortHint').textContent = 'Sorted by priority rank';
    applyFilters();
  });
  document.querySelectorAll('th.sortable').forEach(th => th.addEventListener('click', () => {
    const key = th.dataset.sort;
    $('sortPreset').value = '';
    if (sortKey === key) sortDir *= -1;
    else {
      sortKey = key;
      sortDir = ['priority_score','expected_return_pct','quality_score','value_edge_score'].includes(key) ? -1 : 1;
    }
    $('sortHint').textContent = `Sorted by ${key} ${sortDir === 1 ? 'ascending' : 'descending'}`;
    applyFilters();
  }));
  setupSparkTooltip();
  setupDetailModal();
  applyFilters();
}

wire();
</script>
</body>
</html>"""
    return (
        template
        .replace("__DATA_JSON__", data_json)
        .replace("__SERIES_JSON__", series_json)
        .replace("__VERDICT_COLORS__", json.dumps(VERDICT_COLORS))
        .replace("__VERDICT_ORDER__", json.dumps(VERDICT_ORDER))
    )


def main() -> None:
    analysis = load_analysis()
    history = load_history()
    series = build_history_series(analysis, history)
    records = [row_to_record(row) for _, row in analysis.iterrows()]
    write_priority_csv(analysis)
    html_text = build_html(records, series)
    out_path = OUT_DIR / "sticker_dashboard.html"
    out_path.write_text(html_text, encoding="utf-8")
    print(f"Dashboard written to {out_path}")
    print(f"Rows: {len(records)} | history series: {len(series)}")


if __name__ == "__main__":
    main()
