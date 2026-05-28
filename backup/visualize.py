from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
import json
import math
import html
import urllib.parse
import urllib.request
import pandas as pd
import numpy as np

DATA_DIR = Path("data")
ANALYZE_DIR = Path("analyze")
OUT_DIR = Path("visualized")
OUT_DIR.mkdir(parents=True, exist_ok=True)
STEAM_INSPECT_BASE = "steam://rungame/730/76561202255233023/+csgo_econ_action_preview%20"
STEAM_PREVIEW_CACHE = OUT_DIR / "steam_preview_cache.json"
STEAM_PREVIEW_CACHE_VERSION = 2
STEAM_FETCH_WORKERS = 12
STEAM_FETCH_TIMEOUT = 12

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
    return pd.read_csv(path, encoding="utf-8-sig")


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


def is_qwik_ref(value: str, max_index: int) -> bool:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-z]+", value):
        return False
    try:
        return int(value, 36) < max_index
    except ValueError:
        return False


def decode_qwik_data(value, objs: list, seen: set[int] | None = None):
    if seen is None:
        seen = set()
    if isinstance(value, str) and is_qwik_ref(value, len(objs)):
        index = int(value, 36)
        if index in seen:
            return value
        return decode_qwik_data(objs[index], objs, seen | {index})
    if isinstance(value, list):
        return [decode_qwik_data(item, objs, seen) for item in value]
    if isinstance(value, dict):
        return {key: decode_qwik_data(item, objs, seen) for key, item in value.items()}
    return value


def walk_dicts(value):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from walk_dicts(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_dicts(item)


def steam_market_search_url(name: str) -> str:
    query = urllib.parse.quote(str(name or ""), safe="")
    return f"steam://openurl/https://steamcommunity.com/market/search?appid=730&q={query}"


def steam_market_listing_url(market_hash_name: str) -> str:
    encoded = urllib.parse.quote(str(market_hash_name or ""), safe="")
    return f"steam://openurl/https://steamcommunity.com/market/listings/730/{encoded}"


def steam_open_url(url: str) -> str:
    url = str(url or "").strip()
    if not url:
        return ""
    if url.startswith("steam://"):
        return url
    if url.startswith("https://steamcommunity.com/"):
        return f"steam://openurl/{url}"
    return url


def fetch_preview_metadata(item_url: str) -> dict:
    qdata_url = item_url.rstrip("/") + "/q-data.json"
    result = {
        "version": STEAM_PREVIEW_CACHE_VERSION,
        "item_url": item_url,
        "qdata_url": qdata_url,
        "status": "missing",
        "steam_preview_url": "",
        "market_hash_name": "",
        "defindex": None,
        "paint_index": None,
        "rarity_id": "",
    }
    try:
        request = urllib.request.Request(qdata_url, headers={"User-Agent": "cs2-sticker-dashboard/1.0"})
        with urllib.request.urlopen(request, timeout=STEAM_FETCH_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
        objs = payload.get("_objs", [])
        decoded = decode_qwik_data(payload, objs)
        metadata = None
        for item in walk_dicts(decoded):
            if "defindex" in item and "paint_index" in item:
                metadata = item
                break
        catalog = metadata.get("catalog", {}) if isinstance(metadata, dict) else {}
        defindex = safe_float(metadata.get("defindex"), None) if metadata else None
        paintindex = safe_float(metadata.get("paint_index"), None) if metadata else None
        rarity_id = str(catalog.get("rarity_id", "") or "")
        market_hash_name = str(catalog.get("market_hash_name", "") or catalog.get("name", "") or "")
        if not market_hash_name:
            return result | {"status": "metadata_missing", "market_hash_name": market_hash_name, "rarity_id": rarity_id}

        return result | {
            "status": "steam_page",
            "steam_preview_url": steam_market_listing_url(market_hash_name),
            "market_hash_name": market_hash_name,
            "defindex": int(defindex) if defindex is not None else None,
            "paint_index": int(paintindex) if paintindex is not None else None,
            "rarity_id": rarity_id,
        }
    except Exception as exc:
        return result | {"status": "fetch_failed", "error": str(exc)}


def load_steam_preview_cache() -> dict:
    if not STEAM_PREVIEW_CACHE.exists():
        return {}
    try:
        data = json.loads(STEAM_PREVIEW_CACHE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_steam_preview_cache(cache: dict) -> None:
    STEAM_PREVIEW_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def enrich_steam_preview_links(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "market_hash_name" in df.columns or "steam_market_url" in df.columns:
        if "market_hash_name" in df.columns:
            df["steam_market_hash_name"] = df["market_hash_name"].fillna("").astype(str)
        else:
            source_names = df["sticker"] if "sticker" in df.columns else pd.Series([""] * len(df), index=df.index)
            df["steam_market_hash_name"] = source_names.map(lambda name: f"Sticker | {name} | Cologne 2026")
        if "steam_market_url" in df.columns:
            source_url = df["steam_market_url"].fillna("").astype(str)
            fallback = df["steam_market_hash_name"].map(steam_market_listing_url)
            df["steam_preview_url"] = [
                steam_open_url(url) if url else fallback_url
                for url, fallback_url in zip(source_url, fallback)
            ]
        else:
            df["steam_preview_url"] = df["steam_market_hash_name"].map(steam_market_listing_url)
            df["steam_market_url"] = df["steam_preview_url"]
        if "steam_preview_status" not in df.columns:
            df["steam_preview_status"] = "market_listing"
        return df

    if "item_url" not in df.columns:
        df["steam_preview_url"] = ""
        df["steam_preview_status"] = "missing"
        source_names = df["sticker"] if "sticker" in df.columns else pd.Series([""] * len(df), index=df.index)
        df["steam_market_url"] = source_names.map(steam_market_search_url)
        return df

    cache = load_steam_preview_cache()
    item_urls = sorted({
        str(value).strip()
        for value in df["item_url"].fillna("")
        if str(value).strip().startswith("https://cs2tokens.com/")
    })
    missing = [
        url for url in item_urls
        if cache.get(url, {}).get("version") != STEAM_PREVIEW_CACHE_VERSION
        or not cache.get(url, {}).get("steam_preview_url")
    ]
    if missing:
        print(f"Fetching Steam preview metadata: {len(missing)} missing")
        with ThreadPoolExecutor(max_workers=STEAM_FETCH_WORKERS) as executor:
            futures = {executor.submit(fetch_preview_metadata, url): url for url in missing}
            for future in as_completed(futures):
                cache[futures[future]] = future.result()
        write_steam_preview_cache(cache)

    def cached_value(row: pd.Series, key: str, default=""):
        item_url = str(row.get("item_url", "") or "")
        return cache.get(item_url, {}).get(key, default)

    df["steam_preview_url"] = df.apply(lambda row: cached_value(row, "steam_preview_url", ""), axis=1)
    df["steam_preview_status"] = df.apply(lambda row: cached_value(row, "status", "missing"), axis=1)
    df["steam_market_hash_name"] = df.apply(
        lambda row: cached_value(row, "market_hash_name", "") or f"Sticker | {row.get('sticker', '')} | Cologne 2026",
        axis=1,
    )
    df["steam_market_url"] = df["steam_market_hash_name"].map(steam_market_search_url)
    return df


def short_text(value: str, limit: int = 76) -> str:
    text = str(value or "").strip()
    text = " | ".join([part.strip() for part in text.split("|") if part.strip()])
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def make_steam_url(row: pd.Series) -> str:
    for key in ["inspect_url", "inspect_link", "steam_url", "steam_preview_url"]:
        value = row.get(key, "")
        if isinstance(value, str) and (value.startswith("steam://") or value.startswith("https://steamcommunity.com/")):
            return steam_open_url(value)
    market_url = row.get("steam_market_url", "")
    if isinstance(market_url, str) and (market_url.startswith("steam://") or market_url.startswith("https://steamcommunity.com/")):
        return steam_open_url(market_url)
    return steam_market_search_url(str(row.get("sticker", "")))


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
                merge_key, "image_url", "item_url", "recent_return_pct", "hist_min", "hist_max",
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
        "positive_popularity_sum", "hist_min", "hist_max", "hist_points", "recent_return_pct",
        "value_edge_score", "expected_return_pct", "expected_return_score", "robust_reference_price",
        "robust_peak_price", "discount_from_robust_peak_pct", "downside_to_floor_pct",
        "downside_risk_score", "demand_momentum_score", "demand_price_divergence_score",
        "falling_demand_penalty", "prediction_confidence", "score_confidence", "manual_score_count",
        "history_coverage_score", "entry_change_score", "snapshot_price_change_pct",
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
            point_time = p.get("point_time", "")
            if pd.isna(point_time):
                point_time = p.get("tooltip_time_raw", "")
            points.append({
                "i": i,
                "price": token,
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
        "price_tokens": safe_float(val("price_tokens"), 0) or 0,
        "usd_price": safe_float(val("usd_price"), 0) or 0,
        "recent_return_pct": safe_float(val("recent_return_pct"), None),
        "hist_min": safe_float(val("hist_min"), None),
        "hist_max": safe_float(val("hist_max"), None),
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
        "steam_url": make_steam_url(row),
        "steam_preview_status": str(val("steam_preview_status", "")),
        "steam_market_hash_name": str(val("steam_market_hash_name", "")),
        "market_hash_name": str(val("market_hash_name", "")),
        "metadata_status": str(val("metadata_status", "")),
        "sticker_id": str(val("sticker_id", "")),
    }


def write_priority_csv(df: pd.DataFrame) -> None:
    cols = [
        "priority_rank", "priority_score", "priority_tier", "verdict", "sticker", "category", "variant",
        "sticker_type", "catalog_type", "player_name", "team", "team_name",
        "price_tokens", "usd_price", "suggested_size", "entry_tier", "flood_risk",
        "hist_min", "hist_max", "hist_points", "history_span_hours", "snapshot_points",
        "quality_score", "history_score", "decision_score", "trend_score", "value_edge_score",
        "expected_return_pct", "demand_momentum_score", "demand_price_divergence_score",
        "prediction_confidence", "score_confidence", "quick_reason", "risk_note", "action_note",
        "item_url", "image_url", "steam_preview_url", "steam_preview_status", "steam_market_hash_name",
        "market_hash_name", "steam_market_url", "metadata_status",
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
  renderCharts();
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
  .filters {
    position:sticky;
    top:0;
    z-index:30;
    display:grid;
    grid-template-columns:minmax(260px,1.25fr) repeat(6, minmax(130px,.6fr)) 120px 112px;
    gap:10px;
    margin:12px 0;
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
  .table-wrap { max-height:82vh; overflow:auto; }
  table { width:100%; border-collapse:separate; border-spacing:0; table-layout:fixed; }
  col.rank-col { width:64px; }
  col.sticker-col { width:28%; }
  col.price-col { width:9%; }
  col.decision-col { width:15%; }
  col.edge-col { width:16%; }
  col.market-col { width:17%; }
  col.notes-col { width:15%; }
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
  tbody tr { background:#101722; }
  tbody tr:nth-child(even) { background:#0e1520; }
  tbody tr:hover { background:#162233; }
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
  }
  .name { display:inline; font-size:16px; line-height:1.25; font-weight:900; color:#fff; }
  .name:hover { text-decoration:underline; text-decoration-thickness:1px; }
  .meta { margin-top:7px; color:var(--muted); font-size:12px; line-height:1.4; }
  .chips { display:flex; flex-wrap:wrap; gap:6px; margin-top:9px; }
  .chip { display:inline-flex; align-items:center; max-width:100%; padding:4px 7px; border:1px solid var(--line); border-radius:999px; background:rgba(13,20,32,.8); color:#d8e1ee; font-size:12px; line-height:1.2; }
  .actions { display:flex; flex-wrap:wrap; gap:7px; margin-top:10px; }
  .action { display:inline-flex; align-items:center; justify-content:center; min-height:30px; padding:6px 9px; border:1px solid var(--line); border-radius:6px; font-weight:800; font-size:12px; }
  .action.primary { color:#c9dbff; border-color:rgba(91,140,255,.58); }
  .action.steam { color:#a8f3c8; border-color:rgba(52,211,153,.52); }
  .price-main { font-size:30px; line-height:1; font-weight:950; font-variant-numeric:tabular-nums; color:#fff; }
  .price-sub { margin-top:6px; color:var(--muted); font-size:14px; font-weight:800; font-variant-numeric:tabular-nums; }
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
  .note-block { display:grid; gap:9px; line-height:1.45; font-size:13px; color:#d8e1ee; }
  .note-block label { display:block; color:var(--muted); font-size:11px; margin-bottom:2px; }
  .note-action { color:#f7df9d; }
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
    .app { width:calc(100vw - 18px); padding-top:9px; }
    .filters { grid-template-columns:1fr 1fr; }
    .stats { grid-template-columns:1fr 1fr; }
    .sticker-cell { grid-template-columns:142px minmax(0,1fr); }
    .thumb { width:142px; height:142px; }
  }
</style>
</head>
<body>
<div class="app">
  <header class="topbar">
    <div>
      <h1>CS2 Sticker Decision Dashboard</h1>
      <p class="sub">Analyzer output with Paper, Foil and Holo coverage. Use the filters to isolate a variant or sticker type, then sort by price, expected return, confidence, flood risk or manual score.</p>
    </div>
    <div class="stats">
      <div class="stat"><span>Visible</span><b id="visibleCount">0</b></div>
      <div class="stat"><span>Total</span><b id="totalCount">0</b></div>
      <div class="stat"><span>Avg Expected</span><b id="avgExpected">0%</b></div>
      <div class="stat"><span>Avg Edge</span><b id="avgEdge">0.00</b></div>
      <div class="stat"><span>Scored</span><b id="scoredCount">0</b></div>
    </div>
  </header>

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
    <div class="field"><label for="scoredFilter">Scored</label><select id="scoredFilter"><option value="">All</option><option value="true">Scored</option><option value="false">Unscored</option></select></div>
    <div class="field"><label>&nbsp;</label><button id="resetBtn">Reset</button></div>
  </section>

  <main class="content">
    <section class="chart-grid">
      <section class="panel">
        <div class="panel-head">
          <div><div class="panel-title">Opportunity Map</div><div class="hint">Expected return versus flood risk. Lower flood and higher expected return is the preferred zone.</div></div>
        </div>
        <div class="chart-body">
          <div class="chart-frame"><svg id="opportunityPlot" class="chart" viewBox="0 0 920 420"></svg></div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div><div class="panel-title">Variant Decision Mix</div><div class="hint">Counts after the active filters, grouped by Paper, Foil and Holo.</div></div>
        </div>
        <div class="chart-body">
          <div class="legend" id="legend"></div>
          <div class="chart-frame" style="margin-top:10px"><svg id="variantPlot" class="chart" viewBox="0 0 920 420"></svg></div>
        </div>
      </section>

      <section class="panel" style="grid-column:1 / -1">
        <div class="panel-head">
          <div><div class="panel-title">Top Priority Movement</div><div class="hint">Uses collected history points. If a variant only has one point so far, it will appear after more collections.</div></div>
          <div class="hint" id="sortHint">Sorted by priority rank</div>
        </div>
        <div class="chart-controls">
          <label>Top N <select id="topN" class="inline-select"><option>10</option><option selected>16</option><option>24</option><option>32</option></select></label>
          <label>Scale <select id="movementScale" class="inline-select"><option value="normalized" selected>Normalized</option><option value="price">Token price</option></select></label>
          <button id="rerenderCharts">Update charts</button>
        </div>
        <div class="chart-body" style="padding-top:0">
          <div class="chart-frame tall"><svg id="movementPlot" class="chart tall" viewBox="0 0 1320 500"></svg></div>
        </div>
      </section>
    </section>

    <section class="panel">
      <div class="panel-head">
        <div><div class="panel-title">Priority Table</div><div class="hint">Click headers to sort. Sticker buttons stay beside the image so actions do not require horizontal scrolling.</div></div>
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
      <div class="footer-note">Generated files are written under <code>visualized/</code>. Steam Preview opens the exact Steam Community Market listing from collector metadata when available.</div>
    </section>
  </main>
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

const $ = (id) => document.getElementById(id);
const hasNum = (v) => v !== null && v !== undefined && v !== '' && Number.isFinite(Number(v));
const num = (v) => hasNum(v) ? Number(v) : null;
const fmt = (v, d=0) => hasNum(v) ? Number(v).toFixed(d) : '-';
const pct = (v, d=0) => hasNum(v) ? `${Number(v).toFixed(d)}%` : '-';
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const tokens = (v) => hasNum(v) ? Math.round(Number(v)).toLocaleString() : '-';
const money = (v) => hasNum(v) ? '$' + Number(v).toFixed(2) : '-';
function rangeMoney(tokenValue, r) {
  const currentTokens = num(r.price_tokens);
  const currentUsd = num(r.usd_price);
  const historicalTokens = num(tokenValue);
  if (historicalTokens === null) return '-';
  if (currentTokens === null || currentTokens <= 0 || currentUsd === null) return `${tokens(historicalTokens)} tokens`;
  return `${tokens(historicalTokens)} / ${money((historicalTokens / currentTokens) * currentUsd)}`;
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
  const coords = points.map((p, i) => {
    const x = 7 + i * ((width - 14) / Math.max(points.length - 1, 1));
    const y = height - 8 - ((Number(p.price) - min) / span) * (height - 18);
    return [x, y];
  });
  const line = coords.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ');
  const area = `${line} L${coords.at(-1)[0].toFixed(1)},${height - 6} L${coords[0][0].toFixed(1)},${height - 6} Z`;
  const up = coords.at(-1)[1] < coords[0][1];
  const stroke = up ? '#5ee592' : '#fb7185';
  const dash = chart.source === 'history' ? '' : ' stroke-dasharray="5 4" opacity=".82"';
  const label = chart.source === 'history' ? '' : `<text x="8" y="14" fill="#97a6b9" font-size="10">${esc(chart.label)}</text>`;
  return `<svg class="spark" viewBox="0 0 ${width} ${height}" aria-label="price trend">${label}<path class="area" d="${area}" fill="${stroke}"></path><path class="line" d="${line}" stroke="${stroke}"${dash}></path><circle cx="${coords.at(-1)[0].toFixed(1)}" cy="${coords.at(-1)[1].toFixed(1)}" r="4" fill="${stroke}" stroke="#080d14" stroke-width="2"></circle></svg>`;
}

function rowHtml(r) {
  const points = historySeries[r.sticker_id] || [];
  const link = r.item_url || '#';
  const steam = r.steam_url || '#';
  const vcolor = colorForVerdict(r.verdict);
  const expectedClass = pctClass(r.expected_return_pct);
  const demandClass = pctClass(r.demand_momentum_score);
  const changeValue = hasNum(r.snapshot_price_change_pct) ? r.snapshot_price_change_pct : r.recent_return_pct;
  const image = r.image_url || '';
  const typeLabel = r.display_type || r.category || '-';
  return `<tr>
    <td><div class="rank">#${esc(r.priority_rank)}</div><div class="tier">${esc(r.priority_tier || '')}</div></td>
    <td>
      <div class="sticker-cell">
        <img class="thumb" src="${esc(image)}" loading="lazy" onerror="this.style.visibility='hidden'" />
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
            <a class="action steam" href="${esc(steam)}">Preview</a>
          </div>
        </div>
      </div>
    </td>
    <td>
      <div class="price-main">${money(r.usd_price)}</div>
      <div class="price-sub">${tokens(r.price_tokens)} tokens</div>
      <div class="metric-list">
        <div class="metric-row"><span>Entry</span><b>${esc(r.entry_tier || '-')}</b></div>
        <div class="metric-row"><span>Low</span><b>${rangeMoney(r.hist_min, r)}</b></div>
        <div class="metric-row"><span>High</span><b>${rangeMoney(r.hist_max, r)}</b></div>
      </div>
    </td>
    <td>
      <span class="verdict" style="background:${vcolor}">${esc(r.verdict || '-')}</span>
      <div class="metric-list">
        <div class="metric-row"><span>Priority</span><b>${fmt(r.priority_score,1)}</b></div>
        <div class="metric-row"><span>Size</span><b>${esc(r.suggested_size || '-')}</b></div>
        <div class="metric-row"><span>Confidence</span><b>${fmt(r.prediction_confidence,2)}</b></div>
      </div>
    </td>
    <td>
      <div class="metric-list" style="margin-top:0">
        <div class="metric-row"><span>Expected</span><b class="${expectedClass}">${pct(r.expected_return_pct,0)}</b></div>
        <div class="metric-row"><span>Value Edge</span><b>${fmt(r.value_edge_score,2)}</b></div>
        <div class="metric-row"><span>Quality</span><b>${fmt(r.quality_score,2)}</b></div>
        <div class="metric-row"><span>Score Conf.</span><b>${fmt(r.score_confidence,2)}</b></div>
        <div class="metric-row"><span>Manual</span><b>${fmt(r.manual_score_count,0)}</b></div>
      </div>
    </td>
    <td>
      <div class="metric-list" style="margin-top:0">
        <div class="metric-row"><span>Flood</span><b>${esc(r.flood_risk || '-')} (${fmt(r.flood_risk_score,2)})</b></div>
        <div class="metric-row"><span>Discount</span><b>${pct(r.discount_from_high_pct,0)}</b></div>
        <div class="metric-row"><span>Demand</span><b class="${demandClass}">${fmt(r.demand_momentum_score,2)}</b></div>
        <div class="metric-row"><span>Change</span><b class="${pctClass(changeValue)}">${pct(changeValue,1)}</b></div>
      </div>
      ${sparkline(points, r)}
    </td>
    <td>
      <div class="note-block">
        <div><label>Reason</label><div>${esc(r.quick_reason || '-')}</div></div>
        <div><label>Risk</label><div>${esc(r.risk_note || '-')}</div></div>
        <div><label>Action</label><div class="note-action">${esc(r.action_note || '-')}</div></div>
      </div>
    </td>
  </tr>`;
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
  const minConfidence = num($('confidenceFilter').value);
  const maxPrice = num($('priceMax').value);

  filtered = records.filter(r => {
    if (verdict && r.verdict !== verdict) return false;
    if (variant && r.variant !== variant) return false;
    if (type && r.display_type !== type) return false;
    if (category && r.category !== category) return false;
    if (entry && r.entry_tier !== entry) return false;
    if (flood && r.flood_risk !== flood) return false;
    if (scored && String(r.scored) !== scored) return false;
    if (minConfidence !== null && (!hasNum(r.prediction_confidence) || Number(r.prediction_confidence) < minConfidence)) return false;
    if (maxPrice !== null && Number(r.price_tokens || 0) > maxPrice) return false;
    if (q) {
      const hay = [
        r.sticker, r.team, r.team_name, r.player_name, r.variant, r.display_type, r.category,
        r.verdict, r.quick_reason, r.risk_note, r.action_note, r.notes, r.flood_risk,
        r.entry_tier, r.price_tokens, r.usd_price, r.steam_market_hash_name
      ].join(' ').toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });
  sortRows();
  renderTable();
  renderCharts();
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

function renderTable() {
  $('tbody').innerHTML = filtered.map(rowHtml).join('') || `<tr><td colspan="7" class="empty">No stickers match the active filters.</td></tr>`;
  $('visibleCount').textContent = filtered.length.toLocaleString();
  $('totalCount').textContent = records.length.toLocaleString();
  const expected = filtered.map(r => num(r.expected_return_pct)).filter(v => v !== null);
  const edge = filtered.map(r => num(r.value_edge_score)).filter(v => v !== null);
  const scored = filtered.filter(r => r.scored).length;
  $('avgExpected').textContent = expected.length ? pct(expected.reduce((a,b) => a + b, 0) / expected.length, 0) : '-';
  $('avgEdge').textContent = edge.length ? fmt(edge.reduce((a,b) => a + b, 0) / edge.length, 2) : '-';
  $('scoredCount').textContent = `${scored}/${filtered.length}`;
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
  $('opportunityPlot').innerHTML = opportunityChart();
  $('variantPlot').innerHTML = variantChart();
  $('movementPlot').innerHTML = movementChart();
}

function wire() {
  makeOptions();
  ['search','verdictFilter','variantFilter','typeFilter','categoryFilter','entryFilter','floodFilter','confidenceFilter','priceMax','scoredFilter']
    .forEach(id => $(id).addEventListener('input', applyFilters));
  $('resetBtn').addEventListener('click', () => {
    ['search','verdictFilter','variantFilter','typeFilter','categoryFilter','entryFilter','floodFilter','confidenceFilter','priceMax','scoredFilter']
      .forEach(id => $(id).value = '');
    sortKey = 'priority_rank';
    sortDir = 1;
    $('sortHint').textContent = 'Sorted by priority rank';
    applyFilters();
  });
  $('rerenderCharts').addEventListener('click', renderCharts);
  document.querySelectorAll('th.sortable').forEach(th => th.addEventListener('click', () => {
    const key = th.dataset.sort;
    if (sortKey === key) sortDir *= -1;
    else {
      sortKey = key;
      sortDir = ['priority_score','expected_return_pct','quality_score','value_edge_score'].includes(key) ? -1 : 1;
    }
    $('sortHint').textContent = `Sorted by ${key} ${sortDir === 1 ? 'ascending' : 'descending'}`;
    applyFilters();
  }));
  $('topN').addEventListener('input', renderCharts);
  $('movementScale').addEventListener('input', renderCharts);
  $('legend').innerHTML = Object.entries(verdictColors)
    .map(([k,v]) => `<span><span class="legend-dot" style="background:${v}"></span>${esc(k)}</span>`)
    .join('');
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
    analysis = enrich_steam_preview_links(analysis)
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
