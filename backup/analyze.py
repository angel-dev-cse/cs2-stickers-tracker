from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable
import math
import re

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
SNAPSHOT_DIR = DATA_DIR / "snapshots"
LATEST_SNAPSHOT_PATH = DATA_DIR / "latest_snapshot.csv"
SCORES_PATH = DATA_DIR / "scores.csv"
HISTORY_PATH = DATA_DIR / "history_points.csv"

OUT_DIR = Path("analyze")
DEFAULT_VARIANTS = "Paper,Foil,Holo,Gold"


# -----------------------------
# File loading helpers
# -----------------------------

def candidate_files(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    search_roots = [Path("."), Path("/mnt/data"), DATA_DIR, Path("/mnt/data/data"), SNAPSHOT_DIR]
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            files.extend(root.glob(pattern))
    # Deduplicate while preserving file identity.
    unique = {p.resolve(): p for p in files if p.is_file()}
    return sorted(unique.values(), key=lambda p: p.stat().st_mtime)


def output_dir() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


def read_csv_best(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def read_csv_loose(path: Path) -> pd.DataFrame:
    """Read CSVs written by older and newer collectors without losing the whole file."""
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


def normalize_variant(value: str | None) -> str:
    raw = "" if value is None else str(value).strip()
    if not raw:
        return ""
    known = {
        "paper": "Paper",
        "foil": "Foil",
        "holo": "Holo",
        "gold": "Gold",
        "glitter": "Glitter",
        "lenticular": "Lenticular",
    }
    return known.get(raw.lower(), raw[:1].upper() + raw[1:])


def parse_variants(value: str) -> list[str]:
    variants: list[str] = []
    seen: set[str] = set()
    for part in value.split(","):
        variant = normalize_variant(part)
        key = variant.lower()
        if variant and key not in seen:
            variants.append(variant)
            seen.add(key)
    return variants


def load_snapshots() -> pd.DataFrame:
    files = candidate_files(["snapshot_*.csv", "latest_snapshot*.csv"])
    frames: list[pd.DataFrame] = []
    for path in files:
        try:
            df = read_csv_best(path)
        except Exception:
            continue
        if "timestamp" not in df.columns or "sticker_id" not in df.columns:
            continue
        frames.append(df)
    if frames:
        return pd.concat(frames, ignore_index=True)
    raise SystemExit("No snapshot found. Run collect.py first or place latest_snapshot.csv in data/.")


def load_scores() -> pd.DataFrame:
    score_cols = [
        "sticker_id",
        "visual_score",
        "craft_score",
        "demand_score",
        "color_score",
        "readability_score",
        "notes",
    ]
    files = candidate_files(["scores*.csv"])
    frames: list[pd.DataFrame] = []
    for path in files:
        try:
            raw = read_csv_best(path)
        except Exception:
            continue
        cols = [c for c in score_cols if c in raw.columns]
        if "sticker_id" in cols:
            frames.append(raw[cols].copy())
    if not frames:
        return pd.DataFrame(columns=score_cols)
    scores = pd.concat(frames, ignore_index=True)
    # Merge duplicate score templates safely: keep the last non-empty value per sticker/column.
    for col in score_cols:
        if col not in scores.columns:
            scores[col] = np.nan
    merged_rows = []
    for sticker_id, group in scores.groupby("sticker_id", sort=False):
        row = {"sticker_id": sticker_id}
        for col in score_cols:
            if col == "sticker_id":
                continue
            values = group[col].dropna()
            # Treat blank strings as missing for notes too.
            values = values[values.astype(str).str.strip() != ""]
            row[col] = values.iloc[-1] if len(values) else np.nan
        merged_rows.append(row)
    return pd.DataFrame(merged_rows, columns=score_cols)


def load_history() -> pd.DataFrame:
    files = candidate_files(["history_points*.csv", "latest_history*.csv"])
    frames: list[pd.DataFrame] = []
    for path in files:
        try:
            df = read_csv_loose(path)
        except Exception:
            continue
        if "sticker_id" not in df.columns:
            continue
        if "token_cost" not in df.columns and "token_cost_est" not in df.columns:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

# -----------------------------
# Generic helpers
# -----------------------------

def numeric(series: pd.Series, default: float | None = np.nan) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def datetime_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(pd.NaT, index=df.index)
    parsed = pd.to_datetime(df[column], errors="coerce", utc=True, format="mixed")
    return parsed.dt.tz_convert(None)


def pct_change(new: float, old: float) -> float:
    if old is None or old == 0 or pd.isna(old) or pd.isna(new):
        return np.nan
    return ((new / old) - 1) * 100


def safe_div(num: float, den: float, default: float = np.nan) -> float:
    if den is None or den == 0 or pd.isna(den) or pd.isna(num):
        return default
    return num / den


def clean_name(name: str) -> str:
    name = str(name)
    name = name.replace("Sticker | ", "")
    name = name.replace(" | Cologne 2026", "")
    return name


def percentile_rank(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() <= 1:
        return pd.Series(0.0, index=series.index)
    return s.rank(method="average", pct=True).fillna(0.0)


def score_clip(value: pd.Series | float, low: float = 0, high: float = 1):
    return pd.Series(value).clip(low, high) if not isinstance(value, pd.Series) else value.clip(low, high)


def linear_slope(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if not pd.isna(v)]
    if len(vals) < 2:
        return 0.0
    x = np.arange(len(vals), dtype=float)
    y = np.array(vals, dtype=float)
    if np.nanmean(y) == 0:
        return 0.0
    slope = np.polyfit(x, y, 1)[0]
    # Normalized slope: percent of average value per point.
    return float((slope / np.nanmean(y)) * 100)


def largest_previous_bounce_pct(prices: list[float]) -> float:
    """Largest low -> later high bounce found before the final point."""
    vals = [float(p) for p in prices if not pd.isna(p) and p > 0]
    if len(vals) < 3:
        return 0.0

    best = 0.0
    # Ignore the final current-ish point when looking for historical defended zones.
    for i in range(0, len(vals) - 2):
        low = vals[i]
        future_high = max(vals[i + 1 : -1]) if len(vals[i + 1 : -1]) else np.nan
        if low and not pd.isna(future_high):
            best = max(best, ((future_high / low) - 1) * 100)
    return round(best, 2)


def count_lower_highs(prices: list[float]) -> int:
    vals = [float(p) for p in prices if not pd.isna(p) and p > 0]
    if len(vals) < 4:
        return 0
    peaks: list[float] = []
    for i in range(1, len(vals) - 1):
        if vals[i] >= vals[i - 1] and vals[i] >= vals[i + 1]:
            peaks.append(vals[i])
    if len(peaks) < 2:
        return 0
    return sum(1 for a, b in zip(peaks, peaks[1:]) if b < a)


def weighted_reference_price(row: pd.Series) -> float:
    parts = [
        ("hist_median", 0.24),
        ("hist_avg", 0.16),
        ("recent_avg_price", 0.20),
        ("early_avg_price", 0.10),
        ("snapshot_median_price", 0.14),
        ("snapshot_recent_avg_price", 0.16),
    ]
    values: list[float] = []
    weights: list[float] = []

    for col, weight in parts:
        value = row.get(col, np.nan)
        if pd.notna(value) and float(value) > 0:
            values.append(float(value))
            weights.append(weight)

    if not values:
        return np.nan

    return float(np.average(values, weights=weights))


def pct_change_series(new: pd.Series, old: pd.Series) -> pd.Series:
    new_num = pd.to_numeric(new, errors="coerce")
    old_num = pd.to_numeric(old, errors="coerce")
    return np.where((old_num > 0) & new_num.notna(), ((new_num / old_num) - 1) * 100, np.nan)


def bounded_score(series: pd.Series, center: float = 0, scale: float = 1) -> pd.Series:
    return ((pd.to_numeric(series, errors="coerce").fillna(center) - center) / scale).clip(0, 1)


def summarize_snapshot_trends(snapshots: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "sticker_id",
        "snapshot_points",
        "snapshot_span_hours",
        "snapshot_first_price",
        "snapshot_prev_price",
        "snapshot_recent_avg_price",
        "snapshot_median_price",
        "snapshot_min_price",
        "snapshot_max_price",
        "snapshot_robust_peak",
        "snapshot_return_pct",
        "snapshot_price_change_pct",
        "snapshot_price_velocity_pct_per_day",
        "snapshot_price_slope",
        "snapshot_price_acceleration",
        "rank_change",
        "rank_percentile_change",
        "rank_improvement_score",
        "price_drop_opportunity_score",
    ]
    if snapshots.empty:
        return pd.DataFrame(columns=cols)

    hist = snapshots.copy()
    hist = hist.dropna(subset=["timestamp", "sticker_id", "price_tokens"])
    hist = hist.sort_values(["timestamp", "sticker_id"])
    hist = hist.drop_duplicates(subset=["timestamp", "sticker_id"], keep="last")

    group_cols = ["timestamp", "category", "variant"]
    hist["snapshot_rank"] = hist.groupby(group_cols)["price_tokens"].rank(method="first", ascending=True)
    hist["snapshot_group_total"] = hist.groupby(group_cols)["price_tokens"].transform("count")
    hist["snapshot_price_percentile"] = (
        (hist["snapshot_rank"] - 1) / (hist["snapshot_group_total"] - 1)
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    rows = []
    for sticker_id, group in hist.groupby("sticker_id", sort=False):
        g = group.sort_values("timestamp")
        prices = g["price_tokens"].astype(float).tolist()
        if not prices:
            continue

        latest = g.iloc[-1]
        previous = g.iloc[-2] if len(g) >= 2 else None
        first = g.iloc[0]
        recent = prices[-min(3, len(prices)) :]
        prior_recent = prices[-min(6, len(prices)) : -min(3, len(prices))] if len(prices) >= 6 else []
        snapshot_span_hours = (g["timestamp"].max() - g["timestamp"].min()).total_seconds() / 3600

        if previous is not None:
            price_change = pct_change(float(latest["price_tokens"]), float(previous["price_tokens"]))
            elapsed_hours = max((latest["timestamp"] - previous["timestamp"]).total_seconds() / 3600, 1)
            velocity = price_change / max(elapsed_hours / 24, 1 / 24) if not pd.isna(price_change) else np.nan
            rank_change = float(latest["snapshot_rank"] - previous["snapshot_rank"])
            rank_percentile_change = float(latest["snapshot_price_percentile"] - previous["snapshot_price_percentile"])
        else:
            price_change = np.nan
            velocity = np.nan
            rank_change = np.nan
            rank_percentile_change = np.nan

        recent_slope = linear_slope(recent)
        prior_slope = linear_slope(prior_recent)
        acceleration = recent_slope - prior_slope if prior_recent else 0.0
        rank_improvement = max(0.0, min(1.0, (-rank_percentile_change) / 0.25)) if not pd.isna(rank_percentile_change) else 0.0
        price_drop_opportunity = max(0.0, min(1.0, (-(price_change or 0)) / 35)) if not pd.isna(price_change) else 0.0

        rows.append({
            "sticker_id": sticker_id,
            "snapshot_points": len(prices),
            "snapshot_span_hours": round(snapshot_span_hours, 2),
            "snapshot_first_price": round(float(first["price_tokens"]), 2),
            "snapshot_prev_price": round(float(previous["price_tokens"]), 2) if previous is not None else np.nan,
            "snapshot_recent_avg_price": round(float(np.mean(recent)), 2),
            "snapshot_median_price": round(float(np.median(prices)), 2),
            "snapshot_min_price": round(float(np.min(prices)), 2),
            "snapshot_max_price": round(float(np.max(prices)), 2),
            "snapshot_robust_peak": round(float(np.quantile(prices, 0.85)), 2),
            "snapshot_return_pct": round(pct_change(float(latest["price_tokens"]), float(first["price_tokens"])), 2),
            "snapshot_price_change_pct": round(price_change, 2) if not pd.isna(price_change) else np.nan,
            "snapshot_price_velocity_pct_per_day": round(velocity, 2) if not pd.isna(velocity) else np.nan,
            "snapshot_price_slope": round(linear_slope(prices), 2),
            "snapshot_price_acceleration": round(acceleration, 2),
            "rank_change": round(rank_change, 2) if not pd.isna(rank_change) else np.nan,
            "rank_percentile_change": round(rank_percentile_change, 4) if not pd.isna(rank_percentile_change) else np.nan,
            "rank_improvement_score": round(rank_improvement, 4),
            "price_drop_opportunity_score": round(price_drop_opportunity, 4),
        })

    return pd.DataFrame(rows, columns=cols)


def build_snapshot_validation(snapshots: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "signal",
        "observations",
        "overall_mean_next_return_pct",
        "top_quintile_mean_next_return_pct",
        "top_quintile_hit_rate_pct",
        "bottom_quintile_mean_next_return_pct",
        "bottom_quintile_hit_rate_pct",
        "spread_top_minus_bottom_pct",
    ]
    if snapshots.empty:
        return pd.DataFrame(columns=cols)

    hist = snapshots.copy()
    hist = hist.dropna(subset=["timestamp", "sticker_id", "price_tokens"])
    hist = hist.sort_values(["timestamp", "sticker_id"])
    hist = hist.drop_duplicates(subset=["timestamp", "sticker_id"], keep="last")

    group_cols = ["timestamp", "category", "variant"]
    hist["price_percentile_at_time"] = (
        hist.groupby(group_cols)["price_tokens"].rank(method="average", pct=True).fillna(0)
    )
    hist["prev_price"] = hist.groupby("sticker_id")["price_tokens"].shift(1)
    hist["next_price"] = hist.groupby("sticker_id")["price_tokens"].shift(-1)
    hist["prev_return_pct"] = pct_change_series(hist["price_tokens"], hist["prev_price"])
    hist["next_return_pct"] = pct_change_series(hist["next_price"], hist["price_tokens"])
    hist = hist.dropna(subset=["next_return_pct"]).copy()

    hist["cheap_percentile_signal"] = (1 - hist["price_percentile_at_time"]).clip(0, 1)
    hist["recent_drop_signal"] = (hist["prev_return_pct"].fillna(0).clip(upper=0).abs() / 35).clip(0, 1)
    hist["recent_momentum_signal"] = (hist["prev_return_pct"].fillna(0).clip(lower=0) / 50).clip(0, 1)

    signals = [
        "cheap_percentile_signal",
        "recent_drop_signal",
        "recent_momentum_signal",
    ]
    rows = []
    for signal in signals:
        scored = hist[[signal, "next_return_pct"]].dropna().copy()
        if len(scored) < 10 or scored[signal].nunique() <= 1:
            continue
        bucket_size = max(1, int(len(scored) * 0.20))
        top = scored.nlargest(bucket_size, signal)
        bottom = scored.nsmallest(bucket_size, signal)
        if top.empty or bottom.empty:
            continue
        rows.append({
            "signal": signal,
            "observations": len(scored),
            "overall_mean_next_return_pct": round(float(scored["next_return_pct"].mean()), 2),
            "top_quintile_mean_next_return_pct": round(float(top["next_return_pct"].mean()), 2),
            "top_quintile_hit_rate_pct": round(float((top["next_return_pct"] > 0).mean() * 100), 2),
            "bottom_quintile_mean_next_return_pct": round(float(bottom["next_return_pct"].mean()), 2),
            "bottom_quintile_hit_rate_pct": round(float((bottom["next_return_pct"] > 0).mean() * 100), 2),
            "spread_top_minus_bottom_pct": round(float(top["next_return_pct"].mean() - bottom["next_return_pct"].mean()), 2),
        })

    return pd.DataFrame(rows, columns=cols)


# -----------------------------
# History processing
# -----------------------------

def clean_history(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw

    hist = raw.copy()
    if "token_cost" not in hist.columns:
        if "token_cost_est" in hist.columns:
            hist["token_cost"] = hist["token_cost_est"]
        else:
            return pd.DataFrame()

    hist["token_cost"] = pd.to_numeric(hist["token_cost"], errors="coerce")
    hist["usd_price"] = pd.to_numeric(hist.get("usd_price"), errors="coerce")
    hist["popularity"] = pd.to_numeric(hist.get("popularity"), errors="coerce")
    hist["history_scrape_timestamp"] = datetime_column(hist, "history_scrape_timestamp")
    hist["point_index"] = pd.to_numeric(hist.get("point_index"), errors="coerce")

    fetched_time = datetime_column(hist, "fetched_at")
    tooltip_time = datetime_column(hist, "tooltip_time_raw")
    hist["point_time"] = fetched_time.fillna(tooltip_time)
    hist["point_time_source"] = np.where(fetched_time.notna(), "fetched_at", "tooltip_time_raw")

    if "tooltip_time_raw" in hist.columns:
        bad = hist["tooltip_time_raw"].astype(str).str.contains(
            "Cologne 2026|PRICE HISTORY|TOKEN COST|Sticker \\|", case=False, na=False
        )
        hist = hist[~bad].copy()

    hist = hist.dropna(subset=["sticker_id", "token_cost"])
    hist = hist[hist["token_cost"] > 0].copy()

    if "history_range" in hist.columns and (hist["history_range"] == "30D").any():
        hist = hist[hist["history_range"] == "30D"].copy()

    dedupe_cols = [col for col in ["sticker_id", "history_range", "point_time", "token_cost"] if col in hist.columns]
    if dedupe_cols:
        hist = hist.drop_duplicates(subset=dedupe_cols, keep="last")

    if hist["point_time"].notna().any():
        hist["point_bucket"] = hist["point_time"].dt.floor("h")
        abs_popularity = hist["popularity"].abs().fillna(0)
        positive_popularity = hist["popularity"].clip(lower=0).fillna(0)
        total_abs = abs_popularity.groupby(hist["point_bucket"]).transform("sum")
        total_positive = positive_popularity.groupby(hist["point_bucket"]).transform("sum")
        hist["relative_demand_share"] = (abs_popularity / total_abs.replace(0, np.nan)).fillna(0)
        hist["positive_demand_share"] = (positive_popularity / total_positive.replace(0, np.nan)).fillna(0)
    else:
        hist["point_bucket"] = pd.NaT
        hist["relative_demand_share"] = np.nan
        hist["positive_demand_share"] = np.nan

    sort_cols = ["sticker_id"]
    if hist["point_time"].notna().any():
        sort_cols += ["point_time", "point_index"]
    else:
        sort_cols += ["point_index"]
    hist = hist.sort_values(sort_cols)
    return hist


def summarize_history(hist: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "sticker_id",
        "hist_points",
        "launch_price",
        "early_avg_price",
        "hist_last",
        "hist_min",
        "hist_max",
        "hist_avg",
        "hist_median",
        "recent_avg_price",
        "hist_return_pct",
        "hist_range_pct",
        "hist_volatility_pct",
        "recent_return_pct",
        "price_slope_full",
        "price_slope_recent",
        "popularity_slope_full",
        "popularity_slope_recent",
        "positive_popularity_sum",
        "absolute_popularity_pressure",
        "peak_positive_popularity",
        "latest_popularity",
        "popularity_points",
        "avg_relative_demand_share",
        "latest_relative_demand_share",
        "relative_demand_share_change_pct",
        "relative_demand_share_slope_full",
        "relative_demand_share_slope_recent",
        "demand_share_acceleration",
        "avg_positive_demand_share",
        "latest_positive_demand_share",
        "max_previous_bounce_pct",
        "lower_high_count",
        "history_span_hours",
        "history_coverage_score",
        "last_tooltip_time_raw",
        "last_point_time",
    ]
    if hist.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for sticker_id, g in hist.groupby("sticker_id", sort=False):
        g = g.sort_values(["point_time", "point_index"], na_position="last")
        prices = g["token_cost"].dropna().astype(float).tolist()
        if not prices:
            continue

        first = prices[0]
        last = prices[-1]
        hist_min = min(prices)
        hist_max = max(prices)
        hist_avg = float(np.mean(prices))
        hist_median = float(np.median(prices))
        early_avg = float(np.mean(prices[: min(3, len(prices))]))
        recent_avg = float(np.mean(prices[-min(3, len(prices)) :]))

        returns = pd.Series(prices).pct_change().dropna() * 100
        volatility = float(returns.std()) if len(returns) > 1 else 0.0
        recent_return = pct_change(prices[-1], prices[-2]) if len(prices) >= 2 else np.nan

        pops = g["popularity"].dropna().astype(float).tolist() if "popularity" in g.columns else []
        positive_popularity_sum = float(sum(p for p in pops if p > 0)) if pops else 0.0
        absolute_popularity_pressure = float(sum(abs(p) for p in pops)) if pops else 0.0
        peak_positive_popularity = float(max([p for p in pops if p > 0], default=0.0)) if pops else 0.0
        latest_popularity = pops[-1] if pops else np.nan
        shares = g["relative_demand_share"].dropna().astype(float).tolist() if "relative_demand_share" in g.columns else []
        positive_shares = g["positive_demand_share"].dropna().astype(float).tolist() if "positive_demand_share" in g.columns else []
        latest_share = shares[-1] if shares else np.nan
        avg_share = float(np.mean(shares)) if shares else np.nan
        share_change = pct_change(latest_share, shares[0]) if shares and shares[0] > 0 else np.nan
        share_slope_full = linear_slope(shares) if len(shares) >= 2 else np.nan
        share_slope_recent = linear_slope(shares[-3:]) if len(shares) >= 2 else np.nan
        prior_share_slope = linear_slope(shares[-6:-3]) if len(shares) >= 6 else 0.0
        share_acceleration = share_slope_recent - prior_share_slope if not pd.isna(share_slope_recent) else np.nan
        latest_positive_share = positive_shares[-1] if positive_shares else np.nan
        avg_positive_share = float(np.mean(positive_shares)) if positive_shares else np.nan

        tooltip_times = g.get("tooltip_time_raw", pd.Series(dtype=str)).dropna().tolist()
        last_tooltip_time = tooltip_times[-1] if tooltip_times else ""
        point_times = g["point_time"].dropna() if "point_time" in g.columns else pd.Series(dtype="datetime64[ns]")
        if len(point_times) >= 2:
            history_span_hours = (point_times.max() - point_times.min()).total_seconds() / 3600
        else:
            history_span_hours = 0.0
        point_score = min(len(prices) / 8, 1.0)
        span_score = min(history_span_hours / 72, 1.0)
        history_coverage_score = 0.70 * point_score + 0.30 * span_score
        last_point_time = point_times.max().isoformat() if len(point_times) else ""

        rows.append({
            "sticker_id": sticker_id,
            "hist_points": len(prices),
            "launch_price": round(first, 2),
            "early_avg_price": round(early_avg, 2),
            "hist_last": round(last, 2),
            "hist_min": round(hist_min, 2),
            "hist_max": round(hist_max, 2),
            "hist_avg": round(hist_avg, 2),
            "hist_median": round(hist_median, 2),
            "recent_avg_price": round(recent_avg, 2),
            "hist_return_pct": round(pct_change(last, first), 2) if first else np.nan,
            "hist_range_pct": round(((hist_max / hist_min) - 1) * 100, 2) if hist_min else np.nan,
            "hist_volatility_pct": round(volatility, 2),
            "recent_return_pct": round(recent_return, 2) if not pd.isna(recent_return) else np.nan,
            "price_slope_full": round(linear_slope(prices), 2),
            "price_slope_recent": round(linear_slope(prices[-3:]), 2),
            "popularity_slope_full": round(linear_slope(pops), 2) if len(pops) >= 2 else np.nan,
            "popularity_slope_recent": round(linear_slope(pops[-3:]), 2) if len(pops) >= 2 else np.nan,
            "positive_popularity_sum": round(positive_popularity_sum, 2),
            "absolute_popularity_pressure": round(absolute_popularity_pressure, 2),
            "peak_positive_popularity": round(peak_positive_popularity, 2),
            "latest_popularity": latest_popularity,
            "popularity_points": len(pops),
            "avg_relative_demand_share": round(avg_share, 6) if not pd.isna(avg_share) else np.nan,
            "latest_relative_demand_share": round(latest_share, 6) if not pd.isna(latest_share) else np.nan,
            "relative_demand_share_change_pct": round(share_change, 2) if not pd.isna(share_change) else np.nan,
            "relative_demand_share_slope_full": round(share_slope_full, 2) if not pd.isna(share_slope_full) else np.nan,
            "relative_demand_share_slope_recent": round(share_slope_recent, 2) if not pd.isna(share_slope_recent) else np.nan,
            "demand_share_acceleration": round(share_acceleration, 2) if not pd.isna(share_acceleration) else np.nan,
            "avg_positive_demand_share": round(avg_positive_share, 6) if not pd.isna(avg_positive_share) else np.nan,
            "latest_positive_demand_share": round(latest_positive_share, 6) if not pd.isna(latest_positive_share) else np.nan,
            "max_previous_bounce_pct": largest_previous_bounce_pct(prices),
            "lower_high_count": count_lower_highs(prices),
            "history_span_hours": round(history_span_hours, 2),
            "history_coverage_score": round(history_coverage_score, 4),
            "last_tooltip_time_raw": last_tooltip_time,
            "last_point_time": last_point_time,
        })

    return pd.DataFrame(rows)


# -----------------------------
# Metrics and score construction
# -----------------------------

def add_history_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in [
        "hist_min",
        "hist_max",
        "hist_avg",
        "hist_median",
        "hist_points",
        "launch_price",
        "early_avg_price",
        "recent_avg_price",
        "history_coverage_score",
        "snapshot_points",
        "snapshot_recent_avg_price",
        "snapshot_median_price",
        "snapshot_min_price",
        "snapshot_max_price",
        "snapshot_robust_peak",
        "snapshot_price_change_pct",
        "snapshot_price_velocity_pct_per_day",
        "snapshot_price_slope",
        "snapshot_price_acceleration",
        "rank_percentile_change",
        "rank_improvement_score",
        "price_drop_opportunity_score",
    ]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["discount_from_high_pct"] = np.where(
        df["hist_max"] > 0,
        ((df["hist_max"] - df["price_tokens"]) / df["hist_max"]) * 100,
        np.nan,
    )
    df["upside_to_high_pct"] = np.where(
        df["price_tokens"] > 0,
        ((df["hist_max"] / df["price_tokens"]) - 1) * 100,
        np.nan,
    )
    df["position_in_range"] = np.where(
        df["hist_max"] > df["hist_min"],
        (df["price_tokens"] - df["hist_min"]) / (df["hist_max"] - df["hist_min"]),
        0.5,
    )
    df["position_in_range"] = pd.to_numeric(df["position_in_range"], errors="coerce").clip(0, 1).fillna(0.5)

    df["launch_gap_pct"] = np.where(
        df["launch_price"] > 0,
        ((df["price_tokens"] / df["launch_price"]) - 1) * 100,
        np.nan,
    )
    df["early_avg_gap_pct"] = np.where(
        df["early_avg_price"] > 0,
        ((df["price_tokens"] / df["early_avg_price"]) - 1) * 100,
        np.nan,
    )
    df["robust_reference_price"] = df.apply(weighted_reference_price, axis=1).round(2)
    df["expected_return_pct"] = np.where(
        (df["robust_reference_price"] > 0) & (df["price_tokens"] > 0),
        ((df["robust_reference_price"] / df["price_tokens"]) - 1) * 100,
        np.nan,
    )
    df["expected_return_score"] = (df["expected_return_pct"].fillna(0) / 120).clip(0, 1).round(4)
    df["robust_peak_price"] = df[["hist_max", "snapshot_robust_peak"]].max(axis=1, skipna=True)
    df["discount_from_robust_peak_pct"] = np.where(
        df["robust_peak_price"] > 0,
        ((df["robust_peak_price"] - df["price_tokens"]) / df["robust_peak_price"]) * 100,
        np.nan,
    )
    df["downside_to_floor_pct"] = np.where(
        (df["hist_min"] > 0) & (df["price_tokens"] > 0),
        np.minimum(((df["hist_min"] / df["price_tokens"]) - 1) * 100, 0),
        np.nan,
    )
    df["downside_risk_score"] = (df["downside_to_floor_pct"].fillna(0).abs() / 80).clip(0, 1).round(4)
    df["history_coverage_score"] = df["history_coverage_score"].fillna(0).clip(0, 1)
    df["history_uncertainty_score"] = (1 - df["history_coverage_score"]).clip(0, 1).round(4)

    group_cols = ["category", "variant"]
    df["group_median_discount_pct"] = df.groupby(group_cols)["discount_from_high_pct"].transform("median")
    df["group_median_position"] = df.groupby(group_cols)["position_in_range"].transform("median")
    df["relative_discount_pct"] = df["discount_from_high_pct"] - df["group_median_discount_pct"]
    df["relative_floor_bonus"] = df["group_median_position"] - df["position_in_range"]

    return df


def add_scores(df: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(scores, on="sticker_id", how="left")
    fields = ["visual_score", "craft_score", "demand_score", "color_score", "readability_score"]
    for field in fields:
        if field not in df.columns:
            df[field] = np.nan
        df[field] = pd.to_numeric(df[field], errors="coerce")

    df["manual_score_count"] = df[fields].notna().sum(axis=1)
    df["score_confidence"] = (df["manual_score_count"] / len(fields)).round(4)
    df["scored"] = df[fields].notna().all(axis=1)
    temp = df[fields].fillna(6)
    df["quality_score"] = (
        0.30 * temp["visual_score"]
        + 0.25 * temp["demand_score"]
        + 0.20 * temp["craft_score"]
        + 0.15 * temp["color_score"]
        + 0.10 * temp["readability_score"]
    ).round(2)

    # Separate because demand is important enough to be visible in final board.
    df["manual_demand_score"] = df["demand_score"].fillna(6)
    df["manual_color_craft_score"] = (0.55 * temp["color_score"] + 0.45 * temp["craft_score"]).round(2)
    notes = df.get("notes", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    df["note_positive_flag"] = notes.str.contains(
        r"\b(?:good|great|clean|nice|strong|unique|interesting|underrated|best|love|pop|hype)\b",
        regex=True,
    ).astype(int)
    df["note_negative_flag"] = notes.str.contains(
        r"\b(?:bad|weak|ugly|avoid|poor|meh|messy|boring|overpriced|hard to read|not good)\b",
        regex=True,
    ).astype(int)
    df["note_hype_flag"] = notes.str.contains(r"\b(?:hype|popular|fan|meme|star|donk|s1mple|zywoo|m0nesy)\b", regex=True).astype(int)
    df["note_avoid_flag"] = notes.str.contains(r"\b(?:avoid|skip|do not|don't|no buy|trap)\b", regex=True).astype(int)
    df["note_score_adjustment"] = (
        0.04 * df["note_positive_flag"]
        + 0.03 * df["note_hype_flag"]
        - 0.06 * df["note_negative_flag"]
        - 0.10 * df["note_avoid_flag"]
    ).clip(-0.15, 0.10).round(4)
    return df


def entry_bands(category: str, variant: str) -> list[tuple[float, str, float]]:
    """
    Token-price entry bands. 100 tokens ≈ $0.99.

    Important correction: an $8 Team Holo (~800 tokens) is no longer labeled
    Cheap. It is Fair. Cheap should mean cheap enough to diversify without
    much emotional pressure, not merely cheaper than elite team holos.
    """
    c = str(category).lower()
    v = str(variant).lower()

    if c == "team" and v == "holo":
        return [
            (300, "Very Cheap", 1.00),
            (500, "Cheap", 0.86),
            (1000, "Fair", 0.58),
            (2500, "Expensive", 0.25),
            (float("inf"), "Premium", 0.06),
        ]
    if c == "team" and v == "foil":
        return [
            (120, "Very Cheap", 1.00),
            (250, "Cheap", 0.86),
            (600, "Fair", 0.55),
            (1200, "Expensive", 0.22),
            (float("inf"), "Premium", 0.05),
        ]
    if c == "team" and v == "paper":
        return [
            (45, "Very Cheap", 1.00),
            (90, "Cheap", 0.84),
            (180, "Fair", 0.52),
            (400, "Expensive", 0.20),
            (float("inf"), "Premium", 0.04),
        ]
    if c == "player" and v == "holo":
        return [
            (100, "Very Cheap", 1.00),
            (150, "Cheap", 0.86),
            (400, "Fair", 0.55),
            (1000, "Expensive", 0.22),
            (float("inf"), "Premium", 0.05),
        ]
    if c == "player" and v == "foil":
        return [
            (70, "Very Cheap", 1.00),
            (120, "Cheap", 0.86),
            (250, "Fair", 0.52),
            (600, "Expensive", 0.20),
            (float("inf"), "Premium", 0.04),
        ]
    if c == "player" and v == "paper":
        return [
            (20, "Very Cheap", 1.00),
            (45, "Cheap", 0.84),
            (100, "Fair", 0.50),
            (250, "Expensive", 0.18),
            (float("inf"), "Premium", 0.03),
        ]

    return [
        (100, "Very Cheap", 1.00),
        (150, "Cheap", 0.86),
        (400, "Fair", 0.55),
        (1000, "Expensive", 0.22),
        (float("inf"), "Premium", 0.05),
    ]


def add_entry_tier(df: pd.DataFrame) -> pd.DataFrame:
    tiers: list[str] = []
    scores: list[float] = []

    for _, row in df.iterrows():
        price = float(row.get("price_tokens", np.nan))
        if pd.isna(price):
            tiers.append("Unknown")
            scores.append(0.0)
            continue

        chosen_tier = "Premium"
        chosen_score = 0.0
        previous_limit = 0.0

        for limit, label, base_score in entry_bands(row.get("category"), row.get("variant")):
            if price <= limit:
                chosen_tier = label
                # Smoothly decay inside each band except the first.
                if previous_limit > 0 and np.isfinite(limit):
                    span = max(limit - previous_limit, 1)
                    progress = max(0.0, min(1.0, (price - previous_limit) / span))
                    next_score = base_score
                    prev_score = min(1.0, base_score + 0.18)
                    chosen_score = prev_score - (prev_score - next_score) * progress
                else:
                    chosen_score = base_score
                break
            previous_limit = limit

        tiers.append(chosen_tier)
        scores.append(float(chosen_score))

    df["entry_tier"] = tiers
    df["entry_score"] = pd.Series(scores, index=df.index).clip(0, 1).round(4)
    df["entry_change_score"] = (
        0.50 * df["entry_score"].fillna(0)
        + 0.25 * df.get("rank_improvement_score", pd.Series(0, index=df.index)).fillna(0)
        + 0.25 * df.get("price_drop_opportunity_score", pd.Series(0, index=df.index)).fillna(0)
    ).clip(0, 1).round(4)
    return df


def add_trend_and_flood(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in [
        "price_slope_full",
        "price_slope_recent",
        "popularity_slope_full",
        "popularity_slope_recent",
        "recent_return_pct",
        "max_previous_bounce_pct",
        "lower_high_count",
        "positive_popularity_sum",
        "absolute_popularity_pressure",
        "peak_positive_popularity",
        "latest_popularity",
        "history_coverage_score",
        "latest_relative_demand_share",
        "avg_relative_demand_share",
        "relative_demand_share_change_pct",
        "relative_demand_share_slope_full",
        "relative_demand_share_slope_recent",
        "demand_share_acceleration",
        "latest_positive_demand_share",
        "snapshot_price_change_pct",
        "snapshot_price_velocity_pct_per_day",
        "snapshot_price_slope",
        "snapshot_price_acceleration",
        "rank_percentile_change",
        "price_drop_opportunity_score",
    ]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Trend score: reward floor + previous bounce + non-overextended setups.
    floor = (1 - df["position_in_range"]).clip(0, 1)
    previous_bounce = (df["max_previous_bounce_pct"].fillna(0) / 250).clip(0, 1)
    recent_recovery = (df["recent_return_pct"].fillna(0).clip(lower=0) / 100).clip(0, 1)
    negative_recent_ok = np.where(df["recent_return_pct"].fillna(0) <= 25, 1, 0)
    lower_high_penalty = (df["lower_high_count"].fillna(0) / 3).clip(0, 1)
    overextended_penalty = df["position_in_range"].clip(0, 1)
    snapshot_momentum = (df["snapshot_price_velocity_pct_per_day"].fillna(0).clip(lower=0) / 80).clip(0, 1)
    snapshot_acceleration = ((df["snapshot_price_acceleration"].fillna(0) + 30) / 60).clip(0, 1)
    demand_share_recent = ((df["relative_demand_share_slope_recent"].fillna(0) + 35) / 70).clip(0, 1)

    df["trend_score"] = (
        0.27 * floor
        + 0.20 * previous_bounce
        + 0.13 * recent_recovery
        + 0.11 * snapshot_momentum
        + 0.10 * demand_share_recent
        + 0.09 * snapshot_acceleration
        + 0.10 * negative_recent_ok
        - 0.10 * lower_high_penalty
        - 0.10 * overextended_penalty
    ).clip(0, 1).round(4)

    trend_labels = []
    for _, row in df.iterrows():
        pos = row.get("position_in_range", 0.5)
        bounce = row.get("max_previous_bounce_pct", 0)
        recent = row.get("recent_return_pct", np.nan)
        lower_highs = row.get("lower_high_count", 0)
        pfull = row.get("price_slope_full", 0)
        precent = row.get("price_slope_recent", 0)
        poprecent = row.get("popularity_slope_recent", np.nan)
        share_recent = row.get("relative_demand_share_slope_recent", np.nan)
        snapshot_velocity = row.get("snapshot_price_velocity_pct_per_day", np.nan)
        points = row.get("hist_points", 0)

        if pd.isna(points) or points < 3:
            trend_labels.append("Limited history")
        elif pd.notna(share_recent) and share_recent > 8 and pd.notna(snapshot_velocity) and snapshot_velocity <= 0:
            trend_labels.append("Demand rising while price soft")
        elif pos <= 0.20 and bounce >= 80 and lower_highs >= 1:
            trend_labels.append("Floor + prior bounce, but lower highs")
        elif pos <= 0.25 and bounce >= 80:
            trend_labels.append("Floor + prior bounce")
        elif pfull > 5 and precent > 5:
            trend_labels.append("Confirmed upward momentum")
        elif precent < -5 and (pd.isna(poprecent) or poprecent <= 0):
            trend_labels.append("Falling / demand weak")
        elif pos >= 0.75:
            trend_labels.append("Near high / chase risk")
        elif recent > 30:
            trend_labels.append("Recent bounce")
        else:
            trend_labels.append("Neutral")
    df["trend_signal"] = trend_labels

    # Flood/crowding proxy. Popularity is not exact sales, so treat it as pressure/crowdedness.
    group = ["category", "variant"]
    df["crowding_percentile"] = df.groupby(group)["absolute_popularity_pressure"].transform(percentile_rank).fillna(0)
    df["crowding_percentile"] = np.where(df["absolute_popularity_pressure"].fillna(0) > 0, df["crowding_percentile"], 0)
    high_price_interest = (df["hist_avg"].fillna(0) / df["price_tokens"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1)
    high_price_interest_score = ((high_price_interest - 1) / 4).clip(0, 1)
    spike_score = df.groupby(group)["peak_positive_popularity"].transform(percentile_rank).fillna(0)
    spike_score = np.where(df["peak_positive_popularity"].fillna(0) > 0, spike_score, 0)
    popularity_balance = (
        df["positive_popularity_sum"].fillna(0)
        / df["absolute_popularity_pressure"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.5).clip(0, 1)
    popularity_velocity = ((df["popularity_slope_recent"].fillna(0) + 25) / 50).clip(0, 1)
    latest_popularity_score = df.groupby(group)["latest_popularity"].transform(percentile_rank).fillna(0)
    latest_popularity_score = np.where(df["latest_popularity"].notna(), latest_popularity_score, 0.5)
    latest_share_percentile = df.groupby(group)["latest_relative_demand_share"].transform(percentile_rank).fillna(0)
    latest_share_percentile = np.where(df["latest_relative_demand_share"].notna(), latest_share_percentile, 0.5)
    share_trend_score = ((df["relative_demand_share_slope_recent"].fillna(0) + 35) / 70).clip(0, 1)
    share_acceleration_score = ((df["demand_share_acceleration"].fillna(0) + 30) / 60).clip(0, 1)
    price_velocity = df["snapshot_price_velocity_pct_per_day"].fillna(df["price_slope_recent"]).fillna(0)
    demand_price_divergence = (
        (df["relative_demand_share_slope_recent"].fillna(0).clip(lower=0) / 35).clip(0, 1)
        * (price_velocity.clip(upper=0).abs() / 50).clip(0, 1)
    )
    df["demand_price_divergence_score"] = demand_price_divergence.round(4)
    df["falling_demand_penalty"] = (
        (price_velocity.clip(upper=0).abs() / 50).clip(0, 1)
        * (df["relative_demand_share_slope_recent"].fillna(0).clip(upper=0).abs() / 35).clip(0, 1)
    ).round(4)
    df["demand_momentum_score"] = (
        0.24 * popularity_balance
        + 0.18 * popularity_velocity
        + 0.18 * share_trend_score
        + 0.14 * latest_share_percentile
        + 0.10 * latest_popularity_score
        + 0.08 * share_acceleration_score
        + 0.05 * df["demand_price_divergence_score"]
        + 0.03 * df["history_coverage_score"].fillna(0)
    ).clip(0, 1).round(4)
    share_crowding = latest_share_percentile * (0.45 + 0.55 * df["position_in_range"].clip(0, 1))
    flood_raw = (
        0.42 * df["crowding_percentile"]
        + 0.22 * spike_score
        + 0.18 * high_price_interest_score
        + 0.18 * share_crowding
    ).clip(0, 1)
    df["flood_risk_score"] = flood_raw.round(4)

    bins = []
    for score in df["flood_risk_score"]:
        if score >= 0.80:
            bins.append("Extreme")
        elif score >= 0.60:
            bins.append("High")
        elif score >= 0.35:
            bins.append("Medium")
        else:
            bins.append("Low")
    df["flood_risk"] = bins
    return df


def add_signal_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    discount = (df["discount_from_high_pct"].fillna(0) / 100).clip(0, 1)
    relative_discount = ((df["relative_discount_pct"].fillna(0) + 50) / 100).clip(0, 1)
    floor = (1 - df["position_in_range"]).clip(0, 1)
    rank = (1 - df["price_percentile"]).clip(0, 1)
    quality_scaled = ((df["quality_score"] - 5) / 5).clip(0, 1)
    score_confidence = df["score_confidence"].fillna(0).clip(0, 1)
    manual_quality = quality_scaled * (0.35 + 0.65 * score_confidence)
    entry = df["entry_score"].clip(0, 1)
    entry_change = df["entry_change_score"].fillna(entry).clip(0, 1)
    expected = df["expected_return_score"].fillna(0).clip(0, 1)
    demand = df["demand_momentum_score"].fillna(0.5).clip(0, 1)
    downside = df["downside_risk_score"].fillna(0).clip(0, 1)
    divergence = df["demand_price_divergence_score"].fillna(0).clip(0, 1)
    falling_demand = df["falling_demand_penalty"].fillna(0).clip(0, 1)
    note_adjustment = df["note_score_adjustment"].fillna(0).clip(-0.15, 0.10)
    history_coverage = df["history_coverage_score"].fillna(0).clip(0, 1)
    metadata_ok = (
        df["metadata_status"].astype(str).str.lower().eq("ok")
        if "metadata_status" in df.columns
        else pd.Series(True, index=df.index)
    )
    metadata_confidence = np.where(metadata_ok, 1.0, 0.5)
    df["prediction_confidence"] = (
        0.45 * history_coverage
        + 0.35 * score_confidence
        + 0.20 * metadata_confidence
    ).clip(0, 1).round(4)
    uncertainty = (1 - df["prediction_confidence"]).clip(0, 1)

    df["history_score"] = (
        0.18 * discount
        + 0.17 * floor
        + 0.15 * expected
        + 0.13 * relative_discount
        + 0.12 * demand
        + 0.08 * rank
        + 0.08 * df["trend_score"]
        + 0.06 * entry_change
        + 0.03 * divergence
        - 0.10 * downside
        - 0.07 * falling_demand
        - 0.04 * (1 - history_coverage)
    ).clip(0, 1).round(4)

    df["discovery_score"] = (
        0.28 * df["history_score"]
        + 0.20 * df["trend_score"]
        + 0.18 * entry
        + 0.16 * expected
        + 0.13 * demand
        + 0.06 * divergence
        + 0.05 * discount
        - 0.15 * df["flood_risk_score"]
        - 0.08 * falling_demand
        - 0.05 * uncertainty
        + note_adjustment
    ).clip(0, 1).round(4)

    df["decision_score"] = (
        0.18 * manual_quality
        + 0.22 * df["history_score"]
        + 0.15 * df["trend_score"]
        + 0.12 * entry_change
        + 0.12 * expected
        + 0.10 * demand
        + 0.06 * divergence
        + 0.05 * ((df["manual_demand_score"].fillna(6) - 5) / 5).clip(0, 1) * score_confidence
        + 0.04 * ((df["manual_color_craft_score"].fillna(6) - 5) / 5).clip(0, 1) * score_confidence
        - 0.13 * df["flood_risk_score"]
        - 0.08 * df["position_in_range"].clip(0, 1)
        - 0.08 * falling_demand
        - 0.07 * uncertainty
        + note_adjustment
    ).clip(0, 1).round(4)
    df["value_edge_score"] = (
        0.28 * df["history_score"]
        + 0.22 * expected
        + 0.17 * demand
        + 0.12 * entry_change
        + 0.10 * df["trend_score"]
        + 0.08 * divergence
        + 0.09 * rank
        - 0.18 * df["flood_risk_score"]
        - 0.10 * downside
        - 0.08 * falling_demand
        - 0.06 * uncertainty
        + note_adjustment
    ).clip(0, 1).round(4)

    return df


def add_portfolio_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    team = df.get("team", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
    subject = df.get("subject", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
    player = df.get("player_name", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
    fallback = subject.where(subject != "", player)
    df["portfolio_group"] = team.where(team != "", fallback).replace("", "Unknown")
    df["portfolio_group_count"] = df.groupby("portfolio_group")["sticker_id"].transform("count")
    df["portfolio_variant_count"] = df.groupby(["portfolio_group", "variant"])["sticker_id"].transform("count")
    df["team_exposure_score"] = ((df["portfolio_group_count"] - 1) / 12).clip(0, 1).round(4)
    return df



# -----------------------------
# Verdicts, priority ranking, and outputs
# -----------------------------

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


def metric_value(row: pd.Series, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def short_percent(value: float | int | str | None, decimals: int = 0, signed: bool = False) -> str:
    try:
        if pd.isna(value):
            return "n/a"
        value = float(value)
    except Exception:
        return "n/a"
    sign = "+" if signed and value > 0 else ""
    return f"{sign}{value:.{decimals}f}%"


def build_quick_reason(row: pd.Series) -> str:
    """Small table-friendly reason. Full metrics remain in debug_metrics.csv."""
    parts: list[str] = []
    discount = row.get("discount_from_high_pct", np.nan)
    upside = row.get("upside_to_high_pct", np.nan)
    expected = row.get("expected_return_pct", np.nan)
    launch = row.get("launch_gap_pct", np.nan)
    trend = str(row.get("trend_signal", "") or "")
    demand = row.get("demand_momentum_score", np.nan)

    if pd.notna(expected):
        parts.append(f"{expected:+.0f}% expected")
    if pd.notna(demand):
        parts.append(f"{float(demand):.2f} demand")
    if pd.notna(discount):
        parts.append(f"{discount:.0f}% below high")
    if pd.notna(upside):
        parts.append(f"{upside:.0f}% upside")
    if pd.notna(launch):
        parts.append(f"{launch:+.0f}% vs launch")
    if trend and trend != "Neutral":
        # Keep trend readable in the table; details are in separate columns.
        parts.append(trend.replace("Floor + prior bounce, but lower highs", "Floor + bounce, lower highs"))
    if not bool(row.get("scored", False)):
        parts.append("score first")
    if metric_value(row, "prediction_confidence", 0) < 0.45:
        parts.append("low confidence")
    if str(row.get("flood_risk", "")) in {"High", "Extreme"}:
        parts.append(f"{row['flood_risk']} flood")

    return " | ".join(parts[:5])


def build_action_note(row: pd.Series) -> str:
    verdict = str(row.get("verdict", ""))
    if verdict == "CORE BUY CANDIDATE":
        return "Best scored candidate; still size carefully."
    if verdict == "SMALL BUY":
        return "Small position only; not a core hold."
    if verdict == "CHEAP HISTORY PUNT":
        return "Cheap punt; small size is acceptable even before full scoring."
    if verdict == "VISUAL CHECK NOW":
        return "Open preview, score it, then rerun analysis."
    if verdict == "SCORE FIRST":
        return "Potentially interesting; needs manual score."
    if verdict == "WAIT FOR DROP":
        return "Good/interesting sticker but entry is not ideal."
    if verdict == "DO NOT CHASE":
        return "Too extended or too close to high."
    if verdict == "FLOOD RISK":
        return "Crowded setup; avoid unless conviction is very high."
    if verdict == "SCORE/WAIT":
        return "Scored, but edge is not strong enough yet."
    return "No clear edge."


def build_risk_note(row: pd.Series) -> str:
    risks: list[str] = []
    flood = str(row.get("flood_risk", ""))
    if flood in {"High", "Extreme"}:
        risks.append(f"{flood} crowding")
    if metric_value(row, "lower_high_count") >= 1:
        risks.append("lower highs")
    if metric_value(row, "position_in_range", 0.5) >= 0.70:
        risks.append("near range high")
    if metric_value(row, "falling_demand_penalty") >= 0.45:
        risks.append("falling demand")
    if bool(row.get("scored", False)) and metric_value(row, "quality_score") < 7.2:
        risks.append("modest quality score")
    if not bool(row.get("scored", False)):
        risks.append("unscored")
    if metric_value(row, "prediction_confidence", 1) < 0.45:
        risks.append("low confidence")
    return ", ".join(risks) if risks else "manageable"


def suggested_size(row: pd.Series) -> str:
    verdict = row.get("verdict", "")
    price = row.get("price_tokens", np.nan)
    category = str(row.get("category", ""))

    if verdict == "CORE BUY CANDIDATE":
        if price <= 150:
            return "3-8 max"
        if price <= 1000:
            return "2-5 max"
        return "1-2 max"
    if verdict == "SMALL BUY":
        return "1-3 max"
    if verdict == "CHEAP HISTORY PUNT":
        if not bool(row.get("scored", False)):
            return "1-2 max"
        return "1-3 max"
    if verdict in {"VISUAL CHECK NOW", "SCORE FIRST"}:
        return "0 until scored"
    return "0"


def final_verdict(row: pd.Series) -> str:
    scored = bool(row.get("scored", False))
    quality = float(row.get("quality_score", 0) or 0)
    decision = float(row.get("decision_score", 0) or 0)
    discovery = float(row.get("discovery_score", 0) or 0)
    history = float(row.get("history_score", 0) or 0)
    value_edge = float(row.get("value_edge_score", 0) or 0)
    expected = metric_value(row, "expected_return_pct", 0)
    confidence = metric_value(row, "prediction_confidence", 0)
    falling_demand = metric_value(row, "falling_demand_penalty", 0)
    raw_pos = row.get("position_in_range", 0.5)
    pos = 0.5 if pd.isna(raw_pos) else float(raw_pos)
    entry = str(row.get("entry_tier", ""))
    flood = str(row.get("flood_risk", ""))

    # Universal risk guards. They prevent late chase entries.
    if pos >= 0.85 and row.get("discount_from_high_pct", 0) < 25:
        return "DO NOT CHASE"
    if flood == "Extreme" and pos > 0.45:
        return "FLOOD RISK"

    cheap_punt = (
        entry in {"Very Cheap", "Cheap"}
        and value_edge >= 0.45
        and history >= 0.55
        and pos <= 0.40
        and expected >= 15
        and confidence >= 0.35
        and falling_demand < 0.55
        and flood not in {"Extreme"}
    )
    if cheap_punt:
        return "CHEAP HISTORY PUNT"

    if scored:
        if (
            quality >= 8.0
            and decision >= 0.56
            and value_edge >= 0.55
            and history >= 0.62
            and pos <= 0.35
            and falling_demand < 0.35
            and flood not in {"High", "Extreme"}
        ):
            return "CORE BUY CANDIDATE"
        if quality >= 7.2 and decision >= 0.48 and value_edge >= 0.47 and falling_demand < 0.50 and flood not in {"Extreme"}:
            return "SMALL BUY"
        if pos >= 0.70:
            return "WAIT FOR DROP"
        return "SCORE/WAIT"

    # Discovery mode: unscored stickers can be surfaced but never become buys.
    if discovery >= 0.62 and pos <= 0.35:
        return "VISUAL CHECK NOW"
    if discovery >= 0.50 or (history >= 0.70 and pos <= 0.35):
        return "SCORE FIRST"
    if pos >= 0.75:
        return "WAIT FOR DROP"
    return "IGNORE"


def add_priority_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a clear rank inside each verdict so equal verdicts are still prioritized."""
    df = df.copy()
    verdict_bonus = df["verdict"].map({
        "CORE BUY CANDIDATE": 0.10,
        "SMALL BUY": 0.06,
        "CHEAP HISTORY PUNT": 0.02,
        "VISUAL CHECK NOW": 0.00,
        "SCORE FIRST": -0.02,
        "SCORE/WAIT": -0.05,
        "WAIT FOR DROP": -0.12,
        "DO NOT CHASE": -0.22,
        "FLOOD RISK": -0.25,
        "IGNORE": -0.30,
    }).fillna(-0.10)

    quality_scaled = ((pd.to_numeric(df["quality_score"], errors="coerce").fillna(6) - 5) / 5).clip(0, 1)
    flood_good = 1 - pd.to_numeric(df["flood_risk_score"], errors="coerce").fillna(0.5).clip(0, 1)
    floor_good = 1 - pd.to_numeric(df["position_in_range"], errors="coerce").fillna(0.5).clip(0, 1)
    confidence = pd.to_numeric(df["prediction_confidence"], errors="coerce").fillna(0).clip(0, 1)

    raw = (
        0.20 * pd.to_numeric(df["decision_score"], errors="coerce").fillna(0)
        + 0.18 * pd.to_numeric(df["discovery_score"], errors="coerce").fillna(0)
        + 0.18 * pd.to_numeric(df["value_edge_score"], errors="coerce").fillna(0)
        + 0.13 * pd.to_numeric(df["history_score"], errors="coerce").fillna(0)
        + 0.10 * pd.to_numeric(df["trend_score"], errors="coerce").fillna(0)
        + 0.08 * pd.to_numeric(df["entry_score"], errors="coerce").fillna(0)
        + 0.06 * pd.to_numeric(df["demand_momentum_score"], errors="coerce").fillna(0.5)
        + 0.05 * quality_scaled
        + 0.05 * flood_good
        + 0.04 * confidence
        + 0.03 * floor_good
        + verdict_bonus
    ).clip(0, 1)

    df["priority_score"] = (raw * 100).round(1)

    tiers = []
    for score in df["priority_score"]:
        if score >= 72:
            tiers.append("P1")
        elif score >= 62:
            tiers.append("P2")
        elif score >= 52:
            tiers.append("P3")
        else:
            tiers.append("P4")
    df["priority_tier"] = tiers

    # Human-facing rank after final sorting.
    df["verdict_rank"] = df["verdict"].map(VERDICT_ORDER).fillna(99).astype(int)
    df = df.sort_values(
        ["verdict_rank", "priority_score", "value_edge_score", "decision_score", "discovery_score", "history_score"],
        ascending=[True, False, False, False, False, False],
    ).copy()
    df.insert(0, "priority_rank", range(1, len(df) + 1))
    return df


def compact_outputs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["sticker"] = df["name"].map(clean_name)
    df["verdict"] = df.apply(final_verdict, axis=1)
    df["suggested_size"] = df.apply(suggested_size, axis=1)
    df["quick_reason"] = df.apply(build_quick_reason, axis=1)
    df["action_note"] = df.apply(build_action_note, axis=1)
    df["risk_note"] = df.apply(build_risk_note, axis=1)
    df["reason"] = df["quick_reason"]
    df = add_priority_fields(df)

    front_cols = [
        "priority_rank",
        "priority_score",
        "priority_tier",
        "verdict",
        "sticker",
        "category",
        "variant",
        "team",
        "price_tokens",
        "usd_price",
        "suggested_size",
        "entry_tier",
        "flood_risk",
        "quality_score",
        "history_score",
        "decision_score",
        "discovery_score",
        "value_edge_score",
        "trend_score",
        "expected_return_pct",
        "expected_return_score",
        "robust_reference_price",
        "robust_peak_price",
        "discount_from_robust_peak_pct",
        "downside_to_floor_pct",
        "downside_risk_score",
        "demand_momentum_score",
        "demand_price_divergence_score",
        "falling_demand_penalty",
        "prediction_confidence",
        "score_confidence",
        "manual_score_count",
        "history_coverage_score",
        "entry_change_score",
        "discount_from_high_pct",
        "upside_to_high_pct",
        "position_in_range",
        "trend_signal",
        "quick_reason",
        "risk_note",
        "action_note",
        "scored",
        "launch_price",
        "launch_gap_pct",
        "early_avg_price",
        "early_avg_gap_pct",
        "max_previous_bounce_pct",
        "lower_high_count",
        "recent_return_pct",
        "price_slope_full",
        "price_slope_recent",
        "snapshot_points",
        "snapshot_span_hours",
        "snapshot_price_change_pct",
        "snapshot_price_velocity_pct_per_day",
        "snapshot_price_slope",
        "snapshot_price_acceleration",
        "rank_change",
        "rank_percentile_change",
        "rank_improvement_score",
        "price_drop_opportunity_score",
        "positive_popularity_sum",
        "absolute_popularity_pressure",
        "peak_positive_popularity",
        "latest_popularity",
        "avg_relative_demand_share",
        "latest_relative_demand_share",
        "relative_demand_share_change_pct",
        "relative_demand_share_slope_full",
        "relative_demand_share_slope_recent",
        "demand_share_acceleration",
        "avg_positive_demand_share",
        "latest_positive_demand_share",
        "crowding_percentile",
        "entry_score",
        "flood_risk_score",
        "price_percentile",
        "hist_min",
        "hist_max",
        "hist_median",
        "recent_avg_price",
        "hist_points",
        "history_span_hours",
        "last_point_time",
        "visual_score",
        "craft_score",
        "demand_score",
        "color_score",
        "readability_score",
        "notes",
        "note_positive_flag",
        "note_negative_flag",
        "note_hype_flag",
        "note_avoid_flag",
        "note_score_adjustment",
        "portfolio_group",
        "portfolio_group_count",
        "portfolio_variant_count",
        "team_exposure_score",
        "image_formula",
        "image_url",
        "item_url",
        "market_hash_name",
        "steam_market_url",
        "metadata_status",
        "paint_index",
        "volatile_item_id",
        "campaign_id",
        "campaign_slug",
        "token_key",
        "catalog_type",
        "sticker_type",
        "player_name",
        "team_name",
        "rarity_id",
        "sticker_id",
    ]
    for col in front_cols:
        if col not in df.columns:
            df[col] = np.nan

    clean = df[front_cols].copy()

    watch_verdicts = [
        "CORE BUY CANDIDATE",
        "SMALL BUY",
        "CHEAP HISTORY PUNT",
        "VISUAL CHECK NOW",
        "SCORE FIRST",
    ]
    buy_watchlist = clean[clean["verdict"].isin(watch_verdicts)].copy()

    score_targets = clean[
        (~clean["scored"].fillna(False).astype(bool))
        & (clean["verdict"].isin(["CHEAP HISTORY PUNT", "VISUAL CHECK NOW", "SCORE FIRST"]))
    ].copy()

    return clean, buy_watchlist, score_targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    args = parser.parse_args()
    wanted_variants = parse_variants(args.variants)
    if not wanted_variants:
        raise SystemExit("No variants requested. Use --variants Paper,Foil,Holo,Gold for example.")

    out_dir = output_dir()

    snapshots = load_snapshots()
    scores = load_scores()
    history = clean_history(load_history())
    history_summary = summarize_history(history)

    snapshots["timestamp"] = pd.to_datetime(snapshots["timestamp"], errors="coerce")
    snapshots = snapshots.dropna(subset=["timestamp"])
    snapshots["variant"] = snapshots["variant"].map(normalize_variant)
    snapshots = snapshots[snapshots["variant"].isin(wanted_variants)].copy()
    snapshots["price_tokens"] = numeric(snapshots["price_tokens"])
    snapshots["usd_price"] = numeric(snapshots["usd_price"])
    snapshots = snapshots.dropna(subset=["price_tokens"])
    if snapshots.empty:
        raise SystemExit(f"No snapshot rows found for variants: {', '.join(wanted_variants)}")
    snapshot_summary = summarize_snapshot_trends(snapshots)
    validation = build_snapshot_validation(snapshots)

    latest_time = snapshots["timestamp"].max()
    latest = snapshots[snapshots["timestamp"] == latest_time].copy()
    latest = latest.drop_duplicates(subset=["sticker_id"], keep="last")

    latest["rank_low_to_high"] = latest.groupby(["category", "variant"])["price_tokens"].rank(method="first", ascending=True)
    latest["total_in_group"] = latest.groupby(["category", "variant"])["price_tokens"].transform("count")
    latest["price_percentile"] = ((latest["rank_low_to_high"] - 1) / (latest["total_in_group"] - 1)).fillna(0)

    df = latest.merge(snapshot_summary, on="sticker_id", how="left")
    df = df.merge(history_summary, on="sticker_id", how="left")
    df = add_history_metrics(df)
    df = add_scores(df, scores)
    df = add_entry_tier(df)
    df = add_trend_and_flood(df)
    df = add_signal_scores(df)
    df = add_portfolio_fields(df)

    decision_board, buy_watchlist, score_targets = compact_outputs(df)

    decision_path = out_dir / "decision_board.csv"
    watch_path = out_dir / "buy_watchlist_clean.csv"
    score_targets_path = out_dir / "score_targets.csv"
    debug_path = out_dir / "debug_metrics.csv"
    legacy_path = out_dir / "latest_analysis_clean.csv"
    validation_path = out_dir / "model_validation.csv"

    decision_board.to_csv(decision_path, index=False, encoding="utf-8-sig")
    buy_watchlist.to_csv(watch_path, index=False, encoding="utf-8-sig")
    score_targets.to_csv(score_targets_path, index=False, encoding="utf-8-sig")
    df.to_csv(debug_path, index=False, encoding="utf-8-sig")
    decision_board.to_csv(legacy_path, index=False, encoding="utf-8-sig")
    validation.to_csv(validation_path, index=False, encoding="utf-8-sig")

    print(f"Latest snapshot: {latest_time}")
    print(f"History points used: {len(history)}")
    print(f"Stickers with history: {history_summary['sticker_id'].nunique() if not history_summary.empty else 0}")
    print(f"Saved decision board: {decision_path}")
    print(f"Saved watchlist: {watch_path}")
    print(f"Saved score targets: {score_targets_path}")
    print(f"Saved debug metrics: {debug_path}")
    print(f"Saved model validation: {validation_path}")

    display_cols = [
        "priority_rank",
        "priority_score",
        "verdict",
        "sticker",
        "team",
        "price_tokens",
        "usd_price",
        "suggested_size",
        "entry_tier",
        "flood_risk",
        "quality_score",
        "history_score",
        "value_edge_score",
        "demand_momentum_score",
        "expected_return_pct",
        "prediction_confidence",
        "quick_reason",
    ]
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 260)
    print("\nTop decision board:")
    print(decision_board[display_cols].head(40).to_string(index=False))


if __name__ == "__main__":
    main()
