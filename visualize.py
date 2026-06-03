from __future__ import annotations

import argparse
import csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import html
import json
import math
import re
import time
import unicodedata
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
import pandas as pd
import numpy as np

DATA_DIR = Path("data")
ANALYZE_DIR = Path("analyze")
OUT_DIR = Path("visualized")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSGOSKINS_CACHE_PATH = OUT_DIR / "csgoskins_prices.json"
SECOND_MARKET_HISTORY_PATH = OUT_DIR / "second_market_history.csv"
FAVORITES_PATH = OUT_DIR / "favorites.json"
CSGOSKINS_CACHE_TTL_SECONDS = 6 * 60 * 60
CSGOSKINS_ERROR_RETRY_SECONDS = 4 * 60 * 60
CSGOSKINS_WORKERS = 3
CSGOSKINS_GENERATION_FETCH_LIMIT = 90
UUSKINS_STICKER_CATEGORY_ID = 106
MIN_REASONABLE_2P_USD = 0.10
GENERIC_2P_OUTLIER_RATIO = 0.45
SECOND_MARKET_HISTORY_FIELDS = [
    "timestamp",
    "fetched_at",
    "source",
    "sticker_id",
    "market_hash_name",
    "sticker",
    "variant",
    "csgoskins_url",
    "status",
    "low_usd",
    "csfloat_usd",
    "uuskins_usd",
    "csfloat_source",
    "uuskins_source",
]

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
    "CORE BUY CANDIDATE": "#00e676",
    "SMALL BUY": "#9cff2e",
    "CHEAP HISTORY PUNT": "#ffd400",
    "VISUAL CHECK NOW": "#00d5ff",
    "SCORE FIRST": "#a855f7",
    "WAIT FOR DROP": "#ff8a00",
    "DO NOT CHASE": "#ff3b30",
    "FLOOD RISK": "#ff2b6a",
    "SCORE/WAIT": "#c7d2fe",
    "IGNORE": "#6b7280",
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


def trusted_2p_price_candidates(entry: dict[str, object]) -> list[float]:
    """Return marketplace prices after rejecting parser artifacts such as $0.01 UI text."""
    if not isinstance(entry, dict):
        return []
    markets = entry.get("markets") if isinstance(entry.get("markets"), dict) else {}
    market_prices = [
        price
        for price in (safe_float(value, None) for value in markets.values())
        if price is not None and price >= MIN_REASONABLE_2P_USD
    ]
    direct = safe_float(entry.get("price"), None)
    candidates = list(market_prices)

    if direct is not None and direct >= MIN_REASONABLE_2P_USD:
        if market_prices:
            # Generic CSGOSkins page parsing can catch unrelated tiny amounts. Only trust
            # a direct price when it is in the same range as named marketplace offers.
            if direct >= min(market_prices) * GENERIC_2P_OUTLIER_RATIO:
                candidates.append(direct)
        else:
            candidates.append(direct)
    return candidates


def trusted_2p_low(entry: dict[str, object]) -> float | None:
    candidates = trusted_2p_price_candidates(entry)
    return min(candidates) if candidates else None


def safe_bool(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def rarity_for_variant(variant: str) -> str:
    value = str(variant or "").lower()
    if "gold" in value:
        return "rarity_ancient"
    if "holo" in value:
        return "rarity_legendary"
    if "foil" in value:
        return "rarity_mythical"
    return "rarity_rare"


def csgoskins_slug(market_hash_name: str, sticker: str) -> str:
    base = str(market_hash_name or "").strip()
    if not base:
        base = f"Sticker | {sticker} | Cologne 2026"
    normalized = unicodedata.normalize("NFKD", base)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower().replace("&", " and ")
    normalized = normalized.replace("thunderdownunder", "thunder downunder")
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    return normalized.strip("-")


def csgoskins_url(market_hash_name: str, sticker: str) -> str:
    return f"https://csgoskins.gg/items/{csgoskins_slug(market_hash_name, sticker)}"


def skinsniper_url_from_csgoskins(url: str) -> str:
    slug = str(url or "").split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]
    if not slug:
        return ""
    return f"https://skinsniper.com/stickers/{slug}"


def steam_market_url(market_hash_name: str, sticker: str) -> str:
    name = str(market_hash_name or "").strip()
    if not name:
        name = f"Sticker | {sticker} | Cologne 2026"
    return f"https://steamcommunity.com/market/listings/730/{quote(name, safe='')}"


def market_hash_from_csgoskins_url(url: str) -> str:
    slug = str(url or "").split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]
    if slug.startswith("sticker-"):
        slug = slug[len("sticker-"):]
    if slug.endswith("-cologne-2026"):
        slug = slug[: -len("-cologne-2026")]
    variant = ""
    for suffix, label in (("-holo", "Holo"), ("-foil", "Foil"), ("-gold", "Gold")):
        if slug.endswith(suffix):
            slug = slug[: -len(suffix)]
            variant = label
            break
    base = slug.replace("-", " ").strip()
    if variant:
        return f"Sticker | {base} ({variant}) | Cologne 2026"
    return f"Sticker | {base} | Cologne 2026"


def normalize_market_hash(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def read_csgoskins_cache() -> dict[str, dict[str, object]]:
    if not CSGOSKINS_CACHE_PATH.exists():
        return {}
    try:
        data = json.loads(CSGOSKINS_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): v for k, v in data.items() if isinstance(v, dict)}


def write_csgoskins_cache(cache: dict[str, dict[str, object]]) -> None:
    CSGOSKINS_CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def timestamp_iso(value: object) -> str:
    ts = safe_float(value, None)
    if ts is None:
        ts = time.time()
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat().replace("+00:00", "Z")


def second_market_history_row(
    base: dict[str, object],
    entry: dict[str, object],
    source: str = "visualize",
) -> dict[str, object] | None:
    entry = entry if isinstance(entry, dict) else {}
    markets = entry.get("markets") if isinstance(entry.get("markets"), dict) else {}
    market_sources = entry.get("market_sources") if isinstance(entry.get("market_sources"), dict) else {}
    low = trusted_2p_low(entry)
    csfloat = safe_float(markets.get("CSFloat"), None)
    uuskins = safe_float(markets.get("UUSkins"), None)
    if low is None and csfloat is None and uuskins is None:
        return None
    fetched_at = int(safe_float(entry.get("fetched_at"), time.time()) or time.time())
    return {
        "timestamp": timestamp_iso(fetched_at),
        "fetched_at": fetched_at,
        "source": source,
        "sticker_id": str(base.get("sticker_id", "") or ""),
        "market_hash_name": str(base.get("market_hash_name", "") or ""),
        "sticker": str(base.get("sticker", "") or ""),
        "variant": str(base.get("variant", "") or ""),
        "csgoskins_url": str(base.get("csgoskins_url", "") or ""),
        "status": str(entry.get("status", "") or ""),
        "low_usd": low,
        "csfloat_usd": csfloat,
        "uuskins_usd": uuskins,
        "csfloat_source": str(market_sources.get("CSFloat", "") or ""),
        "uuskins_source": str(market_sources.get("UUSkins", "") or ""),
    }


def append_second_market_history(rows: list[dict[str, object]]) -> int:
    clean_rows = [row for row in rows if row]
    if not clean_rows:
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing_keys: set[tuple[str, str, str, str, str]] = set()
    if SECOND_MARKET_HISTORY_PATH.exists():
        try:
            with open(SECOND_MARKET_HISTORY_PATH, newline="", encoding="utf-8-sig") as file:
                for row in csv.DictReader(file):
                    existing_keys.add((
                        str(row.get("fetched_at", "")),
                        str(row.get("csgoskins_url", "")),
                        str(row.get("low_usd", "")),
                        str(row.get("csfloat_usd", "")),
                        str(row.get("uuskins_usd", "")),
                    ))
        except OSError:
            existing_keys = set()

    to_write: list[dict[str, object]] = []
    for row in clean_rows:
        key = (
            str(row.get("fetched_at", "")),
            str(row.get("csgoskins_url", "")),
            "" if row.get("low_usd") is None else str(row.get("low_usd")),
            "" if row.get("csfloat_usd") is None else str(row.get("csfloat_usd")),
            "" if row.get("uuskins_usd") is None else str(row.get("uuskins_usd")),
        )
        if key in existing_keys:
            continue
        existing_keys.add(key)
        to_write.append({field: row.get(field, "") for field in SECOND_MARKET_HISTORY_FIELDS})

    if not to_write:
        return 0
    exists = SECOND_MARKET_HISTORY_PATH.exists() and SECOND_MARKET_HISTORY_PATH.stat().st_size > 0
    with open(SECOND_MARKET_HISTORY_PATH, "a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=SECOND_MARKET_HISTORY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerows(to_write)
    return len(to_write)


def seed_second_market_history_from_records(records: list[dict]) -> int:
    cache = read_csgoskins_cache()
    rows: list[dict[str, object]] = []
    for record in records:
        url = str(record.get("csgoskins_url", "") or "")
        entry = cache.get(url)
        if not isinstance(entry, dict):
            continue
        row = second_market_history_row(record, entry, source="cache_seed")
        if row:
            rows.append(row)
    return append_second_market_history(rows)


def build_second_market_series(records: list[dict]) -> dict[str, list[dict]]:
    if not SECOND_MARKET_HISTORY_PATH.exists():
        return {}

    by_url = {str(record.get("csgoskins_url", "")): str(record.get("sticker_id", "")) for record in records}
    by_name = {str(record.get("market_hash_name", "")).lower(): str(record.get("sticker_id", "")) for record in records}
    valid_ids = {str(record.get("sticker_id", "")) for record in records}
    grouped: dict[str, list[dict]] = {}

    try:
        with open(SECOND_MARKET_HISTORY_PATH, newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            for row in reader:
                sid = str(row.get("sticker_id", "") or "").strip()
                if not sid:
                    sid = by_url.get(str(row.get("csgoskins_url", "") or ""), "")
                if not sid:
                    sid = by_name.get(str(row.get("market_hash_name", "") or "").lower(), "")
                if not sid or sid not in valid_ids:
                    continue

                low = safe_float(row.get("low_usd"), None)
                csfloat = safe_float(row.get("csfloat_usd"), None)
                uuskins = safe_float(row.get("uuskins_usd"), None)
                if low is None and csfloat is None and uuskins is None:
                    continue
                fetched_at = safe_float(row.get("fetched_at"), None)
                time_label = str(row.get("timestamp", "") or "")
                grouped.setdefault(sid, []).append({
                    "time": time_label,
                    "fetched_at": fetched_at,
                    "low": low,
                    "csfloat": csfloat,
                    "uuskins": uuskins,
                    "source": str(row.get("source", "") or ""),
                })
    except OSError:
        return {}

    series: dict[str, list[dict]] = {}
    for sid, points in grouped.items():
        points.sort(key=lambda p: (p.get("fetched_at") is None, p.get("fetched_at") or 0, p.get("time") or ""))
        deduped: dict[tuple[object, object, object, object], dict] = {}
        for point in points:
            key = (
                point.get("fetched_at"),
                point.get("low"),
                point.get("csfloat"),
                point.get("uuskins"),
            )
            deduped[key] = point
        clean = list(deduped.values())
        clean.sort(key=lambda p: (p.get("fetched_at") is None, p.get("fetched_at") or 0, p.get("time") or ""))
        series[sid] = clean[-80:]
    return series


def read_favorites() -> set[str]:
    if not FAVORITES_PATH.exists():
        return set()
    try:
        payload = json.loads(FAVORITES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    values = payload.get("favorites", payload) if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        return set()
    return {str(value).strip() for value in values if str(value).strip()}


def csgoskins_text(raw_html: str) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", raw_html))
    return re.sub(r"\s+", " ", text)


def prices_from_text(text: str) -> list[float]:
    values = [
        safe_float(match.replace(",", ""), None)
        for match in re.findall(r"\$([0-9][0-9,.]*)", text)
    ]
    return [price for price in values if price is not None and price > 0]


def parse_marketplace_offer(text: str, aliases: list[str]) -> float | None:
    lower = text.lower()
    prices: list[float] = []
    for alias in aliases:
        pattern = re.escape(alias.lower())
        for match in re.finditer(pattern, lower):
            segment = text[match.start(): match.start() + 520]
            found = prices_from_text(segment)
            if found:
                prices.append(found[0])
    return min(prices) if prices else None


def parse_csgoskins_prices(raw_html: str) -> dict[str, object]:
    text = csgoskins_text(raw_html)
    normal_match = re.search(r"\bNormal\s*\$([0-9][0-9,.]*)", text, flags=re.IGNORECASE)
    normal_price = safe_float(normal_match.group(1).replace(",", ""), None) if normal_match else None

    active_start = text.lower().find("active offers")
    active_text = text[active_start:active_start + 9000] if active_start >= 0 else text
    markets = {
        "CSFloat": parse_marketplace_offer(active_text, ["CSFloat", "CS Float"]),
        "UUSkins": parse_marketplace_offer(active_text, ["UUSKINS", "UU SKINS", "UUSkins"]),
    }
    markets = {key: value for key, value in markets.items() if value is not None}
    candidates = list(markets.values())
    if normal_price is not None:
        candidates.append(normal_price)
    parsed = {"price": min(candidates) if candidates else None, "markets": markets}
    trusted = trusted_2p_low(parsed)
    parsed["price"] = trusted
    return parsed


def parse_csgoskins_price(raw_html: str) -> float | None:
    return safe_float(parse_csgoskins_prices(raw_html).get("price"), None)


def parse_skinsniper_market_prices(raw_html: str) -> dict[str, float]:
    markets: dict[str, float] = {}
    aliases = {
        "CSFloat": ["CSFloat", "CS Float"],
        "UUSkins": ["UUSkins", "UUSKINS", "UU SKINS"],
    }
    lower_html = raw_html.lower()
    for market_name, market_aliases in aliases.items():
        prices: list[float] = []
        for alias in market_aliases:
            start = 0
            needle = alias.lower()
            while True:
                idx = lower_html.find(needle, start)
                if idx < 0:
                    break
                segment = raw_html[idx: idx + 5200]
                price_match = re.search(
                    r'class="market-price"[^>]*>\s*\$([0-9][0-9,.]*)',
                    segment,
                    flags=re.IGNORECASE,
                )
                if price_match:
                    price = safe_float(price_match.group(1).replace(",", ""), None)
                    if price is not None and price > 0:
                        prices.append(price)
                start = idx + max(1, len(needle))
        if prices:
            markets[market_name] = min(prices)
    return markets


def fetch_skinsniper_markets(csgoskins_item_url: str) -> dict[str, object]:
    url = skinsniper_url_from_csgoskins(csgoskins_item_url)
    if not url:
        return {"markets": {}, "status": "no_skinsniper_url"}
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/125.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    try:
        with urlopen(request, timeout=10) as response:
            raw_html = response.read().decode("utf-8", "ignore")
    except HTTPError as exc:
        return {"markets": {}, "status": f"skinsniper HTTP {exc.code}", "source_url": url}
    except (URLError, TimeoutError, OSError) as exc:
        return {"markets": {}, "status": f"skinsniper {exc.__class__.__name__}", "source_url": url}

    markets = parse_skinsniper_market_prices(raw_html)
    return {
        "markets": markets,
        "status": "ok" if markets else "no_skinsniper_market_detail",
        "source_url": url,
    }


def parse_uuskins_spu_price(payload: dict[str, object], market_hash_name: str) -> float | None:
    target = normalize_market_hash(market_hash_name)
    data = payload.get("data") if isinstance(payload, dict) else {}
    items = data.get("items") if isinstance(data, dict) else []
    if not isinstance(items, list):
        return None
    candidates: list[float] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        names = [
            normalize_market_hash(item.get("hashName", "")),
            normalize_market_hash(item.get("name", "")),
        ]
        if target and target not in names:
            continue
        for field in ("minPrice", "pointMinPrice"):
            price = safe_float(item.get(field), None)
            if price is not None and price > 0:
                candidates.append(price)
    return min(candidates) if candidates else None


def fetch_uuskins_market_price(market_hash_name: str) -> dict[str, object]:
    name = str(market_hash_name or "").strip()
    if not name:
        return {"price": None, "status": "no_uuskins_name"}
    url = "https://api.uuskins.com/api/vertex/commodity/query/spu/list"
    payload = {
        "language": "en",
        "appId": "730",
        "businessType": 1,
        "categoryId": UUSKINS_STICKER_CATEGORY_ID,
        "pageIndex": 1,
        "pageSize": 5,
        "sortType": 2,
        "keyword": name,
    }
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/125.0 Safari/537.36",
            "Accept": "application/json,text/plain,*/*",
            "Content-Type": "application/json",
            "Origin": "https://www.uuskins.com",
            "Referer": f"https://www.uuskins.com/items?search_word={quote(name, safe='')}",
        },
    )
    try:
        with urlopen(request, timeout=10) as response:
            body = response.read().decode("utf-8", "ignore")
        parsed = json.loads(body)
    except HTTPError as exc:
        return {"price": None, "status": f"uuskins HTTP {exc.code}", "source_url": url}
    except (URLError, TimeoutError, OSError) as exc:
        return {"price": None, "status": f"uuskins {exc.__class__.__name__}", "source_url": url}
    except json.JSONDecodeError:
        return {"price": None, "status": "uuskins bad_json", "source_url": url}

    if not isinstance(parsed, dict):
        return {"price": None, "status": "uuskins bad_payload", "source_url": url}
    price = parse_uuskins_spu_price(parsed, name)
    return {
        "price": price,
        "status": "ok" if price is not None else f"uuskins no_match code {parsed.get('code', '-')}",
        "source_url": url,
    }


def has_market_detail(entry: dict[str, object]) -> bool:
    markets = entry.get("markets") if isinstance(entry, dict) else {}
    if not isinstance(markets, dict):
        return False
    has_csfloat = safe_float(markets.get("CSFloat"), None) is not None
    has_uuskins = safe_float(markets.get("UUSkins"), None) is not None
    uuskins_checked = "uuskins_status" in entry
    return has_csfloat and (has_uuskins or uuskins_checked)


def fetch_csgoskins_price(url: str, market_hash_name: str = "", sticker: str = "") -> dict[str, object]:
    resolved_market_hash_name = str(market_hash_name or "").strip() or market_hash_from_csgoskins_url(url)
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/125.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    last_status = "error"
    result: dict[str, object] = {"price": None, "markets": {}, "status": "error", "fetched_at": int(time.time())}
    for attempt in range(2):
        now = int(time.time())
        try:
            with urlopen(request, timeout=8) as response:
                raw_html = response.read().decode("utf-8", "ignore")
            parsed = parse_csgoskins_prices(raw_html)
            price = safe_float(parsed.get("price"), None)
            markets = parsed.get("markets") if isinstance(parsed.get("markets"), dict) else {}
            result = {
                "price": price,
                "markets": markets,
                "market_sources": {key: "CSGOSkins" for key in markets},
                "status": "ok" if price is not None else "no_price",
                "fetched_at": now,
            }
            break
        except HTTPError as exc:
            last_status = f"error: HTTP {exc.code}"
        except (URLError, TimeoutError, OSError) as exc:
            last_status = f"error: {exc.__class__.__name__}"
        if attempt < 1:
            time.sleep(0.35 * (attempt + 1))
    else:
        result = {"price": None, "markets": {}, "status": last_status, "fetched_at": int(time.time())}

    markets = result.get("markets") if isinstance(result.get("markets"), dict) else {}
    if "CSFloat" not in markets or "UUSkins" not in markets:
        fallback = fetch_skinsniper_markets(url)
        fallback_markets = fallback.get("markets") if isinstance(fallback.get("markets"), dict) else {}
        if fallback_markets:
            merged_markets = {**markets, **fallback_markets}
            result["markets"] = merged_markets
            trusted = trusted_2p_low({"price": result.get("price"), "markets": merged_markets})
            result["price"] = trusted if trusted is not None else result.get("price")
            result["market_sources"] = {
                **({} if not isinstance(result.get("market_sources"), dict) else result.get("market_sources")),
                **{key: "SkinSniper" for key in fallback_markets},
            }
            result["fallback_status"] = fallback.get("status")
            result["fallback_url"] = fallback.get("source_url")
            if str(result.get("status", "")).startswith("error"):
                result["last_error"] = result.get("status")
            if result.get("price") is not None:
                result["status"] = "ok"
        elif str(result.get("status", "")).startswith("error"):
            result["fallback_status"] = fallback.get("status")
            result["fallback_url"] = fallback.get("source_url")

    markets = result.get("markets") if isinstance(result.get("markets"), dict) else {}
    if "UUSkins" not in markets and resolved_market_hash_name:
        uuskins = fetch_uuskins_market_price(resolved_market_hash_name)
        uuskins_price = safe_float(uuskins.get("price"), None)
        result["uuskins_status"] = uuskins.get("status")
        if uuskins_price is not None:
            markets = {**markets, "UUSkins": uuskins_price}
            result["markets"] = markets
            trusted = trusted_2p_low({"price": result.get("price"), "markets": markets})
            result["price"] = trusted if trusted is not None else result.get("price")
            result["market_sources"] = {
                **({} if not isinstance(result.get("market_sources"), dict) else result.get("market_sources")),
                "UUSkins": "UUSkins",
            }
            if str(result.get("status", "")).startswith("error"):
                result["last_error"] = result.get("status")
            if result.get("price") is not None:
                result["status"] = "ok"
    return result


def merge_csgoskins_cache_entry(previous: dict[str, object], result: dict[str, object]) -> dict[str, object]:
    previous = previous if isinstance(previous, dict) else {}
    result = result if isinstance(result, dict) else {}
    result_status = str(result.get("status", "error"))
    previous_status = str(previous.get("status", ""))

    if result_status == "ok":
        merged_markets = {
            **(previous.get("markets") if isinstance(previous.get("markets"), dict) else {}),
            **(result.get("markets") if isinstance(result.get("markets"), dict) else {}),
        }
        merged_sources = {
            **(previous.get("market_sources") if isinstance(previous.get("market_sources"), dict) else {}),
            **(result.get("market_sources") if isinstance(result.get("market_sources"), dict) else {}),
        }
        merged = {**previous, **result, "markets": merged_markets, "market_sources": merged_sources}
        low = trusted_2p_low(merged)
        if low is not None:
            merged["price"] = low
        return merged

    if previous_status == "ok":
        merged = {**previous}
        result_markets = result.get("markets") if isinstance(result.get("markets"), dict) else {}
        if result_markets:
            merged["markets"] = {
                **(previous.get("markets") if isinstance(previous.get("markets"), dict) else {}),
                **result_markets,
            }
        result_sources = result.get("market_sources") if isinstance(result.get("market_sources"), dict) else {}
        if result_sources:
            merged["market_sources"] = {
                **(previous.get("market_sources") if isinstance(previous.get("market_sources"), dict) else {}),
                **result_sources,
            }
        merged["last_error"] = result_status
        merged["last_error_at"] = result.get("fetched_at")
        if result.get("fallback_status"):
            merged["fallback_status"] = result.get("fallback_status")
        if result.get("fallback_url"):
            merged["fallback_url"] = result.get("fallback_url")
        low = trusted_2p_low(merged)
        if low is not None:
            merged["price"] = low
        return merged

    low = trusted_2p_low(result)
    if low is not None:
        return {**result, "price": low}
    return result


def enrich_csgoskins_prices(records: list[dict], fetch_stale: bool = True) -> None:
    url_variant: dict[str, str] = {}
    url_favorite: dict[str, bool] = {}
    url_record: dict[str, dict] = {}
    favorites = read_favorites()
    for record in records:
        url = csgoskins_url(str(record.get("market_hash_name", "")), str(record.get("sticker", "")))
        record["csgoskins_url"] = url
        record["csgoskins_low_usd"] = None
        record["csgoskins_markets"] = {}
        record["csfloat_low_usd"] = None
        record["uuskins_low_usd"] = None
        record["csgoskins_status"] = "pending"
        record["csgoskins_market_sources"] = {}
        url_record[url] = record
        variant = str(record.get("variant") or "")
        if "holo" in variant.lower():
            url_variant[url] = "Holo"
        elif "foil" in variant.lower():
            url_variant[url] = "Foil"
        elif "gold" in variant.lower():
            url_variant[url] = "Gold"
        else:
            url_variant[url] = "Paper"
        favorite_keys = {
            str(record.get("sticker_id", "")).strip(),
            str(record.get("market_hash_name", "")).strip(),
            str(record.get("sticker", "")).strip(),
        }
        url_favorite[url] = bool(favorites.intersection(favorite_keys))

    cache = read_csgoskins_cache()
    cache_repaired = False
    for url, entry in list(cache.items()):
        if not isinstance(entry, dict):
            continue
        current = safe_float(entry.get("price"), None)
        trusted = trusted_2p_low(entry)
        if trusted is not None and (current is None or abs(current - trusted) > 0.004):
            entry["price"] = trusted
            cache[url] = entry
            cache_repaired = True
        elif trusted is None and current is not None and current < MIN_REASONABLE_2P_USD:
            entry["price"] = None
            entry["status"] = "no_trusted_price"
            cache[url] = entry
            cache_repaired = True
    now = time.time()
    eligible = {"Holo", "Foil"}
    urls = {
        str(record["csgoskins_url"])
        for record in records
        if record.get("csgoskins_url") and url_variant.get(str(record["csgoskins_url"])) in eligible
    }
    missing = [
        url for url in urls
        if (
            not cache.get(url)
            or (
                cache[url].get("status") == "ok"
                and not has_market_detail(cache[url])
            )
            or (
                cache[url].get("status") != "ok"
                and now - float(cache[url].get("fetched_at", 0) or 0) > CSGOSKINS_ERROR_RETRY_SECONDS
            )
            or now - float(cache[url].get("fetched_at", 0) or 0) > CSGOSKINS_CACHE_TTL_SECONDS
        )
    ]

    force_favorite_urls = sorted([url for url in urls if url_favorite.get(url)])
    target_urls = sorted(set(missing).union(force_favorite_urls)) if fetch_stale else []

    if fetch_stale and target_urls:
        history_rows: list[dict[str, object]] = []
        favorite_urls = sorted([url for url in target_urls if url_favorite.get(url)])
        holo_urls = sorted([url for url in target_urls if not url_favorite.get(url) and url_variant.get(url) == "Holo"])
        foil_urls = sorted([url for url in target_urls if not url_favorite.get(url) and url_variant.get(url) == "Foil"])
        remaining_budget = max(0, CSGOSKINS_GENERATION_FETCH_LIMIT - len(favorite_urls))
        selected_holo = holo_urls[:remaining_budget]
        remaining_budget = max(0, remaining_budget - len(selected_holo))
        selected_foil = foil_urls[:remaining_budget]
        selected_total = len(favorite_urls) + len(selected_holo) + len(selected_foil)
        skipped = len(target_urls) - selected_total
        print(f"Fetching CSGOSkins prices: {selected_total} prioritized URLs of {len(target_urls)} stale/missing/favorite Holo/Foil URLs")
        if skipped > 0:
            print(f"  CSGOSkins skipped this run: {skipped} URLs. Use dashboard fetch buttons to refresh specific stickers.")
        groups = [
            ("favorites", favorite_urls),
            ("holo", selected_holo),
            ("foil", selected_foil),
        ]
        for group_name, group_urls in groups:
            if not group_urls:
                continue
            print(f"  CSGOSkins {group_name}: {len(group_urls)} URLs")
            with ThreadPoolExecutor(max_workers=min(CSGOSKINS_WORKERS, len(group_urls))) as executor:
                future_map = {
                    executor.submit(
                        fetch_csgoskins_price,
                        url,
                        str(url_record.get(url, {}).get("market_hash_name", "")),
                        str(url_record.get(url, {}).get("sticker", "")),
                    ): url
                    for url in group_urls
                }
                for future in as_completed(future_map):
                    url = future_map[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {"price": None, "markets": {}, "status": f"error: {exc.__class__.__name__}", "fetched_at": int(time.time())}
                    cache_entry = merge_csgoskins_cache_entry(cache.get(url, {}), result)
                    cache[url] = cache_entry
                    history_row = second_market_history_row(url_record.get(url, {}), cache_entry, source=f"visualize:{group_name}")
                    if history_row:
                        history_rows.append(history_row)
        write_csgoskins_cache(cache)
        appended = append_second_market_history(history_rows)
        if appended:
            print(f"  2P history points saved: {appended}")
    elif cache_repaired:
        write_csgoskins_cache(cache)

    for record in records:
        url = str(record.get("csgoskins_url"))
        if url_variant.get(url) not in eligible:
            record["csgoskins_low_usd"] = None
            record["csgoskins_status"] = "not_requested"
            continue
        entry = cache.get(url, {})
        markets = entry.get("markets") if isinstance(entry.get("markets"), dict) else {}
        record["csgoskins_low_usd"] = trusted_2p_low(entry)
        record["csgoskins_markets"] = markets
        record["csfloat_low_usd"] = safe_float(markets.get("CSFloat"), None)
        record["uuskins_low_usd"] = safe_float(markets.get("UUSkins"), None)
        record["csgoskins_status"] = str(entry.get("status", "not_cached"))
        record["csgoskins_market_sources"] = entry.get("market_sources") if isinstance(entry.get("market_sources"), dict) else {}


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
    market_hash_name = str(val("market_hash_name", ""))
    steam_url = str(val("steam_market_url", "")).strip() or steam_market_url(market_hash_name, name)
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
        "latest_popularity": safe_float(val("latest_popularity"), None),
        "positive_popularity_sum": safe_float(val("positive_popularity_sum"), None),
        "absolute_popularity_pressure": safe_float(val("absolute_popularity_pressure"), None),
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
        "second_market_latest_low_usd": safe_float(val("second_market_latest_low_usd"), None),
        "second_market_previous_low_usd": safe_float(val("second_market_previous_low_usd"), None),
        "second_market_low_usd": safe_float(val("second_market_low_usd"), None),
        "second_market_high_usd": safe_float(val("second_market_high_usd"), None),
        "second_market_change_pct": safe_float(val("second_market_change_pct"), None),
        "second_market_slope_pct": safe_float(val("second_market_slope_pct"), None),
        "second_market_true_edge_pct": safe_float(val("second_market_true_edge_pct"), None),
        "second_market_discount_to_steam_pct": safe_float(val("second_market_discount_to_steam_pct"), None),
        "second_market_wait_penalty": safe_float(val("second_market_wait_penalty"), None),
        "second_market_score": safe_float(val("second_market_score"), None),
        "second_market_points": safe_float(val("second_market_points"), None),
        "second_market_new_low": safe_bool(val("second_market_new_low", False)),
        "csfloat_latest_usd": safe_float(val("csfloat_latest_usd"), None),
        "uuskins_latest_usd": safe_float(val("uuskins_latest_usd"), None),
        "quick_reason": short_text(str(val("quick_reason", val("reason", ""))), 240),
        "risk_note": short_text(str(val("risk_note", "")), 180),
        "action_note": short_text(str(val("action_note", val("suggested_size", ""))), 220),
        "notes": short_text(str(val("notes", "")), 220),
        "scored": safe_bool(val("scored", False)),
        "image_url": str(val("image_url", "")),
        "item_url": item_url,
        "market_hash_name": market_hash_name,
        "steam_market_url": steam_url,
        "metadata_status": str(val("metadata_status", "")),
        "rarity_id": rarity_for_variant(variant),
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
        "second_market_latest_low_usd", "second_market_previous_low_usd", "second_market_low_usd",
        "second_market_high_usd", "second_market_change_pct", "second_market_slope_pct",
        "second_market_true_edge_pct", "second_market_discount_to_steam_pct",
        "second_market_wait_penalty", "second_market_score", "second_market_points",
        "second_market_new_low", "csfloat_latest_usd", "uuskins_latest_usd",
        "item_url", "image_url", "market_hash_name", "steam_market_url", "metadata_status",
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
let modalHistoryOpen = false;
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


def build_html(records: list[dict], series: dict[str, list[dict]], second_market_series: dict[str, list[dict]]) -> str:
    data_json = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    series_json = json.dumps(series, ensure_ascii=False).replace("</", "<\\/")
    second_market_json = json.dumps(second_market_series, ensure_ascii=False).replace("</", "<\\/")
    favorites_json = json.dumps(sorted(read_favorites()), ensure_ascii=False).replace("</", "<\\/")

    template = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>CS2 Sticker Decision Dashboard</title>
<style>
  :root {
    --bg:#050506;
    --panel:#111113;
    --panel-2:#17181b;
    --panel-3:#202126;
    --line:rgba(255,255,255,.12);
    --line-soft:rgba(255,255,255,.075);
    --text:#f4f7fb;
    --muted:#a8adb7;
    --faint:#717783;
    --blue:#2f7dff;
    --green:#00e676;
    --yellow:#ffd400;
    --red:#ff3b5f;
    --shadow:0 18px 46px rgba(0,0,0,.42);
  }
  * { box-sizing:border-box; }
  html { color-scheme:dark; }
  body {
    margin:0;
    background:
      linear-gradient(180deg, #101014 0, #08080a 290px, var(--bg) 100%),
      linear-gradient(90deg, rgba(47,125,255,.08), rgba(168,85,247,.055) 36%, rgba(255,43,214,.05) 67%, rgba(255,180,0,.055)),
      repeating-linear-gradient(90deg, rgba(255,255,255,.022) 0 1px, transparent 1px 86px),
      repeating-linear-gradient(0deg, rgba(255,255,255,.015) 0 1px, transparent 1px 86px),
      var(--bg);
    color:var(--text);
    font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size:14px;
    letter-spacing:0;
  }
  a { color:inherit; text-decoration:none; }
  .rarity-accent {
    --rarity:#2f7dff;
    --rarity2:#35e4ff;
    --rarity-bg:rgba(47,125,255,.22);
    --rarity-soft:rgba(47,125,255,.11);
  }
  button, input, select {
    font:inherit;
    color:var(--text);
    background:#111216;
    border:1px solid var(--line);
    border-radius:6px;
    min-height:36px;
  }
  input, select { width:100%; padding:8px 10px; }
  button { padding:8px 12px; cursor:pointer; font-weight:750; }
  button:hover, a.action:hover { border-color:rgba(0,213,255,.58); background:#191b21; }
  input:focus, select:focus { outline:2px solid rgba(0,213,255,.22); border-color:#00d5ff; }
  .app { width:min(1840px, calc(100vw - 28px)); margin:0 auto; padding:16px 0 28px; }
  .topbar {
    display:grid;
    grid-template-columns:minmax(320px,1fr) auto;
    gap:18px;
    align-items:start;
    padding:18px 20px;
    background:
      linear-gradient(135deg, rgba(20,30,44,.98), rgba(10,16,25,.97) 62%, rgba(20,27,42,.98)),
      linear-gradient(90deg, rgba(91,140,255,.10), rgba(52,211,153,.06));
    border:1px solid rgba(169,180,196,.18);
    border-radius:12px;
    box-shadow:var(--shadow), inset 0 1px 0 rgba(255,255,255,.04);
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
    background:linear-gradient(180deg, rgba(18,26,38,.96), rgba(12,18,28,.96));
    border:1px solid rgba(169,180,196,.16);
    border-radius:12px;
    box-shadow:var(--shadow), inset 0 1px 0 rgba(255,255,255,.035);
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
    --rarity:#5b8cff;
    --rarity2:#9cc3ff;
    --rarity-bg:rgba(91,140,255,.14);
    --rarity-soft:rgba(91,140,255,.08);
    position:relative;
    display:grid;
    grid-template-rows:auto minmax(0,1fr) auto;
    gap:8px;
    min-width:0;
    min-height:262px;
    padding:10px;
    border:1px solid color-mix(in srgb, var(--rarity) 26%, rgba(169,180,196,.16));
    border-radius:10px;
    background:
      linear-gradient(180deg, var(--rarity-soft), transparent 42%),
      linear-gradient(180deg, rgba(21,29,42,.98), rgba(10,15,23,.99));
    color:var(--text);
    text-align:left;
    cursor:pointer;
    font:inherit;
    overflow:hidden;
    box-shadow:0 14px 30px rgba(0,0,0,.24), inset 0 1px 0 rgba(255,255,255,.04);
    transition:transform .16s ease, border-color .16s ease, box-shadow .16s ease, background-color .16s ease;
  }
  .grid-card:hover,
  .grid-card:focus-visible {
    transform:translateY(-2px);
    border-color:color-mix(in srgb, var(--rarity) 62%, white 6%);
    box-shadow:0 20px 40px rgba(0,0,0,.34), 0 0 0 1px var(--rarity-bg), inset 0 1px 0 rgba(255,255,255,.05);
    outline:none;
  }
  .grid-card::after {
    content:"";
    position:absolute;
    left:0;
    right:0;
    bottom:0;
    height:3px;
    background:linear-gradient(90deg, var(--rarity), var(--rarity2));
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
    background:linear-gradient(135deg, var(--rarity), var(--rarity2));
    color:#07101b;
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
    border-radius:9px;
    background:linear-gradient(180deg, rgba(255,255,255,.045), var(--rarity-soft));
    box-shadow:inset 0 0 0 1px rgba(255,255,255,.035);
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
  .grid-variant { flex:0 0 auto; padding:2px 5px; border:1px solid color-mix(in srgb, var(--rarity) 45%, var(--line)); border-radius:999px; color:#d8e1ee; background:var(--rarity-soft); }
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
  .grid-view[data-density="dense"] {
    gap:8px;
    padding:10px;
  }
  .grid-view[data-density="dense"] .grid-card { min-height:190px; gap:6px; padding:7px; border-radius:7px; }
  .grid-view[data-density="dense"] .grid-rank,
  .grid-view[data-density="dense"] .grid-tier {
    top:6px;
    min-height:20px;
    padding:2px 5px;
    font-size:10px;
  }
  .grid-view[data-density="dense"] .grid-rank { left:6px; }
  .grid-view[data-density="dense"] .grid-tier { right:6px; }
  .grid-view[data-density="dense"] .grid-image { padding:12px 3px 2px; }
  .grid-view[data-density="dense"] .grid-name { font-size:11px; }
  .grid-view[data-density="dense"] .grid-meta,
  .grid-view[data-density="dense"] .grid-kpi small { display:none; }
  .grid-view[data-density="dense"] .grid-kpis { grid-template-columns:1fr 1fr; }
  .grid-view[data-density="dense"] .grid-kpi:last-child { display:none; }
  .grid-view[data-density="dense"] .grid-verdict-pill { padding:4px 5px; font-size:8.5px; }
  .grid-view[data-density="dense"] .grid-price { font-size:12px; }
  .grid-view[data-density="dense"] .grid-kpi { padding:3px 4px; }
  .grid-view[data-density="dense"] .grid-kpi b { font-size:10px; gap:3px; }
  .grid-view[data-density="dense"] .signal-dot { width:6px; height:6px; }
  .grid-view[data-density="ultra"] {
    gap:6px;
    padding:8px;
  }
  .grid-view[data-density="ultra"] .grid-card { min-height:142px; gap:4px; padding:6px; border-radius:6px; }
  .grid-view[data-density="ultra"] .grid-rank,
  .grid-view[data-density="ultra"] .grid-tier {
    top:5px;
    min-height:18px;
    padding:2px 5px;
    font-size:9px;
  }
  .grid-view[data-density="ultra"] .grid-rank { left:5px; }
  .grid-view[data-density="ultra"] .grid-tier { right:5px; }
  .grid-view[data-density="ultra"] .grid-image { padding:11px 2px 1px; }
  .grid-view[data-density="ultra"] .grid-name { font-size:9.5px; line-height:1.1; }
  .grid-view[data-density="ultra"] .grid-meta { display:none; }
  .grid-view[data-density="ultra"] .grid-kpis { display:none; }
  .grid-view[data-density="ultra"] .grid-verdict { display:block; }
  .grid-view[data-density="ultra"] .grid-verdict-pill { display:block; margin-bottom:4px; padding:3px 4px; font-size:7.5px; }
  .grid-view[data-density="ultra"] .grid-price { font-size:10.5px; }
  .grid-empty { grid-column:1 / -1; padding:38px; text-align:center; color:var(--muted); }
  .portfolio-focus-grid {
    display:grid;
    grid-template-columns:repeat(4, minmax(0,1fr));
    gap:10px;
    padding:14px;
  }
  .focus-card {
    --rarity:#5b8cff;
    --rarity2:#9cc3ff;
    --rarity-bg:rgba(91,140,255,.14);
    --rarity-soft:rgba(91,140,255,.08);
    display:grid;
    grid-template-columns:72px minmax(0,1fr);
    gap:10px;
    align-items:center;
    padding:10px;
    border:1px solid color-mix(in srgb, var(--rarity) 28%, rgba(169,180,196,.15));
    border-radius:10px;
    background:linear-gradient(145deg, var(--rarity-soft), rgba(10,16,25,.96) 56%);
    box-shadow:inset 0 -2px 0 var(--rarity-bg);
  }
  .focus-card img { width:72px; height:72px; object-fit:contain; filter:drop-shadow(0 10px 14px rgba(0,0,0,.30)); }
  .focus-title { display:flex; align-items:center; justify-content:space-between; gap:8px; min-width:0; }
  .focus-title b { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:#fff; font-size:13px; }
  .focus-rank { flex:0 0 auto; color:#dce8ff; font-size:11px; font-weight:950; }
  .focus-note { margin-top:5px; color:var(--muted); font-size:11px; line-height:1.35; }
  .focus-meta { display:flex; flex-wrap:wrap; gap:5px; margin-top:7px; }
  .focus-chip { padding:3px 6px; border-radius:999px; border:1px solid var(--line); background:#0d1420; color:#d8e1ee; font-size:10px; font-weight:850; }
  .focus-empty { padding:18px 16px; color:var(--muted); line-height:1.45; }
  .inventory-shell > summary { list-style:none; cursor:pointer; }
  .inventory-shell > summary::-webkit-details-marker { display:none; }
  .recommendations-shell > summary { list-style:none; cursor:pointer; }
  .recommendations-shell > summary::-webkit-details-marker { display:none; }
  .collapse-cue {
    flex:0 0 auto;
    color:var(--muted);
    font-size:11px;
    font-weight:900;
    text-transform:uppercase;
  }
  .recommendations-shell[open] .collapse-cue::after { content:"Collapse"; }
  .recommendations-shell:not([open]) .collapse-cue::after { content:"Expand"; }
  .inventory-summary-stats { display:flex; flex-wrap:wrap; justify-content:flex-end; gap:8px; }
  .inventory-stat {
    min-width:96px;
    padding:8px 10px;
    border:1px solid var(--line);
    border-radius:7px;
    background:rgba(13,20,32,.78);
  }
  .inventory-stat span { display:block; color:var(--muted); font-size:10px; margin-bottom:3px; text-transform:uppercase; }
  .inventory-stat b { display:block; color:#fff; font-size:15px; font-variant-numeric:tabular-nums; }
  .inventory-body { display:grid; gap:12px; padding:14px; border-top:1px solid var(--line-soft); }
  .inventory-form {
    display:grid;
    grid-template-columns:minmax(240px,1.4fr) minmax(86px,.42fr) repeat(4, minmax(112px,.68fr)) minmax(180px,1fr) auto;
    gap:10px;
    align-items:end;
  }
  .inventory-rate-note {
    grid-column:1 / -1;
    display:flex;
    flex-wrap:wrap;
    align-items:center;
    gap:8px;
    margin-bottom:-2px;
    color:#aebeda;
    font-size:12px;
  }
  .inventory-rate-note b { color:#edf4ff; font-variant-numeric:tabular-nums; }
  .inventory-form .field.notes-field { grid-column:auto; }
  .inventory-form-actions { display:flex; gap:7px; align-items:end; }
  .inventory-toolbar { display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:10px; }
  .inventory-toolbar-left { display:flex; flex-wrap:wrap; align-items:center; gap:9px; }
  .inventory-toolbar-right { display:flex; flex-wrap:wrap; align-items:center; justify-content:flex-end; gap:8px; }
  .inventory-grid-controls {
    display:flex;
    align-items:center;
    gap:6px;
    padding:3px;
    border:1px solid rgba(169,180,196,.15);
    border-radius:8px;
    background:rgba(8,13,20,.48);
  }
  .inventory-grid-controls label { color:var(--muted); font-size:11px; font-weight:850; padding-left:6px; }
  .inventory-grid-controls select { width:auto; min-width:108px; min-height:30px; padding:5px 8px; }
  .inventory-grid-controls input { width:72px; min-height:30px; padding:5px 8px; }
  .inventory-filter-input { width:210px; }
  .inventory-account-filter { width:160px; }
  .inventory-sort-filter { width:170px; }
  .inventory-status { color:var(--muted); font-size:12px; }
  .inventory-status.ok { color:#86efac; }
  .inventory-status.warn { color:#f8dfa5; }
  .inventory-status.error { color:#fecdd3; }
  .inventory-ops {
    display:grid;
    grid-template-columns:minmax(0,1.25fr) minmax(0,1fr);
    gap:10px;
  }
  .inventory-op {
    border:1px solid rgba(169,180,196,.14);
    border-radius:8px;
    background:rgba(8,13,20,.38);
  }
  .inventory-op > summary {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:10px;
    padding:10px 12px;
    cursor:pointer;
    list-style:none;
    color:#edf4ff;
    font-size:13px;
    font-weight:900;
  }
  .inventory-op > summary::-webkit-details-marker { display:none; }
  .inventory-op > summary span { color:var(--muted); font-size:11px; font-weight:750; }
  .inventory-op-body { display:grid; gap:10px; padding:0 12px 12px; }
  .inventory-op-grid {
    display:grid;
    grid-template-columns:repeat(4, minmax(0,1fr));
    gap:8px;
    align-items:end;
  }
  .inventory-op-grid.three { grid-template-columns:repeat(3, minmax(0,1fr)); }
  .inventory-op textarea {
    width:100%;
    min-height:112px;
    resize:vertical;
    padding:9px 10px;
    border:1px solid var(--line);
    border-radius:7px;
    background:#0f1724;
    color:#edf4ff;
    font:12px/1.45 Consolas, "SFMono-Regular", monospace;
  }
  .inventory-op-actions { display:flex; flex-wrap:wrap; align-items:center; gap:8px; }
  .inventory-op-hint { color:var(--muted); font-size:11px; line-height:1.35; }
  .check-row {
    display:flex;
    align-items:center;
    gap:7px;
    color:#c9d5e5;
    font-size:12px;
    font-weight:800;
  }
  .check-row input { width:16px; min-height:16px; }
  .inventory-selection-bar {
    display:flex;
    flex-wrap:wrap;
    align-items:center;
    justify-content:space-between;
    gap:10px;
    padding:8px 10px;
    border:1px solid rgba(169,180,196,.13);
    border-radius:8px;
    background:rgba(13,20,32,.62);
  }
  .inventory-selection-actions { display:flex; flex-wrap:wrap; align-items:center; gap:7px; }
  .inventory-selected-count { color:#dbeafe; font-size:12px; font-weight:900; }
  .inventory-drawer-stats {
    display:grid;
    grid-template-columns:repeat(4, minmax(0,1fr));
    gap:8px;
    padding:10px 12px;
    border-top:1px solid rgba(169,180,196,.12);
    border-bottom:1px solid rgba(169,180,196,.12);
    background:rgba(8,13,20,.34);
  }
  .inventory-drawer-stat {
    min-width:0;
    padding:9px 10px;
    border:1px solid rgba(169,180,196,.14);
    border-radius:10px;
    background:rgba(255,255,255,.035);
  }
  .inventory-drawer-stat span {
    display:block;
    color:var(--muted);
    font-size:10px;
    font-weight:900;
    text-transform:uppercase;
  }
  .inventory-drawer-stat b {
    display:block;
    margin-top:4px;
    color:#fff;
    font-size:16px;
    font-weight:950;
    font-variant-numeric:tabular-nums;
  }
  .inventory-context-panel {
    display:grid;
    grid-template-columns:repeat(4, minmax(0,1fr));
    gap:8px;
    margin:12px 0;
    padding:10px;
    border:1px solid rgba(169,180,196,.16);
    border-radius:12px;
    background:rgba(13,20,32,.72);
  }
  .inventory-context-panel .inventory-context-item {
    min-width:0;
    padding:8px 9px;
    border-radius:9px;
    background:rgba(255,255,255,.04);
  }
  .inventory-context-panel span {
    display:block;
    color:var(--muted);
    font-size:10px;
    font-weight:900;
    text-transform:uppercase;
  }
  .inventory-context-panel b {
    display:block;
    margin-top:4px;
    color:#fff;
    font-size:14px;
    font-weight:950;
    font-variant-numeric:tabular-nums;
    overflow-wrap:anywhere;
  }
  .inventory-context-panel small {
    display:block;
    margin-top:3px;
    color:#9eabbf;
    font-size:11px;
    line-height:1.25;
  }
  .inventory-list-wrap { overflow:auto; border:1px solid rgba(169,180,196,.13); border-radius:8px; }
  .inventory-table { min-width:1040px; }
  .inventory-table th { position:static; }
  .inventory-table td { padding:10px; }
  .inventory-table .select-col { width:46px; text-align:center; }
  .inventory-select-cell { text-align:center; }
  .inventory-select {
    width:18px;
    height:18px;
    min-height:18px;
    accent-color:#34d399;
    cursor:pointer;
  }
  .inventory-table tr.inventory-selected-row,
  .inventory-card.inventory-selected-card {
    box-shadow:inset 0 0 0 1px rgba(52,211,153,.42);
    border-color:rgba(52,211,153,.36);
  }
  .inventory-sticker-cell { display:grid; grid-template-columns:58px minmax(0,1fr); gap:9px; align-items:center; }
  .inventory-sticker-cell img { width:58px; height:58px; object-fit:contain; }
  .inventory-item-title { color:#fff; font-weight:900; line-height:1.2; }
  .inventory-item-sub { color:var(--muted); font-size:11px; margin-top:3px; }
  .inventory-actions { display:flex; flex-wrap:wrap; gap:6px; }
  .mini-btn {
    min-height:28px;
    padding:5px 8px;
    border-radius:6px;
    font-size:11px;
    font-weight:850;
  }
  .mini-btn.danger { color:#fecdd3; border-color:rgba(251,113,133,.36); }
  .inventory-grid-view {
    display:grid;
    grid-template-columns:repeat(auto-fill, minmax(160px,1fr));
    gap:12px;
    perspective:900px;
  }
  .inventory-grid-view[hidden], .inventory-list-wrap[hidden] { display:none !important; }
  .inventory-card {
    --rarity:#5b8cff;
    --rarity2:#9cc3ff;
    --rarity-bg:rgba(91,140,255,.14);
    --rarity-soft:rgba(91,140,255,.08);
    position:relative;
    display:grid;
    grid-template-rows:minmax(150px,.68fr) auto auto auto;
    min-width:0;
    min-height:352px;
    gap:8px;
    padding:10px;
    border:1px solid color-mix(in srgb, var(--rarity) 30%, rgba(169,180,196,.14));
    border-radius:12px;
    background:
      linear-gradient(180deg, rgba(255,255,255,.045), transparent 18%),
      linear-gradient(180deg, var(--rarity-soft), transparent 52%),
      linear-gradient(180deg, rgba(20,27,39,.98), rgba(9,14,22,.99));
    box-shadow:0 16px 32px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.05);
    overflow:hidden;
    transform:translateZ(0);
    transition:transform .16s ease, border-color .16s ease, box-shadow .16s ease;
  }
  .inventory-card::before {
    content:"";
    position:absolute;
    left:0;
    right:0;
    bottom:0;
    height:4px;
    background:linear-gradient(90deg, var(--rarity), var(--rarity2));
    z-index:2;
  }
  .inventory-card:hover {
    transform:translateY(-3px) rotateX(1deg);
    border-color:color-mix(in srgb, var(--rarity) 64%, white 4%);
    box-shadow:0 22px 42px rgba(0,0,0,.38), 0 0 0 1px var(--rarity-bg), inset 0 1px 0 rgba(255,255,255,.06);
  }
  .inventory-card-select {
    position:absolute;
    top:8px;
    right:8px;
    z-index:2;
    display:inline-flex;
    align-items:center;
    justify-content:center;
    width:30px;
    height:30px;
    padding:0;
    border:1px solid rgba(169,180,196,.20);
    border-radius:999px;
    background:rgba(8,13,20,.72);
    color:#dbeafe;
    backdrop-filter:blur(8px);
  }
  .inventory-card-select input { margin:0; }
  .inventory-card-art {
    position:relative;
    display:flex;
    align-items:center;
    justify-content:center;
    min-height:142px;
    height:100%;
    padding:10px 12px 6px;
    border-radius:10px;
    background:
      linear-gradient(180deg, rgba(255,255,255,.065), rgba(255,255,255,.01)),
      linear-gradient(180deg, var(--rarity-bg), rgba(8,13,20,.12));
    box-shadow:inset 0 0 0 1px rgba(255,255,255,.04);
  }
  .inventory-card-art img {
    width:100%;
    height:100%;
    object-fit:contain;
    filter:drop-shadow(0 14px 18px rgba(0,0,0,.38));
    transition:transform .18s ease;
  }
  .inventory-card:hover .inventory-card-art img { transform:scale(1.045); }
  .inventory-card-title {
    display:block;
    min-width:0;
    overflow:visible;
    text-overflow:clip;
    white-space:normal;
    color:#fff;
    font-size:13px;
    font-weight:950;
    line-height:1.22;
  }
  .inventory-card-meta { display:flex; align-items:center; gap:5px; min-width:0; color:var(--muted); font-size:11px; margin-top:5px; }
  .inventory-card-meta span { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .inventory-card-market {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:6px;
    margin-top:8px;
  }
  .inventory-current-price { color:#fff; font-size:18px; font-weight:950; line-height:1; font-variant-numeric:tabular-nums; }
  .inventory-cost-pill {
    display:inline-flex;
    flex-direction:column;
    align-items:flex-start;
    justify-content:center;
    min-height:24px;
    padding:3px 7px;
    border-radius:8px;
    background:rgba(255,255,255,.045);
    border:1px solid rgba(169,180,196,.13);
    color:#d8e1ee;
    font-size:9px;
    font-weight:850;
    line-height:1.05;
  }
  .inventory-cost-pill b {
    color:#fff;
    font-size:11px;
    font-variant-numeric:tabular-nums;
  }
  .inventory-pnl-pill {
    display:inline-flex;
    align-items:center;
    min-height:24px;
    padding:3px 7px;
    border-radius:999px;
    background:rgba(8,13,20,.58);
    border:1px solid rgba(169,180,196,.14);
    font-size:11px;
    font-weight:950;
    font-variant-numeric:tabular-nums;
  }
  .inventory-guard-badge {
    display:inline-flex;
    align-items:center;
    width:max-content;
    max-width:100%;
    min-height:20px;
    margin-top:4px;
    padding:3px 7px;
    border-radius:999px;
    font-size:10px;
    font-weight:950;
    letter-spacing:.01em;
    font-variant-numeric:tabular-nums;
    border:1px solid rgba(255,255,255,.14);
    color:#dbeafe;
    background:rgba(148,163,184,.12);
  }
  .inventory-guard-badge.danger {
    color:#fecaca;
    background:rgba(239,68,68,.16);
    border-color:rgba(248,113,113,.35);
  }
  .inventory-guard-badge.good {
    color:#bbf7d0;
    background:rgba(34,197,94,.15);
    border-color:rgba(74,222,128,.32);
  }
  .inventory-guard-badge.neutral {
    color:#dbeafe;
    background:rgba(96,165,250,.13);
    border-color:rgba(147,197,253,.28);
  }
  .inventory-card-subrow {
    display:grid;
    grid-template-columns:1fr;
    align-items:start;
    gap:7px;
    color:var(--muted);
    font-size:11px;
    font-weight:800;
    min-width:0;
  }
  .inventory-card-subrow span { min-width:0; overflow:visible; text-overflow:clip; white-space:normal; line-height:1.25; }
  .inventory-card-actions {
    display:flex;
    gap:5px;
    opacity:1;
    transform:none;
    margin-top:auto;
    transition:opacity .14s ease, transform .14s ease;
  }
  .inventory-card:hover .inventory-card-actions,
  .inventory-card:focus-within .inventory-card-actions { opacity:1; transform:translateY(0); }
  .inventory-card-actions .mini-btn { flex:1; min-height:26px; padding:4px 6px; font-size:10px; }
  .inventory-grid-view[data-density="dense"] { gap:9px; }
  .inventory-grid-view[data-density="dense"] .inventory-card { min-height:322px; padding:8px; border-radius:10px; }
  .inventory-grid-view[data-density="dense"] .inventory-card-title { font-size:11.2px; line-height:1.2; }
  .inventory-grid-view[data-density="dense"] .inventory-current-price { font-size:15px; }
  .inventory-grid-view[data-density="dense"] .inventory-card-art { min-height:134px; padding:8px 7px 5px; }
  .inventory-grid-view[data-density="ultra"] { gap:7px; }
  .inventory-grid-view[data-density="ultra"] .inventory-card { min-height:292px; padding:6px; border-radius:8px; }
  .inventory-grid-view[data-density="ultra"] .inventory-card-art { min-height:112px; padding:7px 4px 4px; border-radius:7px; }
  .inventory-grid-view[data-density="ultra"] .inventory-card-select { width:24px; height:24px; top:6px; right:6px; }
  .inventory-grid-view[data-density="ultra"] .inventory-card-title { font-size:10px; }
  .inventory-grid-view[data-density="ultra"] .inventory-card-meta { display:flex; font-size:9.5px; }
  .inventory-grid-view[data-density="ultra"] .inventory-card-subrow { display:grid; font-size:9px; }
  .inventory-grid-view[data-density="ultra"] .inventory-current-price { font-size:13px; }
  .inventory-grid-view[data-density="ultra"] .inventory-pnl-pill { min-height:20px; padding:2px 5px; font-size:9px; }
  .inventory-grid-view[data-density="dense"] .inventory-card-actions .mini-btn,
  .inventory-grid-view[data-density="ultra"] .inventory-card-actions .mini-btn { min-height:24px; padding:3px 5px; font-size:9px; }
  .inventory-pnl { font-weight:950; }
  .inventory-pnl.pos { color:#86efac !important; }
  .inventory-pnl.neg { color:#fecdd3 !important; }
  .inventory-pnl.flat { color:#d8e1ee !important; }
  .inventory-empty { padding:18px; color:var(--muted); text-align:center; border:1px dashed rgba(169,180,196,.18); border-radius:8px; }
  table { width:100%; border-collapse:separate; border-spacing:0; table-layout:fixed; }
  col.rank-col { width:64px; }
  col.sticker-col { width:26%; }
  col.price-col { width:14%; }
  col.decision-col { width:14%; }
  col.edge-col { width:15%; }
  col.market-col { width:15%; }
  col.notes-col { width:12%; }
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
  tbody tr {
    --rarity:#5b8cff;
    --rarity2:#9cc3ff;
    --rarity-bg:rgba(91,140,255,.14);
    --rarity-soft:rgba(91,140,255,.08);
    background:#101722;
    transition:background-color .14s ease, box-shadow .14s ease;
  }
  tbody tr:nth-child(even) { background:#0e1520; }
  tbody tr:hover { background:#162233; box-shadow:inset 0 0 0 1px var(--rarity-bg); }
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
    border:1px solid color-mix(in srgb, var(--rarity) 30%, rgba(169,180,196,.22));
    box-shadow:inset 0 -3px 0 var(--rarity-soft);
    transition:transform .14s ease, border-color .14s ease, box-shadow .14s ease;
  }
  tbody tr:hover .thumb { transform:translateY(-1px); border-color:color-mix(in srgb, var(--rarity) 55%, white 6%); box-shadow:0 10px 24px rgba(0,0,0,.20), inset 0 -3px 0 var(--rarity); }
  .name { display:inline; font-size:16px; line-height:1.25; font-weight:900; color:#fff; }
  .name:hover { text-decoration:underline; text-decoration-thickness:1px; }
  .meta { margin-top:7px; color:var(--muted); font-size:12px; line-height:1.4; }
  .chips { display:flex; flex-wrap:wrap; gap:6px; margin-top:9px; }
  .chip { display:inline-flex; align-items:center; max-width:100%; padding:4px 7px; border:1px solid var(--line); border-radius:999px; background:rgba(13,20,32,.8); color:#d8e1ee; font-size:12px; line-height:1.2; }
  .rarity-accent .chip:first-child { border-color:color-mix(in srgb, var(--rarity) 45%, var(--line)); background:var(--rarity-soft); }
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
  .owned-price-panel {
    display:grid;
    gap:6px;
    margin-top:11px;
    padding:8px;
    border:1px solid rgba(94,230,168,.22);
    border-radius:10px;
    background:
      linear-gradient(180deg, rgba(94,230,168,.08), rgba(59,130,246,.035)),
      rgba(255,255,255,.026);
  }
  .owned-price-head {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:8px;
    color:#dff7ea;
    font-size:10px;
    font-weight:950;
    text-transform:uppercase;
    letter-spacing:.04em;
  }
  .owned-price-head b {
    display:inline-flex;
    align-items:center;
    justify-content:center;
    min-width:22px;
    height:20px;
    padding:0 7px;
    border-radius:999px;
    background:rgba(94,230,168,.14);
    border:1px solid rgba(94,230,168,.25);
    color:#bbf7d0;
    font-size:11px;
    font-variant-numeric:tabular-nums;
  }
  .owned-price-list {
    display:grid;
    gap:5px;
    max-height:132px;
    overflow:auto;
    padding-right:2px;
  }
  .owned-price-item {
    display:grid;
    grid-template-columns:minmax(0,1fr) auto;
    align-items:center;
    gap:6px;
    width:100%;
    padding:7px 8px;
    border:1px solid rgba(169,180,196,.14);
    border-radius:8px;
    background:rgba(8,13,20,.58);
    color:#eaf2ff;
    text-align:left;
    cursor:pointer;
    transition:transform .14s ease, border-color .14s ease, background-color .14s ease;
  }
  .owned-price-item:hover,
  .owned-price-item:focus-visible {
    transform:translateY(-1px);
    border-color:rgba(94,230,168,.42);
    background:rgba(20,35,48,.86);
    outline:none;
  }
  .owned-price-account {
    min-width:0;
    color:#dbeafe;
    font-size:11px;
    font-weight:900;
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
  }
  .owned-price-meta {
    grid-column:1 / -1;
    color:#9fb0c5;
    font-size:10px;
    font-weight:750;
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
  }
  .owned-price-value {
    color:#fff;
    font-size:12px;
    font-weight:950;
    font-variant-numeric:tabular-nums;
    white-space:nowrap;
  }
  .inventory-jump-highlight {
    animation:inventoryJumpPulse 1.8s ease-in-out 1;
    box-shadow:0 0 0 2px rgba(94,230,168,.62), 0 0 32px rgba(94,230,168,.20) !important;
  }
  @keyframes inventoryJumpPulse {
    0%, 100% { box-shadow:0 0 0 1px rgba(94,230,168,.24), 0 0 0 rgba(94,230,168,0); }
    35% { box-shadow:0 0 0 3px rgba(94,230,168,.72), 0 0 36px rgba(94,230,168,.26); }
  }
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
  .second-market-charts {
    display:grid;
    gap:8px;
    margin-top:10px;
  }
  .second-market-chart {
    border:1px solid rgba(255,255,255,.09);
    border-radius:9px;
    background:rgba(255,255,255,.025);
    padding:7px 8px 5px;
  }
  .second-market-chart header {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:8px;
    margin-bottom:2px;
    color:#d8e0eb;
    font-size:10px;
    font-weight:900;
    letter-spacing:.04em;
    text-transform:uppercase;
  }
  .second-market-chart header small {
    color:#9ca9ba;
    font-size:10px;
    font-weight:850;
    letter-spacing:0;
    text-transform:none;
    font-variant-numeric:tabular-nums;
  }
  .second-market-chart.cf {
    border-color:rgba(80,174,255,.20);
    background:rgba(41,123,255,.05);
  }
  .second-market-chart.uu {
    border-color:rgba(255,214,76,.22);
    background:rgba(255,214,76,.05);
  }
  .second-market-spark {
    width:100%;
    height:54px;
    display:block;
  }
  .second-market-spark .line {
    fill:none;
    stroke-width:2.8;
    stroke-linecap:round;
    stroke-linejoin:round;
  }
  .second-market-spark .area { opacity:.12; }
  .second-market-empty {
    color:#768398;
    font-size:11px;
    padding:8px 0 6px;
  }
  .second-market-signal {
    display:inline-flex;
    align-items:center;
    gap:6px;
    margin-top:8px;
    padding:5px 8px;
    border-radius:999px;
    border:1px solid rgba(255,255,255,.10);
    background:rgba(255,255,255,.04);
    color:#cbd5e1;
    font-size:11px;
    font-weight:900;
  }
  .second-market-signal.wait {
    border-color:rgba(251,113,133,.35);
    background:rgba(251,113,133,.10);
    color:#fecdd3;
  }
  .second-market-signal.edge {
    border-color:rgba(94,229,146,.34);
    background:rgba(94,229,146,.10);
    color:#bbf7d0;
  }
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
    position:fixed;
    top:calc(env(safe-area-inset-top, 0px) + 12px);
    right:calc(env(safe-area-inset-right, 0px) + 12px);
    z-index:2003;
    display:inline-flex;
    align-items:center;
    justify-content:center;
    gap:7px;
    min-width:88px;
    height:42px;
    min-height:42px;
    margin:0;
    padding:0 13px;
    border:1px solid rgba(169,180,196,.30);
    border-radius:999px;
    background:rgba(16,26,41,.96);
    color:#edf4ff;
    box-shadow:0 14px 34px rgba(0,0,0,.45);
    font-size:18px;
    font-weight:950;
    line-height:1;
    backdrop-filter:blur(10px);
  }
  .modal-close::before { content:"Close"; font-size:12px; line-height:1; }
  .modal-content { padding:18px; }
  .modal-grid { display:grid; grid-template-columns:minmax(270px,.9fr) minmax(0,1.1fr); gap:18px; clear:both; }
  .modal-visual {
    --rarity:#5b8cff;
    --rarity2:#9cc3ff;
    --rarity-bg:rgba(91,140,255,.14);
    --rarity-soft:rgba(91,140,255,.08);
    position:relative;
    display:grid;
    gap:12px;
    align-content:start;
    padding:14px;
    border:1px solid color-mix(in srgb, var(--rarity) 32%, rgba(169,180,196,.14));
    border-radius:11px;
    background:linear-gradient(180deg, var(--rarity-soft), rgba(10,16,25,.98) 55%);
    box-shadow:inset 0 -4px 0 var(--rarity-bg);
  }
  .modal-visual img { width:100%; max-height:430px; object-fit:contain; filter:drop-shadow(0 18px 22px rgba(0,0,0,.38)); }
  .modal-rank { position:absolute; top:12px; left:12px; padding:5px 9px; border-radius:999px; background:rgba(8,13,20,.76); border:1px solid color-mix(in srgb, var(--rarity) 45%, rgba(169,180,196,.22)); font-weight:950; }
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

  /* Visual design pass: graphite base, decisive rarity accents, sharper hierarchy. */
  .topbar {
    position:relative;
    overflow:hidden;
    border-radius:8px;
    background:
      linear-gradient(135deg, rgba(25,25,29,.98), rgba(9,9,11,.98) 58%, rgba(17,15,22,.98)),
      linear-gradient(90deg, rgba(0,213,255,.10), rgba(168,85,247,.10), rgba(255,43,214,.08), rgba(255,180,0,.08));
    border-color:rgba(255,255,255,.12);
  }
  .topbar::before {
    content:"";
    position:absolute;
    inset:0 0 auto;
    height:3px;
    background:linear-gradient(90deg, #2f7dff, #35e4ff 25%, #a855f7 50%, #ff2bd6 74%, #ffb000);
  }
  h1 { color:#fff; font-weight:950; }
  .sub { color:#b8bec9; }
  .stat {
    border-color:rgba(255,255,255,.12);
    background:linear-gradient(180deg, rgba(255,255,255,.055), rgba(255,255,255,.018));
    box-shadow:inset 0 1px 0 rgba(255,255,255,.05);
  }
  .stat span { color:#989fab; }
  .stat b { color:#fff; }
  .filters {
    background:rgba(7,7,9,.92);
    border-color:rgba(255,255,255,.12);
    box-shadow:0 16px 38px rgba(0,0,0,.36), inset 0 1px 0 rgba(255,255,255,.035);
  }
  .field label {
    color:#9da4af;
    font-weight:800;
    letter-spacing:.02em;
    text-transform:uppercase;
  }
  input, select {
    background:linear-gradient(180deg, #15161b, #101115);
    border-color:rgba(255,255,255,.12);
  }
  button, .mini-btn, .action {
    background:linear-gradient(180deg, #1a1b20, #121318);
    border-color:rgba(255,255,255,.13);
    box-shadow:inset 0 1px 0 rgba(255,255,255,.04);
  }
  .panel {
    border-radius:8px;
    background:
      linear-gradient(180deg, rgba(255,255,255,.045), transparent 110px),
      linear-gradient(180deg, #121317, #0b0c10);
    border-color:rgba(255,255,255,.11);
  }
  .panel-head {
    background:linear-gradient(180deg, rgba(255,255,255,.045), rgba(255,255,255,.012));
    border-bottom-color:rgba(255,255,255,.075);
  }
  .panel-title {
    color:#fff;
    font-weight:950;
  }
  .hint { color:#a8adb7; }
  .view-toggle,
  .grid-controls,
  .inventory-grid-controls {
    background:#0d0e11;
    border-color:rgba(255,255,255,.12);
  }
  .view-btn.active {
    background:linear-gradient(135deg, rgba(47,125,255,.28), rgba(0,213,255,.16));
    box-shadow:inset 0 0 0 1px rgba(0,213,255,.38);
    color:#fff;
  }
  .grid-card,
  .inventory-card {
    border-radius:8px;
    border-color:color-mix(in srgb, var(--rarity) 58%, rgba(255,255,255,.12));
    background:
      linear-gradient(180deg, color-mix(in srgb, var(--rarity) 18%, transparent), transparent 46%),
      linear-gradient(180deg, #191a1f, #0b0c10 78%);
    box-shadow:
      0 16px 38px rgba(0,0,0,.38),
      0 0 0 1px rgba(255,255,255,.035),
      inset 0 1px 0 rgba(255,255,255,.06);
  }
  .grid-card::after,
  .inventory-card::before {
    height:5px;
    background:linear-gradient(90deg, var(--rarity), var(--rarity2));
    box-shadow:0 0 18px var(--rarity-bg);
  }
  .grid-card:hover,
  .grid-card:focus-visible,
  .inventory-card:hover {
    border-color:var(--rarity);
    box-shadow:
      0 22px 52px rgba(0,0,0,.48),
      0 0 0 1px var(--rarity),
      0 0 30px var(--rarity-bg),
      inset 0 1px 0 rgba(255,255,255,.08);
  }
  .grid-image,
  .inventory-card-art {
    border-radius:7px;
    background:
      linear-gradient(180deg, rgba(255,255,255,.075), rgba(255,255,255,.01)),
      linear-gradient(180deg, var(--rarity-bg), rgba(255,255,255,.02));
    box-shadow:
      inset 0 0 0 1px rgba(255,255,255,.055),
      inset 0 -3px 0 var(--rarity);
  }
  .grid-tier,
  .grid-variant {
    color:#050506;
    background:linear-gradient(135deg, var(--rarity), var(--rarity2));
    border-color:rgba(255,255,255,.22);
    box-shadow:0 0 16px var(--rarity-bg);
  }
  .grid-rank,
  .inventory-card-select,
  .modal-rank {
    background:rgba(5,5,6,.74);
    border-color:rgba(255,255,255,.16);
  }
  .inventory-current-price,
  .grid-price,
  .price-main {
    color:#fff;
    text-shadow:0 1px 14px rgba(255,255,255,.12);
  }
  .inventory-pnl-pill,
  .grid-kpi,
  .inventory-op,
  .inventory-selection-bar,
  .inventory-stat,
  .modal-section {
    background:linear-gradient(180deg, rgba(255,255,255,.052), rgba(255,255,255,.018));
    border-color:rgba(255,255,255,.105);
  }
  .chip {
    background:#101115;
    border-color:rgba(255,255,255,.12);
  }
  .rarity-accent .chip:first-child {
    color:#050506;
    font-weight:900;
    border-color:rgba(255,255,255,.20);
    background:linear-gradient(135deg, var(--rarity), var(--rarity2));
  }
  thead th {
    background:#111216;
    color:#d9dee7;
    border-bottom-color:rgba(255,255,255,.10);
  }
  tbody tr {
    background:#0f1014;
  }
  tbody tr:nth-child(even) { background:#0b0c10; }
  tbody tr:hover {
    background:color-mix(in srgb, var(--rarity) 11%, #101115);
    box-shadow:inset 0 0 0 1px color-mix(in srgb, var(--rarity) 50%, rgba(255,255,255,.06));
  }
  tbody tr td:first-child {
    border-left:4px solid var(--rarity);
  }
  .thumb {
    background:
      linear-gradient(180deg, rgba(255,255,255,.065), rgba(255,255,255,.012)),
      linear-gradient(180deg, var(--rarity-bg), rgba(5,5,6,.8));
    border-color:color-mix(in srgb, var(--rarity) 58%, rgba(255,255,255,.12));
  }
  .verdict {
    box-shadow:0 0 20px rgba(255,255,255,.08);
  }
  .modal-dialog {
    border-radius:8px;
    background:linear-gradient(180deg, #15161a, #090a0d);
    border-color:rgba(255,255,255,.14);
  }
  .modal-visual {
    border-radius:8px;
    border-color:color-mix(in srgb, var(--rarity) 60%, rgba(255,255,255,.12));
    background:
      linear-gradient(180deg, color-mix(in srgb, var(--rarity) 15%, transparent), transparent 50%),
      linear-gradient(180deg, #18191e, #0b0c10);
  }
  .pos { color:#00e676 !important; }
  .neg { color:#ff3b5f !important; }
  .flat { color:#c3cad5 !important; }
  .signal-dot.up,
  .signal-dot.low { background:#00e676; }
  .signal-dot.down { background:#ff3b5f; }
  .signal-dot.watch { background:#ffd400; }
  .release-low-badge,
  .low-gap-badge.near {
    border-color:rgba(0,230,118,.45);
    background:rgba(0,230,118,.12);
    color:#9dffc7;
  }
  .low-gap-badge.mid {
    border-color:rgba(255,212,0,.48);
    background:rgba(255,212,0,.12);
    color:#ffe56b;
  }

  /* Product polish pass: softer hierarchy, lighter accents, large-list containment. */
  :root {
    --bg:#0b0d12;
    --panel:#151820;
    --panel-2:#1b202a;
    --panel-3:#242a35;
    --line:rgba(229,236,247,.13);
    --line-soft:rgba(229,236,247,.075);
    --text:#f5f7fb;
    --muted:#b3bbc8;
    --faint:#7f8795;
    --blue:#6aa8ff;
    --green:#5ee6a8;
    --yellow:#ffd86b;
    --red:#ff6b83;
    --shadow:0 22px 58px rgba(0,0,0,.34);
  }
  body {
    background:
      radial-gradient(circle at 15% -8%, rgba(106,168,255,.14), transparent 35%),
      radial-gradient(circle at 88% 0%, rgba(255,122,217,.095), transparent 32%),
      linear-gradient(180deg, #171a22 0, #10131a 270px, #0b0d12 100%);
  }
  .app { width:min(1860px, calc(100vw - 32px)); }
  .topbar,
  .panel,
  .filter-panel {
    border-radius:18px;
  }
  .topbar {
    background:
      linear-gradient(135deg, rgba(32,36,47,.92), rgba(17,20,28,.96) 62%, rgba(24,24,32,.94)),
      linear-gradient(90deg, rgba(106,168,255,.12), rgba(94,230,168,.06), rgba(255,122,217,.08));
    border-color:rgba(229,236,247,.14);
    box-shadow:0 28px 70px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.06);
  }
  .topbar::before {
    height:2px;
    opacity:.72;
    background:linear-gradient(90deg, rgba(106,168,255,.85), rgba(120,240,255,.75), rgba(167,139,250,.70), rgba(255,122,217,.70), rgba(251,191,36,.72));
  }
  .stats { gap:10px; }
  .topbar-side {
    display:grid;
    grid-template-columns:auto 1fr;
    gap:12px;
    align-items:stretch;
    min-width:760px;
  }
  .inventory-banner-btn {
    display:flex;
    align-items:center;
    gap:10px;
    min-width:150px;
    min-height:62px;
    padding:10px 12px;
    text-align:left;
    border-radius:15px;
    background:
      linear-gradient(180deg, rgba(94,230,168,.13), rgba(255,255,255,.035)),
      rgba(255,255,255,.035);
    border-color:rgba(94,230,168,.22);
  }
  .inventory-banner-btn b,
  .inventory-banner-btn small {
    display:block;
    line-height:1.15;
  }
  .inventory-banner-btn b { color:#fff; font-size:13px; font-weight:950; }
  .inventory-banner-btn small { margin-top:4px; color:#aeb7c6; font-size:11px; font-weight:800; }
  .inventory-banner-icon {
    position:relative;
    width:28px;
    height:28px;
    flex:0 0 auto;
    border:1px solid rgba(94,230,168,.38);
    border-radius:9px;
    background:linear-gradient(135deg, rgba(94,230,168,.22), rgba(106,168,255,.12));
  }
  .inventory-banner-icon::before,
  .inventory-banner-icon::after {
    content:"";
    position:absolute;
    inset:7px;
    border:2px solid #dfffee;
    border-top:0;
    border-radius:2px 2px 5px 5px;
  }
  .inventory-banner-icon::after {
    inset:4px 8px auto;
    height:7px;
    border:2px solid #dfffee;
    border-bottom:0;
    border-radius:999px 999px 0 0;
    background:transparent;
  }
  .stat,
  .filters,
  .panel,
  .modal-dialog {
    backdrop-filter:blur(16px);
  }
  .stat {
    border-radius:14px;
    background:linear-gradient(180deg, rgba(255,255,255,.075), rgba(255,255,255,.025));
  }
  .filters {
    top:8px;
    padding:12px;
    border-radius:16px;
    background:rgba(16,19,27,.84);
    box-shadow:0 18px 48px rgba(0,0,0,.26), inset 0 1px 0 rgba(255,255,255,.055);
  }
  input,
  select,
  button,
  .mini-btn,
  .action {
    border-radius:11px;
  }
  input,
  select {
    background:linear-gradient(180deg, rgba(255,255,255,.075), rgba(255,255,255,.032));
    border-color:rgba(229,236,247,.13);
  }
  select option,
  select optgroup {
    color:#f5f7fb;
    background:#171b24;
  }
  button,
  .mini-btn,
  .action {
    background:linear-gradient(180deg, rgba(255,255,255,.095), rgba(255,255,255,.04));
    border-color:rgba(229,236,247,.14);
    transition:transform .16s ease, border-color .16s ease, background-color .16s ease, box-shadow .16s ease;
  }
  button:hover,
  a.action:hover {
    transform:translateY(-1px);
    background:linear-gradient(180deg, rgba(255,255,255,.13), rgba(255,255,255,.055));
    border-color:rgba(120,196,255,.36);
    box-shadow:0 10px 24px rgba(0,0,0,.18);
  }
  .panel {
    background:
      linear-gradient(180deg, rgba(255,255,255,.058), rgba(255,255,255,.012) 150px),
      linear-gradient(180deg, rgba(23,27,36,.96), rgba(12,14,20,.98));
    border-color:rgba(229,236,247,.12);
  }
  .panel-head {
    padding:16px 18px;
    background:linear-gradient(180deg, rgba(255,255,255,.052), rgba(255,255,255,.018));
  }
  .panel-title { font-size:17px; letter-spacing:0; }
  .view-toggle,
  .grid-controls,
  .inventory-grid-controls {
    border-radius:13px;
    background:rgba(255,255,255,.045);
  }
  .view-btn { border-radius:10px; }
  .view-btn.active {
    background:linear-gradient(135deg, rgba(106,168,255,.24), rgba(120,240,255,.10));
    box-shadow:inset 0 0 0 1px rgba(122,184,255,.24);
  }
  .grid-card,
  .inventory-card,
  .focus-card,
  tbody tr {
    content-visibility:auto;
  }
  .grid-card { contain-intrinsic-size:258px; }
  .inventory-card { contain-intrinsic-size:352px; }
  .focus-card { contain-intrinsic-size:96px; }
  tbody tr { contain-intrinsic-size:210px; }
  .grid-card,
  .inventory-card {
    border-radius:16px;
    border-color:color-mix(in srgb, var(--rarity) 28%, rgba(229,236,247,.12));
    background:
      linear-gradient(180deg, color-mix(in srgb, var(--rarity) 8%, transparent), transparent 48%),
      linear-gradient(180deg, rgba(32,36,47,.94), rgba(14,16,22,.98) 82%);
    box-shadow:0 18px 42px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.055);
  }
  .grid-card::after,
  .inventory-card::before {
    height:3px;
    opacity:.78;
    box-shadow:none;
  }
  .grid-card:hover,
  .grid-card:focus-visible,
  .inventory-card:hover {
    transform:translateY(-3px);
    border-color:color-mix(in srgb, var(--rarity) 52%, rgba(255,255,255,.12));
    box-shadow:0 24px 56px rgba(0,0,0,.34), 0 0 0 1px rgba(255,255,255,.035), 0 0 18px var(--rarity-bg);
  }
  .grid-image,
  .inventory-card-art,
  .thumb,
  .modal-visual {
    background:
      radial-gradient(circle at 50% 26%, color-mix(in srgb, var(--rarity) 18%, transparent), transparent 55%),
      linear-gradient(180deg, rgba(255,255,255,.07), rgba(255,255,255,.018));
    box-shadow:inset 0 0 0 1px rgba(255,255,255,.052);
  }
  .grid-tier,
  .grid-variant,
  .rarity-accent .chip:first-child {
    color:#10131a;
    background:linear-gradient(135deg, color-mix(in srgb, var(--rarity) 88%, white 12%), color-mix(in srgb, var(--rarity2) 82%, white 18%));
    box-shadow:none;
  }
  .grid-kpis { grid-template-columns:repeat(2, minmax(0,1fr)); }
  .grid-kpi {
    border-radius:10px;
    background:rgba(255,255,255,.047);
    border-color:rgba(229,236,247,.095);
  }
  .grid-market-counter {
    display:flex;
    min-width:0;
  }
  .market-count {
    display:inline-flex;
    align-items:center;
    gap:6px;
    max-width:100%;
    min-height:24px;
    padding:4px 8px;
    border:1px solid rgba(229,236,247,.115);
    border-radius:999px;
    background:linear-gradient(180deg, rgba(255,255,255,.075), rgba(255,255,255,.025));
    color:#e8eef8;
    font-size:11px;
    font-weight:850;
    line-height:1.1;
    white-space:nowrap;
    overflow:hidden;
    text-overflow:ellipsis;
    font-variant-numeric:tabular-nums;
  }
  .market-count b {
    color:inherit;
    font:inherit;
    overflow:hidden;
    text-overflow:ellipsis;
  }
  .market-count small {
    color:var(--muted);
    font-size:10px;
    font-weight:750;
  }
  .market-count.mini {
    padding:4px 7px;
    font-size:10.5px;
  }
  .market-count.muted { color:#929baa; }
  .market-count-dot {
    width:7px;
    height:7px;
    flex:0 0 auto;
    border-radius:999px;
    background:linear-gradient(135deg, var(--green), var(--blue));
    box-shadow:0 0 0 3px rgba(94,230,168,.10);
  }
  .price-compare {
    display:grid;
    grid-template-columns:repeat(2, minmax(0,1fr));
    gap:7px;
    margin-top:11px;
  }
  .market-price-card {
    display:block;
    min-width:0;
    padding:8px;
    border:1px solid rgba(229,236,247,.11);
    border-radius:12px;
    background:linear-gradient(180deg, rgba(255,255,255,.072), rgba(255,255,255,.028));
    font-variant-numeric:tabular-nums;
  }
  .market-price-card span {
    display:block;
    margin-bottom:4px;
    color:#aeb7c6;
    font-size:10px;
    font-weight:900;
    letter-spacing:.04em;
    text-transform:uppercase;
  }
  .market-price-card b {
    display:block;
    color:#fff;
    font-size:14px;
    line-height:1.1;
    font-weight:950;
  }
  .market-price-card small {
    display:block;
    margin-top:4px;
    color:#aeb7c6;
    font-size:10px;
    line-height:1.25;
  }
  .market-price-card em {
    display:block;
    margin-top:2px;
    font-style:normal;
    font-weight:900;
  }
  .market-price-card.skins {
    border-color:rgba(80,143,255,.22);
    background:
      linear-gradient(180deg, rgba(59,130,246,.105), rgba(255,255,255,.026)),
      rgba(255,255,255,.028);
  }
  .market-price-card.skins.unavailable b { color:#c5cedb; }
  .store-offers {
    grid-column:1 / -1;
    display:grid;
    grid-template-columns:repeat(2, minmax(0,1fr));
    gap:6px;
  }
  .store-chip {
    display:flex;
    align-items:center;
    justify-content:space-between;
    min-width:0;
    min-height:28px;
    padding:5px 7px;
    border:1px solid rgba(229,236,247,.12);
    border-radius:9px;
    background:rgba(255,255,255,.045);
    color:#edf4ff;
    font-size:11px;
    font-weight:950;
    font-variant-numeric:tabular-nums;
  }
  .store-chip.unavailable {
    color:#8792a3;
    background:rgba(255,255,255,.025);
  }
  .store-icon {
    display:inline-flex;
    align-items:center;
    justify-content:center;
    width:21px;
    height:21px;
    border-radius:7px;
    color:#071018;
    font-size:9px;
    font-weight:1000;
    letter-spacing:0;
  }
  .store-chip.csfloat .store-icon { background:linear-gradient(135deg,#44e2ff,#3b82f6); }
  .store-chip.uuskins .store-icon { background:linear-gradient(135deg,#f8d24a,#ff8a00); }
  .store-chip b { margin-left:6px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .second-market-price-grid {
    grid-column:1 / -1;
    display:grid;
    gap:6px;
    margin-top:2px;
  }
  .second-market-price-row {
    display:grid;
    grid-template-columns:54px repeat(3, minmax(0,1fr));
    gap:6px;
    align-items:stretch;
    padding:7px;
    border:1px solid rgba(229,236,247,.10);
    border-radius:10px;
    background:rgba(255,255,255,.032);
    font-variant-numeric:tabular-nums;
    color:inherit;
    text-decoration:none;
    transition:border-color .16s ease, background-color .16s ease, transform .16s ease;
  }
  .second-market-price-row:hover {
    transform:translateY(-1px);
    border-color:rgba(229,236,247,.20);
    background:rgba(255,255,255,.055);
  }
  .second-market-price-row.cf {
    border-color:rgba(80,174,255,.20);
    background:rgba(41,123,255,.045);
  }
  .second-market-price-row.uu {
    border-color:rgba(255,214,76,.22);
    background:rgba(255,214,76,.045);
  }
  .second-market-price-source {
    display:flex;
    align-items:center;
    gap:5px;
    color:#f4f8ff;
    font-size:11px;
    font-weight:950;
  }
  .second-market-price-row .store-icon {
    width:20px;
    height:20px;
  }
  .second-market-price-cell {
    min-width:0;
    padding:4px 5px;
    border-radius:8px;
    background:rgba(0,0,0,.14);
  }
  .second-market-price-cell span {
    display:block;
    color:#9da9ba;
    font-size:9px;
    font-weight:950;
    letter-spacing:.04em;
    text-transform:uppercase;
  }
  .second-market-price-cell b {
    display:block;
    margin-top:2px;
    color:#f8fbff;
    font-size:12px;
    line-height:1.05;
    font-weight:950;
    white-space:nowrap;
    overflow:hidden;
    text-overflow:ellipsis;
  }
  .second-market-price-cell.prev b { color:#d7e0ee; }
  .second-market-price-cell.low b { color:#78f3a6; }
  .skins-action {
    color:#d8e8ff;
    border-color:rgba(80,143,255,.36);
    background:linear-gradient(180deg, rgba(80,143,255,.15), rgba(255,255,255,.035));
  }
  .steam-action {
    color:#d8ffe9;
    border-color:rgba(52,211,153,.34);
    background:linear-gradient(180deg, rgba(52,211,153,.13), rgba(255,255,255,.032));
  }
  .inventory-inline-note {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:12px;
    padding:12px;
    border:1px solid rgba(229,236,247,.10);
    border-radius:14px;
    background:rgba(255,255,255,.035);
  }
  .inventory-drawer {
    place-items:start end;
    padding:18px;
  }
  .inventory-drawer-dialog {
    position:relative;
    display:grid;
    grid-template-rows:auto auto auto minmax(0,1fr);
    gap:12px;
    width:min(1380px, calc(100vw - 36px));
    height:min(900px, calc(100vh - 36px));
    padding:16px;
    border:1px solid rgba(229,236,247,.14);
    border-radius:20px;
    background:
      linear-gradient(180deg, rgba(255,255,255,.07), rgba(255,255,255,.025)),
      linear-gradient(180deg, #171b24, #0d1017);
    box-shadow:0 28px 80px rgba(0,0,0,.46);
    overflow:hidden;
  }
  .inventory-drawer-head {
    display:flex;
    align-items:flex-start;
    justify-content:space-between;
    gap:14px;
    padding:2px 52px 2px 2px;
  }
  .inventory-drawer-head h2 {
    margin:0;
    color:#fff;
    font-size:22px;
    line-height:1.1;
    font-weight:950;
  }
  .inventory-drawer-head p {
    margin:5px 0 0;
    color:var(--muted);
    font-size:12px;
  }
  .inventory-drawer-close {
    top:14px;
    right:14px;
  }
  .inventory-drawer .inventory-grid-view,
  .inventory-drawer .inventory-list-wrap {
    min-height:0;
    overflow:auto;
  }
  .inventory-drawer .inventory-grid-view {
    padding:2px 2px 10px;
  }
  .market-activity-row b { min-width:0; }
  .modal-market-count { margin:-2px 0 10px; }
  tbody tr {
    background:rgba(20,24,32,.88);
  }
  tbody tr:nth-child(even) { background:rgba(16,19,26,.90); }
  tbody tr:hover {
    background:color-mix(in srgb, var(--rarity) 5%, rgba(28,32,42,.94));
    box-shadow:inset 0 0 0 1px color-mix(in srgb, var(--rarity) 24%, rgba(255,255,255,.05));
  }
  tbody tr td:first-child {
    border-left:3px solid color-mix(in srgb, var(--rarity) 78%, white 4%);
  }
  .portfolio-focus-grid {
    grid-template-columns:repeat(2, minmax(0,1fr));
    gap:14px;
    padding:16px;
  }
  .recommendation-group {
    min-width:0;
    border:1px solid color-mix(in srgb, var(--rarity) 26%, rgba(229,236,247,.115));
    border-radius:16px;
    background:
      linear-gradient(180deg, color-mix(in srgb, var(--rarity) 7%, transparent), transparent 50%),
      rgba(255,255,255,.032);
    overflow:hidden;
  }
  .recommendation-head {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:10px;
    padding:11px 12px;
    border-bottom:1px solid rgba(229,236,247,.075);
  }
  .recommendation-head span {
    color:#fff;
    font-size:13px;
    font-weight:950;
  }
  .recommendation-head b {
    color:color-mix(in srgb, var(--rarity) 72%, white 28%);
    font-size:11px;
    font-weight:900;
    text-transform:uppercase;
  }
  .recommendation-cards {
    display:grid;
    grid-template-columns:1fr;
    gap:8px;
    padding:10px;
  }
  .focus-card {
    width:100%;
    min-height:86px;
    grid-template-columns:66px minmax(0,1fr);
    padding:9px;
    border-radius:14px;
    background:
      linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.022)),
      linear-gradient(135deg, color-mix(in srgb, var(--rarity) 6%, transparent), transparent 68%);
    color:var(--text);
    text-align:left;
    cursor:pointer;
    box-shadow:none;
    transition:transform .16s ease, border-color .16s ease, background-color .16s ease, box-shadow .16s ease;
  }
  .focus-card:hover,
  .focus-card:focus-visible {
    transform:translateY(-1px);
    border-color:color-mix(in srgb, var(--rarity) 46%, rgba(255,255,255,.14));
    box-shadow:0 12px 28px rgba(0,0,0,.20);
    outline:none;
  }
  .focus-card img {
    width:66px;
    height:66px;
    border-radius:12px;
    background:rgba(255,255,255,.035);
  }
  .focus-card-body {
    display:block;
    min-width:0;
  }
  .focus-note {
    display:block;
    color:#aeb7c6;
  }
  .focus-meta {
    align-items:center;
  }
  .focus-chip {
    border-radius:999px;
    background:rgba(255,255,255,.05);
    border-color:rgba(229,236,247,.11);
  }
  .focus-empty.small {
    padding:14px;
    font-size:12px;
  }
  .verdict {
    box-shadow:none;
  }
  /* Consistency pass: solid canvas, calmer surfaces, semantic accents only. */
  body {
    background:#0d1118;
  }
  .topbar,
  .panel,
  .filter-panel,
  .inventory-drawer-dialog,
  .modal-dialog {
    background:#151a23;
    border-color:#293142;
    box-shadow:0 18px 48px rgba(0,0,0,.30);
  }
  .topbar::before { display:none; }
  .filters,
  .panel-head,
  .stat,
  .inventory-inline-note,
  .recommendation-group,
  .focus-card,
  .market-price-card,
  .grid-card,
  .inventory-card,
  tbody tr,
  tbody tr:nth-child(even) {
    background:#171d27;
  }
  .filters {
    grid-template-columns:repeat(auto-fit, minmax(150px, 1fr));
  }
  .filters .field:first-child {
    grid-column:span 2;
  }
  tbody tr:hover {
    background:#1b2330;
    box-shadow:inset 0 0 0 1px color-mix(in srgb, var(--rarity) 24%, #364155);
  }
  input,
  select,
  button,
  .mini-btn,
  .action,
  .view-toggle,
  .grid-controls,
  .inventory-grid-controls {
    background:#202837;
    border-color:#384355;
    box-shadow:none;
  }
  input:focus,
  select:focus,
  .multi-select-button:focus-visible,
  button:focus-visible {
    outline:2px solid rgba(94,156,255,.34);
    outline-offset:2px;
  }
  button:hover,
  a.action:hover,
  .mini-btn:hover {
    transform:translateY(-1px);
    background:#263044;
    border-color:#4b5d78;
    box-shadow:0 10px 24px rgba(0,0,0,.18);
  }
  select option,
  select optgroup {
    background:#202837;
    color:#f5f7fb;
  }
  .multi-select {
    position:relative;
    min-width:0;
  }
  .multi-select-button {
    width:100%;
    min-height:36px;
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:10px;
    padding:8px 10px;
    color:var(--text);
    font-weight:800;
    text-align:left;
  }
  .multi-select-button::after {
    content:"";
    width:8px;
    height:8px;
    border-right:2px solid #aeb7c6;
    border-bottom:2px solid #aeb7c6;
    transform:rotate(45deg) translateY(-2px);
    transition:transform .16s ease;
  }
  .multi-select[data-open="true"] .multi-select-button::after {
    transform:rotate(225deg) translateY(-2px);
  }
  .multi-select-menu {
    position:absolute;
    top:calc(100% + 6px);
    left:0;
    z-index:80;
    width:min(260px, 92vw);
    display:none;
    gap:4px;
    padding:8px;
    border:1px solid #384355;
    border-radius:12px;
    background:#202837;
    box-shadow:0 18px 42px rgba(0,0,0,.34);
  }
  .multi-select[data-open="true"] .multi-select-menu {
    display:grid;
  }
  .multi-select-menu label {
    display:flex;
    align-items:center;
    gap:9px;
    min-height:34px;
    padding:7px 8px;
    margin:0;
    color:#e6edf8;
    border-radius:9px;
    cursor:pointer;
    font-size:13px;
    font-weight:800;
  }
  .multi-select-menu label:hover {
    background:#283244;
  }
  .multi-select-menu input {
    width:16px;
    height:16px;
    min-height:0;
    accent-color:#6aa8ff;
  }
  .metric-label,
  .metric-row span[title],
  .inventory-stat span[title],
  .market-price-card[title],
  .price-range-row[title] {
    cursor:help;
    text-decoration:underline dotted rgba(174,183,198,.45);
    text-underline-offset:3px;
  }
  .price-opportunity-tag {
    display:inline-flex;
    align-items:center;
    gap:6px;
    width:100%;
    margin-top:8px;
    padding:7px 8px;
    border:1px solid rgba(94,230,168,.32);
    border-radius:11px;
    background:#172820;
    color:#c8ffe5;
    font-size:11px;
    font-weight:900;
    line-height:1.2;
  }
  .price-opportunity-tag::before {
    content:"";
    width:8px;
    height:8px;
    border-radius:999px;
    background:#5ee6a8;
    box-shadow:0 0 0 3px rgba(94,230,168,.12);
  }
  .third-party-row td:first-child {
    border-left-color:#5ee6a8;
  }
  .grid-deal-tag {
    display:inline-flex;
    align-items:center;
    justify-content:center;
    min-height:22px;
    padding:3px 7px;
    border:1px solid rgba(94,230,168,.30);
    border-radius:999px;
    background:#172820;
    color:#c8ffe5;
    font-size:10px;
    font-weight:950;
    white-space:nowrap;
  }
  .grid-prices {
    display:grid;
    grid-template-columns:repeat(5, minmax(0,1fr));
    gap:4px;
    min-width:0;
  }
  .grid-prices span {
    min-width:0;
    padding:4px 5px;
    border:1px solid #303a4d;
    border-radius:8px;
    background:#141b27;
    overflow:hidden;
  }
  .grid-prices small,
  .grid-prices b {
    display:block;
    min-width:0;
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
    font-variant-numeric:tabular-nums;
  }
  .grid-prices small {
    color:#8f9aab;
    font-size:8px;
    font-weight:950;
    text-transform:uppercase;
  }
  .grid-prices b {
    margin-top:1px;
    color:#f5f7fb;
    font-size:10px;
    font-weight:950;
  }
  .grid-view[data-density="dense"] .grid-prices {
    grid-template-columns:repeat(3, minmax(0,1fr));
  }
  .grid-view[data-density="dense"] .grid-prices span {
    padding:3px 4px;
  }
  .grid-view[data-density="dense"] .grid-prices span:nth-child(3),
  .grid-view[data-density="dense"] .grid-prices span:nth-child(5) {
    display:none;
  }
  .grid-view[data-density="ultra"] .grid-prices {
    grid-template-columns:repeat(2, minmax(0,1fr));
  }
  .grid-view[data-density="ultra"] .grid-prices span {
    padding:2px 3px;
    border-radius:6px;
  }
  .grid-view[data-density="ultra"] .grid-prices span:nth-child(n+3),
  .grid-view[data-density="ultra"] .grid-market-counter,
  .grid-view[data-density="ultra"] .grid-deal-tag {
    display:none;
  }
  .favorite-btn {
    display:inline-flex;
    align-items:center;
    justify-content:center;
    gap:6px;
    min-height:30px;
    padding:5px 9px;
    border-radius:999px;
    color:#cbd5e1;
    font-size:12px;
    font-weight:900;
  }
  .favorite-btn.active {
    color:#ffe7a3;
    border-color:rgba(255,216,107,.42);
    background:#2a2516;
  }
  .favorite-btn.compact {
    position:absolute;
    top:8px;
    left:50%;
    z-index:4;
    min-width:30px;
    width:30px;
    height:30px;
    padding:0;
    font-size:16px;
    background:#202837;
  }
  .refresh-prices-btn,
  .fetch-price-btn {
    display:inline-flex;
    align-items:center;
    justify-content:center;
    gap:6px;
    min-height:30px;
    padding:6px 10px;
    border:1px solid rgba(57,217,138,.36);
    border-radius:999px;
    background:linear-gradient(180deg, rgba(57,217,138,.18), rgba(10,18,28,.92));
    color:#d8ffe9;
    font-size:12px;
    font-weight:950;
    box-shadow:0 8px 18px rgba(0,0,0,.18), inset 0 1px 0 rgba(255,255,255,.06);
  }
  .refresh-prices-btn:hover,
  .fetch-price-btn:hover {
    border-color:rgba(52,235,161,.72);
    background:linear-gradient(180deg, rgba(57,217,138,.28), rgba(12,24,34,.96));
  }
  .fetch-price-btn:disabled {
    cursor:not-allowed;
    opacity:.45;
    color:#9aa8b9;
    border-color:rgba(169,180,196,.18);
    background:#151b25;
  }
  .fetch-price-btn.compact {
    width:100%;
    min-height:23px;
    padding:3px 6px;
    border-radius:7px;
    font-size:10px;
    letter-spacing:.01em;
  }
  .fetch-price-btn[data-state="busy"] {
    border-color:rgba(57,217,138,.48);
    color:#dfffee;
  }
  .fetch-price-btn[data-state="ok"] {
    border-color:rgba(94,229,146,.42);
    color:#bbf7d0;
  }
  .fetch-price-btn[data-state="warn"] {
    border-color:rgba(255,214,76,.42);
    color:#fde68a;
  }
  .fetch-price-btn[data-state="error"] {
    border-color:rgba(251,113,133,.46);
    color:#fecdd3;
  }
  .price-fetch-status {
    display:inline-flex;
    align-items:center;
    min-height:24px;
    margin-left:10px;
    padding:4px 9px;
    border:1px solid rgba(169,180,196,.16);
    border-radius:999px;
    background:rgba(255,255,255,.04);
    color:#aebbd0;
    font-weight:850;
  }
  .price-fetch-inline-status {
    min-height:28px;
    margin:6px 0 0;
    width:100%;
    justify-content:center;
    white-space:normal;
    text-align:center;
    line-height:1.25;
  }
  .price-fetching .refresh-prices-btn,
  .price-fetching .fetch-price-btn {
    cursor:progress;
    opacity:.72;
  }
  .refresh-prices-btn[disabled],
  .fetch-price-btn.busy {
    pointer-events:none;
  }
  .price-fetch-toast {
    position:fixed;
    right:18px;
    bottom:18px;
    z-index:95;
    max-width:min(460px, calc(100vw - 28px));
    margin:0;
    box-shadow:0 18px 46px rgba(0,0,0,.38), inset 0 1px 0 rgba(255,255,255,.06);
    pointer-events:none;
    opacity:0;
    transform:translateY(10px);
    transition:opacity .16s ease, transform .16s ease;
  }
  .price-fetch-toast[data-visible="true"] {
    opacity:1;
    transform:translateY(0);
  }
  .footer-note {
    display:flex;
    flex-wrap:wrap;
    align-items:center;
    gap:8px;
  }
  .price-fetch-status[data-tone="ok"] {
    color:#baffdf;
    border-color:rgba(52,235,161,.34);
    background:rgba(18,111,72,.18);
  }
  .price-fetch-status[data-tone="warn"] {
    color:#ffe7a3;
    border-color:rgba(255,201,71,.34);
    background:rgba(134,89,8,.18);
  }
  .price-fetch-status[data-tone="error"] {
    color:#ffc4cf;
    border-color:rgba(255,91,122,.38);
    background:rgba(112,24,44,.20);
  }
  .price-fetch-status[data-tone="busy"]::before {
    content:"";
    width:7px;
    height:7px;
    margin-right:7px;
    border-radius:999px;
    background:#39d98a;
    box-shadow:0 0 0 0 rgba(57,217,138,.45);
    animation:fetchPulse 1.15s ease-out infinite;
  }
  @keyframes fetchPulse {
    0% { box-shadow:0 0 0 0 rgba(57,217,138,.42); }
    100% { box-shadow:0 0 0 9px rgba(57,217,138,0); }
  }
  .signal-tags {
    display:flex;
    flex-direction:column;
    align-items:flex-start;
    gap:4px;
  }
  .signal-tags.row {
    margin-top:8px;
    width:max-content;
    max-width:96px;
  }
  .signal-tags.grid {
    position:absolute;
    top:44px;
    left:8px;
    z-index:3;
    max-width:74px;
  }
  .signal-tag {
    display:inline-flex;
    align-items:center;
    max-width:100%;
    min-height:20px;
    padding:3px 7px;
    border:0;
    border-radius:999px;
    background:#252d3b;
    color:#f4f8ff;
    font-size:9.5px;
    line-height:1.1;
    font-weight:950;
    text-transform:uppercase;
    white-space:nowrap;
    overflow:hidden;
    text-overflow:ellipsis;
    box-shadow:0 10px 20px rgba(0,0,0,.22), inset 0 1px 0 rgba(255,255,255,.18);
  }
  .signal-tag.edge {
    color:#04130d;
    background:linear-gradient(135deg, #40f6a0, #22d3ee);
  }
  .signal-tag.low {
    color:#061424;
    background:linear-gradient(135deg, #8bd3ff, #5b8cff);
  }
  .signal-tag.discount {
    color:#211402;
    background:linear-gradient(135deg, #ffe66d, #ff9f1c);
  }
  .signal-tag.watch {
    color:#14091f;
    background:linear-gradient(135deg, #c4b5fd, #fb7bdc);
  }
  .signal-tag.favorite {
    color:#211402;
    background:linear-gradient(135deg, #fff176, #ffbf3f);
  }
  .inventory-drawer {
    place-items:center;
    z-index:2100;
  }
  #detailModal { z-index:2200; }
  .grid-image,
  .inventory-card-art,
  .thumb,
  .modal-visual {
    background:#1d2532;
  }
  .market-price-card.skins {
    background:#182338;
  }
  .market-price-card.skins.deal {
    border-color:rgba(94,230,168,.34);
    background:#172820;
  }
  .market-price-card.true-edge.deal {
    border-color:rgba(94,230,168,.38);
    background:#172820;
  }
  .market-price-card.skins.expensive {
    border-color:rgba(255,107,131,.28);
    background:#2a1d24;
  }
  .market-price-card.true-edge.expensive {
    border-color:rgba(255,107,131,.28);
    background:#2a1d24;
  }
  @media (prefers-reduced-motion:no-preference) {
    tbody tr { animation:none; }
    .signal-dot.up,
    .signal-dot.low { animation:none; }
    .release-low-badge {
      transition:border-color .18s ease, background-color .18s ease, transform .18s ease;
    }
    .release-low-badge:hover { transform:translateY(-1px); }
  }

  @media (max-width:1200px) {
    .topbar { grid-template-columns:1fr; }
    .topbar-side { min-width:0; grid-template-columns:1fr; }
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
    .topbar-side { display:grid; grid-template-columns:1fr; gap:8px; margin-top:12px; }
    .inventory-banner-btn { min-height:52px; width:100%; }
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
    .portfolio-focus-grid { grid-template-columns:1fr; gap:8px; padding:10px; }
    .focus-card { grid-template-columns:64px minmax(0,1fr); padding:9px; }
    .focus-card img { width:64px; height:64px; }
    .inventory-summary { display:block; }
    .inventory-summary-stats { justify-content:flex-start; margin-top:10px; }
    .inventory-stat { min-width:calc(50% - 4px); }
    .inventory-body { padding:10px; gap:10px; }
    .inventory-form { grid-template-columns:1fr 1fr; gap:8px; }
    .inventory-rate-note { font-size:11px; }
    .inventory-form .field:first-child,
    .inventory-form .field.notes-field,
    .inventory-form-actions { grid-column:1 / -1; }
    .inventory-form-actions { display:grid; grid-template-columns:1fr 1fr; }
    .inventory-toolbar { display:grid; gap:8px; }
    .inventory-toolbar-left { display:grid; grid-template-columns:1fr; }
    .inventory-toolbar-right { justify-content:stretch; }
    .inventory-toolbar-right > *,
    .inventory-grid-controls { width:100%; }
    .inventory-grid-controls select { flex:1; min-width:0; }
    .inventory-grid-controls input { flex:0 0 86px; }
    .inventory-filter-input,
    .inventory-account-filter,
    .inventory-sort-filter { width:100%; }
    .inventory-context-panel { grid-template-columns:1fr 1fr; }
    .inventory-drawer-stats { grid-template-columns:1fr 1fr; }
    .inventory-ops { grid-template-columns:1fr; }
    .inventory-op-grid,
    .inventory-op-grid.three { grid-template-columns:1fr 1fr; }
    .inventory-op textarea { min-height:132px; }
    .inventory-selection-bar { display:grid; gap:8px; }
    .inventory-selection-actions { display:grid; grid-template-columns:1fr 1fr; }
    .inventory-selection-actions .mini-btn { width:100%; }
    .inventory-list-wrap { overflow:visible; border:0; }
    .inventory-table { min-width:0; display:block; }
    .inventory-table thead { display:none; }
    .inventory-table tbody { display:grid; gap:10px; }
    .inventory-table tr { display:block; border:1px solid rgba(169,180,196,.16); border-radius:9px; background:#101722; overflow:hidden; }
    .inventory-table td { display:block; width:100%; padding:10px 12px; border-bottom:1px solid rgba(152,166,184,.12); }
    .inventory-table td:last-child { border-bottom:0; }
    .inventory-select-cell { text-align:left; }
    .inventory-table td::before {
      content:attr(data-label);
      display:block;
      margin-bottom:5px;
      color:var(--muted);
      font-size:10px;
      font-weight:900;
      text-transform:uppercase;
    }
    .inventory-grid-view { grid-template-columns:repeat(2, minmax(0,1fr)); gap:8px; }
    .inventory-drawer { padding:8px; }
    .inventory-drawer-dialog { width:calc(100vw - 16px); height:calc(100vh - 16px); padding:10px; border-radius:14px; }
    .inventory-drawer-head { padding-right:52px; }
    .inventory-drawer-head h2 { font-size:18px; }
    .inventory-drawer-head p { font-size:11px; }
    .inventory-card { min-height:336px; padding:8px; border-radius:10px; }
    .inventory-card-art { min-height:132px; padding:7px 5px 5px; }
    .inventory-card-title { font-size:11px; }
    .inventory-card-meta { font-size:10px; }
    .inventory-card-subrow { font-size:9.5px; }
    .inventory-current-price { font-size:14px; }
    .inventory-card-actions { display:flex; }
    .inventory-card-actions .mini-btn { min-height:24px; padding:3px 5px; font-size:9px; }
    .inventory-card-select { width:26px; height:26px; }
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
    .modal-close {
      top:calc(env(safe-area-inset-top, 0px) + 10px);
      right:calc(env(safe-area-inset-right, 0px) + 10px);
      min-width:92px;
      height:42px;
      font-size:18px;
    }
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
    <div class="topbar-side">
      <button class="inventory-banner-btn" id="inventoryDrawerBtn" type="button" aria-haspopup="dialog" aria-controls="inventoryDrawer">
        <span class="inventory-banner-icon" aria-hidden="true"></span>
        <span><b>Inventory</b><small id="inventoryTopCount">0 items</small></span>
      </button>
      <div class="stats">
        <div class="stat"><span>Shown</span><b id="visibleCount">0</b></div>
        <div class="stat"><span>Total</span><b id="totalCount">0</b></div>
        <div class="stat"><span>Avg Expected</span><b id="avgExpected">0%</b></div>
        <div class="stat"><span>Avg Edge</span><b id="avgEdge">0.00</b></div>
        <div class="stat"><span>Scored</span><b id="scoredCount">0</b></div>
      </div>
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
      <div class="field variant-field">
        <label id="variantFilterLabel" for="variantFilterButton">Variant</label>
        <div class="multi-select" id="variantFilter" data-open="false">
          <button id="variantFilterButton" class="multi-select-button" type="button" aria-haspopup="listbox" aria-expanded="false" aria-labelledby="variantFilterLabel variantFilterButton">Holo, Foil</button>
          <div class="multi-select-menu" id="variantFilterMenu" role="listbox" aria-label="Variant filter">
            <label><input type="checkbox" data-variant-option value="Paper" />Paper</label>
            <label><input type="checkbox" data-variant-option value="Foil" checked />Foil</label>
            <label><input type="checkbox" data-variant-option value="Holo" checked />Holo</label>
            <label><input type="checkbox" data-variant-option value="Gold" />Gold</label>
          </div>
        </div>
      </div>
      <div class="field"><label for="typeFilter">Type</label><select id="typeFilter"><option value="">All types</option></select></div>
      <div class="field"><label for="categoryFilter">Category</label><select id="categoryFilter"><option value="">All categories</option></select></div>
      <div class="field"><label for="entryFilter">Entry</label><select id="entryFilter"><option value="">All entries</option></select></div>
      <div class="field"><label for="floodFilter">Flood</label><select id="floodFilter"><option value="">All flood levels</option></select></div>
      <div class="field"><label for="confidenceFilter">Confidence</label><select id="confidenceFilter"><option value="">Any</option><option value="0.35">35%+</option><option value="0.50">50%+</option><option value="0.70">70%+</option></select></div>
      <div class="field"><label for="priceMax">Max tokens</label><input id="priceMax" type="number" min="0" step="1" placeholder="Any" /></div>
      <div class="field"><label for="priceStateFilter">Price state</label><select id="priceStateFilter"><option value="">All</option><option value="current_low">Current low</option><option value="above_low">Above low</option></select></div>
      <div class="field"><label for="favoriteFilter">Bookmarks</label><select id="favoriteFilter"><option value="">All</option><option value="favorites">Favorites only</option><option value="not_favorites">Not favorites</option></select></div>
      <div class="field"><label for="refreshFavoritePricesBtn">2P prices</label><button id="refreshFavoritePricesBtn" class="refresh-prices-btn" type="button">Refresh Favorites</button><div id="priceFetchInlineStatus" class="price-fetch-status price-fetch-inline-status">2P refresh idle.</div></div>
      <div class="field"><label for="lowGapMax">Within low %</label><input id="lowGapMax" type="number" min="0" step="0.5" placeholder="5 or 10" /></div>
      <div class="field"><label for="sortPreset">Sort</label><select id="sortPreset"><option value="">Priority rank</option><option value="third_party_edge">2nd-party true edge</option><option value="third_party_low">2P lowest price</option><option value="demand_desc">Demand high first</option><option value="quality_desc">Quality high first</option><option value="flood_low">Flood low first</option><option value="flood_high">Flood high first</option><option value="expected_desc">Expected high first</option><option value="confidence_desc">Confidence high first</option><option value="value_edge_desc">Value edge high first</option><option value="current_low">Current low first</option><option value="low_gap">Closest to low</option><option value="price_asc">Price low to high</option><option value="price_desc">Price high to low</option></select></div>
      <div class="field"><label for="rowLimit">Rows</label><select id="rowLimit"><option value="120" selected>120 first</option><option value="240">240</option><option value="480">480</option><option value="0">All gradual</option></select></div>
      <div class="field"><label for="scoredFilter">Scored</label><select id="scoredFilter"><option value="">All</option><option value="true">Scored</option><option value="false">Unscored</option></select></div>
      <div class="field"><label>&nbsp;</label><button id="resetBtn">Reset</button></div>
    </section>
  </details>

  <main class="content">
    <details class="panel recommendations-shell" id="portfolioFocusPanel" open>
      <summary class="panel-head">
        <div>
          <div class="panel-title">Inventory-Aware Buy Focus by Finish</div>
          <div class="hint">Recommendations are split into Paper, Foil, Holo and Gold so you can judge each market separately while avoiding inventory concentration.</div>
        </div>
        <div class="hint" id="portfolioFocusHint">Load inventory to personalize suggestions.</div>
        <span class="collapse-cue" aria-hidden="true"></span>
      </summary>
      <div id="portfolioFocus" class="portfolio-focus-grid"></div>
    </details>

    <details class="panel inventory-shell" id="inventoryShell">
      <summary class="panel-head inventory-summary">
        <div>
          <div class="panel-title">Inventory Tracker</div>
          <div class="hint" id="inventorySaveHint">Serve through <code>inventory_server.py</code> to save edits into <code>Inventory/sticker_inventory.csv</code>.</div>
        </div>
        <div class="inventory-summary-stats">
          <div class="inventory-stat"><span>Items</span><b id="inventoryCount">0</b></div>
          <div class="inventory-stat"><span title="Estimated current market value of all tracked inventory items using the dashboard's latest Steam-side price.">Current</span><b id="inventoryCurrentValue">$0.00</b></div>
          <div class="inventory-stat"><span title="The purchase cost you entered for inventory rows. Blank purchase prices are excluded from this total.">Known Cost</span><b id="inventoryKnownCost">$0.00</b></div>
          <div class="inventory-stat"><span title="Profit or loss versus known purchase cost. It is only calculated for items with a saved buy price.">P/L</span><b id="inventoryPnl">$0.00</b></div>
        </div>
      </summary>
      <div class="inventory-body">
        <details class="inventory-op inventory-add-panel" id="inventoryAddPanel">
          <summary>Add / Edit Item <span>Single item entry</span></summary>
          <div class="inventory-op-body">
            <form id="inventoryForm" class="inventory-form">
              <input type="hidden" id="inventoryId" />
              <div class="inventory-rate-note"><b>100 tokens = $0.99</b><span>Fill either tokens or USD; the paired value is calculated automatically.</span></div>
              <div class="field">
                <label for="inventoryStickerInput">Sticker</label>
                <input id="inventoryStickerInput" list="inventoryStickerOptions" placeholder="Start typing sticker name" required />
                <datalist id="inventoryStickerOptions"></datalist>
              </div>
              <div class="field"><label for="inventoryQuantity">Qty</label><input id="inventoryQuantity" type="number" min="1" step="1" value="1" /></div>
              <div class="field"><label for="inventoryAccount">Steam account</label><input id="inventoryAccount" list="inventoryAccountOptions" placeholder="Main / Alt" /><datalist id="inventoryAccountOptions"></datalist></div>
              <div class="field"><label for="inventoryBoughtTokens">Bought tokens</label><input id="inventoryBoughtTokens" type="number" min="0" step="1" placeholder="Optional" /></div>
              <div class="field"><label for="inventoryBoughtUsd">Bought USD</label><input id="inventoryBoughtUsd" type="number" min="0" step="0.01" placeholder="Optional" /></div>
              <div class="field"><label for="inventoryAcquiredAt">Date</label><input id="inventoryAcquiredAt" type="date" /></div>
              <div class="field notes-field"><label for="inventoryNotes">Notes</label><input id="inventoryNotes" placeholder="Trade note, reason, storage" /></div>
              <div class="inventory-form-actions">
                <button id="inventorySubmit" type="submit">Add Item</button>
                <button id="inventoryCancel" type="button">Cancel</button>
              </div>
            </form>
          </div>
        </details>
        <div class="inventory-ops">
          <details class="inventory-op" id="inventoryBatchPanel">
            <summary>Batch Add <span>CSV or spreadsheet rows</span></summary>
            <div class="inventory-op-body">
              <div class="inventory-op-grid">
                <div class="field"><label for="inventoryBatchAccount">Default account</label><input id="inventoryBatchAccount" list="inventoryAccountOptions" placeholder="Used when row is blank" /></div>
                <div class="field"><label for="inventoryBatchDate">Default date</label><input id="inventoryBatchDate" type="date" /></div>
                <label class="check-row"><input id="inventoryBatchUseCurrentPrice" type="checkbox" checked />Use current market if price is blank</label>
                <div class="inventory-op-actions"><button class="mini-btn" id="inventoryBatchAddBtn" type="button">Add Batch</button></div>
              </div>
              <textarea id="inventoryBatchText" placeholder="Sticker, qty, account, bought tokens, bought USD, date, notes&#10;Example: nettik (Holo), 2, Main, 119, , 2026-05-30, first buy&#10;Example: Team Liquid (Holo), 1, Alt, , 112.00, 2026-05-30, FOMO check"></textarea>
              <div class="inventory-op-hint">Each quantity creates separate inventory rows. Exact names are best; partial names are accepted only when they match one sticker.</div>
            </div>
          </details>
          <details class="inventory-op" id="inventoryBulkPanel">
            <summary>Batch Edit Selected <span>Apply only filled fields</span></summary>
            <div class="inventory-op-body">
              <div class="inventory-op-grid three">
                <div class="field"><label for="inventoryBulkAccount">Account</label><input id="inventoryBulkAccount" list="inventoryAccountOptions" placeholder="Leave blank to keep" /></div>
                <div class="field"><label for="inventoryBulkBoughtTokens">Bought tokens</label><input id="inventoryBulkBoughtTokens" type="number" min="0" step="1" placeholder="Optional" /></div>
                <div class="field"><label for="inventoryBulkBoughtUsd">Bought USD</label><input id="inventoryBulkBoughtUsd" type="number" min="0" step="0.01" placeholder="Optional" /></div>
                <div class="field"><label for="inventoryBulkDate">Date</label><input id="inventoryBulkDate" type="date" /></div>
                <div class="field"><label for="inventoryBulkNotesMode">Notes</label><select id="inventoryBulkNotesMode"><option value="append">Append notes</option><option value="replace">Replace notes</option></select></div>
                <div class="field"><label for="inventoryBulkNotes">Note text</label><input id="inventoryBulkNotes" placeholder="Leave blank to keep" /></div>
              </div>
              <div class="inventory-op-actions">
                <button class="mini-btn" id="inventoryBulkApplyBtn" type="button">Apply to Selected</button>
                <span class="inventory-op-hint">Use selection controls below to target visible or manually checked items.</span>
              </div>
            </div>
          </details>
        </div>
        <div class="inventory-inline-note">
          <span class="inventory-status" id="inventoryStatus">Inventory not loaded yet.</span>
          <button class="mini-btn" id="inventoryDrawerInlineBtn" type="button">Open Inventory Window</button>
        </div>
      </div>
    </details>

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
      <div class="footer-note"><span id="renderHint">All matched records render in small batches so the dashboard remains responsive.</span><span id="priceFetchStatus" class="price-fetch-status">2P refresh idle.</span> Generated files are written under <code>visualized/</code>.</div>
    </section>
  </main>
  <div id="sparkTip" class="spark-tip" role="tooltip"></div>
  <div id="priceFetchToast" class="price-fetch-status price-fetch-toast" role="status" aria-live="polite">2P refresh idle.</div>
  <div id="detailModal" class="modal" hidden>
    <div class="modal-backdrop" data-close-modal></div>
    <div class="modal-dialog" role="dialog" aria-modal="true" aria-labelledby="detailTitle">
      <button class="modal-close" id="modalClose" type="button" aria-label="Close details">&times;</button>
      <div class="modal-content" id="modalContent"></div>
    </div>
  </div>
  <div id="inventoryDrawer" class="inventory-drawer modal" hidden>
    <div class="modal-backdrop" data-close-inventory-drawer></div>
    <div class="inventory-drawer-dialog" role="dialog" aria-modal="true" aria-labelledby="inventoryDrawerTitle">
      <div class="inventory-drawer-head">
        <div>
          <h2 id="inventoryDrawerTitle">Inventory</h2>
          <p>Each owned sticker is shown as a separate item, with current price, P/L, and market context.</p>
        </div>
        <button class="modal-close inventory-drawer-close" id="inventoryDrawerClose" type="button" aria-label="Close inventory">&times;</button>
      </div>
      <div class="inventory-toolbar">
        <div class="inventory-toolbar-left">
          <div class="view-toggle" role="group" aria-label="Inventory view mode">
            <button class="view-btn active" id="inventoryGridBtn" type="button" aria-pressed="true"><span class="view-icon grid-icon"></span>Grid</button>
            <button class="view-btn" id="inventoryListBtn" type="button" aria-pressed="false"><span class="view-icon list-icon"></span>List</button>
          </div>
          <input class="inventory-filter-input" id="inventorySearch" placeholder="Search inventory" />
          <select class="inventory-account-filter" id="inventoryAccountFilter"><option value="">All accounts</option></select>
          <select class="inventory-sort-filter" id="inventorySort" title="Sort inventory">
            <option value="date_desc">Newest first</option>
            <option value="date_asc">Oldest first</option>
            <option value="name_asc">Sticker A-Z</option>
            <option value="current_desc">Current value</option>
            <option value="cost_desc">Buy price</option>
            <option value="pnl_desc">P/L value</option>
            <option value="pnl_pct_desc">P/L percent</option>
            <option value="overpay_desc">2P overpay</option>
            <option value="overpay_pct_desc">2P overpay %</option>
            <option value="account_asc">Account</option>
          </select>
          <button class="mini-btn" id="inventoryClearFiltersBtn" type="button">Clear</button>
        </div>
        <div class="inventory-toolbar-right">
          <div class="inventory-grid-controls" id="inventoryGridControls">
            <label for="inventoryGridCols">Tiles</label>
            <select id="inventoryGridCols">
              <option value="auto">Auto</option>
              <option value="5" selected>5 per row</option>
              <option value="8">8 per row</option>
              <option value="12">12 per row</option>
              <option value="custom">Custom</option>
            </select>
            <input id="inventoryCustomCols" type="number" min="1" max="18" step="1" placeholder="More" hidden />
          </div>
          <button class="mini-btn" id="inventoryExportBtn" type="button">Download CSV</button>
        </div>
      </div>
      <div class="inventory-drawer-stats">
        <div class="inventory-drawer-stat"><span>Visible Items</span><b id="inventoryDrawerCount">0</b></div>
        <div class="inventory-drawer-stat"><span>Visible Current</span><b id="inventoryDrawerCurrentValue">$0.00</b></div>
        <div class="inventory-drawer-stat"><span>Visible Cost</span><b id="inventoryDrawerKnownCost">$0.00</b></div>
        <div class="inventory-drawer-stat"><span>Visible P/L</span><b id="inventoryDrawerPnl">$0.00</b></div>
      </div>
      <div class="inventory-selection-bar">
        <div class="inventory-selected-count" id="inventorySelectedCount">0 selected</div>
        <div class="inventory-selection-actions">
          <button class="mini-btn" id="inventorySelectVisibleBtn" type="button">Select Visible</button>
          <button class="mini-btn" id="inventoryClearSelectionBtn" type="button">Clear Selection</button>
          <button class="mini-btn danger" id="inventoryDeleteSelectedBtn" type="button">Delete Selected</button>
        </div>
      </div>
      <div id="inventoryGridView" class="inventory-grid-view"></div>
      <div id="inventoryListView" class="inventory-list-wrap" hidden>
        <table class="inventory-table">
          <thead>
            <tr>
              <th class="select-col">Select</th>
              <th>Sticker</th>
              <th>Account</th>
              <th title="The known purchase price saved for this inventory row.">Bought</th>
              <th title="Latest dashboard market value using collected Steam-side price data.">Current</th>
              <th title="Profit or loss compared with the saved purchase price.">P/L</th>
              <th>Market</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody id="inventoryTbody"></tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<script id="records-json" type="application/json">__DATA_JSON__</script>
<script id="series-json" type="application/json">__SERIES_JSON__</script>
<script id="second-market-json" type="application/json">__SECOND_MARKET_JSON__</script>
<script id="favorites-json" type="application/json">__FAVORITES_JSON__</script>
<script>
const records = JSON.parse(document.getElementById('records-json').textContent);
const historySeries = JSON.parse(document.getElementById('series-json').textContent);
const secondMarketSeries = JSON.parse(document.getElementById('second-market-json').textContent);
const embeddedFavoriteIds = JSON.parse(document.getElementById('favorites-json').textContent);
const verdictColors = __VERDICT_COLORS__;
const verdictOrder = __VERDICT_ORDER__;
let sortKey = 'priority_rank';
let sortDir = 1;
let filtered = [];
let viewMode = 'list';
let modalHistoryOpen = false;
let activeStickerModalId = null;
let activeInventoryModalId = null;
let priceFetchBusy = false;
let priceFetchState = new Map();
let inventoryItems = [];
let inventoryViewMode = 'grid';
let inventorySortMode = localStorage.getItem('cs2StickerInventorySort') || 'date_desc';
let inventoryApiOnline = false;
let selectedInventoryIds = new Set();
let renderSequence = 0;
let favoriteIds = new Set(Array.isArray(embeddedFavoriteIds) ? embeddedFavoriteIds.map(String) : []);
let topTrueEdgeIds = new Set();
const RENDER_CHUNK_SIZE = 70;
const USD_PER_TOKEN = 0.99 / 100;
const TOKENS_PER_USD = 100 / 0.99;
const MIN_REASONABLE_2P_USD = 0.10;
const GENERIC_2P_OUTLIER_RATIO = 0.45;
const TWO_P_STORAGE_KEY = 'cs2StickerFetched2PPricesV1';
const DEFAULT_VARIANTS = new Set(['Foil', 'Holo']);
const ALL_VARIANTS = ['Paper', 'Foil', 'Holo', 'Gold'];
const recordById = new Map(records.map(r => [String(r.sticker_id), r]));
const FAVORITES_STORAGE_KEY = 'cs2StickerFavorites';

const $ = (id) => document.getElementById(id);
const hasNum = (v) => v !== null && v !== undefined && v !== '' && Number.isFinite(Number(v));
const num = (v) => hasNum(v) ? Number(v) : null;
const fmt = (v, d=0) => hasNum(v) ? Number(v).toFixed(d) : '-';
const pct = (v, d=0) => hasNum(v) ? `${Number(v).toFixed(d)}%` : '-';
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const tokens = (v) => hasNum(v) ? Math.round(Number(v)).toLocaleString() : '-';
const money = (v) => hasNum(v) ? '$' + Number(v).toFixed(2) : '-';
function favoriteId(r) {
  return String(r?.sticker_id || r?.market_hash_name || r?.sticker || '');
}
function loadFavorites() {
  try {
    const parsed = JSON.parse(localStorage.getItem(FAVORITES_STORAGE_KEY) || '[]');
    favoriteIds = new Set([
      ...(Array.isArray(embeddedFavoriteIds) ? embeddedFavoriteIds.map(String) : []),
      ...(Array.isArray(parsed) ? parsed.map(String) : [])
    ]);
  } catch {
    favoriteIds = new Set(Array.isArray(embeddedFavoriteIds) ? embeddedFavoriteIds.map(String) : []);
  }
}
function saveFavorites() {
  localStorage.setItem(FAVORITES_STORAGE_KEY, JSON.stringify([...favoriteIds]));
  syncFavoritesToServer();
}
async function syncFavoritesToServer() {
  try {
    await fetch('/api/favorites', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({favorites:[...favoriteIds]})
    });
  } catch {
    // File-open mode keeps favorites in localStorage only.
  }
}
async function mergeServerFavorites() {
  try {
    const response = await fetch('/api/favorites', {cache:'no-store'});
    if (!response.ok) return;
    const payload = await response.json();
    const serverFavorites = Array.isArray(payload.favorites) ? payload.favorites.map(String) : [];
    if (!serverFavorites.length) return;
    const before = favoriteIds.size;
    serverFavorites.forEach(id => favoriteIds.add(id));
    if (favoriteIds.size !== before) {
      localStorage.setItem(FAVORITES_STORAGE_KEY, JSON.stringify([...favoriteIds]));
      applyFiltersPreservingScroll();
    }
  } catch {
    // The static dashboard can run without inventory_server.py.
  }
}
function isFavorite(r) {
  const id = favoriteId(r);
  return Boolean(id && favoriteIds.has(id));
}
function favoriteButtonHtml(r, compact=false) {
  const id = favoriteId(r);
  const active = isFavorite(r);
  const label = active ? 'Remove favorite' : 'Add favorite';
  return `<button class="favorite-btn ${compact ? 'compact' : ''} ${active ? 'active' : ''}" type="button" data-favorite="${esc(id)}" aria-pressed="${active}" title="${label}">${active ? '&#9733;' : '&#9734;'}${compact ? '' : `<span>${active ? 'Saved' : 'Favorite'}</span>`}</button>`;
}
function csgoskinsFetchable(r) {
  return ['Holo', 'Foil'].includes(normalizedVariant(r));
}
function priceFetchButtonHtml(r, compact=false) {
  const id = favoriteId(r);
  const fetchable = csgoskinsFetchable(r);
  const state = priceFetchState.get(id);
  const stateAttr = state ? ` data-state="${esc(state.tone || '')}"` : '';
  const label = state?.label || (compact ? '2P' : 'Fetch 2P');
  const title = fetchable
    ? state?.message || 'Try to refresh the CSGOSkins lowest price and update true edge.'
    : 'Live CSGOSkins refresh is limited to Holo/Foil to avoid slow or excessive requests.';
  return `<button class="fetch-price-btn ${compact ? 'compact' : ''}" type="button" data-fetch-price="${esc(id)}"${stateAttr} ${fetchable ? '' : 'disabled'} title="${esc(title)}">${esc(label)}</button>`;
}
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
    <div class="price-range-row prev${previousClass}">${metricLabel('Prev')}<div><b>${esc(previous.usd)}${previousChange}</b><small>${esc(previous.tokens)} tokens before current</small></div></div>
    <div class="price-range-row low">${metricLabel('Low')}<div><b>${esc(low.usd)}</b><small>${esc(low.tokens)} tokens</small></div></div>
    <div class="price-range-row high">${metricLabel('High')}<div><b>${esc(high.usd)}</b><small>${esc(high.tokens)} tokens</small></div></div>
  </div>`;
}

function inventoryItemsForRecord(r) {
  const stickerId = String(r?.sticker_id || '').trim();
  const sticker = String(r?.sticker || '').trim().toLowerCase();
  const variant = normalizedVariant(r);
  return inventoryItems.filter(item => {
    const itemStickerId = String(item.sticker_id || '').trim();
    if (stickerId && itemStickerId && itemStickerId === stickerId) return true;
    return String(item.sticker || '').trim().toLowerCase() === sticker
      && normalizedVariant(item) === variant;
  });
}

function ownedInventoryPriceHtml(r) {
  const owned = inventoryItemsForRecord(r);
  if (!owned.length) return '';
  const rows = owned.map((item, index) => {
    const bought = boughtLabel(item);
    const account = item.steam_account || 'No account';
    const meta = [item.acquired_at || '', item.notes || ''].filter(Boolean).join(' | ');
    return `<button class="owned-price-item" type="button" data-owned-inventory="${esc(item.inventory_id)}" title="Open this owned sticker in inventory">
      <span class="owned-price-account">#${index + 1} ${esc(account)}</span>
      <span class="owned-price-value">${esc(boughtShortLabel(item))}</span>
      <span class="owned-price-meta">${esc(bought)}${meta ? ` | ${esc(meta)}` : ''}</span>
    </button>`;
  }).join('');
  return `<div class="owned-price-panel" aria-label="Owned inventory copies">
    <div class="owned-price-head"><span>Owned in inventory</span><b>${owned.length}</b></div>
    <div class="owned-price-list">${rows}</div>
  </div>`;
}

function usdToTokens(value) {
  const n = num(value);
  return n === null ? null : n * TOKENS_PER_USD;
}
function steamLowUsd(r) {
  const lowTokens = num(r?.hist_min);
  const currentTokens = num(r?.price_tokens);
  const currentUsd = num(r?.usd_price);
  if (lowTokens === null || currentTokens === null || currentTokens <= 0 || currentUsd === null) return null;
  return (lowTokens / currentTokens) * currentUsd;
}
function marketplacePrice(r, key) {
  const markets = r?.csgoskins_markets && typeof r.csgoskins_markets === 'object' ? r.csgoskins_markets : {};
  if (key === 'CSFloat') return num(r?.csfloat_low_usd ?? markets.CSFloat);
  if (key === 'UUSkins') return num(r?.uuskins_low_usd ?? markets.UUSkins);
  return null;
}
function trustedThirdPartyLow(r) {
  const named = [marketplacePrice(r, 'CSFloat'), marketplacePrice(r, 'UUSkins')]
    .filter(v => v !== null && v >= MIN_REASONABLE_2P_USD);
  const direct = num(r?.csgoskins_low_usd);
  const candidates = [...named];
  if (direct !== null && direct >= MIN_REASONABLE_2P_USD) {
    if (!named.length || direct >= Math.min(...named) * GENERIC_2P_OUTLIER_RATIO) {
      candidates.push(direct);
    }
  }
  return candidates.length ? Math.min(...candidates) : null;
}
function csgoskinsDiscountPct(r) {
  const csgPrice = trustedThirdPartyLow(r);
  const steamUsd = num(r?.usd_price);
  if (csgPrice === null || steamUsd === null || steamUsd <= 0 || csgPrice <= 0) return null;
  return ((steamUsd - csgPrice) / steamUsd) * 100;
}
function csgoskinsDiscountAbs(r) {
  const csgPrice = trustedThirdPartyLow(r);
  const steamUsd = num(r?.usd_price);
  if (csgPrice === null || steamUsd === null || csgPrice <= 0) return null;
  return steamUsd - csgPrice;
}
function csgoskinsTrueEdgePct(r) {
  const csgPrice = trustedThirdPartyLow(r);
  const lowUsd = steamLowUsd(r);
  if (csgPrice === null || lowUsd === null || lowUsd <= 0 || csgPrice <= 0) return null;
  return ((lowUsd - csgPrice) / lowUsd) * 100;
}
function csgoskinsTrueEdgeAbs(r) {
  const csgPrice = trustedThirdPartyLow(r);
  const lowUsd = steamLowUsd(r);
  if (csgPrice === null || lowUsd === null || csgPrice <= 0) return null;
  return lowUsd - csgPrice;
}
function marketplaceSource(r, key) {
  const sources = r?.csgoskins_market_sources && typeof r.csgoskins_market_sources === 'object' ? r.csgoskins_market_sources : {};
  return sources[key] || '2P cache';
}

function marketplaceSearchName(r) {
  return String(r?.market_hash_name || r?.sticker || '').trim();
}

function marketplaceUrl(r, key) {
  const name = marketplaceSearchName(r);
  const encoded = encodeURIComponent(name);
  if (key === 'CSFloat' || key === 'csfloat') return `https://csfloat.com/search?sort_by=lowest_price&type=buy_now&market_hash_name=${encoded}`;
  if (key === 'UUSkins' || key === 'uuskins') return `https://www.uuskins.com/items?search_word=${encoded}`;
  return r?.csgoskins_url || '#';
}

function marketplaceChipHtml(r, key, label, icon, cls) {
  const price = marketplacePrice(r, key);
  const unavailable = price === null;
  const source = marketplaceSource(r, key);
  const url = marketplaceUrl(r, key);
  const title = unavailable
    ? `${label} price was not found in the cached CSGOSkins/SkinSniper offer data. Opens ${label} search.`
    : `${label} lowest offer parsed from ${source}. Opens ${label} search.`;
  return `<a class="store-chip ${cls} ${unavailable ? 'unavailable' : ''}" href="${esc(url)}" target="_blank" rel="noopener" title="${esc(title)}"><span class="store-icon">${esc(icon)}</span><b>${unavailable ? '-' : money(price)}</b></a>`;
}
function marketplaceOffersHtml(r) {
  if (!csgoskinsFetchable(r)) return '';
  return `<div class="store-offers" aria-label="Marketplace offer prices parsed from CSGOSkins/SkinSniper">
    ${marketplaceChipHtml(r, 'CSFloat', 'CSFloat', 'CF', 'csfloat')}
    ${marketplaceChipHtml(r, 'UUSkins', 'UUSkins', 'UU', 'uuskins')}
  </div>`;
}

function secondMarketStats(r, key) {
  const points = secondMarketPointsFor(r, key);
  if (!points.length) {
    const current = key === 'csfloat' ? marketplacePrice(r, 'CSFloat') : marketplacePrice(r, 'UUSkins');
    return {
      current,
      previous:null,
      low:current,
      count:current === null ? 0 : 1,
      lastTime:'',
      lowTime:''
    };
  }
  const currentPoint = points[points.length - 1];
  const previousPoint = points.length > 1 ? points[points.length - 2] : null;
  let lowPoint = points[0];
  points.forEach(point => {
    if (point.price !== null && (lowPoint.price === null || point.price < lowPoint.price)) lowPoint = point;
  });
  return {
    current:currentPoint.price,
    previous:previousPoint ? previousPoint.price : null,
    low:lowPoint ? lowPoint.price : null,
    count:points.length,
    lastTime:currentPoint.time || '',
    lowTime:lowPoint ? (lowPoint.time || '') : ''
  };
}

function secondMarketPriceCell(label, value, cls='') {
  return `<span class="second-market-price-cell ${cls}"><span>${esc(label)}</span><b>${value === null ? '-' : money(value)}</b></span>`;
}

function secondMarketPriceRowHtml(r, key, label, icon, cls) {
  const stats = secondMarketStats(r, key);
  const url = marketplaceUrl(r, key);
  if (!stats.count) {
    return `<a class="second-market-price-row ${cls}" href="${esc(url)}" target="_blank" rel="noopener" title="${esc(label)} has no saved price history yet. Opens ${esc(label)} search. Use Fetch 2P for this sticker.">
      <span class="second-market-price-source"><span class="store-icon">${esc(icon)}</span>${esc(label)}</span>
      ${secondMarketPriceCell('Now', null)}
      ${secondMarketPriceCell('Prev', null, 'prev')}
      ${secondMarketPriceCell('Low', null, 'low')}
    </a>`;
  }
  const tip = [
    `${label} saved 2P history`,
    `Current: ${money(stats.current)}`,
    `Previous: ${money(stats.previous)}`,
    `Lowest saved: ${money(stats.low)}`,
    `${stats.count} saved point${stats.count === 1 ? '' : 's'}`,
    stats.lastTime ? `Last: ${String(stats.lastTime).replace('T', ' ').replace('.000Z', 'Z')}` : '',
    stats.lowTime ? `Low: ${String(stats.lowTime).replace('T', ' ').replace('.000Z', 'Z')}` : ''
  ].filter(Boolean).join('\n');
  return `<a class="second-market-price-row ${cls}" href="${esc(url)}" target="_blank" rel="noopener" title="${esc(tip)}">
    <span class="second-market-price-source"><span class="store-icon">${esc(icon)}</span>${esc(label)}</span>
    ${secondMarketPriceCell('Now', stats.current)}
    ${secondMarketPriceCell('Prev', stats.previous, 'prev')}
    ${secondMarketPriceCell('Low', stats.low, 'low')}
  </a>`;
}

function secondMarketPriceGridHtml(r) {
  if (!csgoskinsFetchable(r)) return '';
  return `<div class="second-market-price-grid" aria-label="Saved second-market current, previous and low prices">
    ${secondMarketPriceRowHtml(r, 'csfloat', 'CSFloat', 'CF', 'cf')}
    ${secondMarketPriceRowHtml(r, 'uuskins', 'UUSkins', 'UU', 'uu')}
  </div>`;
}
function isCsgoskinsOpportunity(r) {
  const variant = normalizedVariant(r);
  const pctValue = csgoskinsTrueEdgePct(r);
  const absValue = csgoskinsTrueEdgeAbs(r);
  return ['Foil', 'Holo'].includes(variant) && pctValue !== null && absValue !== null && pctValue >= 10 && absValue >= 0.05;
}
function csgoskinsOpportunityTagHtml(r) {
  if (!isCsgoskinsOpportunity(r)) return '';
  return `<div class="price-opportunity-tag" title="CSGOSkins is materially cheaper than the collected Steam historical low. Verify liquidity, fees and seller reputation before buying.">True edge ${fmt(csgoskinsTrueEdgePct(r), 0)}% (${money(csgoskinsTrueEdgeAbs(r))})</div>`;
}
function csgoskinsTrueEdgeCardHtml(r) {
  const csgPrice = trustedThirdPartyLow(r);
  const lowUsd = steamLowUsd(r);
  if (csgPrice === null || lowUsd === null) {
    return `<div class="market-price-card true-edge unavailable" title="Needs both trusted 2P price and collected Steam historical low."><span>True Edge</span><b>-</b><small>vs Steam low unavailable</small></div>`;
  }
  const edgePct = csgoskinsTrueEdgePct(r);
  const edgeAbs = csgoskinsTrueEdgeAbs(r);
  const cls = edgePct === null || Math.abs(edgePct) < 0.5 ? 'flat' : edgePct > 0 ? 'pos' : 'neg';
  const sign = edgePct !== null && edgePct > 0 ? '+' : '';
  const cardClass = edgePct !== null && edgePct > 0 ? ' deal' : edgePct !== null && edgePct < -5 ? ' expensive' : '';
  return `<div class="market-price-card true-edge${cardClass}" title="True edge compares the trusted 2P low with the collected Steam historical low for this sticker. Positive means 2P is below the Steam low."><span>True Edge</span><b class="${cls}">${sign}${fmt(edgePct, 1)}%</b><small>${money(edgeAbs)} vs Steam low ${money(lowUsd)}</small></div>`;
}
function csgoskinsPriceHtml(r) {
  const csgPrice = trustedThirdPartyLow(r);
  const steamUsd = num(r.usd_price);
  const url = r.csgoskins_url || '#';
  if (csgPrice === null) {
    return `<a class="market-price-card skins unavailable" href="${esc(url)}" target="_blank" rel="noopener" title="${esc(r.csgoskins_status || 'No trusted 2P price cached')}"><span>2P Low</span><b>Check</b><small>price unavailable</small></a>`;
  }
  const tokenEquivalent = usdToTokens(csgPrice);
  const discount = csgoskinsDiscountPct(r);
  const diffClass = discount === null || Math.abs(discount) < 0.5 ? 'flat' : discount > 0 ? 'pos' : 'neg';
  const diffText = discount === null ? 'compare live' : discount > 0 ? `${fmt(discount, 1)}% cheaper` : `${fmt(Math.abs(discount), 1)}% higher`;
  const cardClass = isCsgoskinsOpportunity(r) ? ' deal' : discount !== null && discount < -5 ? ' expensive' : '';
  return `<a class="market-price-card skins${cardClass}" href="${esc(url)}" target="_blank" rel="noopener" title="Open CSGOSkins comparison. Positive discount means the trusted 2P low is cheaper than the dashboard Steam price."><span>2P Low</span><b>${money(csgPrice)}</b><small>${tokens(tokenEquivalent)} token eq. <em class="${diffClass}">${diffText}</em></small></a>`;
}
function priceCompareHtml(r) {
  const steamUrl = r.steam_market_url || '#';
  return `<div class="price-compare">
    <a class="market-price-card steam" href="${esc(steamUrl)}" target="_blank" rel="noopener" title="Open this sticker on Steam Community Market. Price shown is from collected CS2Tokens data."><span>Steam</span><b>${money(r.usd_price)}</b><small>${tokens(r.price_tokens)} tokens</small></a>
    <div class="market-price-card steam-low" title="Collected historical low converted to USD using the current token-to-USD ratio."><span>Steam Low</span><b>${money(steamLowUsd(r))}</b><small>${tokens(r.hist_min)} low tokens</small></div>
    ${csgoskinsPriceHtml(r)}
    ${csgoskinsTrueEdgeCardHtml(r)}
    ${marketplaceOffersHtml(r)}
    ${secondMarketPriceGridHtml(r)}
  </div>`;
}
function gridPriceStackHtml(r) {
  const high = tokenUsdPair(r.hist_max, r);
  const thirdPartyLow = trustedThirdPartyLow(r);
  const trueEdge = csgoskinsTrueEdgePct(r);
  const trueEdgeClass = trueEdge === null || Math.abs(trueEdge) < 0.5 ? 'flat' : trueEdge > 0 ? 'pos' : 'neg';
  const trueEdgeText = trueEdge === null ? '-' : `${trueEdge > 0 ? '+' : ''}${fmt(trueEdge, 0)}%`;
  return `<span class="grid-prices" title="Current Steam, Steam low, Steam high, trusted 2P low, and true edge versus Steam low.">
    <span><small>Steam</small><b>${money(r.usd_price)}</b></span>
    <span><small>Low</small><b>${money(steamLowUsd(r))}</b></span>
    <span><small>High</small><b>${esc(high.usd)}</b></span>
    <span><small>2P</small><b>${money(thirdPartyLow)}</b></span>
    <span><small>Edge</small><b class="${trueEdgeClass}">${trueEdgeText}</b></span>
  </span>`;
}
function recordForFetchId(id) {
  const key = String(id || '');
  return records.find(r =>
    favoriteId(r) === key ||
    String(r.sticker_id || '') === key ||
    String(r.market_hash_name || '') === key ||
    String(r.sticker || '') === key ||
    String(r.csgoskins_url || '') === key
  ) || null;
}

function readTwoPStore() {
  try {
    const parsed = JSON.parse(localStorage.getItem(TWO_P_STORAGE_KEY) || '{}');
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch {
    return {};
  }
}

function twoPStoreKey(item) {
  return String(item?.csgoskins_url || item?.sticker_id || item?.market_hash_name || item?.sticker || item?.id || '').trim();
}

function persistFetchedCsgoskinsItems(items) {
  if (!Array.isArray(items) || !items.length) return 0;
  const store = readTwoPStore();
  let saved = 0;
  items.forEach(item => {
    const key = twoPStoreKey(item);
    if (!key) return;
    store[key] = {
      ...item,
      saved_at: Math.floor(Date.now() / 1000)
    };
    saved += 1;
  });
  try {
    localStorage.setItem(TWO_P_STORAGE_KEY, JSON.stringify(store));
  } catch {
    return 0;
  }
  return saved;
}

function hydratePersistedCsgoskinsPrices() {
  const store = readTwoPStore();
  let applied = 0;
  Object.values(store).forEach(item => {
    if (applyFetchedCsgoskinsPrice(item)) applied += 1;
  });
  return applied;
}

async function hydrateServerCsgoskinsCache() {
  try {
    const response = await fetch('csgoskins_prices.json', {cache:'no-store'});
    if (!response.ok) return;
    const cache = await response.json();
    if (!cache || typeof cache !== 'object') return;
    const items = Object.entries(cache).map(([url, entry]) => {
      if (!entry || typeof entry !== 'object') return null;
      const r = recordForFetchId(url);
      const markets = entry.markets && typeof entry.markets === 'object' ? entry.markets : {};
      return {
        ...entry,
        id:r ? favoriteId(r) : url,
        sticker_id:r?.sticker_id || entry.sticker_id || '',
        sticker:r?.sticker || entry.sticker || '',
        variant:r ? normalizedVariant(r) : entry.variant || '',
        market_hash_name:r?.market_hash_name || entry.market_hash_name || '',
        csgoskins_url:url,
        price:entry.price,
        markets,
        csfloat_low_usd:markets.CSFloat,
        uuskins_low_usd:markets.UUSkins,
      };
    }).filter(Boolean);
    let applied = 0;
    items.forEach(item => {
      if (applyFetchedCsgoskinsPrice(item)) applied += 1;
    });
    if (applied) {
      persistFetchedCsgoskinsItems(items);
      computeSignalSets();
      applyFiltersPreservingScroll();
    }
  } catch {
    // Cache hydration is optional; live/API fetch still reports visible errors.
  }
}
function priceFetchPayload(r) {
  return {
    id: favoriteId(r),
    sticker_id: r.sticker_id || '',
    sticker: r.sticker || '',
    variant: normalizedVariant(r),
    market_hash_name: r.market_hash_name || '',
    csgoskins_url: r.csgoskins_url || '',
    steam_market_url: r.steam_market_url || ''
  };
}
function setPriceFetchStatus(message, tone='') {
  ['priceFetchStatus', 'priceFetchInlineStatus', 'priceFetchToast'].forEach(id => {
    const el = $(id);
    if (!el) return;
    el.textContent = message;
    el.dataset.tone = tone;
    if (id === 'priceFetchToast') {
      const visible = tone && message && !/idle/i.test(message);
      el.dataset.visible = visible ? 'true' : 'false';
    }
  });
}
function setPriceFetchButtonsBusy(isBusy) {
  const refresh = $('refreshFavoritePricesBtn');
  if (refresh) {
    refresh.disabled = isBusy;
    refresh.textContent = isBusy ? 'Refreshing...' : 'Refresh Favorites';
  }
  document.querySelectorAll('.fetch-price-btn').forEach(button => {
    button.classList.toggle('busy', isBusy);
    if (isBusy) button.setAttribute('aria-busy', 'true');
    else button.removeAttribute('aria-busy');
  });
}
function applyFetchedCsgoskinsPrice(item) {
  const r = recordForFetchId(item?.id || item?.sticker_id || item?.market_hash_name || item?.sticker);
  if (!r) return false;
  if (item.csgoskins_url) r.csgoskins_url = item.csgoskins_url;
  if ('price' in item) r.csgoskins_low_usd = num(item.price);
  const markets = item.markets && typeof item.markets === 'object' ? item.markets : {};
  r.csgoskins_markets = markets;
  r.csgoskins_market_sources = item.market_sources && typeof item.market_sources === 'object' ? item.market_sources : {};
  r.csfloat_low_usd = num(item.csfloat_low_usd ?? markets.CSFloat);
  r.uuskins_low_usd = num(item.uuskins_low_usd ?? markets.UUSkins);
  if (item.status) r.csgoskins_status = String(item.status);
  if (item.last_error) r.csgoskins_last_error = String(item.last_error);
  const fetchedAt = num(item.fetched_at) || Math.floor(Date.now() / 1000);
  if (r.sticker_id && (r.csfloat_low_usd !== null || r.uuskins_low_usd !== null || r.csgoskins_low_usd !== null)) {
    const list = secondMarketSeries[r.sticker_id] || [];
    const point = {
      time: new Date(fetchedAt * 1000).toISOString(),
      fetched_at: fetchedAt,
      low: r.csgoskins_low_usd,
      csfloat: r.csfloat_low_usd,
      uuskins: r.uuskins_low_usd,
      source: 'api'
    };
    const key = `${point.fetched_at}|${point.low}|${point.csfloat}|${point.uuskins}`;
    const exists = list.some(p => `${p.fetched_at}|${p.low}|${p.csfloat}|${p.uuskins}` === key);
    if (!exists) list.push(point);
    secondMarketSeries[r.sticker_id] = list.slice(-80);
  }
  return true;
}
function refreshOpenStickerModal() {
  const modal = $('detailModal');
  const content = $('modalContent');
  if (!modal || modal.hidden || !content || !activeStickerModalId) return;
  const r = recordForFetchId(activeStickerModalId);
  const item = activeInventoryModalId ? inventoryItems.find(row => row.inventory_id === activeInventoryModalId) : null;
  if (r) content.innerHTML = stickerDetailsHtml(r, item || null);
}
async function fetchCsgoskinsPricesFor(rows, label='selected stickers', triggerButton=null) {
  if (priceFetchBusy) {
    setPriceFetchStatus('A 2P price refresh is already running.', 'warn');
    return;
  }
  const originalButtonText = triggerButton ? triggerButton.textContent : '';
  const unique = [];
  const seen = new Set();
  rows.forEach(r => {
    if (!r) return;
    const id = favoriteId(r);
    if (!id || seen.has(id)) return;
    seen.add(id);
    unique.push(r);
  });
  const eligible = unique.filter(csgoskinsFetchable);
  if (!eligible.length) {
    setPriceFetchStatus('No Holo/Foil stickers selected for 2P refresh.', 'warn');
    return;
  }

  priceFetchBusy = true;
  document.body.classList.add('price-fetching');
  setPriceFetchButtonsBusy(true);
  if (triggerButton) {
    triggerButton.disabled = true;
    triggerButton.textContent = 'Fetching...';
  }
  setPriceFetchStatus(`Refreshing ${eligible.length} ${label}...`, 'busy');
  try {
    const response = await fetch('/api/csgoskins-price', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({items:eligible.map(priceFetchPayload)})
    });
    if (!response.ok) {
      const text = await response.text().catch(() => '');
      throw new Error(`HTTP ${response.status}${text ? ` - ${text.slice(0, 160)}` : ''}`);
    }
    const payload = await response.json();
    const items = Array.isArray(payload.items) ? payload.items : [];
    items.forEach(applyFetchedCsgoskinsPrice);
    const saved = persistFetchedCsgoskinsItems(items);
    computeSignalSets();
    applyFiltersPreservingScroll();
    refreshOpenStickerModal();
    const priced = items.filter(item => num(item.price) !== null).length;
    const csfloat = items.filter(item => num(item.csfloat_low_usd ?? item.markets?.CSFloat) !== null).length;
    const uuskins = items.filter(item => num(item.uuskins_low_usd ?? item.markets?.UUSkins) !== null).length;
    const fallback = items.filter(item => item.fallback_status || Object.values(item.market_sources || {}).includes('SkinSniper')).length;
    const live = items.filter(item => item.status === 'ok').length;
    const cached = items.filter(item => item.status === 'ok_cached_after_error').length;
    const failed = items.filter(item => String(item.status || '').startsWith('error')).length;
    const errorKinds = [...new Set(items
      .map(item => String(item.status || ''))
      .filter(status => status.startsWith('error'))
    )].slice(0, 3).join(', ');
    const details = [
      live ? `${live} live` : '',
      cached ? `${cached} cached` : '',
      fallback ? `${fallback} fallback` : '',
      `CF ${csfloat}`,
      `UU ${uuskins}`,
      failed ? `${failed} failed${errorKinds ? `: ${errorKinds}` : ''}` : ''
    ].filter(Boolean).join(', ');
    setPriceFetchStatus(`2P refresh done: ${priced}/${items.length} priced${details ? ` (${details})` : ''}. Saved ${saved} reload cache row${saved === 1 ? '' : 's'}.`, failed ? 'warn' : 'ok');
  } catch (error) {
    setPriceFetchStatus(`2P refresh failed: ${error.message || error}. Run python inventory_server.py and open the localhost dashboard URL.`, 'error');
  } finally {
    priceFetchBusy = false;
    document.body.classList.remove('price-fetching');
    setPriceFetchButtonsBusy(false);
    if (triggerButton) {
      triggerButton.disabled = false;
      triggerButton.textContent = originalButtonText || 'Fetch 2P';
    }
  }
}
function lowGapHtml(r) {
  if (isReleaseLow(r)) return '<div class="release-low-badge">Current low</div>';
  const gap = num(r.low_gap_pct);
  if (gap === null) return '';
  const cls = gap <= 5 ? ' near' : gap <= 10 ? ' mid' : '';
  const digits = gap < 10 ? 1 : 0;
  return `<div class="low-gap-badge${cls}">+${fmt(gap, digits)}% above low</div>`;
}
function compactNumber(v) {
  const n = num(v);
  if (n === null) return '-';
  const abs = Math.abs(n);
  if (abs >= 1000000) return `${(n / 1000000).toFixed(abs >= 10000000 ? 0 : 1)}M`;
  if (abs >= 1000) return `${(n / 1000).toFixed(abs >= 10000 ? 0 : 1)}k`;
  return Math.round(n).toLocaleString();
}
function activityMetrics(r) {
  const row = r || {};
  const latest = num(row.latest_popularity);
  const positive = num(row.positive_popularity_sum);
  const pressure = num(row.absolute_popularity_pressure);
  const share = num(row.latest_relative_demand_share);
  const primary = latest !== null ? latest : positive !== null ? positive : pressure;
  return {latest, positive, pressure, share, primary};
}
function marketCounterHtml(r, mode='full') {
  const metrics = activityMetrics(r);
  const title = [
    'CS2Tokens market activity counter',
    metrics.latest !== null ? `Latest popularity: ${Math.round(metrics.latest).toLocaleString()}` : '',
    metrics.positive !== null ? `Positive activity sum: ${Math.round(metrics.positive).toLocaleString()}` : '',
    metrics.pressure !== null ? `Activity pressure: ${Math.round(metrics.pressure).toLocaleString()}` : '',
    metrics.share !== null ? `Relative demand share: ${metrics.share.toFixed(6)}` : '',
    'This is collected activity/popularity, not a verified Steam listing supply count.'
  ].filter(Boolean).join('\n');
  if (metrics.primary === null) {
    return `<span class="market-count muted" title="${esc(title)}"><span class="market-count-dot"></span>No activity</span>`;
  }
  const label = mode === 'mini' ? compactNumber(metrics.primary) : `${compactNumber(metrics.primary)} activity`;
  const extra = mode === 'full' && metrics.positive !== null ? `<small>${compactNumber(metrics.positive)} positive</small>` : '';
  return `<span class="market-count ${mode}" title="${esc(title)}"><span class="market-count-dot"></span><b>${esc(label)}</b>${extra}</span>`;
}
const metricDescriptions = {
  'P/L':'Profit or loss compared with the known purchase price saved in inventory. Blank buy prices are excluded.',
  'Known Cost':'The buy price you entered for the inventory row or portfolio summary.',
  'Current':'Latest dashboard market value using collected Steam-side price data.',
  'Expected':'Model-estimated upside from current price, based on discount, trend, demand, quality and risk features.',
  'Value Edge':'Combined relative-value score. Higher means the sticker looks cheaper versus its history and peers after risk adjustments.',
  'Quality':'Manual visual grade from scores.csv where available, with neutral fallback for unscored stickers.',
  'Score Conf.':'Confidence in the quality score. Low values usually mean the sticker has not been manually scored yet.',
  'Manual':'Count or weight of manual score inputs used for the sticker.',
  'Manual Score':'Count or weight of manual score inputs used for the sticker.',
  'Demand':'Demand momentum from CS2Tokens popularity/activity history. Higher is stronger collected demand.',
  'Activity':'CS2Tokens popularity/activity counter, not verified Steam listing supply.',
  'Flood':'Model estimate of listing/supply pressure. Lower flood is usually healthier for upside.',
  'Discount':'How far current price is below the collected historical high.',
  'Change':'Most recent price change from snapshots/history.',
  'Entry':'Price bucket. Cheaper entries allow wider diversification, but still need quality and demand.',
  'Priority':'Final ranking score combining the model signals.',
  'Size':'Suggested maximum buy size from the model.',
  'Confidence':'Prediction confidence from available data/history.',
  'Prev':'Price point immediately before the current/latest point in the collected series.',
  'Low':'Lowest collected price since release/history start.',
  'High':'Highest collected price since release/history start.'
};
function metricLabel(label) {
  const title = metricDescriptions[label];
  return `<span${title ? ` class="metric-label" title="${esc(title)}"` : ''}>${esc(label)}</span>`;
}
const colorForVerdict = (v) => verdictColors[v] || '#94a3b8';
const pctClass = (v) => !hasNum(v) ? 'flat' : Number(v) > 0.5 ? 'pos' : Number(v) < -0.5 ? 'neg' : 'flat';
const shortName = (s, n=25) => String(s || '').replace(/\s*\((Paper|Foil|Holo)\)/i, '').slice(0, n);
const rarityPalette = {
  rarity_rare: {label:'High Grade', cls:'rarity-rare', accent:'#64a8ff', accent2:'#78f0ff', bg:'rgba(100,168,255,.18)', soft:'rgba(100,168,255,.075)'},
  rarity_mythical: {label:'Remarkable', cls:'rarity-mythical', accent:'#a78bfa', accent2:'#f0abfc', bg:'rgba(167,139,250,.18)', soft:'rgba(167,139,250,.075)'},
  rarity_legendary: {label:'Exotic', cls:'rarity-legendary', accent:'#ff7ad9', accent2:'#7dd3fc', bg:'rgba(255,122,217,.18)', soft:'rgba(255,122,217,.075)'},
  rarity_ancient: {label:'Extraordinary', cls:'rarity-ancient', accent:'#fbbf24', accent2:'#fde68a', bg:'rgba(251,191,36,.18)', soft:'rgba(251,191,36,.075)'},
};

function rarityInfo(r) {
  const id = String(r?.rarity_id || '').trim();
  if (rarityPalette[id]) return rarityPalette[id];
  const variant = String(r?.variant || r?.category || '').toLowerCase();
  if (variant.includes('gold')) return rarityPalette.rarity_ancient;
  if (variant.includes('holo')) return rarityPalette.rarity_legendary;
  if (variant.includes('foil')) return rarityPalette.rarity_mythical;
  return rarityPalette.rarity_rare;
}

function rarityStyleAttr(r) {
  const info = rarityInfo(r);
  return `style="--rarity:${info.accent};--rarity2:${info.accent2};--rarity-bg:${info.bg};--rarity-soft:${info.soft};"`;
}

function rarityClass(r) {
  return `rarity-accent ${rarityInfo(r).cls}`;
}

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

function variantCheckboxes() {
  return Array.from(document.querySelectorAll('[data-variant-option]'));
}

function selectedVariants() {
  return variantCheckboxes()
    .filter(input => input.checked)
    .map(input => normalizedVariant({variant:input.value}));
}

function syncVariantFilterLabel() {
  const button = $('variantFilterButton');
  if (!button) return;
  const selected = selectedVariants();
  button.textContent = selected.length ? selected.join(', ') : 'All variants';
}

function resetVariantFilter() {
  variantCheckboxes().forEach(input => {
    input.checked = DEFAULT_VARIANTS.has(normalizedVariant({variant:input.value}));
  });
  syncVariantFilterLabel();
}

function closeVariantMenu() {
  const shell = $('variantFilter');
  if (!shell) return;
  shell.dataset.open = 'false';
  $('variantFilterButton')?.setAttribute('aria-expanded', 'false');
}

function toggleVariantMenu() {
  const shell = $('variantFilter');
  if (!shell) return;
  const open = shell.dataset.open !== 'true';
  shell.dataset.open = String(open);
  $('variantFilterButton')?.setAttribute('aria-expanded', String(open));
}

function makeOptions() {
  fillSelect('verdictFilter', 'verdict');
  fillSelect('typeFilter', 'display_type');
  fillSelect('categoryFilter', 'category');
  fillSelect('entryFilter', 'entry_tier');
  fillSelect('floodFilter', 'flood_risk');
  syncVariantFilterLabel();
  const inventoryOptions = $('inventoryStickerOptions');
  if (inventoryOptions) {
    inventoryOptions.innerHTML = records.map(r => `<option value="${esc(r.sticker)}"></option>`).join('');
  }
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

function secondMarketPointsFor(r, key) {
  return (secondMarketSeries[r.sticker_id] || [])
    .map(point => ({
      price: num(point[key]),
      low: num(point.low),
      time: point.time || '',
      fetched_at: point.fetched_at,
      source: point.source || '2P history'
    }))
    .filter(point => point.price !== null);
}

function secondMarketChartFor(r, key, label, cls, width=230, height=54) {
  let points = secondMarketPointsFor(r, key);
  if (!points.length) {
    return `<div class="second-market-chart ${cls}"><header><span>${esc(label)}</span><small>-</small></header><div class="second-market-empty">2P history pending</div></div>`;
  }
  if (points.length === 1) {
    points = [{...points[0], synthetic:true}, {...points[0], synthetic:true}];
  }
  const prices = points.map(p => Number(p.price)).filter(Number.isFinite);
  const min = Math.min(...prices);
  const max = Math.max(...prices);
  const span = Math.max(max - min, 1e-9);
  const topPad = 8;
  const bottomPad = 9;
  const coords = points.map((p, i) => {
    const x = 8 + i * ((width - 16) / Math.max(points.length - 1, 1));
    const y = height - bottomPad - ((Number(p.price) - min) / span) * (height - topPad - bottomPad);
    return [x, y];
  });
  const line = coords.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ');
  const area = `${line} L${coords[coords.length - 1][0].toFixed(1)},${height - bottomPad} L${coords[0][0].toFixed(1)},${height - bottomPad} Z`;
  const latest = prices[prices.length - 1];
  const prior = prices.length > 1 ? prices[prices.length - 2] : null;
  const change = prior && prior > 0 ? ((latest - prior) / prior) * 100 : null;
  const color = cls === 'cf' ? '#50aeff' : '#ffd64c';
  const trendColor = change === null || Math.abs(change) < 0.5 ? '#bac6d6' : change > 0 ? '#5ee592' : '#fb7185';
  const dots = coords.map(([x, y], i) => {
    const p = points[i];
    const tip = [
      `${r.sticker}`,
      `${label}: ${money(p.price)}`,
      p.low !== null ? `Trusted 2P low: ${money(p.low)}` : '',
      p.time ? String(p.time).replace('T', ' ').replace('.000Z', 'Z') : '',
      p.source || '2P history'
    ].filter(Boolean).join('\n');
    return `<circle class="spark-point" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="3.8" fill="${color}" stroke="#080d14" stroke-width="1.6" data-tip="${esc(tip)}"></circle>`;
  }).join('');
  const changeText = change === null ? `${money(latest)}` : `${money(latest)} ${change > 0 ? '+' : ''}${fmt(change, 1)}%`;
  return `<div class="second-market-chart ${cls}">
    <header><span>${esc(label)}</span><small style="color:${trendColor}">${esc(changeText)}</small></header>
    <svg class="second-market-spark" viewBox="0 0 ${width} ${height}" aria-label="${esc(label)} second market trend">
      <path class="area" d="${area}" fill="${color}"></path>
      <path class="line" d="${line}" stroke="${color}"></path>
      ${dots}
    </svg>
  </div>`;
}

function secondMarketChartsHtml(r) {
  const points = secondMarketSeries[r.sticker_id] || [];
  if (!points.length && !csgoskinsFetchable(r)) return '';
  return `<div class="second-market-charts" aria-label="Second-market trend">
    ${secondMarketChartFor(r, 'csfloat', 'CSFloat', 'cf')}
    ${secondMarketChartFor(r, 'uuskins', 'UUSkins', 'uu')}
  </div>`;
}

function secondMarketSignalHtml(r) {
  const wait = num(r.second_market_wait_penalty);
  const edge = num(r.second_market_true_edge_pct);
  const change = num(r.second_market_change_pct);
  if (wait !== null && wait >= 0.55) {
    const reason = change !== null ? `2P falling ${pct(change, 1)}` : '2P still falling';
    return `<span class="second-market-signal wait" title="The off-market price trend is still making lower lows, so the analyzer reduces buy urgency.">${esc(reason)} · wait</span>`;
  }
  if (edge !== null && edge > 8) {
    return `<span class="second-market-signal edge" title="Trusted 2P price is below the collected Steam historical low.">${esc(`2P true edge +${fmt(edge, 1)}%`)}</span>`;
  }
  return '';
}

function rowHtml(r) {
  const points = historySeries[r.sticker_id] || [];
  const link = r.item_url || '#';
  const steamUrl = r.steam_market_url || '#';
  const vcolor = colorForVerdict(r.verdict);
  const expectedClass = pctClass(r.expected_return_pct);
  const demandClass = pctClass(r.demand_momentum_score);
  const changeValue = hasNum(r.snapshot_price_change_pct) ? r.snapshot_price_change_pct : r.recent_return_pct;
  const image = r.image_url || '';
  const typeLabel = r.display_type || r.category || '-';
  const atReleaseLow = isReleaseLow(r);
  const thirdPartyDeal = isCsgoskinsOpportunity(r);
  return `<tr class="${atReleaseLow ? 'release-low-row' : ''} ${thirdPartyDeal ? 'third-party-row' : ''} ${rarityClass(r)}" ${rarityStyleAttr(r)}>
    <td data-label="Rank"><div class="rank">#${esc(r.priority_rank)}</div><div class="tier">${esc(r.priority_tier || '')}</div>${signalTagsHtml(r, 'row')}</td>
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
            ${favoriteButtonHtml(r)}
            <a class="action primary" href="${esc(link)}" target="_blank" rel="noopener">CS2Tokens</a>
            <a class="action steam-action" href="${esc(steamUrl)}" target="_blank" rel="noopener">Steam</a>
            <a class="action skins-action" href="${esc(r.csgoskins_url || '#')}" target="_blank" rel="noopener">CSGOSkins</a>
            ${priceFetchButtonHtml(r)}
          </div>
        </div>
      </div>
    </td>
    <td data-label="Price">
      <div class="price-main">${money(r.usd_price)}</div>
      <div class="price-sub">${tokens(r.price_tokens)} tokens</div>
      ${priceCompareHtml(r)}
      ${ownedInventoryPriceHtml(r)}
      ${secondMarketSignalHtml(r)}
      ${csgoskinsOpportunityTagHtml(r)}
      ${lowGapHtml(r)}
      ${priceRangeHtml(r, points)}
      <div class="metric-list">
        <div class="metric-row">${metricLabel('Entry')}<b>${esc(r.entry_tier || '-')}</b></div>
      </div>
    </td>
    <td data-label="Decision">
      <span class="verdict" style="background:${vcolor}">${esc(r.verdict || '-')}</span>
      <div class="metric-list">
        <div class="metric-row">${metricLabel('Priority')}<b>${fmt(r.priority_score,1)}</b></div>
        <div class="metric-row">${metricLabel('Size')}<b>${esc(r.suggested_size || '-')}</b></div>
        <div class="metric-row">${metricLabel('Confidence')}<b>${fmt(r.prediction_confidence,2)}</b></div>
      </div>
    </td>
    <td data-label="Edge & Scores">
      <div class="metric-list" style="margin-top:0">
        <div class="metric-row">${metricLabel('Expected')}<b class="${expectedClass}">${pct(r.expected_return_pct,0)}</b></div>
        <div class="metric-row">${metricLabel('Value Edge')}<b>${fmt(r.value_edge_score,2)}</b></div>
        <div class="metric-row">${metricLabel('Quality')}<b>${fmt(r.quality_score,2)}</b></div>
        <div class="metric-row">${metricLabel('Score Conf.')}<b>${fmt(r.score_confidence,2)}</b></div>
        <div class="metric-row">${metricLabel('Manual')}<b>${fmt(r.manual_score_count,0)}</b></div>
      </div>
    </td>
    <td data-label="Market">
      <div class="metric-list" style="margin-top:0">
        <div class="metric-row market-activity-row">${metricLabel('Activity')}<b>${marketCounterHtml(r, 'full')}</b></div>
        <div class="metric-row">${metricLabel('Flood')}<b>${esc(r.flood_risk || '-')} (${fmt(r.flood_risk_score,2)})</b></div>
        <div class="metric-row">${metricLabel('Discount')}<b>${pct(r.discount_from_high_pct,0)}</b></div>
        <div class="metric-row">${metricLabel('Demand')}<b class="${demandClass}">${fmt(r.demand_momentum_score,2)}</b></div>
        <div class="metric-row">${metricLabel('Change')}<b class="${pctClass(changeValue)}">${pct(changeValue,1)}</b></div>
      </div>
      ${sparkline(points, r)}
      ${secondMarketChartsHtml(r)}
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
  const thirdPartyDeal = isCsgoskinsOpportunity(r);
  return `<div class="grid-card ${rarityClass(r)} ${atReleaseLow ? 'release-low-card' : ''} ${thirdPartyDeal ? 'third-party-card' : ''}" ${rarityStyleAttr(r)} role="button" tabindex="0" data-id="${esc(id)}" aria-label="Open details for ${esc(r.sticker)}">
    <span class="grid-rank">#${esc(r.priority_rank)}</span>
    <span class="grid-tier">${esc(r.priority_tier || '')}</span>
    ${favoriteButtonHtml(r, true)}
    ${signalTagsHtml(r, 'grid')}
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
      ${thirdPartyDeal ? `<span class="grid-deal-tag" title="CSGOSkins is below the collected Steam low by ${fmt(csgoskinsTrueEdgePct(r), 1)}%.">True edge ${fmt(csgoskinsTrueEdgePct(r), 0)}%</span>` : ''}
      ${gridPriceStackHtml(r)}
      ${priceFetchButtonHtml(r, true)}
      <span class="grid-market-counter">${marketCounterHtml(r, 'mini')}</span>
      <span class="grid-kpis">
        <span class="grid-kpi" title="${esc(metricDescriptions.Expected)}"><small>Expected</small><b class="${expectedClass}"><span class="signal-dot ${signalFor(r.expected_return_pct, atReleaseLow)}"></span>${pct(r.expected_return_pct,0)}</b></span>
        <span class="grid-kpi" title="${esc(metricDescriptions.Demand)}"><small>Demand</small><b class="${demandClass}"><span class="signal-dot ${signalFor(r.demand_momentum_score)}"></span>${fmt(r.demand_momentum_score,2)}</b></span>
      </span>
    </span>
  </div>`;
  }

function modalMetric(label, value, cls='') {
  return `<div class="metric-row">${metricLabel(label)}<b class="${cls}">${value}</b></div>`;
}

function inventoryContextHtml(item, r) {
  if (!item) return '';
  const pnl = inventoryPnl(item, r);
  const pnlValue = pnl.usdPct !== null
    ? `${pnl.usdAbs >= 0 ? '+' : ''}${money(pnl.usdAbs)} (${pnl.usdPct >= 0 ? '+' : ''}${fmt(pnl.usdPct, 1)}%)`
    : pnl.tokenPct !== null
      ? `${pnl.tokenPct >= 0 ? '+' : ''}${fmt(pnl.tokenPct, 1)}% tokens`
      : '-';
  const guard = inventoryOverpayInfo(item, r);
  const guardText = guard
    ? guard.pct >= 8
      ? `Paid ${money(guard.diff)} above 2P low`
      : guard.pct <= -8
        ? `Paid ${money(Math.abs(guard.diff))} below 2P low`
        : 'Near current 2P low'
    : '2P comparison unavailable';
  const guardClass = guard ? pnlClass(-guard.pct) : 'flat';
  return `<div class="inventory-context-panel" aria-label="Inventory purchase context">
    <div class="inventory-context-item"><span>Account</span><b>${esc(item.steam_account || '-')}</b><small>${esc(item.acquired_at || 'No buy date')}</small></div>
    <div class="inventory-context-item"><span>Buying Price</span><b>${esc(boughtLabel(item))}</b><small>${esc(metricDescriptions['Known Cost'])}</small></div>
    <div class="inventory-context-item"><span>Current Value</span><b>${money(pnl.currentUsd)}</b><small>${tokens(pnl.currentTokens)} tokens now</small></div>
    <div class="inventory-context-item"><span>P/L</span><b class="inventory-pnl ${pnlClass(pnl.usdPct ?? pnl.tokenPct)}">${esc(pnlValue)}</b><small>${esc(metricDescriptions['P/L'])}</small></div>
    <div class="inventory-context-item"><span>Buy Guard</span><b class="inventory-pnl ${guardClass}">${esc(guardText)}</b><small>${guard ? `${fmt(guard.pct, 1)}% vs trusted 2P low ${money(guard.best)}` : 'Refresh 2P prices first'}</small></div>
  </div>`;
}

function stickerDetailsHtml(r, inventoryItem=null) {
  const points = historySeries[r.sticker_id] || [];
  const vcolor = colorForVerdict(r.verdict);
  const expectedClass = pctClass(r.expected_return_pct);
  const demandClass = pctClass(r.demand_momentum_score);
  const changeValue = hasNum(r.snapshot_price_change_pct) ? r.snapshot_price_change_pct : r.recent_return_pct;
  const changeClass = pctClass(changeValue);
  const typeLabel = r.display_type || r.category || '-';
  const link = r.item_url || '#';
  const steamUrl = r.steam_market_url || '#';
  const image = r.image_url || '';
  return `<div class="modal-grid">
    <div class="modal-visual ${rarityClass(r)}" ${rarityStyleAttr(r)}>
      <span class="modal-rank">#${esc(r.priority_rank)} ${esc(r.priority_tier || '')}</span>
      <img src="${esc(image)}" alt="${esc(r.sticker)}" />
      <div class="actions">${favoriteButtonHtml(r)}<a class="action primary" href="${esc(link)}" target="_blank" rel="noopener">Open CS2Tokens</a><a class="action steam-action" href="${esc(steamUrl)}" target="_blank" rel="noopener">Open Steam</a><a class="action skins-action" href="${esc(r.csgoskins_url || '#')}" target="_blank" rel="noopener">Open CSGOSkins</a>${priceFetchButtonHtml(r)}<button class="action" type="button" data-add-inventory="${esc(r.sticker_id)}">Add to Inventory</button></div>
    </div>
    <div class="modal-main">
      <div class="modal-title-row">
        <h2 class="modal-title" id="detailTitle">${esc(r.sticker)}</h2>
        <span class="verdict" style="background:${vcolor}">${esc(r.verdict || '-')}</span>
      </div>
      <div class="modal-meta">${esc(r.player_name || r.team_name || r.team || 'No team')} | ${esc(typeLabel)} | ${esc(r.variant || '-')}</div>
      <div class="modal-price"><b>${money(r.usd_price)}</b><span>${tokens(r.price_tokens)} tokens</span></div>
      ${inventoryContextHtml(inventoryItem, r)}
      ${priceCompareHtml(r)}
      ${csgoskinsOpportunityTagHtml(r)}
      <div class="modal-market-count">${marketCounterHtml(r, 'full')}</div>
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
            ${modalMetric('Activity', marketCounterHtml(r, 'full'))}
            ${modalMetric('Discount', pct(r.discount_from_high_pct,0))}
            ${modalMetric('Demand', fmt(r.demand_momentum_score,2), demandClass)}
            ${modalMetric('Change', pct(changeValue,1), changeClass)}
          </div>
        </div>
        <div class="modal-section">
          <h3>Trend</h3>
          ${sparkline(points, r, 420, 118)}
        </div>
        <div class="modal-section">
          <h3>2P Trend</h3>
          ${secondMarketChartsHtml(r)}
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
  const variants = selectedVariants();
  const type = $('typeFilter').value;
  const category = $('categoryFilter').value;
  const entry = $('entryFilter').value;
  const flood = $('floodFilter').value;
  const scored = $('scoredFilter').value;
  const priceState = $('priceStateFilter').value;
  const favoriteFilter = $('favoriteFilter').value;
  const sortPreset = $('sortPreset').value;
  const minConfidence = num($('confidenceFilter').value);
  const maxPrice = num($('priceMax').value);
  const maxLowGap = num($('lowGapMax').value);

  filtered = records.filter(r => {
    if (verdict && r.verdict !== verdict) return false;
    if (variants.length && !variants.includes(normalizedVariant(r))) return false;
    if (type && r.display_type !== type) return false;
    if (category && r.category !== category) return false;
    if (entry && r.entry_tier !== entry) return false;
    if (flood && r.flood_risk !== flood) return false;
    if (scored && String(r.scored) !== scored) return false;
    if (favoriteFilter === 'favorites' && !isFavorite(r)) return false;
    if (favoriteFilter === 'not_favorites' && isFavorite(r)) return false;
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

function scheduleFrame(callback) {
  let called = false;
  const run = () => {
    if (called) return;
    called = true;
    callback();
  };
  if (typeof window.requestAnimationFrame === 'function') {
    window.requestAnimationFrame(run);
  }
  window.setTimeout(run, document.hidden ? 16 : 80);
}

function applyFiltersPreservingScroll() {
  const y = window.scrollY;
  const x = window.scrollX;
  const tableWrap = document.querySelector('.table-wrap');
  const tableLeft = tableWrap ? tableWrap.scrollLeft : 0;
  applyFilters();
  const restore = () => {
    window.scrollTo(x, y);
    if (tableWrap) tableWrap.scrollLeft = tableLeft;
  };
  scheduleFrame(restore);
  setTimeout(restore, 80);
  setTimeout(restore, 260);
}

function applySortPreset(sortPreset) {
  if (!sortPreset) return;
  if (sortPreset === 'third_party_edge') {
    sortKey = 'third_party_discount_pct';
    sortDir = -1;
  } else if (sortPreset === 'third_party_low') {
    sortKey = 'third_party_low_usd';
    sortDir = 1;
  } else if (sortPreset === 'demand_desc') {
    sortKey = 'demand_momentum_score';
    sortDir = -1;
  } else if (sortPreset === 'quality_desc') {
    sortKey = 'quality_score';
    sortDir = -1;
  } else if (sortPreset === 'flood_low') {
    sortKey = 'flood_risk_score';
    sortDir = 1;
  } else if (sortPreset === 'flood_high') {
    sortKey = 'flood_risk_score';
    sortDir = -1;
  } else if (sortPreset === 'expected_desc') {
    sortKey = 'expected_return_pct';
    sortDir = -1;
  } else if (sortPreset === 'confidence_desc') {
    sortKey = 'prediction_confidence';
    sortDir = -1;
  } else if (sortPreset === 'value_edge_desc') {
    sortKey = 'value_edge_score';
    sortDir = -1;
  } else if (sortPreset === 'current_low') {
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
  const sortLabels = {
    third_party_edge:'2nd-party true edge',
    third_party_low:'2P lowest price',
    demand_desc:'demand high first',
    quality_desc:'quality high first',
    flood_low:'flood low first',
    flood_high:'flood high first',
    expected_desc:'expected return high first',
    confidence_desc:'confidence high first',
    value_edge_desc:'value edge high first',
    current_low:'current low',
    low_gap:'distance from low',
    price_asc:'price low to high',
    price_desc:'price high to low'
  };
  $('sortHint').textContent = `Sorted by ${sortLabels[sortPreset] || sortPreset.replace(/_/g, ' ')}`;
}

function compareValues(a, b) {
  if (sortKey === 'verdict') {
    return (verdictOrder[a.verdict] ?? 99) - (verdictOrder[b.verdict] ?? 99);
  }
  if (sortKey === 'third_party_discount_pct') {
    const an = csgoskinsTrueEdgePct(a);
    const bn = csgoskinsTrueEdgePct(b);
    if (an === null && bn === null) return Number(b.priority_rank || 9999) - Number(a.priority_rank || 9999);
    if (an === null) return -1;
    if (bn === null) return 1;
    return an - bn;
  }
  if (sortKey === 'third_party_low_usd') {
    const an = trustedThirdPartyLow(a);
    const bn = trustedThirdPartyLow(b);
    if (an === null && bn === null) return Number(a.priority_rank || 9999) - Number(b.priority_rank || 9999);
    if (an === null) return 1;
    if (bn === null) return -1;
    return an - bn;
  }
  const av = a[sortKey], bv = b[sortKey];
  const an = num(av), bn = num(bv);
  if (an !== null || bn !== null) {
    if (an === null) return sortDir === -1 ? -1 : 1;
    if (bn === null) return sortDir === -1 ? 1 : -1;
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

function renderChunked(container, rows, renderer, emptyHtml, done) {
  const token = ++renderSequence;
  if (!container) return;
  const previousHeight = container.offsetHeight;
  if (previousHeight > 0) container.style.minHeight = `${previousHeight}px`;
  container.innerHTML = '';
  if (!rows.length) {
    container.innerHTML = emptyHtml;
    container.style.minHeight = '';
    if (done) done();
    return;
  }
  if (typeof window.requestAnimationFrame !== 'function' || document.hidden) {
    container.innerHTML = rows.map(renderer).join('');
    container.style.minHeight = '';
    if (done) done();
    return;
  }
  let index = 0;
  const step = () => {
    if (token !== renderSequence) return;
    const chunk = rows.slice(index, index + RENDER_CHUNK_SIZE).map(renderer).join('');
    container.insertAdjacentHTML('beforeend', chunk);
    index += RENDER_CHUNK_SIZE;
    const hint = $('renderHint');
    if (hint && rows.length > RENDER_CHUNK_SIZE) {
      hint.textContent = `Rendering ${Math.min(index, rows.length).toLocaleString()} of ${rows.length.toLocaleString()} matched rows.`;
    }
    if (index < rows.length) {
      scheduleFrame(step);
    } else {
      container.style.minHeight = '';
      if (done) done();
    }
  };
  scheduleFrame(step);
}

function inventoryGridColumnCount() {
  const mode = $('inventoryGridCols')?.value || '5';
  const custom = $('inventoryCustomCols');
  let count = mode === 'auto' ? (isMobileLayout() ? 2 : 5) : mode === 'custom' ? num(custom?.value) : num(mode);
  if (count === null) count = isMobileLayout() ? 2 : 5;
  return Math.max(1, Math.min(18, Math.round(count)));
}

function applyInventoryGridColumnSetting() {
  const grid = $('inventoryGridView');
  const custom = $('inventoryCustomCols');
  const controls = $('inventoryGridControls');
  if (!grid) return;
  const mode = $('inventoryGridCols')?.value || '5';
  const count = inventoryGridColumnCount();
  if (isMobileLayout()) {
    grid.style.gridTemplateColumns = `repeat(${Math.min(count, 2)}, minmax(0, 1fr))`;
  } else if (mode === 'auto') {
    grid.style.gridTemplateColumns = 'repeat(auto-fill, minmax(160px, 1fr))';
  } else {
    grid.style.gridTemplateColumns = `repeat(${count}, minmax(0, 1fr))`;
  }
  grid.dataset.density = count >= 12 ? 'ultra' : count >= 8 ? 'dense' : 'normal';
  if (custom) custom.hidden = mode !== 'custom';
  if (controls) controls.classList.toggle('active', inventoryViewMode === 'grid');
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
  renderChunked(grid, rows, gridCardHtml, '<div class="grid-empty">No stickers match the active filters.</div>', updateRenderHint);
}

function renderResults() {
  const rows = displayedRows();
  syncViewMode();
  if (viewMode === 'grid') {
    ++renderSequence;
    $('tbody').innerHTML = '';
    renderGrid(rows);
  } else {
    const grid = $('gridView');
    if (grid) grid.innerHTML = '';
    renderChunked($('tbody'), rows, rowHtml, `<tr><td colspan="7" class="empty">No stickers match the active filters.</td></tr>`, updateRenderHint);
  }
  $('visibleCount').textContent = `${rows.length.toLocaleString()}/${filtered.length.toLocaleString()}`;
  $('totalCount').textContent = records.length.toLocaleString();
  const expected = filtered.map(r => num(r.expected_return_pct)).filter(v => v !== null);
  const edge = filtered.map(r => num(r.value_edge_score)).filter(v => v !== null);
  const scored = filtered.filter(r => r.scored).length;
  $('avgExpected').textContent = expected.length ? pct(expected.reduce((a,b) => a + b, 0) / expected.length, 0) : '-';
  $('avgEdge').textContent = edge.length ? fmt(edge.reduce((a,b) => a + b, 0) / edge.length, 2) : '-';
  $('scoredCount').textContent = `${scored}/${filtered.length}`;
  updateRenderHint();
  updateMobileFilterSummary();
  renderPortfolioFocus();
}

function updateRenderHint() {
  const hint = $('renderHint');
  if (!hint) return;
  const rows = displayedRows();
  hint.textContent = rows.length < filtered.length
    ? `Showing ${rows.length.toLocaleString()} of ${filtered.length.toLocaleString()} matched rows. Choose All gradual for the full set.`
    : `Showing all ${filtered.length.toLocaleString()} matched rows with gradual rendering.`;
}

function nowIso() {
  return new Date().toISOString().replace(/\.\d{3}Z$/, 'Z');
}

function makeInventoryId() {
  if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
  return `inv_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function usdFromTokens(value) {
  const n = num(value);
  return n === null ? '' : (n * USD_PER_TOKEN).toFixed(2);
}

function tokensFromUsd(value) {
  const n = num(value);
  return n === null ? '' : String(Math.round(n * TOKENS_PER_USD));
}

function normalizeCostFields(tokenValue, usdValue) {
  const tokenText = String(tokenValue ?? '').trim();
  const usdText = String(usdValue ?? '').trim();
  const tokenNumber = num(tokenText);
  const usdNumber = num(usdText);
  return {
    bought_tokens: tokenNumber !== null ? String(Math.round(tokenNumber)) : (usdNumber !== null ? tokensFromUsd(usdNumber) : ''),
    bought_usd: usdNumber !== null ? Number(usdNumber).toFixed(2) : (tokenNumber !== null ? usdFromTokens(tokenNumber) : ''),
  };
}

function syncCostInputs(tokensId, usdId, source) {
  const tokenEl = $(tokensId);
  const usdEl = $(usdId);
  if (!tokenEl || !usdEl) return;
  if (source === 'tokens') {
    const next = usdFromTokens(tokenEl.value);
    usdEl.value = tokenEl.value.trim() ? next : '';
  } else {
    const next = tokensFromUsd(usdEl.value);
    tokenEl.value = usdEl.value.trim() ? next : '';
  }
}

function currentCostForRecord(r) {
  if (!r) return {bought_tokens:'', bought_usd:''};
  return normalizeCostFields(hasNum(r.price_tokens) ? Math.round(Number(r.price_tokens)) : '', hasNum(r.usd_price) ? Number(r.usd_price).toFixed(2) : '');
}

function inventoryFields() {
  return ['inventory_id','sticker_id','sticker','variant','category','steam_account','bought_tokens','bought_usd','acquired_at','notes','created_at','updated_at'];
}

function normalizeInventoryItem(item) {
  const r = recordById.get(String(item.sticker_id || '')) || records.find(row => String(row.sticker).toLowerCase() === String(item.sticker || '').toLowerCase());
  const stamp = nowIso();
  const cost = normalizeCostFields(item.bought_tokens, item.bought_usd);
  return {
    inventory_id: String(item.inventory_id || makeInventoryId()),
    sticker_id: String(item.sticker_id || r?.sticker_id || ''),
    sticker: String(item.sticker || r?.sticker || ''),
    variant: String(item.variant || r?.variant || ''),
    category: String(item.category || r?.category || ''),
    steam_account: String(item.steam_account || ''),
    bought_tokens: cost.bought_tokens,
    bought_usd: cost.bought_usd,
    acquired_at: String(item.acquired_at || ''),
    notes: String(item.notes || ''),
    created_at: String(item.created_at || stamp),
    updated_at: String(item.updated_at || stamp),
  };
}

function inventoryRecord(item) {
  return recordById.get(String(item.sticker_id || '')) || records.find(r => String(r.sticker).toLowerCase() === String(item.sticker || '').toLowerCase()) || null;
}

function resolveStickerInput(value) {
  const text = String(value || '').trim().toLowerCase();
  if (!text) return null;
  return records.find(r => String(r.sticker_id).toLowerCase() === text)
    || records.find(r => String(r.sticker).toLowerCase() === text)
    || records.find(r => String(r.market_hash_name || '').toLowerCase() === text)
    || records.find(r => String(r.sticker).toLowerCase().includes(text));
}

function resolveStickerBatchInput(value) {
  const text = String(value || '').trim().toLowerCase();
  if (!text) return {record:null, error:'missing sticker'};
  const exact = records.find(r => String(r.sticker_id).toLowerCase() === text)
    || records.find(r => String(r.sticker).toLowerCase() === text)
    || records.find(r => String(r.market_hash_name || '').toLowerCase() === text);
  if (exact) return {record:exact, error:''};
  const matches = records.filter(r => String(r.sticker).toLowerCase().includes(text));
  if (matches.length === 1) return {record:matches[0], error:''};
  if (matches.length > 1) return {record:null, error:`ambiguous: ${matches.slice(0, 3).map(r => r.sticker).join(' | ')}`};
  return {record:null, error:'not found'};
}

function inventoryFilteredItems() {
  const query = String($('inventorySearch')?.value || '').trim().toLowerCase();
  const account = String($('inventoryAccountFilter')?.value || '').trim();
  const filteredItems = inventoryItems.filter(item => {
    const r = inventoryRecord(item);
    if (account && item.steam_account !== account) return false;
    if (!query) return true;
    const haystack = [
      item.sticker, item.variant, item.category, item.steam_account, item.notes, item.acquired_at,
      r?.sticker, r?.team, r?.team_name, r?.player_name, r?.verdict, r?.display_type,
      item.bought_tokens, item.bought_usd, r?.price_tokens, r?.usd_price,
    ].join(' ').toLowerCase();
    return haystack.includes(query);
  });
  return sortInventoryItems(filteredItems);
}

function inventoryDateValue(item) {
  const value = Date.parse(item.acquired_at || item.updated_at || item.created_at || '');
  return Number.isFinite(value) ? value : 0;
}

function inventorySortValue(item, key) {
  const r = inventoryRecord(item);
  const pnl = inventoryPnl(item, r);
  if (key === 'name') return String(r?.sticker || item.sticker || '').toLowerCase();
  if (key === 'account') return String(item.steam_account || '').toLowerCase();
  if (key === 'date') return inventoryDateValue(item);
  if (key === 'current') return pnl.currentUsd ?? -Infinity;
  if (key === 'cost') return pnl.boughtUsd ?? -Infinity;
  if (key === 'pnl') return pnl.usdAbs ?? -Infinity;
  if (key === 'pnl_pct') return pnl.usdPct ?? pnl.tokenPct ?? -Infinity;
  if (key === 'overpay') return inventoryOverpayInfo(item, r)?.diff ?? -Infinity;
  if (key === 'overpay_pct') return inventoryOverpayInfo(item, r)?.pct ?? -Infinity;
  return 0;
}

function sortInventoryItems(items) {
  const mode = inventorySortMode || 'date_desc';
  const [key, dir] = mode.endsWith('_asc') ? [mode.replace(/_asc$/, ''), 'asc'] : [mode.replace(/_desc$/, ''), 'desc'];
  return [...items].sort((a, b) => {
    const av = inventorySortValue(a, key);
    const bv = inventorySortValue(b, key);
    let cmp = 0;
    if (typeof av === 'string' || typeof bv === 'string') cmp = String(av).localeCompare(String(bv));
    else cmp = Number(av) - Number(bv);
    if (cmp === 0) cmp = String(a.sticker || '').localeCompare(String(b.sticker || ''));
    return dir === 'asc' ? cmp : -cmp;
  });
}

function updateInventoryAccountFilter() {
  const select = $('inventoryAccountFilter');
  const accounts = [...new Set(inventoryItems.map(item => item.steam_account).filter(Boolean))].sort((a, b) => a.localeCompare(b));
  if (select) {
    const current = select.value;
    select.innerHTML = '<option value="">All accounts</option>' + accounts.map(account => `<option value="${esc(account)}">${esc(account)}</option>`).join('');
    if (accounts.includes(current)) select.value = current;
  }
  const datalist = $('inventoryAccountOptions');
  if (datalist) datalist.innerHTML = accounts.map(account => `<option value="${esc(account)}"></option>`).join('');
}

function pruneInventorySelection() {
  const valid = new Set(inventoryItems.map(item => item.inventory_id));
  selectedInventoryIds = new Set([...selectedInventoryIds].filter(id => valid.has(id)));
}

function updateInventorySelectionUi(visibleItems) {
  pruneInventorySelection();
  const count = selectedInventoryIds.size;
  const visibleCount = visibleItems.filter(item => selectedInventoryIds.has(item.inventory_id)).length;
  const text = count
    ? `${count.toLocaleString()} selected${visibleCount !== count ? ` (${visibleCount.toLocaleString()} visible)` : ''}`
    : '0 selected';
  const label = $('inventorySelectedCount');
  if (label) label.textContent = text;
  const hasSelection = count > 0;
  $('inventoryClearSelectionBtn')?.toggleAttribute('disabled', !hasSelection);
  $('inventoryDeleteSelectedBtn')?.toggleAttribute('disabled', !hasSelection);
  $('inventoryBulkApplyBtn')?.toggleAttribute('disabled', !hasSelection);
  $('inventorySelectVisibleBtn')?.toggleAttribute('disabled', visibleItems.length === 0);
}

function inventoryPnl(item, r) {
  const currentTokens = num(r?.price_tokens);
  const currentUsd = num(r?.usd_price);
  const boughtTokens = num(item.bought_tokens);
  const boughtUsd = num(item.bought_usd);
  const tokenPct = currentTokens !== null && boughtTokens !== null && boughtTokens > 0
    ? ((currentTokens - boughtTokens) / boughtTokens) * 100
    : null;
  const usdPct = currentUsd !== null && boughtUsd !== null && boughtUsd > 0
    ? ((currentUsd - boughtUsd) / boughtUsd) * 100
    : null;
  const usdAbs = currentUsd !== null && boughtUsd !== null && boughtUsd > 0 ? currentUsd - boughtUsd : null;
  return {currentTokens, currentUsd, boughtTokens, boughtUsd, tokenPct, usdPct, usdAbs};
}

function pnlClass(value) {
  const n = num(value);
  if (n === null || Math.abs(n) < 0.05) return 'flat';
  return n > 0 ? 'pos' : 'neg';
}

function boughtLabel(item) {
  const parts = [];
  if (num(item.bought_usd) !== null) parts.push(money(item.bought_usd));
  if (num(item.bought_tokens) !== null) parts.push(`${tokens(item.bought_tokens)} tokens`);
  return parts.length ? parts.join(' / ') : 'Not set';
}

function boughtShortLabel(item) {
  if (num(item.bought_usd) !== null) return money(item.bought_usd);
  if (num(item.bought_tokens) !== null) return `${tokens(item.bought_tokens)}t`;
  return '-';
}

function inventoryOverpayInfo(item, r) {
  const paid = num(item.bought_usd);
  const best = trustedThirdPartyLow(r);
  if (paid === null || best === null || best <= 0) return null;
  const diff = paid - best;
  return {paid, best, diff, pct: (diff / best) * 100};
}

function inventoryOverpayBadgeHtml(item, r, compact=false) {
  const info = inventoryOverpayInfo(item, r);
  if (!info) return '';
  if (info.pct >= 8) {
    return `<span class="inventory-guard-badge danger" title="Your saved buy price is ${money(info.diff)} above the current trusted 2P low of ${money(info.best)}. Compare stores before adding more.">${compact ? 'Over 2P' : `Paid +${fmt(info.pct, 1)}% vs 2P`}</span>`;
  }
  if (info.pct <= -8) {
    return `<span class="inventory-guard-badge good" title="Your saved buy price is ${money(Math.abs(info.diff))} below the current trusted 2P low of ${money(info.best)}.">${compact ? 'Below 2P' : `Paid ${fmt(info.pct, 1)}% vs 2P`}</span>`;
  }
  return `<span class="inventory-guard-badge neutral" title="Your saved buy price is close to the current trusted 2P low of ${money(info.best)}.">${compact ? 'Fair 2P' : 'Near 2P low'}</span>`;
}

function inventoryItemListHtml(item, index) {
  const r = inventoryRecord(item);
  const pnl = inventoryPnl(item, r);
  const image = r?.image_url || '';
  const title = r?.sticker || item.sticker || 'Unknown sticker';
  const market = r ? `${pct(r.expected_return_pct,0)} expected | ${esc(r.verdict || '-')}` : 'No market match';
  const selected = selectedInventoryIds.has(item.inventory_id);
  const pnlValue = pnl.usdPct !== null ? `${pnl.usdAbs >= 0 ? '+' : ''}${money(pnl.usdAbs)} (${pnl.usdPct >= 0 ? '+' : ''}${fmt(pnl.usdPct,1)}%)`
    : pnl.tokenPct !== null ? `${pnl.tokenPct >= 0 ? '+' : ''}${fmt(pnl.tokenPct,1)}% tokens`
    : '-';
  return `<tr class="${selected ? 'inventory-selected-row' : ''} ${rarityClass(r || item)}" ${rarityStyleAttr(r || item)} data-inventory-row="${esc(item.inventory_id)}" tabindex="-1">
    <td data-label="Select" class="inventory-select-cell"><input class="inventory-select" type="checkbox" data-select-inventory="${esc(item.inventory_id)}" ${selected ? 'checked' : ''} aria-label="Select ${esc(title)}" /></td>
    <td data-label="Sticker"><div class="inventory-sticker-cell"><img src="${esc(image)}" loading="lazy" decoding="async" onerror="this.style.visibility='hidden'" /><div><div class="inventory-item-title">#${index + 1} ${esc(title)}</div><div class="inventory-item-sub">${esc(r?.variant || item.variant || '-')} | ${esc(r?.team || r?.player_name || r?.team_name || '')}</div></div></div></td>
    <td data-label="Account">${esc(item.steam_account || '-')}</td>
    <td data-label="Bought">${esc(boughtLabel(item))}<div class="inventory-item-sub">${esc(item.acquired_at || '')}</div></td>
    <td data-label="Current">${money(pnl.currentUsd)}<div class="inventory-item-sub">${tokens(pnl.currentTokens)} tokens</div></td>
    <td data-label="P/L"><span class="inventory-pnl ${pnlClass(pnl.usdPct ?? pnl.tokenPct)}" title="${esc(metricDescriptions['P/L'])}">${esc(pnlValue)}</span><div class="inventory-item-sub">${inventoryOverpayBadgeHtml(item, r)}</div></td>
    <td data-label="Market">${market}<div class="inventory-item-sub">${marketCounterHtml(r, 'mini')} | Low ${tokens(r?.hist_min)} | High ${tokens(r?.hist_max)}</div></td>
    <td data-label="Actions"><div class="inventory-actions"><button class="mini-btn" type="button" data-inventory-details="${esc(item.inventory_id)}">Details</button><button class="mini-btn" type="button" data-edit-inventory="${esc(item.inventory_id)}">Edit</button><button class="mini-btn danger" type="button" data-delete-inventory="${esc(item.inventory_id)}">Delete</button></div></td>
  </tr>`;
}

function inventoryItemCardHtml(item, index) {
  const r = inventoryRecord(item);
  const pnl = inventoryPnl(item, r);
  const image = r?.image_url || '';
  const title = r?.sticker || item.sticker || 'Unknown sticker';
  const selected = selectedInventoryIds.has(item.inventory_id);
  const pnlText = pnl.usdPct !== null ? `${pnl.usdPct >= 0 ? '+' : ''}${fmt(pnl.usdPct,1)}%` : pnl.tokenPct !== null ? `${pnl.tokenPct >= 0 ? '+' : ''}${fmt(pnl.tokenPct,1)}%` : '-';
  const pnlCls = pnlClass(pnl.usdPct ?? pnl.tokenPct);
  const account = item.steam_account || 'No account';
  const bought = boughtLabel(item);
  return `<div class="inventory-card ${rarityClass(r || item)} ${selected ? 'inventory-selected-card' : ''}" ${rarityStyleAttr(r || item)} data-inventory-card="${esc(item.inventory_id)}" role="button" tabindex="0" aria-label="Open details for ${esc(title)}">
    <label class="inventory-card-select" title="Select item"><input class="inventory-select" type="checkbox" data-select-inventory="${esc(item.inventory_id)}" ${selected ? 'checked' : ''} /></label>
    <div class="inventory-card-art">
      <img src="${esc(image)}" loading="lazy" decoding="async" onerror="this.style.visibility='hidden'" alt="${esc(title)}" />
    </div>
    <div>
      <div class="inventory-card-title" title="${esc(title)}">#${index + 1} ${esc(title)}</div>
      <div class="inventory-card-meta"><span>${esc(account)}</span><span>|</span><span>${esc(item.acquired_at || 'No date')}</span></div>
      <div class="inventory-card-market">
        <span class="inventory-current-price" title="${esc(metricDescriptions.Current)}">${money(pnl.currentUsd)}</span>
        <span class="inventory-cost-pill" title="${esc(`Known cost: ${bought}`)}"><span>Cost</span><b>${esc(boughtShortLabel(item))}</b></span>
        <span class="inventory-pnl-pill inventory-pnl ${pnlCls}" title="${esc(metricDescriptions['P/L'])}">${esc(pnlText)}</span>
      </div>
      <div class="inventory-card-counter">${marketCounterHtml(r, 'mini')} ${inventoryOverpayBadgeHtml(item, r, true)}</div>
    </div>
    <div class="inventory-card-subrow">
      <span title="${esc(bought)}">Bought ${esc(bought)}</span>
      <span title="Low ${tokens(r?.hist_min)} | High ${tokens(r?.hist_max)}">L ${tokens(r?.hist_min)} / H ${tokens(r?.hist_max)}</span>
    </div>
    <div class="inventory-card-actions"><button class="mini-btn" type="button" data-inventory-details="${esc(item.inventory_id)}">Details</button><button class="mini-btn" type="button" data-edit-inventory="${esc(item.inventory_id)}">Edit</button><button class="mini-btn danger" type="button" data-delete-inventory="${esc(item.inventory_id)}">Delete</button></div>
  </div>`;
}

function portfolioKey(r) {
  return String(r?.portfolio_group || r?.team_name || r?.team || r?.player_name || r?.sticker || '').trim() || 'Unknown';
}

function portfolioExposureKey(r) {
  return `${normalizedVariant(r)} | ${portfolioKey(r)}`;
}

function goodBuyCandidate(r) {
  const verdictRank = verdictOrder[r.verdict] ?? 99;
  return verdictRank <= 4 && String(r.suggested_size || '') !== '0' && Number(r.priority_score || 0) > 0;
}

function watchCandidate(r) {
  const verdict = String(r.verdict || '');
  return ['SCORE/WAIT', 'WAIT FOR DROP'].includes(verdict) && Number(r.priority_score || 0) > 0;
}

function normalizedVariant(r) {
  const raw = String(r?.variant || '').trim();
  const lower = raw.toLowerCase();
  if (lower.includes('gold')) return 'Gold';
  if (lower.includes('holo')) return 'Holo';
  if (lower.includes('foil')) return 'Foil';
  if (lower.includes('paper')) return 'Paper';
  return raw || 'Paper';
}

function computeSignalSets() {
  const ranked = records
    .filter(r => ['Holo', 'Foil'].includes(normalizedVariant(r)) && csgoskinsTrueEdgePct(r) !== null)
    .sort((a, b) => (csgoskinsTrueEdgePct(b) ?? -Infinity) - (csgoskinsTrueEdgePct(a) ?? -Infinity));
  const limit = Math.max(12, Math.ceil(ranked.length * 0.08));
  topTrueEdgeIds = new Set(
    ranked
      .slice(0, limit)
      .filter(r => (csgoskinsTrueEdgePct(r) ?? -Infinity) > 0)
      .map(favoriteId)
  );
}

function isTopTrueEdge(r) {
  return topTrueEdgeIds.has(favoriteId(r));
}

function signalTags(r) {
  const out = [];
  const trueEdge = csgoskinsTrueEdgePct(r);
  const currentEdge = csgoskinsDiscountPct(r);
  const lowGap = num(r.low_gap_pct);
  if (isFavorite(r)) out.push({cls:'favorite', label:'Favorite', title:'Bookmarked sticker'});
  if (isTopTrueEdge(r)) out.push({cls:'edge', label:'Top 2P edge', title:'Among the strongest CSGOSkins prices versus Steam historical low'});
  if (trueEdge !== null && trueEdge > 0) out.push({cls:'edge', label:`Below low ${fmt(trueEdge,0)}%`, title:'Csgoskins is below the collected Steam historical low'});
  else if (currentEdge !== null && currentEdge > 10) out.push({cls:'watch', label:`2P cheaper ${fmt(currentEdge,0)}%`, title:'Csgoskins is cheaper than current Steam price, but not below the collected Steam low'});
  if (isReleaseLow(r)) out.push({cls:'low', label:'Steam low', title:'Current Steam-side price is at the collected low'});
  else if (lowGap !== null && lowGap <= 5) out.push({cls:'low', label:`Near low +${fmt(lowGap,1)}%`, title:'Current Steam-side price is within 5% of the collected low'});
  if (num(r.discount_from_high_pct) !== null && Number(r.discount_from_high_pct) >= 80) out.push({cls:'discount', label:'Deep discount', title:'Current price is at least 80% below the collected high'});
  return out.slice(0, 5);
}

function signalTagsHtml(r, mode='row') {
  const tags = signalTags(r);
  if (!tags.length) return '';
  return `<div class="signal-tags ${mode}">${tags.map(tag => `<span class="signal-tag ${tag.cls}" title="${esc(tag.title)}">${esc(tag.label)}</span>`).join('')}</div>`;
}

function renderPortfolioFocus() {
  const box = $('portfolioFocus');
  if (!box) return;
  const groupCounts = new Map();
  const heldIds = new Map();
  inventoryItems.forEach(item => {
    const r = inventoryRecord(item);
    if (!r) return;
    const key = portfolioExposureKey(r);
    groupCounts.set(key, (groupCounts.get(key) || 0) + 1);
    heldIds.set(String(r.sticker_id), (heldIds.get(String(r.sticker_id)) || 0) + 1);
  });
  const saturated = [...groupCounts.entries()].filter(([, count]) => count >= 2).sort((a, b) => b[1] - a[1]).slice(0, 4);

  const activeVariants = selectedVariants();
  const activeFilterIds = ['search','verdictFilter','typeFilter','categoryFilter','entryFilter','floodFilter','confidenceFilter','priceMax','priceStateFilter','favoriteFilter','lowGapMax','scoredFilter'];
  const anyFilterActive = activeFilterIds.some(id => String($(id)?.value || '').trim());
  const variants = activeVariants.length ? activeVariants : ALL_VARIANTS;
  const sourceRows = anyFilterActive ? filtered : records;
  const sortedCandidate = (variant) => {
    const base = sourceRows
      .filter(r => normalizedVariant(r) === variant)
      .map(r => ({r, exposure:groupCounts.get(portfolioExposureKey(r)) || 0, held:heldIds.get(String(r.sticker_id)) || 0}))
      .filter(item => item.held === 0);
    const buys = base
      .filter(item => goodBuyCandidate(item.r))
      .sort((a, b) => (a.exposure - b.exposure) || (Number(a.r.priority_rank || 9999) - Number(b.r.priority_rank || 9999)))
      .slice(0, 5)
      .map(item => ({...item, mode:'buy'}));
    if (buys.length) return buys;
    return base
      .filter(item => watchCandidate(item.r))
      .sort((a, b) => (Number(b.r.priority_score || 0) - Number(a.r.priority_score || 0)) || (Number(a.r.priority_rank || 9999) - Number(b.r.priority_rank || 9999)))
      .slice(0, 5)
      .map(item => ({...item, mode:'watch'}));
  };

  $('portfolioFocusHint').textContent = inventoryItems.length
    ? saturated.length
      ? `${inventoryItems.length} inventory items tracked. Avoid adding more in the same finish/group: ${saturated.map(([key, count]) => `${key} (${count})`).join(', ')}.`
      : `${inventoryItems.length} inventory items tracked; each finish favors groups with less exposure.`
    : activeVariants.length
      ? `Showing ${activeVariants.join(', ')} recommendations from the active filters.`
      : 'Split by Paper, Foil, Holo and Gold from the active filters.';

  const sections = variants.map(variant => ({variant, items:sortedCandidate(variant)}));
  if (!sections.some(section => section.items.length)) {
    box.innerHTML = '<div class="focus-empty">No underexposed buy candidates after the current filters. That usually means your inventory already overlaps the stronger candidates, the selected finish is too extended, or the model is asking you to wait.</div>';
    return;
  }

  box.innerHTML = sections.map(({variant, items}) => {
    const shellRecord = records.find(r => normalizedVariant(r) === variant) || {variant};
    const empty = `<div class="focus-empty small">No ${esc(variant)} candidate in the active filter set.</div>`;
    const cards = items.map(({r, exposure, mode}) => {
      const vcolor = colorForVerdict(r.verdict);
      const reason = exposure
        ? `${exposure} ${esc(normalizedVariant(r))} held in ${esc(portfolioKey(r))}; size carefully.`
        : mode === 'watch'
          ? `Best watch candidate; model is not calling this a buy yet.`
          : `Clean diversification against your current inventory.`;
      return `<button class="focus-card ${rarityClass(r)}" ${rarityStyleAttr(r)} type="button" data-id="${esc(r.sticker_id || r.sticker)}">
        <img src="${esc(r.image_url || '')}" loading="lazy" decoding="async" fetchpriority="low" onerror="this.style.visibility='hidden'" />
        <span class="focus-card-body">
          <span class="focus-title"><b>${esc(r.sticker)}</b><span class="focus-rank">#${esc(r.priority_rank)}</span></span>
          <span class="focus-note">${reason}</span>
          <span class="focus-meta"><span class="focus-chip" style="border-color:${vcolor}">${mode === 'watch' ? 'Best wait' : esc(r.verdict || '-')}</span><span class="focus-chip">${money(r.usd_price)}</span><span class="focus-chip">${pct(r.expected_return_pct,0)} exp.</span>${marketCounterHtml(r, 'mini')}</span>
        </span>
      </button>`;
    }).join('');
    return `<section class="recommendation-group ${rarityClass(shellRecord)}" ${rarityStyleAttr(shellRecord)}>
      <div class="recommendation-head"><span>${esc(variant)}</span><b>${items.length ? `${items.length} ${items[0].mode === 'watch' ? 'watch' : 'focus'}` : 'Waiting'}</b></div>
      <div class="recommendation-cards">${items.length ? cards : empty}</div>
    </section>`;
  }).join('');
}

function renderInventory() {
  updateInventoryAccountFilter();
  const visibleItems = inventoryFilteredItems();
  const grid = $('inventoryGridView');
  const tbody = $('inventoryTbody');
  applyInventoryGridColumnSetting();
  const emptyText = inventoryItems.length ? 'No inventory items match the active inventory filters.' : 'No inventory items yet. Add each physical sticker as a separate row.';
  if (grid) grid.innerHTML = visibleItems.length ? visibleItems.map(inventoryItemCardHtml).join('') : `<div class="inventory-empty">${emptyText}</div>`;
  if (tbody) tbody.innerHTML = visibleItems.length ? visibleItems.map(inventoryItemListHtml).join('') : `<tr><td colspan="8" class="empty">${emptyText}</td></tr>`;

  let currentValue = 0;
  let knownCost = 0;
  let knownPnl = 0;
  let visibleCurrentValue = 0;
  let visibleKnownCost = 0;
  let visibleKnownPnl = 0;
  inventoryItems.forEach(item => {
    const r = inventoryRecord(item);
    const pnl = inventoryPnl(item, r);
    if (pnl.currentUsd !== null) currentValue += pnl.currentUsd;
    if (pnl.boughtUsd !== null) {
      knownCost += pnl.boughtUsd;
      if (pnl.currentUsd !== null) knownPnl += pnl.currentUsd - pnl.boughtUsd;
    }
  });
  visibleItems.forEach(item => {
    const r = inventoryRecord(item);
    const pnl = inventoryPnl(item, r);
    if (pnl.currentUsd !== null) visibleCurrentValue += pnl.currentUsd;
    if (pnl.boughtUsd !== null) {
      visibleKnownCost += pnl.boughtUsd;
      if (pnl.currentUsd !== null) visibleKnownPnl += pnl.currentUsd - pnl.boughtUsd;
    }
  });
  $('inventoryCount').textContent = inventoryItems.length.toLocaleString();
  if ($('inventoryTopCount')) $('inventoryTopCount').textContent = `${inventoryItems.length.toLocaleString()} item${inventoryItems.length === 1 ? '' : 's'}`;
  $('inventoryCurrentValue').textContent = money(currentValue);
  $('inventoryKnownCost').textContent = money(knownCost);
  $('inventoryPnl').textContent = `${knownPnl >= 0 ? '+' : ''}${money(knownPnl)}`;
  $('inventoryPnl').className = `inventory-pnl ${pnlClass(knownPnl)}`;
  if ($('inventoryDrawerCount')) $('inventoryDrawerCount').textContent = visibleItems.length.toLocaleString();
  if ($('inventoryDrawerCurrentValue')) $('inventoryDrawerCurrentValue').textContent = money(visibleCurrentValue);
  if ($('inventoryDrawerKnownCost')) $('inventoryDrawerKnownCost').textContent = money(visibleKnownCost);
  if ($('inventoryDrawerPnl')) {
    $('inventoryDrawerPnl').textContent = `${visibleKnownPnl >= 0 ? '+' : ''}${money(visibleKnownPnl)}`;
    $('inventoryDrawerPnl').className = `inventory-pnl ${pnlClass(visibleKnownPnl)}`;
  }
  $('inventoryGridView')?.toggleAttribute('hidden', inventoryViewMode !== 'grid');
  $('inventoryListView')?.toggleAttribute('hidden', inventoryViewMode !== 'list');
  $('inventoryGridBtn')?.classList.toggle('active', inventoryViewMode === 'grid');
  $('inventoryListBtn')?.classList.toggle('active', inventoryViewMode === 'list');
  $('inventoryGridBtn')?.setAttribute('aria-pressed', String(inventoryViewMode === 'grid'));
  $('inventoryListBtn')?.setAttribute('aria-pressed', String(inventoryViewMode === 'list'));
  updateInventorySelectionUi(visibleItems);
  renderPortfolioFocus();
}

function setInventoryStatus(text, cls='') {
  const el = $('inventoryStatus');
  if (!el) return;
  el.textContent = text;
  el.className = `inventory-status ${cls}`;
}

async function loadInventory() {
  try {
    const response = await fetch('/api/inventory', {cache:'no-store'});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    inventoryItems = (payload.items || []).map(normalizeInventoryItem);
    inventoryApiOnline = true;
    setInventoryStatus(`CSV connected: ${payload.path || 'Inventory/sticker_inventory.csv'}`, 'ok');
  } catch (error) {
    inventoryApiOnline = false;
    const fallback = localStorage.getItem('cs2StickerInventory');
    inventoryItems = fallback ? JSON.parse(fallback).map(normalizeInventoryItem) : [];
    setInventoryStatus('Local browser mode. Run inventory_server.py to save CSV.', 'warn');
  }
  renderInventory();
  applyFiltersPreservingScroll();
}

async function persistInventory() {
  inventoryItems = inventoryItems.map(normalizeInventoryItem);
  localStorage.setItem('cs2StickerInventory', JSON.stringify(inventoryItems));
  if (!inventoryApiOnline) {
    setInventoryStatus('Saved in browser only. Start inventory_server.py for CSV persistence.', 'warn');
    renderInventory();
    applyFiltersPreservingScroll();
    return;
  }
  try {
    const response = await fetch('/api/inventory', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({items:inventoryItems})
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    inventoryItems = (payload.items || []).map(normalizeInventoryItem);
    setInventoryStatus(`Saved to ${payload.path || 'Inventory/sticker_inventory.csv'}`, 'ok');
  } catch (error) {
    inventoryApiOnline = false;
    setInventoryStatus(`CSV save failed; kept browser copy. ${error.message}`, 'error');
  }
  renderInventory();
  applyFiltersPreservingScroll();
}

function clearInventoryForm() {
  ['inventoryId','inventoryStickerInput','inventoryAccount','inventoryBoughtTokens','inventoryBoughtUsd','inventoryAcquiredAt','inventoryNotes']
    .forEach(id => { const el = $(id); if (el) el.value = ''; });
  if ($('inventoryQuantity')) {
    $('inventoryQuantity').value = '1';
    $('inventoryQuantity').disabled = false;
  }
  $('inventorySubmit').textContent = 'Add Item';
}

function fillInventoryForm(item) {
  closeInventoryDrawer();
  $('inventoryId').value = item.inventory_id || '';
  $('inventoryStickerInput').value = item.sticker || '';
  if ($('inventoryQuantity')) {
    $('inventoryQuantity').value = '1';
    $('inventoryQuantity').disabled = true;
  }
  $('inventoryAccount').value = item.steam_account || '';
  $('inventoryBoughtTokens').value = item.bought_tokens || '';
  $('inventoryBoughtUsd').value = item.bought_usd || '';
  $('inventoryAcquiredAt').value = item.acquired_at || '';
  $('inventoryNotes').value = item.notes || '';
  $('inventorySubmit').textContent = 'Update Item';
  $('inventoryShell')?.setAttribute('open', '');
  $('inventoryAddPanel')?.setAttribute('open', '');
  $('inventoryStickerInput')?.focus({preventScroll:false});
}

function startInventoryAdd(stickerId) {
  const r = recordById.get(String(stickerId));
  if (!r) return;
  closeInventoryDrawer();
  clearInventoryForm();
  const cost = currentCostForRecord(r);
  $('inventoryStickerInput').value = r.sticker;
  $('inventoryBoughtTokens').value = cost.bought_tokens;
  $('inventoryBoughtUsd').value = cost.bought_usd;
  $('inventoryAcquiredAt').valueAsDate = new Date();
  $('inventoryShell')?.setAttribute('open', '');
  $('inventoryAddPanel')?.setAttribute('open', '');
  $('inventoryAccount')?.focus({preventScroll:false});
}

function submitInventoryForm(event) {
  event.preventDefault();
  const r = resolveStickerInput($('inventoryStickerInput').value);
  if (!r) {
    setInventoryStatus('Sticker not found. Use the exact dashboard sticker name.', 'error');
    return;
  }
  const existingId = $('inventoryId').value;
  const existing = existingId ? inventoryItems.find(item => item.inventory_id === existingId) : null;
  const stamp = nowIso();
  const cost = normalizeCostFields($('inventoryBoughtTokens').value, $('inventoryBoughtUsd').value);
  const baseItem = {
    sticker_id: r.sticker_id,
    sticker: r.sticker,
    variant: r.variant,
    category: r.category,
    steam_account: $('inventoryAccount').value.trim(),
    bought_tokens: cost.bought_tokens,
    bought_usd: cost.bought_usd,
    acquired_at: $('inventoryAcquiredAt').value,
    notes: $('inventoryNotes').value.trim(),
    updated_at: stamp,
  };
  if (existingId) {
    const item = normalizeInventoryItem({
      ...baseItem,
      inventory_id: existingId,
      created_at: existing?.created_at || stamp,
    });
    inventoryItems = inventoryItems.map(row => row.inventory_id === existingId ? item : row);
  } else {
    const qty = Math.max(1, Math.min(500, Math.floor(num($('inventoryQuantity')?.value) || 1)));
    const newItems = Array.from({length:qty}, () => normalizeInventoryItem({
      ...baseItem,
      inventory_id: makeInventoryId(),
      created_at: stamp,
    }));
    inventoryItems = [...newItems, ...inventoryItems];
  }
  clearInventoryForm();
  persistInventory();
}

function parseDelimitedLine(line) {
  const delimiter = line.includes('\t') && !line.includes(',') ? '\t' : ',';
  const out = [];
  let current = '';
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (quoted && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else {
        quoted = !quoted;
      }
    } else if (ch === delimiter && !quoted) {
      out.push(current.trim());
      current = '';
    } else {
      current += ch;
    }
  }
  out.push(current.trim());
  return out;
}

function addInventoryBatch() {
  const text = String($('inventoryBatchText')?.value || '').trim();
  if (!text) {
    setInventoryStatus('Batch add needs at least one row.', 'error');
    return;
  }
  const defaultAccount = String($('inventoryBatchAccount')?.value || '').trim();
  const defaultDate = $('inventoryBatchDate')?.value || '';
  const useCurrent = Boolean($('inventoryBatchUseCurrentPrice')?.checked);
  const stamp = nowIso();
  const newItems = [];
  const errors = [];
  text.split(/\r?\n/).forEach((rawLine, idx) => {
    const line = rawLine.trim();
    if (!line) return;
    const cols = parseDelimitedLine(line);
    if (idx === 0 && String(cols[0] || '').toLowerCase() === 'sticker') return;
    const [stickerName, qtyText, accountText, tokenText, usdText, dateText, ...noteParts] = cols;
    const resolved = resolveStickerBatchInput(stickerName);
    if (!resolved.record) {
      errors.push(`line ${idx + 1}: ${resolved.error}`);
      return;
    }
    const qty = Math.max(1, Math.min(500, Math.floor(num(qtyText) || 1)));
    let cost = normalizeCostFields(tokenText, usdText);
    if (useCurrent && !cost.bought_tokens && !cost.bought_usd) cost = currentCostForRecord(resolved.record);
    const note = noteParts.join(', ').trim();
    for (let i = 0; i < qty; i++) {
      newItems.push(normalizeInventoryItem({
        inventory_id: makeInventoryId(),
        sticker_id: resolved.record.sticker_id,
        sticker: resolved.record.sticker,
        variant: resolved.record.variant,
        category: resolved.record.category,
        steam_account: String(accountText || defaultAccount || '').trim(),
        bought_tokens: cost.bought_tokens,
        bought_usd: cost.bought_usd,
        acquired_at: String(dateText || defaultDate || '').trim(),
        notes: note,
        created_at: stamp,
        updated_at: stamp,
      }));
    }
  });
  if (!newItems.length) {
    setInventoryStatus(`No batch rows were added. ${errors.slice(0, 3).join(' | ')}`, 'error');
    return;
  }
  inventoryItems = [...newItems, ...inventoryItems];
  $('inventoryBatchText').value = errors.length ? text : '';
  setInventoryStatus(`Added ${newItems.length.toLocaleString()} inventory rows${errors.length ? `; ${errors.length} row(s) skipped.` : '.'}`, errors.length ? 'warn' : 'ok');
  persistInventory();
}

function handleInventorySelectionChange(event) {
  const select = event.target.closest('[data-select-inventory]');
  if (!select) return;
  const id = select.dataset.selectInventory;
  if (select.checked) selectedInventoryIds.add(id);
  else selectedInventoryIds.delete(id);
  renderInventory();
}

function selectVisibleInventory() {
  inventoryFilteredItems().forEach(item => selectedInventoryIds.add(item.inventory_id));
  renderInventory();
}

function clearInventorySelection() {
  selectedInventoryIds.clear();
  renderInventory();
}

function deleteSelectedInventory() {
  const ids = [...selectedInventoryIds];
  if (!ids.length) {
    setInventoryStatus('Select inventory rows before deleting.', 'warn');
    return;
  }
  if (!confirm(`Delete ${ids.length.toLocaleString()} selected inventory item(s)?`)) return;
  const idSet = new Set(ids);
  inventoryItems = inventoryItems.filter(item => !idSet.has(item.inventory_id));
  selectedInventoryIds.clear();
  persistInventory();
}

function applyInventoryBulkEdit() {
  const ids = [...selectedInventoryIds];
  if (!ids.length) {
    setInventoryStatus('Select inventory rows before applying a batch edit.', 'warn');
    return;
  }
  const account = String($('inventoryBulkAccount')?.value || '').trim();
  const date = $('inventoryBulkDate')?.value || '';
  const note = String($('inventoryBulkNotes')?.value || '').trim();
  const noteMode = $('inventoryBulkNotesMode')?.value || 'append';
  const rawTokens = String($('inventoryBulkBoughtTokens')?.value || '').trim();
  const rawUsd = String($('inventoryBulkBoughtUsd')?.value || '').trim();
  const hasCost = Boolean(rawTokens || rawUsd);
  if (!account && !date && !note && !hasCost) {
    setInventoryStatus('Batch edit has no filled fields to apply.', 'warn');
    return;
  }
  const cost = normalizeCostFields(rawTokens, rawUsd);
  const stamp = nowIso();
  const idSet = new Set(ids);
  inventoryItems = inventoryItems.map(item => {
    if (!idSet.has(item.inventory_id)) return item;
    const next = {...item, updated_at:stamp};
    if (account) next.steam_account = account;
    if (date) next.acquired_at = date;
    if (hasCost) {
      next.bought_tokens = cost.bought_tokens;
      next.bought_usd = cost.bought_usd;
    }
    if (note) {
      next.notes = noteMode === 'replace' || !next.notes ? note : `${next.notes}; ${note}`;
    }
    return normalizeInventoryItem(next);
  });
  ['inventoryBulkAccount','inventoryBulkBoughtTokens','inventoryBulkBoughtUsd','inventoryBulkDate','inventoryBulkNotes']
    .forEach(id => { const el = $(id); if (el) el.value = ''; });
  persistInventory();
}

function handleInventoryClick(event) {
  const detail = event.target.closest('[data-inventory-details]');
  const edit = event.target.closest('[data-edit-inventory]');
  const del = event.target.closest('[data-delete-inventory]');
  const card = event.target.closest('.inventory-card[data-inventory-card]');
  if (detail) {
    const item = inventoryItems.find(row => row.inventory_id === detail.dataset.inventoryDetails);
    const r = item ? inventoryRecord(item) : null;
    if (r) openStickerModal(r.sticker_id, item.inventory_id);
  } else if (edit) {
    const item = inventoryItems.find(row => row.inventory_id === edit.dataset.editInventory);
    if (item) fillInventoryForm(item);
  } else if (del) {
    const item = inventoryItems.find(row => row.inventory_id === del.dataset.deleteInventory);
    if (item && confirm(`Delete ${item.sticker} from inventory?`)) {
      inventoryItems = inventoryItems.filter(row => row.inventory_id !== item.inventory_id);
      persistInventory();
    }
  } else if (card && !event.target.closest('button, a, input, label, select, textarea')) {
    const item = inventoryItems.find(row => row.inventory_id === card.dataset.inventoryCard);
    const r = item ? inventoryRecord(item) : null;
    if (r) openStickerModal(r.sticker_id, item.inventory_id);
  }
}

function selectorEscape(value) {
  if (window.CSS && typeof window.CSS.escape === 'function') return window.CSS.escape(String(value));
  return String(value).replace(/["\\]/g, '\\$&');
}

function openInventoryAtItem(inventoryId) {
  const id = String(inventoryId || '');
  if (!id) return;
  const item = inventoryItems.find(row => row.inventory_id === id);
  if (!item) {
    setInventoryStatus('Inventory item was not found. Refresh inventory and try again.', 'error');
    return;
  }
  openInventoryDrawer();
  if ($('inventorySearch')) $('inventorySearch').value = '';
  if ($('inventoryAccountFilter')) $('inventoryAccountFilter').value = '';
  renderInventory();
  requestAnimationFrame(() => {
    const escaped = selectorEscape(id);
    const candidates = [
      ...document.querySelectorAll(`[data-inventory-card="${escaped}"], [data-inventory-row="${escaped}"]`)
    ];
    const target = candidates.find(el => el.offsetParent !== null) || candidates[0];
    if (!target) return;
    target.scrollIntoView({behavior:'smooth', block:'center', inline:'nearest'});
    target.classList.add('inventory-jump-highlight');
    if (typeof target.focus === 'function') target.focus({preventScroll:true});
    window.setTimeout(() => target.classList.remove('inventory-jump-highlight'), 2100);
  });
}

function setupOwnedInventoryLinks() {
  document.addEventListener('click', event => {
    const button = event.target && event.target.closest ? event.target.closest('[data-owned-inventory]') : null;
    if (!button) return;
    event.preventDefault();
    event.stopPropagation();
    openInventoryAtItem(button.dataset.ownedInventory);
  });
}

function inventoryCsvText() {
  const fields = inventoryFields();
  const quote = value => `"${String(value ?? '').replace(/"/g, '""')}"`;
  return [fields.join(','), ...inventoryItems.map(item => fields.map(field => quote(item[field])).join(','))].join('\n');
}

function downloadInventoryCsv() {
  const blob = new Blob([inventoryCsvText()], {type:'text/csv;charset=utf-8'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'sticker_inventory.csv';
  a.click();
  URL.revokeObjectURL(url);
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
    applyInventoryGridColumnSetting();
    applyGridColumnSetting();
  }, {passive:true});
}

function updateMobileFilterSummary() {
  const summary = document.getElementById('mobileFilterSummary');
  if (!summary) return;
  const ids = ['search','verdictFilter','typeFilter','categoryFilter','entryFilter','floodFilter','confidenceFilter','priceMax','priceStateFilter','favoriteFilter','lowGapMax','sortPreset','scoredFilter'];
  const active = ids.reduce((count, id) => {
    const el = document.getElementById(id);
    return count + (el && String(el.value || '').trim() ? 1 : 0);
  }, selectedVariants().length ? 1 : 0);
  summary.textContent = active
    ? `${active} active - ${filtered.length.toLocaleString()} matches`
    : 'Tap to refine';
}

function setViewMode(mode) {
  viewMode = mode === 'grid' ? 'grid' : 'list';
  renderResults();
}

function openInventoryDrawer() {
  const drawer = $('inventoryDrawer');
  if (!drawer) return;
  drawer.hidden = false;
  document.body.style.overflow = 'hidden';
  renderInventory();
  $('inventoryDrawerClose')?.focus({preventScroll:true});
}

function closeInventoryDrawer() {
  const drawer = $('inventoryDrawer');
  if (!drawer || drawer.hidden) return;
  drawer.hidden = true;
  if ($('detailModal')?.hidden !== false) document.body.style.overflow = '';
}

function openStickerModal(id, inventoryId=null) {
  const r = recordById.get(String(id));
  const modal = $('detailModal');
  const content = $('modalContent');
  if (!r || !modal || !content) return;
  activeStickerModalId = String(id);
  activeInventoryModalId = inventoryId ? String(inventoryId) : null;
  const inventoryItem = activeInventoryModalId ? inventoryItems.find(row => row.inventory_id === activeInventoryModalId) : null;
  content.innerHTML = stickerDetailsHtml(r, inventoryItem || null);
  modal.hidden = false;
  document.body.style.overflow = 'hidden';
  if (!modalHistoryOpen && window.history && window.history.pushState) {
    window.history.pushState({stickerModal:true}, '', window.location.href);
    modalHistoryOpen = true;
  }
  $('modalClose')?.focus({preventScroll:true});
}

function closeStickerModal(fromPop=false) {
  const modal = $('detailModal');
  if (!modal || modal.hidden) return;
  modal.hidden = true;
  if ($('inventoryDrawer')?.hidden !== false) document.body.style.overflow = '';
  activeStickerModalId = null;
  activeInventoryModalId = null;
  const content = $('modalContent');
  if (content) content.innerHTML = '';
  if (modalHistoryOpen) {
    modalHistoryOpen = false;
    if (!fromPop && window.history) window.history.back();
  }
}

function setupDetailModal() {
  const grid = $('gridView');
  if (grid) {
    grid.addEventListener('click', event => {
      if (event.target && event.target.closest && event.target.closest('[data-favorite]')) return;
      if (event.target && event.target.closest && event.target.closest('[data-fetch-price]')) return;
      const card = event.target && event.target.closest ? event.target.closest('.grid-card') : null;
      if (card) openStickerModal(card.dataset.id);
    });
    grid.addEventListener('keydown', event => {
      if (event.key !== 'Enter' && event.key !== ' ') return;
      if (event.target && event.target.closest && event.target.closest('[data-favorite]')) return;
      if (event.target && event.target.closest && event.target.closest('[data-fetch-price]')) return;
      const card = event.target && event.target.closest ? event.target.closest('.grid-card') : null;
      if (card) {
        event.preventDefault();
        openStickerModal(card.dataset.id);
      }
    });
  }
  $('portfolioFocus')?.addEventListener('click', event => {
    const card = event.target && event.target.closest ? event.target.closest('.focus-card[data-id]') : null;
    if (card) openStickerModal(card.dataset.id);
  });
  $('modalContent')?.addEventListener('click', event => {
    const add = event.target && event.target.closest ? event.target.closest('[data-add-inventory]') : null;
    if (add) {
      const stickerId = add.dataset.addInventory;
      closeStickerModal();
      startInventoryAdd(stickerId);
    }
  });
  $('modalClose')?.addEventListener('click', closeStickerModal);
  document.querySelector('[data-close-modal]')?.addEventListener('click', closeStickerModal);
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape') {
      closeStickerModal();
      closeInventoryDrawer();
    }
  });
  window.addEventListener('popstate', () => {
    closeStickerModal(true);
  });
}

function setupInventoryDrawer() {
  $('inventoryDrawerBtn')?.addEventListener('click', openInventoryDrawer);
  $('inventoryDrawerInlineBtn')?.addEventListener('click', openInventoryDrawer);
  $('inventoryDrawerClose')?.addEventListener('click', closeInventoryDrawer);
  document.querySelector('[data-close-inventory-drawer]')?.addEventListener('click', closeInventoryDrawer);
}

function setupFavorites() {
  document.addEventListener('click', event => {
    const button = event.target && event.target.closest ? event.target.closest('[data-favorite]') : null;
    if (!button) return;
    event.preventDefault();
    event.stopPropagation();
    const id = String(button.dataset.favorite || '');
    if (!id) return;
    if (favoriteIds.has(id)) favoriteIds.delete(id);
    else favoriteIds.add(id);
    saveFavorites();
    applyFiltersPreservingScroll();
  });
}

function setupPriceFetch() {
  $('refreshFavoritePricesBtn')?.addEventListener('click', () => {
    const favoriteRows = records.filter(r => isFavorite(r) && csgoskinsFetchable(r));
    fetchCsgoskinsPricesFor(favoriteRows, 'favorite stickers');
  });
  document.addEventListener('click', event => {
    const button = event.target && event.target.closest ? event.target.closest('[data-fetch-price]') : null;
    if (!button) return;
    event.preventDefault();
    event.stopPropagation();
    const r = recordForFetchId(button.dataset.fetchPrice);
    if (!r) {
      setPriceFetchStatus('Could not find that sticker in the loaded dashboard data.', 'error');
      return;
    }
    setPriceFetchStatus(`Starting 2P refresh for ${r.sticker || 'selected sticker'}...`, 'busy');
    fetchCsgoskinsPricesFor([r], r.sticker || 'selected sticker', button);
  });
}

function wire() {
  loadFavorites();
  hydratePersistedCsgoskinsPrices();
  computeSignalSets();
  makeOptions();
  setupFilterPanel();
  if ($('inventoryBatchDate') && !$('inventoryBatchDate').value) $('inventoryBatchDate').valueAsDate = new Date();
  ['search','verdictFilter','typeFilter','categoryFilter','entryFilter','floodFilter','confidenceFilter','priceMax','priceStateFilter','favoriteFilter','lowGapMax','sortPreset','rowLimit','scoredFilter']
    .forEach(id => $(id).addEventListener('input', applyFilters));
  $('variantFilterButton')?.addEventListener('click', toggleVariantMenu);
  variantCheckboxes().forEach(input => input.addEventListener('change', () => {
    syncVariantFilterLabel();
    applyFilters();
  }));
  document.addEventListener('click', event => {
    const shell = $('variantFilter');
    if (shell && !shell.contains(event.target)) closeVariantMenu();
  });
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
  $('inventoryForm')?.addEventListener('submit', submitInventoryForm);
  $('inventoryCancel')?.addEventListener('click', clearInventoryForm);
  $('inventoryBoughtTokens')?.addEventListener('input', () => syncCostInputs('inventoryBoughtTokens', 'inventoryBoughtUsd', 'tokens'));
  $('inventoryBoughtUsd')?.addEventListener('input', () => syncCostInputs('inventoryBoughtTokens', 'inventoryBoughtUsd', 'usd'));
  $('inventoryBulkBoughtTokens')?.addEventListener('input', () => syncCostInputs('inventoryBulkBoughtTokens', 'inventoryBulkBoughtUsd', 'tokens'));
  $('inventoryBulkBoughtUsd')?.addEventListener('input', () => syncCostInputs('inventoryBulkBoughtTokens', 'inventoryBulkBoughtUsd', 'usd'));
  $('inventoryBatchAddBtn')?.addEventListener('click', addInventoryBatch);
  $('inventoryBulkApplyBtn')?.addEventListener('click', applyInventoryBulkEdit);
  $('inventorySearch')?.addEventListener('input', renderInventory);
  $('inventoryAccountFilter')?.addEventListener('input', renderInventory);
  if ($('inventorySort')) $('inventorySort').value = inventorySortMode;
  $('inventorySort')?.addEventListener('input', () => {
    inventorySortMode = $('inventorySort').value || 'date_desc';
    localStorage.setItem('cs2StickerInventorySort', inventorySortMode);
    renderInventory();
  });
  $('inventoryClearFiltersBtn')?.addEventListener('click', () => {
    if ($('inventorySearch')) $('inventorySearch').value = '';
    if ($('inventoryAccountFilter')) $('inventoryAccountFilter').value = '';
    if ($('inventorySort')) {
      $('inventorySort').value = 'date_desc';
      inventorySortMode = 'date_desc';
      localStorage.setItem('cs2StickerInventorySort', inventorySortMode);
    }
    renderInventory();
  });
  $('inventorySelectVisibleBtn')?.addEventListener('click', selectVisibleInventory);
  $('inventoryClearSelectionBtn')?.addEventListener('click', clearInventorySelection);
  $('inventoryDeleteSelectedBtn')?.addEventListener('click', deleteSelectedInventory);
  $('inventoryGridCols')?.addEventListener('input', () => {
    applyInventoryGridColumnSetting();
    renderInventory();
  });
  $('inventoryCustomCols')?.addEventListener('input', () => {
    applyInventoryGridColumnSetting();
    renderInventory();
  });
  $('inventoryGridBtn')?.addEventListener('click', () => {
    inventoryViewMode = 'grid';
    renderInventory();
  });
  $('inventoryListBtn')?.addEventListener('click', () => {
    inventoryViewMode = 'list';
    renderInventory();
  });
  $('inventoryGridView')?.addEventListener('click', handleInventoryClick);
  $('inventoryTbody')?.addEventListener('click', handleInventoryClick);
  $('inventoryGridView')?.addEventListener('change', handleInventorySelectionChange);
  $('inventoryTbody')?.addEventListener('change', handleInventorySelectionChange);
  $('inventoryExportBtn')?.addEventListener('click', downloadInventoryCsv);
  $('resetBtn').addEventListener('click', () => {
    ['search','verdictFilter','typeFilter','categoryFilter','entryFilter','floodFilter','confidenceFilter','priceMax','priceStateFilter','favoriteFilter','lowGapMax','sortPreset','scoredFilter']
      .forEach(id => $(id).value = '');
    resetVariantFilter();
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
  setupInventoryDrawer();
  setupOwnedInventoryLinks();
  setupFavorites();
  setupPriceFetch();
  applyFilters();
  hydrateServerCsgoskinsCache();
  mergeServerFavorites();
  loadInventory();
}

wire();
</script>
</body>
</html>"""
    return (
        template
        .replace("__DATA_JSON__", data_json)
        .replace("__SERIES_JSON__", series_json)
        .replace("__SECOND_MARKET_JSON__", second_market_json)
        .replace("__FAVORITES_JSON__", favorites_json)
        .replace("__VERDICT_COLORS__", json.dumps(VERDICT_COLORS))
        .replace("__VERDICT_ORDER__", json.dumps(VERDICT_ORDER))
    )


def main(fetch_2p: bool = True) -> None:
    analysis = load_analysis()
    history = load_history()
    series = build_history_series(analysis, history)
    records = [row_to_record(row) for _, row in analysis.iterrows()]
    enrich_csgoskins_prices(records, fetch_stale=fetch_2p)
    seeded = seed_second_market_history_from_records(records)
    if seeded:
        print(f"2P history cache seed points saved: {seeded}")
    second_market_series = build_second_market_series(records)
    write_priority_csv(analysis)
    html_text = build_html(records, series, second_market_series)
    out_path = OUT_DIR / "sticker_dashboard.html"
    out_path.write_text(html_text, encoding="utf-8")
    print(f"Dashboard written to {out_path}")
    print(f"Rows: {len(records)} | history series: {len(series)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-fetch-2p", action="store_true", help="Rebuild the dashboard from cached 2P prices without fetching CSGOSkins/UUSkins/CSFloat.")
    args = parser.parse_args()
    main(fetch_2p=not args.no_fetch_2p)
