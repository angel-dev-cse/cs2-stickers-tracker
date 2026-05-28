from __future__ import annotations

import argparse
import csv
import hashlib
from html.parser import HTMLParser
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_URL = "https://cs2tokens.com/browse?sort=price-asc"
DEFAULT_VARIANTS = "Paper,Foil,Holo,Gold"
USER_AGENT = "Mozilla/5.0 (Linux; Android; cs2-sticker-tracker/1.0)"
METADATA_TIMEOUT = 35
METADATA_RETRIES = 6
METADATA_RETRY_BASE_DELAY = 1.75

OUT_DIR = Path("data/snapshots")
HISTORY_DIR = Path("data/history")
OUT_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def parse_tokens(value: str | None) -> int | None:
    if not value:
        return None
    value = re.sub(r"[^\d]", "", value.replace(",", ""))
    return int(value) if value else None


def parse_usd(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value.replace("$", "").replace(",", "").strip())
    except ValueError:
        return None


def make_id(name: str, href: str) -> str:
    raw = f"{name}|{href}".lower()
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def normalize_variant(value: str | None) -> str:
    raw = clean_text(value)
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


def infer_variant_from_name(name: str) -> str:
    match = re.search(r"\((Paper|Foil|Holo|Gold|Glitter|Lenticular)\)", name, flags=re.I)
    return normalize_variant(match.group(1)) if match else "Paper"


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


def absolute_url(href: str) -> str:
    return urllib.parse.urljoin("https://cs2tokens.com", href)


def page_url(base_url: str, page: int) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    query = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    query.pop("finish", None)
    query.setdefault("sort", "price-asc")
    if page > 1:
        query["page"] = str(page)
    else:
        query.pop("page", None)
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urllib.parse.urlencode(query), parsed.fragment))


def item_slug_from_url(value: str) -> str:
    path = urllib.parse.urlsplit(value).path.rstrip("/")
    return path.rsplit("/", 1)[-1] if path else ""


def steam_market_listing_url(market_hash_name: str) -> str:
    if not market_hash_name:
        return ""
    return f"https://steamcommunity.com/market/listings/730/{urllib.parse.quote(market_hash_name, safe='')}"


def get_subject(name: str) -> str:
    name = name.replace("Sticker |", "").strip()
    before_event = name.split("|")[0].strip()
    return re.sub(r"\s*\((Paper|Holo|Foil|Gold|Glitter|Lenticular)\)\s*", "", before_event, flags=re.I).strip()


def get_event(name: str) -> str:
    parts = [p.strip() for p in name.split("|")]
    return parts[-1] if len(parts) >= 3 else ""


def infer_category(name: str, team: str) -> str:
    subject = get_subject(name).lower()
    team_norm = clean_text(team).lower()
    return "Team" if subject and team_norm and subject == team_norm else "Player"


def retry_delay_seconds(url: str, attempt: int, retry_after: str | None = None) -> float:
    if retry_after:
        try:
            return min(35.0, max(0.75, float(retry_after)))
        except ValueError:
            pass
    jitter = (int(hashlib.md5(url.encode("utf-8")).hexdigest()[:4], 16) % 1000) / 1000
    return min(35.0, METADATA_RETRY_BASE_DELAY * (2 ** attempt) + jitter)


def fetch_bytes(url: str, timeout: int = METADATA_TIMEOUT) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    retry_statuses = {429, 500, 502, 503, 504}
    for attempt in range(METADATA_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in retry_statuses or attempt >= METADATA_RETRIES:
                raise
            delay = retry_delay_seconds(url, attempt, exc.headers.get("Retry-After"))
            print(f"HTTP {exc.code}; retrying in {delay:.1f}s: {url}")
            time.sleep(delay)
        except urllib.error.URLError:
            if attempt >= METADATA_RETRIES:
                raise
            delay = retry_delay_seconds(url, attempt)
            print(f"Network error; retrying in {delay:.1f}s: {url}")
            time.sleep(delay)
    raise RuntimeError(f"Failed to fetch after retries: {url}")


def fetch_html(url: str) -> str:
    return fetch_bytes(url).decode("utf-8", errors="replace")


def fetch_json(url: str) -> Any:
    return json.loads(fetch_bytes(url).decode("utf-8"))


def attr_dict(attrs: list[tuple[str, str | None]]) -> dict[str, str]:
    return {key.lower(): value or "" for key, value in attrs}


def has_class(attrs: dict[str, str], class_name: str) -> bool:
    return class_name in attrs.get("class", "").split()


class BrowseHTMLParser(HTMLParser):
    void_tags = {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }

    field_classes = {
        "px-browse-grid__name": "name",
        "px-browse-grid__team": "team",
        "px-browse-grid__rarity": "rarity",
        "px-browse-grid__finish": "finish",
        "px-browse-grid__price-tkn": "price_tokens_raw",
        "px-browse-grid__price-usd": "price_usd_raw",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.cards: list[dict[str, Any]] = []
        self.page_numbers: list[int] = []
        self.text_parts: list[str] = []
        self.in_card = False
        self.card_depth = 0
        self.current_card: dict[str, Any] = {}
        self.active_fields: list[tuple[str, int]] = []

    def handle_starttag(self, tag: str, attrs_raw: list[tuple[str, str | None]]) -> None:
        attrs = attr_dict(attrs_raw)

        href = attrs.get("href", "")
        if tag == "a" and href:
            query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(href).query))
            if query.get("page", "").isdigit():
                self.page_numbers.append(int(query["page"]))

        if tag == "li" and has_class(attrs, "px-browse-grid__card") and not self.in_card:
            self.in_card = True
            self.card_depth = 1
            self.current_card = {
                "href": "",
                "aria_label": "",
                "name": "",
                "title": "",
                "team": "",
                "rarity": "",
                "rarity_color": "",
                "finish": "",
                "price_tokens_raw": "",
                "price_usd_raw": "",
                "image_url": "",
                "_text": {},
            }
            self.active_fields = []
            return

        if not self.in_card:
            return

        if tag not in self.void_tags:
            self.card_depth += 1

        if tag == "a" and has_class(attrs, "px-browse-grid__link"):
            self.current_card["href"] = attrs.get("href", "")
            self.current_card["aria_label"] = attrs.get("aria-label", "")

        if tag == "img" and attrs.get("src") and not self.current_card.get("image_url"):
            self.current_card["image_url"] = attrs.get("src", "")

        for class_name, field in self.field_classes.items():
            if has_class(attrs, class_name):
                self.active_fields.append((field, self.card_depth))
                if field == "name" and attrs.get("title"):
                    self.current_card["title"] = attrs["title"]
                if field == "rarity" and attrs.get("style"):
                    match = re.search(r"--rarity-color\s*:\s*([^;]+)", attrs["style"])
                    if match:
                        self.current_card["rarity_color"] = match.group(1).strip()

    def handle_data(self, data: str) -> None:
        if data:
            self.text_parts.append(data)
        if not self.in_card or not data.strip():
            return
        text_store = self.current_card.setdefault("_text", {})
        for field, _depth in self.active_fields:
            text_store.setdefault(field, []).append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self.in_card:
            return

        self.active_fields = [(field, depth) for field, depth in self.active_fields if depth != self.card_depth]

        if self.card_depth == 1 and tag == "li":
            text_store = self.current_card.pop("_text", {})
            for field, parts in text_store.items():
                if not self.current_card.get(field):
                    self.current_card[field] = clean_text(" ".join(parts))
            self.cards.append(dict(self.current_card))
            self.current_card = {}
            self.in_card = False
            self.card_depth = 0
            self.active_fields = []
            return

        self.card_depth = max(0, self.card_depth - 1)


def parse_browse_html(html: str) -> tuple[list[dict[str, Any]], int]:
    parser = BrowseHTMLParser()
    parser.feed(html)
    text = clean_text(" ".join(parser.text_parts))
    total_items = total_items_from_text(text, parser.page_numbers, len(parser.cards))
    return parser.cards, total_items


def total_items_from_text(text: str, page_numbers: list[int], fallback: int) -> int:
    match = re.search(r"Loading more\s+[-—]\s+[\d,]+\s+of\s+([\d,]+)", text, flags=re.I)
    if match:
        return int(match.group(1).replace(",", ""))
    match = re.search(r"All\s+([\d,]+)\s+items\s+loaded", text, flags=re.I)
    if match:
        return int(match.group(1).replace(",", ""))
    if page_numbers:
        return max(page_numbers) * 60
    return fallback


def parse_card(card: dict[str, Any], timestamp: str, run_id: str, source_url: str) -> dict[str, Any] | None:
    name = clean_text(card.get("title")) or clean_text(card.get("name"))
    if not name:
        return None

    finish = normalize_variant(card.get("finish")) or infer_variant_from_name(name)
    href = absolute_url(clean_text(card.get("href")))
    team = clean_text(card.get("team"))
    price_tokens = parse_tokens(card.get("price_tokens_raw"))
    usd_price = parse_usd(card.get("price_usd_raw"))
    image_url = absolute_url(clean_text(card.get("image_url")))

    if price_tokens is None:
        return None

    return {
        "timestamp": timestamp,
        "scrape_run_id": run_id,
        "source_url": source_url,
        "requested_variant": finish,
        "sticker_id": make_id(name, href),
        "name": name,
        "subject": get_subject(name),
        "event": get_event(name),
        "category": infer_category(name, team),
        "variant": finish,
        "team": team,
        "rarity": clean_text(card.get("rarity")),
        "rarity_color": clean_text(card.get("rarity_color")),
        "price_tokens": price_tokens,
        "usd_price": usd_price,
        "image_url": image_url,
        "image_formula": f'=IMAGE("{image_url}")' if image_url else "",
        "item_url": href,
    }


def collect_browse_rows(args, timestamp: str, run_id: str) -> list[dict[str, Any]]:
    wanted_variants = {v.lower() for v in parse_variants(args.variants)}
    rows_by_id: dict[str, dict[str, Any]] = {}

    first_url = page_url(args.url, 1)
    print(f"Collecting browse page 1: {first_url}")
    first_html = fetch_html(first_url)
    first_cards, total_items = parse_browse_html(first_html)
    page_count = max(1, math.ceil(total_items / max(1, len(first_cards) or 60)))
    if args.max_pages > 0:
        page_count = min(page_count, args.max_pages)

    for page in range(1, page_count + 1):
        url = first_url if page == 1 else page_url(args.url, page)
        if page == 1:
            cards = first_cards
        else:
            print(f"Collecting browse page {page}/{page_count}: {url}")
            cards, _total_items = parse_browse_html(fetch_html(url))
            if args.page_delay > 0:
                time.sleep(args.page_delay)

        parsed = 0
        for card in cards:
            row = parse_card(card, timestamp, run_id, url)
            if not row:
                continue
            if args.event.lower() not in row["event"].lower():
                continue
            if row["variant"].lower() not in wanted_variants:
                continue
            rows_by_id[row["sticker_id"]] = row
            parsed += 1
        print(f"Parsed {parsed} matching rows from page {page}")

    rows = list(rows_by_id.values())
    print(f"Collected {len(rows)} browse rows")
    return rows


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


def fetch_item_metadata(item_url: str) -> dict[str, Any]:
    qdata_url = item_url.rstrip("/") + "/q-data.json"
    base = {
        "metadata_status": "missing",
        "item_slug": item_slug_from_url(item_url),
        "qdata_url": qdata_url,
        "market_hash_name": "",
        "steam_market_url": "",
        "defindex": "",
        "paint_index": "",
        "volatile_item_id": "",
        "campaign_id": "",
        "campaign_slug": "",
        "token_key": "",
        "catalog_type": "",
        "sticker_type": "",
        "player_name": "",
        "team_name": "",
        "rarity_id": "",
        "_history_points": [],
    }
    try:
        payload = fetch_json(qdata_url)
        decoded = decode_qwik_data(payload, payload.get("_objs", []))
        metadata = None
        for item in walk_dicts(decoded):
            if "catalog" in item and "points" in item:
                metadata = item
                break
        if not isinstance(metadata, dict):
            return base | {"metadata_status": "metadata_missing"}

        catalog = metadata.get("catalog") if isinstance(metadata.get("catalog"), dict) else {}
        points = metadata.get("points") if isinstance(metadata.get("points"), list) else []
        first_point = next((p for p in points if isinstance(p, dict)), {})
        campaign = first_point.get("campaign") if isinstance(first_point.get("campaign"), dict) else {}
        market_hash_name = clean_text(catalog.get("market_hash_name") or catalog.get("name") or "")

        return base | {
            "metadata_status": "ok",
            "market_hash_name": market_hash_name,
            "steam_market_url": steam_market_listing_url(market_hash_name),
            "defindex": metadata.get("defindex", ""),
            "paint_index": metadata.get("paint_index", ""),
            "volatile_item_id": metadata.get("volatile_item_id", ""),
            "campaign_id": first_point.get("campaign_id", ""),
            "campaign_slug": clean_text(campaign.get("slug") or ""),
            "token_key": clean_text(first_point.get("token_key") or ""),
            "catalog_type": clean_text(catalog.get("type") or ""),
            "sticker_type": clean_text(catalog.get("sticker_type") or ""),
            "player_name": clean_text(catalog.get("player_name") or ""),
            "team_name": clean_text(catalog.get("team_name") or ""),
            "rarity_id": clean_text(catalog.get("rarity_id") or ""),
            "_history_points": points,
        }
    except Exception as exc:
        return base | {"metadata_status": "fetch_failed", "metadata_error": str(exc)}


def enrich_rows_with_metadata(rows: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    urls = sorted({row["item_url"] for row in rows if row.get("item_url")})
    metadata_by_url: dict[str, dict[str, Any]] = {}
    if not urls:
        return rows

    print(f"Fetching item metadata: {len(urls)} items with {workers} workers")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(fetch_item_metadata, url): url for url in urls}
        for future in as_completed(futures):
            url = futures[future]
            metadata_by_url[url] = future.result()

    failed_urls = [
        url for url, metadata in metadata_by_url.items()
        if metadata.get("metadata_status") == "fetch_failed"
    ]
    if failed_urls:
        print(f"Retrying failed metadata sequentially: {len(failed_urls)} items")
        for index, url in enumerate(failed_urls, start=1):
            time.sleep(1.0)
            print(f"  retry metadata {index}/{len(failed_urls)}")
            metadata_by_url[url] = fetch_item_metadata(url)

    for row in rows:
        metadata = metadata_by_url.get(row.get("item_url"), {})
        row.update(metadata)
        if row.get("market_hash_name"):
            row["name"] = clean_text(row["market_hash_name"])
            row["subject"] = get_subject(row["name"])
        if row.get("team_name"):
            row["team"] = clean_text(row["team_name"])
        if row.get("sticker_type"):
            row["category"] = "Team" if row["sticker_type"].lower() == "team" else "Player"
        if row.get("rarity_id") and not row.get("rarity"):
            row["rarity"] = row["rarity_id"]
    return rows


def rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in sorted(rows, key=lambda r: (r["category"], r["variant"], r["price_tokens"])):
        grouped.setdefault((row["category"], row["variant"]), []).append(row)

    final_rows: list[dict[str, Any]] = []
    for group in grouped.values():
        total = len(group)
        for rank, row in enumerate(group, start=1):
            row["rank_low_to_high"] = rank
            row["total_in_group"] = total
            row["price_percentile"] = round((rank - 1) / (total - 1), 4) if total > 1 else 0
            final_rows.append(row)
    return final_rows


def collect_detail_history(row: dict[str, Any], range_label: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    points = row.get("_history_points") or []
    for index, point in enumerate(points, start=1):
        if not isinstance(point, dict):
            continue
        token_cost = point.get("token_cost")
        fetched_at = point.get("fetched_at")
        if token_cost in (None, "") or not fetched_at:
            continue
        out.append({
            "history_scrape_timestamp": row["timestamp"],
            "sticker_id": row["sticker_id"],
            "name": row["name"],
            "category": row["category"],
            "variant": row["variant"],
            "team": row["team"],
            "current_price_tokens": row["price_tokens"],
            "current_usd_price": row["usd_price"],
            "item_url": row["item_url"],
            "history_range": range_label,
            "point_index": index,
            "x": "",
            "y": "",
            "token_cost": token_cost,
            "usd_price": point.get("usd_cost"),
            "popularity": point.get("popularity"),
            "tooltip_time_raw": fetched_at,
            "tooltip_raw": "",
            "fetched_at": fetched_at,
            "paint_index": point.get("paint_index", row.get("paint_index", "")),
            "volatile_item_id": point.get("volatile_item_id", row.get("volatile_item_id", "")),
            "campaign_id": point.get("campaign_id", row.get("campaign_id", "")),
            "token_key": clean_text(point.get("token_key") or row.get("token_key") or ""),
        })
    return out


def select_history_targets(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    targets = sorted(rows, key=lambda row: row["price_tokens"])
    return targets[:limit] if limit > 0 else targets


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def history_dedupe_key(row: dict[str, Any]) -> tuple:
    return (
        row.get("sticker_id", ""),
        row.get("history_range", ""),
        row.get("fetched_at") or row.get("tooltip_time_raw", ""),
        row.get("token_cost", ""),
    )


def append_deduped_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> int:
    existing_rows: list[dict[str, Any]] = []
    existing_keys = set()
    existing_fieldnames: list[str] = []

    if path.exists() and path.stat().st_size > 0:
        with open(path, newline="", encoding="utf-8-sig") as file:
            reader = csv.reader(file)
            try:
                existing_fieldnames = next(reader)
            except StopIteration:
                existing_fieldnames = []

            merged_fieldnames = list(existing_fieldnames)
            for field in fieldnames:
                if field not in merged_fieldnames:
                    merged_fieldnames.append(field)

            for raw_row in reader:
                if len(raw_row) == len(fieldnames):
                    row = dict(zip(fieldnames, raw_row))
                else:
                    row = dict(zip(existing_fieldnames, raw_row[:len(existing_fieldnames)]))
                    for index, field in enumerate(merged_fieldnames[len(existing_fieldnames):], start=len(existing_fieldnames)):
                        row[field] = raw_row[index] if index < len(raw_row) else ""
                existing_rows.append(row)
                existing_keys.add(history_dedupe_key(row))
    else:
        merged_fieldnames = list(fieldnames)

    new_rows = [row for row in rows if history_dedupe_key(row) not in existing_keys]

    if existing_fieldnames and existing_fieldnames != merged_fieldnames:
        with open(path, "w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=merged_fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(existing_rows)

    mode = "a" if path.exists() and path.stat().st_size > 0 else "w"
    with open(path, mode, newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=merged_fieldnames, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
        writer.writerows(new_rows)

    return len(new_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Android-friendly CS2Tokens collector without Playwright.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--event", default="Cologne 2026")
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--max-pages", type=int, default=0, help="Debug limit. 0 means all pages.")
    parser.add_argument("--page-delay", type=float, default=0.15)
    parser.add_argument("--metadata-workers", type=int, default=4)
    parser.add_argument("--no-metadata", action="store_true")
    parser.add_argument("--history", dest="history", action="store_true", default=True)
    parser.add_argument("--no-history", dest="history", action="store_false")
    parser.add_argument("--history-range", default="30D")
    parser.add_argument("--history-limit", type=int, default=0)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = file_stamp

    rows = collect_browse_rows(args, timestamp, run_id)
    if not rows:
        raise SystemExit("No rows collected. Check network, event, or variant filters.")

    if not args.no_metadata:
        rows = enrich_rows_with_metadata(rows, args.metadata_workers)

    final_rows = rank_rows(rows)

    snapshot_file = OUT_DIR / f"snapshot_{file_stamp}.csv"
    latest_file = Path("data/latest_snapshot.csv")
    fieldnames = [
        "timestamp", "scrape_run_id", "source_url", "requested_variant", "sticker_id", "name",
        "subject", "event", "category", "variant", "team", "rarity", "rarity_color",
        "price_tokens", "usd_price", "rank_low_to_high", "total_in_group", "price_percentile",
        "image_url", "image_formula", "item_url", "item_slug", "metadata_status", "metadata_error",
        "qdata_url", "market_hash_name", "steam_market_url", "defindex", "paint_index",
        "volatile_item_id", "campaign_id", "campaign_slug", "token_key", "catalog_type",
        "sticker_type", "player_name", "team_name", "rarity_id",
    ]
    write_csv(snapshot_file, final_rows, fieldnames)
    write_csv(latest_file, final_rows, fieldnames)

    print(f"Collected {len(final_rows)} stickers")
    print(f"Saved snapshot: {snapshot_file}")
    print(f"Saved latest:   {latest_file}")

    if args.history:
        history_targets = select_history_targets(final_rows, args.history_limit)
        history_rows: list[dict[str, Any]] = []
        for index, row in enumerate(history_targets, start=1):
            print(f"[{index}/{len(history_targets)}] history: {row['name']}")
            history_rows.extend(collect_detail_history(row, args.history_range))

        if history_rows:
            history_file = HISTORY_DIR / f"history_{file_stamp}.csv"
            latest_history_file = Path("data/latest_history.csv")
            cumulative_file = Path("data/history_points.csv")
            history_fields = list(history_rows[0].keys())
            write_csv(history_file, history_rows, history_fields)
            write_csv(latest_history_file, history_rows, history_fields)
            appended_count = append_deduped_csv(cumulative_file, history_rows, history_fields)
            print(f"Saved history: {history_file}")
            print(f"Saved latest history: {latest_history_file}")
            print(f"Appended cumulative history: {cumulative_file} ({appended_count} new rows)")


if __name__ == "__main__":
    main()
