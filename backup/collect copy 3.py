import argparse
import csv
import hashlib
import json
import re
import time
import urllib.parse
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright


DEFAULT_URL = "https://cs2tokens.com/browse?sort=price-asc"
DEFAULT_VARIANTS = "Paper,Foil,Holo,Gold"
METADATA_WORKERS = 8
METADATA_TIMEOUT = 20
METADATA_RETRIES = 5
METADATA_RETRY_BASE_DELAY = 1.5
BROWSE_WORKERS = 4
USER_AGENT = "Mozilla/5.0 (compatible; cs2-sticker-tracker/1.0)"

OUT_DIR = Path("data/snapshots")
HISTORY_DIR = Path("data/history")

OUT_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def clean_text_multiline(value: str | None) -> str:
    if not value:
        return ""
    lines = [clean_text(line) for line in value.splitlines() if clean_text(line)]
    return "\n".join(lines)


def parse_tokens(value: str | None) -> int | None:
    if not value:
        return None

    value = value.replace(",", "")
    value = re.sub(r"[^\d]", "", value)

    if not value:
        return None

    return int(value)


def parse_usd(value: str | None) -> float | None:
    if not value:
        return None

    value = value.replace("$", "").replace(",", "").strip()

    try:
        return float(value)
    except ValueError:
        return None


def parse_number(value: str | None) -> float | None:
    if not value:
        return None

    value = value.replace(",", "").replace("−", "-")
    value = re.sub(r"[^0-9.\-]", "", value)

    try:
        return float(value)
    except ValueError:
        return None


def make_id(name: str, href: str) -> str:
    raw = f"{name}|{href}".lower()
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def normalize_variant(value: str | None) -> str:
    raw = clean_text(value)
    if not raw:
        return ""
    lower = raw.lower()
    known = {
        "paper": "Paper",
        "foil": "Foil",
        "holo": "Holo",
        "gold": "Gold",
        "glitter": "Glitter",
        "lenticular": "Lenticular",
    }
    return known.get(lower, raw[:1].upper() + raw[1:])


def infer_variant_from_name(name: str) -> str:
    match = re.search(r"\((Paper|Foil|Holo|Gold|Glitter|Lenticular)\)", name, flags=re.I)
    if match:
        return normalize_variant(match.group(1))
    return "Paper"


def parse_variants(value: str) -> list[str]:
    variants = []
    seen = set()
    for part in value.split(","):
        variant = normalize_variant(part)
        key = variant.lower()
        if variant and key not in seen:
            variants.append(variant)
            seen.add(key)
    return variants


def browse_finish_param(variant: str) -> str | None:
    variant = normalize_variant(variant)
    if variant.lower() == "paper":
        return None
    return variant


def build_browse_url(base_url: str, variant: str) -> str:
    if "{variant}" in base_url:
        finish = browse_finish_param(variant) or ""
        url = base_url.replace("{variant}", urllib.parse.quote(finish, safe=""))
        if finish:
            return url
        parsed = urllib.parse.urlsplit(url)
        query = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
        query.pop("finish", None)
        query.setdefault("sort", "price-asc")
        return urllib.parse.urlunsplit((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urllib.parse.urlencode(query),
            parsed.fragment,
        ))

    parsed = urllib.parse.urlsplit(base_url)
    query = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    finish = browse_finish_param(variant)
    if finish:
        query["finish"] = finish
    else:
        query.pop("finish", None)
    query.setdefault("sort", "price-asc")
    return urllib.parse.urlunsplit((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        urllib.parse.urlencode(query),
        parsed.fragment,
    ))


def item_slug_from_url(value: str) -> str:
    path = urllib.parse.urlsplit(value).path.rstrip("/")
    return path.rsplit("/", 1)[-1] if path else ""


def steam_market_listing_url(market_hash_name: str) -> str:
    if not market_hash_name:
        return ""
    encoded = urllib.parse.quote(market_hash_name, safe="")
    return f"https://steamcommunity.com/market/listings/730/{encoded}"


def get_subject(name: str) -> str:
    name = name.replace("Sticker |", "").strip()
    before_event = name.split("|")[0].strip()
    subject = re.sub(r"\s*\((Paper|Holo|Foil|Gold|Glitter|Lenticular)\)\s*", "", before_event, flags=re.I)
    return subject.strip()


def get_event(name: str) -> str:
    parts = [p.strip() for p in name.split("|")]
    return parts[-1] if len(parts) >= 3 else ""


def infer_category(name: str, team: str) -> str:
    subject = get_subject(name).lower()
    team_norm = clean_text(team).lower()

    if subject and team_norm and subject == team_norm:
        return "Team"

    return "Player"


def browse_progress(page) -> dict[str, Any]:
    return page.evaluate(
        """
        () => {
          const root = document.scrollingElement || document.documentElement || document.body;
          const text = document.body?.innerText || "";
          const loaded = text.match(/ALL\\s+([\\d,]+)\\s+ITEMS\\s+LOADED/i);
          return {
            count: document.querySelectorAll("li.px-browse-grid__card").length,
            scrollTop: root.scrollTop || window.scrollY || 0,
            scrollHeight: root.scrollHeight || document.body.scrollHeight || 0,
            clientHeight: root.clientHeight || window.innerHeight || 0,
            loadedTotal: loaded ? Number(loaded[1].replace(/,/g, "")) : null
          };
        }
        """
    )


def auto_scroll(page, max_rounds: int = 160, pause_ms: int = 850) -> None:
    stable_rounds = 0
    last_state: tuple[int, int, int] | None = None

    for _ in range(max_rounds):
        before = browse_progress(page)
        loaded_total = before.get("loadedTotal")
        count = int(before.get("count") or 0)

        if loaded_total and count >= int(loaded_total):
            break

        page.mouse.move(720, 900)
        page.mouse.wheel(0, 5000)
        page.evaluate(
            """
            () => {
              const root = document.scrollingElement || document.documentElement || document.body;
              root.scrollTo({ top: root.scrollHeight, behavior: "instant" });
            }
            """
        )
        page.wait_for_timeout(pause_ms)

        after = browse_progress(page)
        state = (
            int(after.get("count") or 0),
            int(after.get("scrollHeight") or 0),
            int(after.get("scrollTop") or 0),
        )
        bottom_gap = int(after.get("scrollHeight") or 0) - int(after.get("scrollTop") or 0) - int(after.get("clientHeight") or 0)

        if state == last_state and bottom_gap <= 20:
            stable_rounds += 1
        else:
            stable_rounds = 0

        loaded_total = after.get("loadedTotal")
        if loaded_total and int(after.get("count") or 0) >= int(loaded_total):
            break

        if stable_rounds >= 8:
            break

        last_state = state


def collect_browse_cards(page) -> list[dict[str, Any]]:
    return page.evaluate(
        """
        () => {
          const cards = Array.from(document.querySelectorAll("li.px-browse-grid__card"));

          return cards.map((card, index) => {
            const link = card.querySelector("a.px-browse-grid__link");
            const nameEl = card.querySelector(".px-browse-grid__name");
            const teamEl = card.querySelector(".px-browse-grid__team");
            const rarityEl = card.querySelector(".px-browse-grid__rarity");
            const finishEl = card.querySelector(".px-browse-grid__finish");
            const priceTknEl = card.querySelector(".px-browse-grid__price-tkn");
            const priceUsdEl = card.querySelector(".px-browse-grid__price-usd");
            const imgEl = card.querySelector(".px-browse-grid__art img");

            return {
              index: index + 1,
              href: link ? link.href : "",
              aria_label: link ? link.getAttribute("aria-label") : "",
              name: nameEl ? nameEl.textContent.trim() : "",
              title: nameEl ? nameEl.getAttribute("title") : "",
              team: teamEl ? teamEl.textContent.trim() : "",
              rarity: rarityEl ? rarityEl.textContent.trim() : "",
              rarity_color: rarityEl ? rarityEl.style.getPropertyValue("--rarity-color") : "",
              finish: finishEl ? finishEl.textContent.trim() : "",
              price_tokens_raw: priceTknEl ? priceTknEl.textContent.trim() : "",
              price_usd_raw: priceUsdEl ? priceUsdEl.textContent.trim() : "",
              image_url: imgEl ? imgEl.src : ""
            };
          });
        }
        """
    )


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


def retry_delay_seconds(url: str, attempt: int, retry_after: str | None = None) -> float:
    if retry_after:
        try:
            return min(30.0, max(0.5, float(retry_after)))
        except ValueError:
            pass
    jitter = (int(hashlib.md5(url.encode("utf-8")).hexdigest()[:4], 16) % 1000) / 1000
    return min(30.0, METADATA_RETRY_BASE_DELAY * (2 ** attempt) + jitter)


def fetch_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    retry_statuses = {429, 500, 502, 503, 504}

    for attempt in range(METADATA_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=METADATA_TIMEOUT) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code not in retry_statuses or attempt >= METADATA_RETRIES:
                raise
            delay = retry_delay_seconds(url, attempt, exc.headers.get("Retry-After"))
            print(f"Metadata HTTP {exc.code}; retrying in {delay:.1f}s: {url}")
            time.sleep(delay)
        except urllib.error.URLError:
            if attempt >= METADATA_RETRIES:
                raise
            delay = retry_delay_seconds(url, attempt)
            print(f"Metadata network error; retrying in {delay:.1f}s: {url}")
            time.sleep(delay)

    raise RuntimeError(f"Failed to fetch JSON after retries: {url}")


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
        objs = payload.get("_objs", [])
        decoded = decode_qwik_data(payload, objs)
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


def enrich_rows_with_metadata(rows: list[dict[str, Any]], workers: int = METADATA_WORKERS) -> list[dict[str, Any]]:
    urls = sorted({row["item_url"] for row in rows if row.get("item_url")})
    metadata_by_url: dict[str, dict[str, Any]] = {}

    if not urls:
        return rows

    print(f"Fetching item metadata: {len(urls)} items")

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(fetch_item_metadata, url): url for url in urls}
        for future in as_completed(futures):
            url = futures[future]
            metadata_by_url[url] = future.result()

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


def parse_svg_price_points(page) -> tuple[list[dict[str, float]], dict[str, float] | None]:
    data = page.evaluate(
        """
        () => {
          const svg = document.querySelector(".px-detail-chart svg");
          if (!svg) return null;

          const viewBoxRaw = svg.getAttribute("viewBox") || "0 0 800 220";
          const parts = viewBoxRaw.split(/\\s+/).map(Number);

          const viewBox = {
            x: Number.isFinite(parts[0]) ? parts[0] : 0,
            y: Number.isFinite(parts[1]) ? parts[1] : 0,
            width: Number.isFinite(parts[2]) ? parts[2] : 800,
            height: Number.isFinite(parts[3]) ? parts[3] : 220
          };

          const paths = Array.from(svg.querySelectorAll("path"));

          const candidates = paths
            .map((p, idx) => ({
              idx,
              d: p.getAttribute("d") || "",
              fill: p.getAttribute("fill") || "",
              stroke: p.getAttribute("stroke") || "",
              dash: p.getAttribute("stroke-dasharray") || "",
              width: p.getAttribute("stroke-width") || "",
            }))
            .filter(p =>
              p.fill === "none" &&
              !p.dash &&
              p.d.includes("L") &&
              !p.stroke.includes("--chart-secondary")
            );

          if (!candidates.length) {
            return { points: [], viewBox };
          }

          const pricePath = candidates.sort((a, b) => {
            const ac = (a.d.match(/[ML]/g) || []).length;
            const bc = (b.d.match(/[ML]/g) || []).length;
            return bc - ac;
          })[0];

          const matches = [...pricePath.d.matchAll(/[ML]\\s*([0-9.]+),([0-9.]+)/g)];

          const points = matches.map((m, i) => ({
            point_index: i + 1,
            x: Number(m[1]),
            y: Number(m[2])
          }));

          return { points, viewBox };
        }
        """
    )

    if not data:
        return [], None

    return data["points"], data["viewBox"]


def parse_tooltip_text(text: str) -> dict[str, Any]:
    lines = [clean_text(x) for x in text.splitlines() if clean_text(x)]

    token_cost = None
    usd_price = None
    popularity = None
    tooltip_time = None

    for line in lines:
        if token_cost is None and re.fullmatch(r"\d[\d,]*", line):
            token_cost = parse_tokens(line)
            continue

        if usd_price is None and line.startswith("$"):
            usd_price = parse_usd(line)
            continue

        if "Popularity" in line:
            popularity = parse_number(line)
            continue

        if re.search(r"\d{1,2}/\d{1,2}/\d{4}", line):
            tooltip_time = line
            continue

    return {
        "token_cost": token_cost,
        "usd_price": usd_price,
        "popularity": popularity,
        "tooltip_time_raw": tooltip_time,
    }


def read_tooltip_text(page, mouse_x: float | None = None, mouse_y: float | None = None) -> str:
    tooltip_text = page.evaluate(
        """
        ({ mouseX, mouseY }) => {
          const all = Array.from(document.querySelectorAll("body *"));
          const candidates = [];

          for (const el of all) {
            const text = (el.innerText || el.textContent || "").trim();
            if (!text) continue;

            const hasUsd = /\\$\\s*\\d/.test(text);
            const hasDate = /\\d{1,2}\\/\\d{1,2}\\/\\d{4}/.test(text);
            const hasToken = /(^|\\n)\\s*\\d[\\d,]*\\s*(\\n|$)/.test(text);

            if (!hasUsd || !hasDate || !hasToken) continue;

            const dateCount = (text.match(/\\d{1,2}\\/\\d{1,2}\\/\\d{4}/g) || []).length;
            const lineCount = text.split(/\\n+/).filter(Boolean).length;

            if (dateCount !== 1) continue;
            if (lineCount > 6) continue;

            if (text.includes("Cologne 2026")) continue;
            if (text.includes("PRICE HISTORY")) continue;
            if (text.includes("TOKEN COST")) continue;
            if (text.includes("Sticker |")) continue;
            if (text.includes("Apply Sticker")) continue;

            const rect = el.getBoundingClientRect();

            const visible =
              rect.width > 0 &&
              rect.height > 0 &&
              rect.bottom >= 0 &&
              rect.right >= 0 &&
              rect.top <= window.innerHeight &&
              rect.left <= window.innerWidth;

            if (!visible) continue;

            let distance = 0;

            if (typeof mouseX === "number" && typeof mouseY === "number") {
              const cx = rect.left + rect.width / 2;
              const cy = rect.top + rect.height / 2;
              distance = Math.hypot(cx - mouseX, cy - mouseY);

              if (distance > 500) continue;
            }

            candidates.push({
              text,
              area: rect.width * rect.height,
              distance
            });
          }

          candidates.sort((a, b) => {
            if (a.distance !== b.distance) return a.distance - b.distance;
            return a.area - b.area;
          });

          return candidates.length ? candidates[0].text : "";
        }
        """,
        {"mouseX": mouse_x, "mouseY": mouse_y},
    )

    return clean_text_multiline(tooltip_text)


def map_svg_to_screen(
    svg_x: float,
    svg_y: float,
    box: dict[str, float],
    view_box: dict[str, float],
) -> tuple[float, float]:
    view_x = view_box.get("x", 0)
    view_y = view_box.get("y", 0)
    view_w = view_box.get("width", 800)
    view_h = view_box.get("height", 220)

    screen_x = box["x"] + ((svg_x - view_x) / view_w) * box["width"]
    screen_y = box["y"] + ((svg_y - view_y) / view_h) * box["height"]

    return screen_x, screen_y


def hover_point_and_get_tooltip(page, screen_x: float, screen_y: float) -> dict[str, Any] | None:
    offsets = [
        (0, 0),
        (0, -4),
        (0, 4),
        (-4, 0),
        (4, 0),
        (-6, -3),
        (6, -3),
        (-6, 3),
        (6, 3),
        (0, -8),
        (0, 8),
        (-10, 0),
        (10, 0),
    ]

    page.mouse.move(10, 10)
    page.wait_for_timeout(120)

    for dx, dy in offsets:
        mx = screen_x + dx
        my = screen_y + dy

        page.mouse.move(mx, my)
        page.wait_for_timeout(300)

        raw = read_tooltip_text(page, mx, my)
        parsed = parse_tooltip_text(raw)

        if (
            parsed["token_cost"] is not None
            and parsed["usd_price"] is not None
            and parsed["tooltip_time_raw"]
            and "Cologne 2026" not in raw
            and "PRICE HISTORY" not in raw
            and "Sticker |" not in raw
        ):
            parsed["tooltip_raw"] = raw
            return parsed

    return None


def extract_chart_history(page, range_label: str) -> list[dict[str, Any]]:
    try:
        btn = page.locator(".px-detail-range-btn", has_text=range_label).first
        if btn.count() > 0:
            btn.click()
            page.wait_for_timeout(1200)
    except Exception:
        pass

    points, view_box = parse_svg_price_points(page)

    if not points or not view_box:
        return []

    chart = page.locator(".px-detail-chart svg").first
    box = chart.bounding_box()

    if not box:
        return []

    rows = []

    for point in points:
        screen_x, screen_y = map_svg_to_screen(point["x"], point["y"], box, view_box)
        parsed = hover_point_and_get_tooltip(page, screen_x, screen_y)

        if not parsed:
            print(f"Skipped bad tooltip point {point['point_index']}")
            continue

        rows.append({
            "history_range": range_label,
            "point_index": point["point_index"],
            "x": point["x"],
            "y": point["y"],
            "token_cost": parsed["token_cost"],
            "usd_price": parsed["usd_price"],
            "popularity": parsed["popularity"],
            "tooltip_time_raw": parsed["tooltip_time_raw"],
            "tooltip_raw": parsed["tooltip_raw"],
        })

    return rows


def collect_detail_history(row: dict[str, Any], range_label: str) -> list[dict[str, Any]]:
    out = []
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


def collect_rows_from_page(page, source_url: str, requested_variant: str, args, timestamp: str, run_id: str) -> list[dict[str, Any]]:
    print(f"Collecting {requested_variant}: {source_url}")
    page.goto(source_url, wait_until="networkidle", timeout=90000)
    try:
        page.wait_for_selector(".px-browse-grid__card", timeout=60000)
    except Exception:
        print(f"No browse cards found for {requested_variant}")
        return []
    auto_scroll(page)

    loaded_cards = page.locator(".px-browse-grid__card").count()
    raw_cards = collect_browse_cards(page)
    rows = []

    for card in raw_cards:
        name = clean_text(card.get("title") or card.get("name"))

        if not name:
            continue

        event = get_event(name)
        finish = normalize_variant(card.get("finish"))
        if not finish:
            finish = infer_variant_from_name(name)

        if args.event.lower() not in event.lower():
            continue

        if finish.lower() != requested_variant.lower():
            continue

        href = clean_text(card.get("href"))
        team = clean_text(card.get("team"))
        price_tokens = parse_tokens(card.get("price_tokens_raw"))
        usd_price = parse_usd(card.get("price_usd_raw"))
        image_url = clean_text(card.get("image_url"))

        if price_tokens is None:
            continue

        category = infer_category(name, team)
        sticker_id = make_id(name, href)

        rows.append({
            "timestamp": timestamp,
            "scrape_run_id": run_id,
            "source_url": source_url,
            "requested_variant": requested_variant,
            "sticker_id": sticker_id,
            "name": name,
            "subject": get_subject(name),
            "event": event,
            "category": category,
            "variant": finish,
            "team": team,
            "rarity": clean_text(card.get("rarity")),
            "rarity_color": clean_text(card.get("rarity_color")),
            "price_tokens": price_tokens,
            "usd_price": usd_price,
            "image_url": image_url,
            "image_formula": f'=IMAGE("{image_url}")' if image_url else "",
            "item_url": href,
        })

    print(f"Collected {len(rows)} {requested_variant} rows from {loaded_cards} loaded cards")
    return rows


def collect_variant_in_browser(variant: str, args, timestamp: str, run_id: str) -> list[dict[str, Any]]:
    source_url = build_browse_url(args.url, variant)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=args.headless)
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1500})
            return collect_rows_from_page(page, source_url, variant, args, timestamp, run_id)
        finally:
            browser.close()


def collect_variants_parallel(wanted_variants: list[str], args, timestamp: str, run_id: str) -> list[dict[str, Any]]:
    workers = max(1, min(int(args.browse_workers or 1), len(wanted_variants)))
    if workers <= 1:
        rows: list[dict[str, Any]] = []
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=args.headless)
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 1500})
                for variant in wanted_variants:
                    source_url = build_browse_url(args.url, variant)
                    rows.extend(collect_rows_from_page(page, source_url, variant, args, timestamp, run_id))
            finally:
                browser.close()
        return rows

    print(f"Collecting {len(wanted_variants)} variants with {workers} parallel browser workers")
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(collect_variant_in_browser, variant, args, timestamp, run_id): variant
            for variant in wanted_variants
        }
        for future in as_completed(futures):
            variant = futures[future]
            try:
                rows.extend(future.result())
            except Exception as exc:
                print(f"Failed collecting {variant}: {exc}")
                raise
    return rows


def rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_sorted = sorted(rows, key=lambda row: (row["category"], row["variant"], row["price_tokens"]))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for row in rows_sorted:
        key = (row["category"], row["variant"])
        grouped.setdefault(key, []).append(row)

    final_rows = []

    for group in grouped.values():
        total = len(group)

        for rank, row in enumerate(group, start=1):
            row["rank_low_to_high"] = rank
            row["total_in_group"] = total
            row["price_percentile"] = round((rank - 1) / (total - 1), 4) if total > 1 else 0
            final_rows.append(row)

    return final_rows


def select_history_targets(rows: list[dict[str, Any]], limit: int, per_variant_limit: int) -> list[dict[str, Any]]:
    if per_variant_limit > 0:
        targets = []
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault((row["category"], row["variant"]), []).append(row)
        for group in grouped.values():
            targets.extend(sorted(group, key=lambda row: row["price_tokens"])[:per_variant_limit])
        return sorted(targets, key=lambda row: (row["variant"], row["category"], row["price_tokens"]))

    targets = sorted(rows, key=lambda row: row["price_tokens"])
    if limit > 0:
        targets = targets[:limit]
    return targets


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--event", default="Cologne 2026")
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--headless", dest="headless", action="store_true", default=True)
    parser.add_argument("--headed", dest="headless", action="store_false")
    parser.add_argument("--browse-workers", type=int, default=BROWSE_WORKERS)
    parser.add_argument("--no-metadata", action="store_true")
    parser.add_argument("--metadata-workers", type=int, default=METADATA_WORKERS)

    parser.add_argument("--history", dest="history", action="store_true", default=True)
    parser.add_argument("--no-history", dest="history", action="store_false")
    parser.add_argument("--history-range", default="30D")
    parser.add_argument("--history-limit", type=int, default=0)
    parser.add_argument("--history-limit-per-variant", type=int, default=0)
    parser.add_argument("--history-delay", type=float, default=0.0)

    args = parser.parse_args()

    wanted_variants = parse_variants(args.variants)

    if not wanted_variants:
        raise SystemExit("No variants requested. Use --variants Paper,Foil,Holo for example.")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = file_stamp

    rows = collect_variants_parallel(wanted_variants, args, timestamp, run_id)

    if not rows:
        raise SystemExit("No rows collected. Check URL, event, or variant filters.")

    if not args.no_metadata:
        rows = enrich_rows_with_metadata(rows, args.metadata_workers)

    final_rows = rank_rows(rows)

    snapshot_file = OUT_DIR / f"snapshot_{file_stamp}.csv"
    latest_file = Path("data/latest_snapshot.csv")

    fieldnames = [
        "timestamp",
        "scrape_run_id",
        "source_url",
        "requested_variant",
        "sticker_id",
        "name",
        "subject",
        "event",
        "category",
        "variant",
        "team",
        "rarity",
        "rarity_color",
        "price_tokens",
        "usd_price",
        "rank_low_to_high",
        "total_in_group",
        "price_percentile",
        "image_url",
        "image_formula",
        "item_url",
        "item_slug",
        "metadata_status",
        "metadata_error",
        "qdata_url",
        "market_hash_name",
        "steam_market_url",
        "defindex",
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
    ]

    write_csv(snapshot_file, final_rows, fieldnames)
    write_csv(latest_file, final_rows, fieldnames)

    print(f"Collected {len(final_rows)} stickers")
    print(f"Saved snapshot: {snapshot_file}")
    print(f"Saved latest:   {latest_file}")

    if args.history:
        history_targets = select_history_targets(
            final_rows,
            args.history_limit,
            args.history_limit_per_variant,
        )

        history_rows = []

        for index, row in enumerate(history_targets, start=1):
            print(f"[{index}/{len(history_targets)}] history: {row['name']}")
            history_rows.extend(collect_detail_history(row, args.history_range))
            if args.history_delay > 0:
                time.sleep(args.history_delay)

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
