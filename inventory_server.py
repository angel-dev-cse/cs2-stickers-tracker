from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile


ROOT = Path(__file__).resolve().parent
INVENTORY_DIR = ROOT / "Inventory"
INVENTORY_CSV = INVENTORY_DIR / "sticker_inventory.csv"
VISUALIZED_DIR = ROOT / "visualized"
FAVORITES_JSON = VISUALIZED_DIR / "favorites.json"
INVENTORY_FIELDS = [
    "inventory_id",
    "sticker_id",
    "sticker",
    "variant",
    "category",
    "steam_account",
    "bought_tokens",
    "bought_usd",
    "acquired_at",
    "notes",
    "created_at",
    "updated_at",
]


def ensure_inventory_file() -> None:
    INVENTORY_DIR.mkdir(parents=True, exist_ok=True)
    if not INVENTORY_CSV.exists():
        with INVENTORY_CSV.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=INVENTORY_FIELDS)
            writer.writeheader()


def read_inventory() -> list[dict[str, str]]:
    ensure_inventory_file()
    with INVENTORY_CSV.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        rows = []
        for row in reader:
            clean = {field: str(row.get(field, "") or "") for field in INVENTORY_FIELDS}
            if clean["inventory_id"] or clean["sticker_id"] or clean["sticker"]:
                rows.append(clean)
        return rows


def write_inventory(rows: list[dict[str, object]]) -> None:
    ensure_inventory_file()
    clean_rows: list[dict[str, str]] = []
    for row in rows:
        clean = {field: str(row.get(field, "") or "") for field in INVENTORY_FIELDS}
        if clean["inventory_id"] and (clean["sticker_id"] or clean["sticker"]):
            clean_rows.append(clean)

    with NamedTemporaryFile("w", delete=False, newline="", encoding="utf-8-sig", dir=INVENTORY_DIR) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=INVENTORY_FIELDS)
        writer.writeheader()
        writer.writerows(clean_rows)
        tmp_path = Path(tmp.name)
    tmp_path.replace(INVENTORY_CSV)


def read_favorites() -> list[str]:
    if not FAVORITES_JSON.exists():
        return []
    try:
        payload = json.loads(FAVORITES_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    values = payload.get("favorites", payload) if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if str(value)]


def write_favorites(values: list[object]) -> None:
    VISUALIZED_DIR.mkdir(parents=True, exist_ok=True)
    favorites = sorted({str(value).strip() for value in values if str(value).strip()})
    FAVORITES_JSON.write_text(json.dumps({"favorites": favorites}, indent=2), encoding="utf-8")


def normalized_variant(item: dict[str, object]) -> str:
    raw = " ".join(
        str(item.get(key, "") or "")
        for key in ("variant", "market_hash_name", "sticker")
    ).lower()
    if "gold" in raw:
        return "Gold"
    if "holo" in raw:
        return "Holo"
    if "foil" in raw:
        return "Foil"
    return "Paper"


def refresh_csgoskins_prices(items: list[object]) -> list[dict[str, object]]:
    from visualize import (
        CSGOSKINS_WORKERS,
        csgoskins_url,
        fetch_csgoskins_price,
        merge_csgoskins_cache_entry,
        read_csgoskins_cache,
        safe_float,
        write_csgoskins_cache,
    )

    cache = read_csgoskins_cache()
    refreshed: list[dict[str, object]] = []
    favorites = set(read_favorites())
    eligible: list[tuple[int, dict[str, object], str, dict[str, object]]] = []

    for raw_item in items:
        if not isinstance(raw_item, dict):
            continue
        item = {str(key): value for key, value in raw_item.items()}
        sticker = str(item.get("sticker", "") or "")
        market_hash_name = str(item.get("market_hash_name", "") or "")
        sticker_id = str(item.get("sticker_id", "") or "")
        row_id = str(item.get("id", "") or sticker_id or market_hash_name or sticker)
        variant = normalized_variant(item)
        url = str(item.get("csgoskins_url", "") or csgoskins_url(market_hash_name, sticker))

        base = {
            "id": row_id,
            "sticker_id": sticker_id,
            "sticker": sticker,
            "variant": variant,
            "market_hash_name": market_hash_name,
            "csgoskins_url": url,
        }
        if variant not in {"Holo", "Foil"}:
            refreshed.append({**base, "price": None, "status": "not_requested"})
            continue

        previous = cache.get(url) if isinstance(cache.get(url), dict) else {}
        is_favorite = bool(favorites.intersection({row_id, sticker_id, market_hash_name, sticker}))
        group = 0 if is_favorite else 1 if variant == "Holo" else 2
        eligible.append((group, base, url, previous))

    def fetch_one(base: dict[str, object], url: str, previous: dict[str, object]) -> tuple[str, dict[str, object], dict[str, object]]:
        result = fetch_csgoskins_price(url, str(base.get("market_hash_name", "")), str(base.get("sticker", "")))
        cache_entry = merge_csgoskins_cache_entry(previous, result)
        status = str(result.get("status", "error"))
        fetched_at = result.get("fetched_at")

        if status != "ok" and previous.get("status") == "ok":
            row_status = "ok_cached_after_error"
        else:
            row_status = str(cache_entry.get("status", status))
        markets = cache_entry.get("markets") if isinstance(cache_entry.get("markets"), dict) else {}
        market_sources = cache_entry.get("market_sources") if isinstance(cache_entry.get("market_sources"), dict) else {}
        price_candidates = [safe_float(cache_entry.get("price"), None)]
        price_candidates.extend(safe_float(price, None) for price in markets.values())
        price_candidates = [price for price in price_candidates if price is not None and price > 0]
        row = {
            **base,
            "price": min(price_candidates) if price_candidates else None,
            "status": row_status,
            "fetched_at": cache_entry.get("fetched_at", fetched_at),
            "markets": markets,
            "market_sources": market_sources,
            "fallback_status": cache_entry.get("fallback_status"),
            "fallback_url": cache_entry.get("fallback_url"),
            "last_error": cache_entry.get("last_error"),
            "csfloat_low_usd": safe_float(markets.get("CSFloat"), None),
            "uuskins_low_usd": safe_float(markets.get("UUSkins"), None),
        }
        return url, cache_entry, row

    if eligible:
        for group in (0, 1, 2):
            group_items = [(base, url, previous) for item_group, base, url, previous in eligible if item_group == group]
            if not group_items:
                continue
            workers = max(1, min(int(CSGOSKINS_WORKERS), len(group_items)))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(fetch_one, base, url, previous) for base, url, previous in group_items]
                for future in as_completed(futures):
                    url, cache_entry, row = future.result()
                    cache[url] = cache_entry
                    refreshed.append(row)

    write_csgoskins_cache(cache)
    return refreshed


class InventoryHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, format: str, *args) -> None:
        return

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/api/inventory":
            try:
                self.send_json(200, {"items": read_inventory(), "path": str(INVENTORY_CSV.relative_to(ROOT))})
            except Exception as exc:
                self.send_json(500, {"error": str(exc)})
            return
        if path == "/api/favorites":
            try:
                self.send_json(200, {"favorites": read_favorites(), "path": str(FAVORITES_JSON.relative_to(ROOT))})
            except Exception as exc:
                self.send_json(500, {"error": str(exc)})
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path not in {"/api/inventory", "/api/favorites", "/api/csgoskins-price"}:
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8")) if raw else {}
            if path == "/api/csgoskins-price":
                rows = payload.get("items", payload if isinstance(payload, list) else [])
                if not isinstance(rows, list):
                    raise ValueError("Expected JSON payload with an items array.")
                self.send_json(200, {"items": refresh_csgoskins_prices(rows)})
                return
            if path == "/api/favorites":
                values = payload.get("favorites", payload if isinstance(payload, list) else [])
                if not isinstance(values, list):
                    raise ValueError("Expected JSON payload with a favorites array.")
                write_favorites(values)
                self.send_json(200, {"favorites": read_favorites(), "path": str(FAVORITES_JSON.relative_to(ROOT))})
                return
            rows = payload.get("items", payload if isinstance(payload, list) else [])
            if not isinstance(rows, list):
                raise ValueError("Expected JSON payload with an items array.")
            write_inventory(rows)
            self.send_json(200, {"items": read_inventory(), "path": str(INVENTORY_CSV.relative_to(ROOT))})
        except Exception as exc:
            self.send_json(400, {"error": str(exc)})


def main() -> None:
    ensure_inventory_file()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    url = f"http://127.0.0.1:{port}/visualized/sticker_dashboard.html"
    print(f"Serving dashboard at:\n  {url}")
    print(f"Inventory CSV:\n  {INVENTORY_CSV}")
    print("Press Ctrl+C to stop the server.")
    ThreadingHTTPServer(("127.0.0.1", port), InventoryHandler).serve_forever()


if __name__ == "__main__":
    main()
