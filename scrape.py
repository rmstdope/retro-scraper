#!/usr/bin/env python3
"""
scrape.py — Download ROMs from romsgames.net.

Usage:
    pip install -r requirements.txt
    python scrape.py nes   # download NES ROMs into roms/nes/
    python scrape.py gb    # download Game Boy ROMs into roms/gb/
    python scrape.py cgb   # download Game Boy Color ROMs into roms/cgb/
    python scrape.py gba   # download Game Boy Advance ROMs into roms/gba/
"""

import argparse
import io
import re
import time
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://www.romsgames.net"
REQUEST_DELAY = 0.75  # seconds between HTTP requests
CACHE_DIR = Path(".cache")


@dataclass
class Platform:
    key: str           # CLI argument name (nes / gb)
    listing_path: str  # e.g. /roms/nintendo/
    slug_prefix: str   # e.g. nintendo-rom-
    total_pages: int
    roms_dir: Path
    rom_ext: str       # file extension to save (.nes / .gb)
    magic: bytes       # expected ROM magic bytes (for validation)


PLATFORMS: dict[str, Platform] = {
    "nes": Platform(
        key="nes",
        listing_path="/roms/nintendo/",
        slug_prefix="nintendo-rom-",
        total_pages=72,
        roms_dir=Path("roms/nes"),
        rom_ext=".nes",
        magic=b"NES\x1a",
    ),
    "gb": Platform(
        key="gb",
        listing_path="/roms/gameboy/",
        slug_prefix="gameboy-rom-",
        total_pages=33,
        roms_dir=Path("roms/gb"),
        rom_ext=".gb",
        magic=b"",  # GB ROMs have no universal fixed magic; accept any non-HTML content
    ),
    "cgb": Platform(
        key="cgb",
        listing_path="/roms/gameboy-color/",
        slug_prefix="gameboy-color-rom-",
        total_pages=24,
        roms_dir=Path("roms/cgb"),
        rom_ext=".gbc",
        magic=b"",  # GBC ROMs have no universal fixed magic; accept any non-HTML content
    ),
    "gba": Platform(
        key="gba",
        listing_path="/roms/gameboy-advance/",
        slug_prefix="gameboy-advance-rom-",
        total_pages=64,
        roms_dir=Path("roms/gba"),
        rom_ext=".gba",
        magic=b"",  # GBA ROMs have no universal fixed magic; accept any non-HTML content
    ),
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# Matches /<prefix><slug>/ hrefs on listing pages (compiled per-platform at runtime)
_slug_re_cache: dict[str, re.Pattern] = {}

# ---------------------------------------------------------------------------
# Phase 1: collect all game slugs from listing pages
# ---------------------------------------------------------------------------


def get_all_game_slugs(platform: Platform) -> list[str]:
    """Scrape all listing pages and return a deduplicated list of ROM slugs.

    Each page's HTML is cached under .cache/{key}-listing-page-{n}.html so the
    listing phase can be resumed if interrupted.
    """
    slug_re = _slug_re_cache.setdefault(
        platform.key,
        re.compile(rf"^/{re.escape(platform.slug_prefix)}(.+)/$"),
    )
    CACHE_DIR.mkdir(exist_ok=True)
    listing_url = f"{BASE_URL}{platform.listing_path}"
    slugs: list[str] = []
    seen: set[str] = set()

    for page in tqdm(range(1, platform.total_pages + 1), desc="Scraping listing pages", unit="page"):
        cache_file = CACHE_DIR / f"{platform.key}-listing-page-{page}.html"

        if cache_file.exists():
            html = cache_file.read_text(encoding="utf-8")
        else:
            url = f"{listing_url}?page={page}&sort=popularity"
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            html = response.text
            cache_file.write_text(html, encoding="utf-8")
            time.sleep(REQUEST_DELAY)

        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            m = slug_re.match(a["href"])
            if m:
                slug = m.group(1)
                if slug not in seen:
                    seen.add(slug)
                    slugs.append(slug)

    return slugs


# ---------------------------------------------------------------------------
# Phase 2: get the media ID for a game (cached per slug)
# ---------------------------------------------------------------------------


def get_media_id(slug: str, platform: Platform) -> str:
    """Fetch the game detail page and return the numeric media ID.

    The detail page HTML is cached under .cache/{key}-detail-{slug}.html so that
    re-runs don't re-fetch pages for already-known slugs.
    """
    cache_file = CACHE_DIR / f"{platform.key}-detail-{slug}.html"

    if cache_file.exists():
        html = cache_file.read_text(encoding="utf-8")
    else:
        url = f"{BASE_URL}/{platform.slug_prefix}{slug}/"
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        html = response.text
        cache_file.write_text(html, encoding="utf-8")
        time.sleep(REQUEST_DELAY)

    soup = BeautifulSoup(html, "html.parser")
    btn = soup.find(attrs={"data-media-id": True})
    if not btn:
        raise ValueError(f"No data-media-id button found for slug '{slug}'")
    return str(btn["data-media-id"])


# ---------------------------------------------------------------------------
# Phase 3: get a signed, time-limited download URL from the API
# ---------------------------------------------------------------------------


def get_download_url(slug: str, media_id: str, platform: Platform) -> tuple[str, str]:
    """POST to the download API and return (downloadUrl, downloadName)."""
    page_url = f"{BASE_URL}/{platform.slug_prefix}{slug}/"
    api_url = f"{page_url}?download"
    response = requests.post(
        api_url,
        headers={**HEADERS, "Accept": "application/json", "Referer": page_url},
        data={"mediaId": media_id, "g-recaptcha-response": ""},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    if not data.get("downloadUrl"):
        raise ValueError(f"No downloadUrl in API response for slug '{slug}': {data}")
    return data["downloadUrl"], data.get("downloadName", "rom.zip")


# ---------------------------------------------------------------------------
# Phase 4: download and extract a single ROM
# ---------------------------------------------------------------------------


def download_rom(slug: str, download_url: str, download_name: str, platform: Platform) -> None:
    """Download the ROM from the signed URL and save it into the platform's roms_dir.

    If the file is a zip, only the first entry matching the platform extension is
    extracted. If the file is not a zip, it is saved directly after validation.
    """
    dest = platform.roms_dir / f"{slug}{platform.rom_ext}"
    if dest.exists():
        return  # already downloaded — resume skip

    # The signed download URL requires the page Referer to be set.
    referer = f"{BASE_URL}/{platform.slug_prefix}{slug}/"
    dl_headers = {**HEADERS, "Referer": referer}
    response = requests.get(download_url, headers=dl_headers, timeout=120, stream=True)
    response.raise_for_status()

    data = b"".join(response.iter_content(chunk_size=65536))

    if data.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                entries = [name for name in zf.namelist() if name.lower().endswith(platform.rom_ext)]
                if entries:
                    dest.write_bytes(zf.read(entries[0]))
                # No matching entry inside the zip — silently skip
        except (zipfile.BadZipFile, zlib.error) as exc:
            print(f"\nSKIP {slug}: Corrupted zip — {exc}")
    elif platform.magic and not data.startswith(platform.magic):
        raise ValueError(
            f"Unexpected content for '{slug}': magic={data[:4]!r}"
        )
    else:
        dest.write_bytes(data)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ROMs from romsgames.net")
    parser.add_argument(
        "platform",
        choices=list(PLATFORMS),
        help="Platform to download: 'nes' (roms/nes/), 'gb' (roms/gb/), 'cgb' (roms/cgb/), or 'gba' (roms/gba/)",
    )
    args = parser.parse_args()
    platform = PLATFORMS[args.platform]

    platform.roms_dir.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    print(f"Platform : {platform.key.upper()}")
    print(f"Output   : {platform.roms_dir}/")
    print()

    print("Phase 1: Collecting game slugs from all listing pages…")
    slugs = get_all_game_slugs(platform)
    print(f"Found {len(slugs)} games.\n")

    print("Phase 2: Downloading ROMs…")
    for slug in tqdm(slugs, desc="Downloading ROMs", unit="rom"):
        dest = platform.roms_dir / f"{slug}{platform.rom_ext}"
        if dest.exists():
            if not platform.magic or dest.read_bytes()[:4] == platform.magic:
                continue  # valid ROM — skip
            dest.unlink()  # corrupt/HTML file from a previous run — re-download

        try:
            media_id = get_media_id(slug, platform)
            download_url, download_name = get_download_url(slug, media_id, platform)
            time.sleep(5)  # signed URL is not active until ~5s after being issued
            download_rom(slug, download_url, download_name, platform)
            time.sleep(REQUEST_DELAY)  # polite delay after download
        except (requests.exceptions.RequestException, zipfile.BadZipFile, ValueError) as exc:
            tqdm.write(f"SKIP {slug}: {exc}")


if __name__ == "__main__":
    main()
