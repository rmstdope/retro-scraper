#!/usr/bin/env python3
"""
sort.py — Sort ROMs by hardware metadata.

NES: reads iNES header, looks up CRC in rom_db.csv, patches header, copies to
     roms/nes/mappers/<mapper>/<submapper>/<name>.nes

GB:  reads GB cartridge header, copies to
     roms/gb/mappers/<mbc>/<cgb-mode>/<name>.gb
     where cgb-mode is one of: dmg-only, cgb-compat, cgb-only

CGB: reads GBC cartridge header, copies to
     roms/cgb/mappers/<mbc>/<cgb-mode>/<name>.gbc

GBA: reads GBA cartridge header, copies to
     roms/gba/makers/<maker-code>/<name>.gba

Usage:
    python sort.py nes
    python sort.py gb
    python sort.py cgb
    python sort.py gba
"""

import argparse
import zlib
from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROM_DB_PATH = Path("rom_db.csv")

INES_MAGIC = b"NES\x1a"
HEADER_SIZE = 16
TRAINER_SIZE = 512

# Game Boy cartridge header offsets
_GB_MBC_OFFSET = 0x0147
_GB_CGB_OFFSET = 0x0143
_GB_HEADER_MIN = 0x0150  # minimum file size to contain a full GB header

# Game Boy Advance cartridge header offsets
_GBA_MAKER_OFFSET = 0xB0  # 2-byte ASCII maker/publisher code
_GBA_HEADER_MIN = 0xC0    # minimum file size to contain a full GBA header

# MBC type byte → human-readable folder name
_GB_MBC_NAMES: dict[int, str] = {
    0x00: "rom-only",
    0x01: "mbc1", 0x02: "mbc1-ram", 0x03: "mbc1-ram-battery",
    0x05: "mbc2", 0x06: "mbc2-battery",
    0x08: "rom-ram", 0x09: "rom-ram-battery",
    0x0B: "mmm01", 0x0C: "mmm01-ram", 0x0D: "mmm01-ram-battery",
    0x0F: "mbc3-rtc-battery", 0x10: "mbc3-ram-rtc-battery",
    0x11: "mbc3", 0x12: "mbc3-ram", 0x13: "mbc3-ram-battery",
    0x19: "mbc5", 0x1A: "mbc5-ram", 0x1B: "mbc5-ram-battery",
    0x1C: "mbc5-rumble", 0x1D: "mbc5-ram-rumble", 0x1E: "mbc5-ram-battery-rumble",
    0x20: "mbc6",
    0x22: "mbc7-sensor-rumble-ram-battery",
    0xFC: "pocket-camera", 0xFD: "bandai-tama5",
    0xFE: "huc3", 0xFF: "huc1-ram-battery",
}

_CSV_FIELDS = [
    "rom_id", "name", "country", "crc", "hardware", "rom_class",
    "mapper", "submapper", "nametable_layout", "prg_rom_size",
    "prg_rom_crc", "prg_nvram_size", "prg_ram_size", "chr_rom_size",
    "chr_rom_crc", "chr_nvram_size", "chr_ram_size", "battery",
    "vs_hardware_type", "vs_ppu_type", "expansion_type",
]

# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def load_rom_db(csv_path: Path) -> dict[str, dict]:
    """Load rom_db.csv into a dict keyed by uppercase hex CRC."""
    db: dict[str, dict] = {}
    with csv_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            parts += [""] * (len(_CSV_FIELDS) - len(parts))
            entry = dict(zip(_CSV_FIELDS, parts))
            crc = entry["crc"].strip().upper()
            if crc:
                db[crc] = entry
    return db


# ---------------------------------------------------------------------------
# iNES header parsing
# ---------------------------------------------------------------------------


def parse_header(header: bytes) -> dict:
    """Return key fields from a 16-byte iNES header."""
    flags6 = header[6]
    flags7 = header[7]
    flags8 = header[8]

    nes2 = (flags7 & 0x0C) == 0x08

    if nes2:
        mapper = ((flags8 & 0x0F) << 8) | (flags7 & 0xF0) | (flags6 >> 4)
        submapper = (flags8 >> 4) & 0x0F
    else:
        mapper = (flags7 & 0xF0) | (flags6 >> 4)
        submapper = 0

    has_trainer = bool(flags6 & 0x04)

    return {
        "mapper": mapper,
        "submapper": submapper,
        "has_trainer": has_trainer,
        "nes2": nes2,
    }


# ---------------------------------------------------------------------------
# CRC calculation
# ---------------------------------------------------------------------------


def rom_crc32(data: bytes, has_trainer: bool) -> str:
    """CRC32 of PRG+CHR payload (after header, trainer skipped if present)."""
    payload = data[HEADER_SIZE:]
    if has_trainer:
        payload = payload[TRAINER_SIZE:]
    return format(zlib.crc32(payload) & 0xFFFFFFFF, "08X")


# ---------------------------------------------------------------------------
# Header patching
# ---------------------------------------------------------------------------


def patch_header(header: bytearray, entry: dict) -> None:
    """Apply rom_db.csv overrides to a mutable 16-byte iNES header in-place."""
    flags6 = header[6]
    flags7 = header[7]
    flags8 = header[8]
    nes2 = (flags7 & 0x0C) == 0x08

    if entry.get("mapper"):
        mapper = int(entry["mapper"])
        flags6 = (flags6 & 0x0F) | ((mapper & 0x0F) << 4)
        flags7 = (flags7 & 0x0F) | (mapper & 0xF0)
        if nes2:
            flags8 = (flags8 & 0xF0) | ((mapper >> 8) & 0x0F)

    if entry.get("submapper") and nes2:
        submapper = int(entry["submapper"])
        flags8 = (flags8 & 0x0F) | ((submapper & 0x0F) << 4)

    if entry.get("nametable_layout"):
        layout = entry["nametable_layout"].strip().upper()
        if layout == "4":
            flags6 = flags6 | 0x08
        elif layout == "H":
            flags6 = flags6 & ~0x09 & 0xFF  # clear bit0 (V mirror) and bit3 (4-screen)
        elif layout == "V":
            flags6 = (flags6 & ~0x08 & 0xFF) | 0x01  # clear 4-screen, set V mirror

    if entry.get("battery"):
        if entry["battery"].strip() == "1":
            flags6 = flags6 | 0x02
        elif entry["battery"].strip() == "0":
            flags6 = flags6 & ~0x02 & 0xFF

    if entry.get("prg_rom_size"):
        prg_pages = int(entry["prg_rom_size"]) // 16384
        if 0 < prg_pages < 256:
            header[4] = prg_pages

    if entry.get("chr_rom_size"):
        chr_pages = int(entry["chr_rom_size"]) // 8192
        if chr_pages < 256:
            header[5] = chr_pages

    header[6] = flags6
    header[7] = flags7
    header[8] = flags8


# ---------------------------------------------------------------------------
# Game Boy header parsing
# ---------------------------------------------------------------------------


def parse_gb_header(data: bytes) -> dict:
    """Return MBC type and CGB mode from a GB cartridge image."""
    mbc_byte = data[_GB_MBC_OFFSET]
    cgb_byte = data[_GB_CGB_OFFSET]

    mbc_name = _GB_MBC_NAMES.get(mbc_byte, f"unknown-{mbc_byte:#04x}")

    if cgb_byte == 0xC0:
        cgb_mode = "cgb-only"
    elif cgb_byte == 0x80:
        cgb_mode = "cgb-compat"
    else:
        cgb_mode = "dmg-only"

    return {"mbc": mbc_name, "cgb_mode": cgb_mode}


# ---------------------------------------------------------------------------
# Game Boy Advance header parsing
# ---------------------------------------------------------------------------


def parse_gba_header(data: bytes) -> dict:
    """Return maker code from a GBA cartridge image."""
    raw = data[_GBA_MAKER_OFFSET:_GBA_MAKER_OFFSET + 2]
    try:
        maker_code = raw.decode("ascii").strip("\x00").strip() or "unknown"
    except UnicodeDecodeError:
        maker_code = "unknown"
    return {"maker_code": maker_code}


# ---------------------------------------------------------------------------
# Platform sort routines
# ---------------------------------------------------------------------------


def sort_nes(db: dict[str, dict]) -> None:
    roms_dir = Path("roms/nes")
    mappers_dir = roms_dir / "mappers"
    nes_files = sorted(roms_dir.glob("*.nes"))
    print(f"Found {len(nes_files)} .nes files in {roms_dir}/\n")

    copied = 0
    for rom_path in tqdm(nes_files, desc="Sorting NES ROMs", unit="rom"):
        data = rom_path.read_bytes()

        if data[:4] != INES_MAGIC:
            raise ValueError(f"{rom_path.name}: not a valid iNES file (magic={data[:4]!r})")

        header_info = parse_header(data[:HEADER_SIZE])
        crc = rom_crc32(data, header_info["has_trainer"])
        entry = db.get(crc)

        header = bytearray(data[:HEADER_SIZE])
        if entry:
            patch_header(header, entry)
            final_info = parse_header(bytes(header))
        else:
            final_info = header_info

        dest_dir = mappers_dir / str(final_info["mapper"]) / str(final_info["submapper"])
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / rom_path.name).write_bytes(bytes(header) + data[HEADER_SIZE:])
        copied += 1

    print(f"\nDone. Copied: {copied}")


def sort_gb() -> None:
    roms_dir = Path("roms/gb")
    mappers_dir = roms_dir / "mappers"
    gb_files = sorted(roms_dir.glob("*.gb"))
    print(f"Found {len(gb_files)} .gb files in {roms_dir}/\n")

    copied = 0
    for rom_path in tqdm(gb_files, desc="Sorting GB ROMs", unit="rom"):
        data = rom_path.read_bytes()

        if len(data) < _GB_HEADER_MIN:
            raise ValueError(f"{rom_path.name}: file too small to be a valid GB ROM ({len(data)} bytes)")

        info = parse_gb_header(data)
        dest_dir = mappers_dir / info["mbc"] / info["cgb_mode"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / rom_path.name).write_bytes(data)
        copied += 1

    print(f"\nDone. Copied: {copied}")


def sort_cgb() -> None:
    roms_dir = Path("roms/cgb")
    mappers_dir = roms_dir / "mappers"
    cgb_files = sorted(roms_dir.glob("*.gbc"))
    print(f"Found {len(cgb_files)} .gbc files in {roms_dir}/\n")

    copied = 0
    for rom_path in tqdm(cgb_files, desc="Sorting CGB ROMs", unit="rom"):
        data = rom_path.read_bytes()

        if len(data) < _GB_HEADER_MIN:
            raise ValueError(f"{rom_path.name}: file too small to be a valid GBC ROM ({len(data)} bytes)")

        info = parse_gb_header(data)
        dest_dir = mappers_dir / info["mbc"] / info["cgb_mode"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / rom_path.name).write_bytes(data)
        copied += 1

    print(f"\nDone. Copied: {copied}")


def sort_gba() -> None:
    roms_dir = Path("roms/gba")
    makers_dir = roms_dir / "makers"
    gba_files = sorted(roms_dir.glob("*.gba"))
    print(f"Found {len(gba_files)} .gba files in {roms_dir}/\n")

    copied = 0
    for rom_path in tqdm(gba_files, desc="Sorting GBA ROMs", unit="rom"):
        data = rom_path.read_bytes()

        if len(data) < _GBA_HEADER_MIN:
            raise ValueError(f"{rom_path.name}: file too small to be a valid GBA ROM ({len(data)} bytes)")

        info = parse_gba_header(data)
        dest_dir = makers_dir / info["maker_code"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / rom_path.name).write_bytes(data)
        copied += 1

    print(f"\nDone. Copied: {copied}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Sort ROMs by hardware metadata")
    parser.add_argument(
        "platform",
        choices=["nes", "gb", "cgb", "gba"],
        help="Platform: 'nes' (roms/nes/mappers/), 'gb' (roms/gb/mappers/), 'cgb' (roms/cgb/mappers/), or 'gba' (roms/gba/makers/)",
    )
    args = parser.parse_args()

    if args.platform == "nes":
        db = load_rom_db(ROM_DB_PATH)
        print(f"Loaded {len(db)} entries from {ROM_DB_PATH}")
        sort_nes(db)
    elif args.platform == "gb":
        sort_gb()
    elif args.platform == "cgb":
        sort_cgb()
    else:
        sort_gba()


if __name__ == "__main__":
    main()
