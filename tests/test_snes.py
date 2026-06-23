"""Unit tests for SNES support.

Covers the pure header-parsing/sorting logic in sort.py (copier-header
detection, checksum-scoring header location, mapping-mode and chipset folder
names, graceful fallback) and the zip-entry selector in scrape.py.
"""

# Tests deliberately exercise module-internal helpers and layout constants.
# Missing per-test docstrings and a large helper signature are expected in tests.
# pylint: disable=protected-access,missing-function-docstring,too-many-arguments

from __future__ import annotations

import pytest

import scrape
import sort

LOROM = 0x7FC0
HIROM = 0xFFC0
EXHIROM = 0x40FFC0


def build_rom(
    loc: int,
    *,
    size: int | None = None,
    map_byte: int = 0x20,
    cart_byte: int = 0x00,
    title: str = "TEST ROM",
    checksum: int = 0x1234,
    valid_checksum: bool = True,
    copier: bool = False,
) -> bytes:
    """Build a minimal synthetic SNES ROM with a header at file offset `loc`.

    When `valid_checksum` is True the checksum/complement pair sums to 0xFFFF.
    When `copier` is True a 512-byte copier header is prepended.
    """
    if size is None:
        size = loc + 0x40
    data = bytearray(size)

    title_bytes = title.encode("ascii").ljust(sort._SNES_TITLE_SIZE)[
        : sort._SNES_TITLE_SIZE
    ]
    start = loc + sort._SNES_TITLE_OFFSET
    data[start : start + sort._SNES_TITLE_SIZE] = title_bytes
    data[loc + sort._SNES_MAP_OFFSET] = map_byte
    data[loc + sort._SNES_CART_OFFSET] = cart_byte

    complement = (0xFFFF ^ checksum) if valid_checksum else ((checksum + 1) & 0xFFFF)
    data[loc + sort._SNES_COMPLEMENT_OFFSET] = complement & 0xFF
    data[loc + sort._SNES_COMPLEMENT_OFFSET + 1] = (complement >> 8) & 0xFF
    data[loc + sort._SNES_CHECKSUM_OFFSET] = checksum & 0xFF
    data[loc + sort._SNES_CHECKSUM_OFFSET + 1] = (checksum >> 8) & 0xFF

    if copier:
        return bytes(b"\x00" * sort._SNES_COPIER_HEADER_SIZE + data)
    return bytes(data)


# ---------------------------------------------------------------------------
# Header detection
# ---------------------------------------------------------------------------


def test_detect_lorom():
    rom = build_rom(LOROM, size=0x8000, map_byte=0x20, cart_byte=0x02)
    info = sort.detect_snes_header(rom)
    assert info is not None
    off, map_byte, cart_byte = info
    assert off == LOROM
    assert map_byte == 0x20
    assert cart_byte == 0x02


def test_detect_hirom():
    rom = build_rom(HIROM, size=0x10000, map_byte=0x21)
    info = sort.detect_snes_header(rom)
    assert info is not None
    assert info[0] == HIROM
    assert info[1] == 0x21


def test_detect_exhirom():
    rom = build_rom(EXHIROM, map_byte=0x25)
    info = sort.detect_snes_header(rom)
    assert info is not None
    assert info[0] == EXHIROM


def test_detect_hirom_not_confused_by_lorom_region():
    # The LoROM candidate location lies within a HiROM image; the valid HiROM
    # checksum must make HiROM win.
    rom = build_rom(HIROM, size=0x10000, map_byte=0x21)
    info = sort.detect_snes_header(rom)
    assert info is not None
    assert info[0] == HIROM


def test_detect_without_valid_checksum_uses_mapmode():
    # Hacks/translations often carry wrong checksums; map-mode + printable title
    # still classify the ROM rather than dropping it to unknown.
    rom = build_rom(LOROM, size=0x8000, map_byte=0x20, valid_checksum=False)
    info = sort.detect_snes_header(rom)
    assert info is not None
    assert info[0] == LOROM


def test_detect_garbage_returns_none():
    assert sort.detect_snes_header(b"\x00" * 0x8000) is None
    assert sort.detect_snes_header(b"\xff" * 0x8000) is None


def test_detect_too_small_returns_none():
    assert sort.detect_snes_header(b"\x00" * 0x10) is None


# ---------------------------------------------------------------------------
# Copier header
# ---------------------------------------------------------------------------


def test_has_copier_header():
    assert sort._has_copier_header(0x8000 + 512)
    assert not sort._has_copier_header(0x8000)
    assert sort._has_copier_header(512)
    assert not sort._has_copier_header(0)


def test_detect_after_stripping_copier_header():
    raw = build_rom(LOROM, size=0x8000, map_byte=0x20, copier=True)
    assert sort._has_copier_header(len(raw))
    stripped = raw[sort._SNES_COPIER_HEADER_SIZE :]
    info = sort.detect_snes_header(stripped)
    assert info is not None
    assert info[0] == LOROM


# ---------------------------------------------------------------------------
# Folder-name mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "map_byte, expected",
    [
        (0x20, "lorom"),
        (0x30, "lorom"),  # fast-rom speed bit ignored
        (0x21, "hirom"),
        (0x31, "hirom"),
        (0x23, "sa1"),
        (0x25, "exhirom"),
        (0x22, "exlorom"),
        (0x2A, "spc7110"),
    ],
)
def test_map_mode_names(map_byte, expected):
    assert sort.snes_map_mode_name(map_byte, LOROM) == expected


def test_map_mode_name_unknown_falls_back_to_location():
    assert sort.snes_map_mode_name(0x00, HIROM) == "hirom"
    assert sort.snes_map_mode_name(0x00, 0x12345).startswith("unknown-")


@pytest.mark.parametrize(
    "cart_byte, expected",
    [
        (0x00, "rom-only"),
        (0x01, "rom-ram"),
        (0x02, "rom-ram-battery"),
        (0x03, "dsp"),
        (0x13, "superfx"),
        (0x14, "superfx-ram"),
        (0x15, "superfx-ram-battery"),
        (0x16, "superfx-battery"),
        (0x35, "sa1-ram-battery"),
        (0x45, "sdd1-ram-battery"),
        (0xF5, "custom-ram-battery"),
    ],
)
def test_chipset_names(cart_byte, expected):
    assert sort.snes_chipset_name(cart_byte) == expected


# ---------------------------------------------------------------------------
# sort_snes integration (uses a temporary working directory)
# ---------------------------------------------------------------------------


def test_sort_snes_routes_to_mapping_and_chipset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    roms = tmp_path / "roms" / "snes"
    roms.mkdir(parents=True)
    rom = build_rom(LOROM, size=0x8000, map_byte=0x20, cart_byte=0x02)
    (roms / "test-game.sfc").write_bytes(rom)

    sort.sort_snes()

    out = roms / "mappers" / "lorom" / "rom-ram-battery" / "test-game.sfc"
    assert out.exists()
    assert out.read_bytes() == rom


def test_sort_snes_strips_copier_header_and_renames_smc(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    roms = tmp_path / "roms" / "snes"
    roms.mkdir(parents=True)
    raw = build_rom(LOROM, size=0x8000, map_byte=0x20, cart_byte=0x00, copier=True)
    (roms / "smc-game.smc").write_bytes(raw)

    sort.sort_snes()

    out = roms / "mappers" / "lorom" / "rom-only" / "smc-game.sfc"
    assert out.exists()
    assert out.read_bytes() == raw[sort._SNES_COPIER_HEADER_SIZE :]
    assert len(out.read_bytes()) == 0x8000


def test_sort_snes_unknown_fallback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    roms = tmp_path / "roms" / "snes"
    roms.mkdir(parents=True)
    (roms / "garbage.sfc").write_bytes(b"\x00" * 0x8000)

    sort.sort_snes()

    assert (roms / "mappers" / "unknown" / "unknown" / "garbage.sfc").exists()


# ---------------------------------------------------------------------------
# scrape._select_zip_entry
# ---------------------------------------------------------------------------


def test_select_zip_entry_snes_accepts_sfc_and_smc():
    plat = scrape.PLATFORMS["snes"]
    assert scrape._select_zip_entry(["readme.txt", "Game.smc"], plat) == "Game.smc"
    assert scrape._select_zip_entry(["Game.sfc"], plat) == "Game.sfc"
    assert scrape._select_zip_entry(["cover.png", "notes.txt"], plat) is None


def test_select_zip_entry_nes_uses_rom_ext_only():
    plat = scrape.PLATFORMS["nes"]
    assert scrape._select_zip_entry(["x.nes"], plat) == "x.nes"
    assert scrape._select_zip_entry(["x.sfc"], plat) is None
