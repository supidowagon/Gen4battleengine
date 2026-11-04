"""The NDS file system of a ROM, read through ndspy.

``RomFiles`` unpacks the FNT/FAT files, NARC archives, the ARM9 binary with its
sections, the ARM9 overlays, and the parts outside the file system (overlay tables, ARM9
footer, ARM7 binary, banner). ndspy is imported inside ``open`` only, so the rest
of ``tools.rom`` imports without it. Nothing is skipped silently: unknown paths and
overlay ids raise, naming what was asked for.
"""

from __future__ import annotations


class RomFiles:
    """Read-only view of one NDS ROM; build it with ``RomFiles.open``."""

    def __init__(self, rom, data: bytes, paths: dict[str, int]):
        self._rom = rom
        self._data = data
        self._paths = paths
        self._overlays: dict | None = None
        self._arm9 = None

    @classmethod
    def open(cls, data: bytes) -> RomFiles:
        import ndspy.rom

        if len(data) < 0x200:
            raise ValueError(f"ROM too short for an NDS header ({len(data)} bytes)")
        rom = ndspy.rom.NintendoDSRom(data)
        paths: dict[str, int] = {}

        def walk(folder, prefix: str) -> None:
            for index, name in enumerate(folder.files):
                paths[prefix + name] = folder.firstID + index
            for name, sub in folder.folders:
                walk(sub, f"{prefix}{name}/")

        walk(rom.filenames, "")
        return cls(rom, bytes(data), paths)

    @property
    def game_code(self) -> str:
        return self._data[0x0C:0x10].decode("ascii")

    def header(self) -> bytes:
        """The raw 0x200-byte NDS header."""
        return self._data[:0x200]

    def paths(self) -> list[str]:
        """Every FNT file path, ``/``-separated without a leading slash, sorted."""
        return sorted(self._paths)

    def file(self, path: str) -> bytes:
        try:
            file_id = self._paths[path]
        except KeyError:
            raise KeyError(f"no such file in the ROM file system: {path}") from None
        return bytes(self._rom.files[file_id])

    def narc(self, path: str) -> list[bytes]:
        """The members of the NARC at ``path``, in archive order."""
        import ndspy.narc

        try:
            return [bytes(m) for m in ndspy.narc.NARC(self.file(path)).files]
        except KeyError:
            raise
        except Exception as e:
            raise ValueError(f"{path}: not a readable NARC ({e})") from e

    def _code(self):
        if self._arm9 is None:
            self._arm9 = self._rom.loadArm9()
        return self._arm9

    def arm9(self) -> bytes:
        """The ARM9 binary, decompressed when the ROM stores it compressed."""
        from ndspy import codeCompression

        return bytes(codeCompression.decompress(self._rom.arm9))

    def arm9_sections(self) -> list[tuple[int, bytes]]:
        """(RAM address, data) per ARM9 section; the static section comes first."""
        return [(s.ramAddress, bytes(s.data)) for s in self._code().sections]

    def _overlay_table(self) -> dict:
        if self._overlays is None:
            self._overlays = self._rom.loadArm9Overlays()
        return self._overlays

    def overlay_ids(self) -> list[int]:
        return sorted(self._overlay_table())

    def _ov(self, i: int):
        try:
            return self._overlay_table()[i]
        except KeyError:
            raise KeyError(f"no ARM9 overlay {i} in the ROM") from None

    def overlay(self, i: int) -> bytes:
        """Overlay ``i``, decompressed when flagged."""
        return bytes(self._ov(i).data)

    def arm9_overlay_table(self) -> bytes:
        """The raw ARM9 overlay table (``y9``): 32 bytes per overlay."""
        return bytes(self._rom.arm9OverlayTable)

    def arm9_footer(self) -> bytes:
        """The 12 bytes stored after the ARM9 binary (ndspy ``arm9PostData``)."""
        return bytes(self._rom.arm9PostData)

    def arm7(self) -> bytes:
        """The ARM7 binary as stored."""
        return bytes(self._rom.arm7)

    def arm7_overlay_table(self) -> bytes:
        """The raw ARM7 overlay table (``y7``)."""
        return bytes(self._rom.arm7OverlayTable)

    def banner(self) -> bytes:
        """The icon/title banner as stored."""
        return bytes(self._rom.iconBanner)

    def overlay_info(self, i: int) -> tuple[int, int, int]:
        """(RAM address, RAM size, BSS size) of overlay ``i``."""
        o = self._ov(i)
        return o.ramAddress, o.ramSize, o.bssSize
