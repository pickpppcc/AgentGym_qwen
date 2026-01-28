#!/usr/bin/env python3
"""
Extract files from a *prefix* of a split ZIP archive (e.g. image.part.aa).

Why:
  Your previous "scan for JPEG markers" approach will almost always produce
  corrupted files because the bytes are (usually) compressed ZIP payload.

This script parses ZIP local file headers (PK\\x03\\x04) and extracts entries
sequentially until EOF. This works even without the ZIP central directory
(which lives at the end, in later parts).

Typical usage (extract first 100 images from part aa):
  python AgentGym2/AgentGym/scripts/unzip_file.py \\
    --input /data/pengchen/AgentGym_qwen/data/hf_grounding_dataset/image.part.aa \\
    --output /data/pengchen/OSWorld/Osworld/extracted_raw_images \\
    --include-ext .jpg .jpeg .png .webp \\
    --limit 100
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import struct
import zlib
from pathlib import Path
from typing import BinaryIO, Iterable, Optional, Tuple


ZIP_LOCAL_FILE_HEADER_SIG = b"PK\x03\x04"
ZIP_DATA_DESCRIPTOR_SIG = b"PK\x07\x08"


class _BufferedReader:
    """Tiny buffered reader that supports pushing back unused bytes."""

    def __init__(self, f: BinaryIO):
        self._f = f
        self._buf = b""

    def tell(self) -> int:
        # Position in the underlying file minus buffered unread bytes.
        return self._f.tell() - len(self._buf)

    def read(self, n: int) -> bytes:
        if n <= 0:
            return b""
        if self._buf:
            out = self._buf[:n]
            self._buf = self._buf[n:]
            if len(out) == n:
                return out
            rest = self._f.read(n - len(out))
            return out + rest
        return self._f.read(n)

    def pushback(self, data: bytes) -> None:
        if data:
            self._buf = data + self._buf

    def read_exact(self, n: int) -> Optional[bytes]:
        data = self.read(n)
        if len(data) != n:
            return None
        return data


def _find_first_sig(f: BinaryIO, sig: bytes = ZIP_LOCAL_FILE_HEADER_SIG) -> Optional[int]:
    """
    Find first local-header signature in a large binary file without reading it all.
    Returns file offset (int) or None.
    """
    chunk = 8 * 1024 * 1024
    overlap = len(sig) - 1
    pos = 0
    tail = b""
    while True:
        data = f.read(chunk)
        if not data:
            return None
        buf = tail + data
        idx = buf.find(sig)
        if idx != -1:
            return pos - len(tail) + idx
        tail = buf[-overlap:]
        pos += len(data)


def _safe_member_path(out_dir: Path, name: str) -> Optional[Path]:
    # Prevent path traversal (../../etc/passwd) when extracting.
    name = name.replace("\\", "/")
    if name.startswith("/") or name.startswith("../") or "/../" in name:
        return None
    p = (out_dir / name).resolve()
    try:
        out_dir_resolved = out_dir.resolve()
    except Exception:
        out_dir_resolved = out_dir
    if not str(p).startswith(str(out_dir_resolved)):
        return None
    return p


def _match_filters(name: str, include_globs: Iterable[str], include_ext: Iterable[str]) -> bool:
    n = name.replace("\\", "/")
    if include_ext:
        lower = n.lower()
        if not any(lower.endswith(ext.lower()) for ext in include_ext):
            return False
    if include_globs:
        if not any(fnmatch.fnmatch(n, g) for g in include_globs):
            return False
    return True


def _read_data_descriptor(br: _BufferedReader) -> Optional[Tuple[int, int, int]]:
    """
    Read ZIP data descriptor and return (crc32, compressed_size, uncompressed_size).
    Descriptor may start with optional signature PK\\x07\\x08.
    """
    head = br.read_exact(16)
    if head is None:
        return None
    if head[:4] == ZIP_DATA_DESCRIPTOR_SIG:
        crc32, csize, usize = struct.unpack("<III", head[4:16])
        return crc32, csize, usize
    # No signature; first 12 bytes are the descriptor.
    crc32, csize, usize = struct.unpack("<III", head[:12])
    br.pushback(head[12:16])
    return crc32, csize, usize


def _extract_deflate_unknown_size(br: _BufferedReader, out_f: BinaryIO) -> bool:
    """
    Stream-decompress a deflate stream until zlib signals end-of-stream.
    Leaves br positioned right after the end-of-stream (with unused bytes pushed back).
    """
    decomp = zlib.decompressobj(wbits=-15)  # raw deflate stream in ZIP
    while True:
        chunk = br.read(1024 * 1024)
        if not chunk:
            return False  # truncated
        try:
            out = decomp.decompress(chunk)
        except zlib.error:
            return False
        if out:
            out_f.write(out)
        if decomp.eof:
            # Push back bytes read past the end of this stream (descriptor starts here).
            br.pushback(decomp.unused_data)
            return True


def extract_prefix_zip(
    input_path: Path,
    output_dir: Path,
    include_globs: Iterable[str],
    include_ext: Iterable[str],
    limit: int,
    overwrite: bool,
    verbose: bool = False,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0
    scanned = 0

    with input_path.open("rb") as f:
        start = _find_first_sig(f)
        if start is None:
            raise RuntimeError("No ZIP local file header signature found (PK\\x03\\x04).")
        f.seek(start)
        br = _BufferedReader(f)

        while True:
            if limit and extracted >= limit:
                break

            sig = br.read_exact(4)
            if sig is None or sig != ZIP_LOCAL_FILE_HEADER_SIG:
                break

            fixed = br.read_exact(26)
            if fixed is None:
                break

            (
                _version_needed,
                flags,
                method,
                _mtime,
                _mdate,
                _crc32,
                comp_size,
                _uncomp_size,
                fname_len,
                extra_len,
            ) = struct.unpack("<HHHHHIIIHH", fixed)

            name_b = br.read_exact(fname_len)
            extra_b = br.read_exact(extra_len)
            if name_b is None or extra_b is None:
                break

            # UTF-8 flag is bit 11; still try utf-8 with replacement.
            name = name_b.decode("utf-8", errors="replace")
            scanned += 1
            if verbose:
                print(f"[info] entry#{scanned} method={method} flags=0x{flags:x} csize={comp_size} name={name!r}")

            # Directories: just ensure it exists and continue.
            if name.endswith("/") or name.endswith("\\"):
                p = _safe_member_path(output_dir, name)
                if p is not None:
                    p.mkdir(parents=True, exist_ok=True)
                continue

            do_extract = _match_filters(name, include_globs, include_ext)
            out_path = _safe_member_path(output_dir, name) if do_extract else None
            if out_path is not None:
                out_path.parent.mkdir(parents=True, exist_ok=True)

            # Some ZIPs use data descriptor (bit 3). Then comp_size may be 0 in local header.
            has_descriptor = bool(flags & (1 << 3))

            if method not in (0, 8):
                # Unsupported compression; we can only skip if comp_size is known.
                if not has_descriptor and comp_size:
                    _ = br.read(comp_size)
                    continue
                raise RuntimeError(
                    f"Unsupported compression method={method} for member {name!r} with unknown size; "
                    f"need full zip (central directory) or add support for this method."
                )

            if method == 0:
                # Stored (no compression).
                if has_descriptor:
                    # Can't skip/parse without knowing the size (no end marker).
                    raise RuntimeError(
                        f"Member {name!r} is STORED with data descriptor; cannot extract from prefix-only zip reliably."
                    )
                payload = br.read_exact(comp_size)
                if payload is None:
                    break
                if out_path is not None:
                    if out_path.exists() and not overwrite:
                        pass
                    else:
                        out_path.write_bytes(payload)
                        extracted += 1
                continue

            # Deflate
            if not has_descriptor:
                # Known compressed size.
                comp = br.read_exact(comp_size)
                if comp is None:
                    break
                if out_path is not None:
                    if out_path.exists() and not overwrite:
                        continue
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    # Stream to disk to keep memory bounded.
                    try:
                        decomp = zlib.decompressobj(wbits=-15)
                        with out_path.open("wb") as out_f:
                            out_f.write(decomp.decompress(comp))
                            out_f.write(decomp.flush())
                        extracted += 1
                    except zlib.error:
                        # Likely hit an entry that spans into the next split part or a corrupt header.
                        # Keep going (best-effort for prefix-only extraction).
                        try:
                            if out_path.exists():
                                out_path.unlink()
                        except Exception:
                            pass
                        if verbose:
                            print(f"[warn] zlib failed for {name!r}; skipping")
                continue

            # Unknown compressed size -> stream until EOF of deflate member, then read data descriptor.
            if out_path is not None and out_path.exists() and not overwrite:
                # If not overwriting, still need to consume the stream to reach next header.
                with open(os.devnull, "wb") as out_f:
                    ok = _extract_deflate_unknown_size(br, out_f)
                if not ok:
                    break
            else:
                if out_path is None:
                    with open(os.devnull, "wb") as out_f:
                        ok = _extract_deflate_unknown_size(br, out_f)
                    if not ok:
                        break
                else:
                    with out_path.open("wb") as out_f:
                        ok = _extract_deflate_unknown_size(br, out_f)
                    if not ok:
                        break
                    extracted += 1

            dd = _read_data_descriptor(br)
            if dd is None:
                break

    print(f"[ok] scanned {scanned} entries, extracted {extracted} files into: {output_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="/data/pengchen/AgentGym_qwen/data/hf_grounding_dataset/image.part.aa",
        help="A split ZIP prefix file (e.g. image.part.aa).",
    )
    ap.add_argument(
        "--output",
        default="/data/pengchen/OSWorld/Osworld/extracted_raw_images",
        help="Output directory.",
    )
    ap.add_argument("--include-glob", action="append", default=[], help="Glob filter (repeatable).")
    ap.add_argument(
        "--include-ext",
        nargs="*",
        default=[".jpg", ".jpeg", ".png", ".webp"],
        help="Only extract files with these extensions (space-separated). Use empty to disable.",
    )
    ap.add_argument("--limit", type=int, default=1000, help="Max files to extract (0 = no limit).")
    ap.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing files.")
    ap.add_argument("--verbose", action="store_true", default=False, help="Print entry-level progress.")
    args = ap.parse_args()

    return extract_prefix_zip(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        include_globs=args.include_glob,
        include_ext=args.include_ext,
        limit=args.limit,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    raise SystemExit(main())
