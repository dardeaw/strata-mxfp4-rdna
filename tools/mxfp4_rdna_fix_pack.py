#!/usr/bin/env python3
"""Drop the in-file MTP block from an MXFP4-for-RDNA pack.

The tsaipifong Swift-1.5 Qwen3.8-Flash-Next GGUF carries the MTP head as block
48 (`qwen4exp.nextn_predict_layers = 1`, `block_count = 49`).  Strata runs the
48-layer trunk; the engine's geometry/native-dense fixes skip block 48, and the
pack must not list it as a native expert layer:

  * `native_experts.txt`: remove the layers >= --layers and update the header's
    `total` (the last kept layer's offset + blob_bytes * n_expert);
  * `index.txt`: remove the `blk.<n>.` tensors with n >= --layers and update the
    `tensors` count.

`dense.bin` is left alone: the extra bytes are not referenced by the trimmed
index.  Run this after `tools/iq_pack.py`:

    python3 tools/mxfp4_rdna_fix_pack.py --pack ~/Strata-data/packs/tsaipifong-mxfp4 --layers 48
"""
import argparse
import pathlib
import re
import sys


def block_index(name: str) -> int | None:
    m = re.match(r"blk\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def fix_native(path: pathlib.Path, layers: int) -> tuple[int, int, int]:
    lines = path.read_text().splitlines()
    n_expert = 512
    for line in lines:
        if line.startswith("#") and "n_expert" in line:
            m = re.search(r"n_expert\s+(\d+)", line)
            if m:
                n_expert = int(m.group(1))
    kept = []
    last = None
    removed = 0
    for line in lines:
        if line.startswith("#") or not line.strip():
            kept.append(line)
            continue
        parts = line.split()
        layer = int(parts[0])
        if layer >= layers:
            removed += 1
            continue
        kept.append(line)
        last = parts
    if last is not None:
        total = int(last[3]) + int(last[4]) * n_expert
        for i, line in enumerate(kept):
            if line.startswith("#"):
                kept[i] = re.sub(r"total\s+\d+", f"total {total}", line, count=1)
                break
    path.write_text("\n".join(kept) + "\n")
    return removed, (int(last[0]) + 1 if last else 0), (total if last else 0)


def fix_index(path: pathlib.Path, layers: int) -> tuple[int, int]:
    lines = path.read_text().splitlines()
    kept = []
    removed = 0
    for line in lines:
        if line.startswith("blk."):
            b = block_index(line)
            if b is not None and b >= layers:
                removed += 1
                continue
        kept.append(line)
    tensors = sum(1 for line in kept if line and not line.startswith("#"))
    for i, line in enumerate(kept):
        if line.startswith("# align"):
            kept[i] = re.sub(r"tensors\s+\d+", f"tensors {tensors}", line, count=1)
            break
    path.write_text("\n".join(kept) + "\n")
    return removed, tensors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", required=True, help="pack directory (native_experts.txt, index.txt)")
    ap.add_argument("--layers", type=int, default=48, help="trunk layer count (default: 48)")
    args = ap.parse_args()
    pack = pathlib.Path(args.pack)
    native = pack / "native_experts.txt"
    index = pack / "index.txt"
    if not native.is_file() or not index.is_file():
        print(f"error: {pack} has no native_experts.txt/index.txt", file=sys.stderr)
        return 1
    n_removed, n_layers, total = fix_native(native, args.layers)
    i_removed, tensors = fix_index(index, args.layers)
    print(f"native_experts.txt: removed {n_removed} MTP layer line(s); trunk {n_layers} layers, total {total}")
    print(f"index.txt: removed {i_removed} blk.{args.layers}+ tensor(s); {tensors} tensors left")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
