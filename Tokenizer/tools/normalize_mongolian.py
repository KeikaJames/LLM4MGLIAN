# -*- coding: utf-8 -*-
"""CLI bridge to the Rust normalizer.

This module shells out to `cargo run --example normalize` inside the
`Encoding Mapping/` crate. PyO3 binding is a planned follow-up.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _find_crate() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "Encoding Mapping"
        if (cand / "Cargo.toml").exists():
            return cand
    raise FileNotFoundError(
        "could not locate 'Encoding Mapping/' crate next to this repo. "
        "Run from inside the LLM4MGLIAN checkout."
    )


def normalize(text: str, nominal: bool = False) -> str:
    crate = _find_crate()
    if not _has_cargo():
        raise RuntimeError(
            "cargo not found in PATH; install Rust or use a PyO3 build when available."
        )
    args = ["cargo", "run", "--quiet", "--example", "normalize"]
    if nominal:
        args += ["--", "--nominal"]
    result = subprocess.run(
        args,
        cwd=str(crate),
        input=text,
        # Force UTF-8 for the Mongolian payload instead of relying on the host
        # locale (which may be C/POSIX in CI/containers and would mojibake or
        # raise). ``encoding`` implies text mode.
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"normalize binary failed (rc={result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout


def _has_cargo() -> bool:
    try:
        subprocess.run(
            ["cargo", "--version"], capture_output=True, check=True, text=True
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nominal",
        action="store_true",
        help="collapse to nominal Unicode (recommended for tokenizer ingest)",
    )
    parser.add_argument(
        "--input",
        help="input file (defaults to stdin)",
    )
    parser.add_argument(
        "--output",
        help="output file (defaults to stdout)",
    )
    args = parser.parse_args()

    if args.input:
        text = Path(args.input).read_text(encoding="utf-8")
    else:
        # Decode stdin as UTF-8 explicitly rather than via the host locale.
        text = sys.stdin.buffer.read().decode("utf-8")
    out = normalize(text, nominal=args.nominal)
    if args.output:
        Path(args.output).write_text(out, encoding="utf-8")
    else:
        sys.stdout.buffer.write(out.encode("utf-8"))


if __name__ == "__main__":
    main()
