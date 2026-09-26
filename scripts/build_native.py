"""Build the optional Heliax CPU kernel library."""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "heliax" / "native_ops.c"
OUTPUT = ROOT / "src" / "heliax" / "_native.so"


def main() -> None:
    compiler = os.environ.get("CC", "gcc")
    output = Path(os.environ.get("HELIAX_NATIVE_OUT", OUTPUT))
    command = [
        compiler,
        "-O3",
        "-std=c11",
        "-fPIC",
        "-shared",
        "-fno-math-errno",
        str(SOURCE),
        "-o",
        str(output),
        "-lm",
    ]
    if os.environ.get("HELIAX_NATIVE_OPENMP", "").lower() in {"1", "true", "yes"}:
        command[1:1] = ["-fopenmp"]
    if os.environ.get("HELIAX_NATIVE_NATIVE_ARCH", "").lower() in {"1", "true", "yes"}:
        command.insert(1, "-march=native")
    print("building:", " ".join(command))
    subprocess.run(command, check=True)
    print(f"built {output} ({platform.machine()})")


if __name__ == "__main__":
    main()
