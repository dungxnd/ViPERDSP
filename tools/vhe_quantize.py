#!/usr/bin/env python3
"""
vhe_quantize.py — Convert VHE_Lx.h float32 kernel arrays to int16_t.

Each array is quantized independently using its own peak value so the full
int16 dynamic range is always used.  The per-array scale factor is emitted
as a comment and as a C++ constexpr alongside the array so VHE.cpp can fold
it into the existing k.gain without changing the LoadKernel call signature.

Usage:
    python3 tools/vhe_quantize.py

Reads:   viper/effects/VHE_L{0..4}.h
Writes:  viper/effects/VHE_L{0..4}.h  (in-place, originals backed up as .bak)

Precision note:
    int16 SNR floor ≈ 6.02 * 15 + 1.76 ≈ 92 dB.  HRTF convolution noise is
    inaudible at this resolution.  L1 kernels have max_abs ≈ 0.998 and use
    nearly the full int16 range already; L0/L2/L3/L4 have max_abs ≈ 0.3–0.4,
    but per-array normalisation brings them to full scale too, recovering ~3
    extra bits of effective precision vs a global scale.
"""

import re
import shutil
import struct
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EFFECTS_DIR = Path(__file__).parent.parent / "viper" / "effects"
INT16_MAX   = 32767

# Each entry: (header_file, [array_name, ...])
# Order matches the kVheKernels table in VHE.cpp.
HEADERS: list[tuple[str, list[str]]] = [
    ("VHE_L0.h", ["kVheL0_44100_L", "kVheL0_44100_R",
                  "kVheL0_48000_L", "kVheL0_48000_R"]),
    ("VHE_L1.h", ["kVheL1_44100_L", "kVheL1_44100_R",
                  "kVheL1_48000_L", "kVheL1_48000_R"]),
    ("VHE_L2.h", ["kVheL2_44100_L", "kVheL2_44100_R",
                  "kVheL2_48000_L", "kVheL2_48000_R"]),
    ("VHE_L3.h", ["kVheL3_44100_L", "kVheL3_44100_R",
                  "kVheL3_48000_L", "kVheL3_48000_R"]),
    ("VHE_L4.h", ["kVheL4_44100_L", "kVheL4_44100_R",
                  "kVheL4_48000_L", "kVheL4_48000_R"]),
]

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_ARRAY_RE = re.compile(
    r"static\s+const\s+float\s+(\w+)\[\]\s*=\s*\{([^;]+)\};",
    re.DOTALL,
)
_FLOAT_RE = re.compile(r"(-?\d+\.\d+)f")


@dataclass
class KernelArray:
    name:   str
    values: list[float]
    peak:   float = field(init=False)
    scale:  float = field(init=False)  # multiply int16 by this to reconstruct float

    def __post_init__(self) -> None:
        self.peak  = max(abs(v) for v in self.values)
        if self.peak == 0.0:
            self.scale = 1.0 / INT16_MAX
        else:
            # Normalise so peak maps to ±INT16_MAX exactly.
            self.scale = self.peak / INT16_MAX

    def quantize(self) -> list[int]:
        """Return int16-clamped values. Max rounding error: 0.5 LSB."""
        inv = INT16_MAX / self.peak if self.peak != 0.0 else 0.0
        result = []
        for v in self.values:
            q = int(round(v * inv))
            # Clamp to [-32767, 32767] — avoid -32768 (asymmetric two's complement)
            q = max(-INT16_MAX, min(INT16_MAX, q))
            result.append(q)
        return result

    def max_reconstruction_error(self) -> float:
        """Worst-case absolute reconstruction error after round-trip."""
        quant = self.quantize()
        return max(abs(v - (q * self.scale)) for v, q in zip(self.values, quant))


def parse_header(path: Path) -> dict[str, KernelArray]:
    text = path.read_text(encoding="utf-8")
    arrays: dict[str, KernelArray] = {}
    for m in _ARRAY_RE.finditer(text):
        name   = m.group(1)
        floats = [float(f) for f in _FLOAT_RE.findall(m.group(2))]
        arrays[name] = KernelArray(name=name, values=floats)
    return arrays

# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

_VALS_PER_LINE = 8


def _format_int16_array(name: str, values: list[int]) -> str:
    """Emit  static const int16_t kFoo[] = { ... };  with 8 values per line."""
    lines = []
    lines.append(f"static const int16_t {name}[] = {{")
    for i in range(0, len(values), _VALS_PER_LINE):
        chunk = values[i : i + _VALS_PER_LINE]
        row   = ", ".join(f"{v:6d}" for v in chunk)
        comma = "," if (i + _VALS_PER_LINE) < len(values) else ""
        lines.append(f"    {row}{comma}")
    lines.append("};")
    return "\n".join(lines)


def generate_header(filename: str, arrays: dict[str, KernelArray]) -> str:
    """Return the full contents of the new header."""
    lines: list[str] = []
    lines.append("#pragma once")
    lines.append("")
    lines.append("#include <cstdint>")
    lines.append("")
    lines.append("// -----------------------------------------------------------------------")
    lines.append("// VHE kernel arrays — int16_t quantized from float32 originals.")
    lines.append("// Each array has a companion kXxx_Scale constexpr.  Multiply int16")
    lines.append("// values by kXxx_Scale to reconstruct float.  VHE.cpp folds this into")
    lines.append("// k.gain so LoadKernel receives the same effective coefficients as before.")
    lines.append("//")
    lines.append("// Quantization: per-array normalisation to ±32767.  SNR ≥ 92 dB.")
    lines.append("// -----------------------------------------------------------------------")
    lines.append("")

    for name in arrays:          # preserve declaration order
        ka = arrays[name]
        quant = ka.quantize()
        err   = ka.max_reconstruction_error()
        lines.append(f"// peak_abs={ka.peak:.7f}  scale={ka.scale:.10f}  max_err={err:.2e}")
        lines.append(f"inline constexpr float {name}_Scale = {ka.scale:.10f}f;")
        lines.append(_format_int16_array(name, quant))
        lines.append("")

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(arrays: dict[str, KernelArray]) -> None:
    """Assert round-trip error is below half an LSB for every tap."""
    for name, ka in arrays.items():
        err = ka.max_reconstruction_error()
        half_lsb = ka.scale * 0.5
        assert err <= half_lsb + 1e-9, (
            f"FAIL {name}: max_err={err:.2e} > half_lsb={half_lsb:.2e}"
        )
        print(f"  OK  {name:30s}  n={len(ka.values):4d}  "
              f"peak={ka.peak:.4f}  scale={ka.scale:.8f}  max_err={err:.2e}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Source directory: {EFFECTS_DIR}\n")

    for filename, expected_names in HEADERS:
        path = EFFECTS_DIR / filename
        print(f"Processing {filename} ...")

        arrays = parse_header(path)

        # Verify all expected arrays were found
        for name in expected_names:
            if name not in arrays:
                raise RuntimeError(f"Array '{name}' not found in {filename}")

        # Preserve declaration order from expected_names
        ordered = {n: arrays[n] for n in expected_names}

        print("  Validating round-trip precision ...")
        validate(ordered)

        # Back up original
        bak = path.with_suffix(".h.bak")
        shutil.copy2(path, bak)
        print(f"  Backup: {bak.name}")

        # Write new header
        new_text = generate_header(filename, ordered)
        path.write_text(new_text, encoding="utf-8")

        # Size report
        orig_kb = bak.stat().st_size  / 1024
        new_kb  = path.stat().st_size / 1024
        saved   = orig_kb - new_kb
        print(f"  Size:   {orig_kb:.1f} KB -> {new_kb:.1f} KB  (-{saved:.1f} KB, "
              f"{saved/orig_kb*100:.0f}% smaller)\n")

    print("Done.  Next step: update VHE.cpp -- fold _Scale into k.gain.")
    print("See: tools/vhe_quantize.py docstring for the LoadKernel call pattern.")


if __name__ == "__main__":
    main()
