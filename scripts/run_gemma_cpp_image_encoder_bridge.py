#!/usr/bin/env python3
"""Runs the gemma.cpp OpenPI image-encoder bridge and compares F32 outputs."""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from openpi.shared import gemma_cpp_image_encoder_bridge as bridge  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemma-cpp-root", default="~/gemma.cpp")
    parser.add_argument("--binary", default="")
    parser.add_argument("--bundle", default="build/openpi_golden/pi05_base_image_encoder_raw_bundle")
    parser.add_argument("--output-dir", default="build/openpi_golden/gemma_cpp_bridge_smoke")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--image-f32", default="")
    parser.add_argument("--image-ppm", default="")
    parser.add_argument("--attention", choices=("serial", "hybrid", "parallel"), default="hybrid")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--encoded-max-abs-tol", type=float, default=1.5)
    parser.add_argument("--tokens-max-abs-tol", type=float, default=2.5)
    args = parser.parse_args()
    if args.image_f32 and args.image_ppm:
        parser.error("--image-f32 and --image-ppm are mutually exclusive")
    return args


def maybe_path(value: str) -> pathlib.Path | None:
    return bridge.expand_path(value) if value else None


def main() -> int:
    args = parse_args()
    config = bridge.BridgeConfig(
        gemma_cpp_root=bridge.expand_path(args.gemma_cpp_root),
        binary=maybe_path(args.binary),
        bundle=bridge.expand_path(args.bundle),
        output_dir=bridge.expand_path(args.output_dir),
        build=args.build,
        image_f32=maybe_path(args.image_f32),
        image_ppm=maybe_path(args.image_ppm),
        attention=args.attention,
        warmup=args.warmup,
        runs=args.runs,
        compare=not args.skip_compare,
    )

    try:
        result = bridge.run_bridge(config, echo_commands=True)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        print("hint: pass --build or run scripts/build_openpi_image_encoder_hip_run.sh in gemma.cpp", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        return exc.returncode

    print(f"tokens_out {result.tokens_out} bytes={result.tokens_out.stat().st_size}")
    print(f"encoded_out {result.encoded_out} bytes={result.encoded_out.stat().st_size}")
    if config.compare and args.image_ppm:
        print("warning: comparing a custom PPM image against bundle golden tensors", file=sys.stderr)
    if config.compare and config.image_f32 is not None and config.image_f32.resolve() != (
        config.bundle / "golden_image.f32"
    ).resolve():
        print("warning: comparing a custom F32 image against bundle golden tensors", file=sys.stderr)

    if result.encoded_metrics is not None:
        print(bridge.format_metrics("encoded", result.encoded_metrics))
    if result.tokens_metrics is not None:
        print(bridge.format_metrics("tokens", result.tokens_metrics))
    print(f"manifest {result.manifest_path}")

    if args.strict:
        bridge.assert_within_tolerance(
            result,
            encoded_max_abs_tol=args.encoded_max_abs_tol,
            tokens_max_abs_tol=args.tokens_max_abs_tol,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
