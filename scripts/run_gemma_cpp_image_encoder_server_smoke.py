#!/usr/bin/env python3
"""Smoke-tests the resident gemma.cpp OpenPI image-encoder server."""

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
    parser.add_argument("--output-dir", default="build/openpi_golden/gemma_cpp_resident_server_smoke")
    parser.add_argument("--image-f32", default="")
    parser.add_argument("--image-ppm", default="")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--attention", choices=("serial", "hybrid", "parallel"), default="hybrid")
    parser.add_argument("--first-warmup", type=int, default=1)
    parser.add_argument("--first-runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()
    if args.image_f32 and args.image_ppm:
        parser.error("--image-f32 and --image-ppm are mutually exclusive")
    if args.requests < 1:
        parser.error("--requests must be >= 1")
    return args


def maybe_path(value: str) -> pathlib.Path | None:
    return bridge.expand_path(value) if value else None


def main() -> int:
    args = parse_args()
    output_dir = bridge.expand_path(args.output_dir)
    config = bridge.ServerConfig(
        gemma_cpp_root=bridge.expand_path(args.gemma_cpp_root),
        binary=maybe_path(args.binary),
        bundle=bridge.expand_path(args.bundle),
        build=args.build,
        attention=args.attention,
        warmup=args.warmup,
        runs=args.runs,
    )
    image_f32 = maybe_path(args.image_f32)
    image_ppm = maybe_path(args.image_ppm)

    try:
        with bridge.ImageEncoderServer(config, echo_commands=True) as server:
            for index in range(args.requests):
                result = server.run(
                    image_f32=image_f32,
                    image_ppm=image_ppm,
                    tokens_out=output_dir / f"tokens_{index + 1}.f32",
                    encoded_out=output_dir / f"encoded_{index + 1}.f32",
                    warmup=args.first_warmup if index == 0 else args.warmup,
                    runs=args.first_runs if index == 0 else args.runs,
                )
                print(
                    f"request={index + 1} wall_ms={result.wall_ms:.6g} "
                    f"tokens_out={result.tokens_out} encoded_out={result.encoded_out}"
                )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        print(
            "hint: pass --build or run scripts/build_openpi_image_encoder_hip_server.sh in gemma.cpp",
            file=sys.stderr,
        )
        return 1
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
