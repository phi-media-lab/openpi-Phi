from __future__ import annotations

from array import array
import dataclasses
import json
import math
import pathlib
import shlex
import subprocess
import sys


IMAGE_TOKEN_SHAPE = (1, 256, 2048)
ENCODED_SHAPE = (1, 256, 1152)
IMAGE_TOKEN_FLOAT_COUNT = IMAGE_TOKEN_SHAPE[0] * IMAGE_TOKEN_SHAPE[1] * IMAGE_TOKEN_SHAPE[2]
ENCODED_FLOAT_COUNT = ENCODED_SHAPE[0] * ENCODED_SHAPE[1] * ENCODED_SHAPE[2]
FLOAT32_BYTES = 4
FLOAT32_DTYPE = "float32_le"


@dataclasses.dataclass(frozen=True)
class TensorMetrics:
    count: int
    max_abs: float
    rms: float
    mean_abs: float

    def to_json(self) -> dict[str, float | int]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class BridgeConfig:
    gemma_cpp_root: pathlib.Path
    bundle: pathlib.Path
    output_dir: pathlib.Path
    binary: pathlib.Path | None = None
    build: bool = False
    image_f32: pathlib.Path | None = None
    image_ppm: pathlib.Path | None = None
    attention: str = "hybrid"
    warmup: int = 1
    runs: int = 3
    compare: bool = True


@dataclasses.dataclass(frozen=True)
class BridgeResult:
    command: list[str]
    binary: pathlib.Path
    bundle: pathlib.Path
    tokens_out: pathlib.Path
    encoded_out: pathlib.Path
    manifest_path: pathlib.Path
    encoded_metrics: TensorMetrics | None
    tokens_metrics: TensorMetrics | None


@dataclasses.dataclass(frozen=True)
class ServerConfig:
    gemma_cpp_root: pathlib.Path
    bundle: pathlib.Path
    binary: pathlib.Path | None = None
    build: bool = False
    attention: str = "hybrid"
    warmup: int = 0
    runs: int = 1


@dataclasses.dataclass(frozen=True)
class ServerRunResult:
    command: str
    response: dict[str, str]
    tokens_out: pathlib.Path
    encoded_out: pathlib.Path

    @property
    def wall_ms(self) -> float:
        return float(self.response["wall_ms"])


def expand_path(path: str | pathlib.Path) -> pathlib.Path:
    return pathlib.Path(path).expanduser()


def read_f32(path: pathlib.Path) -> array:
    values = array("f")
    data = path.read_bytes()
    if len(data) % values.itemsize != 0:
        raise ValueError(f"{path} byte size is not divisible by float32 size")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def validate_f32_count(path: pathlib.Path, expected_count: int) -> None:
    expected_bytes = expected_count * FLOAT32_BYTES
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(f"{path} has {actual_bytes} bytes, expected {expected_bytes}")


def load_tokens_numpy(path: pathlib.Path, *, dtype=None):
    import numpy as np  # Imported lazily so the CLI can run with system Python.

    validate_f32_count(path, IMAGE_TOKEN_FLOAT_COUNT)
    tokens = np.fromfile(path, dtype="<f4").reshape(IMAGE_TOKEN_SHAPE)
    return tokens.astype(dtype) if dtype is not None else tokens


def load_encoded_numpy(path: pathlib.Path, *, dtype=None):
    import numpy as np  # Imported lazily so the CLI can run with system Python.

    validate_f32_count(path, ENCODED_FLOAT_COUNT)
    encoded = np.fromfile(path, dtype="<f4").reshape(ENCODED_SHAPE)
    return encoded.astype(dtype) if dtype is not None else encoded


def compare_f32(actual_path: pathlib.Path, golden_path: pathlib.Path) -> TensorMetrics:
    actual = read_f32(actual_path)
    golden = read_f32(golden_path)
    if len(actual) != len(golden):
        raise ValueError(f"length mismatch actual={len(actual)} golden={len(golden)}")

    max_abs = 0.0
    sum_abs = 0.0
    sum_sq = 0.0
    for actual_value, golden_value in zip(actual, golden, strict=True):
        diff = float(actual_value) - float(golden_value)
        abs_diff = abs(diff)
        max_abs = max(max_abs, abs_diff)
        sum_abs += abs_diff
        sum_sq += diff * diff

    count = len(actual)
    return TensorMetrics(
        count=count,
        max_abs=max_abs,
        rms=math.sqrt(sum_sq / count),
        mean_abs=sum_abs / count,
    )


def format_metrics(name: str, metrics: TensorMetrics) -> str:
    return (
        f"{name} count={metrics.count} max_abs={metrics.max_abs:.9g} "
        f"rms={metrics.rms:.9g} mean_abs={metrics.mean_abs:.9g}"
    )


def parse_response(line: str) -> tuple[str, dict[str, str]]:
    parts = line.strip().split()
    if not parts:
        raise ValueError("empty server response")
    status = parts[0]
    values = {}
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        values[key] = value
    return status, values


def build_run_command(
    binary: pathlib.Path,
    bundle: pathlib.Path,
    tokens_out: pathlib.Path,
    encoded_out: pathlib.Path,
    config: BridgeConfig,
) -> list[str]:
    command = [
        str(binary),
        "--bundle",
        str(bundle),
        "--attention",
        config.attention,
        "--warmup",
        str(config.warmup),
        "--runs",
        str(config.runs),
        "--tokens_out",
        str(tokens_out),
        "--encoded_out",
        str(encoded_out),
    ]
    if config.image_f32 is not None:
        command.extend(["--image", str(config.image_f32)])
    if config.image_ppm is not None:
        command.extend(["--image_ppm", str(config.image_ppm)])
    return command


def build_server_command(binary: pathlib.Path, bundle: pathlib.Path, config: ServerConfig) -> list[str]:
    return [
        str(binary),
        "--bundle",
        str(bundle),
        "--attention",
        config.attention,
        "--warmup",
        str(config.warmup),
        "--runs",
        str(config.runs),
    ]


def write_manifest(result: BridgeResult, config: BridgeConfig) -> None:
    result.manifest_path.write_text(
        json.dumps(
            {
                "format": "openpi_gemma_cpp_image_encoder_bridge_v1",
                "command": result.command,
                "binary": str(result.binary),
                "bundle": str(result.bundle),
                "image_f32": str(config.image_f32) if config.image_f32 is not None else "",
                "image_ppm": str(config.image_ppm) if config.image_ppm is not None else "",
                "attention": config.attention,
                "warmup": config.warmup,
                "runs": config.runs,
                "tokens_out": str(result.tokens_out),
                "tokens_dtype": FLOAT32_DTYPE,
                "tokens_shape": list(IMAGE_TOKEN_SHAPE),
                "tokens_out_bytes": result.tokens_out.stat().st_size,
                "encoded_out": str(result.encoded_out),
                "encoded_dtype": FLOAT32_DTYPE,
                "encoded_shape": list(ENCODED_SHAPE),
                "encoded_out_bytes": result.encoded_out.stat().st_size,
                "encoded_metrics": result.encoded_metrics.to_json() if result.encoded_metrics else None,
                "tokens_metrics": result.tokens_metrics.to_json() if result.tokens_metrics else None,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run_bridge(config: BridgeConfig, *, echo_commands: bool = False) -> BridgeResult:
    if config.image_f32 is not None and config.image_ppm is not None:
        raise ValueError("image_f32 and image_ppm are mutually exclusive")

    gemma_cpp_root = expand_path(config.gemma_cpp_root)
    bundle = expand_path(config.bundle)
    output_dir = expand_path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.binary is not None:
        binary = expand_path(config.binary)
    else:
        binary = gemma_cpp_root / "build/openpi_image_encoder_hip_run"

    if config.build:
        build_script = gemma_cpp_root / "scripts/build_openpi_image_encoder_hip_run.sh"
        if echo_commands:
            print(shlex.join([str(build_script)]), flush=True)
        build_result = subprocess.run([str(build_script)], check=False)
        if build_result.returncode != 0:
            raise subprocess.CalledProcessError(build_result.returncode, [str(build_script)])

    if not binary.exists():
        raise FileNotFoundError(f"missing gemma.cpp bridge binary: {binary}")

    tokens_out = output_dir / "gemma_cpp_tokens.f32"
    encoded_out = output_dir / "gemma_cpp_encoded.f32"
    command = build_run_command(binary, bundle, tokens_out, encoded_out, config)
    if echo_commands:
        print(shlex.join(command), flush=True)
    run_result = subprocess.run(command, check=False)
    if run_result.returncode != 0:
        raise subprocess.CalledProcessError(run_result.returncode, command)
    validate_f32_count(tokens_out, IMAGE_TOKEN_FLOAT_COUNT)
    validate_f32_count(encoded_out, ENCODED_FLOAT_COUNT)

    encoded_metrics = None
    tokens_metrics = None
    if config.compare:
        encoded_metrics = compare_f32(encoded_out, bundle / "golden_intermediate_encoded.f32")
        tokens_metrics = compare_f32(tokens_out, bundle / "golden_tokens.f32")

    result = BridgeResult(
        command=command,
        binary=binary,
        bundle=bundle,
        tokens_out=tokens_out,
        encoded_out=encoded_out,
        manifest_path=output_dir / "bridge_run_manifest.json",
        encoded_metrics=encoded_metrics,
        tokens_metrics=tokens_metrics,
    )
    write_manifest(result, config)
    return result


class ImageEncoderServer:
    def __init__(self, config: ServerConfig, *, echo_commands: bool = False):
        self._config = config
        self._echo_commands = echo_commands
        self._process: subprocess.Popen[str] | None = None
        self._binary: pathlib.Path | None = None
        self._bundle: pathlib.Path | None = None

    def __enter__(self) -> "ImageEncoderServer":
        self.start()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def start(self) -> None:
        if self._process is not None:
            return

        gemma_cpp_root = expand_path(self._config.gemma_cpp_root)
        self._bundle = expand_path(self._config.bundle)
        if self._config.binary is not None:
            self._binary = expand_path(self._config.binary)
        else:
            self._binary = gemma_cpp_root / "build/openpi_image_encoder_hip_server"

        if self._config.build:
            build_script = gemma_cpp_root / "scripts/build_openpi_image_encoder_hip_server.sh"
            if self._echo_commands:
                print(shlex.join([str(build_script)]), flush=True)
            build_result = subprocess.run([str(build_script)], check=False)
            if build_result.returncode != 0:
                raise subprocess.CalledProcessError(build_result.returncode, [str(build_script)])

        if not self._binary.exists():
            raise FileNotFoundError(f"missing gemma.cpp resident server binary: {self._binary}")

        command = build_server_command(self._binary, self._bundle, self._config)
        if self._echo_commands:
            print(shlex.join(command), flush=True)
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        ready = self._readline()
        status, _ = parse_response(ready)
        if status != "READY":
            self.close()
            raise RuntimeError(f"unexpected server startup response: {ready.strip()}")

    def close(self) -> None:
        if self._process is None:
            return
        process = self._process
        self._process = None
        if process.stdin is not None and process.poll() is None:
            process.stdin.write("QUIT\n")
            process.stdin.flush()
        if process.stdout is not None and process.poll() is None:
            process.stdout.readline()
        process.wait(timeout=10)

    def run(
        self,
        *,
        tokens_out: pathlib.Path,
        encoded_out: pathlib.Path,
        image_f32: pathlib.Path | None = None,
        image_ppm: pathlib.Path | None = None,
        attention: str | None = None,
        warmup: int | None = None,
        runs: int | None = None,
    ) -> ServerRunResult:
        if image_f32 is not None and image_ppm is not None:
            raise ValueError("image_f32 and image_ppm are mutually exclusive")
        if self._process is None:
            self.start()

        tokens_out.parent.mkdir(parents=True, exist_ok=True)
        encoded_out.parent.mkdir(parents=True, exist_ok=True)
        parts = [
            "RUN",
            f"tokens_out={tokens_out}",
            f"encoded_out={encoded_out}",
        ]
        if image_f32 is not None:
            parts.append(f"image_f32={image_f32}")
        if image_ppm is not None:
            parts.append(f"image_ppm={image_ppm}")
        if attention is not None:
            parts.append(f"attention={attention}")
        if warmup is not None:
            parts.append(f"warmup={warmup}")
        if runs is not None:
            parts.append(f"runs={runs}")

        command = " ".join(parts)
        if self._echo_commands:
            print(command, flush=True)
        process = self._require_process()
        process.stdin.write(command + "\n")
        process.stdin.flush()
        line = self._readline()
        status, response = parse_response(line)
        if status != "OK":
            raise RuntimeError(f"server request failed: {line.strip()}")
        validate_f32_count(tokens_out, IMAGE_TOKEN_FLOAT_COUNT)
        validate_f32_count(encoded_out, ENCODED_FLOAT_COUNT)
        return ServerRunResult(
            command=command,
            response=response,
            tokens_out=tokens_out,
            encoded_out=encoded_out,
        )

    def _require_process(self) -> subprocess.Popen[str]:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("server is not running")
        return self._process

    def _readline(self) -> str:
        process = self._require_process()
        if process.stdout is None:
            raise RuntimeError("server stdout is not available")
        line = process.stdout.readline()
        if not line:
            raise RuntimeError("server exited without response")
        return line


def assert_within_tolerance(
    result: BridgeResult,
    *,
    encoded_max_abs_tol: float,
    tokens_max_abs_tol: float,
) -> None:
    if result.encoded_metrics is None or result.tokens_metrics is None:
        raise ValueError("strict tolerance check requires comparison metrics")
    if result.encoded_metrics.max_abs > encoded_max_abs_tol:
        raise ValueError(
            f"encoded max_abs {result.encoded_metrics.max_abs} exceeds tolerance {encoded_max_abs_tol}"
        )
    if result.tokens_metrics.max_abs > tokens_max_abs_tol:
        raise ValueError(f"tokens max_abs {result.tokens_metrics.max_abs} exceeds tolerance {tokens_max_abs_tol}")
