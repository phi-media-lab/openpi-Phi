import dataclasses
import hashlib
import json
import pathlib
from typing import Any

import numpy as np
import tyro


@dataclasses.dataclass
class Args:
    golden: str = "build/openpi_golden/pi05_aloha_base_0_rgb_image_encoder.npz"
    weights: str = "build/openpi_golden/pi05_base_image_encoder_weights_bfloat16_all_layers.npz"
    output_dir: str = "build/openpi_golden/pi05_base_image_encoder_raw_bundle"


def load_npz(path: pathlib.Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def parse_json_scalar(array: np.ndarray) -> dict[str, Any]:
    return json.loads(str(array.item()))


def sha256_file(path: pathlib.Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_array(output_dir: pathlib.Path, name: str, array: np.ndarray) -> dict[str, Any]:
    array = np.asarray(array, dtype="<f4")
    path = output_dir / f"{name}.f32"
    array.tofile(path)
    return {
        "name": name,
        "file": path.name,
        "dtype": "float32_le",
        "shape": list(array.shape),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main(args: Args) -> None:
    golden_path = pathlib.Path(args.golden)
    weights_path = pathlib.Path(args.weights)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    golden = load_npz(golden_path)
    weights = load_npz(weights_path)
    golden_metadata = parse_json_scalar(golden["metadata_json"])
    weight_manifest = parse_json_scalar(weights["manifest_json"])

    arrays = []
    for name in sorted(key for key in weights if key != "manifest_json"):
        arrays.append(write_array(output_dir, f"weight_{name}", weights[name]))
    for name in (
        "image",
        "tokens",
        "intermediate_stem",
        "intermediate_with_posemb",
        "intermediate_encoder_block00_sa",
        "intermediate_encoder_block00_plus_sa",
        "intermediate_encoder_block00_mlp",
        "intermediate_encoder_block00_plus_mlp",
        "intermediate_encoded",
        "intermediate_logits_2d",
    ):
        if name in golden:
            arrays.append(write_array(output_dir, f"golden_{name}", golden[name]))

    manifest = {
        "format": "openpi_image_encoder_raw_bundle_v1",
        "source_golden": str(golden_path),
        "source_golden_sha256": sha256_file(golden_path),
        "source_weights": str(weights_path),
        "source_weights_sha256": sha256_file(weights_path),
        "golden_metadata": golden_metadata,
        "weight_manifest": weight_manifest,
        "arrays": arrays,
        "notes": [
            "All binary files are contiguous little-endian float32 arrays.",
            "Weight values reflect the OpenPI runtime bfloat16 restore path but are stored as float32 for portability.",
            "This bundle is intended for C++ smoke tests before adding a native Orbax/NPZ reader.",
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {output_dir}")
    print(f"manifest {manifest_path}")
    print(f"array_count {len(arrays)}")
    print(f"total_bytes {sum(item['bytes'] for item in arrays)}")


if __name__ == "__main__":
    main(tyro.cli(Args))
