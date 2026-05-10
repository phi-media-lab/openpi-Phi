from array import array

import pytest

from openpi.shared import gemma_cpp_image_encoder_bridge as bridge


def write_f32(path, values):
    data = array("f", values)
    path.write_bytes(data.tobytes())


def test_compare_f32_metrics(tmp_path):
    actual = tmp_path / "actual.f32"
    golden = tmp_path / "golden.f32"
    write_f32(actual, [1.0, 2.0, 4.0])
    write_f32(golden, [1.0, 3.0, 2.0])

    metrics = bridge.compare_f32(actual, golden)

    assert metrics.count == 3
    assert metrics.max_abs == 2.0
    assert metrics.mean_abs == pytest.approx(1.0)
    assert metrics.rms == pytest.approx((5.0 / 3.0) ** 0.5)


def test_assert_within_tolerance_requires_metrics(tmp_path):
    result = bridge.BridgeResult(
        command=[],
        binary=tmp_path / "bin",
        bundle=tmp_path / "bundle",
        tokens_out=tmp_path / "tokens.f32",
        encoded_out=tmp_path / "encoded.f32",
        manifest_path=tmp_path / "manifest.json",
        encoded_metrics=None,
        tokens_metrics=None,
    )

    with pytest.raises(ValueError, match="requires comparison metrics"):
        bridge.assert_within_tolerance(
            result,
            encoded_max_abs_tol=1.0,
            tokens_max_abs_tol=1.0,
        )


def test_validate_f32_count(tmp_path):
    path = tmp_path / "values.f32"
    write_f32(path, [1.0, 2.0])

    bridge.validate_f32_count(path, 2)
    with pytest.raises(ValueError, match="expected 12"):
        bridge.validate_f32_count(path, 3)
