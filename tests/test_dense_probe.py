"""Opt-in prefix execution against the pinned local MiMo BF16 snapshot."""

import json
import math
import os
import subprocess
from pathlib import Path

import pytest

from embedded_jev.inventory import MODEL_REVISION


@pytest.mark.skipif(
    os.environ.get("MIMO_DENSE_PREFIX_TEST") != "1",
    reason="requires isolated CPU Torch/Transformers and the verified local MiMo snapshot",
)
def test_local_dense_text_prefix_captures_real_ffn_input():
    interpreter = os.environ.get("MIMO_DENSE_PYTHON")
    local_dir = os.environ.get("MIMO_LOCAL_DIR")
    if not interpreter or not local_dir or not Path(interpreter).is_file():
        pytest.fail("set MIMO_DENSE_PYTHON and MIMO_LOCAL_DIR for the opt-in prefix test")
    result = subprocess.run(
        [interpreter, "-m", "embedded_jev.dense_probe", "--local-dir", local_dir,
         "--layers", "4", "--compare-ternary"],
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
             "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "4"},
        check=True, capture_output=True, text=True, timeout=120,
    )
    report = json.loads(result.stdout)
    assert report["revision"] == MODEL_REVISION
    assert report["layers"] == 4 and report["tokens"] <= 32
    assert report["weight_bytes"] == 3764136064
    assert report["output_shape"] == [1, report["tokens"], 4096]
    assert report["output_dtype"] == "torch.bfloat16"
    assert report["ffn_down_input_shape"] == [1, report["tokens"], 12288]
    assert len(report["ffn_down_input_sha256"]) == 64
    assert math.isfinite(report["ffn_down_input_rms"])
    comparison = report["ternary_comparison"]
    assert comparison["policy"] == "unrotated_searched_fp16_group128"
    assert comparison["activation_quantized"] is False
    assert math.isfinite(comparison["relative_output_rmse"])
    assert math.isfinite(comparison["weight_mse"])