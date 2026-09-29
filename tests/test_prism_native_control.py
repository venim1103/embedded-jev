"""Optional pinned Prism CPU transform graph control without model weights."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


PRISM_REVISION = "842b1880415d6f508f03b789e5ce70194def7bfd"
BITNET_REVISION = "0b341e582afbf9e1011f24744b554c96a3477eb5"


def test_pinned_prism_cpu_fwht_matches_dense_signed_reference(tmp_path):
    source_dir = os.environ.get("PRISM_SOURCE_DIR")
    library_name = os.environ.get("PRISM_GGML_CPU_LIBRARY")
    if not source_dir or not library_name:
        pytest.skip("requires pinned Prism source and built CPU library")
    source = Path(source_dir)
    library = Path(library_name)
    if not source.is_dir() or not library.is_file():
        pytest.fail("missing pinned Prism source or CPU library")
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    assert revision == PRISM_REVISION
    compiler = shutil.which("clang++-18")
    if compiler is None:
        pytest.skip("clang++-18 is not installed")

    program = Path(__file__).resolve().parents[1] / "native" / "prism_hadamard_smoke.cpp"
    binary = tmp_path / "prism-hadamard-smoke"
    library_dir = library.parent
    subprocess.run(
        [
            compiler, "-std=c++17", "-O2", "-I", str(source / "ggml" / "include"),
            str(program), "-L", str(library_dir), f"-Wl,-rpath,{library_dir}",
            "-lggml-cpu", "-lggml-base", "-o", str(binary),
        ],
        check=True, capture_output=True, text=True,
    )
    result = subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True, timeout=15
    )
    report = json.loads(result.stdout.strip())
    assert report["tokens"] == 2 and report["width"] == 128
    assert report["max_abs_error"] < 1e-4


def test_pinned_prism_fwht_feeds_bitnet_derived_group_scale_kernel(tmp_path):
    prism_source = os.environ.get("PRISM_SOURCE_DIR")
    prism_library = os.environ.get("PRISM_GGML_CPU_LIBRARY")
    bitnet_source = os.environ.get("BITNET_SOURCE_DIR")
    if not prism_source or not prism_library or not bitnet_source:
        pytest.skip("requires pinned Prism and BitNet source plus Prism CPU library")
    revisions = ((prism_source, PRISM_REVISION), (bitnet_source, BITNET_REVISION))
    for directory, expected in revisions:
        revision = subprocess.check_output(
            ["git", "-C", directory, "rev-parse", "HEAD"], text=True
        ).strip()
        assert revision == expected
    compiler = shutil.which("clang++-18")
    if compiler is None:
        pytest.skip("clang++-18 is not installed")
    library_dir = Path(prism_library).parent
    sources = Path(__file__).resolve().parents[1] / "native"
    binary = tmp_path / "prism-bitnet-bridge"
    subprocess.run(
        [
            compiler, "-std=c++17", "-O2", "-mavx2",
            "-I", str(Path(prism_source) / "ggml" / "include"),
            str(sources / "prism_bitnet_bridge.cpp"),
            str(sources / "bitnet_group_scale.cpp"),
            "-L", str(library_dir), f"-Wl,-rpath,{library_dir}",
            "-lggml-cpu", "-lggml-base", "-o", str(binary),
        ],
        check=True, capture_output=True, text=True,
    )
    result = subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True, timeout=15
    )
    report = json.loads(result.stdout.strip())
    assert report["tokens"] == 2 and report["groups"] == 2
    assert report["gdn_v_grouped"] is False
    assert report["graph_op"] == "map_custom2"
    assert report["graph_evaluations"] == report["callback_calls"] == 2
    assert report["max_transform_error"] < 1e-4
    assert report["max_output_error"] < 0.005
    assert report["repeat_scale_error"] < 0.005
    grouped_result = subprocess.run(
        [str(binary), "--grouped-v"], check=True,
        capture_output=True, text=True, timeout=15,
    )
    grouped = json.loads(grouped_result.stdout.strip())
    assert grouped["gdn_v_grouped"] is True
    assert grouped["tokens"] == report["tokens"] == 2
    assert grouped["graph_evaluations"] == grouped["callback_calls"] == 2
    assert grouped["max_transform_error"] < 1e-4
    assert grouped["max_output_error"] < 0.005
    assert grouped["repeat_scale_error"] < 0.005


def test_pinned_prism_qwen35_filter_and_ssm_bias_map_mimo_text_names():
    source_dir = os.environ.get("PRISM_SOURCE_DIR")
    interpreter = os.environ.get("PRISM_CONVERTER_PYTHON")
    if not source_dir or not interpreter:
        pytest.skip("requires pinned Prism source and isolated GGUF Python environment")
    source = Path(source_dir)
    if not source.is_dir() or not Path(interpreter).is_file():
        pytest.fail("missing pinned Prism source or GGUF Python environment")
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    assert revision == PRISM_REVISION
    code = """
from __future__ import annotations
import ast
import json
import sys
from pathlib import Path
from gguf import MODEL_ARCH, get_tensor_name_map

source = Path(sys.argv[1]) / "conversion/base.py"
base_tree = ast.parse(source.read_text())
model = next(node for node in base_tree.body
             if isinstance(node, ast.ClassDef) and node.name == "ModelBase")
method = next(node for node in model.body
              if isinstance(node, ast.FunctionDef) and node.name == "filter_tensors")
filtered_class = ast.ClassDef(name="FilteredModel", bases=[], keywords=[],
                              body=[method], decorator_list=[])
namespace = {}
exec(compile(ast.fix_missing_locations(ast.Module(body=[filtered_class], type_ignores=[])),
             str(source), "exec"), namespace)
text_model = next(node for node in base_tree.body
                  if isinstance(node, ast.ClassDef) and node.name == "TextModel")
text_filter = next(node for node in text_model.body
                   if isinstance(node, ast.FunctionDef) and node.name == "filter_tensors")
filtered_text_class = ast.ClassDef(
    name="FilteredTextModel", bases=[ast.Name(id="FilteredModel", ctx=ast.Load())],
    keywords=[], body=[text_filter], decorator_list=[])
exec(compile(ast.fix_missing_locations(ast.Module(body=[filtered_text_class], type_ignores=[])),
             str(source), "exec"), namespace)
qwen_source = Path(sys.argv[1]) / "conversion/qwen.py"
qwen = next(node for node in ast.parse(qwen_source.read_text()).body
            if isinstance(node, ast.ClassDef) and node.name == "Qwen3NextModel")
modify = next(node for node in qwen.body
              if isinstance(node, ast.FunctionDef) and node.name == "modify_tensors")
bias_branch = next(node for node in ast.walk(modify)
                   if isinstance(node, ast.If) and ast.unparse(node.test) == "name.endswith('.dt_bias')")
rename = ast.parse("def rename_dt_bias(name):\\n    return name").body[0]
rename.body.insert(0, bias_branch.body[0])
exec(compile(ast.fix_missing_locations(ast.Module(body=[rename], type_ignores=[])),
             str(qwen_source), "exec"), namespace)
mapper = get_tensor_name_map(MODEL_ARCH.QWEN35, 32)
results = []
for name in sys.argv[2:]:
    filtered = namespace["FilteredTextModel"].filter_tensors((name, lambda: None))
    if filtered is None:
        results.append([None, None, None])
        continue
    mapped_name = (namespace["rename_dt_bias"](filtered[0])
                   if filtered[0].endswith(".dt_bias") else filtered[0])
    results.append([mapper.get_name(name, try_suffixes=(".weight", ".bias")),
                    filtered[0], mapper.get_name(mapped_name, try_suffixes=(".weight", ".bias"))])
print(json.dumps(results))
"""
    result = subprocess.run(
        [interpreter, "-c", code, str(source),
         "model.language_model.layers.3.mlp.down_proj.weight",
         "model.language_model.layers.3.self_attn.q_proj.weight",
         "model.language_model.layers.0.linear_attn.dt_bias",
         "model.visual.blocks.0.attn.qkv.weight"],
        env={**os.environ, "PYTHONPATH": str(source / "gguf-py")},
        check=True, capture_output=True, text=True, timeout=15,
    )
    assert json.loads(result.stdout) == [
        [None, "model.layers.3.mlp.down_proj.weight", "blk.3.ffn_down.weight"],
        [None, "model.layers.3.self_attn.q_proj.weight", "blk.3.attn_q.weight"],
        [None, "model.layers.0.linear_attn.dt_bias", "blk.0.ssm_dt.bias"],
        [None, None, None],
    ]

    from embedded_jev.inventory import _json_object, _open_bounded

    index_bytes, _ = _open_bounded("model.safetensors.index.json")
    names = sorted(_json_object(index_bytes, "model.safetensors.index.json")["weight_map"])
    assert len(names) == 760
    all_results = subprocess.run(
        [interpreter, "-c", code, str(source), *names],
        env={**os.environ, "PYTHONPATH": str(source / "gguf-py")},
        check=True, capture_output=True, text=True, timeout=15,
    )
    mapped = json.loads(all_results.stdout)
    assert len(mapped) == len(names)
    assert sum(result[1] is None for result in mapped) == 333
    assert sum(result[1] is not None for result in mapped) == 427
    assert all(result[2] is not None for result in mapped if result[1] is not None)
    renamed_biases = [
        result for result in mapped if result[1] is not None and result[1].endswith(".dt_bias")
    ]
    assert len(renamed_biases) == 24
    assert all(result[2].endswith(".ssm_dt.bias") for result in renamed_biases)


def test_pinned_prism_qwen35_value_head_reorder_on_toy_tensors():
    source_dir = os.environ.get("PRISM_SOURCE_DIR")
    interpreter = os.environ.get("PRISM_CONVERTER_PYTHON")
    if not source_dir or not interpreter:
        pytest.skip("requires pinned Prism converter and isolated CPU Torch environment")
    source = Path(source_dir)
    assert subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip() == PRISM_REVISION
    code = """
from __future__ import annotations
import ast
import json
import sys
from pathlib import Path
import numpy as np
import torch

module = ast.parse((Path(sys.argv[1]) / "conversion/qwen.py").read_text())
next_model = next(node for node in module.body
                  if isinstance(node, ast.ClassDef) and node.name == "Qwen3NextModel")
value_model = next(node for node in module.body
                   if isinstance(node, ast.ClassDef) and node.name == "_LinearAttentionVReorderBase")
next_modify = next(node for node in next_model.body
                   if isinstance(node, ast.FunctionDef) and node.name == "modify_tensors")
reorder = next(node for node in value_model.body
               if isinstance(node, ast.FunctionDef) and node.name == "_reorder_v_heads")
folded_check = next(node for node in value_model.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_hadamard_folds_tensor")
value_modify = next(node for node in value_model.body
                    if isinstance(node, ast.FunctionDef) and node.name == "modify_tensors")
base = ast.parse("class Base:\\n def modify_tensors(self, data_torch, name, bid):\\n  yield name, data_torch").body[0]
next_class = ast.ClassDef(name="Qwen3NextModel", bases=[ast.Name(id="Base", ctx=ast.Load())],
                          keywords=[], body=[next_modify], decorator_list=[])
value_class = ast.ClassDef(name="ValueModel", bases=[ast.Name(id="Qwen3NextModel", ctx=ast.Load())],
                           keywords=[], body=[reorder, folded_check, value_modify], decorator_list=[])
namespace = {"torch": torch, "np": np}
exec(compile(ast.fix_missing_locations(ast.Module(body=[base, next_class, value_class], type_ignores=[])),
             str(Path(sys.argv[1]) / "conversion/qwen.py"), "exec"), namespace)
model = namespace["ValueModel"]()
model.hparams = {"linear_num_key_heads": 2, "linear_num_value_heads": 4,
                 "linear_key_head_dim": 2, "linear_value_head_dim": 2, "hidden_size": 8}
model.hadamard_folded_names = lambda: []
model._hadamard_gdn_v_grouped = False
name = "model.layers.0.linear_attn."
qkv = torch.arange(16 * 8, dtype=torch.float32).reshape(16, 8)
result = list(model.modify_tensors(qkv, name + "in_proj_qkv.weight", 0))
assert len(result) == 1
expected_v = qkv[8:].numpy().reshape(2, 2, 2, 8).transpose(1, 0, 2, 3).reshape(8, 8)
assert np.array_equal(result[0][1].numpy(), np.concatenate([qkv[:8].numpy(), expected_v]))
gate = torch.arange(8 * 8, dtype=torch.float32).reshape(8, 8)
result_z = list(model.modify_tensors(gate, name + "in_proj_z.weight", 0))
assert np.array_equal(result_z[0][1].numpy(), gate.numpy().reshape(2, 2, 2, 8).transpose(1, 0, 2, 3).reshape(8, 8))
log_values = torch.arange(4, dtype=torch.float32)
result_log = list(model.modify_tensors(log_values, name + "A_log", 0))
expected_heads = np.array([0, 2, 1, 3], dtype=np.float32)
np.testing.assert_allclose(result_log[0][1].numpy(), -np.exp(expected_heads).astype(np.float32), rtol=1e-6, atol=1e-6)
result_bias = list(model.modify_tensors(log_values, name + "dt_bias", 0))
assert result_bias[0][0].endswith(".dt_proj.bias")
assert np.array_equal(result_bias[0][1].numpy(), expected_heads)
alpha = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)
result_alpha = list(model.modify_tensors(alpha, name + "in_proj_a.weight", 0))
expected_alpha = alpha.numpy().reshape(2, 2, 1, 8).transpose(1, 0, 2, 3).reshape(4, 8)
assert np.array_equal(result_alpha[0][1].numpy(), expected_alpha)
conv = torch.arange(16 * 3, dtype=torch.float32).reshape(16, 1, 3)
result_conv = list(model.modify_tensors(conv, name + "conv1d.weight", 0))
conv_values = conv.numpy().reshape(16, 3)
expected_conv_v = conv_values[8:].reshape(2, 2, 2, 3).transpose(1, 0, 2, 3).reshape(8, 3)
assert np.array_equal(result_conv[0][1].numpy(), np.concatenate([conv_values[:8], expected_conv_v]))
out = torch.arange(3 * 8, dtype=torch.float32).reshape(3, 8)
out_name = name + "out_proj.weight"
unfolded = list(model.modify_tensors(out, out_name, 0))
expected_columns = out.numpy().reshape(3, 2, 2, 2).transpose(0, 2, 1, 3).reshape(3, 8)
assert np.array_equal(unfolded[0][1].numpy(), expected_columns)
assert model._hadamard_gdn_v_grouped is False
model.hadamard_folded_names = lambda: [out_name]
folded = list(model.modify_tensors(out, out_name, 0))
assert np.array_equal(folded[0][1].numpy(), out.numpy())
assert model._hadamard_gdn_v_grouped is True
print(json.dumps({"qkv_shape": list(result[0][1].shape), "v_head_order": [0, 2, 1, 3]}))
"""
    result = subprocess.run(
        [interpreter, "-c", code, str(source)], check=True,
        capture_output=True, text=True, timeout=15,
    )
    assert json.loads(result.stdout) == {"qkv_shape": [16, 8], "v_head_order": [0, 2, 1, 3]}


def test_pinned_prism_grouped_v_metadata_writer_for_folded_out_projection():
    source_dir = os.environ.get("PRISM_SOURCE_DIR")
    interpreter = os.environ.get("PRISM_CONVERTER_PYTHON")
    if not source_dir or not interpreter:
        pytest.skip("requires pinned Prism converter and isolated GGUF Python environment")
    source = Path(source_dir)
    assert subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip() == PRISM_REVISION
    code = """
import ast
import json
import logging
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import gguf

source = Path(sys.argv[1]) / "conversion/base.py"
tree = ast.parse(source.read_text())
base = next(node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ModelBase")
method = next(node for node in base.body
              if isinstance(node, ast.FunctionDef) and node.name == "add_hadamard_metadata")
model_class = ast.ClassDef(name="Model", bases=[], keywords=[], body=[method], decorator_list=[])
namespace = {"json": json, "re": re, "gguf": gguf, "logger": logging.getLogger("prism")}
exec(compile(ast.fix_missing_locations(ast.Module(body=[model_class], type_ignores=[])),
             str(source), "exec"), namespace)
manifest = {
    "schema_version": 1, "kind": "hadamard-weight-fold",
    "status": "requires-matching-runtime",
    "transform": {"block_size": 128, "name": "normalized-signed-sylvester-walsh-hadamard",
                  "sign_mode": "identity"},
    "tensors": [{"name": "model.layers.0.linear_attn.out_proj.weight", "axis": -1}],
}
results = []
with tempfile.TemporaryDirectory() as directory:
    Path(directory, "hadamard_packing.json").write_text(json.dumps(manifest))
    for grouped in (False, True):
        metadata = {}
        bool_keys = []
        def put(key, value):
            metadata[key] = value
        def put_bool(key, value):
            bool_keys.append(key)
            put(key, value)
        model = namespace["Model"]()
        model.dir_model = Path(directory)
        model.model_arch = gguf.MODEL_ARCH.QWEN35
        model.hparams = {"tie_word_embeddings": False}
        model._hadamard_gdn_v_grouped = grouped
        model.filter_tensors = lambda tensor: tensor
        model.map_tensor_name = {
            "model.layers.0.linear_attn.out_proj.weight": "blk.0.ssm_out.weight"
        }.__getitem__
        model.gguf_writer = SimpleNamespace(
            add_bool=put_bool, add_uint32=put, add_string=put, add_array=put,
        )
        model.add_hadamard_metadata()
        assert metadata["prism.hadamard.weight_names"] == ["blk.0.ssm_out.weight"]
        key = "prism.hadamard.gdn_v_grouped"
        assert (key in bool_keys) == grouped
        assert metadata.get(key) is (True if grouped else None)
        gguf_path = Path(directory, f"grouped-{grouped}.gguf")
        writer = gguf.GGUFWriter(gguf_path, "qwen35")
        model.gguf_writer = writer
        model.add_hadamard_metadata()
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        reader = gguf.GGUFReader(gguf_path)
        assert not reader.tensors
        assert reader.get_field("prism.hadamard.weight_names").contents() == [
            "blk.0.ssm_out.weight"
        ]
        field = reader.get_field(key)
        assert (field is not None) == grouped
        if grouped:
            assert field.types[0].name == "BOOL"
            assert field.contents() is True
        results.append(grouped)
print(json.dumps(results))
"""
    result = subprocess.run(
        [interpreter, "-c", code, str(source)],
        env={**os.environ, "PYTHONPATH": str(source / "gguf-py")},
        check=True, capture_output=True, text=True, timeout=15,
    )
    assert json.loads(result.stdout) == [False, True]