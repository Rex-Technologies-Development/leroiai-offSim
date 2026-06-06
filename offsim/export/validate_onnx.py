"""Validate ONNX model: load, run dummy input, compare to PyTorch output."""

from __future__ import annotations
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.config import STATE_DIM


def validate(onnx_path: str, model_path: str | None = None):
    import onnxruntime as ort

    input_dim = STATE_DIM  # single-robot obs, matches the exported model
    test_input = np.random.randn(1, input_dim).astype(np.float32)

    print(f"Loading ONNX model: {onnx_path}")
    session = ort.InferenceSession(onnx_path)
    onnx_result = session.run(None, {"state": test_input})[0]
    print(f"ONNX output shape: {onnx_result.shape}")
    print(f"ONNX output sample: {onnx_result[0][:5]}...")

    if model_path:
        import torch
        from sb3_contrib import MaskablePPO
        from export.export_onnx import SB3PolicyWrapper

        model = MaskablePPO.load(model_path)
        wrapper = SB3PolicyWrapper(model.policy)
        wrapper.eval()

        with torch.no_grad():
            pt_result = wrapper(torch.from_numpy(test_input)).numpy()

        diff = np.abs(onnx_result - pt_result).max()
        print(f"PyTorch output sample: {pt_result[0][:5]}...")
        print(f"Max absolute difference: {diff:.2e}")
        if diff < 1e-5:
            print("PASS: ONNX and PyTorch outputs match.")
        else:
            print("WARNING: Outputs differ. Check export.")
    else:
        print("No PyTorch model provided — skipping parity check.")
        print("PASS: ONNX inference works.")
