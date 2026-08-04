"""Validate Override ONNX actor inference and optional PyTorch parity."""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from ..sim.config import NUM_ACTIONS, TEAM_STATE_DIM
except ImportError:
    from sim.config import NUM_ACTIONS, TEAM_STATE_DIM

def validate(onnx_path: str, model_path: str | None = None):
    import onnxruntime as ort
    sample=np.zeros((1,TEAM_STATE_DIM),dtype=np.float32); session=ort.InferenceSession(onnx_path)
    result=session.run(None,{session.get_inputs()[0].name:sample})[0]
    expected=(1,NUM_ACTIONS*2)
    if result.shape != expected: raise ValueError(f"expected logits {expected}, got {result.shape}")
    if model_path:
        import torch
        from sb3_contrib import MaskablePPO
        try:
            from .export_onnx import SB3PolicyWrapper
        except ImportError:
            from export.export_onnx import SB3PolicyWrapper
        wrapped=SB3PolicyWrapper(MaskablePPO.load(model_path).policy).eval()
        with torch.no_grad(): reference=wrapped(torch.from_numpy(sample)).numpy()
        difference=float(np.max(np.abs(reference-result)))
        if difference >= 1e-5: raise ValueError(f"ONNX parity failed: max difference {difference}")
        print(f"PASS: shape={result.shape}, max difference={difference:.2e}")
    else: print(f"PASS: ONNX inference shape={result.shape}")
