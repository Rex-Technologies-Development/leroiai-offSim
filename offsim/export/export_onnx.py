"""Export trained SB3 policy to ONNX for Jetson deployment."""

from __future__ import annotations
import os
import sys
import numpy as np
import torch
import onnx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from sim.config import STATE_DIM


class SB3PolicyWrapper(torch.nn.Module):
    """Extracts just the actor forward pass for ONNX export."""

    def __init__(self, sb3_policy):
        super().__init__()
        self.features_extractor = sb3_policy.features_extractor
        self.mlp_extractor = sb3_policy.mlp_extractor
        self.action_net = sb3_policy.action_net

    def forward(self, state):
        features = self.features_extractor(state)
        latent_pi, _ = self.mlp_extractor(features)
        return self.action_net(latent_pi)


def export_to_onnx(model_path: str, output_path: str):
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    policy = model.policy
    policy.eval()

    input_dim = STATE_DIM * 2  # concatenated obs for both robots
    dummy = torch.randn(1, input_dim, dtype=torch.float32)

    wrapper = SB3PolicyWrapper(policy)
    wrapper.eval()

    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        wrapper, dummy, output_path,
        input_names=["state"],
        output_names=["action_logits"],
        dynamic_axes={"state": {0: "batch"}, "action_logits": {0: "batch"}},
        opset_version=17,
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX check passed. Saved to {output_path}")
