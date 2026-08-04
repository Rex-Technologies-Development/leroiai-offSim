"""Export a centralized Override MaskablePPO actor to ONNX."""
from __future__ import annotations
import os, sys
import torch
import onnx
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sb3_contrib import MaskablePPO
try:
    from ..sim.config import TEAM_STATE_DIM
except ImportError:
    from sim.config import TEAM_STATE_DIM

class SB3PolicyWrapper(torch.nn.Module):
    def __init__(self, policy):
        super().__init__(); self.features_extractor=policy.features_extractor; self.mlp_extractor=policy.mlp_extractor; self.action_net=policy.action_net
    def forward(self, state):
        features=self.features_extractor(state); latent,_=self.mlp_extractor(features); return self.action_net(latent)

def export_to_onnx(model_path: str, output_path: str):
    model=MaskablePPO.load(model_path); wrapper=SB3PolicyWrapper(model.policy).eval(); dummy=torch.zeros(1,TEAM_STATE_DIM,dtype=torch.float32)
    os.makedirs(os.path.dirname(output_path) or ".",exist_ok=True)
    torch.onnx.export(wrapper,dummy,output_path,input_names=["team_state"],output_names=["action_logits"],dynamic_axes={"team_state":{0:"batch"},"action_logits":{0:"batch"}},opset_version=17)
    exported=onnx.load(output_path); onnx.checker.check_model(exported); print(f"saved {output_path}; input={TEAM_STATE_DIM}, output={exported.graph.output[0].name}")
