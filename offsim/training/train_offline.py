"""Override dataset metadata helpers.

Offline CQL/IQL training is intentionally not exposed: d3rlpy's discrete APIs do
not directly represent this prototype's centralized MultiDiscrete([10, 10]) action.
"""
from __future__ import annotations
import json, os
from datetime import datetime, timezone
import numpy as np
try:
    from ..sim.config import ACTION_NAMES, NUM_ACTIONS, TEAM_STATE_DIM
except ImportError:  # direct top-level import through offsim/main.py
    from sim.config import ACTION_NAMES, NUM_ACTIONS, TEAM_STATE_DIM

ACTION_ENUM_VERSION="override_v1"; REWARD_VERSION="override_score_delta_v1"; FIELD_CONFIG_VERSION="override_proxy_v1"
def _meta_path(path): return os.path.splitext(path)[0]+".meta.json"
def build_dataset_metadata(n_transitions):
    return {"action_enum_version":ACTION_ENUM_VERSION,"reward_version":REWARD_VERSION,"field_config_version":FIELD_CONFIG_VERSION,"state_dim":TEAM_STATE_DIM,"action_branches":[NUM_ACTIONS,NUM_ACTIONS],"action_names":[ACTION_NAMES[i] for i in range(NUM_ACTIONS)],"n_transitions":int(n_transitions),"created_at":datetime.now(timezone.utc).isoformat()}
def save_dataset(path,observations,actions,rewards,terminals):
    os.makedirs(os.path.dirname(path) or ".",exist_ok=True); np.savez(path,observations=np.asarray(observations,np.float32),actions=np.asarray(actions,np.int64),rewards=np.asarray(rewards,np.float32),terminals=np.asarray(terminals,np.float32))
    with open(_meta_path(path),"w",encoding="utf-8") as f: json.dump(build_dataset_metadata(len(observations)),f,indent=2)
def train(*args,**kwargs):
    raise NotImplementedError("Offline RL is not implemented for Override's centralized MultiDiscrete action space; use MaskablePPO training.")
