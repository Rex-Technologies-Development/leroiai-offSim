"""Phase 3: CQL/IQL offline RL on recorded match data via d3rlpy.

Dataset metadata
----------------
Every .npz produced by a data-collection run must have a sibling .meta.json
written by `save_dataset()`. The metadata pins the data to a specific
action enum, state dimension, and reward function — loading raises if the
current env doesn't match, so we never silently train on stale rollouts
from before a refactor.
"""

from __future__ import annotations
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import d3rlpy


# ---------------------------------------------------------------------------
# Dataset metadata — pin every recorded dataset to the env that produced it
# ---------------------------------------------------------------------------

# Bump these when their respective contracts change.
ACTION_ENUM_VERSION   = "v3"           # bumped when Action enum members or order change
REWARD_VERSION        = "winning_v1"   # bumped when REWARD_WEIGHTS / components change
FIELD_CONFIG_VERSION  = "push_back_v1" # bumped when field geometry / shared_config.yaml changes


def _meta_path(npz_path: str) -> str:
    """Sibling metadata path for a given .npz dataset."""
    base, _ = os.path.splitext(npz_path)
    return base + ".meta.json"


def build_dataset_metadata(n_transitions: int) -> dict:
    """Build the metadata dict for a dataset produced by the *current* env.

    Captures everything needed to detect drift between the recorded dataset
    and the env at training time.
    """
    from sim.config import Action, ACTION_NAMES, NUM_ACTIONS, STATE_DIM
    return {
        "action_enum_version":  ACTION_ENUM_VERSION,
        "reward_version":       REWARD_VERSION,
        "field_config_version": FIELD_CONFIG_VERSION,
        "state_dim":            int(STATE_DIM),
        "num_actions":          int(NUM_ACTIONS),
        "action_names":         [ACTION_NAMES[i] for i in range(NUM_ACTIONS)],
        "n_transitions":        int(n_transitions),
        "created_at":           datetime.now(timezone.utc).isoformat(),
    }


def save_dataset(
    npz_path: str,
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    terminals: np.ndarray,
) -> None:
    """Save a transitions .npz with a sibling .meta.json snapshot.

    Use this from any data-collection script — never bypass it, or downstream
    training won't know whether the data is still valid for the current env.
    """
    os.makedirs(os.path.dirname(npz_path) or ".", exist_ok=True)
    np.savez(
        npz_path,
        observations=observations.astype(np.float32),
        actions=actions.astype(np.int64),
        rewards=rewards.astype(np.float32),
        terminals=terminals.astype(np.float32),
    )
    meta = build_dataset_metadata(n_transitions=len(observations))
    with open(_meta_path(npz_path), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[save_dataset] wrote {npz_path}  ({meta['n_transitions']} transitions)")
    print(f"[save_dataset] wrote {_meta_path(npz_path)}")


def verify_dataset_compatibility(npz_path: str, strict: bool = True) -> dict:
    """Load and validate the .meta.json sidecar for an .npz dataset.

    Returns the metadata dict on success.  Raises ValueError on mismatch
    when strict=True; prints a warning otherwise.
    """
    from sim.config import Action, NUM_ACTIONS, STATE_DIM

    meta_path = _meta_path(npz_path)
    if not os.path.exists(meta_path):
        msg = f"No metadata at {meta_path} — dataset predates the metadata scheme."
        if strict:
            raise ValueError(
                msg + "\nRe-record with save_dataset() or pass --no-strict to bypass."
            )
        print(f"[verify_dataset] WARN: {msg}")
        return {}

    with open(meta_path) as f:
        meta = json.load(f)

    expected = {
        "action_enum_version":  ACTION_ENUM_VERSION,
        "reward_version":       REWARD_VERSION,
        "field_config_version": FIELD_CONFIG_VERSION,
        "state_dim":            int(STATE_DIM),
        "num_actions":          int(NUM_ACTIONS),
        "action_names":         [a.name for a in Action],
    }
    mismatches = [
        f"  {k}: dataset={meta.get(k)!r}  current={v!r}"
        for k, v in expected.items()
        if meta.get(k) != v
    ]
    if mismatches:
        msg = (
            f"Dataset {npz_path} is incompatible with the current env:\n"
            + "\n".join(mismatches)
        )
        if strict:
            raise ValueError(msg + "\nRe-record the dataset or pass --no-strict to bypass.")
        print(f"[verify_dataset] WARN:\n{msg}")
    return meta


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def collect_data_paths(inputs: list[str]) -> list[str]:
    paths = []
    for p in inputs:
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                if f.endswith((".npz", ".h5")):
                    paths.append(os.path.join(p, f))
        else:
            paths.append(p)
    return paths


def load_npz_files(paths: list[str], strict_metadata: bool = True):
    """Load and concatenate .npz datasets, verifying metadata for each."""
    all_obs, all_act, all_rew, all_term = [], [], [], []
    for p in paths:
        print(f"Loading {p}...")
        verify_dataset_compatibility(p, strict=strict_metadata)
        data = np.load(p)
        all_obs.append(data["observations"])
        all_act.append(data["actions"])
        all_rew.append(data["rewards"])
        all_term.append(data["terminals"])
    return (
        np.concatenate(all_obs).astype(np.float32),
        np.concatenate(all_act).astype(np.int64),
        np.concatenate(all_rew).astype(np.float32),
        np.concatenate(all_term).astype(np.float32),
    )


def train(
    data: list[str],
    algo: str = "cql",
    n_epochs: int = 100,
    steps_per_epoch: int = 1000,
    batch_size: int = 256,
    lr: float = 1e-4,
    save_interval: int = 10,
    device: str = "cuda:0",
    output_dir: str = "models",
    strict_metadata: bool = True,
):
    data_paths = collect_data_paths(data)
    if not data_paths:
        print("No data files found!")
        return

    h5_paths = [p for p in data_paths if p.endswith(".h5")]
    npz_paths = [p for p in data_paths if p.endswith(".npz")]

    datasets = []
    if h5_paths:
        for p in h5_paths:
            datasets.append(d3rlpy.dataset.MDPDataset.load(p))
    if npz_paths:
        obs, act, rew, term = load_npz_files(npz_paths, strict_metadata=strict_metadata)
        datasets.append(d3rlpy.dataset.MDPDataset(
            observations=obs, actions=act, rewards=rew, terminals=term,
        ))

    if len(datasets) > 1:
        dataset = d3rlpy.dataset.MDPDataset(
            observations=np.vstack([d.observations for d in datasets]),
            actions=np.hstack([d.actions for d in datasets]),
            rewards=np.hstack([d.rewards for d in datasets]),
            terminals=np.hstack([d.terminals for d in datasets]),
        )
    else:
        dataset = datasets[0]

    print(f"Total transitions: {len(dataset.observations)}")

    # d3rlpy ultimately uses PyTorch under the hood. If the environment has a
    # CPU-only torch build, requesting a CUDA device will either error later or
    # silently fall back depending on version. Detect early and fail loudly.
    if str(device).startswith("cuda"):
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"device='{device}' was requested, but torch.cuda.is_available() is False. "
                    f"Detected torch={getattr(torch, '__version__', 'unknown')}, torch.version.cuda={getattr(getattr(torch, 'version', None), 'cuda', None)}. "
                    "Install a CUDA-enabled PyTorch build or pass --device cpu."
                )

            # Guard for very new GPUs (e.g. sm_120) not included in the wheel.
            try:
                cap_major, cap_minor = torch.cuda.get_device_capability(0)
                arch = f"sm_{cap_major}{cap_minor}"
                arch_list = []
                try:
                    arch_list = list(torch.cuda.get_arch_list())
                except Exception:
                    arch_list = []
                if arch_list and arch not in arch_list:
                    raise RuntimeError(
                        "CUDA is available, but your GPU compute capability is not included in this PyTorch wheel. "
                        f"Detected GPU arch={arch}, torch supports: {', '.join(arch_list)}. "
                        "Install a newer CUDA PyTorch build (often CUDA 12.6 wheels or nightly builds), or pass --device cpu."
                    )
            except Exception:
                # If capability/arch list can't be queried, don't block training here.
                pass
        except ImportError:
            raise RuntimeError(
                f"device='{device}' was requested, but PyTorch is not installed. "
                "Install PyTorch (CUDA build if desired) or pass --device cpu."
            )

    if algo == "cql":
        model = d3rlpy.algos.DiscreteCQLConfig(
            learning_rate=lr, batch_size=batch_size,
        ).create(device=device)
    elif algo == "iql":
        model = d3rlpy.algos.DiscreteIQLConfig(
            learning_rate=lr, batch_size=batch_size,
        ).create(device=device)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    print(f"Training {algo.upper()} for {n_epochs} epochs...")
    model.fit(
        dataset,
        n_steps=n_epochs * steps_per_epoch,
        n_steps_per_epoch=steps_per_epoch,
        experiment_name=f"vex_{algo}",
        save_interval=save_interval,
    )

    out_path = os.path.join(output_dir, f"offline_{algo}_final.d3")
    model.save(out_path)
    print(f"Model saved to {out_path}")
