"""Phase 3: CQL/IQL offline RL on recorded match data via d3rlpy."""

from __future__ import annotations
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import d3rlpy


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


def load_npz_files(paths: list[str]):
    all_obs, all_act, all_rew, all_term = [], [], [], []
    for p in paths:
        print(f"Loading {p}...")
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
        obs, act, rew, term = load_npz_files(npz_paths)
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
