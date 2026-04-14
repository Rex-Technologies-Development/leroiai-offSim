"""Convert .npz match logs to d3rlpy H5 dataset format."""

import argparse
import os
import numpy as np
import d3rlpy


def convert(npz_paths: list[str], output: str):
    all_obs, all_act, all_rew, all_term = [], [], [], []
    for p in npz_paths:
        print(f"Loading {p}...")
        data = np.load(p)
        all_obs.append(data["observations"])
        all_act.append(data["actions"])
        all_rew.append(data["rewards"])
        all_term.append(data["terminals"])

    dataset = d3rlpy.dataset.MDPDataset(
        observations=np.concatenate(all_obs).astype(np.float32),
        actions=np.concatenate(all_act).astype(np.int64),
        rewards=np.concatenate(all_rew).astype(np.float32),
        terminals=np.concatenate(all_term).astype(np.float32),
    )
    dataset.dump(output)
    print(f"Saved {len(dataset.observations)} transitions to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help=".npz files or directories")
    parser.add_argument("--output", default="data/matches/combined.h5")
    args = parser.parse_args()

    paths = []
    for p in args.inputs:
        if os.path.isdir(p):
            paths.extend(os.path.join(p, f) for f in sorted(os.listdir(p)) if f.endswith(".npz"))
        else:
            paths.append(p)

    convert(paths, args.output)
