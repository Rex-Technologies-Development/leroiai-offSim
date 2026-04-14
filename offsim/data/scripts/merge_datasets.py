"""Merge sim-generated + match data into one H5 dataset for offline RL."""

import argparse
import numpy as np
import d3rlpy


def merge(paths: list[str], output: str):
    datasets = []
    for p in paths:
        print(f"Loading {p}...")
        datasets.append(d3rlpy.dataset.MDPDataset.load(p))

    combined = d3rlpy.dataset.MDPDataset(
        observations=np.vstack([d.observations for d in datasets]),
        actions=np.hstack([d.actions for d in datasets]),
        rewards=np.hstack([d.rewards for d in datasets]),
        terminals=np.hstack([d.terminals for d in datasets]),
    )
    combined.dump(output)
    total = sum(len(d.observations) for d in datasets)
    print(f"Merged {total} transitions from {len(datasets)} files -> {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help=".h5 dataset files")
    parser.add_argument("--output", default="data/combined.h5")
    args = parser.parse_args()
    merge(args.inputs, args.output)
