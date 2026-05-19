"""VEX AI RL — single entry point for all operations.

Usage:
    python main.py demo                                    # visual sim with random actions
    python main.py train --timesteps 2000000               # PPO training with curriculum
    python main.py train-offline --data data/matches/      # offline RL (CQL/IQL)
    python main.py eval --model models/final_model         # evaluate a trained policy
    python main.py eval --model models/final_model --render # evaluate with Pygame
    python main.py export --model models/final_model       # export to ONNX
    python main.py validate --onnx models/onnx/model.onnx  # validate ONNX output
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def _greedy_demo_action(robot_idx: int, env) -> int:
    """Greedy demo policy: collect until full, then score nearest long goal.

    Priority:
      1. Full (MAX_CARRY balls) → SCORE_LONG_GOAL
      2. Accessible balls exist → COLLECT_NEAREST_BALL
      3. Holding any balls but none accessible → SCORE_LONG_GOAL
      4. No balls anywhere → IDLE
    """
    from sim.route_planner import compute_collection_route
    from sim.config import Action, MAX_CARRY

    robot = env.field.allies[robot_idx]

    if robot.balls_held >= MAX_CARRY:
        return int(Action.SCORE_LONG_GOAL)

    route = compute_collection_route(
        robot.position, env.field,
        already_held=robot.balls_held,
        max_volley=1,
        robot=robot,
    )
    if route:
        return int(Action.COLLECT_NEAREST_BALL)
    if robot.balls_held > 0:
        return int(Action.SCORE_LONG_GOAL)
    return int(Action.COLLECT_NEAREST_BALL)   # fallback: drive toward field to get LOS


def cmd_demo(args):
    """Run the sim with greedy (or random) actions and Pygame visualization."""
    from sim.env import VexAIEnv
    from sim.config import Action
    from sim.failure import FailureConfig

    env = VexAIEnv(
        render_mode="human",
        failure_config=FailureConfig.none(),
        opponent_type=args.opponent,
        num_allies=args.num_robots,
    )
    obs, _ = env.reset(seed=42)
    rng = np.random.default_rng(42)
    episode = 1
    total_reward = 0.0

    print(f"VEX Push Back Sim Demo — {args.num_robots} allied robot(s)")
    print("Controls: Space=pause  S=step  H=heatmap  R=reset  +/-=speed  1/2=robot")
    print("          Tab=WASD mode  (W/S=fwd/rev  A/D=turn  I=intake  F=score)")
    print("          Panel: RUN to run, AUTO PLAY for smart greedy policy")

    while True:
        env.render()
        renderer = env._renderer

        # Reset always takes priority
        if renderer and renderer.should_reset:
            renderer.should_reset = False
            obs, _ = env.reset()
            total_reward = 0.0
            episode += 1
            continue

        # WASD manual mode: one physics tick per render frame, no RL step
        if renderer and renderer.wasd_mode:
            if not renderer.paused or renderer.step_once:
                renderer.step_once = False
                env.manual_tick(renderer)
            continue

        if renderer and renderer.paused and not renderer.step_once:
            continue
        if renderer:
            renderer.step_once = False

        if renderer and renderer.demo_score:
            # Demo scoring: force SCORE_LONG_GOAL until robots have emptied all balls.
            from sim.config import Action
            actions = np.array([int(Action.SCORE_LONG_GOAL)] * 2)
            # Auto-exit demo mode once all robots have finished scoring
            all_done = all(env.field.allies[i].balls_held == 0
                           for i in range(env.num_allies))
            if all_done:
                renderer.demo_score = False
                renderer.paused = True
        elif renderer and renderer.auto_collect:
            # Greedy policy: collect → fill → score nearest goal → repeat
            actions = np.array([
                _greedy_demo_action(0, env),
                _greedy_demo_action(min(1, env.num_allies - 1), env),
            ])
        else:
            demo_actions = [int(Action.COLLECT_NEAREST_BALL), int(Action.SCORE_LONG_GOAL),
                            int(Action.SCORE_CENTER_MID), int(Action.SCORE_CENTER_LOW),
                            int(Action.IDLE)]
            actions = np.array([rng.choice(demo_actions) for _ in range(2)])

        obs, rewards, done, _, _ = env.step(actions)
        total_reward += rewards[0]
        if args.num_robots >= 2:
            total_reward += rewards[1]

        if done:
            print(f"Ep {episode}: Us={env.field.my_score} Opp={env.field.opponent_score} "
                  f"Reward={total_reward:.1f}")
            obs, _ = env.reset()
            total_reward = 0.0
            episode += 1

    env.close()


def cmd_train(args):
    """Phase 1: PPO training in sim with curriculum."""
    from training.train_sim import train
    train(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        lr=args.lr,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
        output_dir=args.output_dir,
        resume=args.resume,
        render=args.render,
        device=args.device,
    )


def cmd_train_offline(args):
    """Phase 3: Offline RL from match data."""
    from training.train_offline import train
    train(
        data=args.data,
        algo=args.algo,
        n_epochs=args.n_epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        save_interval=args.save_interval,
        device=args.device,
        output_dir=args.output_dir,
    )


def _resolve_model_path(arg: str, output_dir: str = "models") -> str:
    """Resolve --model arg. If 'latest', pick the freshest known model.

    Search order (newest first):
      1. models/interrupted_model.zip   (Ctrl+C save)
      2. models/final_model.zip         (completed training)
      3. models/checkpoints/*.zip       (most recent checkpoint by mtime)
      4. models/best/best_model.zip     (best by eval reward)
    """
    import glob
    if arg != "latest":
        return arg

    candidates = []
    for path in [
        os.path.join(output_dir, "interrupted_model.zip"),
        os.path.join(output_dir, "final_model.zip"),
    ]:
        if os.path.exists(path):
            candidates.append((os.path.getmtime(path), path))
    ckpts = sorted(
        glob.glob(os.path.join(output_dir, "checkpoints", "*.zip")),
        key=os.path.getmtime, reverse=True,
    )
    if ckpts:
        candidates.append((os.path.getmtime(ckpts[0]), ckpts[0]))
    best = os.path.join(output_dir, "best", "best_model.zip")
    if os.path.exists(best):
        candidates.append((os.path.getmtime(best), best))

    if not candidates:
        raise FileNotFoundError(
            f"No models found under {output_dir}/. Train a model first with "
            f"`python offsim/main.py train` or pass an explicit --model path."
        )
    candidates.sort(reverse=True)
    return candidates[0][1]


def cmd_eval(args):
    """Evaluate a trained single-agent policy."""
    from stable_baselines3 import PPO
    from sim.env import SingleAgentWrapper
    from sim.failure import FailureConfig

    model_path = _resolve_model_path(args.model)
    failure_config = FailureConfig(
        teammate_fail_rate=args.failure_rate,
        object_stolen_rate=args.failure_rate * 0.5,
        stuck_rate=args.failure_rate * 0.25,
        teammate_offline_rate=args.teammate_offline,
    )

    render_mode = "human" if args.render else None
    env = SingleAgentWrapper(
        render_mode=render_mode,
        failure_config=failure_config,
        opponent_type=args.opponent,
        num_allies=1,
        num_opponents=0,
        use_timer=True,
    )
    if render_mode:
        env.render()

    print(f"Loading model from {model_path}...")
    try:
        model = PPO.load(model_path)
    except ValueError as e:
        if "Observation spaces do not match" in str(e):
            print(f"\n[!] Observation-space mismatch loading {model_path}.")
            print(f"    Current env expects shape={env.observation_space.shape}.")
            print(f"    This model was trained with a different STATE_DIM (e.g. heatmap")
            print(f"    on/off or other observation changes). Retrain from scratch:")
            print(f"      python offsim/main.py train --timesteps 200000")
            raise SystemExit(1)
        raise

    wins, losses, ties = 0, 0, 0
    all_rewards, all_scores, all_opp = [], [], []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(int(action))
            ep_reward += float(reward)

            if render_mode:
                env.render()
                renderer = env.env._renderer
                if renderer and renderer.paused:
                    while renderer.paused and not renderer.step_once:
                        renderer.handle_events()
                    renderer.step_once = False

        inner_field = env.env.field
        diff = inner_field.my_score - inner_field.opponent_score
        result = "WIN" if diff > 0 else ("LOSS" if diff < 0 else "TIE")
        if diff > 0: wins += 1
        elif diff < 0: losses += 1
        else: ties += 1

        all_rewards.append(ep_reward)
        all_scores.append(inner_field.my_score)
        all_opp.append(inner_field.opponent_score)

        print(f"  Ep {ep+1:3d}: Us={inner_field.my_score:3d} Opp={inner_field.opponent_score:3d} "
              f"Reward={ep_reward:7.1f} {result}")

    print(f"\n{'='*50}")
    print(f"Results ({args.episodes} episodes, opponent={args.opponent}, "
          f"failure={args.failure_rate:.0%}):")
    print(f"  W/L/T: {wins}/{losses}/{ties}  Win rate: {wins/args.episodes*100:.1f}%")
    print(f"  Avg reward:  {np.mean(all_rewards):.1f} +/- {np.std(all_rewards):.1f}")
    print(f"  Avg score:   Us={np.mean(all_scores):.1f}  Opp={np.mean(all_opp):.1f}")
    print(f"  Avg margin:  {np.mean(all_scores)-np.mean(all_opp):.1f}")

    env.close()


def cmd_export(args):
    """Export trained model to ONNX."""
    from export.export_onnx import export_to_onnx
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    export_to_onnx(args.model, args.output)


def cmd_validate(args):
    """Validate ONNX model."""
    from export.validate_onnx import validate
    validate(args.onnx, args.model)


# ======================================================================
# Argument parser
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="VEX AI RL — Sim & Trainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- demo ---
    p = sub.add_parser("demo", help="Visual sim with random actions")
    p.add_argument("--opponent", default="mixed",
                   choices=["random", "greedy", "defensive", "mixed"])
    p.add_argument("--num-robots", type=int, default=1, choices=[1, 2],
                   help="Number of allied robots (default: 1)")
    p.set_defaults(func=cmd_demo)

    # --- train ---
    p = sub.add_parser("train", help="PPO training — 1 robot, 60s episodes")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--checkpoint-freq", type=int, default=10_000,
                   help="Save a checkpoint every N timesteps (default: 10000)")
    p.add_argument("--eval-freq", type=int, default=25_000)
    p.add_argument("--output-dir", default="models")
    p.add_argument("--resume", default=None, help="Resume from checkpoint path")
    p.add_argument("--render", action="store_true",
                   help="Show the pygame window while training. If n_envs>1, each SubprocVecEnv worker opens its own window.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="PyTorch device for PPO. 'auto' uses CUDA if available "
                        "(default: auto). Note: for this workload, env stepping "
                        "is the bottleneck so CUDA only gives a modest speedup.")
    p.set_defaults(func=cmd_train)

    # --- train-offline ---
    p = sub.add_parser("train-offline", help="Offline RL from match data")
    p.add_argument("--data", nargs="+", required=True,
                   help=".npz/.h5 files or directories")
    p.add_argument("--algo", choices=["cql", "iql"], default="cql")
    p.add_argument("--n-epochs", type=int, default=100)
    p.add_argument("--steps-per-epoch", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save-interval", type=int, default=10)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", default="models")
    p.set_defaults(func=cmd_train_offline)

    # --- eval ---
    p = sub.add_parser("eval", help="Evaluate trained policy")
    p.add_argument("--model", default="latest",
                   help="SB3 model path, or 'latest' to auto-pick the freshest "
                        "from models/ (default: latest)")
    p.add_argument("--episodes", type=int, default=1,
                   help="Number of matches to play (default: 1)")
    p.add_argument("--render", action="store_true",
                   help="Show the match in the pygame window")
    p.add_argument("--failure-rate", type=float, default=0.0,
                   help="Failure-injection rate during eval (default: 0.0)")
    p.add_argument("--teammate-offline", type=float, default=0.0)
    p.add_argument("--opponent", default="random",
                   choices=["random", "greedy", "defensive", "mixed"])
    p.set_defaults(func=cmd_eval)

    # --- export ---
    p = sub.add_parser("export", help="Export to ONNX")
    p.add_argument("--model", required=True, help="SB3 model path")
    p.add_argument("--output", default="models/onnx/model.onnx")
    p.set_defaults(func=cmd_export)

    # --- validate ---
    p = sub.add_parser("validate", help="Validate ONNX model")
    p.add_argument("--onnx", required=True, help="ONNX file path")
    p.add_argument("--model", default=None, help="SB3 model for parity check")
    p.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
