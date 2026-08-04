"""Command-line entry point for the Override 2D prototype."""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))


def cmd_demo(args):
    from sim.env import OverrideStrategyEnv
    render_mode = None if args.headless else "human"
    env = OverrideStrategyEnv(chassis=args.chassis, render_mode=render_mode, opponent=args.opponent)
    obs, info = env.reset(seed=args.seed)
    if render_mode:
        env.render()
        env._renderer.speed = args.speed
    matches = 0; decisions = 0
    print(f"Override autoplay: blue 2v2 red, chassis={args.chassis}, opponent={args.opponent}")
    if render_mode:
        print(f"Playback: {args.speed:g}x (Space pause, S step, R reset, +/- speed)")
    try:
        while matches < args.matches:
            if env._renderer:
                if not env._renderer.running: break
                if env._renderer.should_reset:
                    env._renderer.should_reset = False; obs, info = env.reset(seed=args.seed); decisions = 0
                if env._renderer.paused and not env._renderer.step_once:
                    env.render(); continue
                env._renderer.step_once = False
            obs, reward, done, _, info = env.step(env.autoplay_actions()); decisions += 1
            if args.max_decisions and decisions >= args.max_decisions: done = True
            if done:
                matches += 1
                print(f"match {matches}: blue={info['blue_score']} red={info['red_score']} bonus={info['opening_bonus']} awp={info['blue_awp']}/{info['red_awp']}")
                if matches < args.matches: obs, info = env.reset(seed=None if args.seed is None else args.seed+matches); decisions = 0
    finally:
        env.close()


def cmd_train(args):
    from training.train_sim import train
    train(total_timesteps=args.timesteps, n_envs=args.n_envs, learning_rate=args.lr,
          output_dir=args.output_dir, chassis=args.chassis, opponent=args.opponent,
          device=args.device, resume=args.resume, n_steps=args.n_steps)


def _resolve_model(path, output_dir="models"):
    if path != "latest": return path
    candidates = [os.path.join(output_dir, name) for name in ("final_model.zip", "interrupted_model.zip")]
    candidates = [p for p in candidates if os.path.exists(p)]
    if not candidates: raise FileNotFoundError(f"no Override model found in {output_dir}")
    return max(candidates, key=os.path.getmtime)


def cmd_eval(args):
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks
    from sim.env import OverrideStrategyEnv
    model_path = _resolve_model(args.model, args.output_dir)
    env = OverrideStrategyEnv(chassis=args.chassis, render_mode="human" if args.render else None, opponent=args.opponent)
    model = MaskablePPO.load(model_path, env=env, device=args.device)
    results = []
    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed+episode); done = False; total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True, action_masks=get_action_masks(env))
            obs, reward, done, _, info = env.step(action); total += float(reward)
            if args.render: env.render()
        results.append((info["blue_score"], info["red_score"], total))
        print(f"episode {episode+1}: blue={results[-1][0]} red={results[-1][1]} reward={total:.2f}")
    env.close()


def cmd_export(args):
    from export.export_onnx import export_to_onnx
    export_to_onnx(_resolve_model(args.model, args.output_dir), args.output)


def cmd_validate(args):
    from export.validate_onnx import validate
    validate(args.onnx, None if args.model is None else _resolve_model(args.model, args.output_dir))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Override 2D simulator and MaskablePPO trainer")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="run deterministic 2v2 autoplay")
    demo.add_argument("--chassis", choices=["tank", "mecanum"], default="tank")
    demo.add_argument("--opponent", choices=["greedy", "toggle", "mixed"], default="mixed")
    demo.add_argument("--matches", type=int, default=1); demo.add_argument("--seed", type=int, default=None)
    demo.add_argument("--speed", type=float, choices=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0], default=1.0,
                      help="visual playback multiplier (default: real-time 1x)")
    demo.add_argument("--headless", action="store_true"); demo.add_argument("--max-decisions", type=int, default=0)
    demo.set_defaults(func=cmd_demo)
    train = sub.add_parser("train", help="train centralized two-allied strategy policy")
    train.add_argument("--chassis", choices=["tank", "mecanum"], default="tank")
    train.add_argument("--opponent", choices=["greedy", "toggle", "mixed"], default="mixed")
    train.add_argument("--timesteps", type=int, default=1_000_000); train.add_argument("--n-envs", type=int, default=4)
    train.add_argument("--n-steps", type=int, default=256); train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--output-dir", default="models"); train.add_argument("--resume", default=None)
    train.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto"); train.set_defaults(func=cmd_train)
    evaluate = sub.add_parser("eval", help="evaluate a trained Override policy")
    evaluate.add_argument("--model", default="latest"); evaluate.add_argument("--output-dir", default="models")
    evaluate.add_argument("--episodes", type=int, default=1); evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--chassis", choices=["tank", "mecanum"], default="tank")
    evaluate.add_argument("--opponent", choices=["greedy", "toggle", "mixed"], default="mixed")
    evaluate.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto"); evaluate.add_argument("--render", action="store_true")
    evaluate.set_defaults(func=cmd_eval)
    export = sub.add_parser("export", help="export centralized policy logits to ONNX")
    export.add_argument("--model", default="latest"); export.add_argument("--output-dir", default="models"); export.add_argument("--output", default="models/onnx/override.onnx"); export.set_defaults(func=cmd_export)
    validate = sub.add_parser("validate", help="validate ONNX inference and optional SB3 parity")
    validate.add_argument("--onnx", required=True); validate.add_argument("--model", default=None); validate.add_argument("--output-dir", default="models"); validate.set_defaults(func=cmd_validate)
    args = parser.parse_args(argv); args.func(args)

if __name__ == "__main__": main()
