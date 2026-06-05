"""Per-decision CSV logger for the VEX Push Back RL environment.

Writes one row per RL decision with:
  policy_action   — what the policy chose (before commitment / masking)
  executed_action — what actually ran (after commitment + mask remapping)
  action_source   — policy / committed / mask_fallback
  target_x/y      — navigation waypoint sent to the robot
  pos_x/y, heading — robot state at decision start
  balls_held, balls_on_field
  us_score, opp_score, margin
  reward, top_component — total reward and the single largest contributor
  travelled_in    — inches covered during the decision
  low_progress    — True if the robot moved less than 6" on a non-IDLE action
  wall_ms         — wall-clock milliseconds to execute this decision

Enable with env.enable_logging(log_dir).  Off by default so training is not
slowed.  The output file is auto-named with a timestamp.
"""

from __future__ import annotations
import csv
import math
import os
import time
from pathlib import Path

from sim.config import Action, OBJ_ON_FIELD


class DecisionLogger:
    """Write one CSV row per RL decision.  Thread-unsafe; use from one process."""

    _FIELDS = [
        "decision", "sim_time_s",
        "robot_id",
        "policy_action", "executed_action", "action_source",
        "pos_x", "pos_y", "heading_deg",
        "balls_held", "target_x", "target_y",
        "us_score", "opp_score", "margin",
        "reward", "top_component",
        "travelled_in", "low_progress",
        "balls_on_field",
        "wall_ms",
    ]

    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self._path = Path(log_dir) / f"decisions_{stamp}.csv"
        self._file = open(self._path, "w", newline="", buffering=1)
        self._writer = csv.DictWriter(self._file, fieldnames=self._FIELDS)
        self._writer.writeheader()
        self._decision = 0
        self._wall_start: float = 0.0
        print(f"[DecisionLogger] writing to {self._path}")

    # ------------------------------------------------------------------

    def mark_step_start(self) -> None:
        """Call at the very beginning of env.step() to start the wall timer."""
        self._wall_start = time.perf_counter()

    def log(
        self,
        env,
        robot_id: int,
        policy_action: int,
        executed_action: int,
        action_source: str,        # "policy" | "committed" | "mask_fallback"
        target,                    # np.ndarray or None
        reward: float,
        travelled_in: float,
    ) -> None:
        robot = env.field.allies[robot_id]
        hdg_deg = math.degrees(robot.heading) % 360.0

        breakdown = env.last_reward_breakdown[robot_id]
        if breakdown:
            top_key = max(breakdown, key=lambda k: abs(breakdown[k]))
            top_str = f"{top_key}={breakdown[top_key]:+.3f}"
        else:
            top_str = ""

        balls_on = sum(1 for o in env.field.objects if o.status == OBJ_ON_FIELD)
        wall_ms = round((time.perf_counter() - self._wall_start) * 1000.0, 1)

        tx = f"{float(target[0]):.1f}" if target is not None else ""
        ty = f"{float(target[1]):.1f}" if target is not None else ""

        self._writer.writerow({
            "decision":        self._decision,
            "sim_time_s":      f"{env.field.time_remaining:.1f}",
            "robot_id":        robot_id,
            "policy_action":   Action(policy_action).name,
            "executed_action": Action(executed_action).name,
            "action_source":   action_source,
            "pos_x":           f"{robot.x:.1f}",
            "pos_y":           f"{robot.y:.1f}",
            "heading_deg":     f"{hdg_deg:.0f}",
            "balls_held":      robot.balls_held,
            "target_x":        tx,
            "target_y":        ty,
            "us_score":        env.field.my_score,
            "opp_score":       env.field.opponent_score,
            "margin":          env.field.my_score - env.field.opponent_score,
            "reward":          f"{reward:.4f}",
            "top_component":   top_str,
            "travelled_in":    f"{travelled_in:.1f}",
            "low_progress":    int(env.low_progress[robot_id]),
            "balls_on_field":  balls_on,
            "wall_ms":         wall_ms,
        })
        self._decision += 1

    def new_episode(self) -> None:
        """Insert a blank separator row so episodes are visually distinct."""
        self._writer.writerow({f: "" for f in self._FIELDS})

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __del__(self) -> None:
        self.close()
