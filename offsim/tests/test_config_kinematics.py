import math
from pathlib import Path
import yaml
import numpy as np
from offsim.sim.config import Action, ChassisType, NUM_ACTIONS, PROFILE, Phase
from offsim.sim.field import OverrideField
from offsim.sim.robot import Robot
from offsim.sim.config import Alliance


def test_shared_config_action_parity_and_profile():
    raw=yaml.safe_load((Path(__file__).parents[1]/"shared"/"config.yaml").read_text())
    assert raw["game"] == "override"
    assert [raw["actions"][i] for i in range(raw["num_actions"])] == [a.name for a in Action]
    assert NUM_ACTIONS == 10
    assert PROFILE["robots"] == 4 and PROFILE["opening_duration"] == 30.0 and PROFILE["match_duration"] == 120.0
    assert PROFILE["drivers"] is False


def test_tank_has_no_strafe_and_mecanum_does():
    tank=Robot(0,Alliance.BLUE,ChassisType.TANK,20,20,0)
    mecanum=Robot(1,Alliance.BLUE,ChassisType.MECANUM,20,20,0)
    for _ in range(20): tank.command(0,1,0,0.05); mecanum.command(0,1,0,0.05)
    assert tank.y == 20
    assert mecanum.y > 25
    assert tank.x == 20 and abs(mecanum.x-20)<1e-6


def test_acceleration_and_deceleration_are_bounded():
    robot=Robot(0,Alliance.BLUE,ChassisType.TANK,20,20,0)
    robot.command(1,0,0,0.1)
    assert math.isclose(robot.forward_velocity,4.8)
    robot.command(0,0,0,0.1)
    assert robot.forward_velocity == 0.0


def test_wall_static_and_robot_collisions():
    field=OverrideField("tank",seed=1); robot=field.robots[0]
    robot.x=1; field.physics_tick({0:(-1,0,0)},0.1)
    assert robot.x >= robot.radius and robot.collisions >= 1
    goal=field.goals[4]; robot.x=goal.x; robot.y=goal.y
    field.physics_tick({},0.05)
    assert math.hypot(robot.x-goal.x,robot.y-goal.y) >= robot.radius+field.goal_radius-1e-6
    first,second=field.robots[:2]; first.x=first.y=50; second.x=second.y=50
    field.physics_tick({},0.05)
    assert np.linalg.norm(first.position-second.position) >= first.radius+second.radius-1e-6


def test_pair_collisions_preserve_wall_and_opening_line_constraints():
    wall=OverrideField(); wall.phase=Phase.INTERACTION; first,second=wall.robots[:2]
    first.x=second.x=first.radius; first.y=second.y=60
    wall.physics_tick({},0.05)
    assert first.x>=first.radius and second.x>=second.radius
    assert np.linalg.norm(first.position-second.position)>=first.radius+second.radius-1e-6

    opening=OverrideField(); first,second=opening.robots[:2]
    first.x=50; first.y=70; second.x=55; second.y=65
    opening.physics_tick({},0.05)
    assert first.x+first.y>=120 and second.x+second.y>=120
    assert np.linalg.norm(first.position-second.position)>=first.radius+second.radius-1e-6


def test_pair_collision_does_not_reintroduce_goal_penetration():
    field=OverrideField(); goal=field.goals[0]; first,second=field.robots[:2]
    first.x=goal.x+first.radius+field.goal_radius; first.y=goal.y
    second.x=first.x+7.5; second.y=goal.y
    field.physics_tick({},0.05)
    assert math.hypot(first.x-goal.x,first.y-goal.y)>=first.radius+field.goal_radius-1e-6
    assert math.hypot(second.x-goal.x,second.y-goal.y)>=second.radius+field.goal_radius-1e-6
    assert np.linalg.norm(first.position-second.position)>=first.radius+second.radius-1e-6
