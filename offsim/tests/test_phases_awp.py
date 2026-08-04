from offsim.sim.config import Alliance, Phase
from offsim.sim.field import OverrideField


def test_opening_line_restrictions_and_interaction_transition():
    field=OverrideField(); blue=field.robots[0]; red=field.robots[2]
    blue.x=50; blue.y=69; red.x=90; red.y=79
    field.physics_tick({},0.05)
    assert blue.x+blue.y>=120 and red.x+red.y<=168
    assert "opening_line_block" in blue.telemetry and "opening_line_block" in red.telemetry
    field.advance_clock(29.95)
    assert field.phase is Phase.INTERACTION and field.elapsed==30.0


def test_opening_bonus_and_awp_proxy():
    field=OverrideField()
    for rid,gid in ((0,7),(1,8)):
        robot=field.robots[rid]; goal=field.goals[gid]; robot.x=goal.x+12.5; robot.y=goal.y
        assert field.place(robot,goal)
    field.toggles[0].owner=Alliance.BLUE; field.toggles[1].owner=Alliance.BLUE
    field.advance_clock(30.0)
    assert field.opening_bonus is Alliance.BLUE
    assert field.awp[Alliance.BLUE] is True
    assert field.score(Alliance.BLUE)==field.raw_score(Alliance.BLUE)+12


def test_match_ends_at_120_seconds():
    field=OverrideField(); field.advance_clock(120.0)
    assert field.done and field.phase is Phase.FINISHED and field.time_remaining==0


def test_opening_snapshot_includes_owned_midfield_robot_points():
    field=OverrideField()
    field.robots[0].x=field.robots[0].y=72
    assert field.raw_score(Alliance.BLUE)==28
    field.advance_clock(30.0)
    assert field.opening_raw_scores[Alliance.BLUE]==28
    assert field.opening_bonus is Alliance.BLUE
