import math
from collections import Counter
from offsim.sim.config import Alliance
from offsim.sim.field import (
    AUTONOMOUS_TAPE_SEGMENTS, MIDFIELD_DIAMOND, OFFICIAL_OBJECT_ANCHORS,
    OverrideField, StackEntry, gps_to_field,
)
from offsim.sim.renderer import PygameRenderer


def test_official_goal_gps_coordinates_colors_and_types():
    field=OverrideField()
    expected={
        0: ((0,0),None,"neutral_tall"),
        1: ((-600,1200),None,"neutral_short"),
        2: ((-1200,600),None,"neutral_short"),
        3: ((1200,-600),None,"neutral_short"),
        4: ((600,-1200),None,"neutral_short"),
        5: ((-1200,-600),Alliance.RED,"alliance"),
        6: ((-600,-1200),Alliance.RED,"alliance"),
        7: ((600,1200),Alliance.BLUE,"alliance"),
        8: ((1200,600),Alliance.BLUE,"alliance"),
    }
    for goal in field.goals:
        gps,alliance,kind=expected[goal.goal_id]
        assert (goal.x,goal.y)==gps_to_field(*gps)
        assert goal.protected_by is alliance and goal.kind==kind


def test_wall_toggles_corner_loaders_and_vexu_starting_setup():
    field=OverrideField()
    assert [(t.compass,t.quadrant) for t in field.toggles]==[("N",0),("E",1),("S",2),("W",3)]
    assert field.toggles[0].y>140 and field.toggles[1].x>140
    assert field.toggles[2].y<4 and field.toggles[3].x<4
    assert Counter(loader.alliance for loader in field.loaders)=={Alliance.RED:2,Alliance.BLUE:2}
    assert all((loader.x<5) if loader.alliance is Alliance.RED else (loader.x>139) for loader in field.loaders)
    assert len(list(field.objects_on_field("pin")))==32
    assert len(list(field.objects_on_field("cup")))==36
    assert len(OFFICIAL_OBJECT_ANCHORS)==20
    assert field.goals[0].stack[0].kind=="pin"
    center_pin=field.pins[field.goals[0].stack[0].object_id]
    assert center_pin.halves==("yellow","yellow")
    assert all(robot.held_pin is not None and robot.held_cup is None for robot in field.robots)


def test_paired_autonomous_line_midfield_diamond_and_goal_status_legend_contract():
    from offsim.sim.field import AUTONOMOUS_LINE_SEGMENTS, CUP_CLUSTER_ANCHORS, PIN_CLUSTER_ANCHORS
    # Four corner→diamond-edge segments form the large X (no fan to diamond vertices).
    assert len(AUTONOMOUS_TAPE_SEGMENTS)==4
    starts={segment[0] for segment in AUTONOMOUS_TAPE_SEGMENTS}
    ends={segment[1] for segment in AUTONOMOUS_TAPE_SEGMENTS}
    assert starts=={(0.0,0.0),(144.0,144.0),(0.0,144.0),(144.0,0.0)}
    assert ends=={(60.19,60.19),(83.81,83.81),(60.19,83.81),(83.81,60.19)}
    assert set(AUTONOMOUS_LINE_SEGMENTS)=={
        ((0.0,144.0),(60.19,83.81)),
        ((144.0,0.0),(83.81,60.19)),
    }
    assert MIDFIELD_DIAMOND==(
        gps_to_field(0,-600),gps_to_field(600,0),
        gps_to_field(0,600),gps_to_field(-600,0),
    )
    assert len(PIN_CLUSTER_ANCHORS)==8 and len(CUP_CLUSTER_ANCHORS)==12
    assert PygameRenderer.GOAL_LEGEND_LABELS==(
        "Base: Goal type / protection",
        "Halo: yellow-Pin owner",
        "Pips: visible Pin halves",
        "Center #: stack entries",
        "T: tall Midfield Goal",
    )


def test_corrected_renderer_frame_includes_expanded_status_panel(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER","dummy")
    from offsim.sim.env import OverrideStrategyEnv
    env=OverrideStrategyEnv(render_mode="rgb_array")
    env.reset(seed=2); frame=env.render()
    assert frame.shape==(760,1060,3)
    assert frame.dtype.name=="uint8"
    # The field and status panel both contain substantial non-background content.
    assert frame[:,:760].std()>20 and frame[:,760:].std()>20
    env.close()


def test_pickup_pulse_arms_on_new_object_and_seeds_without_flashing(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    field = OverrideField("tank", seed=1)
    renderer = PygameRenderer(render_mode="rgb_array", size=400)
    robot = field.robots[0]
    robot.held_pin = None; robot.held_cup = None
    renderer.draw(field)                              # first draw seeds prev-held; must not flash
    assert 0 not in renderer._pickup_pulse
    cup = field.nearest_object(robot, "cup"); robot.x, robot.y = cup.x, cup.y
    assert field.collect(robot, "cup")
    renderer.draw(field)                              # None -> held triggers the pulse
    assert renderer._pickup_pulse.get(0, 0) > 0
    renderer.close()


def test_match_load_pin_color_split_and_goal_halo_gating():
    field=OverrideField(); robot=field.robots[0]
    robot.held_pin=None; robot.held_cup=None
    loader=field.nearest_loader(robot); robot.x=loader.x-5; robot.y=loader.y
    loaded=[]
    for _ in range(13):
        assert field.use_loader(robot)
        loaded.append(field.pins[robot.held_pin].halves)
        robot.held_pin=None
    assert loaded.count(("blue","yellow"))==10
    assert loaded.count(("yellow","yellow"))==3
    assert field.match_loads[Alliance.BLUE]["yellow_pin"]==0

    empty_goal=field.goals[7]
    field.toggles[empty_goal.quadrant].owner=Alliance.BLUE
    assert field.goal_owner(empty_goal) is Alliance.BLUE
    assert field.goal_status_owner(empty_goal) is None
    yellow=field._new_pin(("blue","yellow"),None,None)
    empty_goal.stack.append(StackEntry("pin",yellow))
    assert field.goal_status_owner(empty_goal) is Alliance.BLUE
