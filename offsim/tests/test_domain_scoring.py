from offsim.sim.config import Alliance
from offsim.sim.field import OverrideField, StackEntry, YELLOW


def near(robot,target,distance=12.5):
    robot.x=target.x+distance; robot.y=target.y


def test_setup_preloads_possession_caps_and_match_loaders():
    field=OverrideField("tank"); assert len(field.robots)==4 and len(field.goals)==9 and len(field.toggles)==4
    robot=field.robots[0]
    assert robot.held_pin is not None and robot.held_cup is None
    assert field.collect(robot,"pin") is False
    robot.held_pin=None
    loader=field.nearest_loader(robot); robot.x=loader.x-5; robot.y=loader.y
    assert field.in_load_zone(robot)
    robot.y=72; assert not field.in_load_zone(robot); robot.y=loader.y
    before=field.match_loads[Alliance.BLUE].copy()
    assert field.use_loader(robot); assert field.match_loads[Alliance.BLUE]["pin"]==before["pin"]-1
    assert field.use_loader(robot); assert field.match_loads[Alliance.BLUE]["cup"]==before["cup"]-1
    assert not field.use_loader(robot)


def test_symbolic_order_nesting_visible_halves_and_scoring():
    field=OverrideField(); robot=field.robots[0]; goal=field.goals[1]; near(robot,goal)
    cup=field.nearest_object(robot,"cup"); cup.x=cup.y=None; robot.held_cup=cup.object_id
    pin_id=robot.held_pin; cup_id=robot.held_cup
    assert field.place(robot,goal) and field.place(robot,goal)
    assert [entry.kind for entry in goal.stack]==["pin","cup"]
    assert goal.stack[1].nested_on==pin_id
    assert goal.visible_pin_halves(field.pins)==("blue",YELLOW)
    assert field.raw_score(Alliance.BLUE)==5
    field.toggles[goal.quadrant].owner=Alliance.BLUE
    assert field.raw_score(Alliance.BLUE)==15
    assert field.pins[pin_id].placed_goal==goal.goal_id and field.cups[cup_id].placed_goal==goal.goal_id


def test_protected_goal_and_neutral_removal_prevention_telemetry():
    field=OverrideField(); robot=field.robots[0]; robot.held_pin=None
    protected=next(g for g in field.goals if g.protected_by is Alliance.RED); near(robot,protected)
    blue=field._new_pin(("blue","blue"),None,None); field.pins[blue].placed_goal=protected.goal_id; protected.stack.append(StackEntry("pin",blue))
    assert not field.remove_own_pin(robot,protected); assert "protected_goal_block" in field.telemetry[-1]
    own=next(g for g in field.goals if g.protected_by is Alliance.BLUE); near(robot,own)
    mixed=field._new_pin(("blue",YELLOW),None,None); field.pins[mixed].placed_goal=own.goal_id; own.stack.append(StackEntry("pin",mixed))
    assert not field.remove_own_pin(robot,own); assert "neutral_removal_block" in field.telemetry[-1]


def test_removable_alliance_pin_toggle_and_midfield_robot_points():
    field=OverrideField(); robot=field.robots[0]; robot.held_pin=None
    goal=field.goals[3]; near(robot,goal)
    pin=field._new_pin(("blue","blue"),None,None); field.pins[pin].placed_goal=goal.goal_id; goal.stack.append(StackEntry("pin",pin))
    assert field.remove_own_pin(robot,goal) and robot.held_pin==pin
    for toggle in field.toggles[:3]: toggle.owner=Alliance.BLUE
    field.robots[0].x=field.robots[0].y=72; field.robots[1].x=65; field.robots[1].y=72
    assert field.midfield_owner() is Alliance.BLUE
    assert field.raw_score(Alliance.BLUE)>=16


def test_descore_removes_enemy_pin_and_respects_protection():
    field=OverrideField(); red=field.robots[2]; assert red.alliance is Alliance.RED
    # blue scores a Pin in a neutral Goal; red can come descore it
    neutral=field.goals[1]
    blue_pin=field._new_pin(("blue",YELLOW),None,None); field.pins[blue_pin].placed_goal=neutral.goal_id
    neutral.stack.append(StackEntry("pin",blue_pin)); near(red,neutral)
    before=field.raw_score(Alliance.BLUE)
    assert field.can_descore(red,neutral) and field.descore_pin(red,neutral)
    assert not neutral.stack                                   # pin removed from the goal
    assert field.pins[blue_pin].placed_goal is None and field.pins[blue_pin].x is not None  # dropped loose
    assert field.raw_score(Alliance.BLUE) < before            # blue's credit dropped
    # a blue-protected Alliance Goal cannot be descored by red
    blue_goal=next(g for g in field.goals if g.protected_by is Alliance.BLUE)
    bp=field._new_pin(("blue",YELLOW),None,None); field.pins[bp].placed_goal=blue_goal.goal_id
    blue_goal.stack.append(StackEntry("pin",bp)); near(red,blue_goal)
    assert not field.can_descore(red,blue_goal) and not field.descore_pin(red,blue_goal)


def test_midfield_owner_scores_centered_robot_and_owns_center_pin():
    field=OverrideField()
    for robot in field.robots: robot.x=robot.y=20
    field.robots[2].x=field.robots[2].y=72
    assert field.midfield_owner() is Alliance.RED
    assert field.raw_score(Alliance.BLUE)==0
    assert field.raw_score(Alliance.RED)==28
