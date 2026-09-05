"""Override contested-mechanic tests (plan Section 4, Channel 1 = territory)."""
from offsim.sim.config import Alliance
from offsim.sim.field import OverrideField, StackEntry, YELLOW


def _enable(field, **over):
    field.contested_enabled = True
    field.contested = {**field.contested, "enabled": True, "toggle_claim_dwell": 1.0,
                       "alpha_scale": 1.0, "beta": 1.0, "contest_mode": "majority", **over}


def _far(field):
    for i, robot in enumerate(field.robots):
        robot.x = robot.y = 5.0 + i          # park everyone off in a corner


def test_default_off_preserves_instant_toggle_claim():
    field = OverrideField("tank")
    assert field.contested_enabled is False and field.reversal_events == []
    toggle = field.toggles[0]
    robot = field.robots[0]; robot.x, robot.y = toggle.x, toggle.y
    assert field.claim_toggle(robot, toggle) is True        # instant claim still works when OFF
    assert toggle.owner is field.robots[0].alliance


def test_enabled_disables_instant_claim_and_uses_dwell():
    field = OverrideField("tank"); _enable(field)
    toggle = field.toggles[0]
    _far(field)
    blue = field.robots[0]; blue.x, blue.y = toggle.x, toggle.y   # a lone blue robot on the toggle
    assert field.claim_toggle(blue, toggle) is False        # instant claim disabled when ON

    field.update_contested(0.5)
    assert toggle.owner is None, "should not claim before the dwell elapses"
    field.update_contested(0.5)                             # total 1.0s == toggle_claim_dwell
    assert toggle.owner is Alliance.BLUE, "should claim once dwell is reached"


def test_reversal_emits_event_with_value_delta():
    field = OverrideField("tank"); _enable(field)
    toggle = field.toggles[0]; toggle.owner = Alliance.BLUE
    # a yellow half in this toggle's quadrant is the credit that flips hands
    goal = next(g for g in field.goals if g.quadrant == toggle.quadrant)
    pin = field._new_pin((YELLOW, YELLOW), None, None); field.pins[pin].placed_goal = goal.goal_id
    goal.stack.append(StackEntry("pin", pin))
    _far(field)
    for r in (field.robots[2], field.robots[3]):           # two red attackers, zero blue defenders
        r.x, r.y = toggle.x, toggle.y

    field.update_contested(0.5); field.update_contested(0.5)
    assert toggle.owner is Alliance.RED, "majority of attackers should flip ownership"
    assert len(field.reversal_events) == 1
    ev = field.reversal_events[0]
    assert ev["channel"] == "territory" and ev["site_id"] == toggle.toggle_id
    assert ev["from_alliance"] == "blue" and ev["to_alliance"] == "red"
    assert ev["value_delta"] > 0


def test_dwell_resets_when_challenger_leaves():
    field = OverrideField("tank"); _enable(field)
    toggle = field.toggles[0]
    _far(field)
    blue = field.robots[0]; blue.x, blue.y = toggle.x, toggle.y
    field.update_contested(0.5)
    assert field._toggle_claim_timer[toggle.toggle_id] > 0
    blue.x = blue.y = 5.0                                    # leave the region
    field.update_contested(0.5)
    assert field._toggle_claim_timer[toggle.toggle_id] == 0.0
    assert toggle.owner is None


def test_held_value_integrates_over_time():
    field = OverrideField("tank"); _enable(field)
    # place a scored blue half so blue has standing value
    goal = field.goals[1]
    pin = field._new_pin(("blue", "blue"), None, None); field.pins[pin].placed_goal = goal.goal_id
    goal.stack.append(StackEntry("pin", pin))
    _far(field)
    before = field.held_value[Alliance.BLUE]
    field.update_contested(1.0)
    assert field.held_value[Alliance.BLUE] > before
