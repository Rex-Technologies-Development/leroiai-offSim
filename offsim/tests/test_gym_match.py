import numpy as np
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
from offsim.sim.config import Action, NUM_ACTIONS, TEAM_STATE_DIM, Phase
from offsim.sim.env import OverrideContinuousEnv, OverrideStrategyEnv
from offsim.training.reward import strategy_reward


def test_gym_spaces_and_checker():
    continuous=OverrideContinuousEnv("mecanum"); strategy=OverrideStrategyEnv("tank")
    assert continuous.action_space.shape==(2,6)
    assert strategy.action_space.nvec.tolist()==[NUM_ACTIONS,NUM_ACTIONS]
    assert strategy.observation_space.shape==(TEAM_STATE_DIM,)
    check_env(continuous,skip_render_check=True); check_env(strategy,skip_render_check=True)
    continuous.close(); strategy.close()


def test_masks_are_flattened_per_robot_and_valid():
    env=OverrideStrategyEnv(); env.reset(seed=4); mask=env.action_masks()
    assert mask.shape==(NUM_ACTIONS*2,) and mask.dtype==bool
    branches=mask.reshape(2,NUM_ACTIONS)
    assert branches[:,Action.IDLE].all() and branches[:,Action.SCORE_NEAREST_GOAL].all()
    assert not branches[:,Action.COLLECT_PIN].any()
    assert branches[:,Action.COLLECT_CUP].all()
    for branch in branches: assert branch.any()


def test_continuous_mecanum_lateral_control_and_reward_helper():
    env=OverrideContinuousEnv("mecanum"); env.reset(seed=2)
    start=env.field.robots[0].position.copy()
    action=np.zeros((2,6),np.float32); action[0,1]=1
    for _ in range(10): env.step(action)
    assert np.linalg.norm(env.field.robots[0].position-start)>1.0
    assert strategy_reward((5,0),(15,5))==1.0


def test_strategy_determinism_for_same_seed_and_actions():
    first=OverrideStrategyEnv("mecanum"); second=OverrideStrategyEnv("mecanum")
    o1,_=first.reset(seed=12); o2,_=second.reset(seed=12); np.testing.assert_array_equal(o1,o2)
    actions=[np.array([Action.SCORE_NEAREST_GOAL,Action.SCORE_NEAREST_GOAL]),np.array([Action.CLAIM_TOGGLE,Action.CLAIM_TOGGLE]),np.array([Action.COLLECT_PIN,Action.COLLECT_PIN])]
    for action in actions:
        r1=first.step(action); r2=second.step(action)
        np.testing.assert_array_equal(r1[0],r2[0]); assert r1[1:]==r2[1:]


def test_complete_autoplay_match_smoke():
    env=OverrideStrategyEnv("tank",opponent="mixed"); obs,_=env.reset(seed=9); steps=0
    while not env.field.done:
        obs,reward,done,truncated,info=env.step(env.autoplay_actions()); steps+=1
        assert env.observation_space.contains(obs); assert not truncated
        assert steps<=60
    assert done and env.field.phase is Phase.FINISHED and steps==60
    assert info["time_remaining"]==0 and len(env.field.robots)==4
    assert info["blue_score"]>=0 and info["red_score"]>=0


def test_remove_mask_and_target_match_exact_legality():
    from offsim.sim.field import StackEntry, YELLOW
    from offsim.sim.env import ObjectiveController
    env=OverrideStrategyEnv(); env.reset(seed=1); robot=env.field.robots[0]; robot.held_pin=None
    mixed_goal=env.field.goals[3]
    mixed=env.field._new_pin(("blue",YELLOW),None,None); mixed_goal.stack.append(StackEntry("pin",mixed))
    assert not env.action_masks().reshape(2,NUM_ACTIONS)[0,Action.REMOVE_OWN_PIN]
    legal_goal=env.field.goals[4]
    legal=env.field._new_pin(("blue","blue"),None,None); legal_goal.stack.append(StackEntry("pin",legal))
    assert env.action_masks().reshape(2,NUM_ACTIONS)[0,Action.REMOVE_OWN_PIN]
    assert ObjectiveController(env.field).target(0,Action.REMOVE_OWN_PIN) is legal_goal


def test_training_module_imports_as_package():
    from offsim.training.train_sim import make_env
    created=make_env()(); assert created.action_space.nvec.tolist()==[10,10]; created.close()


def test_visual_strategy_step_renders_each_physics_tick_only_in_human_mode():
    class DummyRenderer:
        def __init__(self): self.calls=0
        def draw(self,field): self.calls+=1
        def close(self): pass
    visual=OverrideStrategyEnv(render_mode="human"); visual.reset(seed=5)
    visual._renderer=DummyRenderer(); visual.step(np.array([Action.IDLE,Action.IDLE]))
    assert visual._renderer.calls==40
    headless=OverrideStrategyEnv(); headless.reset(seed=5)
    headless.step(np.array([Action.IDLE,Action.IDLE]))
    assert headless._renderer is None
