"""Gymnasium wrapper: reset/step API, spaces, truncation, reproducibility."""
import pytest
from gymnasium.utils.env_checker import check_env

from environment.gym_env import SmartDiceEnv


def test_reset_returns_obs_and_info():
    env = SmartDiceEnv(experiment=1)
    obs, info = env.reset(seed=0)
    assert obs == 0 and info["position"] == 1 and info["moves"] == 0


def test_spaces():
    env = SmartDiceEnv(experiment=2)
    assert env.observation_space.n == 50 and env.action_space.n == 6


def test_step_returns_five_values():
    env = SmartDiceEnv(experiment=1)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)   # dice 1
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float) and not terminated and not truncated
    assert info["dice"] == 1 and "event" in info


def test_invalid_action_is_rejected():
    env = SmartDiceEnv(experiment=1)
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(6)


def test_episode_is_truncated_at_max_steps():
    env = SmartDiceEnv(experiment=1, max_steps=5)
    env.reset(seed=0)
    truncated = False
    for _ in range(5):
        _, _, terminated, truncated, _ = env.step(0)
        if terminated:
            break
    assert truncated or terminated


def test_same_seed_same_trajectory_with_noisy_dice():
    def run():
        env = SmartDiceEnv(experiment=1, slip_prob=0.5)
        env.reset(seed=123)
        out = []
        for _ in range(15):
            obs, r, term, trunc, _ = env.step(2)
            out.append((obs, r))
            if term or trunc:
                break
        return out
    assert run() == run()


def test_gymnasium_env_checker():
    check_env(SmartDiceEnv(experiment=1), skip_render_check=True)
