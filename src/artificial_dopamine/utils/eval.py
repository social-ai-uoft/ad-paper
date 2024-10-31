"""Evaluation for DQN models."""
import random
from typing import Callable, Union

import gymnasium as gym
import jax
import numpy as np
import tqdm


def evaluate_policy(
    policy: Callable[[jax.typing.ArrayLike], jax.typing.ArrayLike],
    env: Union[gym.Env, gym.vector.VectorEnv],
    num_episodes: int = 10,
    epsilon: float = 0.05,
    show_progress: bool = False
) -> tuple[float, float]:
    """Evaluate a policy on a given environment.

    Args:
        policy: The policy to evaluate.
        env: The environment to evaluate the policy on.
        num_episodes: The number of episodes to evaluate the policy on.
            Default: 10.
        epsilon: The probability of taking a random action. Default: 0.05.
        show_progress: Whether to show the progress of the evaluation.
            Default: ``False``.

    Returns:
        The mean and standard deviation of the returns of the policy.
    """
    episodic_returns = []
    for _ in tqdm.trange(num_episodes, disable=not show_progress):
        obs, _ = env.reset()
        context_acts = None
        total_reward = 0.0
        while True:
            if random.random() < epsilon:
                actions = env.action_space.sample()
            else:
                actions, context_acts = policy(obs, context_acts)

            obs, reward, terminated, truncated, _ = env.step(actions)
            total_reward += reward

            if terminated or truncated:
                break

        episodic_returns.append(total_reward)

    return np.mean(episodic_returns), np.std(episodic_returns)
