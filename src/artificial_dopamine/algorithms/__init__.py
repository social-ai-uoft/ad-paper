"""Artificial-Dopamine (AD) learning algorithms.

The framework architecture is inspired by OpenAI Baselines and Rljax.
"""
# Supervised learning
# from artificial_dopamine.algorithms.ad_mlp import AD_MLP  # noqa: F401

# Q-learning
from artificial_dopamine.algorithms.ad_dqn import AD_DQN  # noqa: F401
from artificial_dopamine.algorithms.ad_qrdqn import AD_QRDQN  # noqa: F401
# TODO: Refactor algorithms to use a base class, avoiding redundant training logic