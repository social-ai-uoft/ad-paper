"""Shared features extractor modules for RL algorithms."""
from abc import ABC, abstractmethod
from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp
from stable_baselines3.common.preprocessing import get_flattened_obs_dim


class BaseFeaturesExtractor(ABC):
    """Base class for features extractor.

    A features extractor takes as input an observation and returns a
    vectorized representation of this observation. No assumptions are
    made about the kind of data the extractor can take as input or
    what it can return as output.

    For example, a CNN features extractor would take as input an image
    observation and return a vector of features extracted by the CNN.

    Args:
        observation_space: The observation space of the environment.
        features_dim: The dimension of the features extracted.
    """

    # Private Instance Attributes:
    #   _observation_space: The observation space of the environment.
    #   _features_dim: The dimension of the features extracted.
    _observation_space: gym.spaces.Space
    _features_dim: int

    def __init__(self, observation_space: gym.spaces.Space,
                 features_dim: int = 0) -> None:
        """Initialize a new features extractor."""
        assert features_dim > 0, 'The features_dim must be positive'
        self._observation_space = observation_space
        self._features_dim = features_dim

    @abstractmethod
    def __call__(self, observation: jax.typing.ArrayLike) -> jax.Array:
        """Extract features from the observation."""
        raise NotImplementedError

    def __repr__(self) -> str:
        """Return a string representation of the features extractor."""
        return f'{self.__class__.__name__}(' \
            f'observation_space={self._observation_space}, ' \
            f'features_dim={self._features_dim})'

    @property
    def features_dim(self) -> int:
        """Return the dimension of the features extracted."""
        return self._features_dim


class FlattenExtractor(BaseFeaturesExtractor):
    """Flatten all dimensions of the observation space, except the first."""

    def __init__(self, observation_space: gym.spaces.Space) -> None:
        """Initialize a new flatten extractor."""
        features_dim = get_flattened_obs_dim(observation_space)
        super().__init__(observation_space, features_dim)

    @partial(jax.jit, static_argnums=0)
    def __call__(self, observation: jax.typing.ArrayLike) -> jax.Array:
        """Flatten the observation."""
        return jnp.reshape(observation, (observation.shape[0], -1))

# TODO: Add logic for learnable features extractors (i.e. CNNs, RNNs, etc.)
