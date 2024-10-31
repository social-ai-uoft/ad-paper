"""Experience replay buffers for RL agents."""
from typing import Any, Dict, List, Optional

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class TrajectoryReplayBufferSamples(ReplayBufferSamples):
    """Samples from a trajectory replay buffer.

    Instance Attributes:
        observations: An array of shape ``(seq_len, batch_size, *obs_shape)``.
        actions: An array of shape ``(seq_len, batch_size, action_dim)``.
        next_observations: An array of shape ``(seq_len, batch_size, *obs_shape)``.
        dones: An array of shape ``(seq_len, batch_size, 1)``.
        rewards: An array of shape ``(seq_len, batch_size, 1)``.
    """
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    rewards: np.ndarray


class TrajectoryReplayBuffer(ReplayBuffer):
    """Replay buffer allowing to sample subsequences of episodic trajectories.

    Args:
        buffer_size: Max number of element in the buffer
        observation_space: Observation space
        action_space: Action space
        n_envs: Number of parallel environments
        optimize_memory_usage: Enable a memory efficient variant of the replay
            buffer which reduces by almost a factor two the memory used, at a
            cost of more complexity. Cannot be used in combination when
            `handle_timeout_termination` is True.
        handle_timeout_termination: Handle timeout termination (due to timelimit)
            separately and treat the task as infinite horizon task.
    """

    ep_boundaries: np.ndarray  # i-th element points to the index of the first element of the i-th episode

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True
    ) -> None:
        """Initialize the replay buffer."""
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device='cpu',  # Force CPU since we're using JAX, not PyTorch
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination
        )
        self.ep_boundaries = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """Add a new transition to the buffer.

        Args:
            obs: the last observation
            next_obs: the next observation
            action: the action that was selected
            reward: the reward that was received
            done: whether the episode was done or not
            infos: list of additional information about the transition.
        """
        # Cache the current position in the buffer as `super().add` will mutate
        # the buffer and change the value of `self.pos`
        old_pos = self.pos
        super().add(obs, next_obs, action, reward, done, infos)

        if old_pos == 0:
            # First experience of the buffer, so this experience is the start of a new episode
            self.ep_boundaries[old_pos] = 0
        elif self.dones[old_pos - 1]:  # >= 1
            # Previous experience was end of episode, so this experience is the start of a new episode
            self.ep_boundaries[old_pos] = old_pos
        else:
            # Previous experience was not end of episode, so this experience continues the episode
            self.ep_boundaries[old_pos] = self.ep_boundaries[old_pos - 1]

    def sample(
        self,
        batch_size: int,
        seq_len: int = 1,
        env: Optional[VecNormalize] = None
    ) -> TrajectoryReplayBufferSamples:
        """Sample elements from the replay buffer.

        Args:
            batch_size: Number of elements to sample
            seq_len: Length of the sequence to sample
            env: associated gym VecEnv to normalize the observations/rewards
                when sampling

        Returns:
            A TrajectoryReplayBufferSamples object containing the sampled
            sequences.
        """
        if not self.optimize_memory_usage:
            upper_bound = self.buffer_size if self.full else self.pos
            batch_inds = np.random.randint(0, upper_bound, size=batch_size)
            return self._get_samples(batch_inds, seq_len, env=env)

        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, seq_len, env=env)

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        seq_len: int,
        env: Optional[VecNormalize] = None
    ) -> TrajectoryReplayBufferSamples:
        """Get samples from the replay buffer."""
        # Sample randomly the env idx
        env_inds = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        ep_starts = self.ep_boundaries[batch_inds, env_inds]

        # Reshape `env_inds` to be (seq_len, batch_size) so that we can use it
        # to index into `self.observations` and `self.next_observations`
        env_inds = np.tile(env_inds[:, None], (1, seq_len)).T

        # Look back (at most) `seq_len` steps in the past from `batch_inds`,
        # stopping at the beginning of the episode.
        #
        # We do so by constructing a `seq_len` x `batch_size` matrix where each
        # row contains the indices of the transitions to sample for the
        # corresponding timestep in the sequence.
        seq_inds = np.maximum(
            batch_inds[:, None] - np.arange(seq_len - 1, -1, -1),
            ep_starts[:, None]
        ).T

        if self.optimize_memory_usage:
            raise NotImplementedError(
                'Sampling is not implemented when `optimize_memory_usage` is True'
            )
        else:
            next_obs = self._normalize_obs(self.next_observations[seq_inds, env_inds, :], env)

        data = (
            self._normalize_obs(self.observations[seq_inds, env_inds, :], env),
            self.actions[seq_inds, env_inds, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            np.expand_dims(self.dones[seq_inds, env_inds] * (1 - self.timeouts[seq_inds, env_inds]), -1),
            self._normalize_reward(np.expand_dims(self.rewards[seq_inds, env_inds], -1), env),
        )

        return TrajectoryReplayBufferSamples(*data)
