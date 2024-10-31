"""Utility functions."""
from typing import Any, Callable, Literal, Optional, Union

import jax
import jax.numpy as jnp
import dmc2gym
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.preprocessing import is_image_space


def linear_schedule(a: float, b: float, duration: int, t: int) -> float:
    """Linearly interpolate between `a` and `b` over `duration` steps.

    Args:
        a: The starting value.
        b: The ending value.
        duration: The number of steps over which to interpolate.
        t: The current step.

    Returns:
        The interpolated value. If `t >= duration`, then `b` is returned.
    """
    return max((b - a) / duration * t + a, b)


def preprocess_obs(
    obs: jax.typing.ArrayLike,
    observation_space: spaces.Space,
    normalize_images: bool = True
) -> Union[jax.Array, dict[str, jax.Array]]:
    """Preprocess observations to be passed to a neural network.

    The behavior depends on the type of the ``observation_space``:

    * If ``observation_space`` is an image space and ``normalize_images`` is True,
        the image observations are divided by 255 so that the pixel values are in
        [0, 1].
    * If the ``observation_space`` is discrete, it is converted to a
        one-hot encoding.

    Args:
        obs: The observation to preprocess. This will be converted to a
            :class:`jax.Array` object before preprocessing.
        observation_space: The observation space.
        normalize_images: Whether to normalize images or not. Default: ``True``.

    Returns:
        The preprocessed observation.

    Raises:
        NotImplementedError: If the ``observation_space`` is not supported.

    Remarks:
        This function is originally from the Stable Baselines3 library, and has
        been modified to support JAX arrays.
    """
    obs = jnp.asarray(obs, dtype=jnp.float32)
    if isinstance(observation_space, spaces.Box):
        if is_image_space(observation_space) and normalize_images:
            return obs / 255.0
        return obs

    elif isinstance(observation_space, spaces.Discrete):
        # One hot encoding and convert to float to avoid errors
        return jax.nn.one_hot(obs, observation_space.n)

    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Concatenation of one hot encodings of each Categorical sub-space
        enc = jnp.concatenate(
            [
                jax.nn.one_hot(obs_, int(observation_space.nvec[idx]))
                for idx, obs_ in enumerate(jnp.split(obs, 1, axis=1))
            ],
            dim=-1,
        )
        # Reshape to (batch_size, sum(observation_space.nvec))
        return jnp.reshape(enc, (obs.shape[0], sum(observation_space.nvec)))

    elif isinstance(observation_space, spaces.MultiBinary):
        return obs

    elif isinstance(observation_space, spaces.Dict):
        # Do not modify by reference the original observation
        preprocessed_obs = {}
        for key, _obs in obs.items():
            preprocessed_obs[key] = preprocess_obs(
                _obs,
                observation_space[key],
                normalize_images=normalize_images
            )
        return preprocessed_obs

    else:
        raise NotImplementedError(
            f'Preprocessing not implemented for {observation_space}'
        )


class EnvSpec:
    """A factory for creating Gym environments according to a specification.

    Args:
        env_id: The ID of the environment to create.
        num_envs: The number of environments to create.
        vectorization_mode: One "sync" or "async", indicating whether to run
            environments synchronously (serially) or asynchronously (in
            parallel using multiple processes).
        wrap_env_fn: A callable with the following signature:

            .. code-block:: python
                def wrap_env_fn(env: gym.Env, env_spec: EnvSpec, index: int) -> gym.Env:
                    ...

            that wraps the environment in one or more wrappers. If ``None``,
            the identity function is used. This is applied after the environment
            is created, but before it is vectorized. Note that by default, the
            environment is wrapped in a :class:`gym.wrappers.RecordEpisodeStatistics`
            wrapper so there is no need to add this wrapper manually.

        env_factory: A callable with the following signature:

            .. code-block:: python
                def make_env(
                    env_spec: EnvSpec,
                    index: int,
                    record_video: bool = False,
                    record_video_freq: Optional[int] = None,
                    run_log_dir: Optional[str] = None,
                    seed: int = 0
                ) -> gym.Env:
                    ...

            that creates a gym environment for a given index. If only the
            environment is needed, then you may alternatively use the
            signature ``def make_env(env: gym.Env, *_: Any) -> gym.Env``
            to ignore the remaining arguments. If ``None``, then the default
            function is used which creates the environment using
            the given ``env_id`` and ``init_kwargs`` and then wraps it in a
            :class:`gym.wrappers.RecordEpisodeStatistics` wrapper (and if
            ``record_video`` is ``True``, a :class:`gym.wrappers.RecordVideo`
            wrapper). This is useful for environments that require special
            initialization logic. Note that if you override this, the
            ``wrap_env_fn`` argument will not be applied and must manually
            be called in your custom factory function.

        init_kwargs: Keyword arguments to pass to the environment constructor.

    Attributes:
        dummy_env: A dummy environment created at initialization time using
            the provided ``env_id`` and ``init_kwargs``. This is used to
            determine the observation and action spaces of the environment.
    """

    env_id: str
    num_envs: int
    vectorization_mode: Literal['sync', 'async']
    wrap_env_fn: Callable[[gym.Env], gym.Env]
    env_factory: Callable[..., gym.Env]
    init_kwargs: dict[str, Any]
    dummy_env: gym.Env

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        vectorization_mode: Literal['sync', 'async'] = 'sync',
        wrap_env_fn: Optional[Callable[[gym.Env], gym.Env]] = None,
        env_factory: Optional[Callable[..., gym.Env]] = None,
        **init_kwargs: Any
    ) -> None:
        """Initialize the environment specification."""
        assert vectorization_mode in ['sync', 'async'], \
            f'Invalid vectorization mode: {vectorization_mode}'
        assert num_envs > 0, f'Invalid number of environments: {num_envs}'

        self.env_id = env_id
        self.num_envs = num_envs
        self.vectorization_mode = vectorization_mode
        self.init_kwargs = init_kwargs
        self.wrap_env_fn = wrap_env_fn or (lambda env, *_: env)
        self.env_factory = env_factory or self._default_env_factory
        self.dummy_env = self.make_env()

    def make_env(
        self,
        record_video: bool = False,
        record_video_freq: Optional[int] = None,
        run_log_dir: Optional[str] = None,
        seed: int = 0
    ) -> gym.vector.VectorEnv:
        """Returns a vectorized environment created according to the spec.

        Args:
            record_video: Whether to record a video of the environment.
                If multiple environments are created, only the first one
                will record a video.
            record_video_freq: The frequency at which to record videos.
                If ``record_video`` is ``True``, then a video will be recorded
                every ``record_video_freq`` episodes. If ``None``, the default
                schedule provided by the :class:`gym.wrappers.RecordVideo`
                wrapper will be used.
            run_log_dir: The directory where the video will be saved.
                If ``None``, then a temporary directory will be used.
            seed: The seed to use for the environment.
        """
        def make_env(index: int) -> gym.Env:
            def thunk() -> gym.Env:
                return self.env_factory(
                    self,
                    index,
                    record_video,
                    record_video_freq,
                    run_log_dir,
                    seed
                )
            return thunk

        if self.vectorization_mode == 'async':
            vec_env_cls = gym.vector.AsyncVectorEnv
        else:
            vec_env_cls = gym.vector.SyncVectorEnv

        return vec_env_cls([make_env(i) for i in range(self.num_envs)])

    @property
    def observation_space(self) -> gym.Space:
        """The observation space of a single environment."""
        return self.dummy_env.single_observation_space

    @property
    def action_space(self) -> gym.Space:
        """The action space of a single environment."""
        return self.dummy_env.single_action_space

    @staticmethod
    def _default_env_factory(
        env_spec: 'EnvSpec',
        index: int,
        record_video: bool = False,
        record_video_freq: Optional[int] = None,
        run_log_dir: Optional[str] = None,
        seed: int = 0
    ) -> gym.Env:
        """The default function for creating environments.

        Supports both Gym and DeepMind Control Suite environments. In specific,
        the following DeepMind Control Suite domains are supported, which are
        discretized versions of the original continuous control tasks:

        * walker    (walker/walk)
        * runner    (walker/run)
        * cheetah   (cheetah/run)
        * reacher   (reacher/easy, reacher/hard)
        * hopper    (hopper/hop)
        * fish      (fish/swim)
        * acrobot   (acrobot/swingup)
        * quadruped (quadruped/run)

        All other environments are created using Gym.

        Args:
            env_spec: The environment specification.
            index: The index of the environment.
            record_video: Whether to record a video of the environment.
            record_video_freq: The frequency at which to record videos.
            run_log_dir: The directory where the video will be saved.
            seed: The seed to use for the environment.

        Returns:
            The created Gym environment.
        """
        # Map environment IDs to DeepMind Control Suite domain-task pairs
        DMC_ENVS = {
            'walker': ('walker', 'walk'),
            'runner': ('walker', 'run'),
            'cheetah': ('cheetah', 'run'),
            'reacher_easy': ('reacher', 'easy'),
            'reacher_hard': ('reacher', 'hard'),
            'hopper': ('hopper', 'hop'),
            'fish': ('fish', 'swim'),
            'acrobot': ('acrobot', 'swingup'),
            'quadruped': ('quadruped', 'run')
        }

        # A thunk to create the environment
        def _make_env(render: bool = False) -> gym.Env:
            """Create the environment.

            Args:
                render: Whether to render the environment or not. When True,
                    the `render_mode` is set to `rgb_array`. Default: False.

            Returns:
                The created environment.
            """
            if env_spec.env_id in DMC_ENVS:
                domain_name, task_name = DMC_ENVS[env_spec.env_id]
                return dmc2gym.make(
                    domain_name=domain_name, task_name=task_name, discrete=True)
            else:
                kwargs = {'render_mode': 'rgb_array'} if render else {}
                return gym.make(env_spec.env_id, **kwargs, **env_spec.init_kwargs)

        # Create the environment, wrap it in a RecordVideo wrapper if needed
        if record_video and index == 0:
            env = gym.wrappers.RecordVideo(
                _make_env(render=True),
                f'{run_log_dir}/videos/',
                disable_logger=True,
                episode_trigger=(lambda x: x % record_video_freq == 0)
                if record_video_freq is not None else None
            )
        else:
            env = _make_env(render=False)

        if env_spec.env_id.startswith('MiniGrid'):
            from gym_minigrid.wrappers import ImgObsWrapper
            env = ImgObsWrapper(env)  # For minigrid environments only

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = env_spec.wrap_env_fn(env, env_spec, index)
        env.action_space.seed(seed + index)

        return env
