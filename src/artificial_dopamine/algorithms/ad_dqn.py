"""Artificial-Dopamine Deep Q-Network (DQN) agent."""
import random
import time
from pathlib import Path
from collections import deque
from functools import partial
from typing import Any, Callable, Optional, Type, Union

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import tqdm
from flax.training.train_state import TrainState
from tensorboardX import SummaryWriter

import artificial_dopamine.ad_layers as ad
from artificial_dopamine.buffers import (TrajectoryReplayBuffer,
                                         TrajectoryReplayBufferSamples)
from artificial_dopamine.features_extractors import (BaseFeaturesExtractor,
                                                     FlattenExtractor)
from artificial_dopamine.utils import EnvSpec, linear_schedule, preprocess_obs
from artificial_dopamine.utils.eval import evaluate_policy
from artificial_dopamine.utils.logging import console


class DQNTrainState(TrainState):
    """Custom DQN training state that also contains target network parameters."""
    target_params: flax.core.FrozenDict[str, Any]


class AD_DQN:
    r"""Artificial-Dopamine (AD) Deep Q-Network (DQN) agent.

    Default hyperparameters are taken from Nature DQN paper, and modified
    as needed to work with AD layers.

    To improve learning stability, we found that the following changes
    helped: using a larger batch size, updating the target network less
    frequently, and annealing the learning rate over time.

    When `backward_connections` is `True`, layers are connected to the
    layer that precedes them in the network, with activations from layer
    :math:`i` being concatenated to the input of layer :math:`i + 1`.
    Similarly, when `recurrent_connections` is `True`, activations from
    layer :math:`i` from timestep :math:`t` are concatenated to the input
    of layer :math:`i` at timestep :math:`t + 1`.

    In order to compute the per-layer activations of the network at timestep
    :math:`t-1`, contiguous episodic sequences with (at most) `context_size`
    transitions are sampled from the replay buffer and "replayed" to compute
    per-layer activations for timestep :math:`t-1`.

    Args:
        env_spec: The environment to train on. Either the ID of a registered
            environment or an :class:`EnvSpec` object.
        features_extractor_cls: The feature extractor class.
            Default: ``FlattenExtractor``.
        net_arch: A list of integers specifying the number of units in each
            hidden layer of the AD network. Default: ``[64, 64]``.
        input_skip_connections: Whether to use skip connections from the input
            data to each layer. Default: ``False``.
        recurrent_connections: Whether to use recurrent connections between
            layers. Default: ``False``.
        backward_connections: Whether to use backward connections between
            layers. Default: ``True``.
        recurrent_weight: The weight to use for recurrent connections.
            Default: ``1.0``.
        backward_weight: The weight to use for backward connections.
            Default: ``1.0``.
        context_size: The number of previous activations to use as context
            for the AD layers. This will be ignored if
            there are no recurrent or backward connections. Default: 10.
        context_accumulation_alpha: The weight to use for the context
            accumulation. This is a float between 0 and 1. A value of 0
            means that the context will not be accumulated at all (reset to
            zero at each timestep), and a value of 1 means that a hard update
            will be performed (always replace with new activations).
            Default: 0.7.
        average_predictions: Whether to average the Q-values from every layer
            when computing the TD target. Default: ``True``.
        layer_cls: The AD layer class to use. Default:
            :class:`artificial_dopamine.ad_layers.FoldingADCell`.
        layer_kwargs: Keyword arguments to pass to the AD layers.
        learning_rate: A float or :class:`optax.Schedule` specifying the
            learning rate. Default: 1e-4. If a float is given, a constant
            learning rate schedule will be created using that value.
        huber_loss: Whether to use the Huber loss function instead of the
            MSE loss function. Default: ``True``.
        double_q: Whether to use the double Q-learning algorithm. Default:
            ``True``.
        buffer_size: The size of the replay buffer. Default: 1_000_000.
        gamma: The discount factor. Default: 0.99.
        tau: The soft update factor. Default: 1.0.
        target_network_frequency: The frequency (in number of steps) at which
            to update the target network. Default: 10000.
        max_grad_norm: The maximum norm of the gradients. Default: 10.
            Use ``None`` to disable gradient clipping.
        batch_size: The batch size to use for training. Default: 32.
        start_eps: The initial exploration rate. Default: 1.0.
        end_eps: The final exploration rate. Default: 0.05.
        exploration_fraction: The fraction of the total number of steps over
            which the exploration rate is annealed from `start_eps` to
            `end_eps`. Default: 0.1.
        learning_starts: The number of steps to wait before starting training.
            Default: 10000.
        train_frequency: The frequency (in number of steps) at which to train
            the network. Default: 4.
        seed: A seed to use for the environment and JAX. Default: 1.
    """

    # Private Instance Attributes:
    #   _lr_schedule: The learning rate schedule.
    #   _rb: The replay buffer.
    #   _env_spec: The environment specification.
    #   _layers: The layers of the AD Q-network (critic).
    #   _states: The training states of the AD Q-network (critic).
    _lr_schedule: optax.Schedule
    _rb: TrajectoryReplayBuffer
    _env_spec: EnvSpec
    _layers: tuple[nn.Module]
    _states: tuple[DQNTrainState]

    def __init__(
            self,
            env_spec: Union[str, EnvSpec],
            features_extractor_cls: Type[BaseFeaturesExtractor] = FlattenExtractor,
            net_arch: list[int] = [64, 64],
            input_skip_connections: bool = False,
            recurrent_connections: bool = False,
            backward_connections: bool = True,
            recurrent_weight: float = 1.0,
            backward_weight: float = 1.0,
            context_size: int = 10,
            context_accumulation_alpha: float = 0.7,
            average_predictions: bool = True,
            layer_cls: Type[ad.ADCell] = ad.FoldingADCell,
            layer_kwargs: Any = {},
            learning_rate: Union[float, optax.Schedule] = 1e-4,
            huber_loss: bool = True,
            double_q: bool = True,
            buffer_size: int = 1_000_000,
            gamma: float = 0.99,
            tau: float = 1.0,
            target_network_frequency: int = 10000,
            max_grad_norm: Optional[float] = 10,
            batch_size: int = 32,
            start_eps: float = 1.0,
            end_eps: float = 0.05,
            exploration_fraction: float = 0.5,
            learning_starts: int = 10000,
            train_frequency: int = 4,
            seed: Optional[int] = 1
    ) -> None:
        """Initialize the DQN agent."""
        super().__init__()

        self._env_spec = EnvSpec(env_spec) if isinstance(
            env_spec, str) else env_spec

        assert isinstance(self._env_spec.action_space, gym.spaces.Discrete), \
            'Only discrete action space is supported, but got: ' \
            f'{self._env_spec.action_space}'

        self.features_extractor = features_extractor_cls(
            self._env_spec.observation_space)
        num_actions = int(self._env_spec.action_space.n)

        # Create a JAX PRNG key
        self.seed = seed or int(time.time())
        key = jax.random.PRNGKey(self.seed)
        _, q_key = jax.random.split(key)

        # Define the learning rate schedule
        self._lr_schedule = learning_rate if callable(
            learning_rate) else optax.constant_schedule(learning_rate)

        # Define an update rule for the gradients based on the Adam optimizer
        grad_clip_fn = optax.clip_by_global_norm(
            max_grad_norm) if max_grad_norm is not None else optax.identity()
        grad_transform = optax.chain(
            grad_clip_fn,  # Clip the gradients by their global norm
            optax.scale_by_adam(b2=0.99),  # Use the updates from Adam
            # Scale the updates by the learning rate schedule
            optax.scale_by_schedule(self._lr_schedule),
            # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
            optax.scale(-1),
        )

        self.input_skip_connections = input_skip_connections
        self.recurrent_connections = recurrent_connections
        self.backward_connections = backward_connections
        self.recurrent_weight = recurrent_weight
        self.backward_weight = backward_weight
        self.context_size = context_size if recurrent_connections or backward_connections else 0
        self.context_accumulation_alpha = context_accumulation_alpha
        self.average_predictions = average_predictions

        # Create the Q-network and layer training states
        input_dim = self.features_extractor.features_dim
        layers, states = [], []
        for i, hidden_size in enumerate(net_arch):
            # All layers receive the last layer's output as input,
            # except for the first layer, which receives the input data
            dummy_x_size = net_arch[i - 1] if i > 0 else input_dim

            if self.input_skip_connections and i > 0:
                # Layer `i` receives the input data as an additional input
                dummy_x_size += input_dim

            if self.recurrent_connections:
                # Layer `i` has connection with itself from the past, and
                # receives H_{t-1}^{i} as an additional input
                dummy_x_size += hidden_size

            if self.backward_connections and i < len(net_arch) - 1:
                # Layer `i` has connection with the next layer in the past,
                # and receives H_{t-1}^{i+1} as an additional input
                dummy_x_size += net_arch[i + 1]

            layer = layer_cls(
                hidden_size, num_actions, **layer_kwargs)

            dummy_x = jnp.zeros((1, dummy_x_size,))
            train_state = DQNTrainState.create(
                apply_fn=layer.apply,
                params=layer.init(q_key, dummy_x),
                target_params=layer.init(q_key, dummy_x),
                tx=grad_transform,
            )
            # Apply jit to the layer for faster inference
            layer.apply = jax.jit(layer.apply)

            layers.append(layer)
            states.append(train_state)

        self._rb = TrajectoryReplayBuffer(
            buffer_size,
            self._env_spec.observation_space,
            self._env_spec.action_space,
            optimize_memory_usage=False,
            handle_timeout_termination=False,
            n_envs=self._env_spec.num_envs
        )
        self._layers = tuple(layers)
        self._states = tuple(states)

        # Store the hyperparameters
        self.net_arch = net_arch
        self.layer_cls = layer_cls
        self.layer_kwargs = layer_kwargs
        self.learning_rate = learning_rate
        self.huber_loss = huber_loss
        self.double_q = double_q
        self.gamma = gamma
        self.tau = tau
        self.target_network_frequency = target_network_frequency
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.exploration_fraction = exploration_fraction
        self.learning_starts = learning_starts
        self.train_frequency = train_frequency

    def forward(
        self,
        obs: jax.typing.ArrayLike,
        last_activations: jax.typing.ArrayLike,
        layer_index: Optional[int] = None,
        target: bool = False
    ) -> tuple[jax.Array, list[jax.Array]]:
        """Predict the Q-values for the given observation.

        Args:
            obs: The observation to predict an action for. This should be a
                batch of raw observations, not extracted features.
            last_activations: A list of the activations from each layer of the
                network from the previous timestep. If ``None``, default to
                zeros. Default: ``None``. This will automatically be made to
                match the number of layers in the network, using zeros for any
                layers that don't have activations from the previous timestep.
            layer_index: The index of the layer to return the output of.
                If None, return the output of the last layer. Default: ``None``.
            target: Whether to perform the forward pass using the target
                network. Default: ``False``.

        Returns:
            The Q-values for the given observation, and a list of the
            activations from each layer with a soft update applied.

        Raises:
            IndexError: If the layer index is out of range.
        """
        if last_activations is None:
            last_activations = []
        else:
            last_activations = list(last_activations)

        # Pad the list of activations with zeros if necessary
        for i in range(len(last_activations), self.num_layers):
            last_activations.append(jnp.zeros(
                (obs.shape[0], self.net_arch[i])))

        # Ensure layer_index is valid
        if layer_index is None:
            layer_index = self.num_layers - 1
        elif layer_index < 0 or layer_index >= self.num_layers:
            raise IndexError(
                f'Layer index must be between 0 and {self.num_layers - 1}, '
                f'but got {layer_index}'
            )

        params = [state.target_params if target else state.params
                  for state in self._states]
        pred_qs, activations = self._forward_jit(params, obs, last_activations)

        if self.average_predictions:
            # Average the first `layer_index` Q-values
            q_values = jnp.mean(pred_qs[:layer_index + 1], axis=0)
        else:
            # Return the Q-values for the given layer
            q_values = pred_qs[layer_index]

        return q_values, activations

    @partial(jax.jit, static_argnames=('self',))
    def _forward_jit(
        self,
        params: list[flax.core.FrozenDict[str, Any]],
        obs: jax.Array,
        last_activations: list[jax.Array],
    ) -> tuple[list[jax.Array], list[jax.Array]]:
        """A JIT-compiled forward pass.

        Args:
            params: The parameters of the network for as many layers as
                should be used in the forward pass. The length of this list
                must be between 1 and the number of layers.
            obs: The observation to predict an action for.
            last_activations: The per-layer activations from the last time step.

        Preconditions:
            - ``len(last_activations) == self.num_layers``

        Returns:
            A list of Q-values for each layer, and a list of the activations
            from each layer with a soft update applied.
        """
        x = self._extract_features(obs)
        pred_qs = jnp.zeros((
            len(params),  # number of layers to iterate through
            x.shape[0],  # batch size
            self._env_spec.action_space.n  # number of actions
        ))
        activations = []

        # Perform a forward pass through each layer
        h = x
        alpha = self.context_accumulation_alpha
        for i, layer in enumerate(self._layers[:len(params)]):
            # Perform the forward pass
            h_in = self._layer_input(x, h, last_activations, i)
            h, pred_q = layer.apply(params[i], h_in)

            # Record the outputs
            activations.append(
                alpha * h + (1 - alpha) * last_activations[i]
            )
            pred_qs = pred_qs.at[i].set(pred_q)

        return pred_qs, activations

    @partial(jax.jit, static_argnames=('self'))
    def _extract_features(self, obs: jax.typing.ArrayLike) -> jax.Array:
        """Get the features from the observation."""
        return self.features_extractor(
            preprocess_obs(obs, self._env_spec.observation_space)
        )

    @partial(jax.jit, static_argnames=('self', 'layer_index'))
    def _layer_input(
        self,
        x: jax.Array,
        h: jax.Array,
        last_activations: list[jax.Array],
        layer_index: int
    ) -> jax.Array:
        """Return the input to the given layer.

        Args:
            x: Extracted features from the observation.
            h: The input to this layer, i.e. the output of the previous layer
                or the input data (``x``) if this is the first layer.
            last_activations: The per-layer activations from the last timestep.
            layer_index: The index of the layer to return the input to.

        Preconditions:
            - ``0 <= layer_index < self.num_layers``

        Returns:
            The input to the given layer, which is the concatenation of the
            input data and any additional recurrent or backward connections.
        """
        # Add temporal encoding from timestep t
        h_in = [h]

        if self.input_skip_connections and layer_index > 0:
            # Add the input skip connection from the input data
            h_in.append(x)

        if self.recurrent_connections:
            # Add the recurrent connection from the past
            recurrent_act = last_activations[layer_index]
            h_in.append(self.recurrent_weight * recurrent_act)

        if self.backward_connections and layer_index < self.num_layers - 1:
            # Add the backward connection from the past
            backward_act = last_activations[layer_index + 1]
            h_in.append(self.backward_weight * backward_act)

        return jnp.concatenate(h_in, axis=-1)

    def predict(
        self,
        obs: jax.Array,
        last_activations: jax.Array,
        **kwargs: Any
    ) -> np.ndarray:
        """Select actions for the given observations using a greedy policy.

        Args:
            obs: The observations to select actions for.
            last_activations: The per-layer activations from the last time step.
            **kwargs: Additional keyword arguments to pass :meth:`forward`.

        Returns:
            The selected actions, as a numpy array.
        """
        q_values, activations = self.forward(obs, last_activations, **kwargs)
        return jax.device_get(jnp.argmax(q_values, axis=-1)), activations

    def get_policy(
        self,
        layer_index: Optional[int] = None
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Return a function that selects an action given an observations.

        Args:
            layer_index: The index of the layer to return the output of.
                If None, return the output of the last layer. Default: ``None``.

        Returns:
            A function that takes in an observation and returns an action.

        Remarks:
            This is a convenience method that wraps the :meth:`predict` method
            in a lambda function with the given ``layer_index``.
        """
        return lambda obs, last_activations: \
            self.predict(obs, last_activations, layer_index=layer_index)

    def evaluate(
        self,
        global_step: int = 0,
        writer: Optional[SummaryWriter] = None,
        verbose: int = 0,
        num_episodes: int = 10,
        eval_envs: Optional[list[gym.Env]] = None
    ) -> tuple[list[tuple[float, float]], list[gym.Env]]:
        r"""Evaluate the agent's performance layer-by-layer and log the results.

        During evaluation, the agent's policy is replaced with a policy that
        only uses the first ``i`` layers of the network, where ``i`` is the
        current layer being evaluated. This allows us to see how the agent's
        performance changes as it learns to use more layers of the network.

        Args:
            global_step: The current global step. This is used for logging
                purposes only. Default: ``0``.
            writer: The TensorBoard writer to log the results and game videos
                to. If ``None``, no results or videos will be logged.
                Default: ``None``.
            verbose: The verbosity level. If 0, do not print any messages.
                If 1, print a message at the start of each evaluation. If 2,
                print a message for each episode, show the progress bar for
                each episode, and print a summary at the end of each evaluation.
                Default: 0.
            num_episodes: The number of episodes to evaluate for each layer.
                Default: 10.
            eval_envs: A list of environments to use for evaluation, one for
                each layer of the network. If ``None``, a new environment will
                be created for each layer using the env spec of this agent.
                Default: ``None``.

        Returns:
            A list of tuples, where each tuple contains the mean and standard
            deviation of the episodic returns for the corresponding layer, and
            a list of the environments used for evaluation.
        """
        if eval_envs is None:
            eval_envs = [
                self._env_spec.make_env(
                    record_video=writer is not None,
                    record_video_freq=1,  # Record every episode
                    run_log_dir=f'{writer.logdir}/eval/layer_{i + 1}' \
                    if writer is not None else None,
                    # Avoid seed collisions with training env(s)
                    seed=self.seed + self._env_spec.num_envs + 1 + i,
                )
                for i in range(self.num_layers)
            ]

        if verbose > 0:
            console.log(f'Evaluating agent at global step {global_step}...')

        # Evaluate each layer of the network
        eval_results = []
        for i, eval_env in enumerate(eval_envs):
            if verbose > 1:
                console.log(f'\tEvaluating layer {i + 1}...')

            mean_ep_return, std_ep_return = evaluate_policy(
                self.get_policy(i),
                eval_env,
                num_episodes=num_episodes,
                show_progress=verbose > 1,
            )

            eval_results.append((mean_ep_return, std_ep_return))
            if writer is not None:
                # Log the results to TensorBoard
                writer.add_scalar(
                    f'eval/layer_{i + 1}/mean_episodic_return', mean_ep_return, global_step)
                writer.add_scalar(
                    f'eval/layer_{i + 1}/std_episodic_return', std_ep_return, global_step)

            if verbose > 1:
                console.log(
                    f'\t\tMean episodic return: {mean_ep_return:.3f} ± {std_ep_return:.3f}')

        # Evaluate the full network
        mean_ep_return, std_ep_return = evaluate_policy(self.get_policy(),
                                                        eval_env,
                                                        num_episodes=num_episodes,
                                                        show_progress=verbose > 1)
        eval_results.append((mean_ep_return, std_ep_return))

        return eval_results, eval_envs

    def learn(  # noqa: C901
            self,
            total_timesteps: int = 500000,
            log_frequency: int = 100,
            eval: bool = True,
            eval_episodes: int = 10,
            eval_frequency: Optional[int] = None,
            exp_name: str = 'FwdDQN',
            save_checkpoints: bool = True,
            track: bool = False,
            wandb_project_name: Optional[str] = None,
            wandb_entity: Optional[str] = None,
            record_video: bool = False,
            show_progress: bool = True,
            verbose: int = 1
    ) -> dict[str, Any]:
        """Train the agent for the given number of timesteps.

        Args:
            total_timesteps: The total number of timesteps to train for.
                Default: 500000.
            log_frequency: The number of timesteps between logging progress.
                Default: 100.
            eval: Whether to evaluate the agent's performance before, during,
                and after training. Default: ``True``. Each evaluation consists
                of a number of episodes equal to ``eval_episodes``.
            eval_episodes: The number of episodes to evaluate the agent for.
                Default: 10. This is ignored if ``eval`` is ``False``.
            eval_frequency: The number of timesteps between evaluations.
                Evaluations will be performed before training begins (i.e. on
                the network as it was initialized or last trained), every
                ``eval_frequency`` timesteps during training, and after
                training completes. If ``None``, evaluations will only be
                performed before training begins and after training completes.
                Default: ``None``. This is ignored if ``eval`` is ``False``.
            exp_name: The name of the experiment.
            track: Whether to track the experiment using Weights & Biases.
                Default: ``False``.
            wandb_project_name: The name of the Weights & Biases project to
                log to. If ``None``, wandb will generate a project name.
                Default: ``None``.
            wandb_entity: The Weights & Biases entity to log to. Default: ``None``.
            record_video: Whether to record a video of the agent's performance
                every ``log_frequency`` timesteps. Default: ``False``.
                The video will be saved to the TensorBoard log directory and,
                if tracking is enabled, uploaded to Weights & Biases.
            show_progress: Whether to show a progress bar during training.
                Default: ``True``.
            verbose: The verbosity level: 0 none, 1 training information,
                2 debug. Default: 1. If set to 0, the progress bar will not
                be shown, even if ``show_progress`` is ``True``.

        Returns:
            A dictionary containing the training statistics and metadata:

                - ``total_timesteps``: The total number of timesteps trained for.
                - ``wall_time``: The total training time, in seconds.
                - ``total_episodes``: The total number of episodes completed.
                - ``episode_infos``: A list of dictionaries containing info
                    about each episode.
                - ``log_dir``: The path to the TensorBoard log directory.
                - ``eval_results``: A dictionary containing the results of the
                    evaluations, if ``eval`` is ``True``. Each key corresponds
                    to a global step at which the evaluation was performed,
                    and each value is a list of tuples containing the mean and
                    standard deviation of the episodic returns for each layer
                    of the network.
        """
        run_name = f'{self._env_spec.env_id}__{exp_name}__{self.seed}__{int(time.time())}'
        if track:
            import wandb
            wandb.init(
                project=wandb_project_name,
                entity=wandb_entity,
                sync_tensorboard=True,
                config=self.config,
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )

        writer = SummaryWriter(f'runs/{run_name}')
        writer.add_text(
            'hyperparameters',
            '|param|value|\n|-|-|\n%s' % ('\n'.join([
                f'|{key}|{value}|'for key, value in self.config.items()
            ])),
        )

        if save_checkpoints:
            # Save the model with the best performance on the evaluation environment every eval_frequency timesteps
            options = ocp.CheckpointManagerOptions(
                max_to_keep=3,
                best_fn=lambda eval_results: eval_results[-1][-1],
                best_mode='max',
            )

            checkpoint_manager = ocp.CheckpointManager(
                (Path(writer.logdir) / 'checkpoints').absolute().as_posix(),
                ocp.PyTreeCheckpointer(),
                options,
            )

        if verbose > 0:
            console.log(
                f'Training {self.__class__.__name__} on {self._env_spec.env_id}...')
            console.log('Run name:', run_name)
            console.log('Logging to:', writer.logdir)
            console.log('Logging to Weights & Biases:',
                        'Yes' if track else 'No')
            if track:
                console.log(f'\tWeights & Biases project: {wandb_project_name}')
                console.log(f'\tWeights & Biases entity: {wandb_entity}')
            console.log('Logging video:', 'Yes' if record_video else 'No')
            console.log('Total timesteps:', total_timesteps)
            console.log('Log frequency:', log_frequency)
            console.log('Hyperparameters:', self.config)

        # Seeding
        random.seed(self.seed)
        np.random.seed(self.seed)

        # The environment to train on
        env = self._env_spec.make_env(
            record_video=record_video, run_log_dir=writer.logdir, seed=self.seed)

        ep_infos = deque(maxlen=log_frequency)
        start_time = time.time()
        obs, _ = env.reset(seed=self.seed)

        # Copy of the environment for each layer, each acting like
        # a "parallel universe" using a different layer's policy
        pu_envs = [
            self._env_spec.make_env(record_video=False, run_log_dir=None, seed=self.seed)
            for _ in range(self.num_layers - 1)
        ]
        pu_obs = [pu_env.reset(seed=self.seed)[0] for pu_env in pu_envs]
        pu_ep_infos = [deque(maxlen=log_frequency) for _ in range(self.num_layers - 1)]
        assert all((obs == pu_obs_i).all() for pu_obs_i in pu_obs), \
            'Environments are not in sync'

        eval_results = []
        eval_envs = None

        def _eval(global_step: int) -> None:
            """Wrapper for conditionally evaluating the agent and mutating ``eval_results``."""
            if not eval:
                return

            # Evaluate if on the first timestep, the last timestep, or at a multiple of ``eval_frequency``
            is_first_timestep = global_step == 0
            is_last_timestep = global_step == total_timesteps - 1
            is_multiple_of_eval_frequency = eval_frequency is not None and global_step % eval_frequency == 0
            if is_first_timestep or is_last_timestep or is_multiple_of_eval_frequency:
                nonlocal eval_envs
                current_eval_results, eval_envs = self.evaluate(
                    global_step,
                    writer,
                    verbose,
                    eval_episodes,
                    eval_envs
                )
                eval_results.append(current_eval_results)

                # Save the model if it's the best so far
                if save_checkpoints:
                    checkpoint_manager.save(global_step, self._states, metrics=eval_results)

        show_progress = show_progress and verbose > 0
        with tqdm.trange(total_timesteps, desc='Training', disable=not show_progress) as progress:
            context_acts = None
            pu_context_acts = [None for _ in range(self.num_layers - 1)]
            for global_step in progress:
                # Get the current epsilon value according to a linear schedule
                epsilon = self._get_epsilon(global_step, total_timesteps)

                # Randomly decide whether to explore or exploit the environment
                # If exp_exp_tradeoff > greater than epsilon --> exploitation
                # Otherwise, --> exploration (random action)
                if random.random() < epsilon:
                    # Take a random action
                    actions = env.action_space.sample()
                else:
                    # Take the best action according to the current Q-network
                    actions, context_acts = self.predict(obs, context_acts)

                # Simulate each parallel universe using the policy of a different layer
                for i, pu_env in enumerate(pu_envs):
                    # Take an action using the policy of the i-th layer
                    if random.random() < epsilon:
                        pu_action = pu_env.action_space.sample()
                    else:
                        pu_action, pu_context_acts[i] = self.predict(
                            pu_obs[i], pu_context_acts[i], layer_index=i)

                    # Execute the action and log data
                    pu_obs[i], _, pu_terminated, pu_truncated, pu_env_infos = pu_env.step(pu_action)

                    # Record rewards for plotting purposes
                    if 'final_info' in pu_env_infos:
                        for info in pu_env_infos['final_info']:
                            # Skip the envs that are not done
                            if 'episode' not in info:
                                continue
                            pu_ep_infos[i].append(info['episode'])

                    # If the episode is terminated or truncated, reset the context
                    if pu_terminated or pu_truncated:
                        pu_context_acts[i] = None

                # Execute the game and log data
                next_obs, rewards, terminated, truncated, infos = env.step(
                    actions)

                # Record rewards for plotting purposes
                if 'final_info' in infos:
                    for info in infos['final_info']:
                        # Skip the envs that are not done
                        if 'episode' not in info:
                            continue

                        episodic_return = info['episode']['r']
                        episodic_length = info['episode']['l']
                        ep_infos.append(info['episode'])

                        writer.add_scalar(
                            'charts/episodic_return', episodic_return, global_step)
                        writer.add_scalar(
                            'charts/episodic_length', episodic_length, global_step)
                        writer.add_scalar('charts/epsilon',
                                          epsilon, global_step)

                # Save data to reply buffer; handle `final_observation`
                real_next_obs = next_obs.copy()
                for i, d in enumerate(truncated):
                    if d:
                        real_next_obs[i] = infos['final_observation'][i]

                self._rb.add(obs, real_next_obs, actions, rewards, terminated, infos)
                obs = next_obs

                # If the episode is terminated or truncated, reset the context
                if terminated or truncated:
                    context_acts = None

                # Train the Q-network
                if global_step > self.learning_starts:
                    if global_step % self.train_frequency == 0:
                        # Sample sub-trajectories from the replay buffer
                        data = self._rb.sample(self.batch_size, seq_len=self.context_size + 1)

                        # Perform a gradient-descent step on the sampled transitions
                        layer_losses, layer_q_values = self._opt_step(data)

                        # Log metrics
                        if global_step % log_frequency == 0:
                            metrics = {}

                            # Record the mean episodic return and length
                            if len(ep_infos) > 0:
                                metrics['charts/mean_episodic_return'] = np.mean([
                                    ep_info['r'] for ep_info in ep_infos]).item()
                                metrics['charts/mean_episodic_length'] = np.mean([
                                    ep_info['l'] for ep_info in ep_infos]).item()

                            for i, pu_ep_infos_i in enumerate(pu_ep_infos):
                                if len(pu_ep_infos_i) > 0:
                                    metrics[f'charts/mean_episodic_return/layer_{i+1}'] = np.mean([
                                        ep_info['r'] for ep_info in pu_ep_infos_i]).item()
                                    metrics[f'charts/mean_episodic_length/layer_{i+1}'] = np.mean([
                                        ep_info['l'] for ep_info in pu_ep_infos_i]).item()

                            # Record the mean TD-loss and Q-values for each layer
                            for i, (loss, q_values) in enumerate(zip(layer_losses, layer_q_values)):
                                metrics[f'losses/layer_{i+1}/td_loss'] = jax.device_get(
                                    loss).item()
                                metrics[f'losses/layer_{i+1}/q_values'] = jax.device_get(
                                    q_values).mean()

                            # Record additional metrics
                            sps = int(global_step / (time.time() - start_time))
                            lr = self._lr_schedule(global_step)

                            metrics['charts/sps'] = sps
                            metrics['charts/epsilon'] = epsilon
                            metrics['charts/learning_rate'] = lr
                            metrics['charts/buffer_size'] = self._rb.size()
                            metrics['charts/rb_pos'] = self._rb.pos

                            # Write metrics to tensorboard and the progress bar
                            for k, v in metrics.items():
                                writer.add_scalar(k, v, global_step)
                            progress.set_postfix({
                                k.split('/')[-1]: str(round(v, 4))
                                for k, v in metrics.items()
                            })

                    # Update target network parameters with a Polyak average
                    if global_step % self.target_network_frequency == 0:
                        new_states = []
                        for i, state in enumerate(self._states):
                            new_states.append(state.replace(
                                target_params=optax.incremental_update(
                                    state.params,
                                    state.target_params,
                                    self.tau
                                )
                            ))
                        self._states = tuple(new_states)

                # Evaluate the Q-network
                _eval(global_step)

        _eval(global_step)

        # Close envs
        if eval_envs is not None:
            for eval_env in eval_envs:
                eval_env.close()
        env.close()

        # Write monitor result info to disk
        writer.close()
        if track:
            wandb.finish()

        # Return training statistics
        return dict(
            total_timesteps=total_timesteps,
            wall_time=time.time() - start_time,
            total_episodes=len(ep_infos),
            episode_infos=ep_infos,
            log_dir=writer.logdir,
            eval_results=eval_results
        )

    def _accumulate_context(
        self,
        obs_seq: jax.typing.ArrayLike,
        target: bool = False
    ) -> list[list[jax.Array]]:
        """Return contexts accumulated over a sequence of observations.

        Args:
            obs_seq: A sequence of raw observations. An array with shape
                ``(seq_len, batch_size, *obs_shape)``.
            target: Whether to perform the forward pass using the target
                network. Default: ``False``.

        Returns:
            A list of lists of per-layer activations, where the outer list
            corresponds to the sequence of observations, and the inner list
            corresponds to the layers of the network. The activations are
            accumulated over the sequence of observations using a soft update
            with weight ``context_accumulation_alpha``.
        """
        params = [state.target_params if target else state.params
                  for state in self._states]
        return self._accumulate_context_jit(params, obs_seq)

    @partial(jax.jit, static_argnames=('self',))
    def _accumulate_context_jit(
        self,
        params: list[flax.core.FrozenDict[str, Any]],
        obs_seq: jax.typing.ArrayLike
    ) -> list[list[jax.Array]]:
        """A JIT-compiled helper function."""
        context_acts = [
            [jnp.zeros((obs_seq[0].shape[0], self.net_arch[i]))
             for i in range(len(self.net_arch))]
        ]

        # Perform ``seq_len`` forward passes through the network,
        # accumulating the activations for the next timestep
        for obs in obs_seq:
            # Forward pass using the previous context
            # NOTE: ``forward_jit`` already applies the soft update
            _, layer_acts = self._forward_jit(params, obs, context_acts[-1])
            context_acts.append(layer_acts)
        return context_acts

    def _opt_step(self, data: TrajectoryReplayBufferSamples) \
            -> tuple[list[float], list[float]]:
        """Perform an in-place optimization step on the online Q-network.

        This will perform a gradient-descent step on each layer of the online
        Q-network using local TD targets computed on a per-layer basis. This
        means that a single 'global' optimization step actually consists of
        :math:`L` local optimization steps w.r.t. the parameters of each layer
        in the Q-network, where :math:`L` is the number of layers in the
        network.

        Args:
            data: The sampled transitions from the replay buffer.

        Returns:
            The TD-loss and Q-values for each layer.
        """
        # Get data from sampled transitions
        obs, obs_seq = data.observations[-1], data.observations
        next_obs = data.next_observations[-1]
        actions = data.actions[-1]
        rewards = data.rewards[-1].flatten()
        dones = data.dones[-1].flatten()

        # Accumulate the context over the sequence of observations
        t_context_acts = self._accumulate_context(obs_seq, target=False)
        t_target_context_acts = self._accumulate_context(obs_seq, target=True)

        # Extract features from the observations
        x = self._extract_features(obs)
        next_x = self._extract_features(next_obs)
        next_x_target = next_x

        # Initialize layer inputs
        h, next_h, next_h_target = x, next_x, next_x_target

        # Perform a local optimization step on each layer of the network
        losses, preds = [], []
        new_states = []
        for i, (layer, state) in enumerate(zip(self._layers, self._states)):
            # Compute the input activations for the current layer
            h = self._layer_input(x, h, t_context_acts[-2], i)
            next_h = self._layer_input(next_x, next_h, t_context_acts[-1], i)
            next_h_target = self._layer_input(
                next_x_target, next_h_target, t_target_context_acts[-1], i)

            # Perform a local optimization step on the layer
            loss, pred, state, (h, next_h, next_h_target) = self._local_opt_step(
                layer,
                state,
                h,  # observations
                actions,
                next_h,  # next observations
                next_h_target,  # next observations from target network
                rewards,
                dones
            )

            # Update the state and store the loss and predictions
            losses.append(loss)
            preds.append(pred)
            new_states.append(state)

        self._states = tuple(new_states)
        return losses, preds

    @partial(jax.jit, static_argnames=('self', 'layer'))
    def _local_opt_step(
        self,
        layer: nn.Module,
        state: DQNTrainState,
        x: jax.typing.ArrayLike,
        actions: jax.typing.ArrayLike,
        next_x: jax.typing.ArrayLike,
        next_x_target: jax.typing.ArrayLike,
        rewards: jax.typing.ArrayLike,
        dones: jax.typing.ArrayLike
    ) -> tuple[jax.Array, jax.Array, DQNTrainState]:
        """Perform a local optimization step on the given Q-network layer.

        This is a JIT-compiled helper function for the :meth:`_opt_step` method
        that performs a single gradient-descent step on the given Q-network
        layer using local TD targets.

        Args:
            layer: The layer to perform the optimization step on.
            state: The training state of the layer.
            x: The input to the layer at timestep :math:`t`.
            actions: The actions from the replay buffer.
            next_x: The input to the layer at timestep :math:`t+1`.
            next_x_target: The input to the target layer at
                timestep :math:`t+1`.
            rewards: The rewards from the replay buffer.
            dones: The dones from the replay buffer.

        Returns:
            The TD-loss, the Q-values for the given layer, and the updated
            Q-network layer state.
        """
        next_h, next_q_preds = layer.apply(state.params, next_x)
        next_h_target, next_target_q_preds = layer.apply(state.target_params, next_x_target)  # (batch_size, num_actions)
        if self.double_q:
            # Double DQN: select actions for the next states using the
            # online network; a_t+1 = argmax_a Q(s_t+1, a; theta)
            next_actions = jnp.argmax(next_q_preds, axis=-1)  # (batch_size,)

            # Compute the Q-values in the next state for the chosen next actions using the target network
            q_next_target = jnp.take_along_axis(next_target_q_preds, next_actions[:, None], axis=-1)[:, 0]  # (batch_size,)
        else:
            # Regular DQN: compute the Q-values in the next state using the target network
            q_next_target = jnp.max(next_target_q_preds, axis=-1)  # (batch_size,)

        # Compute the temporal difference (TD) targets using the Bellman equation
        # Note that we use a mask to zero out the Q-values for the terminal states
        next_q_value = rewards + (1 - dones) * self.gamma * q_next_target  # (batch_size,)

        def loss_fn(params: flax.core.FrozenDict[str, Any]) \
                -> tuple[jax.Array, tuple[jax.Array]]:
            """The Q-learning objective function to minimize."""
            # Compute Q(s_t, a) - the Q values for the current state
            #
            # The model computes Q(s_t), then we select the columns
            # of actions taken. These are the actions which would've
            # been taken for each batch state according to the q_net
            h, q_pred = layer.apply(params, x)  # (batch_size, num_actions)
            q_pred = jnp.take_along_axis(q_pred, actions, axis=-1).squeeze()  # (batch_size,)

            if self.huber_loss:
                # Use Huber (smooth L1) loss
                td_loss = optax.huber_loss(q_pred, next_q_value).mean()
            else:
                # Use MSE loss
                td_loss = optax.squared_error(q_pred, next_q_value).mean()

            return td_loss, (q_pred, h)

        # Compute the gradients of the objective function with respect to the
        # Q-network parameters for the given layer
        (loss_value, (q_pred, h)), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(state.params)

        state = state.apply_gradients(grads=grads)
        return loss_value, q_pred, state, (h, next_h, next_h_target)

    def _get_epsilon(self, step: int, total_timesteps: int) -> float:
        """Return the epsilon value for the given training step.

        The epsilon value is linearly annealed from ``start_eps`` to
        ``end_eps`` over the first ``exploration_fraction`` of the total
        training steps, excluding the ``learning_starts`` steps at the
        beginning of training, and then kept constant for the remainder
        of training.

        Args:
            step: The current training step.
            total_timesteps: The total number of training steps.

        Returns:
            The epsilon value for the given training step.
        """
        return linear_schedule(
            self.start_eps,
            self.end_eps,
            int(self.exploration_fraction * total_timesteps),
            max(step - self.learning_starts, 0),
        )

    @property
    def config(self) -> dict[str, Any]:
        """Return the network configuration."""
        conf = {k: v for k, v in vars(self).items() if not k.startswith('_')}
        conf['buffer_size'] = self._rb.buffer_size
        return conf

    @property
    def num_layers(self) -> int:
        """Return the number of layers in the network."""
        return len(self._layers)
