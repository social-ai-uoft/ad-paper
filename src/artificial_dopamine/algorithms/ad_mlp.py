"""Artificial-Dopamine network."""
import random
import time
from functools import partial
from typing import Any, Callable, Iterable, Optional, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
from flax.training.train_state import TrainState
from tensorboardX import SummaryWriter

import artificial_dopamine.ad_layers as ad
from artificial_dopamine.utils.logging import console

# Type aliases for convenience
Dataset = Iterable[tuple[jax.Array, jax.Array]]
MetricFn = Callable[[jax.Array, jax.Array], jax.Array]


class AggregateLayer(nn.Module):
    """A layer that aggregates the activations from all previous layers.

    This layer is used to combine the activations from all previous layers
    into a single tensor, which is then fed into a final layer to produce the
    final prediction. This is useful for tasks where the input data is
    high-dimensional, such as image classification, since it allows the
    network to learn a low-dimensional representation of the input data
    before making the final prediction.

    Args:
        out_dim: The dimension of the output data.
    """

    out_dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Perform a forward pass through the layer."""\
            # Return a dummy activation to be consistent with the other layers
        y = nn.Dense(features=self.out_dim)(x)
        # log softmax is used for the final layer
        return x, nn.log_softmax(y)


class ADNet:
    r"""Artificial-Dopamine network.

    Args:
        input_dim: The dimension of the input data.
        out_dim: The dimension of the target data.
        net_arch: A list of integers specifying the number of units in each
            hidden layer of the Artificial-Dopamine network.
        loss_fn: The loss function to use. This is either a single function,
            in which case it is used for all layers, or a list of functions,
            one for each layer. The functions should take two arguments: the
            predictions and the targets, and should return a scalar loss value.
        aggregate_layer: Whether the final layer of the network should be an
            aggregate layer. If ``True``, the activations from all previous
            layers will be concatenated and used as the input to the final
            layer. If ``False``, the final layer will only use the activations
            from the previous layer as input. Default: ``True``.
        input_skip_connections: Whether to use skip connections from the input
            data to each layer. Default: ``False``.
        recurrent_connections: Whether to use recurrent connections between
            layers. Default: ``False``.
        forward_connections: Whether to use forward connections between
            layers. Default: ``True``.
        recurrent_weight: The weight to use for recurrent connections.
            Default: ``1.0``.
        forward_conn_weight: The weight to use for forward connections.
            Default: ``1.0``.
        context_size: The number of previous activations to use as context
            for the Artificial-Dopamine layers. This will be ignored if
            there are no recurrent or forward connections. Default: 10.
        context_accumulation_alpha: The weight to use for the context
            accumulation. This is a float between 0 and 1. A value of 0
            means that the context will not be accumulated at all (reset to
            zero at each timestep), and a value of 1 means that a hard update
            will be performed (always replace with new activations). If this
            is a :class:`optax.Schedule`, it should be a function that takes
            integer values between 0 and ``context_size`` and returns a float
            between 0 and 1. Default: 0.7.
        average_predictions: Whether to average the predictions from each
            layer. If ``True``, the final prediction will be the average of
            the predictions from each layer. If ``False``, the final prediction
            will be the prediction from the last layer. Default: ``True``.
        layer_cls: The Artificial-Dopamine layer class to use. Default:
            :class:`artificial_dopamine.ad_layers.FoldingADCell`.
        layer_kwargs: Keyword arguments to pass to the Artificial-Dopamine layers.
        learning_rate: A float or :class:`optax.Schedule` specifying the
            learning rate. Default: 1e-4. If a float is given, a constant
            learning rate schedule will be created using that value.
        max_grad_norm: The maximum norm of the gradients. Default: 10.
            Use ``None`` to disable gradient clipping.
        seed: A seed to use for JAX's PRNG. Default: 1.
        metric_fns: A dictionary mapping metric names to functions that compute
            the metric. The functions should take two arguments: the
            predictions and the targets, and should return a scalar value.
            Default: ``None``.
    """

    # Private Instance Attributes:
    #   _lr_schedule: The learning rate schedule.
    #   _alpha_schedule: The context accumulation weight schedule.
    #   _layers: The Artificial-Dopamine layers.
    #   _states: The TrainState for each layer.
    _lr_schedule: optax.Schedule
    _alpha_schedule: optax.Schedule
    _layers: list[nn.Module]
    _states: list[TrainState]

    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        net_arch: list[int],
        loss_fn: Union[Callable, list[Callable]],
        aggregate_layer: bool = False,
        input_skip_connections: bool = False,
        recurrent_connections: bool = False,
        forward_connections: bool = True,
        recurrent_weight: float = 1.0,
        forward_conn_weight: float = 1.0,
        context_size: int = 10,
        context_accumulation_alpha: Union[float, optax.Schedule] = 0.7,
        average_predictions: bool = False,
        layer_cls: type[ad.ADCell] = ad.FoldingADCell,
        layer_kwargs: Any = {},
        learning_rate: Union[float, optax.Schedule] = 1e-4,
        max_grad_norm: Optional[float] = 10,
        seed: Optional[int] = 1,
        metric_fns: Optional[dict[str, MetricFn]] = None
    ) -> None:
        """Initialize the Artificial-Dopamine network."""
        super().__init__()

        # Store hyperparameters
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.net_arch = net_arch
        self.loss_fn = [loss_fn] * len(net_arch) if not isinstance(
            loss_fn, list) else loss_fn
        self.input_skip_connections = input_skip_connections
        self.recurrent_connections = recurrent_connections
        self.forward_connections = forward_connections
        self.recurrent_weight = recurrent_weight
        self.forward_conn_weight = forward_conn_weight
        self.context_size = (
            context_size
            if recurrent_connections or forward_connections else 1
        )
        self.context_accumulation_alpha = context_accumulation_alpha
        self.average_predictions = average_predictions
        self.aggregate_layer = aggregate_layer
        self.layer_cls = layer_cls
        self.layer_kwargs = layer_kwargs
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.seed = seed
        self.metric_fns = metric_fns or {}

        # Create a JAX PRNG key
        self.seed = seed or int(time.time())
        key = jax.random.PRNGKey(self.seed)
        _, q_key = jax.random.split(key)

        # Define schedules for the learning rate and context accumulation weight
        self._lr_schedule = learning_rate if callable(
            learning_rate) else optax.constant_schedule(learning_rate)
        self._alpha_schedule = (
            context_accumulation_alpha
            if callable(context_accumulation_alpha)
            else optax.constant_schedule(context_accumulation_alpha)
        )

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

        # Create the layers and their states
        self._layers, self._states = [], []
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

            if self.forward_connections and i < len(net_arch) - 1:
                # Layer `i` has connection with the next layer in the past,
                # and receives H_{t-1}^{i+1} as an additional input
                dummy_x_size += net_arch[i + 1]

            layer = layer_cls(
                hidden_size, out_dim, **layer_kwargs)

            train_state = TrainState.create(
                apply_fn=layer.apply,
                params=layer.init(q_key, jnp.zeros((1, dummy_x_size,))),
                tx=grad_transform,
            )
            # Apply jit to the layer for faster inference
            layer.apply = jax.jit(layer.apply)

            self._layers.append(layer)
            self._states.append(train_state)

        # Add an aggregate layer to combine the activations from all previous layers
        if aggregate_layer:
            dummy_x = jnp.zeros((sum(net_arch),))
            self._layers.append(AggregateLayer(out_dim))
            self._states.append(TrainState.create(
                apply_fn=self._layers[-1].apply,
                params=self._layers[-1].init(q_key, dummy_x),
                tx=grad_transform,
            ))
            # Apply jit to the layer for faster inference
            self._layers[-1].apply = jax.jit(self._layers[-1].apply)
            # Extend the loss function list to include the aggregate layer
            self.loss_fn.append(self.loss_fn[-1])

    def _accumulate_context(
        self,
        x: jax.typing.ArrayLike,
        target: bool = False
    ) -> list[list[jax.Array]]:
        """Return contexts accumulated over a sequence of observations.

        Args:
            x: The input data.
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
        return self._accumulate_context_jit(params, x)

    @partial(jax.jit, static_argnames=('self',))
    def _accumulate_context_jit(
        self,
        params: list[flax.core.FrozenDict[str, Any]],
        x: jax.typing.ArrayLike
    ) -> list[list[jax.Array]]:
        """A JIT-compiled helper function."""
        context_acts = [jnp.zeros((x.shape[0], self.net_arch[i]))
                        for i in range(self.num_layers)]

        # Perform ``context_size`` forward passes through the network,
        # accumulating the activations for the next timestep
        for t in range(self.context_size):
            # Forward pass using the previous context
            _, layer_acts, _ = self._forward_helper(self, params, x, context_acts)
            # Soft update
            alpha = self._alpha_schedule(t)
            context_acts = [
                alpha * layer_acts[i] + (1 - alpha) * context_acts[i]
                for i in range(len(context_acts))
            ]

        return context_acts

    def forward(
        self,
        x: jax.typing.ArrayLike,
        last_activations: Optional[list[jax.Array]] = None,
        layer_index: Optional[int] = None
    ) -> tuple[jax.Array, list[jax.Array]]:
        """Predict the output given an input ``x``.

        Args:
            x: The input data.
            last_activations: A list of the activations from each layer of the
                network from the previous timestep. If ``None``, default to
                zeros. Default: ``None``. This will automatically be made to
                match the number of layers in the network, using zeros for any
                layers that don't have activations from the previous timestep.
            layer_index: The index of the layer to return the output of.
                If None, return the output of the last layer. Default: ``None``.

        Returns:
            The output for the given input data, and a list of the
            activations from each layer.

        Raises:
            IndexError: If the layer index is out of range.
        """
        if last_activations is None:
            last_activations = []
        else:
            last_activations = list(last_activations)

        # Pad the list of activations with zeros if necessary
        for i in range(len(last_activations), len(self._layers)):
            last_activations.append(jnp.zeros(
                (x.shape[0], self.net_arch[i])))

        if layer_index is None:
            layer_index = len(self._layers) - 1
        assert 0 <= layer_index < len(self._layers), 'Invalid layer index'

        params = [state.params for state in self._states[:layer_index + 1]]
        return self._forward_helper(self, params, x, last_activations)

    @partial(jax.jit, static_argnames=('cls', 'net'))
    def _forward_helper(
        cls,
        net: 'ADNet',
        params: list[flax.core.FrozenDict[str, Any]],
        x: jax.Array,
        last_activations: list[jax.Array]
    ) -> tuple[jax.Array, list[jax.Array]]:
        """A JIT-compiled forward pass helper function.

        This will iterate through the first ``len(params)`` layers of the
        network and return the output of the last layer it iterates through.

        Args:
            net: The ADNet instance.
            params: The parameters of the network for as many layers as
                should be used in the forward pass. The length of this list
                must be between 1 and the number of layers.
            x: The input data.
            last_activations: The per-layer activations from the last timestep.

        Returns:
            The predictions for the given input data, and a list of the
            activations from each layer.
        """
        pred_ys = jnp.zeros((
            len(params),  # number of layers to iterate through
            x.shape[0],  # batch size
            net.out_dim,  # output dimension
        ))
        activations = []
        weights = []
        h = x

        for i, layer in enumerate(net._layers[:len(params)]):
            if net.aggregate_layer and i == net.num_layers - 1:
                h = jnp.concatenate(activations, axis=-1)
                _, pred_y, w = layer.apply(params[i], h)
            else:
                h_in = net._layer_input(x, h, last_activations, i)
                h, pred_y, w = layer.apply(params[i], h_in)

                activations.append(h)
                weights.append(w)
                pred_ys = pred_ys.at[i].set(pred_y)

        if net.average_predictions:
            pred_y = pred_ys.mean(axis=0)
        else:
            pred_y = pred_ys[-1]

        return pred_y, activations, weights

    @partial(jax.jit, static_argnames=('self', 'layer_index'))
    def _layer_input(
        self,
        x: jax.Array,
        h: jax.Array,
        last_activations: list[jax.Array],
        layer_index: int,
    ) -> jax.Array:
        """Return the input to the given layer.

        Args:
            x: The original (unmodified) input data. This is used for the
                input skip connections.
            h: The input to this layer, i.e. the output of the previous layer
                or the input data if this is the first layer.
            last_activations: The per-layer activations from the last timestep.
            layer_index: The index of the layer to return the input to.

        Preconditions:
            - ``0 <= layer_index < len(self._layers)``

        Returns:
            The input to the given layer, which is the concatenation of the
            input data and any additional recurrent or forward connections.
        """
        h_in = [h]

        if self.input_skip_connections and layer_index > 0:
            # Add the input skip connection from the input data
            h_in.append(x)

        if self.recurrent_connections:
            # Add the recurrent connection from the past
            h_in.append(self.recurrent_weight * last_activations[layer_index])

        if self.forward_connections and layer_index < len(self._layers) - 1:
            # Add the forward connection from the past
            h_in.append(self.forward_conn_weight * last_activations[layer_index + 1])

        return jnp.concatenate(h_in, axis=-1)

    def evaluate(
        self,
        data: Dataset,
        global_step: int = 0,
        writer: Optional[SummaryWriter] = None,
        verbose: int = 0,
        num_epochs: int = 10
    ) -> list[dict[str, float]]:
        r"""Evaluate the network's performance layer-by-layer and log the results.

        Args:
            data: An iterable of tuples containing the evaluation data.
            global_step: The current global step. This is used for logging
                purposes only. Default: ``0``.
            writer: The TensorBoard writer to log the results and game videos
                to. If ``None``, no results or videos will be logged.
                Default: ``None``.
            verbose: The verbosity level. If 0, do not print any messages.
                If 1, print a message at the start of each evaluation. If 2,
                print a message for each epoch, show the progress bar for
                each epoch, and print a summary at the end of each evaluation.
                Default: 0.
            num_epochs: The number of epochs to evaluate for. Default: 10.

        Returns:
            A list of dicts, where each dict contains the evaluation results
            for a single layer of the network, mapping metric names to values.
        """
        if verbose > 0:
            console.log(f'Evaluating network at global step {global_step}...')

        # Evaluate each layer of the network
        eval_results = []
        for i in range(len(self._layers)):
            if verbose > 1:
                console.log(f'\tEvaluating layer {i + 1}...')

            pbar_kwargs = dict(
                disable=not verbose > 1,
                unit='epoch',
                desc=f'Layer {i + 1}',
                total=num_epochs,
            )

            n = 0
            eval_metrics = {}
            for _ in tqdm.trange(1, num_epochs + 1, **pbar_kwargs):
                for x, y in data:
                    # Perform a forward pass through the network
                    context_acts = self._accumulate_context(x)
                    pred_y = self.forward(x, context_acts, i)[0]

                    # Compute the loss
                    loss = self.loss_fn[i](pred_y, y).mean()

                    # Compute the metrics
                    eval_metrics['loss'] = eval_metrics.get('loss', 0) + loss
                    for k, metric_fn in self.metric_fns.items():
                        v = metric_fn(pred_y, y)
                        eval_metrics[k] = eval_metrics.get(k, 0) + v

                    n += 1

            # Average the metrics across all epochs
            eval_metrics = {k: v / n for k, v in eval_metrics.items()}
            eval_results.append(eval_metrics)

            if writer is not None:
                # Log the results to TensorBoard
                for k, v in eval_metrics.items():
                    writer.add_scalar(f'eval/layer_{i + 1}/{k}', v, global_step)

            if verbose > 1:
                console.log(
                    f'\t\t{", ".join([f"{k}: {v}" for k, v in eval_metrics.items()])}'
                )

        return eval_results

    def learn(  # noqa: C901
            self,
            train_data: Dataset,
            eval_data: Optional[Dataset] = None,
            epochs: int = 100,
            log_frequency: int = 100,
            eval: bool = True,
            eval_epochs: int = 1,
            eval_frequency: Optional[int] = None,
            exp_name: str = 'ADNet',
            track: bool = False,
            wandb_project_name: Optional[str] = None,
            wandb_entity: Optional[str] = None,
            show_progress: bool = True,
            verbose: int = 1
    ) -> dict[str, Any]:
        """Train the Artificial-Dopamine network on the given data.

        Args:
            train_data: An iterable of tuples containing the training data.
                Each tuple should contain the input data and the target data.
            eval_data: An iterable of tuples containing the evaluation data.
                Just like ``train_data``, but for evaluation. Default: ``None``.
            epochs: The number of epochs to train for. Default: 100.
            log_frequency: The number of steps between logging progress.
                Default: 100.
            eval: Whether to evaluate the network's performance before, during,
                and after training. Default: ``True``. Each evaluation consists
                of a number of epochs equal to ``eval_epochs``. This is ignored
                if ``eval_data`` is ``None``.
            eval_epochs: The number of epochs to use for each evaluation.
                Default: 10. This is ignored if ``eval`` is ``False``.
            eval_frequency: The number of steps between evaluations.
                Evaluations will be performed before training begins (i.e. on
                the network as it was initialized or last trained), every
                ``eval_frequency`` steps during training, and after
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
            show_progress: Whether to show a progress bar during training.
                Default: ``True``.
            verbose: The verbosity level: 0 none, 1 training information,
                2 debug. Default: 1. If set to 0, the progress bar will not
                be shown, even if ``show_progress`` is ``True``.

        Returns:
            A dictionary containing the training statistics and metadata:

                - ``total_steps``: The total number of training steps.
                - ``wall_time``: The total training time, in seconds.
                - ``total_epochs``: The total number of epochs.
                - ``log_dir``: The path to the TensorBoard log directory.
                - ``eval_results``: A dictionary containing the results of the
                    evaluations, if ``eval`` is ``True``. Each key corresponds
                    to a global step at which the evaluation was performed,
                    and each value is a list of tuples containing metrics for
                    each layer of the network.
        """
        run_name = f'{exp_name}__{self.seed}__{int(time.time())}'
        if track:
            import wandb
            wandb.init(
                project=wandb_project_name,
                entity=wandb_entity,
                sync_tensorboard=True,
                config=self.config,
                name=run_name,
                save_code=True,
            )

        writer = SummaryWriter(f'runs/{run_name}')
        writer.add_text(
            'hyperparameters',
            '|param|value|\n|-|-|\n%s' % ('\n'.join([
                f'|{key}|{value}|'for key, value in self.config.items()
            ])),
        )

        if verbose > 0:
            console.log(
                f'Training {self.__class__.__name__}...')
            console.log('Run name:', run_name)
            console.log('Logging to:', writer.logdir)
            console.log('Logging to Weights & Biases:',
                        'Yes' if track else 'No')
            if track:
                console.log(f'\tWeights & Biases project: {wandb_project_name}')
                console.log(f'\tWeights & Biases entity: {wandb_entity}')
            console.log('Total epochs:', epochs)
            console.log('Log frequency:', log_frequency)
            console.log('Hyperparameters:', self.config)

        # Seeding
        random.seed(self.seed)
        np.random.seed(self.seed)

        global_step = 0
        start_time = time.time()
        eval_results = []

        pbar_kwargs = dict(
            disable=not show_progress or verbose == 0,
            unit='epoch',
            desc='Training',
            total=epochs,
        )

        def _eval(force: bool = False) -> None:
            is_multiple = global_step % eval_frequency == 0
            do_eval = eval and eval_data is not None and (
                eval_frequency is not None and (force or is_multiple))
            if do_eval:
                eval_results.append(self.evaluate(
                    eval_data,
                    global_step=global_step,
                    writer=writer,
                    verbose=verbose,
                    num_epochs=eval_epochs))

        with tqdm.trange(1, epochs + 1, **pbar_kwargs) as progress:
            for _ in progress:
                metrics = {}
                steps_since_last_log = 0

                for x, y in train_data:
                    # Perform a gradient-descent step
                    layer_losses, layer_preds = self._opt_step(x, y)

                    # Record the loss for each layer
                    for i, loss in enumerate(layer_losses):
                        name = f'charts/layer_{i+1}/loss'
                        metrics[name] = metrics.get(name, 0) + loss.item()

                    # Record the metrics for each layer
                    for i, pred in enumerate(layer_preds):
                        for k, metric_fn in self.metric_fns.items():
                            v = metric_fn(pred, y)
                            name = f'charts/layer_{i+1}/{k}'
                            metrics[name] = metrics.get(name, 0) + v.item()

                    # Record the metrics for the entire network
                    steps_per_second = int(global_step / (time.time() - start_time))
                    lr = self._lr_schedule(global_step)
                    metrics['charts/sps'] = metrics.get('charts/sps', 0) + steps_per_second
                    metrics['charts/lr'] = metrics.get('charts/lr', 0) + lr
                    steps_since_last_log += 1

                    # Log metrics
                    if global_step % log_frequency == 0 and steps_since_last_log > 0:
                        # Average the metrics
                        metrics = {k: v / steps_since_last_log for k, v in metrics.items()}

                        # Write metrics to tensorboard and the progress bar
                        for k, v in metrics.items():
                            writer.add_scalar(k, v, global_step)
                        progress.set_postfix({
                            k.split('/')[-1]: str(round(v, 4))
                            for k, v in metrics.items()
                        })

                        # Reset the metrics
                        metrics = {}
                        steps_since_last_log = 0

                    # Evaluate the network
                    _eval()

                    # Increment the global step
                    global_step += 1

        # Evaluate the network on the final state
        _eval(force=True)

        # Write monitor result info to disk
        writer.close()
        if track:
            wandb.finish()

        return dict(
            total_steps=global_step,
            wall_time=time.time() - start_time,
            total_epochs=epochs,
            log_dir=writer.logdir,
            eval_results=eval_results
        )

    def _opt_step(
        self,
        x: jax.Array,
        y: jax.Array
    ) -> tuple[list[float], list[jax.Array]]:
        """Perform an in-place optimization step on the network.

        This will perform a gradient-descent step on each layer of the network
        using the targets ``y`` and the inputs ``x``. This means that a single
        'global' optimization step actually consists of :math:`L` local
        optimization steps w.r.t. the parameters of each layer in the network,
        where :math:`L` is the number of layers.

        Args:
            x: The input data. A batch of data of shape ``(batch_size, input_dim)``.
            y: The target data. A batch of data of shape ``(batch_size, out_dim)``.

        Returns:
            A tuple containing the losses for each layer, and the predictions
            for each layer.
        """
        losses, preds, activations = [], [], []
        context_acts = self._accumulate_context(x)

        h = x
        for i, (layer, state) in enumerate(zip(self._layers, self._states)):
            # Compute the input for this layer
            if self.aggregate_layer and i == len(self._layers) - 1:
                h = jnp.concatenate(activations, axis=-1)
            else:
                h = self._layer_input(x, h, context_acts, i)

            # Perform a local optimization step on the layer
            loss, h, pred, state = self._local_opt_step(
                layer, self.loss_fn[i], state, h, y)

            # Update the state and store the loss and predictions
            losses.append(loss)
            preds.append(pred)
            activations.append(h)
            self._states[i] = state

        return losses, preds

    @classmethod
    @partial(jax.jit, static_argnames=('cls', 'layer', 'loss_fn'))
    def _local_opt_step(
        cls,
        layer: nn.Module,
        loss_fn: MetricFn,
        state: TrainState,
        x: jax.Array,
        y: jax.Array
    ) -> tuple[jax.Array, jax.Array, TrainState]:
        """Perform a local optimization step on the given layer.

        This is a JIT-compiled helper function for the :meth:`_opt_step` method
        that performs a single gradient-descent step on the given layer.

        Args:
            layer: The layer to optimize. This is treated as a constant value
                by the JAX JIT compiler.
            state: The TrainState for the given layer.
            x: Input data with shape ``(batch_size, input_dim)``.
            y: Target data with shape ``(batch_size, out_dim)``.

        Returns:
            The loss averaged across the batch, the predictions for the given
            input data, and the updated TrainState.
        """
        def objective_fn(params: flax.core.FrozenDict[str, Any]) \
                -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
            """The objective function to minimize."""
            h, pred, _ = layer.apply(params, x)
            return loss_fn(pred, y).mean(), (h, pred)

        # Compute the gradients of the objective function with respect to the
        # parameters for the given layer
        (loss_value, (h, pred)), grads = jax.value_and_grad(
            objective_fn, has_aux=True)(state.params)

        state = state.apply_gradients(grads=grads)
        return loss_value, h, pred, state

    @property
    def config(self) -> dict[str, Any]:
        """Return the network configuration."""
        conf = {k: v for k, v in vars(self).items() if not k.startswith('_')}
        return conf

    @property
    def num_layers(self) -> int:
        """Return the number of layers in the network."""
        return len(self._layers)
