"""Shared layers for Artificial-Dopamine (AD) models."""
from abc import ABC
from dataclasses import field
from functools import partial
from typing import Callable, Optional, Sequence, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp


class ADCell(nn.Module, ABC):
    r"""Base class for Artificial-Dopamine cells.

    The forward pass computes hidden activations :math:`h` and a predicted
    output :math:`y` from an input :math:`x`.

    Args:
        hidden_features: the number of hidden features.
        out_features: number of output features.

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Activations: :math:`(N, *, H_{act})` where all but the last dimension are
          the same shape as the input and :math:`H_{act} = \text{hidden\_features}`.
        - Output: :math:`(N, *, H_{out})` where all but the last dimension are
          the same shape as the input and :math:`H_{out} = \text{out\_features}`.
    """

    hidden_features: int
    out_features: int

    def setup(self) -> None:
        """Setup the layer."""
        raise NotImplementedError

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Compute the forward pass.

        Args:
            x: The input tensor.

        Returns:
            The hidden activations and the output tensor.
        """
        raise NotImplementedError


class MlpADCell(ADCell):
    r"""An Artificial-Dopamine cell that uses a 2-layer MLP for the forward pass.

    A linear transformation is applied to the input to produce hidden activations,
    which are then passed through a second linear transformation to produce the
    output.

    Args:
        hidden_features: the number of hidden features.
        out_features: number of output features.
        use_bias: whether to add a bias term to the output. Default: ``True``
        act_fn: the activation function to use for the layer.
            Default: :func:`nn.relu`
        normalization_method: The normalization method to use for the output.
            Must be one of ``'lp'``, ``'scaled'``, ``'layer_norm'``, or
            None. Default: ``'lp'``.
        normalization_kwargs: Keyword arguments to pass to the normalization
            function. See :meth:`normalize` for details.

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Activations: :math:`(N, *, H_{act})` where all but the last dimension are
          the same shape as the input and :math:`H_{act} = \text{hidden\_features}`.
        - Output: :math:`(N, *, H_{out})` where all but the last dimension are
          the same shape as the input and :math:`H_{out} = \text{out\_features}`.
    """

    use_bias: bool = True
    act_fn: Callable[[jax.Array], jax.Array] = nn.relu
    normalization_method: Optional[str] = 'lp'
    normalization_kwargs: flax.core.FrozenDict = field(
        default_factory=lambda: flax.core.FrozenDict()
    )

    def setup(self) -> None:
        """Setup the layer."""
        self.fc_h = nn.Dense(self.hidden_features, use_bias=self.use_bias)
        self.fc_y = nn.Dense(self.out_features, use_bias=self.use_bias)
        self.layer_norm = nn.LayerNorm(epsilon=1e-6)

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Compute the forward pass.

        Args:
            x: The input tensor.

        Returns:
            The hidden activations and the output tensor.
        """
        h = self.fc_h(x)
        h, y = self.compute_output(x, h)
        if self.normalization_method is not None:
            y = self.normalize(y, **self.normalization_kwargs)

        return h, y

    def compute_output(self, _: jax.Array, h: jax.Array) \
            -> tuple[jax.Array, jax.Array]:
        """Compute the output of the layer.

        Args:
            x: The input tensor.
            h: Raw output of the linear layer.

        Returns:
            The hidden activations and the output tensor. By default, the
            activation function is only applied to the hidden activations.
        """
        h = self.act_fn(h)
        y = self.fc_y(h)
        return h, y

    def normalize(
        self,
        x: jax.Array,
        mode: str = 'lp',
        p: float = 2,
        epsilon: float = 1e-4,
        alpha: float = 0.5,
        axis: int = -1
    ) -> jax.Array:
        """Normalize the output according to the given `mode`.

        The normalization modes are:

        * ``'lp'``: Normalize the output so that it has unit :math:`L_p` norm.
        * ``'scaled'``: Normalize the output so that it has unit norm scaled by
            :math:`H**alpha`, where :math:`H` is the number of features.
        * ``'gaussian'``: Normalize the output so that it has zero mean and unit
            variance.
        * ``'layer_norm'``: Apply layer normalization to the output.

        This helps with training stability as the dot product tends to
        get very large as the number of features increases.

        Args:
            x: The input tensor.
            mode: The normalization mode to use. Must be one of ``'lp'``,
                ``'scaled'``, or ``'layer_norm'``. Default: ``'lp'``.
            p: The norm to use for the ``'lp'`` mode. Default: ``2``.
            epsilon: A small value to add to the :math:`L_p` norm to avoid
                division by zero. Default: ``1e-4``.
            alpha: Exponent coefficient for the ``'scaled'`` mode.
                Default: ``0.5``
            axis: The axis along which to compute the norm. Default: ``-1``.

        Returns:
            The normalized output tensor.
        """
        if mode == 'lp':
            return x / (jnp.linalg.norm(x, ord=p, axis=axis, keepdims=True) + epsilon)
        elif mode == 'scaled':
            return x / x.shape[axis]**alpha
        elif mode == 'gaussian':
            mean = jnp.mean(x, axis=axis, keepdims=True)
            std = jnp.std(x, axis=axis, keepdims=True)
            return (x - mean) / (std + epsilon)
        elif mode == 'layer_norm':
            return self.layer_norm(x)
        else:
            raise ValueError(
                f'Invalid normalization mode {mode}. Must be one of '
                f'"lp", "scaled", or "layer_norm".'
            )


@partial(jax.jit, static_argnames=('k', 'pad'))
def k_folding(x: jax.Array, p0: jax.Array, k: int, pad: bool = True) \
        -> jax.Array:
    r"""Perform a k-folding operation on the input array.

    Applies :math:`k` equally spaced folds to the input array along the
    last dimension to produce an array of shape :math:`(N, *, k)`.

    The output is a :math:`k`-dimensional vector where the :math:`i`-th component
    is the dot product between the activations and the :math:`i`-th weight vector.

    Args:
        x: The input array. An array of shape :math:`(N, *, H_{in})` where
            :math:`*` means any number of dimensions including none and
            :math:`H_{in}` is the number of input features. Note that the last
            dimension must be divisible by :math:`k`.
        p0: The activations. An array of shape where all but the last dimension
            are the same shape as ``x``. The last dimension should be the size
            of a single fold, i.e. :math:`H_{in} / k`.
        k: The number of folds to apply to the input array. Must be positive.
        pad: If ``True``, the input array is padded with zeros along the last
            dimension to ensure that it is divisible by ``k``. The activations
            ``p0`` are also padded with zeros to match the shape of each folded
            component of the input array. If ``False``, an error is raised if
            the last dimension of ``x`` is not divisible by ``k``.

    Returns:
        An array of shape :math:`(N, *, k)` where all but the last dimension are
        the same shape as ``x``.

    Remarks:
        The k-folding operation can be thought of a downsampling operation where
        the :math:`i`-th component of the output array is a linear combination
        of the activations with coefficients determined by the :math:`i`-th
        partition of the input array.

    Raises:
        ValueError: If ``k`` is not positive.
        ValueError: If the shape of ``x`` and ``p0`` are not the same except
            for the last dimension.
        ValueError: If the last dimension of ``x`` is not divisible by ``k``
            and ``pad`` is ``False``.

    Examples::

        >>> x = jax.random.normal(jax.random.PRNGKey(0), (128, 256))
        >>> p0 = jax.random.normal(jax.random.PRNGKey(0), (128, 64))
        >>> y = k_folding(x, p0, 4)
        >>> y.shape
        (128, 4)
        >>> x = jnp.asarray([[1, 2, 3, 4, 5, 6, 7, 8]])
        >>> p0 = jnp.asarray([[-1, 2, 0.5, 0]])
        >>> y = k_folding(x, p0, 2)
        >>> y.tolist()
        [[4.5, 10.5]]
    """
    # Ensure that k is positive
    if k <= 0:
        raise ValueError(f'k must be positive, but got {k}.')

    # Ensure that x and p0 have the same shape except for the last dimension
    if x.shape[:-1] != p0.shape[:-1]:
        raise ValueError(
            f'Input array and activations must have the same shape except '
            f'for the last dimension, but got {x.shape} and {p0.shape}.'
        )

    # Ensure that the last dimension of x is divisible by k, padding if necessary
    h_in = x.shape[-1]
    if h_in % k != 0:
        if pad:
            x = jnp.pad(x, ((0, 0),) * (x.ndim - 1) + ((0, k - h_in % k),))
            h_in = x.shape[-1]
        else:
            raise ValueError(
                f'Input array must have a last dimension divisible by '
                f'{k}, but has shape {x.shape}. Please set pad=True to '
                f'automatically pad it with zeros.'
            )

    # Ensure that p0 has the correct number of features
    chunk_size = int(h_in // k)
    if p0.shape[-1] != chunk_size:
        if pad:
            p0 = jnp.pad(p0, ((0, 0),) * (p0.ndim - 1) + ((0, chunk_size - p0.shape[-1]),))
        else:
            raise ValueError(
                f'Activations must have {chunk_size} features, but got '
                f'{p0.shape[-1]} instead. Please set pad=True to '
                f'automatically pad it with zeros.'
            )

    # Reshape x to be a (k x chunk_size) matrix of weights
    x = x.reshape(x.shape[:-1] + (k, chunk_size))
    return jnp.matmul(x, jnp.expand_dims(p0, axis=-1)).squeeze(-1)


class FoldingADCell(MlpADCell):
    r"""An Artificial-Dopamine cell that uses a 2-layer MLP with K-folding.

    Four folding modes are supported, which differ in how the layer is
    structured and how the output is computed:

    * Compressive folding: The layer has :math:`\text{hidden\_features}` units and
        is divided into :math:`\text{out\_features} + 1` equal parts, such that
        each part is :math:`\text{hidden\_features} / (\text{out\_features} + 1)`
        units wide. The output of the layer is a concatenation of all the parts,
        i.e. the activations and the output weights.

    * Expansive folding: The layer is actually :math:`\text{out\_features} + 1`
        times wider (larger than :math:`\text{hidden\_features}`), and each part is
        :math:`\text{hidden\_features}` units wide. The activations of the
        layer is the first part, i.e. the activations.

    * Attention folding: The layer has :math:`\text{hidden\_features}` units and
        consists of two dense layers: the first for projecting the input into
        activations :math:`a` and the second for projecting the activations into
        the output weights :math:`w_1, \dots, w_{\text{out\_features}}`.

    * Dual folding: The layer has :math:`\text{hidden\_features}` units and
        consists of two dense layers: both take the input and project it into
        activations and output weights, respectively and independently of each
        other. This is similar to the attention folding layer, except that the
        there is no composition between the activations and the output weights.

    For compressive and expansive folding, the first part denotes the activations
    and the remaining parts denote the output weights, which is a matrix of
    shape :math:`(\text{out\_features}, \text{hidden\_features})`.

    In general, the output :math:`y= (y_1, \dots, y_{\text{out\_features}})`
    is a linear combination of the output weights and the activations, given by:

    .. math::
        y_i = w_i \cdot a = \sum_{j=1}^{|w_i|} w_{ij} a_j,

    where :math:`w_i` is the :math:`i`-th row of the weight matrix and :math:`a`
    are the activations.

    Args:
        hidden_features: the number of hidden features.
        out_features: number of output features.
        use_bias: whether to add a bias term to the output. Default: ``True``
        act_fn: The activation function to use for the layer.
            Default: :func:`nn.relu`
        folding_mode: The folding mode to use. Must be one of ``'compressive'``,
            ``'expansive'``, or ``'attention'``. Default: ``'compressive'``.
        weight_fn: The transformation function to use for the attention weights.
            Default: :func:`nn.tanh`
        ouput_act_fn: The transformation function to use for the output.
            Default: a function that returns its input.

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Activations: :math:`(N, *, H_{act})` where all but the last dimension are
          the same shape as the input and :math:`H_{act} = \text{hidden\_features}`.
        - Output: :math:`(N, *, H_{out})` where all but the last dimension are
          the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Examples::

        >>> m = FoldingFwdLinear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> h, y = m(input)
        >>> print(h.size(), y.size())
        torch.Size([128, 30]) torch.Size([128, 1])

    Remarks:
        When `folding_mode` is "expansive",  this layer has
        :math:`\text{hidden\_features} \times (\text{out\_features} + 1)`
        actual output features.
    """

    folding_mode: str = 'compressive'
    weight_fn: Callable[[jax.Array], jax.Array] = nn.tanh
    output_act_fn: Callable[[jax.Array], jax.Array] = lambda x: x

    def setup(self) -> None:
        """Setup the layer."""
        if self.folding_mode == 'expansive':
            # Expansive folding increases the number of features
            self.h_features = self.hidden_features
            out_features = (self.out_features + 1) * self.hidden_features
        elif self.folding_mode == 'compressive':
            # Compressive folding keeps the number of features the same
            self.h_features = self.hidden_features // (self.out_features + 1)
            out_features = self.hidden_features
        elif self.folding_mode == 'attention' or self.folding_mode == 'dual':
            # Attention folding keeps the number of features the same
            self.h_features = self.hidden_features
            out_features = self.hidden_features
            # attn_dense projects the activations into a flattened weight matrix
            self.attn_dense = nn.Dense(self.out_features * out_features, use_bias=False)
        else:
            raise ValueError(
                f'Invalid folding mode {self.folding_mode}. Must be one of '
                f'"compressive", "expansive", "attention", or "dual".'
            )

        self.dense = nn.Dense(out_features, use_bias=self.use_bias)

    def compute_output(self, x: jax.Array, h: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Computes the output."""
        if self.folding_mode == 'attention':
            # Attention folding: activations and attention weights are computed together
            h = self.act_fn(h)
            w = self.attn_dense(h)
            w = self.weight_fn(w)
        elif self.folding_mode == 'dual':
            # Dual folding: activations and attention weights are computed independently
            h = self.act_fn(h)
            w = self.attn_dense(x)
            w = self.weight_fn(w)
        else:
            # Compressive and expansive folding
            chunks = jnp.split(h, [self.y_features], axis=-1)
            h = self.act_fn(chunks[0])
            w = self.weight_fn(chunks[1])

        # Normalize weights to have zero mean and unit variance
        w = (w - jnp.mean(w, axis=-1, keepdims=True)) / (jnp.std(w, axis=-1, keepdims=True) + 1e-6)

        if self.folding_mode == 'compressive':
            h = jnp.pad(h, ((0, 0),) * (h.ndim - 1) + ((0, self.features - self.y_features),))

        y = k_folding(w, h, self.attention_features)
        return h, y

    @staticmethod
    def _he_kernel_init(
        in_axis: Union[int, Sequence[int]] = -2,
        out_axis: Union[int, Sequence[int]] = -1,
        batch_axis: Sequence[int] = (),
        dtype=jnp.float_,
    ) -> jax.nn.initializers.Initializer:
        """Return a kernel initializer function for a dense layer."""
        return jax.nn.initializers.variance_scaling(
            0.3333,
            mode='fan_in',
            distribution='uniform',
            in_axis=in_axis,
            out_axis=out_axis,
            batch_axis=batch_axis,
            dtype=dtype
        )


class AttentionADCell(ADCell):
    r"""An Artificial-Dopamine cell with an attention mechanism.

    Args:
        hidden_features: the number of features in the hidden activations
        out_features: number of features in the output
        num_heads: The number of heads. Default: ``8``
        attn_weights: Whether to return the attention weights in the
            forward pass. Default: ``False``.

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Activations: :math:`(N, *, H_{act})` where all but the last dimension are
          the same shape as the input and :math:`H_{act} = \text{hidden\_features}`.
        - Output: :math:`(N, *, H_{out})` where all but the last dimension are
          the same shape as the input and :math:`H_{out} = \text{out\_features}`.
        - Attention weights: :math:`(N, *, H_{act}, H_{out})` where all but the
            last two dimensions are the same shape as the input and
            :math:`H_{act} = \text{hidden\_features}` and
            :math:`H_{out} = \text{out\_features}`.
    """
    num_heads: int = 8
    attn_weights: bool = False

    def setup(self) -> None:
        """Setup the layer."""
        self.fc_h = nn.Dense(self.hidden_features)
        self.fc_z1 = nn.Dense(self.num_heads * self.hidden_features, use_bias=False)
        self.fc_z2 = nn.Dense(self.num_heads * self.out_features, use_bias=False)
        self.layer_norm = nn.LayerNorm(1e-6)

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Compute the forward pass.

        Args:
            x: The input tensor.
            weights: Whether to return the weights. Default: ``False``.

        Returns:
            The output tensor and the attention weights.
        """
        num_heads, d_h, d_g = self.num_heads, self.hidden_features, self.out_features
        sz_b = x.shape[0]

        # Apply linear projection fc_h to get the hidden activations
        h = self.layer_norm(nn.relu(self.fc_h(x)))

        # Apply linear projections fc_z1 and fc_z2
        z1 = jnp.reshape(self.fc_z1(x), (sz_b, num_heads, d_h))  # (sz_b, num_heads, d_h)
        z2 = jnp.reshape(self.fc_z2(x), (sz_b, num_heads, d_g))  # (sz_b, num_heads, d_g)

        # Multiply z1 and z2 to get a d_g x d_h weight matrix
        w = jnp.matmul(z2.transpose(0, 2, 1), z1)  # (sz_b, d_g, d_h)
        w = (w - jnp.mean(w, axis=-1, keepdims=True)) / (jnp.std(w, axis=-1, keepdims=True) + 1e-6)
        w = nn.tanh(w)

        # Apply the weight matrix to the hidden activations to get the output
        y = jnp.matmul(w, jnp.expand_dims(h, axis=-1)).squeeze(-1)  # (sz_b, d_g)

        if self.attn_weights:
            return h, y, w
        else:
            return h, y
