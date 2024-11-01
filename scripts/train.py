"""Train an Artificial-Dopamine (AD) agent on a task.

Usage:
    python train.py <env> <algo> [options]

See `python train.py --help` for more options.

Hyperparameters are set to the reasonable defaults we found to work well on
MinAtar tasks, as described in our paper. You can adjust them as needed.
"""
import argparse

import optax
from tabulate import tabulate

import artificial_dopamine.ad_layers as ad
from artificial_dopamine.algorithms import AD_DQN, AD_QRDQN
from artificial_dopamine.utils.eval import evaluate_policy


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train an AD agent on a task')
    # Environment and algorithm
    parser.add_argument('env', type=str, help='Environment ID')
    parser.add_argument(
        'algo', choices=['ad_dqn', 'ad_qrdqn'],
        help='Algorithm to use. Choose from: ad_dqn, ad_qrdqn')
    # Model hyperparameters
    parser.add_argument(
        '--net_arch', type=int, nargs='+', default=[128, 96, 96],
        help='Number of units in each hidden layer')
    parser.add_argument(
        '--seed', type=int, default=1977, help='Random seed')
    parser.add_argument(
        '--num_eval_episodes', type=int, default=10,
        help='Number of episodes of policy evaluation')
    parser.add_argument(
        '--total_timesteps', type=int, default=4_000_000,
        help='Total number of timesteps to train the agent for')
    parser.add_argument(
        '--eval_frequency', type=int, default=1_000_000,
        help='Frequency at which to evaluate the policy')

    return parser.parse_args()


def main() -> None:
    """Script entry point."""
    args = get_args()

    # Define model hyperparameters
    model_kwargs = dict(
        # Network architecture configuration
        net_arch=[128, 96, 96],
        input_skip_connections=True,
        recurrent_connections=True,
        forward_connections=True,
        average_predictions=True,  # Final prediction is the average of all layers
        layer_cls=ad.AttentionADCell,  # Use the attention layer
        # Inference configuration
        context_size=10,
        context_accumulation_alpha=1.0,  # Hard update, no accumulation
        # Training configuration
        learning_rate=2.5e-4,
        huber_loss=False,  # Use standard MSE loss
        buffer_size=4_000_000,
        target_network_frequency=1000,
        max_grad_norm=1.0,
        batch_size=512,
        start_eps=1.0,
        end_eps=0.01,
        exploration_fraction=0.1,
        learning_starts=50_000,
        train_frequency=2,
        seed=args.seed,
        double_q=True,
        gamma=0.99
    )

    # Initialize the AD agent, using the specified algorithm, injecting model-
    # specific hyperparameters as needed
    if args.algo == 'ad_dqn':
        model_cls = AD_DQN
    elif args.algo == 'ad_qrdqn':
        model_cls = AD_QRDQN
        model_kwargs['num_quantiles'] = 10
    model = model_cls(args.env, **model_kwargs)

    # Train the agent
    print(f'Training {args.algo} on {args.env}...')
    train_info = model.learn(
        track=True,
        total_timesteps=args.total_timesteps,
        record_video=True,
        eval_frequency=args.eval_frequency,
        save_checkpoints=True
    )

    # Evaluate each layer of the network
    print('Evaluating each layer of the network...')
    eval_info = []
    for i in range(model.num_layers):
        mean_ep_return, std_ep_return = evaluate_policy(
            model.get_policy(layer_index=i),
            model._env_spec.make_env(
                record_video=False,
                record_video_freq=1,  # Record every episode
                run_log_dir=f'{train_info["log_dir"]}/eval/layer_{i + 1}',
                seed=model.seed
            ),
            num_episodes=args.num_eval_episodes,
            show_progress=True
        )

        eval_info.append((
            f'Layer {i + 1}',
            f'{mean_ep_return:.3f} ± {std_ep_return:.3f}')
        )

    print(f'Evaluation results (over {args.num_eval_episodes} episodes):')
    print(tabulate(eval_info, headers=['Layer', 'Episodic return']))


if __name__ == '__main__':
    main()
