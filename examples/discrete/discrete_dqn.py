import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

import tianshou as ts
from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import CollectStats
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.utils.net.common import Net
from tianshou.utils.space_info import SpaceInfo


def main() -> None:
    task = "CartPole-v1"
    # 学习率（learning rate）
    lr = 1e-3
    # 训练轮数（epoch）
    epoch = 10
    # 批量大小（batch size）
    batch_size = 64

    # 训练环境数量（number of parallel training environments）
    num_train_envs = 10
    # 测试环境数量（number of parallel test environments）
    num_test_envs = 100

    # 折扣因子（discount factor，gamma），控制奖励的衰减
    gamma = 0.9
    # n步时序差分（n-step return horizon，n_step），DQN中用于多步回报
    n_step = 3
    # 目标网络更新频率（target network update frequency）
    target_freq = 320

    # 经验回放缓冲区大小（replay buffer size）
    buffer_size = 20000

    # 训练阶段epsilon-greedy策略的epsilon值（训练时探索率）
    eps_train = 0.1
    # 测试阶段epsilon-greedy策略的epsilon值（测试时探索率）
    eps_test = 0.05

    # 每个epoch的总环境步数（number of env steps per epoch）
    epoch_num_steps = 10000
    # 采集数据时每次每个环境采集的步骤数（steps collected per env per collection）
    collection_step_num_env_steps = 10

    logger = ts.utils.TensorboardLogger(SummaryWriter("log/dqn"))  # TensorBoard is supported!
    # For other loggers, see https://tianshou.readthedocs.io/en/master/tutorials/logger.html

    # Create the environments
    # You can also try SubprocVectorEnv, which will use parallelization
    train_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(num_train_envs)])
    test_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(num_test_envs)])

    # Create the network and optimizer
    # Note: You can easily define other networks.
    # See https://tianshou.readthedocs.io/en/master/01_tutorials/00_dqn.html#build-the-network
    env = gym.make(task, render_mode="human")
    assert isinstance(env.action_space, gym.spaces.Discrete)
    space_info = SpaceInfo.from_env(env)
    state_shape = space_info.observation_info.obs_shape
    action_shape = space_info.action_info.action_shape
    net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[128, 128, 128])
    optim = AdamOptimizerFactory(lr=lr)

    policy = DiscreteQLearningPolicy(
        model=net, action_space=env.action_space, eps_training=eps_train, eps_inference=eps_test
    )
    algorithm = ts.algorithm.DQN(
        policy=policy,
        optim=optim,
        gamma=gamma,
        n_step_return_horizon=n_step,
        target_update_freq=target_freq,
    )
    train_collector = ts.data.Collector[CollectStats](
        algorithm,
        train_envs,
        ts.data.VectorReplayBuffer(buffer_size, num_train_envs),
        exploration_noise=True,
    )
    test_collector = ts.data.Collector[CollectStats](
        algorithm,
        test_envs,
        exploration_noise=True,
    )  # because DQN uses epsilon-greedy method

    def stop_fn(mean_rewards: float) -> bool:
        if env.spec:
            if not env.spec.reward_threshold:
                return False
            else:
                return mean_rewards >= env.spec.reward_threshold
        return False

    result = algorithm.run_training(
        OffPolicyTrainerParams(
            train_collector=train_collector,
            test_collector=test_collector,
            max_epochs=epoch,
            epoch_num_steps=epoch_num_steps,
            collection_step_num_env_steps=collection_step_num_env_steps,
            test_step_num_episodes=num_test_envs,
            batch_size=batch_size,
            update_step_num_gradient_steps_per_sample=1 / collection_step_num_env_steps,
            stop_fn=stop_fn,
            logger=logger,
            test_in_train=True,
        )
    )
    print(f"Finished training in {result.timing.total_time} seconds")

    # watch performance
    collector = ts.data.Collector[CollectStats](algorithm, env, exploration_noise=True)
    collector.collect(n_episode=100, render=1 / 35)


if __name__ == "__main__":
    main()
