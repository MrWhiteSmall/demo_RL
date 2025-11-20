from util_for_argparse import get_cuda_args
get_cuda_args()

from demo_dqn import Agent

import torch
import gymnasium as gym

# 创建测试环境
test_env = gym.make('CartPole-v1', render_mode='rgb_array')
state_dim = test_env.observation_space.shape[0]
action_dim = test_env.action_space.n

# 创建智能体
agent = Agent(state_dim=state_dim, action_dim=action_dim)

# 评估阶段，关闭探索
agent.epsilon = 0

from gymnasium.wrappers import RecordVideo
test_env = RecordVideo(test_env, './demo_dqn_videos', episode_trigger=lambda x: True)

# 加载模型权重（注意：权重需要和结构匹配）
# ckpt = torch.load('./demp_dqn.pth')
ckpt = torch.load('./output/best_model.pth')
agent.q_net.load_state_dict(ckpt)

for episode in range(3):
    state = test_env.reset()[0]
    total_reward = 0
    step = 0

    while True:
        action = agent.choose_action(state)
        next_state, reward, done, _, _ = test_env.step(action)

        total_reward += reward
        state = next_state
        step += 1

        if done or step > 1500:
            break

    print(f"Test Episode: {episode}, Total Reward: {total_reward}, Steps: {step}")

test_env.close()