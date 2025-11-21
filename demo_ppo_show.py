from util_for_argparse import get_cuda_args
get_cuda_args()

from demo_ppo import Agent
import gymnasium as gym
import torch

env = gym.make('Pendulum-v1', render_mode='rgb_array')

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = Agent(state_dim, action_dim)

agent.actor.load_state_dict( torch.load('./ppo_actor.pth')) # 只需要actor

from gymnasium.wrappers import RecordVideo
env = RecordVideo(env, './demo_ppo_videos',episode_trigger=lambda x: True)

for episode in range(3):
    state = env.reset()[0]
    episode_rewards = 0
    step = 0 # 计数 多少步
    while True:
        step += 1
        state_tensor = torch.FloatTensor(state).cuda()
        action, log_prod = agent.choose_action(state_tensor)

        next_state,reward,done, truncated ,_ = env.step(action)

        episode_rewards += reward

        if done or truncated or step > 1500 : break
        state = next_state
        # INSERT_YOUR_CODE
    print(f"Episode: {episode}, Total Reward: {episode_rewards:.4f}, Total Step: {step}")

env.close()