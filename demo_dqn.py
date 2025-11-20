

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from collections import deque

import random

device = None

class QNet(nn.Module):
    def __init__(self,state_dim, action_dim) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self,x):
        return self.fc(x)

class Agent:
    def __init__(self, state_dim, action_dim) -> None:
        self.q_net = QNet(state_dim=state_dim, action_dim=action_dim).to(device)
        self.t_net = QNet(state_dim=state_dim, action_dim=action_dim).to(device)
        self.best_net = QNet(state_dim=state_dim, action_dim=action_dim).to(device)
        self.t_net.load_state_dict(self.q_net.state_dict())

        self.optim = optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.replay_buffer = deque(maxlen=10000)
        self.bz = 64
        self.gamma = 0.99

        self.epsilon = 0.1 # 探索概率
        self.update_freq = 100 # 更新target 频率
        
        self.step_cnt = 0
        self.best_reward = 0
        self.best_avg_reward = 0

        self.eval_episodes = 5

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            # 注意action_space.n不一定等于2，应该使用action_dim
            # 但本示例用CartPole，为2
            return np.random.randint(0, self.q_net.fc[-1].out_features)
        
        state_tensor = torch.FloatTensor(state).to(device)
        q_values = self.q_net(state_tensor)
        return q_values.cpu().detach().numpy().argmax()

    def store_exp(self,states,actions,rewards,next_states,dones):
        self.replay_buffer.append(
            (states, actions, rewards, next_states, dones)
        )

    def train(self,):
        if len(self.replay_buffer) < self.bz:
            return
        
        batch = random.sample(self.replay_buffer, self.bz)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        actions = torch.LongTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        dones = torch.FloatTensor(np.array(dones)).to(device)

        cur_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_q = self.t_net(next_states).max(1)[0]
            t_q = rewards + self.gamma * next_q * (1-dones)

        loss = nn.MSELoss()(cur_q, t_q)
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()

        self.step_cnt += 1
        if self.step_cnt % self.update_freq == 0:
            # .clone()可省略，但不影响
            self.t_net.load_state_dict(
               self.q_net.state_dict()
                )

    def eval(self, env):
        origin_epsilon = self.epsilon

        self.epsilon = 0  # 评估时关闭探索
        total_rewards = []

        for _ in range(self.eval_episodes):
            state = env.reset()[0]

            episode_reward = 0

            while True:
                action = self.choose_action(state=state)
                next_state, reward, done, _, _ = env.step(action)

                state = next_state

                episode_reward += reward

                if done or episode_reward > 2e4:
                    break
            
            total_rewards.append( episode_reward )

        self.epsilon = origin_epsilon

        return np.mean(total_rewards)

    def save_model(self,path='./demp_dqn.pth'):
        # 保存的应该是cpu权重，兼容性更好
        torch.save(self.q_net.state_dict(), path)
        print(f"Model saved to {path}")


def main():
    from util_for_argparse import get_cuda_args
    get_cuda_args()



    global device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import gymnasium as gym

    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = Agent(state_dim, action_dim)


    total_episodes  = 250
    epsilon_start = 1
    epsilon_end = 0.01
    epsilon_decay = 0.995

    agent.epsilon = epsilon_start

    for episode in range(total_episodes):
        state = env.reset()[0]
        total_reward = 0

        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)

            agent.store_exp( state, action, reward, next_state, done )
            agent.train()

            total_reward += reward

            state = next_state

            if done:
                break

        agent.epsilon = max( epsilon_end, agent.epsilon * epsilon_decay )

        if episode % 10 == 0:
            env_eval = gym.make('CartPole-v1')
            avg_reward = agent.eval(env_eval)
            env_eval.close()

            if avg_reward > agent.best_avg_reward:
                agent.best_avg_reward = avg_reward

                agent.best_net.load_state_dict(
                    agent.q_net.state_dict()
                )
                agent.save_model()
        
        print(f"Episode: {episode}, Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}, Best Avg Reward: {agent.best_avg_reward:.2f}")
    env.close()


    # 训练结束
    agent.epsilon = 0
    test_env = gym.make('CartPole-v1', render_mode='rgb_array')
    from gymnasium.wrappers import RecordVideo
    test_env = RecordVideo(test_env, './demo_dqn_videos',episode_trigger=lambda x: True)
    agent.q_net.load_state_dict(
        agent.best_net.state_dict()
    )

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
            if step > 1500:break

            if done:
                break

        print(f"Test Episode: {episode}, Total Reward: {total_reward}, Steps: {step}")
            
    test_env.close()

if __name__=='__main__':
    main()