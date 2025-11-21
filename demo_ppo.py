from util_for_argparse import get_cuda_args
get_cuda_args()

import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
import torch
import numpy as np
import gymnasium as gym

import swanlab  # 引入swanlab用于记录和可视化

# 选择设备：优先使用CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super(Net, self).__init__()
        # 简单的三层MLP神经网络
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.fc(x)


class Agent:
    def __init__(self, state_dim, action_dim, lr=1e-3) -> None:
        # Actor和Critic网络
        self.actor = Net(state_dim, action_dim).to(device)
        self.critic = Net(state_dim, 1).to(device)

        # 优化器
        self.optim_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=lr)

        # 动作协方差
        self.conv_var = torch.full(size=(action_dim,), fill_value=0.5).to(device)
        self.conv_mat = torch.diag(self.conv_var).to(device)

        # 超参数
        self.gamma = 0.95
        self.total_steps = 4800
        self.episode_max_steps = 1600
        self.total_train_steps = 200_000_000
        # self.total_steps = 48  # 每轮累积步数
        # self.episode_max_steps = 16  # 每回合最大步数
        # self.total_train_steps = 200 # 总训练"步"
        self.inner_loop_times = 5
        self.save_freq = 10 # 每多少轮保存一次模型
        self.clip = 0.2 # PPO裁剪系数

    def choose_action(self, state):
        """
        给定一个state（tensor），输出actor网络输出的动作采样和log_prob。
        """
        action = self.actor(state)
        # 构建多元高斯分布
        dist = MultivariateNormal(action, self.conv_mat)
        # 动作采样
        action_sample = dist.sample()
        log_prod = dist.log_prob(action_sample)  # 采样动作的概率密度log值

        return action_sample.detach().cpu().numpy(), log_prod.detach()

    def collect_data(self, env):
        """
        在环境中采集一批数据，累计到设定的步数。
        """
        bz_state = []
        bz_action = []
        bz_log_prod = []

        bz_episode_len = []
        bz_episode_reward = []
        # 一共bz个样本
        # 每个样本内是一个完整的游戏状态 长度不等
        bz_step = 0
        # 使得总的数量级大于total ,如果使用bz来控制,也就默认每次玩bz把游戏,如果游戏很快结束,就收集不到比较多的数据
        while bz_step < self.total_steps: 

            state = env.reset()[0]

            episode_reward = []

            for step_i in range(self.episode_max_steps):
                bz_step += 1
                # 状态转为Tensor
                state_tensor = torch.FloatTensor(state).to(device)
                action, log_prod = self.choose_action(state_tensor)
                next_state, reward, done, truncated, _ = env.step(action)
                episode_reward.append(reward)
                bz_state.append(state)
                bz_action.append(action)
                bz_log_prod.append(log_prod.cpu().numpy())
                if done or truncated:
                    break
                state = next_state

            bz_episode_len.append(step_i + 1)
            bz_episode_reward.append(episode_reward)

        # 用numpy先转array再转tensor，避免直接list转tensor的性能和类型问题
        batch_state = torch.from_numpy(np.array(bz_state, dtype=np.float32)).to(device)
        batch_action = torch.from_numpy(np.array(bz_action, dtype=np.float32)).to(device)
        batch_log_prod = torch.from_numpy(np.array(bz_log_prod, dtype=np.float32)).to(device)
        batch_rtgs = self.get_rtgs(bz_episode_reward).to(device)

        return batch_state, batch_action, batch_log_prod, batch_rtgs, bz_episode_len

    def train(self, env):
        """
        PPO主训练循环，负责采集、训练和记录日志。
        """
        i = 0
        train_times = 0

        # 初始化swanlab实验
        swanlab.init(project="ppo-demo")

        while i < self.total_train_steps:
            # 采集一批数据
            batch_state, batch_action, batch_log_prod, batch_rtgs, bz_episode_len = self.collect_data(env)

            i += sum(bz_episode_len)
            train_times += 1

            # 先用critic给出当前（采集过的动作）的价值估计
            bz_value, _ = self.eval(batch_state, batch_action)
            # 计算优势A(s,a)  当前actor的奖励 - 预测的奖励 == 优势
            adv = batch_rtgs - bz_value.detach()
            # 优势归一化
            adv = (adv - adv.mean()) / (adv.std() + 1e-10)

            # 进入PPO主循环
            for _ in range(self.inner_loop_times):
                bz_value, cur_log_prod = self.eval(batch_state, batch_action)
                # 重要性采样权重比率
                ratios = torch.exp(cur_log_prod - batch_log_prod)
                adv1 = adv * ratios
                adv2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * adv

                actor_loss = (-torch.min(adv1, adv2)).mean()  # PPO policy损失
                critic_loss = nn.MSELoss()(bz_value, batch_rtgs)  # value网络损失

                actor_loss.backward()
                self.optim_actor.step()
                self.optim_actor.zero_grad()

                critic_loss.backward()
                self.optim_critic.step()
                self.optim_critic.zero_grad()

            # 只记录本次batch的rtgs均值(优势)和损失
            mean_rtgs = float(batch_rtgs.mean().item())
            swanlab.log(
                {
                    "rtgs/mean": mean_rtgs,
                    "loss/actor": float(actor_loss.item()),
                    "loss/critic": float(critic_loss.item()),
                },
                step=i,
            )
            print(
                f"Update: {train_times}, Steps: {i}, Mean RTGS: {mean_rtgs:.4f}, "
                f"Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}"
            )

            # 定期保存模型
            if train_times % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './ppo_actor.pth')
                torch.save(self.critic.state_dict(), './ppo_critic.pth')

    def eval(self, bz_state, bz_action):
        """
        对一批状态、动作评估，返回value和当前策略下该动作的log_prob。
        """
        bz_value = self.critic(bz_state).squeeze()
        # 用现有策略（重新前向一次actor）构造分布计算log_prob
        action = self.actor(bz_state)
        dist = MultivariateNormal(action, covariance_matrix=self.conv_mat)
        log_prod = dist.log_prob(bz_action)
        return bz_value, log_prod

    def get_rtgs(self, bz_rewards):
        """
        计算累计reward-to-go，用于PPO的优势估计。输入为list的list。
        """
        bz_rtgs = []
        for episode_rewards in reversed(bz_rewards):
            discount_reward = 0
            for reward in reversed(episode_rewards):
                discount_reward = reward + self.gamma * discount_reward
                bz_rtgs.append(discount_reward)
        bz_rtgs = list(reversed(bz_rtgs))
        bz_rtgs = np.array(bz_rtgs, dtype=np.float32)
        bz_rtgs_tensor = torch.from_numpy(bz_rtgs)
        return bz_rtgs_tensor

# ================== Gym环境部分 =====================
'''
倒立摆的物理模型（Pendulum-v1）：
一个单摆，可以绕支点自由旋转
目标：将摆锤摆动到垂直向上​​的位置并保持平衡
挑战：摆锤受到重力作用，自然状态下会下垂

# 状态包含3个值：
state = [cos(theta), sin(theta), theta_dot]
# 其中：
# - theta: 摆的角度（相对于垂直向上位置）
# - theta_dot: 角速度
# - cos(theta), sin(theta): 用三角函数表示角度（避免角度不连续）

# 动作空间：连续值，范围 [-2.0, 2.0]
action_space = Box(low=-2.0, high=2.0, shape=(1,))

# 动作：施加在摆上的扭矩（torque）
# 正值：顺时针，负值：逆时针，绝对值表示大小
'''

def main():
    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = Agent(state_dim, action_dim)
    agent.train(env)

if __name__=='__main__':
    main()