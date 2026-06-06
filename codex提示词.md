下面这份可以直接复制给 **Codex**，让它在你已经克隆好的空 GitHub 仓库中完成项目构建。它是按照“**可直接上手开发**”的方式写的，包括项目结构、代码文件、运行方式、提交方式和检查标准。

------

# 给 Codex 的完整项目构建指令

你现在位于一个已经从 GitHub 克隆到本地的空仓库中。请在当前仓库中构建一个人工智能第四次作业项目。

本次作业主题是 **强化学习实验**，要求实现 DQN 或其他强化学习模型，并提交 code、README、实验报告 PPT 和 5 分钟 PPT 讲述视频。本阶段只需要完成 **code + README** 部分。作业最终需要包含算法流程介绍、实验结果分析，README 要说明如何运行程序。

本项目选择实现：

```text
基于 DQN 的 CartPole-v1 强化学习实验
```

请严格按照下面要求构建项目。

------

## 一、项目目标

请实现一个完整、可运行、适合课程作业展示的 DQN 项目。

项目需要完成以下功能：

```text
1. 使用 Deep Q-Network, DQN 训练智能体完成 CartPole-v1 任务
2. 使用 PyTorch 构建 Q 网络
3. 实现经验回放 Replay Buffer
4. 实现 Target Network
5. 实现 epsilon-greedy 探索策略
6. 训练完成后保存模型
7. 训练完成后保存 reward 曲线
8. 保存训练日志
9. 提供测试脚本加载模型并输出平均奖励
10. 提供 README，说明项目结构、环境配置、训练方法和测试方法
```

------

## 二、项目目录结构

请在当前仓库根目录下创建如下结构：

```text
AI-HW4-DQN-CartPole/
├── README.md
├── requirements.txt
├── .gitignore
├── dqn.py
├── replay_buffer.py
├── utils.py
├── train.py
├── test.py
├── models/
├── results/
└── logs/
```

注意：你当前已经在仓库根目录，不需要再额外创建最外层 `AI-HW4-DQN-CartPole` 文件夹。请直接在当前目录下创建这些文件和文件夹。

------

## 三、依赖文件 requirements.txt

请创建 `requirements.txt`，内容如下：

```txt
gymnasium[classic-control]==0.29.1
torch
numpy
matplotlib
tqdm
```

------

## 四、.gitignore 文件

请创建 `.gitignore`，内容如下：

```gitignore
# Python cache
__pycache__/
*.pyc
*.pyo
*.pyd

# Virtual environments
.venv/
venv/
env/

# IDE
.vscode/
.idea/

# OS files
.DS_Store
Thumbs.db

# Temporary files
*.tmp

# Jupyter
.ipynb_checkpoints/
```

注意：不要忽略 `models/`、`results/` 和 `logs/`，因为本课程作业需要保留模型、结果图和训练日志。

------

## 五、实现 dqn.py

请在 `dqn.py` 中实现 DQN 的 Q 网络。

要求：

```text
1. 使用 PyTorch。
2. 定义类 QNetwork，继承 nn.Module。
3. 初始化参数包括 state_dim、action_dim、hidden_dim。
4. hidden_dim 默认值为 128。
5. 网络结构为：
   Linear(state_dim, hidden_dim)
   ReLU
   Linear(hidden_dim, hidden_dim)
   ReLU
   Linear(hidden_dim, action_dim)
6. forward 输入状态 state，输出每个动作对应的 Q 值。
7. 代码应简洁、规范，并包含必要注释。
```

请实现为：

```python
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    A simple fully connected Q-network for DQN.
    Input: environment state
    Output: Q-values for all possible actions
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(QNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)
```

------

## 六、实现 replay_buffer.py

请在 `replay_buffer.py` 中实现经验回放池。

要求：

```text
1. 使用 collections.deque 存储经验。
2. 定义类 ReplayBuffer。
3. 初始化参数 capacity，表示经验池最大容量。
4. 实现 push(state, action, reward, next_state, done)。
5. 实现 sample(batch_size)。
6. sample 随机采样一个 batch。
7. sample 返回：
   states, actions, rewards, next_states, dones
8. 返回值均为 numpy 数组。
9. 实现 __len__ 方法，返回当前经验数量。
```

请实现为：

```python
import random
from collections import deque
from typing import Tuple

import numpy as np


class ReplayBuffer:
    """
    Experience replay buffer for DQN.
    It stores transitions and provides random mini-batches for training.
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)
```

------

## 七、实现 utils.py

请在 `utils.py` 中实现工具函数。

要求实现两个函数：

```text
1. save_reward_curve(rewards, save_path)
   - 使用 matplotlib 绘制训练奖励曲线
   - 横轴为 Episode
   - 纵轴为 Reward
   - 标题为 Training Reward Curve
   - 保存到 save_path

2. write_log(log_path, message)
   - 将训练日志追加写入 log_path
   - 每条日志单独一行
   - 如果目录不存在，自动创建
```

请实现为：

```python
import os
from typing import List

import matplotlib.pyplot as plt


def save_reward_curve(rewards: List[float], save_path: str) -> None:
    """
    Save the training reward curve as an image.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(rewards) + 1), rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward Curve")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def write_log(log_path: str, message: str) -> None:
    """
    Append a message to the log file.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")
```

------

## 八、实现 train.py

请在 `train.py` 中实现完整 DQN 训练程序。

### 训练环境

使用：

```python
gymnasium.make("CartPole-v1")
```

### 训练参数

请使用以下默认参数：

```python
seed = 42
num_episodes = 400
batch_size = 64
gamma = 0.99
learning_rate = 1e-3
buffer_capacity = 10000
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
target_update_freq = 10
hidden_dim = 128
```

### 训练逻辑要求

请完整实现以下流程：

```text
1. 创建 models、results、logs 目录。
2. 设置随机种子。
3. 创建 CartPole-v1 环境。
4. 获取 state_dim 和 action_dim。
5. 初始化 policy_net 和 target_net。
6. target_net 初始加载 policy_net 参数。
7. 使用 Adam 优化器。
8. 使用 MSELoss。
9. 初始化 ReplayBuffer。
10. 每个 episode reset 环境。
11. 每一步根据 epsilon-greedy 选择动作。
12. 与环境交互，得到 next_state、reward、terminated、truncated、info。
13. done = terminated or truncated。
14. 将 transition 存入 ReplayBuffer。
15. 当 ReplayBuffer 数量达到 batch_size 后开始训练。
16. 从 ReplayBuffer 采样 batch。
17. 计算当前 Q 值：
    current_q = policy_net(states).gather(1, actions)
18. 计算目标 Q 值：
    target_q = rewards + gamma * max(target_net(next_states)) * (1 - dones)
19. 使用 MSELoss 计算 loss。
20. 反向传播并更新 policy_net。
21. 每 target_update_freq 个 episode 同步 target_net。
22. 每个 episode 结束后更新 epsilon，不能低于 epsilon_end。
23. 每个 episode 打印 reward、epsilon、loss。
24. 每个 episode 将日志写入 logs/train_log.txt。
25. 训练结束后保存模型到 models/dqn_cartpole.pth。
26. 训练结束后保存奖励曲线到 results/reward_curve.png。
27. 训练结束后输出最终提示。
```

### train.py 推荐完整代码

请按照下面代码实现：

```python
import os
import random
from typing import List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dqn import QNetwork
from replay_buffer import ReplayBuffer
from utils import save_reward_curve, write_log


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_action(
    state: np.ndarray,
    policy_net: QNetwork,
    epsilon: float,
    action_dim: int,
    device: torch.device,
) -> int:
    """
    Select action using epsilon-greedy strategy.
    """
    if random.random() < epsilon:
        return random.randrange(action_dim)

    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = policy_net(state_tensor)

    return int(torch.argmax(q_values, dim=1).item())


def train() -> None:
    # Paths
    model_path = "models/dqn_cartpole.pth"
    reward_curve_path = "results/reward_curve.png"
    log_path = "logs/train_log.txt"

    # Create output directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Clear old log file
    if os.path.exists(log_path):
        os.remove(log_path)

    # Hyperparameters
    seed = 42
    num_episodes = 400
    batch_size = 64
    gamma = 0.99
    learning_rate = 1e-3
    buffer_capacity = 10000
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    target_update_freq = 10
    hidden_dim = 128

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_log(log_path, f"Using device: {device}")

    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    env.action_space.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    target_net = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    replay_buffer = ReplayBuffer(buffer_capacity)

    rewards: List[float] = []
    epsilon = epsilon_start

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0.0
        episode_losses = []

        done = False

        while not done:
            action = select_action(state, policy_net, epsilon, action_dim, device)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if len(replay_buffer) >= batch_size:
                states, actions, batch_rewards, next_states, dones = replay_buffer.sample(batch_size)

                states_tensor = torch.FloatTensor(states).to(device)
                actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards_tensor = torch.FloatTensor(batch_rewards).unsqueeze(1).to(device)
                next_states_tensor = torch.FloatTensor(next_states).to(device)
                dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(device)

                current_q = policy_net(states_tensor).gather(1, actions_tensor)

                with torch.no_grad():
                    max_next_q = target_net(next_states_tensor).max(1, keepdim=True)[0]
                    target_q = rewards_tensor + gamma * max_next_q * (1 - dones_tensor)

                loss = criterion(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                episode_losses.append(loss.item())

        rewards.append(episode_reward)

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0

        log_message = (
            f"Episode {episode:03d}/{num_episodes} | "
            f"Reward: {episode_reward:.1f} | "
            f"Epsilon: {epsilon:.4f} | "
            f"Loss: {avg_loss:.6f}"
        )

        print(log_message)
        write_log(log_path, log_message)

    env.close()

    torch.save(policy_net.state_dict(), model_path)
    save_reward_curve(rewards, reward_curve_path)

    print("\nTraining finished.")
    print(f"Model saved to: {model_path}")
    print(f"Reward curve saved to: {reward_curve_path}")
    print(f"Training log saved to: {log_path}")


if __name__ == "__main__":
    train()
```

------

## 九、实现 test.py

请在 `test.py` 中实现模型测试程序。

要求：

```text
1. 使用 CartPole-v1 环境。
2. 加载 models/dqn_cartpole.pth。
3. 如果模型不存在，提示用户先运行 python train.py。
4. 构建 QNetwork。
5. 测试 10 个 episodes。
6. 测试阶段不使用 epsilon，直接选择 Q 值最大的动作。
7. 输出每个 episode 的 reward。
8. 输出 average reward。
```

请实现为：

```python
import os

import gymnasium as gym
import numpy as np
import torch

from dqn import QNetwork


def test() -> None:
    model_path = "models/dqn_cartpole.pth"

    if not os.path.exists(model_path):
        print("Model not found. Please run python train.py first.")
        return

    num_episodes = 10
    hidden_dim = 128
    seed = 42

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make("CartPole-v1")
    env.reset(seed=seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()

    rewards = []

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0).to(device)

            with torch.no_grad():
                q_values = policy_net(state_tensor)

            action = int(torch.argmax(q_values, dim=1).item())

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            episode_reward += reward

        rewards.append(episode_reward)
        print(f"Test Episode {episode}: Reward = {episode_reward:.1f}")

    env.close()

    average_reward = np.mean(rewards)
    print(f"\nAverage Reward over {num_episodes} episodes: {average_reward:.2f}")


if __name__ == "__main__":
    test()
```

------

## 十、实现 README.md

请在 `README.md` 中写入以下内容。

~~~markdown
# AI HW4: DQN Reinforcement Learning Experiment

## 1. Project Introduction

This project implements a Deep Q-Network, DQN, to solve the CartPole-v1 reinforcement learning task.

The agent interacts with the environment, receives rewards, and learns an action-value function to maximize the cumulative reward.

This project is built for the fourth homework of the Artificial Intelligence course.

## 2. Task

The task is based on the CartPole-v1 environment.

In this environment, the agent controls a cart by moving it left or right. The goal is to keep the pole balanced for as long as possible. The reward increases when the pole remains balanced.

## 3. Algorithm

The implementation is based on Deep Q-Network, DQN.

The main components include:

- Q-Network
- Experience Replay
- Target Network
- Epsilon-greedy exploration strategy

The Q-network takes the environment state as input and outputs the Q-values of all possible actions. The agent selects actions according to the epsilon-greedy strategy during training.

## 4. Project Structure

```text
AI-HW4-DQN-CartPole/
├── README.md
├── requirements.txt
├── .gitignore
├── dqn.py
├── replay_buffer.py
├── utils.py
├── train.py
├── test.py
├── models/
├── results/
└── logs/
~~~

## 5. Environment

Python 3.10 is recommended.

Main dependencies:

- gymnasium
- torch
- numpy
- matplotlib
- tqdm

## 6. Install Dependencies

```bash
pip install -r requirements.txt
```

If the download speed is slow, use the Tsinghua mirror:

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 7. Train the Model

Run:

```bash
python train.py
```

After training, the model will be saved to:

```text
models/dqn_cartpole.pth
```

The reward curve will be saved to:

```text
results/reward_curve.png
```

The training log will be saved to:

```text
logs/train_log.txt
```

## 8. Test the Model

Run:

```bash
python test.py
```

The program will load the trained model and print the reward of each test episode and the average reward.

## 9. Output Files

After running the training script, the following files will be generated:

```text
models/dqn_cartpole.pth
results/reward_curve.png
logs/train_log.txt
```

## 10. Experimental Result

During training, the reward generally increases as the number of episodes grows. This shows that the agent gradually learns an effective policy for the CartPole-v1 control task.

The reward curve can be used in the final experiment report PPT for result analysis.

## 11. Notes

If the model file does not exist when running the test script, please train the model first:

~~~bash
python train.py
---

## 十一、运行检查步骤

请完成代码后，在当前仓库根目录依次运行：

```bash
python -m py_compile dqn.py
python -m py_compile replay_buffer.py
python -m py_compile utils.py
python -m py_compile train.py
python -m py_compile test.py
~~~

如果没有报错，再运行：

```bash
python train.py
```

训练结束后检查是否生成：

```text
models/dqn_cartpole.pth
results/reward_curve.png
logs/train_log.txt
```

然后运行：

```bash
python test.py
```

测试输出应包含：

```text
Test Episode 1: Reward = ...
Test Episode 2: Reward = ...
...
Average Reward over 10 episodes: ...
```

------

## 十二、Git 提交要求

完成项目结构和代码后，先进行第一次提交：

```bash
git status
git add .
git commit -m "Implement DQN CartPole experiment"
git push
```

训练完成并生成模型、结果图、日志后，再进行第二次提交：

```bash
git status
git add models results logs
git commit -m "Add trained model reward curve and training log"
git push
```

如果第一次提交时还没有配置 Git 用户，请使用：

```bash
git config --global user.name "ChengxianWu"
git config --global user.email "wuchengxian528@gmail.com"
```

然后重新提交。

------

## 十三、最终验收标准

请保证项目最终满足以下标准：

```text
1. 当前仓库不是空的。
2. README.md 存在，并能说明如何运行程序。
3. requirements.txt 存在。
4. dqn.py 存在，并实现 QNetwork。
5. replay_buffer.py 存在，并实现 ReplayBuffer。
6. utils.py 存在，并实现保存曲线和日志的函数。
7. train.py 可以通过 python train.py 运行。
8. test.py 可以通过 python test.py 运行。
9. 训练后生成 models/dqn_cartpole.pth。
10. 训练后生成 results/reward_curve.png。
11. 训练后生成 logs/train_log.txt。
12. GitHub 仓库中能看到完整代码、README 和实验结果。
```

------

