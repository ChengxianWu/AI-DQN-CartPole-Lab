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
                states, actions, batch_rewards, next_states, dones = (
                    replay_buffer.sample(batch_size)
                )

                states_tensor = torch.FloatTensor(states).to(device)
                actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards_tensor = (
                    torch.FloatTensor(batch_rewards).unsqueeze(1).to(device)
                )
                next_states_tensor = torch.FloatTensor(next_states).to(device)
                dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(device)

                current_q = policy_net(states_tensor).gather(1, actions_tensor)

                with torch.no_grad():
                    max_next_q = target_net(next_states_tensor).max(
                        1, keepdim=True
                    )[0]
                    target_q = rewards_tensor + gamma * max_next_q * (
                        1 - dones_tensor
                    )

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
