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
            state_tensor = (
                torch.FloatTensor(np.array(state)).unsqueeze(0).to(device)
            )

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
    print(
        f"\nAverage Reward over {num_episodes} episodes: "
        f"{average_reward:.2f}"
    )


if __name__ == "__main__":
    test()
