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
