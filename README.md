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
```

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

```bash
python train.py
```
