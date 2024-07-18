# ML-MPJ4 Project
Description of ML-MPJ4 project.
Deep Q-Learning and Q-Learning in Wumpus World

## Overview
This project explores the implementation of Deep Q-Learning and Q-Learning algorithms to solve the Wumpus World problem. Wumpus World is a classic problem in artificial intelligence and reinforcement learning, where an agent must navigate a grid-based environment to find gold while avoiding hazards such as pits and the Wumpus.

## Questions

### Question One
The goal of this question is to implement Deep Q-Learning and Q-Learning to solve the Wumpus World problem. The agent must learn to:
- Navigate the grid effectively
- Avoid hazards such as pits and the Wumpus
- Collect gold
- Utilize a 4x4 grid with specified cell contents (empty, pit, Wumpus, or gold)
- Move in four directions and shoot an arrow in four directions
- Implement a reward structure: +100 for gold, -1000 for falling into a pit or being eaten by Wumpus, +50 for killing Wumpus, and -1 for each movement

## Setup Instructions

### Environment
1. WumpusWorld Class: This class simulates the Wumpus World environment.
    - reset(): Resets the environment to the initial state.
    - step(action): Executes an action and returns the next state, reward, and whether the game is done.

### Agent
1. DQNAgent Class: This class defines the agent using Deep Q-Learning.
    - __init__(self, env): Initializes the agent with the environment.
    - remember(self, state, action, reward, next_state, done): Stores experiences in replay memory.
    - select_action(self, state): Selects an action based on the current policy.
    - optimize_model(self): Updates the model using experiences from replay memory.
    - learn(self, num_episodes): Trains the agent over a specified number of episodes.

## Implementation Details

### Libraries and Tools
- numpy
- random
- collections (namedtuple, deque)
- torch (nn, optim)
- matplotlib.pyplot

### Key Components
- ReplayMemory: Stores past experiences to be used for training the agent.
- DQN Model: Defines the neural network architecture for Deep Q-Learning.
- Training Loop: Runs the training process, updating the agent's policy and target networks.

### Parameters
- MEMORY_CAPACITY: Size of the replay memory.
- BATCH_SIZE: Size of training batches.
- GAMMA: Discount factor for future rewards.
- EPS_START: Initial exploration rate.
- EPS_END: Minimum exploration rate.
- EPS_DECAY: Rate of decay for exploration.
- TAU: Rate for updating the target network.
- LR: Learning rate for the optimizer.

## Usage Instructions

1. Initialize the Environment and Agent:
   
   env = WumpusWorld()
   agent = DQNAgent(env)
   

2. Train the Agent:
   
   num_episodes = 1000
   agent.learn(num_episodes)
   

3. Evaluate the Agent:
   After training, evaluate the agent's performance by running it in the environment and observing its behavior.

## Results

- Training Performance: Track the agent's cumulative rewards over episodes to monitor learning progress.
- Evaluation: Assess the agent's ability to navigate the Wumpus World, avoid hazards, and collect gold.

## Report
Provide a detailed report including:
- Introduction to Wumpus World and reinforcement learning concepts.
- Description of the Deep Q-Learning and Q-Learning algorithms.
- Implementation details and challenges faced.
- Results and analysis of the agent's performance.
- Conclusions and future work suggestions.
