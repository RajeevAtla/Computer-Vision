Code Documentation

Purpose

The code implements a Deep Q-Learning (DQN) agent to play the Atari game Ms. Pacman using OpenAI Gym’s Atari environments. It introduces custom reward shaping, frame skipping for faster training, and metrics tracking to evaluate the training process.

Key Components

1. Hyperparameters

Global variables to configure the environment, learning rate, gamma (discount factor), replay buffer size, batch size, epsilon-greedy strategy, target network update frequency, number of episodes, and maximum steps per episode.

2. Neural Network (DQN)

Defines the Deep Q-Network (DQN):
	•	Input: State shape (observation space dimensions).
	•	Output: Q-values for each action in the action space.
	•	Architecture:
	•	Convolutional layers for feature extraction from pixel-based inputs.
	•	Fully connected layers for Q-value computation.

3. Replay Buffer

Implements a memory buffer using deque to store experience tuples (state, action, reward, next_state, done). Supports adding new experiences and sampling random batches for training.

4. DQN Agent

Defines the agent using:
	•	Policy Network: Learns the Q-values for actions.
	•	Target Network: Stabilizes training by using a delayed update mechanism.
	•	Epsilon-Greedy Strategy: Selects actions based on an exploration-exploitation trade-off.
	•	Replay Buffer: Stores past experiences to enable experience replay.
	•	Training Function:
	•	Samples a batch from the replay buffer.
	•	Computes the loss between predicted and target Q-values.
	•	Backpropagates loss to update network weights.
	•	Target Network Update: Periodically syncs weights of the policy network to the target network.

5. Custom Reward Function

Introduces a custom reward system:
	•	Bonus rewards for positive events (e.g., eating dots or clearing levels).
	•	Penalties for negative events (e.g., colliding with ghosts or repetitive actions).

6. Frame Skip Wrapper

Reduces computational overhead by skipping frames:
	•	Executes the same action for skip frames.
	•	Aggregates rewards over skipped frames.

7. Training Loop

Implements the core training process:
	•	Resets the environment and initializes metrics.
	•	Runs episodes, allowing the agent to interact with the environment.
	•	Applies custom rewards and stores experiences in the replay buffer.
	•	Trains the policy network and periodically updates the target network.
	•	Tracks metrics like rewards, losses, epsilon values, episode lengths, and collisions.

8. Metrics Plotting

Generates visualizations to analyze training:
	•	Total rewards per episode.
	•	Loss per episode.
	•	Epsilon decay across episodes.
	•	Episode lengths (steps) per episode.

9. Functions

compute_custom_rewards

Computes rewards by:
	•	Adding bonuses for positive events (e.g., clearing a level).
	•	Applying penalties for negative events (e.g., ghost collisions, repetitive actions).

train_agent_with_custom_rewards

Main function to train the DQN agent:
	•	Prepares the environment and agent.
	•	Executes training episodes.
	•	Applies the custom reward function.
	•	Collects metrics for analysis.



plot_metrics

Visualizes training metrics using Matplotlib:
	•	Displays reward trends, loss evolution, epsilon decay, and episode lengths.

How to Run
	1.	Ensure the required libraries (gymnasium, torch, ale-py, shimmy) are installed.
	2.	Execute the script to train the agent.
	3.	Use plot_metrics to analyze training performance.

Customizations
	•	Modify compute_custom_rewards to change the reward shaping logic.
	•	Adjust hyperparameters (e.g., LEARNING_RATE, GAMMA, EPSILON_DECAY) to experiment with learning behaviors.
	•	Use additional wrappers for preprocessing (e.g., grayscale conversion, resizing) to optimize input processing.

