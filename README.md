# Playing Tetris Using Reinforcement Learning 
**Application of Reinforcement Learning Algorithms to a Simplified Tetris Environment**

**1. Introduction**
In this project, I have implemented and compared three RL algorithms—Deep Q‑Network (DQN), REINFORCE (Monte Carlo Policy Gradient), and Proximal Policy Optimization (PPO), and PPO with Baseline (PPO‑BL)—on a simplified Tetris environment. The aim is to evaluate the improvement speed, stability, and final performance of each method. The objective is to train an agent that learns optimal strategies to maximize its score by interacting with a simulated Tetris environment. The project uses a custom Tetris environment built with Pygame Library. 

**2. Dataset(s) Source and Description**
- The Tetris state space is represented by a 10×20 grid. I am not using any Data to train the model as the agent interacts with the environment and learns from that. 

**3. Data Exploration and Important Features**
This project does not use a external dataset. Instead, it relies on a simulated Tetris environment created programmatically using Python and Pygame. The environment consists of:
* Grid Size: A 10x20 grid (10 columns wide, 20 rows tall), where each cell is either empty (0) or occupied (1).
* Tetriminos: Five simplified piece types (I, O, T, L, J), each represented as a 2D NumPy array defining their shape.
* Action Space: The agent can perform four actions—move left, move right, rotate, and drop—while pieces fall incrementally.
*  Game Mechanics: Lines are cleared when a row is fully occupied, increasing the score, and the game ends when a new piece cannot spawn without overlapping existing blocks.

  The environment generates experiences dynamically as the agent interacts with it, producing states, actions, rewards. 

**4. Data Exploration and Important Features**

Since there is no static dataset, "data exploration" pertains to analyzing the state representation used by the agent. The state is defined by three normalized features extracted from the current grid configuration:
* Height: The maximum height of occupied cells, normalized by dividing by 20 (the grid height). This indicates how full the grid is, influencing the risk of game over.
* Holes: The total number of empty cells below occupied cells in each column, normalized by 50 (an estimated maximum). Holes complicate future piece placements and reduce line-clearing opportunities.
* Bumpiness: The sum of absolute height differences between adjacent columns, normalized by 100 (an estimated maximum). High bumpiness reflects an uneven grid surface, making it harder to fit pieces efficiently.

These features were chosen because they capture critical aspects of Tetris gameplay. They provide a compact yet informative representation for the agent to evaluate its actions.

**5. Methods**
* Tetris Environment
  The TetrisEnv class simulates the game:
1. State: A 3D NumPy array [height, holes, bumpiness].
2. Actions: Four discrete actions (0: move left, 1: move right, 2: rotate, 3: drop).
3. Reward: +1 per line cleared, 0 for game over. (This is different for PPO with BL with +1 per line crossed and -100 for game over)
4. Termination: The episode ends when a new piece cannot spawn (game over).
   
**5.1 Deep Q-Network (DQN)**
* Overview: A value-based method that estimates Q-values (expected rewards) for state-action pairs.
* Neural Network: Two hidden layers (24 units, ReLU), outputting Q-values for 4 actions.
* Exploration: ε-greedy with decay (1.0 to 0.01).
* Replay Buffer: 2000 capacity, batch size 32.
* Hyperparameters: Learning rate = 0.1, discount factor (γ) = 0.95.

**5.2 REINFORCE (Monte Carlo Policy Gradient)**
* Overview: A policy-based method that optimizes the policy directly using episode returns.
* Neural Network: Policy network with two hidden layers (24 units, ReLU), softmax output.
* Training: Full-episode Monte Carlo returns, no baseline.
* Hyperparameters: Learning rate = 0.1, discount factor (γ) = 0.99.

**5.3 Proximal Policy Optimization (PPO)**
* Overview: A policy-based method with a clipped objective for stable updates, paired with a value function.
* Neural Networks: Policy and value networks, each with two hidden layers (24 units, ReLU).
* Hyperparameters: Clip ratio ε = 0.2, learning rate = 0.001, 10 epochs per update, batch = 10 episodes.

**5.4 PPO with Baseline (PPO-BL)**
* Overview: Extends PPO with Generalized Advantage Estimation (GAE) for improved advantage calculation.
* Neural Networks: Same as PPO.
* Hyperparameters: GAE (λ = 0.95), clip ratio ε = 0.2 or 0.4, learning rate = 0.001, 10 epochs per update, batch = 10 episodes.

**6. Experimentation**
- Trained each agent for 1000 - 3000 episodes.
- Track per-episode reward

**7. Observations & Analysis**
7.1 **Convergence Speed**
- DQN - Improvement began around 1950 episodes.
- REINFORCE - No significant progress by 5300 episodes, even up to 10,000.
- PPO -  Improvement started after ~2000 episodes.
- PPO‑BL - surprisingly improved in just 1000 episodes. 

7.2 **Stability**
- DQN - Low variance post-improvement due to ε-decay, indicating stability.
- REINFORCE - Stability unclear due to slow progress; more episodes or implementation review is needed.
- PPO- Low variance and quick stabilization from clipping.
- PPO‑BL- Early improvement with higher initial variance; running 2000+ episodes could confirm long-term stability.
  
7.3 **Computational Cost**
- DQN -  Moderate, with replay buffer overhead. 
- REINFORCE - High cost due to slow convergence.
- PPO -  Moderate speed, slower than DQN in time but more stable.
- PPO‑BL - higher compute per update , but sample efficient.

**8. Conclusion**
* PPO with Value Baseline (PPO-BL) showed the best learning performance among all the algorithms tested. Its ability to clip policy updates helped improve stability, although using a very high clipping range reduced the improvement rate.
* Both DQN and PPO (without baseline) performed well in terms of learning speed and stability. Their convergence rates were quite similar, especially after running for around 3000 episodes.
* REINFORCE, though easy to understand and implement, had the slowest learning progress. This might be due to its high variance or possibly due to issues in the code implementation, which I couldn’t resolved.

**9. References**
- https://github.com/nuno-faria/tetris-ai/blob/master/README.md
- Proximal Policy Optimization Algorithms - https://web.stanford.edu/class/cs234/
- https://www.cs.toronto.edu/~cmaddis/courses/sta4273_w21/studentwork/gae.pdf

**Acknowledgement**
* A significant portion of the code for this project was developed with the assistance of large language models (LLMs), which provided guidance on implementation details and debugging.
* The implementation idea for the DQN agent was inspired by the open-source project available at nuno-faria/tetris-ai, which served as a foundational reference.

