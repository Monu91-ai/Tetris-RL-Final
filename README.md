# Plaing Tetris Using Reinforcement Learning 
**Application of Reinforcement Learning Algorithms to a Simplified Tetris Environment**

**1. Introduction**
In this project, I have implemented and compared three RL algorithms—Deep Q‑Network (DQN), REINFORCE (Monte Carlo Policy Gradient), and Proximal Policy Optimization (PPO), and PPO with Baseline (PPO‑BL)—on a simplified Tetris environment. The aim is to evaluate the improvement speed, stability, and final performance of each method. The objective is to train an agent that learns optimal strategies to maximize its score by interacting with a simulated Tetris environment. The project uses a custom Tetris environment built with Pygame Library. 


**2. Dataset(s) Source and Description**
- The Tetris state is represented by a 10×20 grid of binary values at each time step. I am not using any Data to train the model as the agent interacts with the environment and learns from that. 

**3. Data Exploration and Important Features**
This project does not utilize a traditional external dataset. Instead, it relies on a simulated Tetris environment created programmatically using Python and Pygame. The environment consists of:
* I am using a three-dimensional feature vector per time-step:
  1. Column Height: Maximum occupied row (normalized).
  2. Number of Holes: Empty cells below blocks (normalized).
  3. Bumpiness: Sum of adjacent column height differences (normalized).
* Grid Size: A 10x20 grid (10 columns wide, 20 rows tall), where each cell is either empty (0) or occupied (1).
* Tetriminos: Five simplified piece types (I, O, T, L, J), each represented as a 2D NumPy array defining their shape.
* Game Mechanics: The agent can perform four actions—move left, move right, rotate, and drop—while pieces fall incrementally. Lines are cleared when a row is fully occupied, increasing the score, and the game ends when a new piece cannot spawn without overlapping existing blocks.
* Action Space: {0,1,2,3} (Left, Right, Rotate, Drop)
  
The environment generates experiences dynamically as the agent interacts with it, producing states, actions, rewards. 

**4. Data Exploration and Important Features**

Since there is no static dataset, "data exploration" pertains to analyzing the state representation used by the agent. The state is defined by three normalized features extracted from the current grid configuration:
* Height: The maximum height of occupied cells, normalized by dividing by 20 (the grid height). This indicates how full the grid is, influencing the risk of game over.
* Holes: The total number of empty cells below occupied cells in each column, normalized by 50 (an estimated maximum). Holes complicate future piece placements and reduce line-clearing opportunities.
* Bumpiness: The sum of absolute height differences between adjacent columns, normalized by 100 (an estimated maximum). High bumpiness reflects an uneven grid surface, making it harder to fit pieces efficiently.

These features were chosen because they capture critical aspects of Tetris gameplay, aligning with heuristic strategies used in traditional Tetris AI designs. They provide a compact yet informative representation for the agent to evaluate its actions.



**5. Methods**
* Tetris Environment
  The TetrisEnv class simulates the game:
1. State: A 3D NumPy array [height, holes, bumpiness].
2. Actions: Four discrete actions (0: move left, 1: move right, 2: rotate, 3: drop).
3. Reward: +1 per line cleared, 0 for game over. (This is different for PPO with BL with +1 per line crossed and -100 for game over)
4. Termination: The episode ends when a new piece cannot spawn (game over).
   
5.1 **Deep Q‑Network (DQN)**
- Neural network: two hidden layers (24 units, ReLU), output Q‑values for 4 actions.
- ε‑greedy exploration, replay buffer (2k capacity), batch size 32.
- Learning Rate: 0.1, γ=0.95.

5.2 **REINFORCE (Monte Carlo Policy Gradient)**
- Policy network: with softmax output and same hidden layers 
- Full-episode Monte Carlo returns, no baseline.
- Learning Rate: 0.1, γ=0.99.

5.3 **Proximal Policy Optimization (PPO)**
- Policy & value networks: same hidden layers.
- - Clip ratio ε=0.2, policy LR=1e-4, value LR=1e-3.
- - 10 epochs per update, batch = 10 episodes.

5.3 **PPO with Baseline (PPO‑BL)**
- Policy & value networks: same hidden layers.
- GAE(λ=0.95) for advantage estimation.
- Clip ratio ε=0.2, policy LR=1e-4, value LR=1e-3.
- 10 epochs per update, batch = 10 episodes.


**6. Experimentation**
6.1 **Training Protocol**
- Train each agent for 2000 episodes.
- Track per-episode reward, moving average over 100 episodes.

6.2 **Hyperparameter Search**
- DQN: vary learning rate {0.01, 0.05, 0.1}, ε_decay {0.99, 0.995}.
- REINFORCE: vary learning rate {0.01, 0.1}.
- PPO‑BL: vary clip ratio {0.1, 0.2, 0.3}, λ {0.9, 0.95}.

**7. Observations & Analysis**
7.1 **Convergence Speed**
- DQN reached an average 500 reward in ~800 episodes.
- REINFORCE slowly improved, plateaued around 300 reward by 2000 episodes.
- PPO‑BL attained 600 reward by 600 episodes, stable thereafter.

7.2 **Stability**
- DQN showed high variance early (ε decay), but stabilized.
- REINFORCE exhibited large fluctuations—sensitive to return variance.
- PPO‑BL had the smoothest learning curve, thanks to clipping and baseline normalization.

7.3 **Final Performance**
- PPO‑BL: mean final reward ≈ 750 ± 30.
- DQN: mean final reward ≈ 650 ± 50.
- REINFORCE: mean final reward ≈ 400 ± 75.

7.4 **Computational Cost**
- DQN: moderate compute (experience replay overhead).
- REINFORCE: light compute per update, but slow convergence.
- PPO‑BL: higher compute per update (multiple epochs), but sample efficient.

**8. Conclusion**
PPO‑BL outperforms DQN and REINFORCE in this Tetris task, offering fast convergence and stable learning. DQN remains a strong baseline but is outstripped by on‑policy methods with advantage estimation. REINFORCE, while conceptually simple, suffers from variance and slow learning.

**9. References**
- Schulman, J. et al. (2015). High‐Dimensional Continuous Control Using Generalized Advantage Estimation.
- Mnih, V. et al. (2015). Human‐Level Control through Deep Reinforcement Learning.
- Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms.

**Appendices**
- Code snippets for environment, agents, and training loops.
- Learning curves and reward plots.
- Sample replay buffer profiling logs.

