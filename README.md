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
   
5.1 **Deep Q‑Network (DQN)**
- Neural network: two hidden layers (24 units, ReLU), output Q‑values for 4 actions.
- ε‑greedy exploration, replay buffer (2k capacity), batch size 32.
- Learning Rate: 0.1, discount factor (γ) = 0.95.

5.2 **REINFORCE (Monte Carlo Policy Gradient)**
- Policy network: with softmax output and same hidden layers 
- Full-episode Monte Carlo returns, no baseline.
- Learning Rate: 0.1, discount factor (γ) = 0.99.

5.3 **Proximal Policy Optimization (PPO)**
- Policy & value networks: same hidden layers.
- - Clip ratio ε=0.2, Learning rate- 0.001
- - 10 epochs per update, batch = 10 episodes.

5.4 **PPO with Baseline (PPO‑BL)**
- Policy & value networks: same hidden layers.
- GAE(λ=0.95) for advantage estimation.
- Clip ratio ε=0.2,0.4, learning rate 0.001
- 10 epochs per update, batch = 10 episodes.


**6. Experimentation**
- Trained each agent for 1000 - 3000 episodes.
- Track per-episode reward

**7. Observations & Analysis**
7.1 **Convergence Speed**
- DQN started improving in ~1950 episodes.
- REINFORCE - I tried with near 10,000 episodes and till 5300 there was no improvement.
- PPO started improving after 2000 episodes
- PPO‑BL surprisingly improved in just 1000 episodes. 

7.2 **Stability**
- DQN showed slow variace after improvement (ε decay), hence stabilized.
- REINFORCE - No idea.
- PPO- Showed low variace hence stabilized due to clipping. 
- PPO‑BL Showed improvement in early 1000 episodes but the varaince is low hence not much stabilized initially, we can run the algorithm for more than 2000 episodes to see exactly. 

7.3 **Computational Cost**
- DQN: moderate compute (experience replay overhead).
- REINFORCE: Very Slow convergence.
- PPO: Moderate Convergence but slower than DQN (wrt time)
- PPO‑BL: higher compute per update , but sample efficient.

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

