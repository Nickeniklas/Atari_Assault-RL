# Reinforcement learning: Atari Assault 

Atari Assault solved using PPO, DQN and RainbowDQN models.

### Environment
gymnsasium: https://gymnasium.farama.org/

Ale: https://ale.farama.org/

### Agent Models
Stable-baselines3: https://stable-baselines3.readthedocs.io/en/master/index.html

# Setup
1. create virutal environment

2. install requirements.txt

3. run main.py

Choose model by changing in `__main__`: 
```
agent.set_model(name="PPO", env=env) # "DQN", "PPO" or "RDQN"
```

**Output:** Visualizations of training and saved trained model and tensorboard file.  

## Visualize results in browser (tensorboard)
```
tensorboard --logdir ./logs/assault_tensorboard/
```