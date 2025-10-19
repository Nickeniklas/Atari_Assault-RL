from stable_baselines3 import DQN, PPO
from sb3_contrib import QRDQN
import matplotlib.pyplot as plt
import os

class agent:
    def __init__(self):
        self.name = None
    def set_model(self, name, env, seed=42):
        """ Set up the RL agent model based on the specified name. (Optional)"""
        self.name = name
        self.model = None

        if not os.path.exists('./logs'):
            os.makedirs('./logs')

        if name == "DQN":
            # Dqn Policy for image input, tensorboard logging for visualization, memory handling
            self.model = DQN("CnnPolicy", env, buffer_size=100_000, learning_starts=10_000, verbose=1, tensorboard_log="./logs/assault_tensorboard/")

        elif name == "PPO":    
            self.model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./logs/assault_tensorboard/")

        elif name == "QRDQN":    
            self.model = QRDQN("CnnPolicy", env, seed=seed, buffer_size=100_000, verbose=1, tensorboard_log="./logs/assault_tensorboard/")

        return self.model
