import gymnasium as gym
import ale_py
import optuna
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import pandas as pd
import os

#gym.register_envs(ale_py) # if auto registration fails

def main():
    env = gym.make('ALE/Assault-v5', render_mode="human")
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

    # debugging
    #print("Action space:", env.action_space)
    #print("Observation space:", env.observation_space)

    # test run with random actions
    episodes = 5
    for episode in range(1, episodes+1):
        obs, info = env.reset()
        done = False
        score = 0 

        while not done:
            env.render()
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward
        print(f'Episode:{episode}, Score:{score}')

    env.close()

# script run
if __name__ == "__main__":
    print("Starting main.py")
    main()