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

def setup_env():
    env = gym.make('ALE/Assault-v5', render_mode="human")
    env = Monitor(env)
    
    # debugging
    #print("Action space:", env.action_space)
    #print("Observation space:", env.observation_space)

    return env

class agent:
    def __init__(self):
        self.name = None
    def set_model(self, name, env):
        self.name = name

        if not os.path.exists('./logs'):
            os.makedirs('./logs')

        if name == "DQN":
            # Dqn Policy for image input, tensorboard logging for visualization
            model = DQN("CnnPolicy", env, verbose=1, tensorboard_log="./logs/dqn_assault_tensorboard/")

        if name == "PPO":    
            # Cnn Policy for image input, tensorboard logging for visualization
            model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./logs/ppo_assault_tensorboard/")

        return model

def test_trained_agent(env, model, episodes = 5):
    # testing run loop
    for episode in range(1, episodes+1):
        obs, info = env.reset()
        done = False
        score = 0 

        while not done:
            env.render()
            action = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward
        print(f'Episode:{episode}, Score:{score}')

    env.close()

# script run
if __name__ == "__main__":
    print("Starting main.py")
    
    # setup env
    print("Setting up environment...")
    env = setup_env()

    # setup model
    print("Setting up model...")
    agent = agent()
    model = agent.set_model("PPO", env)

    # train agent
    print("Training model...")
    model.learn(total_timesteps=10000)

    # save agent
    print("Saving model...")
    model.save(f"{model.name}_assault")

    # test trained model
    print("Testing model...")
    test_trained_agent(env, model, episodes=3)


    