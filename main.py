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

def setup_env(render_mode=None):
    env = gym.make('ALE/Assault-v5', render_mode=render_mode)
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
        self.model = None

        if not os.path.exists('./logs'):
            os.makedirs('./logs')

        if name == "DQN":
            # Dqn Policy for image input, tensorboard logging for visualization
            self.model = DQN("CnnPolicy", env, verbose=1, tensorboard_log=f"./logs/{name}_assault_tensorboard/")

        elif name == "PPO":    
            # Cnn Policy for image input, tensorboard logging for visualization
            self.model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=f"./logs/{name}_assault_tensorboard/")

        return self.model

def visualize_trained_agent(env, model, episodes=5):
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

def print_results(env):
    episode_rewards = env.get_episode_rewards()  # only available after Monitor tracks episodes
    print("Episode rewards:", episode_rewards)
    print("Mean reward:", sum(episode_rewards)/len(episode_rewards))
    print("Number of episodes:", len(episode_rewards))

def plot_results(env):
    episode_rewards = env.get_episode_rewards()  # only available after Monitor tracks episodes
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward Curve")
    plt.show()

# script run
if __name__ == "__main__":
    print("Starting main.py")
    
    # setup env
    print("Setting up environment...")
    env = setup_env(render_mode=None) # "human" or "rgb_array" or None

    # setup model
    print("Setting up model...")
    agent = agent()
    model = agent.set_model(name="PPO", env=env) # "DQN" or "PPO"

    # train agent
    print("Training model...")
    agent.model.learn(total_timesteps=10000)

    # save agent
    print("Saving model...")
    if not os.path.exists('./models'):
        os.makedirs('./models')
    agent.model.save(f"./models/{agent.name}_assault")

    # print and plot results
    print("Printing results...")
    print_results(env)
    print("Plotting results...")
    plot_results(env)

    # visualize trained model
    #print("Testing model...")
    #env = setup_env(render_mode="Human")
    #visualize_trained_agent(env, agent.model, episodes=2)


    