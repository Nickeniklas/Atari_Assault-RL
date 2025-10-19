import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
from gymnasium.wrappers import FrameStackObservation
import ale_py
import optuna
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from sb3_contrib import RainbowDQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import pandas as pd
import os
import gc

#gym.register_envs(ale_py) # if auto registration fails

def setup_env(render_mode=None):
    base_env = gym.make('ALE/Assault-v5', render_mode=render_mode, frameskip=1) # disable base env skipping
    monitor_env = Monitor(base_env)
    preprocessing_env = AtariPreprocessing(monitor_env)  # grayscale, downsample, skip frames
    stacked_env = FrameStackObservation(preprocessing_env, 4)       # stacked frames for temporal info
    
    # debugging
    #print("Action space:", env.action_space)
    #print("Observation space:", env.observation_space)

    return stacked_env, monitor_env

class agent:
    def __init__(self):
        self.name = None
    def set_model(self, name, env):
        """ Set up the RL agent model based on the specified name. (Optional)"""
        self.name = name
        self.model = None

        if not os.path.exists('./logs'):
            os.makedirs('./logs')

        if name == "DQN":
            # Dqn Policy for image input, tensorboard logging for visualization, memory handling
            self.model = DQN("CnnPolicy", env, buffer_size=100_000, learning_starts=10_000, verbose=1, tensorboard_log=f"./logs/assault_tensorboard/")

        elif name == "PPO":    
            self.model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=f"./logs/assault_tensorboard/")

        elif name == "RDQN":    
            self.model = RainbowDQN("CnnPolicy", env, verbose=1, tensorboard_log=f"./logs/assault_tensorboard/")

        return self.model

def visualize_trained_agent(env, model, episodes=5):
    """ Visualize a trained agent in the environment. """
    # game loop
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

def print_results(monitor_env):
    """ Print the results of the training in console. """
    episode_rewards = monitor_env.get_episode_rewards()  # only available after Monitor tracks episodes
    print("Episode rewards:", episode_rewards)
    print("Mean reward:", sum(episode_rewards)/len(episode_rewards))
    print("Number of episodes:", len(episode_rewards))

def plot_results(monitor_env):
    """ Plot the results of the training as bar chart. Reward per episode. """
    episode_rewards = monitor_env.get_episode_rewards()  # only available after Monitor tracks episodes
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward Curve")
    plt.show()

# OPTUNA
def tune_ppo(env_id="ALE/Assault-v5", n_trials=10, timesteps=5000, seed=42):
    """
    Tune PPO hyperparameters with Optuna.
    Returns the best hyperparameters and the corresponding study.
    """
    
    def objective(trial):
        # Suggest hyperparameters
        n_steps = trial.suggest_categorical("n_steps", [128, 256, 512, 1024])
        gamma = trial.suggest_float("gamma", 0.9, 0.9999)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        ent_coef = trial.suggest_float("ent_coef", 0.0, 0.01)
        gae_lambda = trial.suggest_float("gae_lambda", 0.8, 1.0)
        clip_range = trial.suggest_float("clip_range", 0.1, 0.4)

        # Create environment
        env, monitor_env = setup_env(render_mode=None)

        # Create PPO model
        model = PPO(
            "CnnPolicy",
            env,
            n_steps=n_steps,
            gamma=gamma,
            learning_rate=learning_rate,
            ent_coef=ent_coef,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            verbose=0,
            seed=seed
        )

        # Train for a short number of timesteps
        model.learn(total_timesteps=timesteps)

        # Compute mean reward
        episode_rewards = monitor_env.get_episode_rewards()
        mean_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0

        env.close()
        return mean_reward

    # Run the study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("Best hyperparameters:", study.best_params)
    print("Best mean reward:", study.best_value)
    return study.best_params, study

# OPTUNA for DQN 
def tune_dqn(env_id="ALE/Assault-v5", n_trials=10, timesteps=5000, seed=42):
    """
    Tune DQN hyperparameters with Optuna.
    Returns the best hyperparameters and the corresponding study.
    """
    def objective(trial):
        # Suggest hyperparameters
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3)
        gamma = trial.suggest_float("gamma", 0.90, 0.999)
        exploration_fraction = trial.suggest_float("exploration_fraction", 0.05, 0.3)
        exploration_final_eps = trial.suggest_float("exploration_fraction", 0.01, 0.1)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

        # Create environment
        env, monitor_env = setup_env(render_mode=None)

        model = DQN(
            "CnnPolicy",
            env,
            buffer_size=100_000, # limit buffer size from memory usage
            learning_starts=10_000,
            learning_rate=learning_rate,
            gamma=gamma,
            exploration_fraction=exploration_fraction,
            exploration_final_eps=exploration_final_eps,
            batch_size=batch_size,
            verbose=0,
            seed=seed
        )

        # Train for a short number of timesteps
        model.learn(total_timesteps=timesteps)

        # Compute mean reward
        episode_rewards = monitor_env.get_episode_rewards()
        mean_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0

        # for memory issues
        env.close()
        del model
        gc.collect()
        
        return mean_reward

    # Run the study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("Best hyperparameters:", study.best_params)
    print("Best mean reward:", study.best_value)
    return study.best_params, study
    

# script run
if __name__ == "__main__":
    print("Starting main.py")
    
    # setup env
    print("Setting up environment...")
    env, monitor_env = setup_env(render_mode=None) # "human" or "rgb_array" or None

    # setup model
    print("Setting up model...")
    agent = agent()
    model = agent.set_model(name="RDQN", env=env) # "DQN", "PPO" or "RDQN" 

    # train agent
    print("Training model...")
    agent.model.learn(total_timesteps=10_000) # 10k for quick test

    # save agent
    print("Saving model...")
    if not os.path.exists('./models'):
        os.makedirs('./models')
    agent.model.save(f"./models/{agent.name}_assault")

    # print and plot results
    print("Printing results...")
    print_results(monitor_env)
    print("Plotting results...")
    plot_results(monitor_env)

    env.close()

    # visualize trained model
    #print("Testing model...")
    #env = setup_env(render_mode="Human")
    #visualize_trained_agent(env, agent.model, episodes=2)

    # -- OPTUNA HYPER PARAMETER TUNING -- 
    if agent.name == "PPO":
        # OPTUNA tuning on PPO agent
        print("Tuning new PPO agent hyperparameters with Optuna...")
        env, monitor_env = setup_env(render_mode=None) 
        best_params, study = tune_ppo(n_trials=15, timesteps=10_000)
        
        # Full training with best hyperparameters (longer)
        best_model = PPO(
            "CnnPolicy",
            env,
            n_steps=best_params["n_steps"],
            gamma=best_params["gamma"],
            learning_rate=best_params["learning_rate"],
            ent_coef=best_params["ent_coef"],
            gae_lambda=best_params["gae_lambda"],
            clip_range=best_params["clip_range"],
            verbose=1
        )
        best_model.learn(total_timesteps=50_000)  # much longer training
        best_model.save("./models/PPO_assault_best")
        print("Best model trained and saved.")

        # print and plot results
        print("Printing results...")
        print_results(monitor_env)
        print("Plotting results...")
        plot_results(monitor_env)

        env.close()

    elif agent.name == "DQN":
        # OPTUNA tuning on DQN agent
        print("Tuning new DQN agent hyperparameters with Optuna...")
        env, monitor_env = setup_env(render_mode=None) 
        best_params, study = tune_dqn(n_trials=10, timesteps=10_000)
        
        # Full training with best hyperparameters (longer)
        best_model = DQN(
            "CnnPolicy",
            env,
            buffer_size=100_000, # limit buffer size from memory usage
            learning_starts=10_000,
            learning_rate=best_params["learning_rate"],
            gamma=best_params["gamma"],
            exploration_fraction=best_params["exploration_fraction"],
            batch_size=best_params["batch_size"],
            exploration_final_eps=best_params["exploration_final_eps"],
            verbose=1,
            tensorboard_log=f"./logs/assault_tensorboard/"
        )
        best_model.learn(total_timesteps=50_000)  
        best_model.save("./models/DQN_assault_best")
        print("Best model trained and saved.")

        # print and plot results
        print("Printing results...")
        print_results(monitor_env)
        print("Plotting results...")
        plot_results(monitor_env)

        env.close()
    
    elif agent.name == "RDQN":
        print("RDQN Finished.")


        