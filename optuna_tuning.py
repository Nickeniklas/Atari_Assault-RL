import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import ale_py
import optuna
from stable_baselines3 import DQN, PPO
from sb3_contrib import QRDQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import gc
import numpy as np

from setup_env import setup_env

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
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

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
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("Best hyperparameters:", study.best_params)
    print("Best mean reward:", study.best_value)
    return study.best_params, study

# OPTUNA for QR-DQN 
def tune_qrdqn(env_id="ALE/Assault-v5", n_trials=10, timesteps=5000, seed=42):
    """
    Tune QR-DQN hyperparameters with Optuna.
    Returns the best hyperparameters and the corresponding study.
    """
    def objective(trial):
        # Unique random seed per trial
        seed = np.random.randint(0, 1_000_000)

        # Suggest hyperparameters
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True) # better sample the log scale
        gamma = trial.suggest_float("gamma", 0.90, 0.999)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        train_freq = trial.suggest_categorical("train_freq", [1, 4])
        exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.5)
        exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.01, 0.1)
        
        # Create environment
        env, monitor_env = setup_env(render_mode=None)

        model = QRDQN(
            "CnnPolicy",
            env,
            buffer_size=100_000, # limit buffer size from memory usage
            learning_starts=10_000,
            learning_rate=learning_rate,
            gamma=gamma,
            batch_size=batch_size,
            train_freq=train_freq,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=1.0,
            exploration_final_eps=exploration_final_eps,
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
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("Best hyperparameters:", study.best_params)
    print("Best mean reward:", study.best_value)
    return study.best_params, study