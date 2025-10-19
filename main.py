from stable_baselines3 import DQN, PPO
from sb3_contrib import QRDQN
import os

from visualize import visualize_trained_agent, print_results, plot_results
from setup_env import setup_env
from optuna_tuning import tune_ppo, tune_dqn, tune_qrdqn
from agent import agent

# script run
if __name__ == "__main__":
    print("Starting main.py")
    
    # setup env
    print("Setting up environment...")
    env, monitor_env = setup_env(render_mode=None) # "human" or "rgb_array" or None

    # setup model
    print("Setting up model...")
    agent = agent()
    model = agent.set_model(name="QRDQN", env=env, seed=12) # "DQN", "PPO" or "QRDQN" 

    # train agent
    print("Training model...")
    agent.model.learn(total_timesteps=10_000) # 10k for quick test

    episode_rewards = []
    for _ in range(10):
        obs, info = env.reset()  # unpack tuple
        done = False
        total = 0
        while not done:
            action, _ = agent.model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            done = terminated or truncated  # reset when episode ends
        episode_rewards.append(total)
    print(episode_rewards)

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

    monitor_env.close()
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
        best_model.save(f"./models/{agent.name}_assault_best")
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
            tensorboard_log="./logs/assault_tensorboard/"
        )
        best_model.learn(total_timesteps=50_000)  
        best_model.save(f"./models/{agent.name}_assault_best")
        print("Best model trained and saved.")

        # print and plot results
        print("Printing results...")
        print_results(monitor_env)
        print("Plotting results...")
        plot_results(monitor_env)

        env.close()
    
    elif agent.name == "QRDQN":
        # OPTUNA tuning on QR-DQN agent
        print("Tuning new QR-DQN agent hyperparameters with Optuna...")
        env, monitor_env = setup_env(render_mode=None) 
        best_params, study = tune_qrdqn(n_trials=20, timesteps=10_000)
        
        # Full training with best hyperparameters (longer)
        best_model = QRDQN(
            "CnnPolicy",
            env,
            buffer_size=100_000, # limit buffer size from memory usage
            learning_starts=10_000,
            learning_rate=best_params["learning_rate"],
            gamma=best_params["gamma"],
            batch_size=best_params["batch_size"],
            train_freq=best_params["train_freq"],
            verbose=1,
            tensorboard_log="./logs/assault_tensorboard/"
        )
        best_model.learn(total_timesteps=50_000)  
        best_model.save(f"./models/{agent.name}_assault_best")
        print("Best model trained and saved.")

        # print and plot results
        print("Printing results...")
        print_results(monitor_env)
        print("Plotting results...")
        plot_results(monitor_env)

        env.close()


        