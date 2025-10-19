import matplotlib.pyplot as plt

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