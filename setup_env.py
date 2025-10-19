import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from stable_baselines3.common.monitor import Monitor
import ale_py

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