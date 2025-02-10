import os
import shutil
import subprocess

import numpy as np
import supersuit as ss
import traci
from pyvirtualdisplay.display import Display
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecMonitor
from tqdm import trange

import sumo_rl

PATH = os.path.dirname(sumo_rl.__file__)

if __name__ == "__main__":
    RESOLUTION = (3200, 1800)

    env= sumo_rl.parallel_env(net_file= PATH + "/nets/RESCO/grid4x4/grid4x4.net.xml", 
                              route_file= PATH + "/nets/RESCO/grid4x4/grid4x4_1.rou.xml", 
                              out_csv_name="outputs/grid4x4/ppo_train",
                              use_gui= True, 
                              num_seconds= 3600,
                              render_mode= "human",)
    #env = sumo_rl.grid4x4(use_gui=True, out_csv_name="outputs/grid4x4/ppo_test",virtual_display=RESOLUTION, render_mode='human')
    max_time = env.unwrapped.env.sim_max_time
    delta_time = env.unwrapped.env.delta_time

    print("Environment created")

    # Debugging
    print("Environment created")
    print(hasattr(env, "render_mode"), type(env))
    print(hasattr(env.unwrapped, "render_mode"))
    print(type(env.unwrapped))
    print(getattr(env.unwrapped, "render_mode", "Not set"))
    print(getattr(env, "render_mode", "Not set"))
    print(hasattr(env,"possible_agents"))
    #print(env.reward_function)


    # Explicitly set render_mode for the environment and its unwrapped version
    env.render_mode = "human"
    env.unwrapped.render_mode = "human"
    '''
    #debugging again 
    if not hasattr(env.unwrapped, "render_mode"):
        env.unwrapped.render_mode = "human"

    #extra debugg
    try:
        print("Checking render_mode attribute...")
        print(f"Render mode in unwrapped: {getattr(env.unwrapped, 'render_mode', 'Not set')}")
    except Exception as e:
        print(f"Error accessing render_mode: {e}")

    '''

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 2, num_cpus=1, base_class="stable_baselines3")
    env = VecMonitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=3,
        gamma=0.95,
        n_steps=256, 
        ent_coef=0.0905168,
        learning_rate= 0.00062211, #0.00062211, 
        vf_coef= 0.5,
        max_grad_norm=0.5, #from 0.9, # better performance
        gae_lambda=0.95, # from 0.99 #better performance
        n_epochs= 10, # from 5 much better #10
        clip_range=0.3 , #0.3,
        batch_size= 256 ,
        device= "cpu",
        tensorboard_log="./logs/grid4x4/ppo_train",
    )

    print("Starting training")
    model.learn(total_timesteps=50000)

    # Saving the trained model
    #model.save("ppo_sumo_rl_4x4loop")
    #print("Model saved successfully.")

    print("Training finished. Starting evaluation")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)

    print(mean_reward)
    print(std_reward)

    # Maximum number of steps before reset, +1 because I'm scared of OBOE
    print("Starting rendering")
    num_steps = (max_time // delta_time) + 1

    obs = env.reset()

    if os.path.exists("temp"):
        shutil.rmtree("temp")

    os.mkdir("temp")
    # img = disp.grab()
    # img.save(f"temp/img0.jpg")

    
  
    
    print("All done, cleaning up")
    shutil.rmtree("temp")
    env.close()
    
