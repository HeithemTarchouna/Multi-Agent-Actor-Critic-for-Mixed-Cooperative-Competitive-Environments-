import numpy as np
from maddpg import MADDPG
from pettingzoo.mpe import simple_adversary_v3,simple_speaker_listener_v4,simple_spread_v3,simple_reference_v3,simple_tag_v3,simple_crypto_v3
import warnings
import time
import matplotlib.pyplot as plt
from IPython import display

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

def visualize_agents(agents, env, n_episodes=20, speed=0.1):
    # Ensure speed is between 0 and 1
    speed = np.clip(speed, 0, 1)

    # Create a figure outside the loop
    plt.figure()
    for episode in range(n_episodes):
        prev_reward = -np.inf
        obs, _ = env.reset()
        terminal = [False] * env.num_agents

        while not any(terminal):
            actions = agents.choose_action(obs)
            obs, rewards, done, truncation, _ = env.step(actions)

            # Sum rewards
            rewards = sum(rewards.values())

            # Determine direction
            direction = "Right direction" if rewards > prev_reward else "Wrong direction"
            prev_reward = rewards
            
            # Render as an RGB array
            img = env.render()

            # Clear the current axes and plot the new image
            plt.clf()
            plt.imshow(img)

            # Determine the center position for the text
            center_x = img.shape[1] / 2

            # Add direction text to the figure, centered horizontally
            plt.text(center_x, 20, direction, fontsize=12, color='white', bbox=dict(facecolor='black', alpha=1), ha='center')

            # Display the updated figure
            display.clear_output(wait=True)
            display.display(plt.gcf())
            plt.pause(0.1 / speed)

            terminal = [d or t for d, t in zip(done.values(), truncation.values())]

        print(f'Episode {episode + 1} completed')
        plt.close()

def solve_env(env,scenario,N_GAMES,evaluate,k=1):
    print("Solving env",scenario)
    obs = env.reset()
 
    n_agents = env.num_agents
    actor_dims = [env.observation_spaces[agent_name].shape[0] for agent_name in env.agents]
    n_actions = [env.action_spaces[agent_name].shape[0] for agent_name in env.agents]
    critic_dims = sum(actor_dims) + sum(n_actions)
    # what eveyone is seeing
    whole_state_observation_dims = sum(actor_dims)

    maddpg_agents = MADDPG(actor_dims, critic_dims,whole_state_observation_dims, n_agents, n_actions,
                           fc1=32, fc2=32,
                           alpha=0.01, beta=0.01, scenario=scenario,
                           chkpt_dir='tmp/maddpg/',env=env,k=k)
 
    LOAD_TYPE = ["Regular","Best"] # Regular: save every 10k, Best: save only if avg_score > best_score
    PRINT_INTERVAL = 500
    SAVE_INTERVAL = 10000
    MAX_STEPS = 25
    total_steps = 0
    score_history = []
    best_score = -3
 
    if evaluate:
        maddpg_agents.load_checkpoint(LOAD_TYPE[1]) # load best
        visualize_agents(maddpg_agents, env, n_episodes=20, speed=10)
    else:
        for i in range(N_GAMES):
            obs,_ = env.reset()
            score = 0
            done = [False] * n_agents
            episode_step = 0
            # each episode, randomly choose a subpolicy
            maddpg_agents.randomly_choose_subpolicy()
            while not any(done):
                actions = maddpg_agents.choose_action(obs)
    
                obs_, reward, termination, truncation, _ = env.step(actions)
    
                state = np.concatenate([i for i in obs.values()])
                state_ = np.concatenate([i for i in obs_.values()])
    
                if episode_step >= MAX_STEPS:
                    done = [True] * n_agents
    
                if any(termination.values()) or any(truncation.values()) or (episode_step >= MAX_STEPS):
                    done = [True] * n_agents
    
                maddpg_agents.store_transition(obs, state, actions, reward, obs_, state_, done)
    
                if total_steps % 5 == 0:
                    maddpg_agents.learn()
    
                obs = obs_
    
                score += sum(reward.values())
                total_steps += 1
                episode_step += 1
    
    
            score_history.append(score)
            avg_score = np.mean(score_history[-100:]) 
            if (avg_score > best_score) and (i > PRINT_INTERVAL):
                print(" avg_score, best_score", avg_score, best_score)
                maddpg_agents.save_checkpoint(LOAD_TYPE[1])


                best_score = avg_score
            if i % PRINT_INTERVAL == 0 and i > 0:
                print('episode', i, 'average score {:.1f}'.format(avg_score))
            if i % SAVE_INTERVAL == 0 and i >0 :
                maddpg_agents.save_checkpoint(LOAD_TYPE[0])


env1,scenario1 = simple_tag_v3.parallel_env(max_cycles=25, continuous_actions=True,render_mode="rgb_array"),"predator_prey"
env2,scenario2 = simple_reference_v3.parallel_env(max_cycles=25, continuous_actions=True),"Physical_Deception"
env3,scenario3 = simple_speaker_listener_v4.parallel_env(max_cycles=25, continuous_actions=True,render_mode="rgb_array"),"Cooperative_Communication"
env4,scenario4 = simple_crypto_v3.parallel_env(max_cycles=25, continuous_actions=True),"Covert_Communication"
env5,scenario5 = simple_spread_v3.parallel_env(max_cycles=25, continuous_actions=True),"Cooperative_Navigation"
env6,scenario6 = simple_adversary_v3.parallel_env(N=2, max_cycles=25, continuous_actions=True,render_mode='rgb_array'),"Keep_Away"

envs =[env1,env2,env3,env4,env5,env6]
scenarios = [scenario1,scenario2,scenario3,scenario4,scenario5,scenario6]


solve_env(env1,scenario1,N_GAMES=25_000,evaluate=True,k=3)

