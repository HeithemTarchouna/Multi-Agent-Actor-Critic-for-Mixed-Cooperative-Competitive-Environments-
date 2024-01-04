import os
import numpy as np
from maddpg import MADDPG
from pettingzoo.mpe import simple_adversary_v3, simple_speaker_listener_v4, simple_spread_v3, simple_reference_v3, simple_tag_v3, simple_crypto_v3,simple_push_v3
import warnings
import time
import matplotlib.pyplot as plt
from IPython import display
from tqdm import tqdm
from pettingzoo.sisl import waterworld_v4

    
# Create a function to save plots
def save_plot(plt, filename, output_dir):
    if not os.path.exists(output_dir):  # Check if the output directory exists, create it if not
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, filename))

# Define plotting functions without output directory argument
def plot_average_episode_rewards(average_rewards, scenario, output_dir):
    plt.figure(figsize=(12, 6))
    plt.xlabel('Episode')
    plt.ylabel('Average Episode Reward')
    plt.title(f'Average Episode Reward Progress - {scenario}')
    plt.plot(range(1, len(average_rewards) + 1), average_rewards)
    plt.grid()
    save_plot(plt, f'Average_Episode_Reward_Progress_{scenario}.png', output_dir)

def plot_individual_agent_rewards(agent_rewards, scenario, output_dir):
    plt.figure(figsize=(12, 6))
    plt.xlabel('Episode')
    plt.ylabel('Agent Reward')
    plt.title(f'Individual Agent Reward Progress - {scenario}')
    
    for agent_name, rewards in agent_rewards.items():
        plt.plot(range(1, len(rewards) + 1), rewards, label=f'Agent {agent_name}')
    
    plt.legend()
    plt.grid()
    save_plot(plt, f'Individual_Agent_Reward_Progress_{scenario}.png', output_dir)

def plot_episode_lengths(episode_lengths, scenario, output_dir):
    plt.figure(figsize=(12, 6))
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.title(f'Episode Length Progress - {scenario}')
    plt.plot(range(1, len(episode_lengths) + 1), episode_lengths)
    plt.grid()
    save_plot(plt, f'Episode_Length_Progress_{scenario}.png', output_dir)

def plot_mean_agent_rewards(mean_agent_rewards, agent_name, scenario, output_dir):
    plt.figure(figsize=(12, 6))
    plt.xlabel('Episode')
    plt.ylabel('Mean Agent Reward')
    plt.title(f'Mean Agent Reward Progress - {scenario} (Agent {agent_name})')
    plt.plot(range(1, len(mean_agent_rewards) + 1), mean_agent_rewards, label=f'Agent {agent_name}')
    plt.legend()
    plt.grid()
    save_plot(plt, f'Mean_Agent_Reward_Progress_{scenario}_Agent_{agent_name}.png', output_dir)

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

def solve_env_with_subpolicies(env, scenario, N_GAMES, evaluate, k_values=[1], plot=True, output_dir=None):
    results = {}  # Store results for different subpolicies

    for k in k_values:
        print(f"Solving env {scenario} with k={k}")
        obs = env.reset()

        n_agents = env.num_agents
        actor_dims = [env.observation_spaces[agent_name].shape[0] for agent_name in env.agents]
        n_actions = [env.action_spaces[agent_name].shape[0] for agent_name in env.agents]
        critic_dims = sum(actor_dims) + sum(n_actions)
        # what everyone is seeing
        whole_state_observation_dims = sum(actor_dims)

        maddpg_agents = MADDPG(actor_dims, critic_dims, whole_state_observation_dims, n_agents, n_actions,
                               fc1=32, fc2=32,
                               alpha=0.01, beta=0.01, scenario=scenario,
                               chkpt_dir=f'tmp/maddpg/k_{k}/', env=env, k=k)

        LOAD_TYPE = ["Regular", "Best"]  # Regular: save every 10k, Best: save only if avg_score > best_score
        PRINT_INTERVAL = 500
        SAVE_INTERVAL = 5000
        MAX_STEPS = 25
        total_steps = 0
        score_history = []
        best_score = -35
        agent_rewards = {agent_name: [] for agent_name in env.agents}
        mean_agent_rewards = {agent_name: [] for agent_name in env.agents}
        episode_lengths = []
        policy_entropies = []  # List to store policy entropies
        
        if evaluate:
            maddpg_agents.load_checkpoint(LOAD_TYPE[0])  # load best
            visualize_agents(maddpg_agents, env, n_episodes=5, speed=10)
        else:
            for i in tqdm(range(N_GAMES), desc=f"Training with k={k}"):
                obs, _ = env.reset()
                score = 0
                done = [False] * n_agents
                episode_step = 0
                episode_length = 0
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

                    for agent_name, r in reward.items():
                        agent_rewards[agent_name].append(r)

                    score += sum(reward.values())
                    total_steps += 1
                    episode_step += 1
                    episode_length += 1
                
                score_history.append(score)
                avg_score = np.mean(score_history[-100:])
                episode_lengths.append(episode_length)

                if (avg_score > best_score) and (i > PRINT_INTERVAL):
                    print(" avg_score, best_score", avg_score, best_score)
                    maddpg_agents.save_checkpoint(LOAD_TYPE[1])

                    best_score = avg_score
                if i % SAVE_INTERVAL == 0 and i > 0:
                    maddpg_agents.save_checkpoint(LOAD_TYPE[0])

                # Compute mean agent rewards
                for agent_name, rewards in agent_rewards.items():
                    mean_agent_reward = np.mean(rewards)
                    mean_agent_rewards[agent_name].append(mean_agent_reward)

        # Store results for this subpolicy
        results[f'k_{k}'] = {
            'score_history': score_history,
            'agent_rewards': agent_rewards,
            'mean_agent_rewards': mean_agent_rewards,
            'episode_lengths': episode_lengths
        }

        if plot:
            # Plot results for different subpolicies
            output_subdir = os.path.join(output_dir, f'scenario_{scenario}', f'k_{k}')
            os.makedirs(output_subdir, exist_ok=True)

            # Plot for average episode rewards
            plot_average_episode_rewards(score_history, f"{scenario} - {k}", output_subdir)

            # Plot for individual agent rewards
            plot_individual_agent_rewards(agent_rewards, f"{scenario} - {k}", output_subdir)

            # Plot for episode lengths
            plot_episode_lengths(episode_lengths, f"{scenario} - {k}", output_subdir)

            # Plot for all agents
            plt.figure(figsize=(12, 6))
            plt.xlabel('Episode')
            plt.ylabel('Mean Agent Reward')
            plt.title(f'Mean Agent Reward Progress - {scenario} - {k}')
            for agent_name, mean_rewards in mean_agent_rewards.items():
                plt.plot(range(1, len(mean_rewards) + 1), mean_rewards, label=f'Agent {agent_name}')
            plt.legend()
            plt.grid()
            save_plot(plt, f'Mean_Agent_Reward_Progress_{scenario}_k_{k}.png', output_subdir)

            # Plot for 'agent_0' only
            agent_name = 'agent_0'
            if agent_name in mean_agent_rewards:
                plot_mean_agent_rewards(mean_agent_rewards[agent_name], agent_name, f"{scenario} - {k}", output_subdir)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    # Specify the output directory
    output_dir = "plots"

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    env1, scenario1 = simple_tag_v3.parallel_env(max_cycles=25, continuous_actions=True, render_mode="rgb_array"), "predator_prey"
    env2, scenario2 = simple_reference_v3.parallel_env(max_cycles=25, continuous_actions=True), "Physical_Deception"
    env3, scenario3 = simple_speaker_listener_v4.parallel_env(max_cycles=25, continuous_actions=True, render_mode="rgb_array"), "Cooperative_Communication"
    env4, scenario4 = simple_crypto_v3.parallel_env(max_cycles=25, continuous_actions=True), "Covert_Communication"
    env5, scenario5 = simple_spread_v3.parallel_env(max_cycles=25, continuous_actions=True), "Cooperative_Navigation"
    env6, scenario6 = simple_adversary_v3.parallel_env(N=2, max_cycles=25, continuous_actions=True, render_mode='rgb_array'), "Keep_Away"
    env7, scenario7 = simple_push_v3.parallel_env(max_cycles=25, continuous_actions=True,render_mode="rgb_array"), "Push"
    #envs = [env1, env2, env3, env4, env5, env6] # remove env if not needed
    #scenarios = [scenario1, scenario2, scenario3, scenario4, scenario5, scenario6]
    env10,scenario10 = waterworld_v4.parallel_env(max_cycles=25, render_mode="rgb_array"), "Waterworld"

    envs = [env3]
    scenarios = [scenario3]
    # K = 4  Cooperative_Communication
    # K = 3  keep-away and cooperative navigation environments,
    # K = 2 for predator-prey
    # 
    k_values = [1,4]  # Add more values if needed

    for env, scenario in zip(envs, scenarios):
        solve_env_with_subpolicies(env, scenario, N_GAMES=25_000, evaluate=False, k_values=k_values, plot=True, output_dir=output_dir)
