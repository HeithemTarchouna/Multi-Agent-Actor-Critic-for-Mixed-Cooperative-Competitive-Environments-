import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)



def calculate_tick_range(values, interval=20):
    start = np.floor(min(values) / interval) * interval
    end = np.ceil(max(values) / interval) * interval
    return np.arange(start, end + 1, interval)


# Create a function to save plots
def save_plot(plt, filename, output_dir):
    if not os.path.exists(output_dir):  # Check if the output directory exists, create it if not
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, filename))

def plot_average_episode_rewards(average_rewards, scenario, output_dir):
    plt.figure(figsize=(12, 6))
    plt.xlabel('Episode')
    plt.ylabel('Average Episode Reward')
    plt.title(f'Average Episode Reward Progress - {scenario}')
    plt.plot(range(1, len(average_rewards) + 1), average_rewards)
    plt.yticks(calculate_tick_range(average_rewards))
    plt.grid()
    save_plot(plt, f'Average_Episode_Reward_Progress_{scenario}.png', output_dir)
    
def plot_average_episode_rewards_rolling(average_rewards, scenario, output_dir):
    plt.figure(figsize=(12, 8))
    plt.xlabel('Episode')
    plt.ylabel('Average Episode Reward')
    plt.title(f'Average Episode Reward Progress - {scenario}')
    
    plt.plot(range(1, len(average_rewards) + 1), average_rewards, alpha=0.3, label='Original')
    
    rewards_series = pd.Series(average_rewards)
    smoothed_rewards = rewards_series.rolling(window=100).mean()

    plt.plot(range(1, len(average_rewards) + 1), smoothed_rewards, color='red', label='Smoothed (Rolling Mean)')
    
    plt.yticks(calculate_tick_range(smoothed_rewards.dropna()))
    plt.legend()
    plt.grid()
    save_plot(plt, f'Average_Episode_Reward_Progress_Rolling_Mean{scenario}.png', output_dir)

def plot_all_agents_rewards(agent_rewards, scenario, output_dir):
    plt.figure(figsize=(12, 6))
    plt.xlabel('Episode')
    plt.ylabel('Agent Reward')
    plt.title(f'All Agents Reward Progress (agent + avdversary) - {scenario}')
    
    all_rewards = np.concatenate([rewards for rewards in agent_rewards.values()])
    for agent_name, rewards in agent_rewards.items():
        plt.plot(range(1, len(rewards) + 1), rewards, label=f'Agent {agent_name}')
    
    plt.yticks(calculate_tick_range(all_rewards))
    plt.legend()
    plt.grid()
    save_plot(plt, f'All_Agents_Reward_Progress_{scenario}.png', output_dir)

def plot_individual_agent_rewards(epsiode_mean_agent_rewards, agent_name, scenario, output_dir):
    plt.figure(figsize=(12, 6))
    plt.xlabel('Episode')
    plt.ylabel('Mean Agent Reward')
    plt.title(f'Mean Agent Reward Progress - {scenario} (Agent {agent_name})')
    plt.plot(range(1, len(epsiode_mean_agent_rewards) + 1), epsiode_mean_agent_rewards, label=f'Agent {agent_name}')
    
    plt.yticks(calculate_tick_range(epsiode_mean_agent_rewards))
    plt.legend()
    plt.grid()
    save_plot(plt, f'Individual_Agent_Reward_Progress_{scenario}_Agent_{agent_name}.png', output_dir)

def plot_everything(output_dir, scenario, k, score_history_100, score_history, epsiode_mean_agent_rewards):
    # Plot results for different subpolicies
            output_subdir = os.path.join(output_dir, f'scenario_{scenario}', f'k_{k}')
            os.makedirs(output_subdir, exist_ok=True)

            # Plot for average episode rewards fancy
            plot_average_episode_rewards(score_history_100, f"{scenario} - {k}", output_subdir)

            # Plot for average episode rewards with rolling mean
            plot_average_episode_rewards_rolling(score_history, f"{scenario} - {k}", output_subdir)
            
            # Plot for individual agent rewards
            plot_all_agents_rewards(epsiode_mean_agent_rewards, f"{scenario} - {k}", output_subdir)
            
            # Plot for 'agent_0' only
            agent_name = 'agent_0'
            if agent_name in epsiode_mean_agent_rewards:
                plot_individual_agent_rewards(epsiode_mean_agent_rewards[agent_name], agent_name, f"{scenario} - {k}", output_subdir)