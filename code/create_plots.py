import numpy as np
import matplotlib.pyplot as plt
from DDPG.run import train_DDPG
from main import solve_env
from pettingzoo.mpe import simple_adversary_v3, simple_speaker_listener_v4, simple_spread_v3, simple_reference_v3, simple_tag_v3, simple_crypto_v3

def normalize_scores(scores):
    min_score = np.min(scores)
    range_score = np.max(scores) - min_score
    if range_score == 0:
        return np.zeros_like(scores)
    return (scores - min_score) / range_score

def train_and_plot(envs, scenarios, N_GAMES, k=1):
    normalized_scores = {}

    for scenario in scenarios:
        env = envs[scenario]
        # Train MADDPG
        maddpg_mean_rewards, _ = solve_env(env, scenario, N_GAMES, evaluate=False, k=k, plot=False)
        maddpg_avg_score = np.mean([np.mean(rewards) for rewards in maddpg_mean_rewards.values()])
        # Train DDPG
        ddpg_avg_score = np.mean(train_DDPG(env, N_GAMES))
        scores = np.array([maddpg_avg_score, ddpg_avg_score])
        normalized_scores[scenario] = normalize_scores(scores)

    # Plotting
    fig, ax = plt.subplots()
    bar_width = 0.35
    index = np.arange(len(scenarios))
    opacity = 0.8

    for i, scenario in enumerate(scenarios):
        maddpg_bar = ax.bar(index[i] - bar_width/2, normalized_scores[scenario][0], bar_width,
                            alpha=opacity, color='b', label='MADDPG' if i == 0 else "")
        ddpg_bar = ax.bar(index[i] + bar_width/2, normalized_scores[scenario][1], bar_width,
                          alpha=opacity, color='r', label='DDPG' if i == 0 else "")

    ax.set_xlabel('Scenarios')
    ax.set_ylabel('Normalized Score')
    ax.set_title('Performance Comparison Across Scenarios')
    ax.set_xticks(index)
    ax.set_xticklabels(scenarios)
    ax.set_ylim(0, 1)
    ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    envs = {
        "predator_prey": simple_tag_v3.parallel_env(max_cycles=25, continuous_actions=True, render_mode="rgb_array"),
        "Keep_Away": simple_adversary_v3.parallel_env(N=2, max_cycles=25, continuous_actions=True, render_mode='rgb_array')
        # Add other environments and scenarios here as needed
    }
    scenarios = ["predator_prey","Keep_Away"]  # Add the keys of other scenarios here as needed
    N_GAMES = 200
    k = 1
    train_and_plot(envs, scenarios, N_GAMES, k)
