import os
import numpy as np
from tqdm import tqdm
from DDPG.agent import Agent
from pettingzoo.mpe import simple_adversary_v3, simple_speaker_listener_v4, simple_spread_v3, simple_reference_v3, simple_tag_v3, simple_crypto_v3


def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state


def train_DDPG(parallel_env, N_GAMES):
    _, _ = parallel_env.reset()
    n_agents = parallel_env.max_num_agents

    n_actions = []
    agents = []

    for agent in parallel_env.agents:
        input_dims = parallel_env.observation_space(agent).shape[0]
        n_actions = parallel_env.action_space(agent).shape[0]

        agents.append(Agent(input_dims=input_dims, n_actions=n_actions,
                            gamma=0.95, tau=0.01, alpha=1e-4, beta=1e-3))

    EVAL_INTERVAL = 1000
    MAX_STEPS = N_GAMES * 25  # 25 steps per episode
    total_steps = 0
    episode = 0

    eval_scores = []
    eval_steps = []
    score = evaluate(agents, parallel_env, episode, total_steps)
    eval_scores.append(score)
    eval_steps.append(total_steps)

    pbar = tqdm(total=MAX_STEPS, desc="Training DDPG")

    while total_steps < MAX_STEPS:
        obs, _ = parallel_env.reset()
        terminal = [False] * n_agents
        obs = list(obs.values())
        while not any(terminal):
            action = [agent.choose_action(obs[idx])
                      for idx, agent in enumerate(agents)]
            action = {agent: act
                      for agent, act in zip(parallel_env.agents, action)}
            obs_, reward, done, truncated, info = parallel_env.step(action)
            list_done = list(done.values())
            list_reward = list(reward.values())
            list_action = list(action.values())
            obs_ = list(obs_.values())
            list_trunc = list(truncated.values())

            terminal = [d or t for d, t in zip(list_done, list_trunc)]

            for idx, agent in enumerate(agents):
                agent.remember(obs[idx], list_action[idx],
                               list_reward[idx], obs_[idx], terminal[idx])

            if total_steps % 125 == 0:
                for agent in agents:
                    agent.learn()
            obs = obs_
            total_steps += 1
            pbar.update(1)

        if total_steps % EVAL_INTERVAL == 0 and total_steps > 0:
            score = evaluate(agents, parallel_env, episode, total_steps)
            eval_scores.append(score)
            eval_steps.append(total_steps)

        episode += 1

        # Create the 'data' directory if it doesn't exist
        if not os.path.exists('data'):
            os.makedirs('data')

        # Save the files in the 'data' directory
        np.save('data/ddpg_scores.npy', np.array(eval_scores))
        np.save('data/ddpg_steps.npy', np.array(eval_steps))

    pbar.close()
    return eval_scores


def evaluate(agents, env, ep, step):
    score_history = []
    for i in range(3):
        obs, _ = env.reset()
        score = 0
        terminal = [False] * env.max_num_agents
        obs = list(obs.values())
        while not any(terminal):
            action = [agent.choose_action(obs[idx], eval=True)
                      for idx, agent in enumerate(agents)]
            action = {agent: act
                      for agent, act in zip(env.agents, action)}

            obs_, reward, done, truncated, info = env.step(action)
            obs_ = list(obs_.values())
            list_trunc = list(truncated.values())
            list_reward = list(reward.values())
            list_done = list(done.values())

            terminal = [d or t for d, t in zip(list_done, list_trunc)]

            obs = obs_
            score += sum(list_reward)
        score_history.append(score)
    avg_score = np.mean(score_history)
    
    return avg_score


if __name__ == '__main__':
    # parallel_env,scenario = simple_tag_v3.parallel_env(max_cycles=25, continuous_actions=True, render_mode="rgb_array"), "predator_prey"
    # train_DDPG(parallel_env=parallel_env,N_GAMES=1000)
    pass