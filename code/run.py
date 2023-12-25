import numpy as np
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
from pettingzoo.mpe import simple_speaker_listener_v4
# save the agents object as a pickle file
import pickle
import os.path


def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        # accumulates the observations of all agents into a single state vector for the critic (global state)
        state = np.concatenate([state, obs])
    return state


def run():
    parallel_env = simple_speaker_listener_v4.parallel_env(
            continuous_actions=True)
    _, _ = parallel_env.reset()
    n_agents = parallel_env.max_num_agents

    actor_dims = []
    n_actions = []
    num_subpolicies = 3
    i = 0
    for agent in parallel_env.agents:
        i += 1
        actor_dims.append(parallel_env.observation_space(agent).shape[0])
        print(f"agent {i}: action_space {parallel_env.action_space(agent).shape[0]}")
        n_actions.append(parallel_env.action_space(agent).shape[0])
        print(f"observation_space {parallel_env.observation_space(agent).shape[0]}")
    


    critic_dims = sum(actor_dims) + sum(n_actions)


    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions,
                        env=parallel_env, gamma=0.95, alpha=1e-4, beta=1e-3,num_subpolicies=num_subpolicies)
    
    critic_dims = sum(actor_dims)
    print(f"critic_dims {critic_dims}")
    print(f"actor_dims {actor_dims}")
    agents_names = [agent.agent_name for agent in maddpg_agents.agents]


    memory = {
        f"{agents_names[0]}":[MultiAgentReplayBuffer(1_000_000, critic_dims, actor_dims,
                                    n_actions, n_agents, batch_size=1024) for _ in range(num_subpolicies)],
        f"{agents_names[1]}":[MultiAgentReplayBuffer(1_000_000, critic_dims, actor_dims,
                                    n_actions, n_agents, batch_size=1024) for _ in range(num_subpolicies)]                        
        }

    # memory = MultiAgentReplayBuffer(1_000_000, critic_dims, actor_dims,
    #                                 n_actions, n_agents, batch_size=1024)

    EVAL_INTERVAL = 1000
    MAX_STEPS = 625_000 # (25_000 episodes)

    total_steps = 0
    episode = 0
    eval_scores = []
    eval_steps = []
    print(f" total steps :{total_steps}")

    # evaluate the agents once before training starts to get a baseline
    score = evaluate(maddpg_agents, parallel_env, episode, total_steps)
    eval_scores.append(score)
    eval_steps.append(total_steps)

    # now train the agents for a specified number of episodes
    while total_steps < MAX_STEPS:
        # reset the environment and get the initial state
        obs, _ = parallel_env.reset() # returns a dictionary of observations for each agent
        terminal = [False] * n_agents
        
        # keep looping until the episode is finished
        while not any(terminal):

            # select actions based on the current state of the environment , returns a list of actions for each agent
            actions,actions_dict = maddpg_agents.choose_action(obs,episode=episode)

            obs_, reward, done, trunc, info = parallel_env.step(actions)

            # store the transition in the replay buffer (experience replay)
            list_done = list(done.values())
            list_obs = list(obs.values())
            list_reward = list(reward.values())
            list_actions = list(actions.values())
            list_obs_ = list(obs_.values())
            list_trunc = list(trunc.values())

            # global state is the observations of all agents
            state = obs_list_to_state_vector(list_obs)
            # global new state is the observations of all agents after each agent's step
            state_ = obs_list_to_state_vector(list_obs_)

            # terminal is true if any agent is done or truncated (either reached the goal or max steps)
            terminal = [d or t for d, t in zip(list_done, list_trunc)]

            # store the transition in the replay buffer
            memory[f"{agents_names[0]}"][maddpg_agents.agents[0].current_subpolicy_idx].store_transition(list_obs, state, list_actions, list_reward,
                                    list_obs_, state_, terminal)
            memory[f"{agents_names[1]}"][maddpg_agents.agents[1].current_subpolicy_idx].store_transition(list_obs, state, list_actions, list_reward,
                                    list_obs_, state_, terminal)

            # memory.store_transition(list_obs, state, list_actions, list_reward,
            #                         list_obs_, state_, terminal)

            if total_steps % 100 == 0:
                maddpg_agents.learn(memory)
            obs = obs_
            total_steps += 1

        if total_steps % EVAL_INTERVAL == 0:
            score = evaluate(maddpg_agents, parallel_env, episode, total_steps)
            eval_scores.append(score)
            eval_steps.append(total_steps)

        episode += 1
        print(f"episode {episode} total steps {total_steps}")

    np.save('data/maddpg_scores.npy', np.array(eval_scores))
    np.save('data/maddpg_steps.npy', np.array(eval_steps))
    return maddpg_agents, parallel_env



def evaluate(agents, env, ep, step, n_eval=3):
    score_history = []
    print(n_eval)
    for i in range(n_eval):
        obs, _ = env.reset()
        score = 0
        terminal = [False] * env.max_num_agents
        while not any(terminal):
            actions,_ = agents.choose_action(obs, evaluate=True)
            obs_, reward, done, trunc, info = env.step(actions)

            list_trunc = list(trunc.values())
            list_reward = list(reward.values())
            list_done = list(done.values())

            terminal = [d or t for d, t in zip(list_done, list_trunc)]

            obs = obs_
            score += sum(list_reward)
        score_history.append(score)
    avg_score = np.mean(score_history)
    print(f'Evaluation episode {ep} train steps {step}'
        f' average score {avg_score:.1f}')
    return avg_score





def visualize_agents(agents, env, n_episodes=20):
    import matplotlib.pyplot as plt
    from IPython import display

    for episode in range(n_episodes):
        obs, _ = env.reset()
        terminal = [False] * env.max_num_agents
        while not any(terminal):
            actions,_ = agents.choose_action(obs, evaluate=False,episode=episode)
            obs, rewards, done, trunc, _ = env.step(actions)
            # sum rewards
            rewards = sum(rewards.values())
            print(rewards)

            terminal = [d or t for d, t in zip(done.values(), trunc.values())]

            # Render as an RGB array and display
            img = env.render()
            plt.imshow(img)
            display.clear_output(wait=True)
            display.display(plt.gcf())
            plt.pause(0.001)  # Adjust as needed for frame rate
        print(any(terminal))


        print(f'Episode {episode + 1} completed')
        plt.close()

if __name__ == '__main__':
    if os.path.isfile('trained_agents.pkl'):
        print('File exists')
        with open('trained_agents.pkl', 'rb') as f:
            agents = pickle.load(f)
            parallel_env = simple_speaker_listener_v4.parallel_env(
            continuous_actions=True,render_mode='rgb_array')
            visualize_agents(agents, parallel_env)

    else:
        print('File does not exist')
        # track how much time it took to train the agents
        import time
        start = time.time()
        agents, env = trained_agents, environment = run()
        end = time.time()
        print(f'Training took {end - start:.2f} seconds')
        with open('trained_agents.pkl', 'wb') as f:
            pickle.dump(agents, f)
    
