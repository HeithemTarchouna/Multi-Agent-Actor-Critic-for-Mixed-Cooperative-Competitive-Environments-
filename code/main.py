import numpy as np
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
from pettingzoo.mpe import simple_adversary_v3,simple_speaker_listener_v4
import warnings
import time

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state


if __name__ == '__main__':
    #scenario = 'simple'
    scenario = 'simple_adversary'
    env = simple_adversary_v3.parallel_env(N=2, max_cycles=40, continuous_actions=True,)

    # env = simple_speaker_listener_v4.parallel_env(
    #         continuous_actions=True)
    
    obs , _ = env.reset()

    n_agents = env.num_agents
    actor_dims = {}
    n_actions = {}
    
    for agent_name in env.agents:
        actor_dims[f"{agent_name}"]= env.observation_space(agent_name)._shape[0]
        #actor_dims.append(env.observation_space(agent_name)._shape[0])

        agent_max_actions = env.action_space(agent_name).shape[0]
        n_actions[f"{agent_name}"] = agent_max_actions
        #print(f"{agent_name} has {agent_max_actions} actions")

    critic_inputs = sum(list(actor_dims.values()) + list(n_actions.values()))
    #print("critic_inputs", critic_inputs)
    

    global_state_dims = sum(list(actor_dims.values()))
    # action space is a list of arrays, assume each agent has same action space
    #n_actions = 5
    maddpg_agents = MADDPG(actor_dims, critic_inputs, n_agents, n_actions,
                           fc1=64, fc2=64,
                           alpha=0.01, beta=0.01, scenario=scenario,
                           chkpt_dir='tmp/maddpg/',env=env)

    memory = MultiAgentReplayBuffer(1000000, global_state_dims, actor_dims,
                                    n_actions, n_agents, batch_size=1024,
                                    agent_names=env.agents)

    PRINT_INTERVAL = 500
    N_GAMES = 500000
    MAX_STEPS = 40
    total_steps = 0
    score_history = []
    evaluate = False
    best_score = -3

    if evaluate:
        maddpg_agents.load_checkpoint()
    time.sleep(1)
    print("starting training")
    for i in range(N_GAMES):
        obs , _ = env.reset()
        score = 0
        done = [False] * n_agents
        episode_step = 0
        while not any(done):
            if evaluate:
                env.render()

                time.sleep(0.12) # to slow down the action for the video

            actions = maddpg_agents.choose_action(obs)

            obs_, reward, termination, truncation, _ = env.step(actions)

            state = np.concatenate([i for i in obs.values()])
            state_ = np.concatenate([i for i in obs_.values()])

            if episode_step >= MAX_STEPS:
                done = [True] * n_agents

            if any(termination.values()) or any(truncation.values()) or (episode_step >= MAX_STEPS):
                done = [True] * n_agents

            memory.store_transition(obs, state, actions, reward, obs_, state_, done)

            if total_steps % 100 == 0 and not evaluate:
                maddpg_agents.learn(memory)

            obs = obs_

            score += sum(reward.values())
            total_steps += 1
            episode_step += 1

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if not evaluate:

            if (avg_score > best_score) and (i > PRINT_INTERVAL):
                print(" avg_score, best_score", avg_score, best_score)
                # maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score))