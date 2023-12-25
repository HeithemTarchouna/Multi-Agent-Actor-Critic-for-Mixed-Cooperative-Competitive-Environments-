from agent import Agent


class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, env,
                 alpha=1e-4, beta=1e-3, fc1=64, fc2=64, gamma=0.95, tau=0.01,
                 chkpt_dir='tmp/maddpg/', scenario='co-op_navigation',num_subpolicies=3):
        self.agents = []
        chkpt_dir += scenario
        for agent_idx in range(n_agents):
            agent = list(env.action_spaces.keys())[agent_idx] # get the agent name
            min_action = env.action_space(agent).low
            max_action = env.action_space(agent).high
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,
                               n_actions[agent_idx],agent_idx,agent,
                               alpha=alpha, beta=beta, tau=tau, fc1=fc1,
                               fc2=fc2, chkpt_dir=chkpt_dir,
                               gamma=gamma, min_action=min_action,
                               max_action=max_action,num_subpolicies=num_subpolicies))
            self.prev_episode = 0


    def choose_action(self, raw_obs, evaluate=False,episode=0):
        actions = {}
        actions_dict = {}
        # check if the episode is finished
        if self.prev_episode == episode:
            new_episode = False
        else:
            new_episode = True
            self.prev_episode = episode
        
        for agent_id, agent in zip(raw_obs, self.agents):
            action = agent.choose_action(raw_obs[agent_id], evaluate,new_episode)
            actions[agent_id] = action
            actions_dict[agent.agent_name] = action



        return actions, actions_dict

    def learn(self, memory):
        for agent in self.agents:
            agent.learn(memory, self.agents)    
