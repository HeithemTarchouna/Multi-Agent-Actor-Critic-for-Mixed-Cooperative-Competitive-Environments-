from agent import Agent

class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, env,
                 alpha=1e-4, beta=1e-3, fc1=64, fc2=64, gamma=0.95, tau=0.01,
                 chkpt_dir='tmp/maddpg/', scenario='co-op_navigation', K=4):
        self.agents = []
        chkpt_dir += scenario
        for agent_idx in range(n_agents):
            agent = list(env.action_spaces.keys())[agent_idx]
            min_action = env.action_space(agent).low
            max_action = env.action_space(agent).high
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,
                                     n_actions[agent_idx], n_agents, agent_idx,
                                     chkpt_dir=chkpt_dir, alpha=alpha, beta=beta,
                                     fc1=fc1, fc2=fc2, gamma=gamma, tau=tau,
                                     min_action=min_action, max_action=max_action, K=K))

    def save_checkpoint(self):
        # Code to save the checkpoints
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        # Code to load the checkpoints
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs, evaluate=False):
        actions = {}
        for agent_id, agent in enumerate(self.agents):
            agent_name = list(raw_obs.keys())[agent_id]  # Get the agent name
            action = agent.choose_action(raw_obs[agent_name], evaluate)
            actions[agent_name] = action  # Use agent name as key
        return actions

    def learn(self, memory):
        for agent in self.agents:
            agent.learn(memory, self.agents, len(self.agents))  # Pass the number of agents
