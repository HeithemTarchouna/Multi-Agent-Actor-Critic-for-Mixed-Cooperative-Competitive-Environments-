import numpy as np

class MultiAgentReplayBuffer:
    def __init__(self, max_size, critic_dims, actor_dims, n_actions, n_agents, batch_size):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.batch_size = batch_size

        self.state_memory = np.zeros((self.mem_size, critic_dims))
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))

        self.reward_memory = np.zeros((self.mem_size, n_agents))
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=np.float32)

        self.actor_state_memory = [np.zeros((self.mem_size, actor_dim)) for actor_dim in actor_dims]
        self.actor_new_state_memory = [np.zeros((self.mem_size, actor_dim)) for actor_dim in actor_dims]
        self.actor_action_memory = [np.zeros((self.mem_size, action_dim)) for action_dim in n_actions]

    def store_transition(self, raw_obs, state, action, reward, raw_obs_, state_, done):
        index = self.mem_cntr % self.mem_size
        if state.shape != self.state_memory[index].shape:
            state = np.reshape(state, self.state_memory[index].shape)
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        for agent_idx, (agent_raw_obs, agent_raw_obs_, agent_action) in enumerate(zip(raw_obs, raw_obs_, action)):
            self.actor_state_memory[agent_idx][index] = agent_raw_obs
            self.actor_new_state_memory[agent_idx][index] = agent_raw_obs_
            self.actor_action_memory[agent_idx][index] = agent_action

        self.mem_cntr += 1

    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]

        actor_states = [self.actor_state_memory[agent_idx][batch] for agent_idx in range(self.n_agents)]
        actor_new_states = [self.actor_new_state_memory[agent_idx][batch] for agent_idx in range(self.n_agents)]
        actions = [self.actor_action_memory[agent_idx][batch] for agent_idx in range(self.n_agents)]

        return actor_states, states, actions, rewards, actor_new_states, states_, terminal

    def ready(self):
        return self.mem_cntr >= self.batch_size
