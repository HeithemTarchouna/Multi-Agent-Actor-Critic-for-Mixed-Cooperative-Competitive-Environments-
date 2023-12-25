import numpy as np

class MultiAgentReplayBuffer:
    def __init__(self, max_size, critic_dims, actor_dims, n_actions, n_agents, batch_size):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.batch_size = batch_size

        self.memory = {
            'state': np.zeros((self.mem_size, critic_dims)), #global state : observations of all agents
            'new_state': np.zeros((self.mem_size, critic_dims)), #global new state, updated after each agent's step
            'reward': np.zeros((self.mem_size, n_agents)), # each agent's reward
            'terminal': np.zeros((self.mem_size, n_agents), dtype=bool) # each agent's done
        }

        self.actor_memory = {
            'state': [np.zeros((self.mem_size, actor_dims[i])) for i in range(n_agents)], # each agent's observations
            'new_state': [np.zeros((self.mem_size, actor_dims[i])) for i in range(n_agents)], # each agent's new observations
            'action': [np.zeros((self.mem_size, n_actions[i])) for i in range(n_agents)] # each agent's action
        }

    def store_transition(self, raw_obs, state, action, reward, raw_obs_, state_, done):
        # index of the current transition in the replay buffer, modulo the buffer size to overwrite old transitions (it's basicall a circular )
        index = self.mem_cntr % self.mem_size 

        for agent_idx in range(self.n_agents):

            self.actor_memory['state'][agent_idx][index] = raw_obs[agent_idx] # assign the oservtion of each agent to the corresponding agent's memory
            self.actor_memory['new_state'][agent_idx][index] = raw_obs_[agent_idx] # assign the new oservtion of each agent to the corresponding agent's memory
            self.actor_memory['action'][agent_idx][index] = action[agent_idx] # assign the action of each agent to the corresponding agent's memory

        self.memory['state'][index] = state # assign the global observation to the global memory
        self.memory['new_state'][index] = state_ # assign the global new observation to the global memory
        self.memory['reward'][index] = reward # assign the global reward to the global memory
        self.memory['terminal'][index] = done # assign the global done to the global memory

        self.mem_cntr += 1

    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size) # if the memory is not full, we sample from the current memory size, otherwise we sample from the max memory size

        batch = np.random.choice(max_mem, self.batch_size, replace=False) # return a list of batch_size random integers from 0 to max_mem


        sample = {
            'states': self.memory['state'][batch],
            'new_states': self.memory['new_state'][batch],
            'rewards': self.memory['reward'][batch],
            'terminals': self.memory['terminal'][batch],
            'actor_states': [self.actor_memory['state'][i][batch] for i in range(self.n_agents)],
            'actor_new_states': [self.actor_memory['new_state'][i][batch] for i in range(self.n_agents)],
            'actions': [self.actor_memory['action'][i][batch] for i in range(self.n_agents)]
        }
        
        #  (the dataset) get the sample of the global state, global new state, global reward, global done, each agent's observation, each agent's new observation, each agent's action

        return sample

    def ready(self):
        # we have enough transitions in the buffer to start learning (batch_size)
        return self.mem_cntr >= self.batch_size
