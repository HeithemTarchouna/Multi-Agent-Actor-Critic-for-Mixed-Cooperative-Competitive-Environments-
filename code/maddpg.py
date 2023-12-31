import torch as T
import torch.nn.functional as F
from agent import Agent
from buffer import MultiAgentReplayBuffer
import numpy as np
 
 
class MADDPG:
    def __init__(self, actor_dims, critic_dims,whole_state_observation_dims, n_agents, n_actions, env,
                 scenario='simple',  alpha=0.01, beta=0.01, fc1=32,
                 fc2=32, gamma=0.95, tau=0.01, chkpt_dir='tmp/maddpg/',k=1):
        

        # Create a memory buffer for each agent and each subpolicy
        self.memory = [[MultiAgentReplayBuffer(1_000_000, whole_state_observation_dims, actor_dims,
                                               n_actions, n_agents, batch_size=32,
                                               agent_names=env.agents)
                        for _ in range(k)] for _ in range(n_agents)]
        self.k = k
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        chkpt_dir += scenario
        self.agents = {}
        for agent_idx, agent_name in enumerate(env.possible_agents):
            self.agents[agent_name] = Agent(actor_dims[agent_idx],
                                            critic_dims,
                                            n_actions[agent_idx], n_agents,
                                            agent_name = agent_name,
                                            alpha=alpha,
                                            beta=beta,
                                            chkpt_dir=chkpt_dir,k=k,agent_type=agent_name[0:-2])
        
        # Create a set of agent types : {'adversary', 'agent', etc..}
        self.agent_types= set([agent.agent_type for agent in self.agents.values()])
 
    def save_checkpoint(self,type):
        print('... saving checkpoint ...')
        for agent_name, agent in self.agents.items():
            agent.save_models(type)
 
    def load_checkpoint(self,type):
        print('... loading checkpoint ...')
        for agent_name, agent in self.agents.items():
            agent.load_models(type)
 
    def choose_action(self, raw_obs):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx])
            actions.append(action)
        return actions
 
    def choose_action(self, raw_obs):
        actions = {agent.agent_name: agent.choose_action(raw_obs[agent.agent_name]) for agent in self.agents.values()}
        return actions
 
    def randomly_choose_subpolicy(self):
        # Generate random subpolicies for each agent type in one go
        random_subpolicies = np.random.randint(0, self.k, size=len(self.agent_types))

        # Create a mapping of agent types to their random subpolicy
        self.type_to_subpolicy = dict(zip(self.agent_types, random_subpolicies))

        # Update subpolicy for each agent only when accessed
        for agent in self.agents.values():
            agent.update_subpolicy(self.type_to_subpolicy[agent.agent_type])
        
    
    def store_transition(self, obs, state, action, reward, obs_, state_, done):
        # Store the transition in the memory buffer of the corresponding agent and active subpolicy:
        # specified by the agent_idx and subpolicy_idx
        for agent_idx, agent in enumerate(self.agents.values()):
            self.memory[agent_idx][agent.current_subpolicy_idx].store_transition(obs, state, action, reward, obs_, state_, done)
    
    def learn(self):
        for agent_idx,agent in enumerate(self.agents.values()):
            agent.learn(self.memory[agent_idx][agent.current_subpolicy_idx],self.agents)


        # # Check if the memory buffer has enough samples for learning
        # if not memory.ready():
        #     return
        

        # # Sample a batch of experiences from the memory buffer
        # actor_states, states, actions, rewards, \
        # actor_new_states, states_, dones = memory.sample_buffer()

        # device = self.agents[list(self.agents.keys())[0]].get_current_subpolicy().actor.device

        # # Convert all the sampled data to tensors and move them to the correct device
        # states = T.tensor(states, dtype=T.float).to(device)
        # actions = [T.tensor(actions[agent_idx], device=device, dtype=T.float)
        #            for agent_idx in range(len(list(self.agents.keys())))]
        # rewards = T.tensor(rewards).to(device)
        # states_ = T.tensor(states_, dtype=T.float).to(device)
        # dones = T.tensor(dones).to(device)

        # # Prepare the new (next state) actions for each agent
        # all_agents_new_actions = []
        # all_agents_new_mu_actions = []
        # old_agents_actions = []
 
        # for agent_idx,(agent_name, agent) in enumerate(self.agents.items()):
        #     # Convert new state observations to tensor and move to device
        #     new_states = T.tensor(actor_new_states[agent_idx],
        #                          dtype=T.float).to(device)
            
        #     # Get the action predictions from the target actor of the current subpolicy
        #     new_pi = agent.get_current_subpolicy().target_actor.forward(new_states)
        #     all_agents_new_actions.append(new_pi)
 
        #     # Get the current state observations as tensor
        #     mu_states = T.tensor(actor_states[agent_idx],
        #                          dtype=T.float).to(device)
            
        #     # Get the action predictions from the current actor of the current subpolicy
        #     pi = agent.get_current_subpolicy().actor.forward(mu_states)
        #     all_agents_new_mu_actions.append(pi)

        #     # Get the old actions
        #     old_agents_actions.append(actions[agent_idx])
 
 
 
        # # Concatenate the actions from all agents
        # new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        # mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        # old_actions = T.cat([acts for acts in old_agents_actions],dim=1)
 
 
        # for agent_idx,(agent_name, agent) in enumerate(self.agents.items()):
        #     # Forward pass in the target critic network
        #     critic_value_ = agent.get_current_subpolicy().target_critic.forward(states_, new_actions).flatten()

        #     # Zero out the critic values for the done states
        #     critic_value_[dones[:,0]] = 0.0

        #     # Forward pass in the current critic network
        #     critic_value = agent.get_current_subpolicy().critic.forward(states, old_actions).flatten()

        #     # Calculate the target for the critic network 
        #     target = rewards[:,agent_idx].float() + agent.get_current_subpolicy().gamma*critic_value_
 
 
        #     # Calculate the critic loss and backpropagate
        #     critic_loss = F.mse_loss(target, critic_value)
        #     agent.get_current_subpolicy().critic.optimizer.zero_grad()
        #     critic_loss.backward(retain_graph=True, inputs=list(agent.get_current_subpolicy().critic.parameters()))
        #     agent.get_current_subpolicy().critic.optimizer.step()

        #     # Calculate the actor loss and backpropagate 
        #     actor_loss = agent.get_current_subpolicy().critic.forward(states, mu).flatten()
        #     actor_loss = -T.mean(actor_loss)
        #     agent.get_current_subpolicy().actor.optimizer.zero_grad()
        #     actor_loss.backward(retain_graph=True, inputs=list(agent.get_current_subpolicy().actor.parameters()))
        #     agent.get_current_subpolicy().actor.optimizer.step()

        #     # Update the network parameters of the current subpolicy
        #     agent.update_network_parameters()