import torch as T
from networks import ActorNetwork, CriticNetwork
import numpy as np
import torch.nn.functional as F



class SubPolicy:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents,  chkpt_dir,agent_name,
                    alpha=0.01, beta=0.01, fc1=32,
                    fc2=32, gamma=0.95, tau=0.01 ,
                    ):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = agent_name
        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                  chkpt_dir=chkpt_dir,  name=self.agent_name+'_actor')
        self.critic = CriticNetwork(beta, critic_dims,
                            fc1, fc2, n_agents, n_actions,
                            chkpt_dir=chkpt_dir, name=self.agent_name+'_critic')
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                        chkpt_dir=chkpt_dir,
                                        name=self.agent_name+'_target_actor')
        self.target_critic = CriticNetwork(beta, critic_dims,
                                            fc1, fc2, n_agents, n_actions,
                                            chkpt_dir=chkpt_dir,
                                            name=self.agent_name+'_target_critic')
        

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        # Ensure observation is a numpy array with standard numeric dtype
        if isinstance(observation, list):
            observations = np.array(observation)
        else:
            observations = observation

        if observations.ndim == 1:
            observations = observations.reshape(1, -1)

        state = T.tensor(observations, dtype=T.float).to(self.actor.device)
        actions = self.actor.forward(state)
        min_v = (1 - actions).min()
        noise = (T.rand(self.n_actions) * min_v).to(self.actor.device)

        action = actions + noise
        return action.detach().cpu().numpy()[0]

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                    (1-tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self,type):
        self.actor.save_checkpoint(type)
        self.target_actor.save_checkpoint(type)
        # for the purpose of the experiment, we don't need to save critic: 
        # for testing, we can just use the actor
        # self.critic.save_checkpoint(type) 
        # self.target_critic.save_checkpoint(type)

    def load_models(self,type):
        self.actor.load_checkpoint(type)
        self.target_actor.load_checkpoint(type)
        # for the purpose of the experiment, we don't need to load critic: 
        # for testing, we can just use the actor
        #self.critic.load_checkpoint(type)
        #self.target_critic.load_checkpoint(type)


class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents,  chkpt_dir,agent_name,
                    alpha=0.01, beta=0.01, fc1=32,
                    fc2=32, gamma=0.95, tau=0.01 ,k=1,agent_type="None"
                    ):
        self.agent_type = agent_type
        self.k = k
        chkpt_dir += agent_name
        # each agent has k subpolicies
        self.subpolicy_list = [SubPolicy(actor_dims, critic_dims, n_actions, n_agents,  f"{chkpt_dir}+{subpolicy}",agent_name,alpha=alpha, beta=beta, fc1=fc1,fc2=fc2, gamma=gamma, tau=tau ) for subpolicy in range(k)]
        self.agent_name = agent_name
        self.current_subpolicy_idx = np.random.randint(0,k)
        self.current_subpolicy = self.get_current_subpolicy()

    def get_current_subpolicy(self):
        return self.subpolicy_list[self.current_subpolicy_idx]
    


    def choose_action(self, observation):
        # Choose an action from the current subpolicy
        return self.current_subpolicy.choose_action(observation)

    def update_network_parameters(self, tau=None):
        # Update the network parameters of the current subpolicy
        self.current_subpolicy.update_network_parameters(tau)

    def learn(self, memory,agents):
        if not memory.ready():
            return
        # Sample a batch of experiences from the memory buffer
        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()

        device = agents[list(agents.keys())[0]].get_current_subpolicy().actor.device

        # Convert all the sampled data to tensors and move them to the correct device
        states = T.tensor(states, dtype=T.float).to(device)
        actions = [T.tensor(actions[agent_idx], device=device, dtype=T.float)
                   for agent_idx in range(len(list(agents.keys())))]
        rewards = T.tensor(rewards).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        # Prepare the new (next state) actions for each agent
        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []
 
        for agent_idx,(agent_name, agent) in enumerate(agents.items()):
            # Convert new state observations to tensor and move to device
            new_states = T.tensor(actor_new_states[agent_idx],
                                 dtype=T.float).to(device)
            
            # Get the action predictions from the target actor of the current subpolicy
            new_pi = agent.get_current_subpolicy().target_actor.forward(new_states)
            all_agents_new_actions.append(new_pi)
 
            # Get the current state observations as tensor
            mu_states = T.tensor(actor_states[agent_idx],
                                 dtype=T.float).to(device)
            
            # Get the action predictions from the current actor of the current subpolicy
            pi = agent.get_current_subpolicy().actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi)

            # Get the old actions
            old_agents_actions.append(actions[agent_idx])
 
 
 
        # Concatenate the actions from all agents
        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions],dim=1)
 
 
        for agent_idx,(agent_name, agent) in enumerate(agents.items()):
            # Forward pass in the target critic network
            critic_value_ = agent.get_current_subpolicy().target_critic.forward(states_, new_actions).flatten()

            # Zero out the critic values for the done states
            critic_value_[dones[:,0]] = 0.0

            # Forward pass in the current critic network
            critic_value = agent.get_current_subpolicy().critic.forward(states, old_actions).flatten()

            # Calculate the target for the critic network 
            target = rewards[:,agent_idx].float() + agent.get_current_subpolicy().gamma*critic_value_
 
 
            # Calculate the critic loss and backpropagate
            critic_loss = F.mse_loss(target, critic_value)
            agent.get_current_subpolicy().critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True, inputs=list(agent.get_current_subpolicy().critic.parameters()))
            agent.get_current_subpolicy().critic.optimizer.step()

            # Calculate the actor loss and backpropagate 
            actor_loss = agent.get_current_subpolicy().critic.forward(states, mu).flatten()
            actor_loss = -T.mean(actor_loss)
            agent.get_current_subpolicy().actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True, inputs=list(agent.get_current_subpolicy().actor.parameters()))
            agent.get_current_subpolicy().actor.optimizer.step()

            # Update the network parameters of the current subpolicy
            agent.update_network_parameters()



    def update_subpolicy(self, subpolicy_index):
            # Update the current subpolicy only if it's different from the current one
            if self.current_subpolicy_idx != subpolicy_index:
                self.current_subpolicy_idx = subpolicy_index
                self.current_subpolicy = self.get_current_subpolicy()


    # saving and loading models
    def save_models(self,type):
        # save all subpolicies
        for subpolicy in self.subpolicy_list:
            subpolicy.save_models(type)

    def load_models(self,type):
        # load all subpolicies
        for subpolicy in self.subpolicy_list:
            subpolicy.load_models(type)
