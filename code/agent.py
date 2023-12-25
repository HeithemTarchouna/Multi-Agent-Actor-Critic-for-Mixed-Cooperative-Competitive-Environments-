import numpy as np
import torch as T
import torch.nn.functional as F
from networks import ActorNetwork, CriticNetwork


class SubPolicy:
    def __init__(self, actor_dims, critic_dims, n_actions,agent_idx,agent_name,chkpt_dir, min_action,
                 max_action, alpha=1e-4, beta=1e-3, fc1=64,
                 fc2=64, gamma=0.95, tau=0.01,subPolicyIdx=0):
        self.gamma = gamma # discount factor (how much we care about future rewards)
        self.tau = tau # soft update parameter for the target network update , if 1 then the target network is the same as the main network (hard update)
        self.n_actions = n_actions
        self.agent_name = agent_name
        self.agent_idx = agent_idx
        self.min_action = min_action # the minimum action value for each agent (in this case it's -1)
        self.max_action = max_action # the maximum action value for each agent (in this case it's 1)
        self.subPolicyIdx =  subPolicyIdx

        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                  chkpt_dir=chkpt_dir,
                                  name=self.agent_name+'_actor')
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2,
                                         n_actions, chkpt_dir=chkpt_dir,
                                         name=self.agent_name+'target__actor')

        self.critic = CriticNetwork(beta, critic_dims, fc1, fc2,
                                    chkpt_dir=chkpt_dir,
                                    name=self.agent_name+'_critic')
        self.target_critic = CriticNetwork(beta, critic_dims, fc1, fc2,
                                           chkpt_dir=chkpt_dir,
                                           name=self.agent_name+'_target__critic')
        # During the creation, we set the target networks to be the same as the main networks.
        self.update_network_parameters(tau=1)

    def choose_action(self, observation, evaluate=False):
        state = T.tensor(observation[np.newaxis, :], dtype=T.float,
                         device=self.actor.device)
        actions = self.actor.forward(state)
        noise = T.randn(size=(self.n_actions,)).to(self.actor.device)

        if not evaluate:
            noise = 0
            
        action = T.clamp(actions + noise,
                         T.tensor(self.min_action, device=self.actor.device),
                         T.tensor(self.max_action, device=self.actor.device))
        return action.data.cpu().numpy()[0]

    def update_network_parameters(self, tau=None):
        tau = tau or self.tau

        src = self.actor
        dest = self.target_actor

        for param, target in zip(src.parameters(), dest.parameters()):
            target.data.copy_(tau * param.data + (1 - tau) * target.data)

        src = self.critic
        dest = self.target_critic

        for param, target in zip(src.parameters(), dest.parameters()):
            target.data.copy_(tau * param.data + (1 - tau) * target.data)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def learn(self, memory, agent_list):
        # If we don't have enough samples in the replay buffer, we don't learn
        if not memory[self.subPolicyIdx].ready():
            return

      # Extract experiences from the buffer
        sample = memory[self.subPolicyIdx].sample_buffer()
        actor_states, states, actions, rewards, actor_new_states, states_, dones = \
            sample['actor_states'], sample['states'], sample['actions'], sample['rewards'], \
            sample['actor_new_states'], sample['new_states'], sample['terminals']

        device = self.actor.device


        # PREPARE THE DATA
        #-----------------------------------------------------------------------------------------
        states = T.tensor(np.array(states), dtype=T.float, device=device) # global state (observations of all agents)
        rewards = T.tensor(np.array(rewards), dtype=T.float, device=device) # global reward
        states_ = T.tensor(np.array(states_), dtype=T.float, device=device) # global new state (observations of all agents after each agent's step)
        dones = T.tensor(np.array(dones), device=device) # global done

        # Convert the experiences of each agent to tensors
        actor_states = [T.tensor(actor_states[idx],
                                 device=device, dtype=T.float)
                        for idx in range(len(agent_list))]
        # Convert the experiences of each agent to tensors
        actor_new_states = [T.tensor(actor_new_states[idx],
                                     device=device, dtype=T.float)
                            for idx in range(len(agent_list))]
        # Convert the experiences of each agent to tensors
        actions = [T.tensor(actions[idx], device=device, dtype=T.float)
                   for idx in range(len(agent_list))]
        #-----------------------------------------------------------------------------------------

        # UPDATE THE CRITIC NETWORK
        #-----------------------------------------------------------------------------------------
        with T.no_grad():
            # Get the actions of all agents in the next state as predicted by the target actor network
            new_actions = T.cat([agent.get_current_subpolicy().target_actor(actor_new_states[idx])
                                 for idx, agent in enumerate(agent_list)],
                                dim=1)
            
            # Pass the next states and new actions to the target critic network to get Q-value predictions for the next states
            critic_value_ = self.target_critic.forward(
                                states_, new_actions).squeeze()
            
            # if the episode is done, the Q-value of the next state is 0 (since there is no next state)
            critic_value_[dones[:, self.agent_idx]] = 0.0 # ( selects the column of the dones matrix corresponding to the current agent) (row is the timestep, column is the agent)

            # Compute the target Q-values using the rewards and discounted Q-values from the target critic 
            target = rewards[:, self.agent_idx] + self.gamma * critic_value_



        # Concatenate the actions from all agents in the current state
        old_actions = T.cat([actions[idx] for idx in range(len(agent_list))],
                            dim=1)
        # Pass the current states and actions to the critic network to get Q-value predictions
        critic_value = self.critic.forward(states, old_actions).squeeze()
        # Calculate the mean squared error loss between the critic's predictions and the target Q-values
        critic_loss = F.mse_loss(target, critic_value)
        # Reset the gradients of the critic network
        self.critic.optimizer.zero_grad()
        # backpropagate the loss
        critic_loss.backward()
        # gradient clipping to avoid exploding gradient
        T.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        # update the weights of the critic network
        self.critic.optimizer.step()
        #-----------------------------------------------------------------------------------------

        # UPDATE THE ACTOR NETWORK : GOAL -> maximize the Q-values predicted by the critic network for the actions it suggests.
        #-----------------------------------------------------------------------------------------
        # Concatenate the actions from all agents in the current state
        actions[self.agent_idx] = self.actor.forward(
                actor_states[self.agent_idx])
        actions = T.cat([actions[i] for i in range(len(agent_list))], dim=1)
        # Calculate the loss for the actor network. It's the negative mean of the Q-values
        # predicted by the critic network for the current states and actions.
        actor_loss = -self.critic.forward(states, actions).mean()
        # Reset the gradients of the actor network
        self.actor.optimizer.zero_grad()
        # backpropagate the loss
        actor_loss.backward()
        # gradient clipping to avoid exploding gradient
        T.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        # update the weights of the actor network
        self.actor.optimizer.step()
    
        self.update_network_parameters()
    #-----------------------------------------------------------------------------------------



class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions,agent_idx,agent_name,chkpt_dir, min_action,
                 max_action, alpha=1e-4, beta=1e-3, fc1=64,
                 fc2=64, gamma=0.95, tau=0.01, num_subpolicies=3):
        self.agent_name = agent_name
        self.agent_idx = agent_idx
        self.n_actions = n_actions
        self.min_action = min_action
        self.max_action = max_action

        # subpolicies co
        self.num_subpolicies = num_subpolicies
        self.subPolicies = self.init_subpolicies(actor_dims, critic_dims, n_actions,agent_idx,agent_name,chkpt_dir, min_action,
                 max_action, alpha=alpha, beta=beta, fc1=fc1,
                 fc2=fc2, gamma=gamma, tau=tau, num_subpolicies=num_subpolicies)
        # randomly select a subpolicy from the subpolicy list (uniform distributin)
        self.current_subpolicy_idx = np.random.randint(0,num_subpolicies)


    def get_current_subpolicy(self):
        return self.subPolicies[self.current_subpolicy_idx]


    def init_subpolicies(self,actor_dims, critic_dims, n_actions,agent_idx,agent_name,chkpt_dir, min_action,
                 max_action, alpha=1e-4, beta=1e-3, fc1=64,
                 fc2=64, gamma=0.95, tau=0.01, num_subpolicies=3):
        subpolicies = []
        for subPolicyIdx in range(num_subpolicies):
            subpolicies.append(SubPolicy(actor_dims, critic_dims, n_actions,agent_idx,agent_name,chkpt_dir, min_action,
                 max_action, alpha,beta, fc1,
                 fc2, gamma, tau,subPolicyIdx))
        return subpolicies
    

    def random_select_subpolicy(self):
        # randomly select a subpolicy from the subpolicy list (uniform distributin)
        self.current_subpolicy_idx = np.random.randint(0,self.num_subpolicies)
    
        
    
    def choose_action(self, observation, evaluate=False,new_episode=False):
        # if it's a new episode, we randomly select a subpolicy from the subpolicy list (uniform distributin)
        if new_episode :
            self.random_select_subpolicy()
            #print(f"agent {self.agent_name} : now using subpolicy {self.current_subpolicy_idx}")
            



        return self.subPolicies[self.current_subpolicy_idx].choose_action(observation, evaluate)
    
    def learn(self, memory, agent_list):
        #self.subPolicies[self.current_subpolicy_idx].learn(memory[self.agent_name][self.current_subpolicy_idx], agent_list)
        self.get_current_subpolicy().learn(memory, agent_list)

    def save_models(self):
        self.get_current_subpolicy().save_models()
    
    def load_models(self):
        self.get_current_subpolicy().current_subPolicy.load_models()
    
    def update_network_parameters(self, tau=None):
       self.get_current_subpolicy().update_network_parameters(tau)
    