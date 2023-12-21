import numpy as np
import torch as T
import torch.nn.functional as F
from networks import ActorNetwork, CriticNetwork

class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, 
                 chkpt_dir, min_action, max_action, alpha=1e-4, beta=1e-3, 
                 fc1=64, fc2=64, gamma=0.95, tau=0.01, K=4):
        self.gamma = gamma
        self.tau = tau
        self.K = K  # Number of sub-policies
        self.n_actions = n_actions
        agent_name = f'agent_{agent_idx}'
        self.agent_idx = agent_idx
        self.min_action = min_action
        self.max_action = max_action
    
        # Initialize multiple actor and critic networks
        self.actors = [ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                     chkpt_dir=chkpt_dir,
                                     name=f'{agent_name}_actor_{k}')
                       for k in range(self.K)]
        self.target_actors = [ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                           chkpt_dir=chkpt_dir,
                                           name=f'{agent_name}_target_actor_{k}')
                              for k in range(self.K)]

        # Single critic per agent as in standard MADDPG
        self.critic = CriticNetwork(beta, critic_dims, fc1, fc2,
                                    chkpt_dir=chkpt_dir,
                                    name=f'{agent_name}_critic')
        self.target_critic = CriticNetwork(beta, critic_dims, fc1, fc2,
                                           chkpt_dir=chkpt_dir,
                                           name=f'{agent_name}_target_critic')

        self.update_network_parameters(tau=1)

    # Choose action from a randomly selected sub-policy
    def choose_action(self, observation, evaluate=False):
        selected_policy = np.random.choice(self.actors)
        return self._choose_action_from_policy(selected_policy, observation, evaluate)

    def _choose_action_from_policy(self, policy, observation, evaluate):
        observation = np.array(observation)
        state = T.tensor(observation, dtype=T.float, device=policy.device).unsqueeze(0)
        actions = policy.forward(state)
        if not evaluate:
            noise = T.randn(size=(self.n_actions,), device=policy.device)
            actions += noise
        min_action = T.tensor(self.min_action, device=policy.device, dtype=T.float)
        max_action = T.tensor(self.max_action, device=policy.device, dtype=T.float)
        actions = T.clamp(actions, min_action, max_action)
        return actions.cpu().data.numpy().flatten()

    def learn(self, memory, agent_list, n_agents):
        if not memory.ready():
            return

        actor_states, states, actions, rewards, actor_new_states, states_, terminal = memory.sample_buffer()

        device = self.actors[0].device  # Assuming all actors are on the same device

        # Convert to tensors
        states = T.tensor(states, dtype=T.float, device=device)
        rewards = T.tensor(rewards, dtype=T.float, device=device)
        dones = T.tensor(terminal, dtype=T.float, device=device)
        actions = T.stack([T.tensor(actions[idx], device=device, dtype=T.float) for idx in range(n_agents)], dim=1)
        
        # Process actor_new_states for each agent separately
        actor_new_states_tensors = [T.tensor(actor_new_states[idx], device=device, dtype=T.float) for idx in range(n_agents)]

        # Loop through each actor network for ensemble policies
        for k, actor in enumerate(self.actors):
            with T.no_grad():
                # Concatenate new actions for all agents
                new_actions = []
                for idx, other_agent in enumerate(agent_list):
                    target_actor = other_agent.target_actors[k]
                    new_action = target_actor(actor_new_states_tensors[idx])
                    new_actions.append(new_action)
                new_actions = T.cat(new_actions, dim=1)

                critic_value_ = self.target_critic(states_, new_actions).squeeze()
                critic_value_[dones] = 0.0
                target = rewards + self.gamma * critic_value_

            # Update the critic network
            critic_value = self.critic(states, T.cat(actions, dim=1)).squeeze()
            critic_loss = F.mse_loss(target, critic_value)

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Update the actor network
            with T.no_grad():
                # Concatenate current actions for all agents
                current_actions = []
                for idx, other_agent in enumerate(agent_list):
                    current_action = other_agent.actors[k](actor_new_states_tensors[idx])
                    current_actions.append(current_action)
                current_actions = T.cat(current_actions, dim=1)

            actor_loss = -self.critic(states, current_actions).mean()
            actor.optimizer.zero_grad()
            actor_loss.backward()
            actor.optimizer.step()

            # Soft update the target networks
            self.soft_update(self.target_actors[k], actor, self.tau)

        self.soft_update(self.target_critic, self.critic, self.tau)

    def soft_update(self, target_network, source_network, tau):
        for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(tau*source_param.data + (1.0-tau)*target_param.data)
    
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # Update the target actor network
        for target_actor, actor in zip(self.target_actors, self.actors):
            with T.no_grad():
                for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                    target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        # Update the target critic network
        with T.no_grad():
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def save_models(self):
        for k, actor in enumerate(self.actors):
            actor.save_checkpoint(k)
        self.critic.save_checkpoint()

    def load_models(self):
        for k, actor in enumerate(self.actors):
            actor.load_checkpoint(k)
        self.critic.load_checkpoint()