import torch as T
import torch.nn.functional as F
from agent import Agent
from pettingzoo.mpe import simple_adversary_v3,simple_speaker_listener_v4



class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, env,
                 scenario='simple',  alpha=0.01, beta=0.01, fc1=64, 
                 fc2=64, gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg/'):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.env = env

        #print(n_actions)
        chkpt_dir += scenario
        self.agents = {}
        for agent_idx, agent_name in enumerate(self.env.possible_agents):
            self.agents[agent_name] = Agent(actor_dims[agent_name],
                                            critic_dims,
                                            n_actions[agent_name], n_agents,
                                            agent_name = agent_name,
                                            alpha=alpha,
                                            beta=beta,
                                            chkpt_dir=chkpt_dir)


    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent_name, agent in self.agents.items():
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent_name, agent in self.agents.items():
            agent.load_models()

    # def choose_action(self, raw_obs):
    #     actions = []
    #     for agent_idx, agent in enumerate(self.agents):
    #         action = agent.choose_action(raw_obs[agent.agent_name])
    #         actions.append(action)
    #     return actions

    # def choose_action(self, raw_obs):
    #     actions = {agent.agent_name: agent.choose_action(raw_obs[agent.agent_name]) for agent in self.agents.values()}
    #     return actions
            

    def choose_action(self, raw_obs):
        actions = {}
        for agent_name, agent in self.agents.items():
            actions[agent_name] = agent.choose_action(raw_obs[agent_name])
        return actions

    def learn(self, memory):
        if not memory.ready():
            return

        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()

        device = self.agents[list(self.agents.keys())[0]].actor.device

        states = T.tensor(states, dtype=T.float).to(device)
        #print(self.agents)
        actions = [T.tensor(actions[agent_idx], device=device, dtype=T.float)
                   for agent_idx in range(len(list(self.agents.keys())))]
        
        rewards = T.tensor(rewards).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []
        #print("-"*10)
        #print(self.agents)
        #print("-"*10)

        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(actor_new_states[agent_idx], 
                                 dtype=T.float).to(device)
            new_pi = self.agents[agent].target_actor.forward(new_states)

            all_agents_new_actions.append(new_pi)
            mu_states = T.tensor(actor_states[agent_idx], 
                                 dtype=T.float).to(device)
            pi = self.agents[agent].actor.forward(mu_states)

            all_agents_new_mu_actions.append(pi)

            old_agents_actions.append(actions[agent_idx])




        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions],dim=1)




        for agent_idx,(agent_name, agent) in enumerate(self.agents.items()):
            #print(agent_name)
            #print(states_.shape)
            #print(new_actions.shape)
            critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()

            critic_value_[dones[:,0]] = 0.0
            critic_value = agent.critic.forward(states, old_actions).flatten()

            target = rewards[:,agent_idx].float() + agent.gamma*critic_value_


            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True, inputs=list(agent.critic.parameters()))
            agent.critic.optimizer.step()

            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = -T.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True, inputs=list(agent.actor.parameters()))
            agent.actor.optimizer.step()

            agent.update_network_parameters()

