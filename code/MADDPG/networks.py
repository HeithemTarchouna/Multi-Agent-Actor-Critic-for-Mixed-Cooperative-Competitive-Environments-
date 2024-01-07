import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Critic Network:
# used to approximate the Q-value function given the state of the environment 
# and the actions of all agents for each agent
class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, 
                    n_agents, n_actions, name, chkpt_dir):
        super(CriticNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def forward(self, state, action):
        x = F.relu(self.fc1(T.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q

    def save_checkpoint(self,type):
        checkpoint_temp = self.chkpt_file + type
        os.makedirs(os.path.dirname(checkpoint_temp), exist_ok=True)
        checkpoint_path = self.chkpt_file + ".pt"  # Ensure the file has an extension
        T.save(self.state_dict(), checkpoint_path)

    
    # def load_checkpoint(self,type):
    #     checkpoint_temp = self.chkpt_file + type
    #     checkpoint_path = checkpoint_temp + ".pt"
    #     return 
    #     self.load_state_dict(T.load(checkpoint_path))

# Actor Network:
# used to approximate the policy function for each agent
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, 
                 n_actions, name, chkpt_dir):
        super(ActorNetwork, self).__init__()
        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = T.softmax(self.pi(x), dim=1)

        return pi

    def save_checkpoint(self,type):
        checkpoint_temp = self.chkpt_file + type
        os.makedirs(os.path.dirname(checkpoint_temp), exist_ok=True)
        checkpoint_path = checkpoint_temp + ".pt"  # Ensure the file has an extension
        T.save(self.state_dict(), checkpoint_path)

    def load_checkpoint(self,type):
        checkpoint_temp = self.chkpt_file + type
        checkpoint_path = checkpoint_temp + ".pt"  # Ensure the file has an extension
        self.load_state_dict(T.load(checkpoint_path))

