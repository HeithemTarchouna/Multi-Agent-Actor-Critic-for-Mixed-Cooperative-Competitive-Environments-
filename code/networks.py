import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CriticNetwork(nn.Module):
    # Critic Network takes the combined state and action information of all agents as input, not just for a single agent.
    # Returns the expected (Q-value) of taking certain actions in certain states,
    # considering both the current policy of its agent (Actor) and the policies of other agents.
    def __init__(self, beta, input_dims, fc1, fc2,
                 name, chkpt_dir):
        super(CriticNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)
        self.fc1 = nn.Linear(input_dims, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.q = nn.Linear(fc2, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        x = F.relu(self.fc1(T.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q


class ActorNetwork(nn.Module):
    # ActorNetwork in MADDPG represents the policy function for an agent.
    # It takes the agent's current state as input and outputs the action values.
    def __init__(self, alpha, input_dims, fc1, fc2,
                 n_actions, name, chkpt_dir):
        super(ActorNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.pi = nn.Linear(fc2, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        pi = self.pi(x)
        pi = T.tanh(pi) # the action values are between -1 and 1
        return pi


