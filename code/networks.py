import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, name, chkpt_dir):
        super(CriticNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)
        self.fc1 = nn.Linear(input_dims, fc1_dims)  # input_dims should be the combined size
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        # Ensure state and action are tensors
        state = self._convert_to_tensor(state)
        action = self._convert_to_tensor(action)

        # Concatenate state and action along the feature dimension (dim=1)
        combined = T.cat([state, action], dim=1)

        # Pass the combined state and action through the network layers
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q

    def _convert_to_tensor(self, numpy_array):
        if not isinstance(numpy_array, T.Tensor):
            numpy_array = T.tensor(numpy_array, dtype=T.float32, device=self.device)
        return numpy_array

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1, fc2, n_actions, name, chkpt_dir):
        super(ActorNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)
        self.fc1 = nn.Linear(input_dims, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.pi = nn.Linear(fc2, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = self._convert_to_tensor(state)

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = T.tanh(self.pi(x))

        return pi

    def _convert_to_tensor(self, numpy_array):
        if not isinstance(numpy_array, T.Tensor):
            numpy_array = T.tensor(numpy_array, dtype=T.float32, device=self.device)
        return numpy_array

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))
