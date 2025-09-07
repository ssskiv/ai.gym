

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch import from_numpy, no_grad, save, load, tensor, clamp
from torch import float as torch_float
from torch import long as torch_long
from torch import min as torch_min
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
from torch import manual_seed, exp, clamp
import torch
from collections import namedtuple

Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])


class LSTMAgent:
    """
    LSTMAgent implements the LSTM RL algorithm (https://arxiv.org/abs/1707.06347).
    It works with a set of discrete actions.
    It uses the Actor and Critic neural network classes defined below.
    """

    # In LSTM_agent.py

    # Note the new, clearer argument names
    def __init__(self, current_input_dim, history_feature_dim, action_dim, device, clip_param=0.2, max_grad_norm=0.5, LSTM_update_iters=10,
                 batch_size=32, gamma=0.99, use_cuda=False, actor_lr=0.0003, critic_lr=0.001, seed=None):
        super().__init__()
        if seed is not None:
            manual_seed(seed)
        
        self.device = device
        
        # Hyper-parameters
        self.clip_param = clip_param
        self.max_grad_norm = max_grad_norm
        self.LSTM_update_iters = LSTM_update_iters
        self.batch_size = batch_size
        self.gamma = gamma
        self.use_cuda = use_cuda

        # Correctly initialize Actor and Critic with all required arguments
        self.actor_net = Actor(
            current_input_dim=current_input_dim, 
            history_feature_dim=history_feature_dim, 
            num_outputs=action_dim
        )
        self.critic_net = Critic(
            current_input_dim=current_input_dim, 
            history_feature_dim=history_feature_dim, 
            action_dim=action_dim
        )
        
        self.actor_net.to(self.device)
        self.critic_net.to(self.device)
        
        if self.use_cuda:
            self.actor_net.cuda()
            self.critic_net.cuda()

        # Create the optimizers
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), actor_lr)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), critic_lr)

        # Training stats
        self.buffer = []

    # In LSTM_agent.py

    # In LSTM_agent.py -> LSTMAgent class

    def work(self, current_obs, history_obs):
        # Convert inputs to tensors and move them to the device
        current_tensor = from_numpy(np.array(current_obs)).float().unsqueeze(0).to(self.device)
        history_tensor = from_numpy(history_obs).float().unsqueeze(0).to(self.device)

        with no_grad():
            mu, log_std = self.actor_net(current_tensor, history_tensor)
        
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        action = dist.sample()
        action_log_prob = dist.log_prob(action).sum(dim=-1)

        return action.squeeze().cpu().numpy(), action_log_prob.item()
    
    # def work(self, current_obs, history_obs):
    #     current_tensor = from_numpy(np.array(current_obs)).float().unsqueeze(0)
    #     history_tensor = from_numpy(history_obs).float().unsqueeze(0)

    #     with no_grad():
    #         # Pass both tensors to the actor network
    #         mu, log_std = self.actor_net(current_tensor, history_tensor)

    #     # Create distribution, sample, and get log_prob
        

    def get_value(self, state):
        """
        Gets the value of the current state according to the critic model.

        :param state: The current state
        :return: state's value
        """
        state = from_numpy(state)
        with no_grad():
            value = self.critic_net(state)
        return value.item()

    def save(self, path):
        """
        Save actor and critic models in the path provided.

        :param path: path to save the models
        :type path: str
        """
        save(self.actor_net.state_dict(), path + '_actor.pkl')
        save(self.critic_net.state_dict(), path + '_critic.pkl')

    def load(self, path):
        """
        Load actor and critic models from the path provided.

        :param path: path where the models are saved
        :type path: str
        """
        actor_state_dict = load(path + '_actor.pkl')
        critic_state_dict = load(path + '_critic.pkl')
        self.actor_net.load_state_dict(actor_state_dict)
        self.critic_net.load_state_dict(critic_state_dict)

    def store_transition(self, transition):
        """
        Stores a transition in the buffer to be used later.

        :param transition: contains state, action, action_prob, reward, next_state
        :type transition: namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])
        """
        self.buffer.append(transition)

    def train_step(self, batch_size=None):
        """
        Performs a training step or update for the actor and critic models, based on transitions gathered in the
        buffer. It then resets the buffer.
        If provided with a batch_size, this is used instead of default self.batch_size

        :param: batch_size: int
        :return: None
        """
        # Default behaviour waits for buffer to collect at least one batch_size of transitions
        if batch_size is None:
            if len(self.buffer) < self.batch_size:
                return
            batch_size = self.batch_size

        # Unpack state tuples
        state_tuples = [t.state for t in self.buffer]
        current_states = [s[0] for s in state_tuples]
        history_states = [s[1] for s in state_tuples]

        # Create tensors from buffer data
        current_batch = tensor(current_states, dtype=torch_float)
        history_batch = tensor(history_states, dtype=torch_float)
        action_batch = tensor([t.action for t in self.buffer], dtype=torch_float)
        old_log_prob_batch = tensor([t.a_log_prob for t in self.buffer], dtype=torch_float).view(-1, 1)
        # ... (calculate Gt as before) ...
        Gt = tensor(Gt, dtype=torch_float)

        # Move all tensors to the device in one place
        current_batch = current_batch.to(self.device)
        history_batch = history_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        old_log_prob_batch = old_log_prob_batch.to(self.device)
        Gt = Gt.to(self.device)


        # Send everything to cuda if used
        if self.use_cuda:
            state, action, old_action_log_prob = state.cuda(), action.cuda(), old_action_log_prob.cuda()
            Gt = Gt.cuda()

        for _ in range(self.LSTM_update_iters):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), batch_size, False):
                # Slicing the tensors that are already on the GPU
                V = self.critic_net(current_batch[index], history_batch[index], action_batch[index])
                Gt_index = Gt[index].view(-1, 1)
                
                delta = Gt_index - V
                advantage = delta.detach()

                # Get the current probabilities
                # Apply past actions with .gather()
                action_prob = self.actor_net(state[index]).gather(1, action[index])  # new policy

                # LSTM
                ratio = (action_prob / old_action_log_prob[index])  # Ratio between current and old policy probabilities
                surr1 = ratio * advantage
                surr2 = clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch_min(surr1, surr2).mean()  # MAX->MIN descent
                self.actor_optimizer.zero_grad()  # Delete old gradients
                action_loss.backward()  # Perform backward step to compute new gradients
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)  # Clip gradients
                self.actor_optimizer.step()  # Perform training step based on gradients

                # update critic network
                value_loss = F.mse_loss(Gt_index, V)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()

        # After each training step, the buffer is cleared
        del self.buffer[:]


# In LSTM_agent.py
from torch.distributions import Normal # Import the Normal distribution

class Actor(nn.Module):
    # The __init__ method defines the layers with the CORRECT input dimensions
    def __init__(self, current_input_dim, history_feature_dim, num_outputs):
        super(Actor, self).__init__()
        # LSTM input_size is the number of features per time step (e.g., 6)
        self.lstm = nn.LSTM(input_size=history_feature_dim, hidden_size=128, batch_first=True)
        
        # MLP input_size is the dimension of the current sensor reading (e.g., 6)
        self.current_mlp = nn.Sequential(nn.Linear(current_input_dim, 32), nn.ReLU())
        
        # This part combines the outputs of the two branches
        combined_dim = 128 + 32  # LSTM hidden size + current MLP output size
        self.common_mlp = nn.Sequential(nn.Linear(combined_dim, 64), nn.ReLU())
        
        # Output heads for CONTINUOUS actions (mean and std deviation)
        self.mu_head = nn.Linear(64, num_outputs)
        self.log_std_head = nn.Linear(64, num_outputs)

    # The forward pass MUST take two separate arguments
    def forward(self, current_input, history_input):
        # 1. Process the 3D history tensor through the LSTM
        _, (h_n, _) = self.lstm(history_input)
        lstm_out = h_n.squeeze(0)
        
        # 2. Process the 2D current data tensor through the MLP
        current_out = self.current_mlp(current_input)
        
        # 3. Combine the results
        combined = torch.cat([lstm_out, current_out], dim=1)
        common_features = self.common_mlp(combined)
        
        # 4. Get the parameters for the continuous action distribution
        mu = self.mu_head(common_features)
        log_std = self.log_std_head(common_features)
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        return mu, log_std

class Critic(nn.Module):
    def __init__(self, current_input_dim, history_feature_dim, action_dim):
        super(Critic, self).__init__()
        self.lstm = nn.LSTM(input_size=history_feature_dim, hidden_size=128, batch_first=True)
        self.current_mlp = nn.Sequential(nn.Linear(current_input_dim, 32), nn.ReLU())

        combined_dim = 128 + 32 + action_dim # State features + action
        self.value_mlp = nn.Sequential(
            nn.Linear(combined_dim, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    # The Critic's forward pass also takes separate inputs
    def forward(self, current_input, history_input, action):
        _, (h_n, _) = self.lstm(history_input)
        lstm_out = h_n.squeeze(0)
        current_out = self.current_mlp(current_input)
        
        state_features = torch.cat([lstm_out, current_out], dim=1)
        combined_with_action = torch.cat([state_features, action], dim=1)
        
        q_value = self.value_mlp(combined_with_action)
        return q_value