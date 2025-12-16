import torch
import torch.nn as nn
import torch.optim as optim

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4):
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        # self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

    def update(self, action_probabilities, reward, optimizer):
        """
        Update the policy network using PPO loss.

        Args:
            state (torch.Tensor): The input state tensor of shape [batch_size, state_dim].
            action_probabilities (torch.Tensor): Probabilities of selecting actions from the policy network.
            reward (torch.Tensor): Reward tensor of shape [batch_size].
        """
        # Compute log probabilities of the selected actions
        log_prob = torch.log(action_probabilities)  # Use precomputed action probabilities
        loss = -(log_prob * reward).mean()  # Negative log probability weighted by reward

        # Optimize the policy network
        # optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def forward(self, state):
        """
        Forward pass through the policy network to compute action probabilities.
        
        Args:
            state (torch.Tensor): The input state tensor of shape [batch_size, state_dim].

        Returns:
            action_probabilities (torch.Tensor): Action probabilities of shape [batch_size, action_dim].
        """
        action_probabilities = self.policy_network(state)
        return action_probabilities