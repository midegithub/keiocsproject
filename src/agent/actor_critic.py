import torch
from torch import nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        """Initializes the Actor-Critic model
        Arguments:
            state_dim (int): Dimension of the state space (here 35 observations)
            action_dim (int): Dimension of the action space (here 12 torques)
            hidden_size (int): Dimension of the hidden layers of neurons"""
        
        super(ActorCritic, self).__init__()
        

        # This part of the network leanrs the shared representation of the state
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )

    def forward(self, state):
        """Performs a forward pass through the Actor-Critic network.
        Arguments:
            state (torch.Tensor): The state of the environment
        Returns:
            the probability distribution of the actions (Actor)
            the estimated value of the state (Critic)"""
        
        x=self.shared_layers(state)
