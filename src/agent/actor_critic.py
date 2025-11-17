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

        #This part of the network learns the mean of the action distribution
        self.actor_mean= nn.Linear(hidden_size, action_dim)
        self.log_std= nn.Parameter(torch.zeros(action_dim)) #Standard log-deviation of the action distribution

        #This part of the network learns the value of the state
        self.critic= nn.Linear(hidden_size, 1)

    def forward(self, state):
        """Performs a forward pass through the Actor-Critic network.
        Arguments:
            state (torch.Tensor): The state of the environment
        Returns:
            the probability distribution of the actions (Actor)
            the estimated value of the state (Critic)"""
        
        x=self.shared_layers(state)

        # Actor output
        action_mean=self.actor_mean(x)

        # Numerical stability: Convert log_std to std
        std = torch.exp(self.log_std)

        # Create the action distribution
        dist = Normal(action_mean, std)

        # Critic output
        value = self.critic(x)

        return dist, value # Action distribution and value of the state

        # Helper method

        def act(self, state):
            """
            Get an action from the policy (for rollout pahse).
            Includes gradients for log_prob and value, but no gradients for the state."""

            dist, value = self.forward(state)

            action = dist.sample() # Sample an action from the distribution

            #Calculate its log-proba

            log_prob = dist.log_prob(action).sum(dim=-1)

            return action, log_prob, value.squeeze()

        def evaluate(self, state, action):
            """
            Get values for a given state and action. for update phase"""

            dist, value = self.forward(state)

            #Get the logproba of the action

            log_prob = dist.log_prob(action).sum(dim=-1)

            #Get the entropy of the distribution

            entropy = dist.entropy().sum(dim=-1)

            return log_prob, value.squeeze(), entropy #Squeeze value to be 1-dimensional
