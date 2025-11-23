import torch
from torch import nn
from torch.distributions import Normal
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=384):
        """Initializes the Actor-Critic model
        Arguments:
            state_dim (int): Dimension of the state space (here 35 observations)
            action_dim (int): Dimension of the action space (here 12 torques)
            hidden_size (int): Dimension of the hidden layers of neurons"""
        
        super(ActorCritic, self).__init__()
        

        # bigger network learns better policies
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.Tanh()
        )

        #separate heads for actor and critic so they dont interfere with each other
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_size//2, hidden_size//2),
            nn.Tanh()
        )
        
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_size//2, hidden_size//2),
            nn.Tanh()
        )

        #output layers
        self.actor_mean= nn.Linear(hidden_size//2, action_dim)
        self.log_std= nn.Parameter(torch.zeros(action_dim)) #Standard log-deviation of the action distribution

        self.critic= nn.Linear(hidden_size//2, 1)
        
        #better weight initialization helps training
        self._init_weights()

    def _init_weights(self):
        """orthogonal initialization works better for RL"""
        # It initializes the weights of the linear layers to be orthogonal, which helps the training of the network by preventing the network from exploding or vanishing gradients.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        #last layer smaller init
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, state):
        """Performs a forward pass through the Actor-Critic network.
        Arguments:
            state (torch.Tensor): The state of the environment
        Returns:
            the probability distribution of the actions (Actor)
            the estimated value of the state (Critic)"""
        
        shared_features=self.shared_layers(state)

        # Actor output
        actor_features = self.actor_head(shared_features)
        action_mean=self.actor_mean(actor_features)
        action_mean = torch.tanh(action_mean) #bound actions

        # Numerical stability: Convert log_std to std and clamp it
        #Clamping is a technique to prevent the standard deviation from becoming too large or too small.
        std = torch.exp(self.log_std).clamp(0.1, 1.0)

        # Create the action distribution
        dist = Normal(action_mean, std)

        # Critic output
        critic_features = self.critic_head(shared_features)
        value = self.critic(critic_features)

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

        return action, log_prob, value.squeeze() # Squeeze value to be 1-dimensional

    def evaluate(self, state, action):
        """
        Get values for a given state and action. for update phase.
        This is the old policy's view of the data, we need new log_probs and values."""

        dist, value = self.forward(state)

        #Get the logproba of the action
        log_prob = dist.log_prob(action).sum(dim=-1)

        #Get the entropy of the distribution
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, value.squeeze(), entropy #Squeeze value to be 1-dimensional
