import torch
import torch.optim as optim
from.actor_critic import ActorCritic # Import our network
from.buffer import RolloutBuffer # Import our buffer


class PPOAgent: # PPOAgent is the class for the PPO agent, PPO means Proximal Policy Optimization
    def __init__(self, state_dim, action_dim, device, hyperparameters):
        self.device=device

        # Store hyperparameters
        self.lr = hyperparameters.get('lr', 3e-4)
        self.gamma=hyperparameters.get('gamma', 0.99)
        self.lambda_gae = hyperparameters.get('lambda_gae',0.95)
        self.clip_epsilon = hyperparameters.get('clip_epsilon',0.2)
        self.v_coef = hyperparameters.get('v_coef', 0.5)
        self.entropy_coef = hyperparameters.get('entropy_coef',0.01)
        self.num_epochs =hyperparameters.get('num_epochs',10)
        self.minibatch_size = hyperparameters.get('minibatch_size',64)

        # Initialize the actor-critic network
        self.model=ActorCritic(state_dim, action_dim).to(device)
        self.optimizer=optim.Adam(self.model.parameters(), lr=self.lr) # Adam optimizer with the learning rate
        # Adam optimizer is a variant of the gradient descent optimizer that is more efficient and stable.

