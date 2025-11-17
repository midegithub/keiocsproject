import torch
import numpy as np

class RolloutBuffer:
    # This class is used to store the data that is used to train the agent.
    def ___init__(self, num_steps, state_dim, action_dim , device):
        """
        Initializes the on-policy rollout buffer.
        Arguments:
        num_steps (int): Max number of steps to store (rollout length).
        state_dim (int): Dimension of the state.
        action_dim (int): Dimension of the action.
        device (torch.device): CPU or CUDA device to store the data.
        """

        self.num_steps=num_steps
        self.device=device

        # Initialize buffers as tensors on the correct device
        # Buffers are used to store the data that is used to train the agent.
        self.states = torch.zeros((num_steps, state_dim)).to(device)
        self.actions = torch.zeros((num_steps, action_dim)).to(device)
        self.rewards = torch.zeros(num_steps).to(device)
        self.log_probs = torch.zeros(num_steps).to(device) # log(pi_old(a|s))
        self.values = torch.zeros(num_steps).to(device) # V_old(s)
        self.dones = torch.zeros(num_steps).to(device) # Terminal flags

        self.step = 0 # Current step index

    def add(self, state, action, reward, log_prob, value, done):
        """Add one step of experience to the buffer."""
        
        
        if self.step >= self.num_steps:
            print("Warning: Buffer is full, cannot add.")
            return # Buffer is full, so we don't add anything
        
        self.states[self.step] = state
        self.actions[self.step] = action
        self.rewards[self.step] = torch.tensor(reward, dtype=torch.float32).to(self.device) # Convert reward to tensor and move to device
        self.log_probs[self.step] = log_prob
        self.values[self.step] = value.squeeze() # Squeeze value to be 1-dimensional
        self.dones[self.step] = torch.tensor(done, dtype=torch.float32).to(self.device) # Convert done to tensor and move to device
        self.step += 1 # Increment step index

    def is_full(self):
        return self.step >= self.num_steps
        
    def clear(self):
        """Clear the buffer."""
        self.step = 0
        # Do we have to reset everything ?
    
    def get_batch(self):
        """Get a batch of experiences from the buffer."""
        # This will be used by the GAE algorithm
        # GAE is a technique to reduce the variance of the policy gradient estimator.

        if not self.is_full():
            print("Warning: Buffer is not full, cannot get batch.")

        return (self.states,self.actions,self.rewards,self.log_probs,self.values,self.dones)

    def get_minibatch_generator(self, advantages, returns, minibatch_size):
        """Creates a generator that yields minibatches of experiences for
        the update phasse, shuffles the data at the start of each epoch"""

        total_size = self.num_steps
        indices= np.arange(total_size)
        np.random.shuffle(indices)

        for start in range(0, total_size, minibatch_size):
            end = start+ minibatch_size
            if end > total_size:
                continue
        
        batch_indices = indices[start:end] 
        
        yield( # Better than returning a tuple, because it is more efficient
            self.states[batch_indices],
            self.actions[batch_indices],
            self.log_probs[batch_indices],
            self.values[batch_indices],# Used for value function loss
            advantages[batch_indices],
            returns[batch_indices]
        )


