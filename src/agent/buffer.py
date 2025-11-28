import torch
import numpy as np

class RolloutBuffer:
    # This class is used to store the data that is used to train the agent.
    def __init__(self, num_steps, state_dim, action_dim , device):
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

        # Pre-allocate tensors directly on device (faster than .to())
        # Buffers are used to store the data that is used to train the agent.
        self.states = torch.zeros((num_steps, state_dim), device=device)
        self.actions = torch.zeros((num_steps, action_dim), device=device)
        self.rewards = torch.zeros(num_steps, device=device) # log(pi_old(a|s))
        self.log_probs = torch.zeros(num_steps, device=device)
        self.values = torch.zeros(num_steps, device=device) # V_old(s)
        self.dones = torch.zeros(num_steps, device=device) # Terminal flags

        self.ptr = 0 # Current position in buffer

    def add(self, state, action, reward, log_prob, value, done):
        """Add one step of experience to the buffer."""
        
        if self.ptr >= self.num_steps:
            print("WARNING: buffer full, skipping")
            return # Buffer is full, so we don't add anything
        
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward # Direct assignment (faster)
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value.squeeze() # Squeeze value to be 1-dimensional
        self.dones[self.ptr] = float(done) # Direct float conversion
        self.ptr += 1 # Increment position

    def is_full(self):
        return self.ptr >= self.num_steps
        
    def clear(self):
        """Reset buffer for new rollout."""
        self.ptr = 0
    
    def get_batch(self):
        """Return all stored data as tuple.
        Used by GAE algorithm to compute advantages."""

        return (self.states,self.actions,self.rewards,self.log_probs,self.values,self.dones)

    def get_minibatch_generator(self, advantages, returns, minibatch_size):
        """Creates a generator that yields minibatches of experiences for
        the update phase. Reshuffles each time its called (for multiple epochs)."""

        total = self.num_steps
        indices= np.arange(total)
        np.random.shuffle(indices)

        for start in range(0, total, minibatch_size):
            end = start + minibatch_size
            
            # Skip incomplete final batch
            if end > total:
                continue
            
            batch_idx = indices[start:end] 
            
            yield( # Better than returning a tuple, because it is more efficient
                self.states[batch_idx],
                self.actions[batch_idx],
                self.log_probs[batch_idx],
                self.values[batch_idx],# Used for value function loss
                advantages[batch_idx],
                returns[batch_idx]
            )


