import torch
import torch.optim as optim
from.actor_critic import ActorCritic # Import our network
from.buffer import RolloutBuffer # Import our buffer


class PPOAgent: # PPOAgent is the class for the PPO agent, PPO means Proximal Policy Optimization
    def __init__(self, state_dim, action_dim, device, hyperparameters):
        self.device=device

        #Values found from research papers and other sources. To be tested
        # lr (learning_rate): How big of a step the agent takes when learning.
        self.lr = hyperparameters.get('lr', 3e-4)
        # gamma (discount_factor): How much the agent cares about future rewards (0=short-sighted, 1=far-sighted).
        self.gamma=hyperparameters.get('gamma',0.99)
        # lambda_gae: Helps the agent assign "credit" or "blame" to a sequence of past actions.
        self.lambda_gae = hyperparameters.get('lambda_gae', 0.95)
        # clip_epsilon: A "leash" that stops the agent from changing its strategy too quickly, keeping it stable.
        self.clip_epsilon = hyperparameters.get('clip_epsilon', 0.2)
        # v_coef: Balances two learning goals: "what action to take?" (Actor) vs. "how good is this situation?" (Critic).
        self.v_coef = hyperparameters.get('v_coef', 0.5)
        # entropy_coef: A bonus for "curiosity." Encourages the agent to keep exploring new actions.
        self.entropy_coef = hyperparameters.get('entropy_coef', 0.01)
        # num_epochs: The number of times the agent will re-study its recent experiences before gathering new ones.
        self.num_epochs = hyperparameters.get('num_epochs', 10)
        # minibatch_size: The size of the "chunks" the agent breaks its experiences into for studying.
        self.minibatch_size=hyperparameters.get('minibatch_size',64)
        # max_grad_norm: Maximum gradient norm for clipping (prevents exploding gradients)
        self.max_grad_norm = hyperparameters.get('max_grad_norm', 0.5)        


        
        # Initialize the actor-critic network
        self.model=ActorCritic(state_dim, action_dim).to(device)
        self.optimizer=optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-5) # Adam optimizer with eps for numerical stability
        # Adam optimizer is a variant of the gradient descent optimizer that is more efficient and stable.

    def compute_advantages_and_returns(self, buffer, last_value, last_done):
        """
        Implements Generalized Advantage Estimation (GAE).
        Iterates *backwards* through the buffer to compute advantages and returns.
        Args:
            buffer (RolloutBuffer): The buffer containing the rollout data.
            last_value (torch.Tensor): The value of the *last state* (S_T).
            last_done (bool): Whether the last state is terminal.
        """

        states, actions, rewards, log_probs, values, dones = buffer.get_batch()
        # Initialize advantages and returns
        advantages = torch.zeros_like(rewards).to(self.device)
        last_gae_lam=0 #Gae means Generalized Advantage Estimation        last_advantage = 0

        for t in reversed(range(buffer.num_steps)):
            if t== buffer.num_steps -1:
                # This is the last step T-1 of the rollout
                #next value is V(S_T)
                #next no terminal is (1.0-Done_T)
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                # This is a middle step T-1 of the rollout
                #next value is V(S_t+1)
                #next no terminal is (1.0-Done_t)
                next_non_terminal = 1.0 - dones[t+1]
                next_value = values[t+1]

            # This is the TD error (delta):
            # delta_t= R_t + gamma * V(S_{t+1}) - V(S_t)
            delta = rewards[t] + self.gamma*next_value*next_non_terminal-values[t]
            # This is the GAE formula: A_t = delta_t + (gamma*lambda)*A_{t+1}
            advantages[t] = last_gae_lam = delta + self.gamma *self.lambda_gae *next_non_terminal*last_gae_lam

        # Returns are the targets value for value function(Critic)
        returns = advantages + values

        return advantages, returns

    def update(self, buffer, advantages, returns):
        """
        The PPO Update Phase.
        Iterates over the data for num_epochs times, and updates the actor-critic network.
        """

        # Normalize advantages (helps with training stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.num_epochs): #Loop for the number of epochs
            # Get minibatch generator (reshuffles each epoch)
            data_generator = buffer.get_minibatch_generator(advantages, returns, self.minibatch_size)
            for batch in data_generator:
                (
                    batch_states,
                    batch_actions,
                    batch_old_log_probs,
                    batch_old_values, #Not used in loss, but good for value clipping
                    batch_advantages,
                    batch_returns
                ) = batch

                # Re evaktuate states and actions 
                #This is the new policy's view of the old data, we need new log_probs and values.

                new_log_probs, new_values, entropy = self.model.evaluate(batch_states, batch_actions)

                # Policy Loss (L-Clip)

                #The probability ratio : r_t(theta) = pi_new(a|s)/pi_old(a|s)
                #In log space: r_t=exp(log(pi_new)-log(pi_old))

                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                #The surrogate objective : surr1= ratio*A_t 
                #It measures how much better the new policy is compared to the old one.
                surr1=ratio*batch_advantages

                #The clipped surrogate objective : surr2= clip(ratio, 1-clip_epsilon, 1+clip_epsilon)*A_t
                #It ensures that the new policy is not too different from the old one.
                surr2=torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon)*batch_advantages

                #Policy loss is the minimum of the two surrogate objectives
                #We use -min because optimisers minimise. but we want to maximise
                policy_loss=-torch.min(surr1, surr2).mean()

                # LVF with value clipping (can help stability)
                value_clipped = batch_old_values + torch.clamp(
                    new_values - batch_old_values,
                    -self.clip_epsilon,
                    self.clip_epsilon
                )
                value_loss_unclipped = (new_values - batch_returns) ** 2
                value_loss_clipped = (value_clipped - batch_returns) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                # Entropy bonus

                entropy_loss=-entropy.mean() #We want to maximise the entropy

                # Total loss
                total_loss=(policy_loss+self.v_coef*value_loss+self.entropy_coef*entropy_loss)

                #Gradient descent step
                self.optimizer.zero_grad() #Clear the gradients
                total_loss.backward() #Backpropagate the loss
                #Gradient clipping (prevents exploding gradients for stable training)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step() #Update the parameters