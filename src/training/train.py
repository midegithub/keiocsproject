import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

#Importing custom modules
from playground.main_playground import MainPlayground
from agent.ppo import PPOAgent
from agent.buffer import RolloutBuffer

#Configuration

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available, otherwise use CPU
print(f"Using device: {DEVICE}")

TOTAL_TIMESTEPS = 200_000 # 200K timesteps is the total number of timesteps the agent will train for.
ROLLOUT_STEPS = 2048 # Number of steps to collect data for each rollout a rollout is a sequence of actions taken by the agent.
MINIBATCH_SIZE = 64 # Size of the mini-batches for training a minibatch is a subset of the data used to train the agent.
NUM_EPOCHS = 10 # Number of epochs for training an epoch is the number of times the agent will re-study its recent experiences before gathering new ones.
SAVE_INTERVAL = 20 # Save model every N Rollouts

#Create output directory
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Hyperparameters chosen by research papers and other sources. To be tested
HYPERPARAMETERS = {
    'lr': 3e-4,
    'gamma': 0.99,
    'lambda_gae': 0.95,
    'clip_epsilon': 0.2,
    'v_coef': 0.5,
    'entropy_coef': 0.01,
    'num_epochs': NUM_EPOCHS,
    'minibatch_size': MINIBATCH_SIZE
    }


#Initialization
def main():
    #Set gui=False for maximum training speed
    env = MainPlayground(gui=False)
    state=env.reset()
    state_dim=env.state_dim
    action_dim=env.action_dim
    print(f'Environment initialized. State dim={state_dim}, Action dim={action_dim}')

    agent = PPOAgent(state_dim, action_dim, DEVICE, HYPERPARAMETERS) # Pass hyperparameters as a single dictionary
    buffer = RolloutBuffer(ROLLOUT_STEPS, state_dim, action_dim, DEVICE)

    #Logging
    all_ep_rewards=[]
    all_avg_rewards=[]
    current_ep_reward=0
    state_tensor=torch.tensor(state, dtype=torch.float32, device=DEVICE) 
    # The tensor is a multi-dimensional array of numbers used for calculations on GPU or CPU.
    num_timesteps=0
    rollout_count=0

    #mMain training loop
    while num_timesteps < TOTAL_TIMESTEPS:
        #Rollout phase

        buffer.clear()
        for _ in range(ROLLOUT_STEPS):
            num_timesteps+=1

            #Get action from agent
            with torch.no_grad(): #We are not calculating gradients for this part, no training here
                action_tensor, log_prob, value = agent.model.act(state_tensor.unsqueeze(0)) #Unsqueeze to add a batch dimension
            
            action = action_tensor.cpu().numpy().squeeze() # Squeeze to remove the batch dimension

            #Step the environment 
            next_state, reward, done, _ = env.step(action)

            #Update metrics
            current_ep_reward+=reward

            #Store data in buffer
            buffer.add(state_tensor, action_tensor, reward, log_prob, value, done)

            state=next_state
            state_tensor=torch.tensor(state, dtype=torch.float32, device=DEVICE)

            if done:
                # Log episode results
                print(f"Timestep {num_timesteps}: Episode reward={current_ep_reward:.2f}") #2 decimal
                all_ep_rewards.append(current_ep_reward)

                #Calculate and log moving average (for smooth plotting)

                avg_reward=np.mean(all_ep_rewards[-50:])
                all_avg_rewards.append(avg_reward)

                #Reset
                state=env.reset()
                state_tensor=torch.tensor(state, dtype=torch.float32, device=DEVICE)
                current_ep_reward=0

        #Update phase
        rollout_count+=1

        #Get the value of the last state
        with torch.no_grad():
            _, last_value = agent.model.forward(state_tensor.unsqueeze(0))

        last_done=done #THis is the 'done' flag from the last step

        # 1 - compute advantages and returns
        advantages, returns = agent.compute_advantages_and_returns(buffer, last_value, last_done)

        # 2 - update the policy
        agent.update(buffer, advantages, returns)

        # 3 - save the model
        if rollout_count % SAVE_INTERVAL == 0:
            save_path=f"models/ppo_spotmicro_{num_timesteps}.pth"
            torch.save(agent.model.state_dict(), save_path)
            print(f"Model saved at timestep {num_timesteps} to {save_path}")
    env.close()

    #Plotting to be implemented
    plot_rewards(all_ep_rewards, all_avg_rewards)

def plot_rewards(rewards, avg_rewards):
    """uses matplotlib to plot the raw and moving average rewards"""
    print("Plotting rewards...")
    plt.figure(figsize=(12,6))

    #Plot raw rewards with transparency
    plt.plot(rewards, label="Episode Rewards", alpha=0.3, color="blue")

    #Plot moving average rewards
    if len(avg_rewards) > 0:
        plt.plot(avg_rewards, label="Moving Average Rewards(50 episodes)", color="red", linewidth=2)

    
    plt.title("PPO Training Progress: Episode Rewards Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    
    #Save the plot
    plot_path="../../plots/training_rewards.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    #plt.show()

if __name__ == "__main__":
    main()
