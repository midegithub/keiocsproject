import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from datetime import datetime

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

TOTAL_TIMESTEPS = 300_000 # Tripled training time for better walking behavior learning
ROLLOUT_STEPS = 4096 # Larger rollouts = better GPU utilization and faster training
MINIBATCH_SIZE = 128 # Larger batches = more efficient GPU usage
NUM_EPOCHS = 6 # Reduced epochs to speed up training loop
SAVE_INTERVAL = 4 # Save more frequently to track best model

# Visualization settings
VISUALIZE_AT_CHECKPOINTS = True  # Set to True to see robot at each checkpoint (pauses training)
VISUALIZE_EPISODES = 1  # Number of episodes to visualize at each checkpoint

#Create output directory
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Optimized hyperparameters for ground-up learning
HYPERPARAMETERS = {
    'lr': 3e-4, # Standard PPO learning rate
    'gamma': 0.99, # Standard discount factor
    'lambda_gae': 0.95, # GAE parameter
    'clip_epsilon': 0.2, # PPO clipping
    'v_coef': 0.5, # Balanced value function weight
    'entropy_coef': 0.01, # Moderate exploration
    'num_epochs': NUM_EPOCHS,
    'minibatch_size': MINIBATCH_SIZE
    }


def visualize_checkpoint(agent, num_episodes=1):
    """
    Simple visualization of robot's current performance with GUI.
    Camera follows the robot for better viewing.
    """
    print(f"\n{'='*50}")
    print(f"VISUALIZING - Watch the robot!")
    print(f"{'='*50}")
    
    import time
    import pybullet as p
    
    viz_env = None
    try:
        # Create temporary environment with GUI
        viz_env = MainPlayground(gui=True, sim_steps_per_action=1)
        
        for ep in range(num_episodes):
            state = viz_env.reset()
            episode_reward = 0
            done = False
            steps = 0
            max_steps = 10000  # Longer visualization to see walking
            
            print(f"Episode {ep+1} - Running for up to {max_steps} steps...")
            print("Camera will follow the robot...")
            
            while not done and steps < max_steps:
                # Check if GUI is still connected
                if not p.isConnected(viz_env.client_id):
                    print("GUI window closed by user")
                    break
                    
                state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE)
                
                with torch.no_grad():
                    action_tensor, _, _ = agent.model.act(state_tensor.unsqueeze(0))
                    action = action_tensor.cpu().numpy().squeeze()
                
                state, reward, done, _ = viz_env.step(action)
                episode_reward += reward
                steps += 1
                
                # Update camera to follow robot every 10 steps
                if steps % 10 == 0:
                    base_pos, _ = p.getBasePositionAndOrientation(viz_env.robot_id, physicsClientId=viz_env.client_id)
                    # Camera follows robot: distance=2.5m, yaw=30deg, pitch=-20deg
                    p.resetDebugVisualizerCamera(
                        cameraDistance=2.5,
                        cameraYaw=30,
                        cameraPitch=-20,
                        cameraTargetPosition=[base_pos[0], base_pos[1], 0.3],
                        physicsClientId=viz_env.client_id
                    )
                
                # Slower visualization - 30 FPS instead of 240 FPS
                time.sleep(1./30)
            
            print(f"Episode {ep+1} done: {steps} steps, reward: {episode_reward:.2f}")
            print(f"Distance traveled: {base_pos[0]:.2f}m")
        
        print(f"Visualization complete!\n")
        
    except Exception as e:
        print(f"Visualization error: {e}")
    finally:
        # Ensure proper cleanup
        if viz_env is not None:
            try:
                if p.isConnected(viz_env.client_id):
                    viz_env.close()
            except:
                pass


#Initialization
def main():
    #Set gui=False for maximum training speed, use multiple sim steps for faster training
    env = MainPlayground(gui=False, sim_steps_per_action=4)
    state=env.reset()
    state_dim=env.state_dim
    action_dim=env.action_dim
    print(f'Environment initialized. State dim={state_dim}, Action dim={action_dim}')
    print(f'Training optimizations: 4 physics steps per action for {4}x speed boost')

    agent = PPOAgent(state_dim, action_dim, DEVICE, HYPERPARAMETERS) # Pass hyperparameters as a single dictionary
    
    # PyTorch 2.0+ compilation for speed boost (disabled on Windows due to C++ compiler issues)
    # If you have Visual Studio C++ Build Tools installed, you can enable this
    USE_TORCH_COMPILE = False  # Set to True if you have VS C++ compiler
    
    if USE_TORCH_COMPILE:
        try:
            if hasattr(torch, 'compile'):
                agent.model = torch.compile(agent.model, mode='reduce-overhead')
                print("Model compiled with torch.compile for faster training")
        except Exception as e:
            print(f"torch.compile failed ({type(e).__name__}), using standard mode")
    else:
        print("Using standard mode (torch.compile disabled)")
    
    # Try loading existing model to continue training
    latest_model = None
    import glob
    model_files = glob.glob("models/ppo_spotmicro_*.pth")
    if model_files:
        # Filter out files with non-numeric suffixes (like BEST)
        numeric_models = []
        for f in model_files:
            try:
                # Extract the part after last underscore and before .pth
                suffix = f.split('_')[-1].split('.')[0]
                timestep = int(suffix)
                numeric_models.append((f, timestep))
            except ValueError:
                # Skip files like ppo_spotmicro_BEST.pth
                continue
        
        if numeric_models:
            # Get the one with highest timestep number
            latest_model = max(numeric_models, key=lambda x: x[1])[0]
            try:
                agent.model.load_state_dict(torch.load(latest_model, map_location=DEVICE))
                print(f"Loaded existing model from {latest_model}")
            except Exception as e:
                print(f"Couldn't load {latest_model}: {e}")
                print("Starting fresh training instead")
                latest_model = None
        else:
            print("No timestep-numbered models found, starting from scratch")
    
    if not latest_model:
        print("Starting fresh training from scratch")
    
    buffer = RolloutBuffer(ROLLOUT_STEPS, state_dim, action_dim, DEVICE)

    #Logging
    all_ep_rewards=[]
    all_avg_rewards=[]
    current_ep_reward=0
    state_tensor=torch.tensor(state, dtype=torch.float32, device=DEVICE) 
    # The tensor is a multi-dimensional array of numbers used for calculations on GPU or CPU.
    num_timesteps=0
    rollout_count=0
    
    # Track best model performance
    best_avg_reward = float('-inf')
    best_model_path = None

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
                all_ep_rewards.append(current_ep_reward)

                #Calculate and log moving average (for smooth plotting)
                avg_reward=np.mean(all_ep_rewards[-50:])
                all_avg_rewards.append(avg_reward)
                
                #print every 10 episodes so we dont spam console
                if len(all_ep_rewards) % 10 == 0:
                    progress = (num_timesteps / TOTAL_TIMESTEPS) * 100
                    print(f"[{progress:5.1f}%] Episode {len(all_ep_rewards):4d} | This episode: {current_ep_reward:6.2f} | Avg(last 50): {avg_reward:6.2f} | Step: {num_timesteps:,}")

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
            
            # Calculate current performance
            current_avg = np.mean(all_ep_rewards[-50:]) if len(all_ep_rewards) >= 50 else np.mean(all_ep_rewards) if all_ep_rewards else float('-inf')
            
            # Save performance metrics
            import pickle
            data_path = f"data/training_data_{num_timesteps}.pkl"
            with open(data_path, 'wb') as f:
                pickle.dump({
                    'rewards': all_ep_rewards, 
                    'avg_rewards': all_avg_rewards,
                    'avg_reward_50': current_avg,
                    'timesteps': num_timesteps
                }, f)
            
            # Track best model
            if current_avg > best_avg_reward:
                improvement = current_avg - best_avg_reward if best_avg_reward != float('-inf') else 0
                best_avg_reward = current_avg
                best_model_path = save_path
                # Save a copy as "best" model with timestamp
                timestamp = datetime.now().strftime("%d%m%H%M")  # day month hour minute
                best_save_path = f"models/ppo_spotmicro_BEST_{timestamp}.pth"
                torch.save(agent.model.state_dict(), best_save_path)
                # Also save without timestamp for easy reference
                torch.save(agent.model.state_dict(), "models/ppo_spotmicro_BEST.pth")
                print(f"[CHECKPOINT] Step {num_timesteps:,} | NEW BEST! Avg(50eps): {current_avg:.2f} (+{improvement:.2f}) | Saved: {best_save_path}")
            else:
                print(f"[CHECKPOINT] Step {num_timesteps:,} | Avg(50eps): {current_avg:.2f} | Best: {best_avg_reward:.2f} | Saved: {save_path}")
            
            # Visualize checkpoint if enabled
            if VISUALIZE_AT_CHECKPOINTS:
                visualize_checkpoint(agent, num_episodes=VISUALIZE_EPISODES)
        
        #print progress every rollout
        if len(all_ep_rewards)>0:
            recent_avg = np.mean(all_ep_rewards[-20:]) if len(all_ep_rewards)>=20 else np.mean(all_ep_rewards)
            total_rollouts = TOTAL_TIMESTEPS // ROLLOUT_STEPS
            print(f"[ROLLOUT {rollout_count}/{total_rollouts}] Policy updated | Avg(last 20eps): {recent_avg:.2f} | Total episodes: {len(all_ep_rewards)}")
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
    plot_path="plots/training_rewards.png"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    #plt.show()

if __name__ == "__main__":
    main()
