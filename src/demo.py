import torch
import time
import os
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Importing custom modules
from playground.main_playground import MainPlayground
from agent.actor_critic import ActorCritic # Import the model class

#Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available, otherwise use CPU
print(f"Using device: {DEVICE}")

def find_best_model():
    """Automatically finds the best trained model to demonstrate"""
    import glob
    import pickle
    
    # First, check if we have a BEST model saved
    best_model = "models/ppo_spotmicro_BEST.pth"
    if os.path.exists(best_model):
        print(f"Found BEST model: {best_model}")
        return best_model
    
    # Otherwise, find model with highest average reward from training data
    data_files = glob.glob("data/training_data_*.pkl")
    if not data_files:
        # Fall back to latest model by timestep
        model_files = glob.glob("models/ppo_spotmicro_*.pth")
        if model_files:
            latest = max(model_files, key=lambda x: int(x.split('_')[-1].split('.')[0]) if x.split('_')[-1].split('.')[0].isdigit() else 0)
            print(f"Using latest model: {latest}")
            return latest
        return None
    
    # Load all training data and find best performing model
    best_reward = float('-inf')
    best_timestep = None
    
    for data_file in data_files:
        try:
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
                avg_reward = data.get('avg_reward_50', float('-inf'))
                timestep = data.get('timesteps', 0)
                
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    best_timestep = timestep
        except:
            continue
    
    if best_timestep:
        model_path = f"models/ppo_spotmicro_{best_timestep}.pth"
        if os.path.exists(model_path):
            print(f"Found best performing model: {model_path} (avg reward: {best_reward:.2f})")
            return model_path
    
    # Last resort - use latest model
    model_files = glob.glob("models/ppo_spotmicro_*.pth")
    if model_files:
        latest = max(model_files, key=lambda x: int(x.split('_')[-1].split('.')[0]) if x.split('_')[-1].split('.')[0].isdigit() else 0)
        print(f"Using latest model: {latest}")
        return latest
    
    return None

MODEL_PATH = find_best_model()  # Auto-detect best model

def main():
    """
    Demo script to load and run a trained PPO agent.
    Uses the mean of the policy distribution for deterministic, stable locomotion"""
    
    # We want to see this, so gui=True. Use 1 sim step for smooth visualization
    env = MainPlayground(gui=True, sim_steps_per_action=1)
    
    # Get state and action dimensions
    state = env.reset()
    state_dim = env.state_dim
    action_dim = env.action_dim
    print(f"Loading trained model for demo. State dim={state_dim}, Action dim={action_dim}")
    
    # --- Load the TRAINED model ---
    model = ActorCritic(state_dim, action_dim).to(DEVICE)
    
    if MODEL_PATH is None or not os.path.exists(MODEL_PATH):
        print(f"Error: No trained model found!")
        print("Please train a model first using: python src/training/train.py")
        env.close()
        return
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() # Set to evaluation mode (e.g., disables dropout if any)
    print(f"Model loaded successfully from {MODEL_PATH}")
    print("Running demo... Press Ctrl+C to exit.")
    
    state = env.reset()
    episode_count = 0
    episode_reward = 0
    step_count = 0
    
    print("\nCamera will follow the robot. Press Ctrl+C to stop.")
    
    try:
        import pybullet as p
        while True:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE)
            
            with torch.no_grad():
                # For demonstration, we take the *mean* action
                # for a more deterministic, less-wobbly gait
                dist, _ = model.forward(state_tensor.unsqueeze(0))
                action=dist.mean # Take the mean instead of sampling
                action=action.cpu().numpy().squeeze()
            
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            step_count += 1
            
            # Update camera to follow robot every 10 steps
            if step_count % 10 == 0:
                base_pos, _ = p.getBasePositionAndOrientation(env.robot_id, physicsClientId=env.client_id)
                # Camera follows robot: distance=2.5m, yaw=30deg, pitch=-20deg
                p.resetDebugVisualizerCamera(
                    cameraDistance=2.5,
                    cameraYaw=30,
                    cameraPitch=-20,
                    cameraTargetPosition=[base_pos[0], base_pos[1], 0.3],
                    physicsClientId=env.client_id
                )
            
            # Slower visualization - 30 FPS for better viewing
            time.sleep(1./30)
            
            if done:
                episode_count+=1
                base_pos, _ = p.getBasePositionAndOrientation(env.robot_id, physicsClientId=env.client_id)
                print(f"Episode {episode_count} finished. Reward: {episode_reward:.2f}, Distance: {base_pos[0]:.2f}m")
                state = env.reset()
                episode_reward = 0
                step_count = 0
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    finally:
        env.close()
        print("Environment closed. Demo ended.")

if __name__ == "__main__":
    main()

