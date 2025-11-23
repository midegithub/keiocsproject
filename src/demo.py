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

#Configuration
MODEL_PATH = "models/ppo_spotmicro_163840.pth" # Model to load for demonstration

def main():
    """
    Demo script to load and run a trained PPO agent.
    Uses the mean of the policy distribution for deterministic, stable locomotion"""
    
    # We want to see this, so gui=True
    env = MainPlayground(gui=True)
    
    # Get state and action dimensions
    state = env.reset()
    state_dim = env.state_dim
    action_dim = env.action_dim
    print(f"Loading trained model for demo. State dim={state_dim}, Action dim={action_dim}")
    
    # --- Load the TRAINED model ---
    model = ActorCritic(state_dim, action_dim).to(DEVICE)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please update MODEL_PATH to point to your trained model.")
        env.close()
        return
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() # Set to evaluation mode (e.g., disables dropout if any)
    print(f"Model loaded successfully from {MODEL_PATH}")
    print("Running demo... Press Ctrl+C to exit.")
    
    state = env.reset()
    episode_count = 0
    episode_reward = 0
    
    try:
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
            
            # Run in real-time (PyBullet default is 240 Hz)
            time.sleep(1./60) #60fps
            
            if done:
                episode_count+=1
                print(f"Episode {episode_count} finished. Reward: {episode_reward:.2f}")
                state = env.reset()
                episode_reward = 0
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    finally:
        env.close()
        print("Environment closed. Demo ended.")

if __name__ == "__main__":
    main()

