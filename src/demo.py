"""
Demo script for trained quadruped robot
Loads the best model and runs it with visualization
"""
import torch
import time
import os
import sys
import glob
import pickle
from pathlib import Path

# setup imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from playground.main_playground import MainPlayground
from agent.actor_critic import ActorCritic

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


def find_best_model():
    """automatically find the best trained model"""
    
    # first check for explicit BEST model
    best_path = "models/ppo_spotmicro_BEST.pth"
    if os.path.exists(best_path):
        print(f"Found BEST model: {best_path}")
        return best_path
    
    # look for best performing from training data
    data_files = glob.glob("data/training_data_*.pkl")
    if data_files:
        best_reward = float('-inf')
        best_timestep = None
        
        for f in data_files:
            try:
                with open(f, 'rb') as fp:
                    data = pickle.load(fp)
                    avg = data.get('avg_reward_50', float('-inf'))
                    ts = data.get('timesteps', 0)
                    
                    if avg > best_reward:
                        best_reward = avg
                        best_timestep = ts
            except:
                continue
        
        if best_timestep:
            model_path = f"models/ppo_spotmicro_{best_timestep}.pth"
            if os.path.exists(model_path):
                print(f"Best model (avg reward {best_reward:.1f}): {model_path}")
                return model_path
    
    # fallback: get latest model by timestep
    model_files = glob.glob("models/ppo_spotmicro_*.pth")
    if model_files:
        numeric = []
        for f in model_files:
            try:
                suffix = f.split('_')[-1].split('.')[0]
                ts = int(suffix)
                numeric.append((f, ts))
            except ValueError:
                continue
        
        if numeric:
            latest = max(numeric, key=lambda x: x[1])[0]
            print(f"Using latest model: {latest}")
            return latest
    
    return None


def main():
    """run the trained robot with visualization"""
    
    # === USER OPTIONS ===
    # True  -> sample actions like training (adds exploration / wobble)
    # False -> use mean actions for smoother, repeatable demos
    USE_STOCHASTIC_ACTIONS = False
    
    # find model
    model_path = find_best_model()
    if model_path is None or not os.path.exists(model_path):
        print("ERROR: No trained model found!")
        print("Train one first: python src/training/train.py")
        return
    
    # create environment with gui
    # IMPORTANT: use same sim_steps_per_action as training (24) for correct behavior
    # The policy was trained with 24 steps, so demo must match for correct behavior
    # demo_mode=True removes time limits and relaxes termination so robot can run freely
    # maxVelocity limits in motor control ensure smooth motion despite faster stepping
    env = MainPlayground(gui=True, sim_steps_per_action=24, use_position_control=True, demo_mode=True)
    
    # Use same randomization as training for consistency
    state = env.reset(randomize=True)
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    
    # load model
    model = ActorCritic(state_dim, action_dim).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    print(f"Loaded model: {model_path}")
    print(f"Action mode: {'STOCHASTIC (like training)' if USE_STOCHASTIC_ACTIONS else 'DETERMINISTIC (mean)'}")
    print("\nRunning demo... Press Ctrl+C to stop")
    print("Camera follows the robot automatically\n")
    
    episode = 0
    ep_reward = 0
    step = 0
    
    import pybullet as p
    
    # timing for real-time playback
    # each step is 24 physics steps * 0.001s = 0.024s of sim time
    # we want roughly real-time, so sleep ~24ms per step
    step_time = 0.024  # 24ms per control step (matches training physics)
    
    # Increase solver iterations for smoother motion
    p.setPhysicsEngineParameter(
        numSolverIterations=50,
        physicsClientId=env.client_id
    )
    
    # debug text IDs for updating stats display
    stats_text_id = None
    
    try:
        last_time = time.time()
        episode_start_time = time.time()
        
        while True:
            state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE)
            
            with torch.no_grad():
                dist, _ = model.forward(state_t.unsqueeze(0))
                
                if USE_STOCHASTIC_ACTIONS:
                    # Sample from distribution (matches training behavior exactly)
                    action = dist.sample().cpu().numpy().squeeze()
                else:
                    # Use mean action for smooth, deterministic behavior
                    action = dist.mean.cpu().numpy().squeeze()
            
            state, reward, done, info = env.step(action)
            ep_reward += reward
            step += 1
            
            # get robot position for camera and stats
            base_pos, _ = p.getBasePositionAndOrientation(env.robot_id, physicsClientId=env.client_id)
            
            # camera follow
            if step % 10 == 0:
                p.resetDebugVisualizerCamera(
                    2.5, 30, -20,
                    [base_pos[0], base_pos[1], 0.3],
                    physicsClientId=env.client_id
                )
            
            # update real-time stats display in PyBullet window
            elapsed_real = time.time() - episode_start_time
            distance = info.get('distance', 0.0)
            speed = distance / elapsed_real if elapsed_real > 0 else 0.0
            
            target_vel = info.get('target_velocity', env.target_velocity)
            stats_text = (
                f"Time: {elapsed_real:.1f}s | Distance: {distance:.2f}m | "
                f"Speed: {speed:.2f}m/s | Target: {target_vel:.2f}m/s | "
                f"Reward: {ep_reward:.1f}"
            )
            
            # position text above the robot
            text_pos = [base_pos[0], base_pos[1], base_pos[2] + 0.5]
            
            if stats_text_id is not None:
                # update existing text
                stats_text_id = p.addUserDebugText(
                    stats_text,
                    text_pos,
                    textColorRGB=[1, 1, 0],  # yellow
                    textSize=1.5,
                    replaceItemUniqueId=stats_text_id,
                    physicsClientId=env.client_id
                )
            else:
                # create new text
                stats_text_id = p.addUserDebugText(
                    stats_text,
                    text_pos,
                    textColorRGB=[1, 1, 0],  # yellow
                    textSize=1.5,
                    physicsClientId=env.client_id
                )
            
            # real-time playback - sleep to match sim time
            elapsed = time.time() - last_time
            sleep_time = step_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_time = time.time()
            
            if done:
                episode += 1
                dist = info.get('distance', 0.0)
                target_vel = info.get('target_velocity', env.target_velocity)
                final_time = time.time() - episode_start_time
                print(
                    f"Episode {episode}: reward={ep_reward:.1f}, "
                    f"distance={dist:.2f}m, time={final_time:.1f}s, "
                    f"target_vel={target_vel:.2f}m/s"
                )
                
                # remove old stats text
                if stats_text_id is not None:
                    p.removeUserDebugItem(stats_text_id, physicsClientId=env.client_id)
                    stats_text_id = None
                
                state = env.reset(randomize=False)
                ep_reward = 0
                step = 0
                episode_start_time = time.time()
                
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    finally:
        env.close()
        print("Done")


if __name__ == "__main__":
    main()
