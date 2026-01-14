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
    print("\n" + "="*60)
    print("DEMO CONTROLS:")
    print("  - Press 'C' in PyBullet window to toggle camera follow")
    print("  - Use mouse to rotate camera when follow is OFF")
    print("  - Press Ctrl+C in terminal to stop")
    print("="*60 + "\n")
    
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
    
    # Camera follow mode (can be toggled)
    camera_follow = True
    camera_follow_interval = 30  # Update camera every N steps (less aggressive)
    
    # debug text IDs for updating stats display
    stats_text_id = None
    barrier_warning_id = None
    
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
            
            # Check for keyboard input to toggle camera follow
            keys = p.getKeyboardEvents(physicsClientId=env.client_id)
            if ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED:
                camera_follow = not camera_follow
                print(f"Camera follow: {'ON' if camera_follow else 'OFF'}")
            
            # get robot position for camera and stats
            base_pos, _ = p.getBasePositionAndOrientation(env.robot_id, physicsClientId=env.client_id)
            
            # camera follow (less aggressive, can be toggled off)
            if camera_follow and step % camera_follow_interval == 0:
                p.resetDebugVisualizerCamera(
                    3.0, 45, -25,  # Wider view
                    [base_pos[0], base_pos[1], 0.2],
                    physicsClientId=env.client_id
                )
            
            # Get barrier info
            barrier_pos = info.get('death_barrier_pos', env.death_barrier_pos)
            distance_from_barrier = info.get('distance_from_barrier', base_pos[0] - barrier_pos)
            
            # update real-time stats display in PyBullet window
            elapsed_real = time.time() - episode_start_time
            distance = info.get('distance', 0.0)
            speed = distance / elapsed_real if elapsed_real > 0 else 0.0
            
            target_vel = info.get('target_velocity', env.target_velocity)
            
            # Main stats text
            stats_text = (
                f"Time: {elapsed_real:.1f}s | Distance: {distance:.2f}m | "
                f"Speed: {speed:.2f}m/s | Reward: {ep_reward:.1f}"
            )
            
            # Barrier warning text (changes color based on danger)
            if distance_from_barrier < 5.0:
                barrier_color = [1, 0, 0]  # Red - DANGER
                barrier_text = f"!!! BARRIER: {distance_from_barrier:.1f}m BEHIND - DANGER !!!"
            elif distance_from_barrier < 10.0:
                barrier_color = [1, 0.5, 0]  # Orange - Warning
                barrier_text = f"BARRIER: {distance_from_barrier:.1f}m behind - Speed up!"
            else:
                barrier_color = [0, 1, 0]  # Green - Safe
                barrier_text = f"Barrier: {distance_from_barrier:.1f}m behind - Safe"
            
            # position text above the robot
            text_pos = [base_pos[0], base_pos[1], base_pos[2] + 0.5]
            barrier_text_pos = [base_pos[0], base_pos[1], base_pos[2] + 0.7]
            
            if stats_text_id is not None:
                stats_text_id = p.addUserDebugText(
                    stats_text,
                    text_pos,
                    textColorRGB=[1, 1, 0],  # yellow
                    textSize=1.5,
                    replaceItemUniqueId=stats_text_id,
                    physicsClientId=env.client_id
                )
            else:
                stats_text_id = p.addUserDebugText(
                    stats_text,
                    text_pos,
                    textColorRGB=[1, 1, 0],  # yellow
                    textSize=1.5,
                    physicsClientId=env.client_id
                )
            
            # Barrier warning text
            if barrier_warning_id is not None:
                barrier_warning_id = p.addUserDebugText(
                    barrier_text,
                    barrier_text_pos,
                    textColorRGB=barrier_color,
                    textSize=1.2,
                    replaceItemUniqueId=barrier_warning_id,
                    physicsClientId=env.client_id
                )
            else:
                barrier_warning_id = p.addUserDebugText(
                    barrier_text,
                    barrier_text_pos,
                    textColorRGB=barrier_color,
                    textSize=1.2,
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
                barrier_dist = info.get('distance_from_barrier', 0.0)
                final_time = time.time() - episode_start_time
                
                # Determine death cause
                if barrier_dist <= 0:
                    death_cause = "KILLED BY BARRIER"
                else:
                    death_cause = "fell/flipped"
                
                print(
                    f"Episode {episode}: reward={ep_reward:.1f}, "
                    f"distance={dist:.2f}m, time={final_time:.1f}s, "
                    f"cause={death_cause}"
                )
                
                # remove old stats text
                if stats_text_id is not None:
                    p.removeUserDebugItem(stats_text_id, physicsClientId=env.client_id)
                    stats_text_id = None
                if barrier_warning_id is not None:
                    p.removeUserDebugItem(barrier_warning_id, physicsClientId=env.client_id)
                    barrier_warning_id = None
                
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
