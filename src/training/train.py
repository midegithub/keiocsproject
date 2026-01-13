import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob
import pickle
from pathlib import Path
from datetime import datetime

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

# Add src directory to Python path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

#Importing custom modules
from playground.main_playground import MainPlayground
from agent.ppo import PPOAgent
from agent.buffer import RolloutBuffer
from training.live_plotter import PlotterThread

# Configuration

# Force CPU usage for faster rollouts in single-process environments
# GPU is often slower due to transfer overhead when stepping environment one by one
DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE} (Forced CPU for faster rollouts)")

TOTAL_TIMESTEPS = 3_000_000 # Enough for 100m goal with denser updates
ROLLOUT_STEPS = 2048 # Longer rollouts stabilize GAE with residual control
MINIBATCH_SIZE = 128 # Slightly larger batches for better gradient estimates
NUM_EPOCHS = 5 # A bit more passes over fresh data
SAVE_INTERVAL = 10 # Save a little more often for analysis

# Visualization settings (will be set by user at start)
SHOW_LIVE_PLOTS = True
SHOW_AT_CHECKPOINTS = False
SHOW_NEW_RECORDS = False

# STABLE hyperparameters - tested for locomotion
HYPERPARAMETERS = {
    'lr': 1e-4, # LOW learning rate = stable learning
    'gamma': 0.99, # Standard discount factor
    'lambda_gae': 0.95, # GAE parameter
    'clip_epsilon': 0.2, # Matches PPO defaults, lets policy improve faster
    'v_coef': 1.0, # Higher value weight = better value learning
    'entropy_coef': 3e-4, # Encourage exploration without destabilizing gait
    'num_epochs': NUM_EPOCHS,
    'minibatch_size': MINIBATCH_SIZE,
    'max_grad_norm': 0.5 # Gradient clipping
    }


def ask_user_preferences():
    """Get user input for visualization settings at start"""
    
    print("\n" + "=" * 55)
    print("QUADRUPED TRAINING - Walk 100 meters!")
    print("=" * 55)
    print(f"Device: {DEVICE}")
    print(f"Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Rollout size: {ROLLOUT_STEPS}")
    print(f"Minibatch: {MINIBATCH_SIZE}")
    print("=" * 55)
    
    # live plots
    print("\n[1/3] Live Plotting (reward/distance graphs)")
    while True:
        try:
            ans = input("Enable live plots? (y/n) [n]: ").strip().lower()
            if ans in ('', 'n', 'no'):
                plots = False
                break
            elif ans in ('y', 'yes'):
                plots = True
                break
        except KeyboardInterrupt:
            plots = False
            break
    
    # checkpoint viz
    print("\n[2/3] Checkpoint Visualization (watch robot at saves)")
    while True:
        try:
            ans = input("Enable checkpoint viz? (y/n) [n]: ").strip().lower()
            if ans in ('', 'n', 'no'):
                checkpoint_viz = False
                break
            elif ans in ('y', 'yes'):
                checkpoint_viz = True
                break
        except KeyboardInterrupt:
            checkpoint_viz = False
            break
    
    # record viz
    print("\n[3/3] Record Visualization (watch when new record)")
    while True:
        try:
            ans = input("Show new records? (y/n) [n]: ").strip().lower()
            if ans in ('', 'n', 'no'):
                record_viz = False
                break
            elif ans in ('y', 'yes'):
                record_viz = True
                break
        except KeyboardInterrupt:
            record_viz = False
            break
    
    print("\n" + "=" * 55)
    print("Settings:")
    print(f"  Plots: {'ON' if plots else 'OFF'}")
    print(f"  Checkpoint viz: {'ON' if checkpoint_viz else 'OFF'}")
    print(f"  Record viz: {'ON' if record_viz else 'OFF'}")
    print("=" * 55)
    print("\nStarting training...\n")
    
    return plots, checkpoint_viz, record_viz


def visualize_checkpoint(agent, num_episodes=1):
    """
    Simple visualization of robot's current performance with GUI.
    Camera follows the robot for better viewing.
    """
    print(f"\n{'='*50}")
    print(f"CHECKPOINT VISUALIZATION")
    print(f"{'='*50}")
    
    import time
    import pybullet as p
    
    viz_env = None
    try:
        # Create temporary environment with GUI - use same settings as training!
        viz_env = MainPlayground(gui=True, sim_steps_per_action=6, use_position_control=True)
        
        # Ensure ground plane is visible - reset camera to show it
        p.resetDebugVisualizerCamera(
            cameraDistance=3.0,
            cameraYaw=45,
            cameraPitch=-25,
            cameraTargetPosition=[0, 0, 0],
            physicsClientId=viz_env.client_id
        )
        
        for ep in range(num_episodes):
            state = viz_env.reset(randomize=False)
            episode_reward = 0
            done = False
            steps = 0
            max_steps = 8000  # Longer visualization to see walking
            
            print(f"Episode {ep+1} - Running for up to {max_steps} steps...")
            
            while not done and steps < max_steps:
                # Check if GUI is still connected
                if not p.isConnected(viz_env.client_id):
                    print("GUI window closed by user")
                    break
                    
                state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE)
                
                with torch.no_grad():
                    # Use mean action (no sampling) for smoother visualization
                    dist, _ = agent.model.forward(state_tensor.unsqueeze(0))
                    action = dist.mean.cpu().numpy().squeeze()
                
                state, reward, done, info = viz_env.step(action)
                episode_reward += reward
                steps += 1
                
                # Update camera to follow robot every 10 steps
                if steps % 10 == 0:
                    base_pos, _ = p.getBasePositionAndOrientation(viz_env.robot_id, physicsClientId=viz_env.client_id)
                    # Camera follows robot: distance=2.5m, yaw=30deg, pitch=-20deg
                    # Keep camera slightly elevated to see ground
                    p.resetDebugVisualizerCamera(
                        cameraDistance=2.5,
                        cameraYaw=30,
                        cameraPitch=-20,
                        cameraTargetPosition=[base_pos[0], base_pos[1], 0.15],
                        physicsClientId=viz_env.client_id
                    )
                
                # Slower visualization - 30 FPS instead of 240 FPS
                time.sleep(1./30)
            
            dist = info.get('distance', 0.0)
            print(f"Episode {ep+1} done: {steps} steps, reward: {episode_reward:.2f}, dist={dist:.2f}m")
        
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


def show_new_record(agent, distance):
    """Celebrate a new distance record with visualization"""
    import pybullet as p
    import time
    
    print(f"\n{'*'*45}")
    print(f"NEW RECORD: {distance:.2f}m!!")
    print(f"{'*'*45}")
    
    viz_env = None
    try:
        viz_env = MainPlayground(gui=True, sim_steps_per_action=6, use_position_control=True)
        
        # Ensure ground plane is visible
        p.resetDebugVisualizerCamera(
            cameraDistance=3.0,
            cameraYaw=45,
            cameraPitch=-25,
            cameraTargetPosition=[0, 0, 0],
            physicsClientId=viz_env.client_id
        )
        
        state = viz_env.reset(randomize=False)
        
        for step in range(5000):
            if not p.isConnected(viz_env.client_id):
                break
            
            state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                dist, _ = agent.model.forward(state_t.unsqueeze(0))
                action = dist.mean.cpu().numpy().squeeze()
            
            state, _, done, info = viz_env.step(action)
            
            if step % 10 == 0:
                base_pos, _ = p.getBasePositionAndOrientation(viz_env.robot_id, physicsClientId=viz_env.client_id)
                p.resetDebugVisualizerCamera(2.5, 30, -20, [base_pos[0], base_pos[1], 0.15], physicsClientId=viz_env.client_id)
            
            time.sleep(0.01)
            if done:
                break
        
        print("Record viz done\n")
        
    except Exception as e:
        print(f"Viz error: {e}")
    finally:
        if viz_env is not None:
            try:
                if p.isConnected(viz_env.client_id):
                    viz_env.close()
            except:
                pass


def find_latest_model():
    """Find most recent saved model to resume training"""
    model_files = glob.glob("models/ppo_spotmicro_*.pth")
    
    numeric_models = []
    for f in model_files:
        try:
            suffix = f.split('_')[-1].split('.')[0]
            steps = int(suffix)
            numeric_models.append((f, steps))
        except ValueError:
            continue  # skip non-numeric like BEST.pth
    
    if numeric_models:
        return max(numeric_models, key=lambda x: x[1])[0]
    return None


#Initialization
def main():
    global SHOW_LIVE_PLOTS, SHOW_AT_CHECKPOINTS, SHOW_NEW_RECORDS
    
    # Get user preferences for visualization
    SHOW_LIVE_PLOTS, SHOW_AT_CHECKPOINTS, SHOW_NEW_RECORDS = ask_user_preferences()
    
    #Create output directory
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Start live plotter if enabled
    plotter = None
    if SHOW_LIVE_PLOTS:
        try:
            plotter = PlotterThread(max_episodes=5000)
            plotter.start()
        except Exception as e:
            print(f"Plotter failed: {e}")
            plotter = None
    #Set gui=False for maximum training speed, use position control for stability
    # Use 24 physics steps per action = MUCH faster training
    env = MainPlayground(gui=False, sim_steps_per_action=24, use_position_control=True)
    state=env.reset()
    state_dim=env.state_dim
    action_dim=env.action_dim
    print(f'Environment initialized. State dim={state_dim}, Action dim={action_dim}')
    print(f'Simulation: {env.sim_steps_per_action}x physics steps per action')

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
    latest_model = find_latest_model()
    if latest_model:
        try:
            agent.model.load_state_dict(torch.load(latest_model, map_location=DEVICE))
            print(f"Loaded model: {latest_model}")
        except Exception as e:
            print(f"Couldn't load {latest_model}: {e}")
            print("Starting fresh")
    else:
        print("No existing model found, starting from scratch")
    
    buffer = RolloutBuffer(ROLLOUT_STEPS, state_dim, action_dim, DEVICE)

    #Logging
    all_ep_rewards=[]
    all_avg_rewards=[]
    all_ep_distances=[]
    all_ep_times=[]
    current_ep_reward=0
    state_tensor=torch.tensor(state, dtype=torch.float32, device=DEVICE) 
    # The tensor is a multi-dimensional array of numbers used for calculations on GPU or CPU.
    num_timesteps=0
    rollout_count=0
    
    # Track best model performance
    best_avg_reward = float('-inf')
    best_distance = 0.0

    print("\n" + "="*55)
    print("TRAINING STARTED")
    print("="*55 + "\n")
    
    #Main training loop
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
            next_state, reward, done, info = env.step(action)

            #Update metrics
            current_ep_reward+=reward

            #Store data in buffer
            buffer.add(state_tensor, action_tensor, reward, log_prob, value, done)

            state=next_state
            state_tensor=torch.tensor(state, dtype=torch.float32, device=DEVICE)

            if done:
                episode_dist = info.get('distance', 0.0)
                survival_time = env.episode_time
                target_vel = info.get('target_velocity', getattr(env, 'target_velocity', 0.6))
                
                # Check for new distance record
                if episode_dist > best_distance:
                    old_best = best_distance
                    best_distance = episode_dist
                    
                    # Save model for significant improvements (> 0.5m)
                    if (episode_dist - old_best) > 0.5:
                        torch.save(agent.model.state_dict(), "models/ppo_spotmicro_BEST.pth")
                        print(f"*** NEW RECORD: {episode_dist:.2f}m (was {old_best:.2f}m) ***")
                        
                        if SHOW_NEW_RECORDS and (episode_dist - old_best) > 1.0:
                            show_new_record(agent, episode_dist)
                
                # Log episode results
                all_ep_rewards.append(current_ep_reward)
                all_ep_distances.append(episode_dist)
                all_ep_times.append(survival_time)

                #Calculate and log moving average (for smooth plotting)
                avg_reward=np.mean(all_ep_rewards[-50:])
                all_avg_rewards.append(avg_reward)
                
                # Update live plot if enabled
                if plotter:
                    try:
                        plotter.update(current_ep_reward, episode_dist, survival_time, num_timesteps)
                    except:
                        pass
                
                #print every 10th episode to reduce I/O overhead
                ep_num = len(all_ep_rewards)
                progress = (num_timesteps / TOTAL_TIMESTEPS) * 100
                if ep_num % 10 == 0 or episode_dist > best_distance * 0.9:  # Print every 10th or near-record
                    print(
                        f"[{progress:5.1f}%] Ep {ep_num:4d} | "
                        f"R={current_ep_reward:7.1f} | Avg={avg_reward:7.1f} | "
                        f"Dist={episode_dist:5.2f}m | T={survival_time:5.1f}s | "
                        f"Best={best_distance:.2f}m | VelTgt={target_vel:.2f}m/s"
                    )

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
            data_path = f"data/training_data_{num_timesteps}.pkl"
            with open(data_path, 'wb') as f:
                pickle.dump({
                    'rewards': all_ep_rewards, 
                    'avg_rewards': all_avg_rewards,
                    'distances': all_ep_distances,
                    'times': all_ep_times,
                    'avg_reward_50': current_avg,
                    'timesteps': num_timesteps,
                    'best_distance': best_distance
                }, f)
            
            # Save training plot at each checkpoint
            if len(all_ep_rewards) > 0:
                try:
                    fig = plot_rewards(all_ep_rewards, all_avg_rewards, all_ep_distances, 
                                      all_ep_times, num_timesteps, best_distance)
                    plot_path = f"plots/training_progress_{num_timesteps}.png"
                    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    # Also save as latest
                    fig = plot_rewards(all_ep_rewards, all_avg_rewards, all_ep_distances, 
                                      all_ep_times, num_timesteps, best_distance)
                    fig.savefig("plots/training_latest.png", dpi=150, bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    print(f"Warning: Could not save plot: {e}")
            
            # Track best model
            if current_avg > best_avg_reward:
                best_avg_reward = current_avg
                # Save a copy as "best" model with timestamp
                timestamp = datetime.now().strftime("%d%m%H%M")  # day month hour minute
                torch.save(agent.model.state_dict(), f"models/ppo_spotmicro_BEST_{timestamp}.pth")
                # Also save without timestamp for easy reference
                torch.save(agent.model.state_dict(), "models/ppo_spotmicro_BEST.pth")
                print(f"[SAVE] Step {num_timesteps:,} | NEW BEST avg={current_avg:.1f} | dist={best_distance:.2f}m | Plot saved")
            else:
                print(f"[SAVE] Step {num_timesteps:,} | avg={current_avg:.1f} | best_dist={best_distance:.2f}m | Plot saved")
            
            # Visualize checkpoint if enabled
            if SHOW_AT_CHECKPOINTS:
                visualize_checkpoint(agent, num_episodes=1)
        
        #print progress every rollout
        if len(all_ep_rewards)>0:
            recent_avg = np.mean(all_ep_rewards[-20:]) if len(all_ep_rewards)>=20 else np.mean(all_ep_rewards)
            total_rollouts = TOTAL_TIMESTEPS // ROLLOUT_STEPS
            print(f"[Rollout {rollout_count}/{total_rollouts}] Updated | Recent20={recent_avg:.1f} | Episodes={len(all_ep_rewards)}")
    
    env.close()
    
    # Save final plot from live plotter
    if plotter:
        try:
            plotter.save("plots/training_live_final.png")
            plotter.close()
        except:
            pass
    
    # Print training summary
    print("\n" + "="*55)
    print("TRAINING COMPLETE")
    print("="*55)
    print(f"Total episodes: {len(all_ep_rewards)}")
    print(f"Total timesteps: {num_timesteps:,}")
    print(f"Best distance: {best_distance:.2f}m")
    print(f"Final avg reward: {all_avg_rewards[-1] if all_avg_rewards else 0:.1f}")
    print(f"\nBest model: models/ppo_spotmicro_BEST.pth")
    print("Run demo: python src/demo.py")
    print("="*55)

    #Plotting final results
    if len(all_ep_rewards) > 0:
        try:
            fig = plot_rewards(all_ep_rewards, all_avg_rewards, all_ep_distances, 
                              all_ep_times, num_timesteps, best_distance)
            fig.savefig("plots/training_final.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            print("Final training plot saved to plots/training_final.png")
        except Exception as e:
            print(f"Warning: Could not save final plot: {e}")

def plot_rewards(rewards, avg_rewards, distances, times, timesteps, best_distance):
    """Create comprehensive training plot matching the live plotter style"""
    print("Creating training plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    
    episodes = list(range(1, len(rewards) + 1))
    
    # ===== PLOT 1: REWARD GRAPH =====
    ax1 = axes[0, 0]
    ax1.plot(episodes, rewards, 'b-', alpha=0.4, linewidth=0.8, label='Episode')
    # Moving average (20 episodes to match live plotter)
    if len(rewards) >= 20:
        avg = np.convolve(rewards, np.ones(20)/20, mode='valid')
        ax1.plot(episodes[19:], avg, 'r-', linewidth=2, label='Avg(20)')
    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode Rewards', fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # ===== PLOT 2: DISTANCE GRAPH (auto-scaled to actual progress) =====
    ax2 = axes[0, 1]
    ax2.plot(episodes, distances, 'g-', alpha=0.4, linewidth=0.8, label='Episode')
    # Moving average
    if len(distances) >= 20:
        avg = np.convolve(distances, np.ones(20)/20, mode='valid')
        ax2.plot(episodes[19:], avg, 'darkgreen', linewidth=2, label='Avg(20)')
    # Best distance line
    ax2.axhline(y=best_distance, color='red', linestyle='--', linewidth=1.5, 
               label=f'Best: {best_distance:.2f}m')
    # Auto-scale Y axis to show actual progress (max of best_distance * 1.5 or 10m minimum)
    y_max = max(best_distance * 1.5, 10.0)
    ax2.set_ylim(0, y_max)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Distance (m)')
    ax2.set_title(f'Distance Traveled (Goal: 100m)', fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # ===== PLOT 3: SURVIVAL TIME GRAPH =====
    ax3 = axes[1, 0]
    best_time = max(times) if times else 0.0
    ax3.plot(episodes, times, 'purple', alpha=0.4, linewidth=0.8, label='Episode')
    # Moving average
    if len(times) >= 20:
        avg = np.convolve(times, np.ones(20)/20, mode='valid')
        ax3.plot(episodes[19:], avg, 'darkviolet', linewidth=2, label='Avg(20)')
    ax3.axhline(y=best_time, color='red', linestyle='--', linewidth=1.5,
               label=f'Best: {best_time:.1f}s')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('Survival Time', fontweight='bold')
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # ===== PLOT 4: STATISTICS =====
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Recent averages (50 episodes to match training logs)
    recent_reward = np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards) if rewards else 0
    recent_dist = np.mean(distances[-50:]) if len(distances) >= 50 else np.mean(distances) if distances else 0
    recent_time = np.mean(times[-50:]) if len(times) >= 50 else np.mean(times) if times else 0
    
    stats_text = f"""
╔══════════════════════════════════════════╗
║          TRAINING STATISTICS             ║
╠══════════════════════════════════════════╣
║  Episodes: {len(episodes):>6}                       ║
║  Timesteps: {timesteps:>10,}                ║
╠══════════════════════════════════════════╣
║         BEST STATS (Independent)         ║
╠──────────────────────────────────────────╣
║  Best Reward:    {max(rewards) if rewards else 0:>10.2f}             ║
║  Best Distance:  {best_distance:>10.2f} m           ║
║  Best Survival:  {best_time:>10.1f} s           ║
╠══════════════════════════════════════════╣
║           RECENT AVERAGES (50ep)         ║
╠──────────────────────────────────────────╣
║  Avg Reward:   {recent_reward:>8.2f}                 ║
║  Avg Distance: {recent_dist:>8.2f} m               ║
║  Avg Survival: {recent_time:>8.1f} s               ║
╚══════════════════════════════════════════╝
"""
    
    ax4.text(0.05, 0.95, stats_text, fontsize=9, family='monospace',
            verticalalignment='top', transform=ax4.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Update title with progress
    progress = (timesteps / TOTAL_TIMESTEPS) * 100
    fig.suptitle(f'Training Progress - {progress:.1f}% | Best Distance: {best_distance:.2f}m / 100m Goal', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    main()
