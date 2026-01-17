# demo.py - trained robot demonstration
# loads model and runs in real time

import torch
import time
import os
import sys
import glob
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from playground.main_playground import MainPlayground
from playground.obstacle_playground import ObstaclePlayground
from agent.actor_critic import ActorCritic

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE}")


def find_models():
    """find all available models"""
    models = []
    patterns = ["models/*.pth", "models/**/*.pth"]
    for pat in patterns:
        models.extend(glob.glob(pat, recursive=True))
    return sorted(set(models))


def find_best():
    """find best model"""
    if os.path.exists("models/obstacles/ppo_obstacle_BEST.pth"):
        return "models/obstacles/ppo_obstacle_BEST.pth"
    if os.path.exists("models/ppo_spotmicro_BEST.pth"):
        return "models/ppo_spotmicro_BEST.pth"
    if os.path.exists("models/ppo_spotmicro_LAST.pth"):
        return "models/ppo_spotmicro_LAST.pth"
    return None


def ask_model():
    """ask which model to use"""
    print("\n" + "="*50)
    print("ROBOT DEMO")
    print("="*50)
    
    all_models = find_models()
    default = find_best()
    
    print("\nAvailable models:")
    for i, m in enumerate(all_models):
        tag = " (default)" if m == default else ""
        print(f"  [{i+1}] {m}{tag}")
    
    while True:
        try:
            inp = input("\nSelect (Enter=default): ").strip()
            
            if inp == "":
                if default:
                    return default
                print("no default model found")
                continue
            
            if inp.isdigit():
                idx = int(inp) - 1
                if 0 <= idx < len(all_models):
                    return all_models[idx]
            
            if os.path.exists(inp):
                return inp
            
            print("invalid choice")
        except KeyboardInterrupt:
            return None


def ask_obstacles():
    """ask obstacle config - single step menu"""
    print("\n" + "="*50)
    print("OBSTACLE CONFIG")
    print("="*50)
    
    print("\n[1] No obstacles (No walls)")
    print("[2] Rectangle obstacles (Steps)")
    print("[3] Cylinder obstacles (Pillars)")
    print("[4] All obstacles (Steps and Pillars)")
    
    while True:
        try:
            c = input("\nSelect [1-4] (default=1): ").strip()
            if c == "" or c == "1":
                return None
            elif c == "2":
                return ['rectangle']
            elif c == "3":
                return ['cylinder']
            elif c == "4":
                return ['rectangle', 'cylinder']
            print("invalid choice")
        except KeyboardInterrupt:
            return None


def main():
    """run demo"""
    
    USE_STOCHASTIC = False
    
    model_path = ask_model()
    if model_path is None:
        return
    
    if not os.path.exists(model_path):
        print(f"file not found: {model_path}")
        return
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model_dim = checkpoint['shared_layers.0.weight'].shape[1]
    
    is_obs_model = model_dim > 37
    
    if is_obs_model:
        print(f"\nOBSTACLE model detected (dim={model_dim})")
    else:
        print(f"\nBASE model detected (dim={model_dim})")
    
    obs_types = ask_obstacles()
    
    # create environment based on user choice
    if obs_types is not None:
        env = ObstaclePlayground(
            gui=True, sim_steps_per_action=24, use_position_control=True,
            demo_mode=True, obstacle_types=obs_types, enable_obstacles=True
        )
        print(f"obstacles: {obs_types}")
    else:
        # no obstacles - use base playground for base model, obstacle playground without obstacles for obstacle model
        if is_obs_model:
            env = ObstaclePlayground(
                gui=True, sim_steps_per_action=24, use_position_control=True,
                demo_mode=True, obstacle_types=[], enable_obstacles=False
            )
        else:
            env = MainPlayground(
                gui=True, sim_steps_per_action=24,
                use_position_control=True, demo_mode=True
            )
        print("flat ground (no obstacles)")
    
    import pybullet as p
    
    # disable useless debug panels
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=env.client_id)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=env.client_id)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=env.client_id)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=env.client_id)
    
    state = env.reset(randomize=True)
    
    if env.state_dim != model_dim:
        print(f"ERROR: dimension mismatch model={model_dim} env={env.state_dim}")
        env.close()
        return
    
    model = ActorCritic(env.state_dim, env.action_dim).to(DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()
    
    print(f"loaded model: {model_path}")
    print("\nControls:")
    print("  C = toggle camera follow")
    print("  R = reset")
    print("  Ctrl+C = stop")
    print()
    
    episode = 0
    ep_reward = 0
    step = 0
    
    step_time = 0.024
    cam_follow = True
    text_id = None
    
    try:
        last_t = time.time()
        ep_start = time.time()
        
        while True:
            state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE)
            
            with torch.no_grad():
                dist, _ = model.forward(state_t.unsqueeze(0))
                if USE_STOCHASTIC:
                    action = dist.sample().cpu().numpy().squeeze()
                else:
                    action = dist.mean.cpu().numpy().squeeze()
            
            state, reward, done, info = env.step(action)
            ep_reward += reward
            step += 1
            
            keys = p.getKeyboardEvents(physicsClientId=env.client_id)
            if ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED:
                cam_follow = not cam_follow
                print(f"camera follow: {cam_follow}")
            if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
                state = env.reset(randomize=True)
                ep_reward = 0
                step = 0
                ep_start = time.time()
                continue
            
            pos, _ = p.getBasePositionAndOrientation(env.robot_id, physicsClientId=env.client_id)
            
            if cam_follow and step % 30 == 0:
                p.resetDebugVisualizerCamera(5.0, -90, -45, [pos[0], pos[1], 0.2],
                                            physicsClientId=env.client_id) 
            
            elapsed = time.time() - ep_start
            dist_val = info.get('distance', 0)
            speed = dist_val / elapsed if elapsed > 0 else 0
            
            txt = f"t={elapsed:.1f}s dist={dist_val:.2f}m speed={speed:.2f}m/s R={ep_reward:.1f}"
            if obs_types is not None:
                txt += f" obs={info.get('obstacles_passed',0)}"
            
            if text_id is not None:
                text_id = p.addUserDebugText(txt, [pos[0], pos[1], pos[2]+0.5],
                                            textColorRGB=[1,1,0], textSize=1.5,
                                            replaceItemUniqueId=text_id,
                                            physicsClientId=env.client_id)
            else:
                text_id = p.addUserDebugText(txt, [pos[0], pos[1], pos[2]+0.5],
                                            textColorRGB=[1,1,0], textSize=1.5,
                                            physicsClientId=env.client_id)
            
            dt = time.time() - last_t
            if dt < step_time:
                time.sleep(step_time - dt)
            last_t = time.time()
            
            if done:
                episode += 1
                print(f"ep {episode}: R={ep_reward:.1f} dist={info.get('distance',0):.2f} "
                      f"obs={info.get('obstacles_passed',0)}")
                
                if text_id:
                    p.removeUserDebugItem(text_id, physicsClientId=env.client_id)
                    text_id = None
                
                state = env.reset(randomize=False)
                ep_reward = 0
                step = 0
                ep_start = time.time()
    
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    finally:
        env.close()
        print("Done")


if __name__ == "__main__":
    main()
