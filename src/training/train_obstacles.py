# train_obstacles.py - training on obstacle courses
# uses transfer learning from walking model

import torch
import numpy as np
import os
import sys
import pickle
from pathlib import Path

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from playground.obstacle_playground import ObstaclePlayground
from agent.ppo import PPOAgent
from agent.buffer import RolloutBuffer
from training.live_plotter import PlotterThread

DEVICE = torch.device("cpu")
print(f"device: {DEVICE}")

# training params
TOTAL_STEPS = 2_000_000
ROLLOUT_LEN = 2048
BATCH_SIZE = 128
EPOCHS = 5
SAVE_EVERY = 10

HYPERPARAMS = {
    'lr': 5e-5,
    'gamma': 0.99,
    'lambda_gae': 0.95,
    'clip_epsilon': 0.15,
    'v_coef': 1.0,
    'entropy_coef': 5e-4,
    'num_epochs': EPOCHS,
    'minibatch_size': BATCH_SIZE,
    'max_grad_norm': 0.5
}


def find_models():
    """find available models"""
    models = []
    
    paths = [
        "models/100m/ppo_spotmicro_BEST.pth",
        "models/ppo_spotmicro_BEST.pth",
        "models/ppo_spotmicro_LAST.pth",
        "models/obstacles/ppo_obstacle_BEST.pth",
    ]
    
    for p in paths:
        if os.path.exists(p):
            models.append(p)
    
    if os.path.exists("models"):
        for sub in sorted(os.listdir("models")):
            subpath = os.path.join("models", sub)
            if os.path.isdir(subpath):
                best = os.path.join(subpath, "ppo_spotmicro_BEST.pth")
                if os.path.exists(best) and best not in models:
                    models.append(best)
    
    return list(dict.fromkeys(models))


def setup():
    """interactive config"""
    print("\n" + "="*50)
    print("OBSTACLE TRAINING")
    print("="*50)
    
    # base model selection
    print("\n[1/2] BASE MODEL")
    models = find_models()
    
    if not models:
        print("no model found, training from scratch")
        base = None
    else:
        print("models:")
        for i, m in enumerate(models):
            print(f"  [{i+1}] {m}")
        print("  [0] from scratch")
        
        while True:
            try:
                c = input("select (default=1): ").strip()
                if c == "": 
                    base = models[0]
                    break
                if c == "0":
                    base = None
                    break
                if c.isdigit() and 0 < int(c) <= len(models):
                    base = models[int(c)-1]
                    break
                print("invalid")
            except KeyboardInterrupt:
                return None
    
    # obstacle types
    print("\n[2/2] OBSTACLE TYPES")
    print("  [1] rectangles only")
    print("  [2] cylinders only")
    print("  [3] both (mixed)")
    
    while True:
        try:
            t = input("select [1-3] (default=3): ").strip()
            if t == "": t = "3"
            if t in ["1","2","3"]: break
        except KeyboardInterrupt:
            return None
    
    types_map = {"1": ["rectangle"], "2": ["cylinder"], "3": ["rectangle", "cylinder"]}
    obs_types = types_map[t]
    
    # plots?
    plots = False
    try:
        a = input("live plots? (y/n) [n]: ").strip().lower()
        plots = a in ('y', 'yes')
    except: pass
    
    print("\n" + "-"*50)
    print(f"base: {base}")
    print(f"obstacles: {obs_types}")
    print("-"*50)
    
    ok = input("start training? (y/n) [y]: ").strip().lower()
    if ok in ('n', 'no'):
        return None
    
    return {'base': base, 'types': obs_types, 'plots': plots}


def main():
    """main training loop"""
    
    cfg = setup()
    if cfg is None:
        return
    
    os.makedirs("models/obstacles", exist_ok=True)
    os.makedirs("data/obstacles", exist_ok=True)
    
    plotter = None
    if cfg['plots']:
        try:
            plotter = PlotterThread(max_episodes=5000)
            plotter.start()
        except: pass
    
    env = ObstaclePlayground(
        gui=False,
        sim_steps_per_action=24,
        use_position_control=True,
        obstacle_types=cfg['types'],
        enable_obstacles=True
    )
    
    state = env.reset()
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    print(f"\nenv: state={state_dim}, action={action_dim}")
    print(f"obstacles: {cfg['types']}")
    
    agent = PPOAgent(state_dim, action_dim, DEVICE, HYPERPARAMS)
    
    if cfg['base'] and os.path.exists(cfg['base']):
        try:
            pretrained = torch.load(cfg['base'], map_location=DEVICE)
            new_state = agent.model.state_dict()
            
            old_dim = pretrained['shared_layers.0.weight'].shape[1]
            new_dim = new_state['shared_layers.0.weight'].shape[1]
            
            if old_dim == new_dim:
                agent.model.load_state_dict(pretrained)
                print(f"loaded: {cfg['base']}")
            else:
                print(f"adapting {old_dim} -> {new_dim} inputs")
                
                loaded = 0
                for name, param in pretrained.items():
                    if name in new_state:
                        if param.shape == new_state[name].shape:
                            new_state[name].copy_(param)
                            loaded += 1
                        elif 'shared_layers.0.weight' in name:
                            old_size = param.shape[1]
                            new_state[name][:, :old_size].copy_(param)
                            torch.nn.init.orthogonal_(new_state[name][:, old_size:], gain=0.1)
                            loaded += 1
                
                agent.model.load_state_dict(new_state)
                print(f"loaded {loaded} layers")
                
        except Exception as e:
            print(f"load error: {e}")
    
    buffer = RolloutBuffer(ROLLOUT_LEN, state_dim, action_dim, DEVICE)
    
    rewards_list = []
    distances = []
    obstacles_list = []
    
    ep_reward = 0
    state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE)
    timesteps = 0
    rollouts = 0
    
    best_avg = float('-inf')
    best_dist = 0.0
    
    print("\n" + "="*50)
    print("TRAINING STARTED")
    print("="*50 + "\n")
    
    while timesteps < TOTAL_STEPS:
        buffer.clear()
        
        for _ in range(ROLLOUT_LEN):
            timesteps += 1
            
            with torch.no_grad():
                act, logp, val = agent.model.act(state_t.unsqueeze(0))
            
            action = act.cpu().numpy().squeeze()
            next_state, reward, done, info = env.step(action)
            
            ep_reward += reward
            buffer.add(state_t, act, reward, logp, val, done)
            
            state = next_state
            state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE)
            
            if done:
                dist = info.get('distance', 0)
                obs_passed = info.get('obstacles_passed', 0)
                
                if dist > best_dist:
                    best_dist = dist
                    torch.save(agent.model.state_dict(), "models/obstacles/ppo_obstacle_BEST.pth")
                    print(f"*** new record: {dist:.2f}m ***")
                
                rewards_list.append(ep_reward)
                distances.append(dist)
                obstacles_list.append(obs_passed)
                
                avg = np.mean(rewards_list[-50:])
                
                if plotter:
                    try: plotter.update(ep_reward, dist, 0, timesteps)
                    except: pass
                
                ep_num = len(rewards_list)
                pct = timesteps / TOTAL_STEPS * 100
                
                if ep_num % 1 == 0:
                    print(f"[{pct:5.1f}%] ep {ep_num} | R={ep_reward:.1f} avg={avg:.1f} | "
                          f"dist={dist:.2f} obs={obs_passed}")
                
                state = env.reset()
                state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE)
                ep_reward = 0
        
        rollouts += 1
        
        with torch.no_grad():
            _, last_val = agent.model.forward(state_t.unsqueeze(0))
        
        adv, ret = agent.compute_advantages_and_returns(buffer, last_val, done)
        agent.update(buffer, adv, ret)
        
        if rollouts % SAVE_EVERY == 0:
            torch.save(agent.model.state_dict(), "models/obstacles/ppo_obstacle_LAST.pth")
            
            with open("data/obstacles/training_data.pkl", 'wb') as f:
                pickle.dump({
                    'rewards': rewards_list,
                    'distances': distances,
                    'obstacles': obstacles_list,
                    'timesteps': timesteps
                }, f)
            
            avg = np.mean(rewards_list[-50:]) if len(rewards_list) >= 50 else np.mean(rewards_list) if rewards_list else 0
            
            if avg > best_avg:
                best_avg = avg
                torch.save(agent.model.state_dict(), "models/obstacles/ppo_obstacle_BEST.pth")
                print(f"[SAVE] step {timesteps:,} | BEST avg={avg:.1f}")
            else:
                print(f"[SAVE] step {timesteps:,} | avg={avg:.1f}")
    
    env.close()
    
    if plotter:
        try:
            plotter.close()
        except: pass
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"episodes: {len(rewards_list)}")
    print(f"best distance: {best_dist:.2f}m")
    print(f"models saved in models/obstacles/")


if __name__ == "__main__":
    main()
