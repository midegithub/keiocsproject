# obstacle_playground.py - environment with obstacles
# extends MainPlayground with obstacle support

import pybullet as p
import numpy as np
import math

from playground.main_playground import MainPlayground
from playground.obstacles import ObstacleManager, ObstacleConfig, CORRIDOR_HALF


class ObstaclePlayground(MainPlayground):
    """extended playground with obstacles (steps and pillars)"""
    
    def __init__(self, gui=True, sim_steps_per_action=6, use_position_control=True,
                 demo_mode=False, obstacle_types=None, enable_obstacles=True):
        
        super().__init__(gui=gui, sim_steps_per_action=sim_steps_per_action,
                        use_position_control=use_position_control, demo_mode=demo_mode)
        
        self.enable_obstacles = enable_obstacles
        # if obstacle_types is None, default to both; if empty list, keep empty
        if obstacle_types is None:
            self._obstacle_types = ['rectangle', 'cylinder']
        else:
            self._obstacle_types = obstacle_types
        
        self.obstacle_manager = ObstacleManager(self.client_id)
        
        self.foot_contacts = np.zeros(4, dtype=np.float32)
        self.obstacles_passed = 0
        self.last_obs_x = 0.0
        
        self.foot_indices = self._find_feet()
        
        # 5 sensors for compatibility with existing models
        self.num_sensors = 5
        self.lookahead = 5.0
        
        # state dim = base (37) + sensors (5) = 42
        self.state_dim = 37 + self.num_sensors
        
        if gui:
            print(f"ObstaclePlayground: types={self._obstacle_types}")
    
    def _find_feet(self):
        """find foot link indices"""
        feet = []
        n = p.getNumJoints(self.robot_id, physicsClientId=self.client_id)
        
        for i in range(n):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client_id)
            name = info[12].decode('utf-8').lower()
            if 'foot' in name or 'toe' in name:
                feet.append(i)
        
        while len(feet) < 4:
            feet.append(-1)
        return feet[:4]
    
    def _detect_contacts(self):
        """detect foot contacts"""
        contacts = np.zeros(4, dtype=np.float32)
        
        for i, idx in enumerate(self.foot_indices):
            if idx < 0: continue
            pts = p.getContactPoints(bodyA=self.robot_id, linkIndexA=idx,
                                    physicsClientId=self.client_id)
            if len(pts) > 0:
                contacts[i] = 1.0
        return contacts
    
    def _check_progress(self, x):
        """count passed obstacles and calculate bonus"""
        passed = 0
        bonus = 0.0
        
        for obs in self.obstacle_manager.obstacles:
            ox = obs.position[0]
            
            if x > ox + 0.5:
                passed += 1
                
                if ox > self.last_obs_x:
                    b = 0.5
                    if 'height' in obs.dimensions:
                        b += obs.dimensions['height'] * 20.0
                    bonus += b
                    self.last_obs_x = ox
        
        return passed, bonus
    
    def reset(self, randomize=True):
        """reset env and regenerate obstacles"""
        self.obstacle_manager.clear_all()
        
        obs = super().reset(randomize=randomize)
        
        if self.enable_obstacles and len(self._obstacle_types) > 0:
            n = self.obstacle_manager.generate_course(
                obstacle_types=self._obstacle_types,
                start_x=5.0, end_x=120.0
            )
            if self.gui_mode:
                print(f"generated {n} obstacles")
        
        self.foot_contacts = np.zeros(4)
        self.obstacles_passed = 0
        self.last_obs_x = 0.0
        
        return obs
    
    def step(self, action):
        """execute step with obstacle handling"""
        obs, reward, done, info = super().step(action)
        
        self.foot_contacts = self._detect_contacts()
        
        pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
        self.obstacles_passed, bonus = self._check_progress(pos[0])
        
        obs_reward = self._obstacle_reward(pos)
        reward += obs_reward + bonus
        
        info['obstacles_passed'] = self.obstacles_passed
        info['total_obstacles'] = len(self.obstacle_manager.obstacles)
        info['foot_contacts'] = self.foot_contacts.copy()
        
        return obs, float(reward), done, info
    
    def _obstacle_reward(self, pos):
        """calculate obstacle-related reward"""
        reward = 0.0
        x, y, z = pos
        
        n_contacts = np.sum(self.foot_contacts)
        if n_contacts >= 2:
            reward += 0.1 * (n_contacts - 1)
        
        for obs in self.obstacle_manager.obstacles:
            ox, oy, oz = obs.position
            dx = abs(x - ox)
            dy = abs(y - oy)
            
            if obs.type == 'rectangle':
                half_l = obs.dimensions['length'] / 2
                half_w = obs.dimensions['width'] / 2
                if dx < half_l + 0.5 and dy < half_w + 0.5:
                    expected_z = 0.23 + obs.dimensions['height']
                    if abs(z - expected_z) < 0.1:
                        reward += 0.2
                    break
            
            elif obs.type == 'cylinder':
                dist = math.sqrt(dx**2 + dy**2)
                if dist < obs.dimensions['radius'] + 1.5:
                    if dist > obs.dimensions['radius'] + 0.3:
                        reward += 0.15
                    else:
                        reward -= 0.1
                    break
        
        # penalty near walls (walls at +/- 2m)
        wall_dist = CORRIDOR_HALF - abs(y)
        if wall_dist < 1.0:
            reward -= 0.2 * (1.0 - wall_dist)
        
        return reward
    
    def get_observation(self):
        """observation with obstacle sensors"""
        base_obs = super().get_observation()
        obs_sense = self._sense_obstacles()
        return np.concatenate([base_obs, obs_sense])
    
    def _sense_obstacles(self):
        """detect obstacle ahead"""
        obs = np.zeros(self.num_sensors, dtype=np.float32)
        obs[0] = 1.0  # max distance = no obstacle
        
        if not self.enable_obstacles or len(self.obstacle_manager.obstacles) == 0:
            return obs
        
        pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
        robot_x = pos[0]
        
        next_obs = None
        min_d = float('inf')
        
        for o in self.obstacle_manager.obstacles:
            d = o.position[0] - robot_x
            if 0 < d < self.lookahead and d < min_d:
                min_d = d
                next_obs = o
        
        if next_obs is None:
            return obs
        
        obs[0] = min_d / self.lookahead
        
        if 'height' in next_obs.dimensions:
            obs[1] = min(next_obs.dimensions['height'] / 0.10, 1.0)
        elif 'radius' in next_obs.dimensions:
            obs[1] = min(next_obs.dimensions['radius'] / 0.5, 1.0)
        
        # one-hot: idx 2=rect, 3=unused, 4=cylinder
        if next_obs.type == 'rectangle':
            obs[2] = 1.0
        elif next_obs.type == 'cylinder':
            obs[4] = 1.0
        
        return obs
    
    def is_done(self):
        """check if episode ended"""
        if super().is_done():
            return True
        
        # only check wall collision if obstacles are enabled (walls exist)
        if self.enable_obstacles and len(self._obstacle_types) > 0:
            pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
            if abs(pos[1]) > CORRIDOR_HALF - 0.2:
                if self.demo_mode:
                    print(f"  hit wall y={pos[1]:.2f}")
                return True
        
        return False
    
    def close(self):
        self.obstacle_manager.clear_all()
        super().close()
    
    def set_obstacle_types(self, types):
        """change types for next reset"""
        self._obstacle_types = types
    
    def set_obstacles_enabled(self, enabled):
        """enable/disable obstacles for next reset"""
        self.enable_obstacles = enabled
    
    def get_obstacle_info(self):
        return self.obstacle_manager.get_info()


def create_rectangle_course(gui=True, **kw):
    return ObstaclePlayground(gui=gui, obstacle_types=['rectangle'], **kw)

def create_cylinder_course(gui=True, **kw):
    return ObstaclePlayground(gui=gui, obstacle_types=['cylinder'], **kw)

def create_mixed_course(gui=True, **kw):
    return ObstaclePlayground(gui=gui, obstacle_types=['rectangle', 'cylinder'], **kw)


if __name__ == "__main__":
    import time
    
    print("testing ObstaclePlayground...")
    
    env = create_mixed_course(gui=True, sim_steps_per_action=6)
    
    print(f"state_dim={env.state_dim}, action_dim={env.action_dim}")
    print(f"obstacles: {env.get_obstacle_info()}")
    
    obs = env.reset(randomize=False)
    zero_act = np.zeros(env.action_dim)
    
    step = 0
    try:
        while True:
            obs, r, done, info = env.step(zero_act)
            step += 1
            
            if step % 60 == 0:
                print(f"t={env.episode_time:.1f}s dist={info['distance']:.2f} obs={info['obstacles_passed']}")
            
            if done:
                print(f"done. dist={info['distance']:.2f}, obstacles={info['obstacles_passed']}")
                obs = env.reset()
                step = 0
            
            time.sleep(1/60)
            
    except KeyboardInterrupt:
        print("stopped")
    finally:
        env.close()
