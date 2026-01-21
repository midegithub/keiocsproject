# obstacles.py - obstacle management for robot training
# rectangles (steps) and cylinders (pillars) only

import pybullet as p
import numpy as np

# corridor config - walls 2m left and 2m right
CORRIDOR_WIDTH = 4.0
CORRIDOR_HALF = 2.0
CORRIDOR_LENGTH = 150.0

# fixed rectangle size
RECT_HEIGHT = 0.04
RECT_LENGTH = 1.0

# fixed cylinder size
CYL_RADIUS = 0.4
CYL_HEIGHT = 2.0

WALL_HEIGHT = 3.0
WALL_THICK = 0.15

# colors
RECT_COLOR = [0.55, 0.55, 0.55, 1.0]
CYL_COLOR = [0.6, 0.4, 0.25, 1.0]
WALL_COLOR = [0.25, 0.25, 0.3, 0.9]

# fixed spacing between obstacles
OBSTACLE_SPACING = 4.0


class Obstacle:
    """single obstacle in simulation"""
    def __init__(self, obs_type, pos, body_id, dims):
        self.type = obs_type
        self.position = pos
        self.body_id = body_id
        self.dimensions = dims


class ObstacleManager:
    """manages obstacles - creation and removal"""
    
    def __init__(self, client_id):
        self.client_id = client_id
        self.obstacles = []
        self.wall_ids = []
        self._types = []
    
    def set_active_types(self, types):
        self._types = types
    
    def get_active_types(self):
        return self._types
    
    def clear_all(self):
        for obs in self.obstacles:
            try:
                p.removeBody(obs.body_id, physicsClientId=self.client_id)
            except: pass
        
        for wid in self.wall_ids:
            try:
                p.removeBody(wid, physicsClientId=self.client_id)
            except: pass
        
        self.obstacles = []
        self.wall_ids = []
    
    def create_walls(self, start_x=-5.0, length=None):
        """create corridor walls at +/- 2m"""
        if length is None:
            length = CORRIDOR_LENGTH
        
        half_ext = [length/2, WALL_THICK/2, WALL_HEIGHT/2]
        
        # left wall
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_ext, 
                                  rgbaColor=WALL_COLOR, physicsClientId=self.client_id)
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_ext, 
                                     physicsClientId=self.client_id)
        left_wall = p.createMultiBody(0, col, vis,
                                      [start_x + length/2, -CORRIDOR_HALF - WALL_THICK/2, WALL_HEIGHT/2],
                                      physicsClientId=self.client_id)
        self.wall_ids.append(left_wall)
        
        # right wall
        vis2 = p.createVisualShape(p.GEOM_BOX, halfExtents=half_ext,
                                   rgbaColor=WALL_COLOR, physicsClientId=self.client_id)
        col2 = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_ext,
                                      physicsClientId=self.client_id)
        right_wall = p.createMultiBody(0, col2, vis2,
                                       [start_x + length/2, CORRIDOR_HALF + WALL_THICK/2, WALL_HEIGHT/2],
                                       physicsClientId=self.client_id)
        self.wall_ids.append(right_wall)
    
    def create_rectangle(self, x, y):
        """create a rectangular step with fixed size"""
        height = RECT_HEIGHT
        width = CORRIDOR_WIDTH
        length = RECT_LENGTH
        
        half = [length/2, width/2, height/2]
        
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half,
                                  rgbaColor=RECT_COLOR, physicsClientId=self.client_id)
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half,
                                     physicsClientId=self.client_id)
        
        body = p.createMultiBody(0, col, vis, [x, y, height/2],
                                 physicsClientId=self.client_id)
        
        p.changeDynamics(body, -1, lateralFriction=1.5, spinningFriction=0.3,
                        rollingFriction=0.1, physicsClientId=self.client_id)
        
        obs = Obstacle('rectangle', (x, y, height/2), body,
                      {'height': height, 'width': width, 'length': length})
        self.obstacles.append(obs)
        return obs
    
    def create_cylinder(self, x, y):
        """create a cylindrical pillar with fixed size"""
        radius = CYL_RADIUS
        height = CYL_HEIGHT
        
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height,
                                  rgbaColor=CYL_COLOR, physicsClientId=self.client_id)
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height,
                                     physicsClientId=self.client_id)
        
        body = p.createMultiBody(0, col, vis, [x, y, height/2],
                                 physicsClientId=self.client_id)
        
        obs = Obstacle('cylinder', (x, y, height/2), body,
                      {'radius': radius, 'height': height})
        self.obstacles.append(obs)
        return obs
    
    def generate_course(self, obstacle_types, start_x=5.0, end_x=120.0, seed=None, create_walls=False):
        """generate obstacle course
        
        Args:
            obstacle_types: List of obstacle types to generate
            start_x: Starting x position for obstacles
            end_x: Ending x position for obstacles  
            seed: Random seed for reproducibility
            create_walls: Whether to create corridor walls (False if parent already has them)
        """
        if seed is not None:
            np.random.seed(seed)
        
        self._types = obstacle_types
        
        # Only create walls if explicitly requested (MainPlayground already creates them)
        if create_walls:
            self.create_walls(start_x=-2.0, length=end_x + 10)
        
        x = start_x
        count = 0
        
        while x < end_x:
            obs_type = np.random.choice(obstacle_types)
            
            # random lateral position for all obstacles
            y_pos = np.random.uniform(-0.5, 0.5)
            
            try:
                if obs_type == 'rectangle':
                    self.create_rectangle(x, y_pos)
                    count += 1
                    x += OBSTACLE_SPACING + np.random.uniform(0, 2)
                    
                elif obs_type == 'cylinder':
                    self.create_cylinder(x, y_pos)
                    count += 1
                    x += OBSTACLE_SPACING + np.random.uniform(0, 1.5)
                    
            except Exception as e:
                print(f"obstacle creation error: {e}")
                x += 2.0
        
        return count
    
    def get_info(self):
        """return info about current obstacles"""
        info = {'total': len(self.obstacles), 'types': {}, 'walls': len(self.wall_ids)}
        
        for obs in self.obstacles:
            t = obs.type
            if t not in info['types']:
                info['types'][t] = 0
            info['types'][t] += 1
        
        return info


class ObstacleConfig:
    CORRIDOR_WIDTH = CORRIDOR_WIDTH
    CORRIDOR_HALF_WIDTH = CORRIDOR_HALF
    CORRIDOR_LENGTH = CORRIDOR_LENGTH
    WALL_HEIGHT = WALL_HEIGHT


class ObstacleType:
    RECTANGLE = 'rectangle'
    CYLINDER = 'cylinder'
