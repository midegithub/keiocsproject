#Import modules for environment implementation
import pybullet as p
import pybullet_data
import numpy as np
import time
import os
import math

# Physics constants for realistic simulation
PHYSICS_TIMESTEP = 0.001  
CONTROL_TIMESTEP = 0.006  # 6ms control loop (realistic for servos)
ACTION_REPEAT = 6  # steps per control 

# Motor parameters (tuned for MG996R-like servos)
MOTOR_KP = 1.2
MOTOR_KD = 0.03
MAX_TORQUE = 3.5
TORQUE_LIMIT = 5.7


class MotorController:
    """Handles the conversion from position commands to torques.
    Simulates real servo motor dynamics with back-EMF."""
    
    def __init__(self, num_motors=12):
        self.num_motors = num_motors
        self.kp = MOTOR_KP
        self.kd = MOTOR_KD
        
        # Empirical current-to-torque curve (from motor datasheet)
        self._current_pts = [0, 10, 20, 30, 40, 50, 60]
        self._torque_pts = [0, 1, 1.9, 2.45, 3.0, 3.25, 3.5]
        
        # Electrical params
        self.resistance = 0.186
        self.voltage = 32.0
        self.torque_const = 0.0954
        
    def compute_torque(self, cmd_pos, curr_pos, curr_vel, true_vel):
        """Convert position command to motor torque using PD + motor model"""
        # PD control to get pwm
        pos_err = cmd_pos - curr_pos
        pwm = self.kp * pos_err - self.kd * curr_vel
        pwm = np.clip(pwm, -1.0, 1.0)
        
        # Motor dynamics - back EMF reduces effective voltage
        voltage_eff = pwm * self.voltage - self.torque_const * true_vel
        voltage_eff = np.clip(voltage_eff, -50, 50)
        
        current = voltage_eff / self.resistance
        current_mag = np.abs(current)
        current_sign = np.sign(current)
        
        # Interpolate torque from current-torque curve
        torque = np.interp(current_mag, self._current_pts, self._torque_pts)
        torque = current_sign * torque
        
        return np.clip(torque, -MAX_TORQUE, MAX_TORQUE)


class GaitGenerator:
    """Generates foot trajectories using bezier curves.
    Much smoother than simple sinusoids."""
    
    def __init__(self, gait_type="trot"):
        self.phase = 0.0
        self.last_time = 0.0
        
        # Phase offsets for trot gait (diagonal pairs move together)
        if gait_type == "trot":
            self.offsets = np.array([0.0, 0.5, 0.5, 0.0])  # FL, FR, RL, RR
        else:  # walk
            self.offsets = np.array([0.0, 0.25, 0.5, 0.75])
            
        self.stance_ratio = 0.5  # Portion of cycle spent on ground
        
        # Bezier curve control points for swing phase
        # X trajectory (forward/back motion)
        self.swing_x = np.array([-0.04, -0.056, -0.06, -0.06, -0.06, 0.0,
                                  0.0, 0.0, 0.06, 0.06, 0.056, 0.04])
        # Z trajectory (lift height) 
        self.swing_z = np.array([0.0, 0.0, 0.04, 0.04, 0.04, 0.04,
                                 0.04, 0.05, 0.05, 0.05, 0.0, 0.0])
    
    def _binomial(self, n, k):
        """Binomial coefficient for bezier curve"""
        return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
    
    def _bezier(self, t, points):
        """Evaluate degree-11 bezier curve at parameter t"""
        n = len(points) - 1
        result = 0.0
        for k, pt in enumerate(points):
            coef = self._binomial(n, k) * (t ** k) * ((1 - t) ** (n - k))
            result += coef * pt
        return result
    
    def _stance_trajectory(self, phase_stance, velocity):
        """Linear trajectory during stance (foot on ground)"""
        half_stride = 0.05
        progress = half_stride * (1 - 2 * phase_stance)
        x = progress * abs(velocity)
        z = -0.001 * math.cos(math.pi * progress / (2 * half_stride))
        return x, z
    
    def _swing_trajectory(self, phase_swing, velocity, direction=1.0):
        """Bezier curve trajectory during swing (foot in air)"""
        x = self._bezier(phase_swing, self.swing_x * direction) * abs(velocity)
        z = self._bezier(phase_swing, self.swing_z) * abs(velocity)
        return x, z
    
    def get_foot_positions(self, t, velocity, period=0.4):
        """Get all 4 foot positions for current time.
        Returns 4x2 array of (x,z) offsets from neutral."""
        
        if period <= 0.01:
            period = 0.01
            
        # Update phase
        if self.phase >= 0.99:
            self.last_time = t
        self.phase = ((t - self.last_time) / period) % 1.0
        
        positions = np.zeros((4, 2))  # 4 feet x (x,z)
        
        direction = 1.0 if velocity >= 0 else -1.0
        
        for leg_id in range(4):
            leg_phase = (self.phase + self.offsets[leg_id]) % 1.0
            
            if leg_phase <= self.stance_ratio:
                # Stance phase - foot on ground
                phase_stance = leg_phase / self.stance_ratio
                x, z = self._stance_trajectory(phase_stance, velocity)
            else:
                # Swing phase - foot in air
                phase_swing = (leg_phase - self.stance_ratio) / (1 - self.stance_ratio)
                x, z = self._swing_trajectory(phase_swing, velocity, direction)
            
            positions[leg_id, 0] = x
            positions[leg_id, 1] = z
            
        return positions


class MainPlayground:
    def __init__(self, gui=True, sim_steps_per_action=6, use_position_control=True, demo_mode=False):
        """
        Initialize the robot simulation environment
        Args:
            gui: Whether to show visualization window
            sim_steps_per_action: Number of physics steps per action (higher = faster training)
            use_position_control: Use PD position control vs raw torque (more stable)
            demo_mode: Relaxed termination for visualization
        """
        #Start Physics server, p.GUI opens a visible window whereas p.DIRECT runs headless
        if gui:
            self.client_id=p.connect(p.GUI)
            #slider for debugging GUI
            p.configureDebugVisualizer(p.COV_ENABLE_GUI,1)
            p.resetDebugVisualizerCamera(cameraDistance=1.5,cameraYaw=20,cameraPitch=-30,cameraTargetPosition=[0,0,0.2])
        else :
            self.client_id=p.connect(p.DIRECT)

        #Setup data paths and world properties
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #Will get plane URDF using taht path
        p.setGravity(0,0,-9.81)
        p.setRealTimeSimulation(0)
        p.setTimeStep(PHYSICS_TIMESTEP, physicsClientId=self.client_id)
        
        # More stable physics solver
        p.setPhysicsEngineParameter(
            numSolverIterations=50,
            enableConeFriction=0,
            physicsClientId=self.client_id
        )
        
        # Speed optimization: run multiple physics steps per action
        self.sim_steps_per_action = sim_steps_per_action
        self.use_position_control = use_position_control
        self.demo_mode = demo_mode  # Relaxed termination for visualization
        self.dt = PHYSICS_TIMESTEP * sim_steps_per_action
        
        #Loading the ground plane with friction
        self.plane_id=p.loadURDF("plane.urdf")
        p.changeDynamics(self.plane_id, -1,
            lateralFriction=1.5,
            spinningFriction=0.3,
            rollingFriction=0.1,
            physicsClientId=self.client_id)

        # Robot Loading, placeholder for now
        self.robot_id=None
        self.actuated_joint_indices=None
        self.action_dim=0
        self.state_dim=None
        print("Environment initialized")




        # Start robot in standing position facing +X direction
        self.start_pos=[0,0,0.25]
        self.start_orn = p.getQuaternionFromEuler([0,0,math.pi]) #Face +X direction
        
        # Default standing joint angles (shoulder, hip, knee for each leg)
        self.rest_pose = np.array([
            0.0, -0.89, 1.30,   # front left
            0.0, -0.89, 1.30,   # front right  
            0.0, -0.89, 1.30,   # rear left
            0.0, -0.89, 1.30    # rear right
        ], dtype=np.float32)
        
        #get the absolute path to the robot URDF (Unified Robotics description format)
        urdf_root= os.path.dirname(os.path.abspath(__file__))
        self.robot_id=p.loadURDF( #Not downloaded yet so this won't work
            os.path.join(urdf_root,"../../assets/rex.urdf"), #Chosen open source urdf in place for real Boston Dynamics's Spot closed source URDF
            self.start_pos,
            self.start_orn,
            useFixedBase=False,#Otherwise the robot is stuck to the world
            physicsClientId=self.client_id
        )

        #Initialize the internal variable and identify joints

        self.actuated_joint_names=[
            'motor_front_left_shoulder', 'motor_front_left_leg', 'foot_motor_front_left',
            'motor_front_right_shoulder', 'motor_front_right_leg', 'foot_motor_front_right',
            'motor_rear_left_shoulder', 'motor_rear_left_leg', 'foot_motor_rear_left',
            'motor_rear_right_shoulder', 'motor_rear_right_leg', 'foot_motor_rear_right'
            ]


        self.actuated_joint_indices=self._identify_actuated_joints()

        if len(self.actuated_joint_indices) !=12:
            raise Exception("Error, did not find 12 actuated joints")
        
        self.action_dim=len(self.actuated_joint_indices)
        
        # Set friction on feet for better ground contact
        self._setup_foot_friction()
        
        # Motor controller and gait generator for smooth walking
        self.motor = MotorController(12)
        self.gait = GaitGenerator("trot")

        #tracking stuff for adaptive rewards
        self.timestep=0
        self.total_episodes=0
        self.episode_time = 0.0
        self.sim_time = 0.0
        
        # Distance tracking
        self.max_x_episode = 0.0
        self.max_x_ever = 0.0
        self.last_x = 0.0
        
        # Reward checkpoints
        self._reached_1m = False
        self._reached_2m = False
        self._reached_3m = False
        self._reached_5m = False
        self._reached_10m = False
        self._reached_25m = False
        self._reached_50m = False
        self._reached_100m = False
        
        # Death ray (forces robot to keep moving forward)
        self.ray_x = -1.0
        self.ray_speed = 0.08
        self.ray_delay = 10.0  # Seconds before ray starts
        
        self.ray_visual = None
        if gui:
            self._create_ray_visual()

        print (f"Environment initialized. Action dim={self.action_dim}, position control={use_position_control}")

    def close(self):
        """Disconnects the Pybullet session"""
        p.disconnect(self.client_id)

    def _identify_actuated_joints(self):
        """To find the joints indices for the 12 joints based on their names"""
        joint_indices=[]
      
        joint_name_to_index={}
        num_joints=p.getNumJoints(self.robot_id,physicsClientId=self.client_id)
        
        for i in range(num_joints):
            joint_info=p.getJointInfo(self.robot_id,i,physicsClientId=self.client_id)
            #Decode the joint name
            joint_name=joint_info[1].decode('utf-8')
            joint_name_to_index[joint_name]=i

        for name in self.actuated_joint_names:
            if name not in joint_name_to_index:
                print(f"Warning:Joint {name} not found in URDF")
                continue
            joint_indices.append(joint_name_to_index[name])
        
        return joint_indices
    
    def _setup_foot_friction(self):
        """High friction on feet for good ground contact"""
        for i in range(p.getNumJoints(self.robot_id, physicsClientId=self.client_id)):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client_id)
            link_name = info[12].decode('utf-8')
            if 'foot' in link_name.lower():
                p.changeDynamics(self.robot_id, i,
                    lateralFriction=1.5,
                    spinningFriction=0.3,
                    rollingFriction=0.1,
                    restitution=0.0,
                    physicsClientId=self.client_id)
    
    def _create_ray_visual(self):
        """Red wall to show death ray position"""
        viz = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.02, 2.0, 1.0],
            rgbaColor=[1, 0, 0, 0.7],
            physicsClientId=self.client_id
        )
        self.ray_visual = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=viz,
            basePosition=[-1.0, 0, 0.5],
            physicsClientId=self.client_id
        )
    
    def _update_ray_visual(self):
        if self.ray_visual is not None:
            p.resetBasePositionAndOrientation(
                self.ray_visual,
                [self.ray_x, 0, 0.5],
                [0, 0, 0, 1],
                physicsClientId=self.client_id
            )
    
    def get_observation(self):
        """Get the complete state representation for the agent.
        State includes:
        Base position (z-height only) - 1 dim
        Base orientation (quaternion) - 4 dims
        Base linear velocity (x,y,z) - 3 dims
        Base angular velocity (x,y,z) - 3 dims
        Joint positions (12) -12 dims
        Joint velocities (12) - 12 dims
        Gait phase sin/cos (2) - for gait synchronization
        Total: 1 + 4 + 3 + 3 + 12 + 12 + 2 = 37 dimensions"""
    
        # p.getBasePositionAndOrientation returns (pos_vec3, orn_quat4)

        base_pos, base_orn_quat=p.getBasePositionAndOrientation(self.robot_id,physicsClientId=self.client_id)


        #Get linvelvec3, ang velvec3

        base_vel_lin, base_vel_ang= p.getBaseVelocity(self.robot_id, physicsClientId=self.client_id)

        # Get joints states

        joint_states=p.getJointStates(self.robot_id, self.actuated_joint_indices, physicsClientId=self.client_id)
        joint_pos=[s[0] for s in joint_states]
        joint_vel=[s[1] for s in joint_states]
        
        # Encode gait phase as sin/cos for continuity
        phase_sin = math.sin(2 * math.pi * self.gait.phase)
        phase_cos = math.cos(2 * math.pi * self.gait.phase)


        # Concatenate all observations

        obs_list= ([base_pos[2]]+
                   list(base_orn_quat)+
                   list(base_vel_lin)+
                   list(base_vel_ang)+
                   list(joint_pos)+
                   list(joint_vel)+
                   [phase_sin, phase_cos])
        
        #Set state dim on firsst call
        if self.state_dim is None:
            self.state_dim=len(obs_list)
            print(f"Observation space dimension set to :{self.state_dim}")

        return np.array(obs_list, dtype = np.float32)
    
    def reset(self, randomize=True):
        """ Resets the environment to the starting state """
        
        if randomize:
            # Small random perturbations for robustness
            random_height = self.start_pos[2] + np.random.uniform(-0.02,0.03)
            random_pos=[self.start_pos[0], self.start_pos[1], random_height]
            
            random_roll = np.random.uniform(-0.05,0.05) 
            random_pitch= np.random.uniform(-0.05,0.05)
            random_orn = p.getQuaternionFromEuler([random_roll,random_pitch,math.pi])
        else:
            random_pos = self.start_pos
            random_orn = self.start_orn
        
        #Reset robot base
        p.resetBasePositionAndOrientation(
            self.robot_id, random_pos, random_orn, physicsClientId=self.client_id
        )

        p.resetBaseVelocity(self.robot_id,[0,0,0],[0,0,0],physicsClientId=self.client_id)

        #Reset joints to rest pose with small noise
        for idx, jid in enumerate(self.actuated_joint_indices):
            noise = np.random.uniform(-0.05,0.05) if randomize else 0
            p.resetJointState(self.robot_id, jid, 
                targetValue=self.rest_pose[idx] + noise, 
                targetVelocity=0, 
                physicsClientId=self.client_id)

        # Reset tracking
        self.timestep=0 
        self.total_episodes+=1
        self.episode_time = 0.0
        self.sim_time = 0.0
        
        self.max_x_episode = 0.0
        self.last_x = random_pos[0]
        
        self._reached_1m = False
        self._reached_2m = False
        self._reached_3m = False
        self._reached_5m = False
        self._reached_10m = False
        self._reached_25m = False
        self._reached_50m = False
        self._reached_100m = False
        
        # Reset gait generator
        self.gait.phase = np.random.uniform(0, 1) if randomize else 0.0
        self.gait.last_time = 0.0
        
        # Reset death ray
        self.ray_x = -2.0
        
        # Curriculum: ray gets faster over time
        if self.total_episodes > 1000:
            self.ray_speed = 0.05
        if self.total_episodes > 5000:
            self.ray_speed = 0.08
        
        self._update_ray_visual()

        #Get initial observation
        return self.get_observation()
    
    def _apply_action(self,action):
        """
        Converts neural network output to motor commands.
        Action is residual adjustment to reference gait (action=0 means follow gait exactly).
        """
        action = np.clip(action, -1.0, 1.0)
        
        # Get reference trajectory from gait generator
        # Target velocity is ~0.5 m/s forward
        foot_pos = self.gait.get_foot_positions(self.sim_time, velocity=0.6, period=0.4)
        
        # Convert foot positions to joint angle offsets
        # Simplified mapping: hip joint controls forward/back, knee controls height
        gait_offset = np.zeros(12, dtype=np.float32)
        for leg in range(4):
            base_idx = leg * 3
            gait_offset[base_idx + 0] = 0.0  # shoulder stays neutral
            gait_offset[base_idx + 1] = foot_pos[leg, 0] * 2.0  # hip for forward/back
            gait_offset[base_idx + 2] = foot_pos[leg, 1] * 3.0  # knee for lift
        
        # Action is residual on top of gait
        residual_scale = 0.15
        target_pos = self.rest_pose + gait_offset + action * residual_scale
        
        if self.use_position_control:
            # Position control mode (more stable)
            p.setJointMotorControlArray(
                self.robot_id,
                self.actuated_joint_indices,
                p.POSITION_CONTROL,
                targetPositions=target_pos.tolist(),
                forces=[MAX_TORQUE] * 12,
                positionGains=[MOTOR_KP] * 12,
                velocityGains=[MOTOR_KD] * 12,
                physicsClientId=self.client_id
            )
        else:
            # Torque control mode (more realistic but harder to learn)
            joint_states = p.getJointStates(self.robot_id, self.actuated_joint_indices, physicsClientId=self.client_id)
            curr_pos = np.array([s[0] for s in joint_states])
            curr_vel = np.array([s[1] for s in joint_states])
            
            torques = self.motor.compute_torque(target_pos, curr_pos, curr_vel, curr_vel)
            
            # Disable default motors first
            p.setJointMotorControlArray(
                bodyUniqueId=self.robot_id,
                jointIndices=self.actuated_joint_indices,
                controlMode=p.VELOCITY_CONTROL,
                forces=np.zeros(self.action_dim),
                physicsClientId=self.client_id
            )

            p.setJointMotorControlArray(
                bodyUniqueId=self.robot_id,
                jointIndices=self.actuated_joint_indices,
                controlMode=p.TORQUE_CONTROL,
                forces=torques,
                physicsClientId=self.client_id
            )



    def step(self,action):
        """Executes one step in the simulation given the action.
        Runs multiple physics steps per action for faster training."""

        self._apply_action(action)

        # Run multiple physics steps for more stable simulation and faster training
        for _ in range(self.sim_steps_per_action):
            p.stepSimulation(physicsClientId=self.client_id)
        
        self.timestep += 1
        self.episode_time += self.dt
        self.sim_time += self.dt
        
        # Update death ray position after delay
        if self.episode_time > self.ray_delay:
            self.ray_x += self.ray_speed * self.dt
            self._update_ray_visual()

        obs=self.get_observation()
        reward=self.get_reward()
        done=self.is_done()
        
        # Track distance
        pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
        if pos[0] > self.max_x_episode:
            self.max_x_episode = pos[0]
        
        new_record = False
        if self.max_x_episode > self.max_x_ever:
            self.max_x_ever = self.max_x_episode
            new_record = True

        info={
            'distance': self.max_x_episode,
            'max_distance_ever': self.max_x_ever,
            'new_distance_record': new_record,
            'death_ray_x': self.ray_x,
            'ahead_of_ray': pos[0] - self.ray_x
        }

        return obs, reward, done, info
            

    def get_reward(self):
        """Reward function focused on forward locomotion.
        Main components: forward velocity, progress bonus, penalties for drift and tilt."""

        base_pos, base_orn_quat=p.getBasePositionAndOrientation(self.robot_id,physicsClientId=self.client_id)
        base_vel_lin, base_vel_ang = p.getBaseVelocity(self.robot_id, physicsClientId=self.client_id)

        x = base_pos[0]
        y = base_pos[1]
        x_velocity = base_vel_lin[0]
        
        # Forward progress since last step
        delta_x = x - self.last_x
        self.last_x = x
        
        # === MAIN REWARDS ===
        
        # Reward forward velocity (the main goal)
        vel_reward = 8.0 * x_velocity
        
        # Bonus for actual progress (not just velocity)
        progress_reward = 15.0 * max(0, delta_x)
        
        # Small alive bonus (reduced to not dominate)
        alive_bonus = 0.2
        
        # Bonus for staying ahead of death ray (pressure to move!)
        if self.episode_time > self.ray_delay:
            ahead_of_ray = x - self.ray_x
            if ahead_of_ray > 0:
                ray_bonus = 0.5 * min(ahead_of_ray, 2.0)
            else:
                ray_bonus = 2.0 * ahead_of_ray  # Negative when behind
        else:
            ray_bonus = 0.0
        
        # === PENALTIES ===
        
        # Drifting sideways is bad
        drift_penalty = 0.3 * abs(y)
        
        # Tilting is bad (check if robot is upright)
        rot_mat = p.getMatrixFromQuaternion(base_orn_quat)
        up_vec = rot_mat[6:9]  # z-axis in world frame
        tilt_penalty = 0.3 * (1.0 - up_vec[2])
        
        # Energy penalty (discourages jerky movements)
        joint_states = p.getJointStates(self.robot_id, self.actuated_joint_indices, physicsClientId=self.client_id)
        torques = np.array([s[3] for s in joint_states])
        velocities = np.array([s[1] for s in joint_states])
        energy = np.abs(np.dot(torques, velocities)) * self.dt
        energy_penalty = 0.0003 * energy
        
        # Combine all components
        reward = alive_bonus + vel_reward + progress_reward + ray_bonus - drift_penalty - tilt_penalty - energy_penalty

        # === DISTANCE MILESTONES ===
        if x > 1.0 and not self._reached_1m:
            reward += 15.0
            self._reached_1m = True
            print("  >> 1m checkpoint!")
        
        if x > 2.0 and not self._reached_2m:
            reward += 20.0
            self._reached_2m = True
            print("  >> 2m checkpoint!")
        
        if x > 3.0 and not self._reached_3m:
            reward += 25.0
            self._reached_3m = True
            print("  >> 3m checkpoint!")
            
        if x > 5.0 and not self._reached_5m:
            reward += 35.0
            self._reached_5m = True
            print("  >> 5m checkpoint!")
            
        if x > 10.0 and not self._reached_10m:
            reward += 50.0
            self._reached_10m = True
            print("  >> 10m checkpoint!")
            
        if x > 25.0 and not self._reached_25m:
            reward += 100.0
            self._reached_25m = True
            print("  >> 25m checkpoint!")
            
        if x > 50.0 and not self._reached_50m:
            reward += 200.0
            self._reached_50m = True
            print("  >> 50m checkpoint!")
            
        if x > 100.0 and not self._reached_100m:
            reward += 500.0
            self._reached_100m = True
            print("  >> 100m GOAL!")

        return reward

    def is_done(self):
        """Checks if the episode is done based on the robot's state."""
        base_pos, base_orn_quat=p.getBasePositionAndOrientation(self.robot_id,physicsClientId=self.client_id)
        
        # Check if robot is upright using rotation matrix
        rot_mat = p.getMatrixFromQuaternion(base_orn_quat)
        up_vec = rot_mat[6:9]
        
        if self.demo_mode:
            # DEMO MODE: very relaxed - only terminate on catastrophic failure
            
            # Completely flipped over (upside down)
            if up_vec[2] < 0.3:
                print("  [Demo] Terminated: Robot flipped over")
                return True
            
            # Robot fell through floor somehow
            if base_pos[2] < 0.03:
                print("  [Demo] Terminated: Robot on ground")
                return True
            
            # No time limit, no death ray in demo mode
            return False
        
        # TRAINING MODE: normal termination conditions
        
        # Robot tilted too much
        if up_vec[2] < 0.7:
            return True
        
        # Too low = fallen
        if base_pos[2] < 0.08:
            return True
        
        # Drifted too far sideways
        if abs(base_pos[1]) > 1.5:
            return True
        
        # Caught by death ray
        if self.episode_time > self.ray_delay:
            if base_pos[0] < self.ray_x:
                return True
        
        # Max episode length
        if self.timestep > 100000:
            return True
        
        return False
    

            







        






















# Testing the interface
if __name__ == "__main__": #Only launches if file ran indepedently

    print("Testing environment...")

    #Call of the main playground class
    env = MainPlayground(gui=True, use_position_control=True) 

    print(f"\nState dim: {env.state_dim}, Action dim: {env.action_dim}")
    print("Robot should walk forward with zero action (following reference gait)")
    print("Red death ray starts after delay\n")

    obs = env.reset(randomize=False)
    zero_action = np.zeros(env.action_dim)
    
    step = 0
    try:
        while True:
            obs, reward, done, info = env.step(zero_action)
            step += 1
            
            if step % 60 == 0:
                t = env.episode_time
                ahead = info['ahead_of_ray']
                if t < env.ray_delay:
                    status = f"grace ({env.ray_delay - t:.1f}s left)"
                else:
                    status = "SAFE" if ahead > 0 else "DANGER"
                print(f"t={t:.1f}s dist={info['distance']:.2f}m ray={ahead:.2f}m [{status}]")
            
            if done:
                print(f"Episode done. Distance: {info['distance']:.2f}m")
                obs = env.reset(randomize=False)
                step = 0
            
            time.sleep(1/60) #60fps
            
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        env.close()



