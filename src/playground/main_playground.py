#Import modules for environment implementation
import pybullet as p
import pybullet_data
import numpy as np
import time
import os

class MainPlayground:
    def __init__(self, gui=True):
        #Start Physics server, p.GUI opens a visible window whereas p.DIRECT runs headless
        if gui:
            self.client_id=p.connect(p.GUI)
            #slider for debugging GUI
            p.configureDebugVisualizer(p.COV_ENABLE_GUI,1)
            p.resetDebugVisualizerCamera(cameraDistance=1.5,cameraYaw=20,cameraPitch=-30,cameraTargetPosition=()) #No target postition
        else :
            self.client_id=p.connect(p.DIRECT)

        #Setup data paths and world properties
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #Will get plane URDF using taht path
        p.setGravity(0,0,-9.81)
        p.setRealTimeSimulation(0) 
        
        #Loading the ground plane
        self.plane_id=p.loadURDF("plane.urdf")

        # Robot Loading, placeholder for now
        self.robot_id=None
        self.actuated_joint_indices=None
        self.action_dim=0
        self.state_dim=None
        print("Environment initialized")




        self.start_pos=[0,0,0.35] #Lower starting height so robot doesnt start floating 
        self.start_orn = p.getQuaternionFromEuler([0,0,0]) #No rotation
        
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

        #tracking stuff for adaptive rewards
        self.timestep=0
        self.total_episodes=0

        print (f"Environment initialized. Action dim={self.action_dim}")

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
    
    def get_observation(self):
        """Get the complete state representation for the agent.
        State includes:
        Base position (z-height only) - 1 dim
        Base orientation (quaternion) - 4 dims
        Base linear velocity (x,y,z) - 3 dims
        Base angular velocity (x,y,z) - 3 dims
        Joint positions (12) -12 dims
        Joint velocities (12) - 12 dims
        Total: 1 + 4 + 3 + 3 + 12 + 12 = 35 dimensions"""
    
        # p.getBasePositionAndOrientation returns (pos_vec3, orn_quat4)

        base_pos, base_orn_quat=p.getBasePositionAndOrientation(self.robot_id,physicsClientId=self.client_id)


        #Get linvelvec3, ang velvec3

        base_vel_lin, base_vel_ang= p.getBaseVelocity(self.robot_id, physicsClientId=self.client_id)

        # Get joints states

        joint_states=p.getJointStates(self.robot_id, self.actuated_joint_indices, physicsClientId=self.client_id)
        joint_pos=[s[0] for s in joint_states]
        joint_vel=[s[1] for s in joint_states]


        # Concatenate all observations

        obs_list= ([base_pos[2]]+
                   list(base_orn_quat)+
                   list(base_vel_lin)+
                   list(base_vel_ang)+
                   list(joint_pos)+
                   list(joint_vel))
        
        #Set state dim on firsst call
        if self.state_dim is None:
            self.state_dim=len(obs_list)
            print(f"Observation space dimension set to :{self.state_dim}")

        return np.array(obs_list, dtype = np.float32)
    
    def reset(self):
        """ Resets the environment to the starting state """
        #add some randomness so robot learns to recover from different positions
        random_height = self.start_pos[2] + np.random.uniform(-0.03,0.03)
        random_pos=[self.start_pos[0], self.start_pos[1], random_height]
        
        #small random rotation so it doesnt always start perfectly level
        random_roll = np.random.uniform(-0.05,0.05) 
        random_pitch= np.random.uniform(-0.05,0.05)
        random_orn = p.getQuaternionFromEuler([random_roll,random_pitch,0])
        
        #Reset robot base
        p.resetBasePositionAndOrientation(
            self.robot_id, random_pos, random_orn, physicsClientId=self.client_id
        )

        p.resetBaseVelocity(self.robot_id,None,None,physicsClientId=self.client_id)

        #randomize joint positions slightly so robot doesnt memorize one starting config
        for i in self.actuated_joint_indices:
            random_joint_pos= np.random.uniform(-0.1,0.1)
            p.resetJointState(self.robot_id,i, targetValue=random_joint_pos, targetVelocity=0, physicsClientId=self.client_id)

        self.reward_checkpoint_1=False
        self.reward_checkpoint_2=False
        self.timestep=0 
        self.total_episodes+=1

        #Get initial observation
        return self.get_observation()
    
    def _apply_action(self,action):
        """
        Applies the given 12 torque values to the robot's actuated joints.
        This is the core of TORQUE CONTROL.
        """

        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=self.actuated_joint_indices,
            controlMode=p.VELOCITY_CONTROL,
            forces=np.zeros(self.action_dim), #set forces to zero
            physicsClientId=self.client_id
        )

        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=self.actuated_joint_indices,
            controlMode=p.TORQUE_CONTROL,
            forces=action,
            physicsClientId=self.client_id
        )



    def step(self,action):
        """executes one step in the simulation given the aciton"""

        self._apply_action(action)

        p.stepSimulation(physicsClientId=self.client_id)

        obs=self.get_observation()
        reward=self.get_reward()
        done=self.is_done()

        info={}

        return obs, reward, done, info
            

    def get_reward(self):
        """Computes the reward with automatic curriculum learning built-in.
        No need to manually switch between balancing and walking modes"""

        base_pos, base_orn_quat=p.getBasePositionAndOrientation(self.robot_id,physicsClientId=self.client_id)
        base_vel_lin, base_vel_ang = p.getBaseVelocity(self.robot_id, physicsClientId=self.client_id)
        base_orn_euler = p.getEulerFromQuaternion(base_orn_quat)

        x_velocity = base_vel_lin[0]
        y_velocity = base_vel_lin[1] 
        base_height = base_pos[2]
        roll = base_orn_euler[0]
        pitch = base_orn_euler[1]
        yaw = base_orn_euler[2]

        #adaptive curriculum - automatically shifts focus from balancing to walking
        #early episodes focus on staying upright, later ones focus on moving forward
        balance_weight = max(0.3, 1.0 - self.total_episodes/500.0) #decreases over time
        locomotion_weight = min(1.0, self.total_episodes/500.0) #increases over time

        #survival bonus - just staying alive is good especially early on
        survival_bonus=0.15
        
        #forward movement reward - only reward forward motion not backward
        reward_forward = locomotion_weight * 2.5*max(0,x_velocity)  
        
        #penalize sideways drift - want straight line walking
        penalty_lateral = 0.5*abs(y_velocity)
        
        #height penalty - want to stay at reasonable height
        #using squared penalty so big deviations hurt more
        target_height = 0.35 
        height_diff = abs(base_height - target_height)
        penalty_height = balance_weight * 2.0*(height_diff**2)
        
        #orientation penalties - both pitch AND roll matter
        #squared penalties again for bigger punishment on large tilts
        penalty_pitch = balance_weight * 3.0*(pitch**2)
        penalty_roll = balance_weight * 3.0*(roll**2) 
        penalty_yaw = 0.3*abs(yaw) #dont spin around
        
        #energy efficiency - penalize crazy spinning and jerky movements
        ang_vel_magnitude = np.sqrt(base_vel_ang[0]**2 + base_vel_ang[1]**2 + base_vel_ang[2]**2)
        penalty_energy = 0.05*ang_vel_magnitude
        
        #combine everything
        reward= (survival_bonus + reward_forward 
                 - penalty_height - penalty_pitch - penalty_roll 
                 - penalty_lateral - penalty_yaw - penalty_energy)

        #milestone bonuses for distance traveled
        if base_pos[0]>1.0 and not self.reward_checkpoint_1:
            reward +=8.0
            self.reward_checkpoint_1=True
        if base_pos[0]>2.0 and not self.reward_checkpoint_2:
            reward +=12.0
            self.reward_checkpoint_2=True
        
        self.timestep+=1
        return reward

    def is_done(self):
        """Checks if the episode is done based on the robot's state."""
        base_pos, base_orn_quat=p.getBasePositionAndOrientation(self.robot_id,physicsClientId=self.client_id)
        base_height=base_pos[2]
        base_orn_euler=p.getEulerFromQuaternion(base_orn_quat)
        roll = base_orn_euler[0]
        pitch=base_orn_euler[1]

        #relaxed termination so robot has more time to learn
        if base_height <0.12: #pretty low before we call it quits
            #print("Episode done: Robot fell over")
            return True
        
        #check both pitch and roll - dont want it tipping any direction
        if abs(pitch) > 1.0 or abs(roll)>1.0: #~57 degrees, more forgiving than before
            #print("Episode done: Robot tilted too much")
            return True
        
        #max episode length so it doesnt run forever if it just stands still
        if self.timestep >2000:
            #print("Episode done: max timesteps reached")
            return True
        
        return False
    

            







        






















# Testing the interface
if __name__ == "__main__": #Only launches if file ran indepedently

    print("Launching")

    #Call of the main playground class
    env = MainPlayground(gui=True) 

    print("\nPybullet environment created")

    dummy_action = np.zeros(env.action_dim)

    try:
        while True:
            env.step(dummy_action)
            time.sleep(1/60) #60fps
    except KeyboardInterrupt:
        print("\nClosed the simulation environment")
    finally:
        if p.isConnected(env.client_id):
            print("Closing environment cleanly.")
            env.close()
        else:
            print("GUI window was closed. No disconnect needed.")



