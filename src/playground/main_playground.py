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




        self.start_pos=[0,0,0.61] #known height for Spot Robot (61cm)
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
        #Reset robot base
        p.resetBasePositionAndOrientation(
            self.robot_id, self.start_pos, self.start_orn, physicsClientId=self.client_id
        )

        p.resetBaseVelocity(self.robot_id,None,None,physicsClientId=self.client_id)

        for i in self.actuated_joint_indices:
            p.resetJointState(self.robot_id,i, targetValue=0, targetVelocity=0, physicsClientId=self.client_id)

        self.reward_checkpoint_1=False
        self.reward_checkpoint_2=False

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
        """Computes the reward for the current state."""

        base_pos, base_orn_quat=p.getBasePositionAndOrientation(self.robot_id,physicsClientId=self.client_id)
        base_vel_lin, _ = p.getBaseVelocity(self.robot_id, physicsClientId=self.client_id)
        base_orn_euler = p.getEulerFromQuaternion(base_orn_quat)

        x_velocity = base_vel_lin[0]
        base_height = base_pos[2]
        pitch = base_orn_euler[1]

        #Reward components
        target_height = 0.61  # Target height for the SPot Robot
        target_pitch = 0.0   # Target pitch angle (level)

        reward_forward = 2.0*x_velocity

        penality_pitch = 1.5*abs(pitch - target_pitch)
        penality_height = 1.0*abs(base_height - target_height)

        #TO DO: Add energy consumption penality?

        reward= reward_forward - penality_pitch - penality_height

        if base_pos[0]>1.0 and not self.reward_checkpoint_1:
            reward +=5.0
            self.reward_checkpoint_1=True
        if base_pos[0]>2.0 and not self.reward_checkpoint_2:
            reward +=5.0
            self.reward_checkpoint_2=True
        
        return reward

    def is_done(self):
        pass
            







        






















# Testing the interface
if __name__ == "__main__": #Only launches if file ran indepedently

    print("Launching")

    #Call of the main playground class
    env = MainPlayground(gui=True) 

    print("\nPybullet environment createed")

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



