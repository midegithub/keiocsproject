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

        def close(self):
            """Disconnects the Pybullet session"""
            p.disconnect(self.client_id)


        self.start_pos=[0,0,0.61] #known height for Spot Robot (61cm)
        self.start_orn = p.getQuaternionFromEuler()
        
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
            'front_left_shoulder_joint', 'front_left_thigh_joint', 'front_left_calf_joint',
            'front_right_shoulder_joint', 'front_right_thigh_joint', 'front_right_calf_joint',
            'rear_left_shoulder_joint', 'rear_left_thigh_joint', 'rear_left_calf_joint',
            'rear_right_shoulder_joint', 'rear_right_thigh_joint', 'rear_right_calf_joint'
            ]


        self.actuated_joint_indices=self._identify_actuated_joints()

        if len(self.actuated_joint_indices) !=12:
            raise Exception("Error, did not find 12 actuated joints")
        
        self.action_dim=len(self.actuated_joint_indices)

        print (f"Environment initialized. Action dim={self.action_dim}")

    def _identify_actuated_joints(self):
        """To find the joints indices for the 12 joints based on their names"""
        joint_indices=None
      
        joint_name_to_index={}
        num_joints=p.getNumJoints(self.robot_id,physicsClientId=self.client_id)
        
        for i in range(num_joints):
            joint_info=p.getJointInfo(self.robot_id,i,physicsClientId=self.client_id)
            #Decode the joint name
            joint_name=joint_info.decode('utf-8')
            joint_name_to_index[joint_name]=i

        for name in self.actuated_joint_names:
            if name not in joint_name_to_index:
                print(f"Warning:Joint {name} not found in URDF")
                continue
            joint_indices.append(joint_name_to_index[name])
        
        return joint_indices
    
    



        






















# Testing the interface
if __name__ == "__main__": #Only launches if file ran indepedently

    print("Launching")

    #Call of the main playground class
    env = MainPlayground(gui=True) 

    print("\nPybullet environment createed")

    try:
        while True:
            time.sleep(1/60) #60fps
    except KeyboardInterrupt:
        print("\nClosed the simulation environment")
    finally:
        env.close()



