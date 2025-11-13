#Import modules for environment implementation
import pybullet as p
import pybullet_data
import numpy as np
import time

class main_playground:
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
        self.actuated_joint_indices=0
        self.action_dim=0
        self.state_dim=None
        print("Environment initialized")

        def close(self):
            """Disconnects the Pybullet session"""
            p.disconnect(self.client_id)


# Testing the interface
if __name__ == "__main__": #Only launches if file ran indepedently

    print("Launching")

    #Call of the main playground class
    env = main_playground(gui=True) 

    print("\nPybullet environment createed")

    try:
        while True:
            time.sleep(1/60) #60fps
    except KeyboardInterrupt:
        print("\nClosed the simulation environment")
    finally:
        env.close()



