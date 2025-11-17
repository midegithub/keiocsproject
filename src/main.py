#Import modules for running the demo
import numpy as np
import time
import pybullet as p
from playground.main_playground import MainPlayground

# Testing the interface
if __name__ == "__main__": #Only launches if file ran independently

    print("Launching")

    #Call of the main playground class
    env = MainPlayground(gui=True) 

    print("\nPybullet environment created")

    dummy_action = np.zeros(env.action_dim)

    try:
        while True:
            # Check if physics server is still connected
            if not p.isConnected(env.client_id):
                print("\nPhysics server disconnected (GUI window closed)")
                break
            
            try:
                obs, reward, done, info = env.step(dummy_action)
                
                # Reset if episode is done
                if done:
                    print("Resetting environment...")
                    env.reset()
            except p.error:
                # Catch PyBullet error when GUI is closed during step
                print("\nPhysics server disconnected (GUI window closed)")
                break
            
            time.sleep(1/60) #60fps
    except KeyboardInterrupt:
        print("\nClosed the simulation environment")
    finally:
        if p.isConnected(env.client_id):
            print("Closing environment cleanly.")
            env.close()
        else:
            print("GUI window was closed. No disconnect needed.")

