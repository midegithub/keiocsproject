"""
DETERMINISTIC WALKING DEMO - Pure Puppet Control
No RL, no randomization, no episodes.
Robot walks using a fixed sinusoidal gait pattern.
"""

import pybullet as p
import pybullet_data
import time
import os
import numpy as np
from playground.main_playground import GaitGenerator

def main():
    print("="*70)
    print("DETERMINISTIC WALKING DEMO - Puppet Control")
    print("="*70)
    print("Robot will walk forward using a fixed gait pattern")
    print("NO randomization, NO episodes, NO RL")
    print("Same behavior every time!")
    print("Press Ctrl+C to stop")
    print("="*70)
    
    # Connect to PyBullet
    client_id = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1./240.)
    
    # Setup camera
    p.resetDebugVisualizerCamera(
        cameraDistance=2.5,
        cameraYaw=30,
        cameraPitch=-20,
        cameraTargetPosition=[0, 0, 0.3]
    )
    
    # Load ground plane
    plane_id = p.loadURDF("plane.urdf")
    
    # Load robot at fixed position
    urdf_path = os.path.join(os.path.dirname(__file__), "../assets/rex.urdf")
    robot_id = p.loadURDF(
        urdf_path,
        [0, 0, 0.30],  # Slightly higher for walking
        p.getQuaternionFromEuler([0, 0, 3.14159]),  # 180° rotation to face +X
        useFixedBase=False
    )
    
    # Find actuated joints
    actuated_joint_names = [
        'motor_front_left_shoulder', 'motor_front_left_leg', 'foot_motor_front_left',
        'motor_front_right_shoulder', 'motor_front_right_leg', 'foot_motor_front_right',
        'motor_rear_left_shoulder', 'motor_rear_left_leg', 'foot_motor_rear_left',
        'motor_rear_right_shoulder', 'motor_rear_right_leg', 'foot_motor_rear_right'
    ]
    
    joint_indices = []
    num_joints = p.getNumJoints(robot_id)
    joint_name_to_index = {}
    
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        joint_name = joint_info[1].decode('utf-8')
        joint_name_to_index[joint_name] = i
    
    for name in actuated_joint_names:
        if name in joint_name_to_index:
            joint_indices.append(joint_name_to_index[name])
    
    print(f"\nFound {len(joint_indices)} actuated joints")
    
    # Initialize to the same rest pose used during PPO training
    standing_pose = [
        0.0, -0.89, 1.30,   # URDF front left (becomes world rear after 180°)
        0.0, -0.89, 1.30,   # URDF front right (becomes world rear)
        0.0, -0.89, 1.30,   # URDF rear left (becomes world FRONT x+)
        0.0, -0.89, 1.30    # URDF rear right (becomes world FRONT x+)
    ]
    for i, joint_idx in enumerate(joint_indices):
        p.resetJointState(robot_id, joint_idx, standing_pose[i], 0)
    
    print("\nStarting deterministic walk...")
    print("Gait: Trotting (diagonal pairs move together)")
    
    gait = GaitGenerator("trot")
    dt = 1./240.
    t = 0.0
    gait_velocity = 0.8
    gait_period = 0.4
    
    step_count = 0
    
    try:
        while True:
            foot_positions = gait.get_foot_positions(t, velocity=gait_velocity, period=gait_period)
            joint_targets = standing_pose.copy()
            for leg in range(4):
                base_idx = leg * 3
                joint_targets[base_idx + 1] += foot_positions[leg, 0] * 2.0
                joint_targets[base_idx + 2] += foot_positions[leg, 1] * 3.0
            
            # Apply joint angles with position control
            for i, joint_idx in enumerate(joint_indices):
                p.setJointMotorControl2(
                    robot_id,
                    joint_idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=joint_targets[i],
                    force=50.0,
                    positionGain=0.3,
                    velocityGain=1.0
                )
            
            # Step simulation
            p.stepSimulation()
            
            # Update camera to follow robot
            if step_count % 50 == 0:
                base_pos, _ = p.getBasePositionAndOrientation(robot_id)
                p.resetDebugVisualizerCamera(
                    cameraDistance=2.5,
                    cameraYaw=30,
                    cameraPitch=-20,
                    cameraTargetPosition=[base_pos[0], base_pos[1], 0.3]
                )
            
            # Print status every second
            if step_count % 240 == 0:
                base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
                base_vel, _ = p.getBaseVelocity(robot_id)
                print(f"Time {t:5.1f}s | Distance: {base_pos[0]:+.2f}m | "
                      f"Height: {base_pos[2]:.3f}m | "
                      f"Vel: {base_vel[0]:+.3f} m/s")
            
            time.sleep(1./240.)  # Real-time
            t += dt
            step_count += 1
            
    except KeyboardInterrupt:
        print("\n\nStopped by user")
        base_pos, _ = p.getBasePositionAndOrientation(robot_id)
        print(f"Total distance traveled: {base_pos[0]:.2f}m")
        print(f"Final position: [{base_pos[0]:.2f}, {base_pos[1]:.2f}, {base_pos[2]:.3f}]")
    finally:
        p.disconnect()
        print("Demo ended")

if __name__ == "__main__":
    main()

