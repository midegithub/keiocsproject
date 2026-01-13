"""
SIMPLE STANDING DEMO - Pure Puppet Control
No RL, no episodes, no randomization.
Just make the robot stand in one pose.
"""

import pybullet as p
import pybullet_data
import time
import os

def main():
    print("="*70)
    print("PURE STANDING DEMO - Deterministic Puppet Control")
    print("="*70)
    print("Robot will spawn and stand in a fixed pose")
    print("NO randomization, NO episodes, NO RL")
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
    
    # Load robot
    urdf_path = os.path.join(os.path.dirname(__file__), "../assets/rex.urdf")
    robot_id = p.loadURDF(
        urdf_path,
        [0, 0, 0.28],  # Fixed spawn position
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
    
    # STANDING POSE - matches the training rest pose used by MainPlayground
    # Structure: [shoulder, hip, knee] × 4 legs
    standing_pose = [
        0.0, -0.89, 1.30,   # URDF front left (becomes world rear after 180°)
        0.0, -0.89, 1.30,   # URDF front right (becomes world rear)
        0.0, -0.89, 1.30,   # URDF rear left (becomes world FRONT x+)
        0.0, -0.89, 1.30    # URDF rear right (becomes world FRONT x+)
    ]
    
    print("\nStanding pose joint angles:")
    for i, angle in enumerate(standing_pose):
        print(f"  Joint {i}: {angle:.2f} rad")
    
    # Set joints to standing pose BEFORE simulation
    for i, joint_idx in enumerate(joint_indices):
        p.resetJointState(robot_id, joint_idx, standing_pose[i], 0)
    
    print("\nHolding standing pose...")
    step_count = 0
    
    try:
        while True:
            # Apply standing pose with position control
            for i, joint_idx in enumerate(joint_indices):
                p.setJointMotorControl2(
                    robot_id,
                    joint_idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=standing_pose[i],
                    force=50.0,
                    positionGain=0.3,
                    velocityGain=1.0
                )
            
            # Step simulation
            p.stepSimulation()
            
            # Print status occasionally
            if step_count % 240 == 0:  # Every second
                base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
                base_vel, _ = p.getBaseVelocity(robot_id)
                print(f"Step {step_count:5d} | Height: {base_pos[2]:.3f}m | "
                      f"Pos: [{base_pos[0]:+.2f}, {base_pos[1]:+.2f}] | "
                      f"Vel: {base_vel[0]:.3f} m/s")
            
            time.sleep(1./240.)  # Real-time
            step_count += 1
            
    except KeyboardInterrupt:
        print("\n\nStopped by user")
        base_pos, _ = p.getBasePositionAndOrientation(robot_id)
        print(f"Final position: [{base_pos[0]:.2f}, {base_pos[1]:.2f}, {base_pos[2]:.3f}]")
    finally:
        p.disconnect()
        print("Demo ended")

if __name__ == "__main__":
    main()

