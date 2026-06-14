# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specif

import time

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)
from lerobot.teleoperators.phone import Phone, PhoneConfig
from lerobot.teleoperators.phone.config_phone import PhoneOS
from lerobot.teleoperators.phone.phone_processor import MapPhoneActionToRobotAction
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 60

# Gripper endpoints for the A-button toggle: each press flips between fully open and
# fully closed and commands that fixed endpoint, so the servo holds torque there (e.g.
# to grip an object). Range is 0..100 — swap these two if your gripper opens/closes the
# other way. Back CLOSED off from 100 (e.g. 90) if the servo stalls/overheats.
GRIPPER_OPEN_POS = 0.0
GRIPPER_CLOSED_POS = 100.0


def main():
    # Initialize the robot and teleoperator
    robot_config = SO100FollowerConfig(
        port="/dev/tty.usbmodem5B7B0166391", id="so101", use_degrees=True
    )
    # ANDROID = the teleop WebXR path, which also drives a Meta Quest 3: open the
    # served https://<this-machine-ip>:4443 page in the Quest Browser, enter VR, and
    # use the right Touch controller (trigger = enable/clutch, A button = gripper).
    teleop_config = PhoneConfig(phone_os=PhoneOS.ANDROID)

    # Initialize the robot and teleoperator
    robot = SO100Follower(robot_config)
    teleop_device = Phone(teleop_config)

    # NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo: https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
    kinematics_solver = RobotKinematics(
        urdf_path="./SO101/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=list(robot.bus.motors.keys()),
    )

    # Flip the IK weight ratio to prioritize ORIENTATION over position.
    # Library default is position=1.0, orientation=0.01 (position wins 100:1).
    # Here we invert it: orientation is held ~100x tighter and position becomes the
    # soft DOF that gives. This suppresses angular creep, but expect x/y/z to drift
    # instead, and near the 5-DOF-unreachable yaw the solver will sacrifice position
    # hard to chase orientation. The IK step calls inverse_kinematics() with only the
    # two positional args, so these flipped defaults take effect.
    _orig_ik = kinematics_solver.inverse_kinematics

    def _ik_orientation_priority(
        current_joint_pos, desired_ee_pose, position_weight=1, orientation_weight=0.01
    ):
        return _orig_ik(current_joint_pos, desired_ee_pose, position_weight, orientation_weight)

    kinematics_solver.inverse_kinematics = _ik_orientation_priority

    # Build pipeline to convert phone action to ee pose action to joint action
    phone_to_robot_joints_processor = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[
            MapPhoneActionToRobotAction(platform=teleop_config.phone_os),
            EEReferenceAndDelta(
                kinematics=kinematics_solver,
                end_effector_step_sizes={"x": 0.1, "y": 0.1, "z": 0.1},
                motor_names=list(robot.bus.motors.keys()),
                use_latched_reference=True,
            ),
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [0.8, 0.8, 0.8]},
                max_ee_step_m=0.3,
            ),
            GripperVelocityToJoint(
                speed_factor=20.0,
            ),
            InverseKinematicsEEToJoints(
                kinematics=kinematics_solver,
                motor_names=list(robot.bus.motors.keys()),
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # Connect to the robot and teleoperator
    robot.connect()
    teleop_device.connect()

    # Init rerun viewer
    init_rerun(session_name="phone_so100_teleop")

    if not robot.is_connected or not teleop_device.is_connected:
        raise ValueError("Robot or teleop is not connected!")

    print("Starting teleop loop. Squeeze the controller trigger to teleoperate...")
    gripper_closed = False  # which endpoint the A button holds at
    prev_button_a = False
    while True:
        t0 = time.perf_counter()

        # Get robot observation
        robot_obs = robot.get_observation()
        # Get teleop action
        phone_obs = teleop_device.get_action()

        # Phone -> EE pose -> Joints transition
        joint_action = phone_to_robot_joints_processor((phone_obs, robot_obs))

        # Gripper: the Quest right-controller A button (reservedButtonA on the WebXR
        # stream) toggles fully closed / fully open. We command that fixed endpoint
        # every frame so the servo keeps applying torque (e.g. to grip and hold an
        # object) instead of just tracking its measured position.
        raw_inputs = phone_obs.get("phone.raw_inputs", {}) if phone_obs else {}
        button_a = bool(raw_inputs.get("reservedButtonA", False))
        if button_a and not prev_button_a:  # rising edge -> flip endpoint
            gripper_closed = not gripper_closed
        prev_button_a = button_a
        joint_action["gripper.pos"] = GRIPPER_CLOSED_POS if gripper_closed else GRIPPER_OPEN_POS

        # Send action to robot
        _ = robot.send_action(joint_action)

        # Visualize
        log_rerun_data(observation=phone_obs, action=joint_action)

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))


if __name__ == "__main__":
    main()
