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

import numpy as np

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

FPS = 60

# Gripper endpoints for the A-button toggle: each press flips between fully open and
# fully closed and commands that fixed endpoint, so the servo holds torque there (e.g.
# to grip an object). Range is 0..100 — swap these two if your gripper opens/closes the
# other way. Back CLOSED off from 100 (e.g. 90) if the servo stalls/overheats.
GRIPPER_OPEN_POS = 0.0
GRIPPER_CLOSED_POS = 100.0


class YawFreeKinematics(RobotKinematics):
    """Kinematics that ignores end-effector yaw (rotation about world vertical).

    The stock solver uses a single 6-DOF frame task, which on the 5-DOF SO-101 is
    over-constrained: position and orientation fight, and orientation creeps.

    Here we replace it with an exactly-determined 5-DOF task:
      * a 3-DOF position task, and
      * a 2-DOF orientation task masked to world X/Y (holds pitch & roll = "level"),
        leaving world-Z yaw unconstrained.
    5 constraints on 5 joints -> square -> position and "level" are solved with ~zero
    residual, and the un-achievable yaw is simply left free instead of fought. Ideal
    for keeping an object flat while moving it in x/y/z (yaw will drift as the arm
    repositions, which is unavoidable on a 5-DOF arm).
    """

    def __init__(self, urdf_path, target_frame_name="gripper_frame_link", joint_names=None):
        super().__init__(urdf_path, target_frame_name, joint_names)
        # Drop the default 6-DOF frame task and build the split position+orientation tasks.
        self.solver.remove_task(self.tip_frame)
        self.position_task = self.solver.add_position_task(self.target_frame_name, np.zeros(3))
        self.position_task.configure("position", "soft", 1.0)
        self.orientation_task = self.solver.add_orientation_task(self.target_frame_name, np.eye(3))
        self.orientation_task.configure("orientation", "soft", 1.0)
        # Express the mask in the world frame and constrain only X/Y rotation (drop yaw/Z).
        self.orientation_task.mask.R_custom_world = np.eye(3)
        self.orientation_task.mask.set_axises("xy")

    def inverse_kinematics(self, current_joint_pos, desired_ee_pose, **_ignored_weights):
        # Seed the solver with the current joints, set targets, solve once (per frame).
        current_joint_rad = np.deg2rad(current_joint_pos[: len(self.joint_names)])
        for i, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, current_joint_rad[i])

        self.position_task.target_world = desired_ee_pose[:3, 3]
        self.orientation_task.R_world_frame = desired_ee_pose[:3, :3]

        self.solver.solve(True)
        self.robot.update_kinematics()

        joint_pos_deg = np.rad2deg([self.robot.get_joint(n) for n in self.joint_names])

        # Preserve trailing values (e.g. gripper) not solved by IK.
        if len(current_joint_pos) > len(self.joint_names):
            result = np.zeros_like(current_joint_pos)
            result[: len(self.joint_names)] = joint_pos_deg
            result[len(self.joint_names) :] = current_joint_pos[len(self.joint_names) :]
            return result
        return joint_pos_deg


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
    # YawFreeKinematics: holds the gripper level (pitch/roll) and lets yaw float, so
    # an object stays flat while moving in x/y/z. See the class docstring.
    kinematics_solver = YawFreeKinematics(
        urdf_path="./SO101/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=list(robot.bus.motors.keys()),
    )

    # Build pipeline to convert phone action to ee pose action to joint action
    phone_to_robot_joints_processor = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[
            MapPhoneActionToRobotAction(platform=teleop_config.phone_os),
            EEReferenceAndDelta(
                kinematics=kinematics_solver,
                end_effector_step_sizes={"x": 0.7, "y": 0.5, "z": 0.5},
                motor_names=list(robot.bus.motors.keys()),
                use_latched_reference=True,
            ),
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [0.8, 0.8, 0.9]},
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

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))


if __name__ == "__main__":
    main()
