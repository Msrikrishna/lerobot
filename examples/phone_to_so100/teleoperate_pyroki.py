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
# See the License for the specific language governing permissions and
# limitations under the License.

"""Teleoperation using pyroki (JAX-based) IK instead of placo.

pyroki's vel_cost_collision mode adds a velocity cost that penalises large
joint-angle changes between frames, which eliminates the elbow-flip problem
that plagued the placo solver at joint limits.

Usage:
    python teleoperate_pyroki.py
"""

import time
from pathlib import Path

import numpy as np
import viser

from robokin.pyroki import PyrokiConfig, PyrokiKinematics
from robokin import pyroki_snippets as pks
from robokin.ui.viser_app import ViserRobotUI

from dataclasses import dataclass, field

from lerobot.processor import (
    RobotActionProcessorStep,
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


@dataclass
class NegateTargetRotation(RobotActionProcessorStep):
    """Negate the phone rotation target to correct for pyroki's inverted EE rotation convention."""

    def action(self, action: RobotAction) -> RobotAction:
        wx, wy = action["target_wx"], action["target_wy"]
        action["target_wx"] = action["target_wz"]
        action["target_wy"] = wx #good
        action["target_wz"] = wy
        return action

    def transform_features(self, features):
        return features


@dataclass
class GateEEDeltas(RobotActionProcessorStep):
    """Selectively freeze translation or rotation deltas before they reach EEReferenceAndDelta.

    When disabled, replays the last seen values so the EE holds its position/orientation
    rather than snapping back to the reference or calibration pose.
    Toggled at runtime via .rotation_enabled and .translation_enabled.
    """

    rotation_enabled: bool = True
    translation_enabled: bool = True
    _frozen_x: float = field(default=0.0, init=False)
    _frozen_y: float = field(default=0.0, init=False)
    _frozen_z: float = field(default=0.0, init=False)
    _frozen_wx: float = field(default=0.0, init=False)
    _frozen_wy: float = field(default=0.0, init=False)
    _frozen_wz: float = field(default=0.0, init=False)

    def action(self, action: RobotAction) -> RobotAction:
        if self.translation_enabled:
            self._frozen_x = action["target_x"]
            self._frozen_y = action["target_y"]
            self._frozen_z = action["target_z"]
        else:
            action["target_x"] = self._frozen_x
            action["target_y"] = self._frozen_y
            action["target_z"] = self._frozen_z

        if self.rotation_enabled:
            self._frozen_wx = action["target_wx"]
            self._frozen_wy = action["target_wy"]
            self._frozen_wz = action["target_wz"]
        else:
            action["target_wx"] = self._frozen_wx
            action["target_wy"] = self._frozen_wy
            action["target_wz"] = self._frozen_wz

        return action

    def transform_features(self, features):
        return features

FPS = 60
DT = 1.0 / FPS

GRIPPER_OPEN_POS = 0.0
GRIPPER_CLOSED_POS = 100.0

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

URDF_PATH = str(Path(__file__).parent / "SO101/so101_new_calib.urdf")
EE_LINK = "gripper_frame_link"

# IK solver weights — tune these to adjust tracking behaviour.
IK_POS_WEIGHT = 50.0    # position tracking strength
IK_ORI_WEIGHT = 10.0    # orientation tracking strength (lower = less jerky rotation)
IK_VEL_WEIGHT = 1     # joint velocity smoothing (higher = smoother but more lag)
IK_LIMIT_WEIGHT = 100.0  # joint limit avoidance (keep high)


class PyrokiKinematicsAdapter:
    """Wraps PyrokiKinematics to match the RobotKinematics (placo) interface.

    The lerobot pipeline works in motor-order degrees; pyroki works in its own
    topological joint order in radians.  This adapter handles both conversions.

    IK uses solve_ik_vel_cost (velocity cost only, no collision avoidance).
    Collision avoidance is deliberately excluded: the SO-101 URDF reports
    self-collisions at the neutral/zero configuration, which causes the
    collision-aware solver to fight the IK and produce erratic motion.
    The velocity cost alone is sufficient to prevent elbow flips.
    """

    def __init__(
        self,
        urdf_path: str,
        ee_link_name: str,
        motor_names: list[str],
        dt: float,
        pos_weight: float,
        ori_weight: float,
        vel_weight: float,
        limit_weight: float,
    ):
        self.solver = PyrokiKinematics(
            urdf_path=urdf_path,
            ee_link_name=ee_link_name,
            cfg=PyrokiConfig(dt=dt, mode="basic_ik", vel_weight=vel_weight),
        )
        self._dt = dt
        self._pos_weight = pos_weight
        self._ori_weight = ori_weight
        self._vel_weight = vel_weight
        self._limit_weight = limit_weight
        self.motor_names = list(motor_names)

        # For each pyroki joint index, the corresponding index in motor_names
        # (None if the joint is not in the motor list).
        self._pyroki_to_motor: list[int | None] = []
        motor_idx_map = {n: i for i, n in enumerate(self.motor_names)}
        for pname in self.solver.joint_names:
            self._pyroki_to_motor.append(motor_idx_map.get(pname))

    def warmup(self) -> None:
        self.solver.warmup()  # JIT-compiles basic_ik via solve_goal
        # Also JIT-compile the vel_cost path used by inverse_kinematics.
        _q0 = np.zeros(self.solver.n_joints)
        _t, _w = self.solver._target_from_transform(self.solver.current_pose())
        pks.solve_ik_vel_cost(
            robot=self.solver.robot,
            target_link_name=self.solver.ee_link_name,
            target_wxyz=_w,
            target_position=_t,
            prev_cfg=_q0,
            dt=self._dt,
            pos_weight=self._pos_weight,
            ori_weight=self._ori_weight,
            vel_weight=self._vel_weight,
            limit_weight=self._limit_weight,
        )

    def _motor_deg_to_pyroki_rad(self, motor_deg: np.ndarray) -> np.ndarray:
        q = np.zeros(self.solver.n_joints, dtype=float)
        for pi, mi in enumerate(self._pyroki_to_motor):
            if mi is not None and mi < len(motor_deg):
                q[pi] = np.deg2rad(motor_deg[mi])
        return q

    def _pyroki_rad_to_motor_deg(
        self, q_rad: np.ndarray, passthrough_deg: np.ndarray
    ) -> np.ndarray:
        result = np.array(passthrough_deg, dtype=float)
        for pi, mi in enumerate(self._pyroki_to_motor):
            if mi is not None and mi < len(result):
                result[mi] = np.rad2deg(q_rad[pi])
        return result

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        q_rad = self._motor_deg_to_pyroki_rad(joint_pos_deg)
        return self.solver.fk(q_rad)

    def inverse_kinematics(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray,
    ) -> np.ndarray:
        q_meas_rad = self._motor_deg_to_pyroki_rad(current_joint_pos)
        target_position, target_wxyz = self.solver._target_from_transform(desired_ee_pose)
        q_sol_rad = pks.solve_ik_vel_cost(
            robot=self.solver.robot,
            target_link_name=self.solver.ee_link_name,
            target_wxyz=target_wxyz,
            target_position=target_position,
            prev_cfg=q_meas_rad,
            dt=self._dt,
            pos_weight=self._pos_weight,
            ori_weight=self._ori_weight,
            vel_weight=self._vel_weight,
            limit_weight=self._limit_weight,
        )
        self.solver.current_cfg = np.asarray(q_sol_rad, dtype=float)
        return self._pyroki_rad_to_motor_deg(q_sol_rad, current_joint_pos)


def main() -> None:
    motor_names = MOTOR_NAMES

    teleop_config = PhoneConfig(phone_os=PhoneOS.ANDROID)
    teleop_device = Phone(teleop_config)

    robot_config = SO100FollowerConfig(
        port="/dev/tty.usbmodem5B7B0166391", id="so101_follower_auto", use_degrees=True
    )
    robot = SO100Follower(robot_config)

    kinematics_adapter = PyrokiKinematicsAdapter(
        urdf_path=URDF_PATH,
        ee_link_name=EE_LINK,
        motor_names=motor_names,
        dt=DT,
        pos_weight=IK_POS_WEIGHT,
        ori_weight=IK_ORI_WEIGHT,
        vel_weight=IK_VEL_WEIGHT,
        limit_weight=IK_LIMIT_WEIGHT,
    )

    print("Warming up JAX JIT (this takes ~10s on first run)...")
    kinematics_adapter.warmup()  # compiles basic_ik + vel_cost paths
    print("JAX ready.")

    gate_step = GateEEDeltas()
    phone_to_robot_joints_processor = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[
            MapPhoneActionToRobotAction(platform=teleop_config.phone_os),
            NegateTargetRotation(),
            gate_step,
            EEReferenceAndDelta(
                kinematics=kinematics_adapter,
                end_effector_step_sizes={"x": 1, "y": 1, "z": 1},
                motor_names=motor_names,
                use_latched_reference=True,
            ),
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [2, 2, 2]},
                max_ee_step_m=0.3,
            ),
            GripperVelocityToJoint(
                speed_factor=20.0,
            ),
            InverseKinematicsEEToJoints(
                kinematics=kinematics_adapter,
                motor_names=motor_names,
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    sim_joints: dict[str, float] = {name: 0.0 for name in MOTOR_NAMES}
    _init_deg = np.array([sim_joints[n] for n in MOTOR_NAMES])
    _init_rad = kinematics_adapter._motor_deg_to_pyroki_rad(_init_deg)
    kinematics_adapter.solver.set_joint_state(_init_rad)
    T_init = kinematics_adapter.solver.current_pose()

    viser_server = viser.ViserServer()
    viser_server.scene.set_up_direction("+z")
    ui = ViserRobotUI(
        server=viser_server,
        urdf=kinematics_adapter.solver.urdf,
        solver_joint_names=kinematics_adapter.solver.joint_names,
        gripper_joint_name="gripper",
    )
    ui.build(initial_q=_init_rad, initial_T=T_init, enable_gizmo=False)
    print("Viser viewer: http://localhost:8080")

    robot.connect()
    if not robot.is_connected:
        raise ValueError("Robot is not connected!")
    _init_obs = robot.get_observation()
    for name in MOTOR_NAMES:
        sim_joints[name] = float(_init_obs.get(f"{name}.pos", 0.0))
    _real_rad = kinematics_adapter._motor_deg_to_pyroki_rad(
        np.array([sim_joints[n] for n in MOTOR_NAMES])
    )
    kinematics_adapter.solver.set_joint_state(_real_rad)
    teleop_device.connect()

    print("Starting pyroki teleop. Squeeze the controller trigger to teleoperate...")

    prev_button_a = False
    prev_button_b = False
    step = 0
    try:
        while True:
            t0 = time.perf_counter()

            robot_obs = robot.get_observation()
            for name in MOTOR_NAMES:
                sim_joints[name] = float(robot_obs.get(f"{name}.pos", 0.0))

            phone_obs = teleop_device.get_action()

            if step % FPS == 0:
                msg = getattr(getattr(teleop_device, "_phone_impl", None), "_latest_message", None) or {}
                print(f"[webxr] device={msg.get('device')} | {msg.get('debug')}")
            step += 1

            try:
                joint_action = phone_to_robot_joints_processor((phone_obs, robot_obs))
            except Exception:
                joint_action = None

            raw_inputs = phone_obs.get("phone.raw_inputs", {}) if phone_obs else {}
            button_a = bool(raw_inputs.get("reservedButtonA", False))
            button_b = bool(raw_inputs.get("reservedButtonB", False))
            if button_a and not prev_button_a:
                gate_step.rotation_enabled = not gate_step.rotation_enabled
                print(f"Rotation: {'ON' if gate_step.rotation_enabled else 'OFF'}")
            if button_b and not prev_button_b:
                gate_step.translation_enabled = not gate_step.translation_enabled
                print(f"Translation: {'ON' if gate_step.translation_enabled else 'OFF'}")
            prev_button_a = button_a
            prev_button_b = button_b

            if joint_action is not None:
                joint_action["gripper.pos"] = GRIPPER_OPEN_POS

                for name in MOTOR_NAMES:
                    key = f"{name}.pos"
                    if key in joint_action:
                        sim_joints[name] = float(joint_action[key])

                robot.send_action(joint_action)

            # Update 3-D viewer with the IK solution computed this frame.
            ui.sync_from_solver(kinematics_adapter.solver, move_gizmo=False)

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    finally:
        print("Releasing servos and disconnecting...")
        robot.disconnect()
        teleop_device.disconnect()


if __name__ == "__main__":
    main()
