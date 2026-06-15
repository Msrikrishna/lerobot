#!/usr/bin/env python

# Teleoperate SO-101 with phone/Quest and visualize live in 3D via viser.
# Open http://localhost:8080 in a browser after running.
#
# Usage (simulation only, no hardware):
#   python visualize_so101.py --no-robot
#
# Usage (with physical robot — also drives the real arm):
#   python visualize_so101.py

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import viser
from viser.extras import ViserUrdf

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
from lerobot.utils.rotation import Rotation

URDF_PATH = Path(__file__).parent / "SO101/so101_new_calib.urdf"
PORT = "/dev/tty.usbmodem5B7B0166391"
ROBOT_ID = "so101_follower_auto"
FPS = 30

GRIPPER_OPEN_POS = 0.0
GRIPPER_CLOSED_POS = 100.0
TRIGGER_DEADZONE = 0.05

# Motor names in bus order (matches kinematics joint_names)
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# Actuated joint order that yourdfpy / ViserUrdf expects for update_cfg()
URDF_JOINT_ORDER = ["gripper", "wrist_roll", "wrist_flex", "elbow_flex", "shoulder_lift", "shoulder_pan"]

TRAIL_LEN = 300


def make_obs(joint_deg: dict[str, float]) -> RobotObservation:
    return {f"{name}.pos": joint_deg[name] for name in MOTOR_NAMES}


def main(connect_robot: bool = True) -> None:
    # ── Viser ────────────────────────────────────────────────────────────────
    server = viser.ViserServer(port=8080)
    server.scene.set_up_direction("+z")
    urdf_vis = ViserUrdf(server, URDF_PATH, root_node_name="/robot")

    # EE trail: point cloud that grows up to TRAIL_LEN points
    trail_pts: list[np.ndarray] = []
    trail_handle = server.scene.add_point_cloud(
        "/ee_trail", points=np.zeros((1, 3)), colors=np.array([[0.2, 0.8, 1.0]]), point_size=0.005
    )
    ee_sphere = server.scene.add_icosphere("/ee_pos", radius=0.012, color=(1.0, 0.4, 0.0))

    # ── Kinematics ───────────────────────────────────────────────────────────
    kinematics_solver = RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name="gripper_frame_link",
        joint_names=MOTOR_NAMES,
    )

    # IK wrapper: orientation priority + CSV logging for singularity analysis
    _ik_n = len(kinematics_solver.joint_names)
    _ik_log_file = open("ik_log.csv", "w", newline="")
    _ik_log = csv.writer(_ik_log_file)
    _ik_log.writerow(["t", "tx", "ty", "tz", "trx", "try", "trz",
                      "pos_err", "rot_err", "dseed", "dprev", "seed", "sol"])
    _ik_t0 = time.perf_counter()
    _ik_prev = {"q": None}
    _orig_ik = kinematics_solver.inverse_kinematics

    def _ik_logged(current_joint_pos, desired_ee_pose, position_weight=1, orientation_weight=0.01):
        sol = _orig_ik(current_joint_pos, desired_ee_pose, position_weight, orientation_weight)
        try:
            seed = np.asarray(current_joint_pos[:_ik_n], dtype=float)
            q = np.asarray(sol[:_ik_n], dtype=float)
            tpos = np.asarray(desired_ee_pose[:3, 3], dtype=float)
            r_t = Rotation.from_matrix(desired_ee_pose[:3, :3])
            trot = r_t.as_rotvec()
            t_fk = kinematics_solver.forward_kinematics(sol)
            pos_err = float(np.linalg.norm(t_fk[:3, 3] - tpos))
            rot_err = float(np.linalg.norm((Rotation.from_matrix(t_fk[:3, :3]) * r_t.inv()).as_rotvec()))
            dseed = float(np.linalg.norm(q - seed))
            dprev = 0.0 if _ik_prev["q"] is None else float(np.linalg.norm(q - _ik_prev["q"]))
            _ik_prev["q"] = q
            _ik_log.writerow([f"{time.perf_counter() - _ik_t0:.4f}",
                               f"{tpos[0]:.5f}", f"{tpos[1]:.5f}", f"{tpos[2]:.5f}",
                               f"{trot[0]:.5f}", f"{trot[1]:.5f}", f"{trot[2]:.5f}",
                               f"{pos_err:.6f}", f"{rot_err:.6f}", f"{dseed:.4f}", f"{dprev:.4f}",
                               " ".join(f"{v:.2f}" for v in seed),
                               " ".join(f"{v:.2f}" for v in q)])
            _ik_log_file.flush()
        except Exception as e:
            print(f"[ik] log error: {e}")
        return sol

    kinematics_solver.inverse_kinematics = _ik_logged

    # ── Teleop pipeline ──────────────────────────────────────────────────────
    teleop_config = PhoneConfig(phone_os=PhoneOS.ANDROID)
    teleop_device = Phone(teleop_config)

    pipeline = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            MapPhoneActionToRobotAction(platform=teleop_config.phone_os),
            EEReferenceAndDelta(
                kinematics=kinematics_solver,
                end_effector_step_sizes={"x": 1, "y": 1, "z": 1},
                motor_names=MOTOR_NAMES,
                use_latched_reference=True,
            ),
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.5, 1.5, 1.5]},
                max_ee_step_m=0.3,
            ),
            GripperVelocityToJoint(speed_factor=20.0),
            InverseKinematicsEEToJoints(
                kinematics=kinematics_solver,
                motor_names=MOTOR_NAMES,
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # ── Robot (optional) ─────────────────────────────────────────────────────
    robot = None
    if connect_robot:
        robot_config = SO100FollowerConfig(port=PORT, id=ROBOT_ID, use_degrees=True)
        robot = SO100Follower(robot_config)
        robot.connect()
        print(f"Robot connected on {PORT}")

    teleop_device.connect()
    print("Viser running at http://localhost:8080")
    print("Squeeze trigger to move. A button resets gripper open.")

    # Simulated joint state (degrees) — used when no robot or as vis state
    sim_joints: dict[str, float] = {name: 0.0 for name in MOTOR_NAMES}
    gripper_closed = False
    prev_button_a = False
    trigger_gripper_pos = GRIPPER_OPEN_POS
    step = 0

    while True:
        t0 = time.perf_counter()

        # ── Get observations ─────────────────────────────────────────────────
        if robot is not None:
            robot_obs = robot.get_observation()
            # Mirror encoder readings into sim_joints for visualization
            for name in MOTOR_NAMES:
                sim_joints[name] = float(robot_obs.get(f"{name}.pos", 0.0))
        else:
            robot_obs = make_obs(sim_joints)

        phone_obs = teleop_device.get_action()
        msg = getattr(getattr(teleop_device, "_phone_impl", None), "_latest_message", None) or {}

        if step % FPS == 0:
            print(f"[webxr] device={msg.get('device')} | {msg.get('debug')}")
        step += 1

        # ── Run teleop pipeline ──────────────────────────────────────────────
        try:
            joint_action = pipeline((phone_obs, robot_obs))
        except Exception:
            joint_action = None

        # ── Gripper control ──────────────────────────────────────────────────
        raw_inputs = phone_obs.get("phone.raw_inputs", {}) if phone_obs else {}
        trigger = float(msg.get("trigger", 0.0))
        button_a = bool(raw_inputs.get("reservedButtonA", False))
        if button_a and not prev_button_a:
            gripper_closed = not gripper_closed
        prev_button_a = button_a

        if trigger > TRIGGER_DEADZONE:
            trigger_gripper_pos = GRIPPER_OPEN_POS + trigger * (GRIPPER_CLOSED_POS - GRIPPER_OPEN_POS)
            gripper_pos = trigger_gripper_pos
        else:
            gripper_pos = GRIPPER_CLOSED_POS if gripper_closed else GRIPPER_OPEN_POS

        if joint_action is not None:
            joint_action["gripper.pos"] = gripper_pos

            # Update simulated state from pipeline output
            for name in MOTOR_NAMES:
                key = f"{name}.pos"
                if key in joint_action:
                    sim_joints[name] = float(joint_action[key])

            # Send to physical robot if connected
            if robot is not None:
                robot.send_action(joint_action)

        # ── Update viser ─────────────────────────────────────────────────────
        cfg = np.array([np.deg2rad(sim_joints.get(j, 0.0)) for j in URDF_JOINT_ORDER])
        urdf_vis.update_cfg(cfg)

        # EE position via FK
        q = np.array([sim_joints[n] for n in MOTOR_NAMES], dtype=float)
        T_ee = kinematics_solver.forward_kinematics(q)
        ee_pos = T_ee[:3, 3].astype(float)

        ee_sphere.position = tuple(ee_pos)

        trail_pts.append(ee_pos.copy())
        if len(trail_pts) > TRAIL_LEN:
            trail_pts.pop(0)
        if len(trail_pts) > 1:
            pts = np.array(trail_pts)
            # Fade from blue (old) to orange (new)
            n = len(pts)
            colors = np.zeros((n, 3))
            colors[:, 0] = np.linspace(0.2, 1.0, n)
            colors[:, 1] = np.linspace(0.6, 0.4, n)
            colors[:, 2] = np.linspace(1.0, 0.0, n)
            trail_handle.points = pts
            trail_handle.colors = colors

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-robot", action="store_true", help="Simulate only, no hardware")
    args = parser.parse_args()
    main(connect_robot=not args.no_robot)
