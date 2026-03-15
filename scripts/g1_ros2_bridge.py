#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import pathlib
import re
import sys
import time
from typing import Sequence

import numpy as np
import yaml


def configure_internal_ros2(ros_distro: str) -> None:
    ext_root = pathlib.Path("/isaac-sim/exts/isaacsim.ros2.bridge")
    ros_lib = ext_root / ros_distro / "lib"
    ros_python = ext_root / ros_distro / "rclpy"
    ros_prefix = ext_root / ros_distro

    if os.environ.get("ISAACSIM_INTERNAL_ROS2_READY") != "1":
        env = os.environ.copy()
        env.setdefault("ROS_DISTRO", ros_distro)
        env.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
        env["LD_LIBRARY_PATH"] = f"{ros_lib}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
        env["PYTHONPATH"] = f"{ros_python}:{env.get('PYTHONPATH', '')}".rstrip(":")
        env["AMENT_PREFIX_PATH"] = f"{ros_prefix}:{env.get('AMENT_PREFIX_PATH', '')}".rstrip(":")
        env["ISAACSIM_INTERNAL_ROS2_READY"] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], env)

    os.environ.setdefault("ROS_DISTRO", ros_distro)
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    os.environ["LD_LIBRARY_PATH"] = f"{ros_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    os.environ["PYTHONPATH"] = f"{ros_python}:{os.environ.get('PYTHONPATH', '')}".rstrip(":")
    os.environ["AMENT_PREFIX_PATH"] = f"{ros_prefix}:{os.environ.get('AMENT_PREFIX_PATH', '')}".rstrip(":")
    if str(ros_python) not in sys.path:
        sys.path.append(str(ros_python))


configure_internal_ros2("humble")

from isaacsim import SimulationApp


EXPECTED_DOF_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "L_index_proximal_joint",
    "L_middle_proximal_joint",
    "L_pinky_proximal_joint",
    "L_ring_proximal_joint",
    "L_thumb_proximal_yaw_joint",
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_pinky_proximal_joint",
    "R_ring_proximal_joint",
    "R_thumb_proximal_yaw_joint",
    "L_index_intermediate_joint",
    "L_middle_intermediate_joint",
    "L_pinky_intermediate_joint",
    "L_ring_intermediate_joint",
    "L_thumb_proximal_pitch_joint",
    "R_index_intermediate_joint",
    "R_middle_intermediate_joint",
    "R_pinky_intermediate_joint",
    "R_ring_intermediate_joint",
    "R_thumb_proximal_pitch_joint",
    "L_thumb_intermediate_joint",
    "R_thumb_intermediate_joint",
    "L_thumb_distal_joint",
    "R_thumb_distal_joint",
]

ACTION_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
]


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def log(message: str) -> None:
    print(message, flush=True)


def create_simulation_app(headless: bool):
    try:
        from isaaclab.app import AppLauncher
    except ImportError:
        return SimulationApp({"headless": headless})
    launcher = AppLauncher(headless=headless)
    return launcher.app


def quat_wxyz_to_rot(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def yaw_from_quat_wxyz(q: Sequence[float]) -> float:
    rot = quat_wxyz_to_rot(q)
    return math.atan2(rot[1, 0], rot[0, 0])


def load_env_yaml(path: pathlib.Path) -> dict:
    class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
        def ignore_unknown(self, node):
            return None

    def tuple_constructor(loader, node):
        return tuple(loader.construct_sequence(node))

    SafeLoaderIgnoreUnknown.add_constructor("tag:yaml.org,2002:python/tuple", tuple_constructor)
    SafeLoaderIgnoreUnknown.add_constructor(None, SafeLoaderIgnoreUnknown.ignore_unknown)
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f, Loader=SafeLoaderIgnoreUnknown)


def _match_pattern(joint_name: str, pattern: str) -> bool:
    return bool(re.fullmatch(pattern, joint_name))


def _resolve_value(joint_name: str, spec, fallback: float) -> float:
    if isinstance(spec, (float, int)):
        return float(spec)
    if isinstance(spec, dict):
        for pattern, value in spec.items():
            if _match_pattern(joint_name, pattern):
                return float(value)
    return float(fallback)


def extract_joint_properties(env_yaml: dict, dof_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    actuator_cfg = env_yaml["scene"]["robot"]["actuators"]
    init_state = env_yaml["scene"]["robot"]["init_state"]

    stiffness = np.zeros(len(dof_names), dtype=np.float32)
    damping = np.zeros(len(dof_names), dtype=np.float32)
    max_effort = np.full(len(dof_names), np.finfo(np.float32).max, dtype=np.float32)
    max_vel = np.full(len(dof_names), np.finfo(np.float32).max, dtype=np.float32)

    for i, joint_name in enumerate(dof_names):
        for actuator in actuator_cfg.values():
            if any(_match_pattern(joint_name, expr) for expr in actuator["joint_names_expr"]):
                stiffness[i] = _resolve_value(joint_name, actuator.get("stiffness"), 0.0)
                damping[i] = _resolve_value(joint_name, actuator.get("damping"), 0.0)
                max_effort[i] = _resolve_value(
                    joint_name,
                    actuator.get("effort_limit_sim", actuator.get("effort_limit")),
                    np.finfo(np.float32).max,
                )
                max_vel[i] = _resolve_value(
                    joint_name,
                    actuator.get("velocity_limit_sim", actuator.get("velocity_limit")),
                    np.finfo(np.float32).max,
                )
                break

    default_pos = np.zeros(len(dof_names), dtype=np.float32)
    default_vel = np.zeros(len(dof_names), dtype=np.float32)
    for i, joint_name in enumerate(dof_names):
        default_pos[i] = _resolve_value(joint_name, init_state["joint_pos"], 0.0)
        default_vel[i] = _resolve_value(joint_name, init_state["joint_vel"], 0.0)

    return default_pos, default_vel, stiffness, damping, max_effort, max_vel


class Ros2PolicyBridge:
    def __init__(self, obs_topic: str, action_topic: str):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float32MultiArray
        from rosgraph_msgs.msg import Clock

        self._rclpy = rclpy
        if not self._rclpy.ok():
            self._rclpy.init(args=None)

        self._float_array_cls = Float32MultiArray
        self._clock_cls = Clock
        self._joint_state_cls = JointState
        self._node = Node("g1_policy_bridge")

        obs_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        action_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        clock_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self._obs_pub = self._node.create_publisher(Float32MultiArray, obs_topic, obs_qos)
        self._clock_pub = self._node.create_publisher(Clock, "/clock", clock_qos)
        self._action_sub = self._node.create_subscription(JointState, action_topic, self._on_action, action_qos)

        self.latest_action = np.zeros(len(ACTION_JOINT_NAMES), dtype=np.float32)
        self.latest_action_stamp_ns = -1
        self.latest_action_received = False

    def _on_action(self, msg) -> None:
        if list(msg.name) != ACTION_JOINT_NAMES:
            self._node.get_logger().error("Received action with unexpected joint ordering")
            return
        if len(msg.position) != len(ACTION_JOINT_NAMES):
            self._node.get_logger().error("Received action with unexpected dimension")
            return
        self.latest_action = np.asarray(msg.position, dtype=np.float32)
        self.latest_action_stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        self.latest_action_received = True

    def publish_clock(self, sim_time_s: float) -> None:
        clock_msg = self._clock_cls()
        sec = int(sim_time_s)
        nanosec = int((sim_time_s - sec) * 1_000_000_000)
        clock_msg.clock.sec = sec
        clock_msg.clock.nanosec = nanosec
        self._clock_pub.publish(clock_msg)

    def publish_obs(self, obs: np.ndarray) -> None:
        msg = self._float_array_cls()
        msg.data = obs.astype(np.float32).tolist()
        self._obs_pub.publish(msg)

    def spin_once(self, timeout_sec: float = 0.0) -> None:
        self._rclpy.spin_once(self._node, timeout_sec=timeout_sec)

    def is_action_fresh(self, sim_time_s: float, timeout_s: float) -> bool:
        if not self.latest_action_received:
            return False
        if self.latest_action_stamp_ns < 0:
            return True
        action_age = sim_time_s - (self.latest_action_stamp_ns / 1_000_000_000.0)
        return action_age <= timeout_s

    def wait_for_action_for_stamp(self, sim_time_s: float, timeout_s: float) -> bool:
        target_stamp_ns = int(round(sim_time_s * 1_000_000_000.0))
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.spin_once(timeout_sec=min(0.001, max(deadline - time.monotonic(), 0.0)))
            if self.latest_action_received and self.latest_action_stamp_ns >= target_stamp_ns:
                return True
        return self.latest_action_received and self.latest_action_stamp_ns >= target_stamp_ns

    def shutdown(self) -> None:
        self._node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Run G1 Isaac Sim deploy and exchange policy tensors over ROS 2.")
    parser.add_argument("--headless", action="store_true", help="Run without GUI.")
    parser.add_argument("--env-yaml", type=pathlib.Path, required=True)
    parser.add_argument("--usd-path", type=pathlib.Path, required=True)
    parser.add_argument("--num-steps", type=int, default=5000)
    parser.add_argument("--lin-vel-x", type=float, default=0.5)
    parser.add_argument("--lin-vel-y", type=float, default=0.0)
    parser.add_argument("--target-heading", type=float, default=0.0)
    parser.add_argument("--spawn-height", type=float, default=0.8)
    parser.add_argument("--obs-topic", default="/g1/policy_obs")
    parser.add_argument("--action-topic", default="/g1/policy_action")
    parser.add_argument("--action-timeout", type=float, default=0.10)
    parser.add_argument("--action-wait-timeout", type=float, default=0.02)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument(
        "--ui-render-interval",
        type=int,
        default=0,
        help="Render the GUI every N physics steps. Defaults to decimation when GUI is enabled.",
    )
    args = parser.parse_args()

    simulation_app = create_simulation_app(args.headless)
    bridge = None
    try:
        import carb
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.prims import define_prim
        from isaacsim.core.utils.types import ArticulationAction
        from omni.physx import get_physx_simulation_interface

        carb.settings.get_settings().set("/persistent/simulation/minFrameRate", 1)

        env_yaml = load_env_yaml(args.env_yaml.resolve())
        sim_dt = float(env_yaml["sim"]["dt"])
        decimation = int(env_yaml["decimation"])
        ui_render_interval = args.ui_render_interval if args.ui_render_interval > 0 else decimation
        command_cfg = env_yaml["commands"]["base_velocity"]
        heading_kp = float(command_cfg["heading_control_stiffness"])
        ang_vel_range = tuple(float(v) for v in command_cfg["ranges"]["ang_vel_z"])
        action_scale = float(env_yaml["actions"]["joint_pos"]["scale"])

        log("Creating World")
        world = World(stage_units_in_meters=1.0, physics_dt=sim_dt, rendering_dt=sim_dt * decimation)
        world.scene.add_default_ground_plane(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)

        log("Referencing G1 USD")
        prim = define_prim("/World/G1", "Xform")
        prim.GetReferences().AddReference(str(args.usd_path.resolve()))
        robot = SingleArticulation(
            prim_path="/World/G1",
            name="g1",
            position=np.array([0.0, 0.0, args.spawn_height], dtype=np.float64),
        )

        log("Resetting world")
        world.reset()
        log("Initializing articulation")
        robot.initialize()

        dof_names = list(robot.dof_names)
        if dof_names != EXPECTED_DOF_NAMES:
            raise RuntimeError("Robot DoF order does not match the training env.")

        log("Extracting joint properties")
        action_joint_indices = [dof_names.index(name) for name in ACTION_JOINT_NAMES]
        default_pos, default_vel, stiffness, damping, max_effort, max_vel = extract_joint_properties(env_yaml, dof_names)

        log("Applying articulation control settings")
        robot.get_articulation_controller().set_effort_modes("force")
        get_physx_simulation_interface().flush_changes()
        robot.get_articulation_controller().switch_control_mode("position")
        robot._articulation_view.set_gains(stiffness, damping)
        robot._articulation_view.set_max_efforts(max_effort)
        get_physx_simulation_interface().flush_changes()
        robot._articulation_view.set_max_joint_velocities(max_vel)

        robot.set_joint_positions(default_pos)
        robot.apply_action(ArticulationAction(joint_positions=default_pos.copy()))

        log("Creating ROS 2 bridge node")
        bridge = Ros2PolicyBridge(args.obs_topic, args.action_topic)
        raw_action = np.zeros(len(ACTION_JOINT_NAMES), dtype=np.float32)
        sim_time_s = 0.0

        def build_obs(command: np.ndarray) -> np.ndarray:
            pos_w, quat_wxyz = robot.get_world_pose()
            lin_vel_w = robot.get_linear_velocity()
            ang_vel_w = robot.get_angular_velocity()

            rot_wb = quat_wxyz_to_rot(quat_wxyz)
            rot_bw = rot_wb.T
            lin_vel_b = rot_bw @ np.asarray(lin_vel_w, dtype=np.float64)
            ang_vel_b = rot_bw @ np.asarray(ang_vel_w, dtype=np.float64)
            gravity_b = rot_bw @ np.array([0.0, 0.0, -1.0], dtype=np.float64)

            current_joint_pos = np.asarray(robot.get_joint_positions(), dtype=np.float32)
            current_joint_vel = np.asarray(robot.get_joint_velocities(), dtype=np.float32)

            obs = np.zeros(138, dtype=np.float32)
            obs[0:3] = lin_vel_b.astype(np.float32)
            obs[3:6] = ang_vel_b.astype(np.float32)
            obs[6:9] = gravity_b.astype(np.float32)
            obs[9:12] = command
            obs[12:65] = current_joint_pos - default_pos
            obs[65:118] = current_joint_vel - default_vel
            obs[118:138] = raw_action
            return obs

        startup_deadline = time.time() + args.startup_timeout
        initial_command = np.array([args.lin_vel_x, args.lin_vel_y, 0.0], dtype=np.float32)
        log("Waiting for first ROS 2 action")
        while not bridge.latest_action_received:
            bridge.publish_clock(sim_time_s)
            bridge.publish_obs(build_obs(initial_command))
            bridge.wait_for_action_for_stamp(sim_time_s, min(args.action_wait_timeout, 0.05))
            if time.time() > startup_deadline:
                raise TimeoutError("Timed out waiting for the first ROS 2 policy action.")
            time.sleep(0.05)

        log(f"ROS 2 action stream received on {args.action_topic}")

        for step in range(args.num_steps):
            bridge.publish_clock(sim_time_s)
            bridge.spin_once(timeout_sec=0.0)

            pos_w, quat_wxyz = robot.get_world_pose()
            current_heading = yaw_from_quat_wxyz(quat_wxyz)
            yaw_rate_cmd = np.clip(
                heading_kp * wrap_to_pi(args.target_heading - current_heading),
                ang_vel_range[0],
                ang_vel_range[1],
            )
            command = np.array([args.lin_vel_x, args.lin_vel_y, yaw_rate_cmd], dtype=np.float32)

            if step % decimation == 0:
                bridge.publish_obs(build_obs(command))
                bridge.wait_for_action_for_stamp(sim_time_s, args.action_wait_timeout)
                if bridge.is_action_fresh(sim_time_s, args.action_timeout):
                    raw_action = bridge.latest_action.copy()

            joint_targets = default_pos.copy()
            for action_idx, joint_idx in enumerate(action_joint_indices):
                joint_targets[joint_idx] = default_pos[joint_idx] + action_scale * raw_action[action_idx]

            robot.apply_action(ArticulationAction(joint_positions=joint_targets))
            should_render_ui = (not args.headless) and (step % ui_render_interval == 0)
            world.step(render=False)
            if should_render_ui:
                world.render()
            sim_time_s += sim_dt

            if step == 0 or (step + 1) % max(args.log_interval, 1) == 0:
                log(
                    f"step={step + 1} "
                    f"cmd=[{command[0]:.3f}, {command[1]:.3f}, {command[2]:.3f}] "
                    f"pos=[{pos_w[0]:.3f}, {pos_w[1]:.3f}, {pos_w[2]:.3f}] "
                    f"heading={current_heading:.3f} "
                    f"action_fresh={bridge.is_action_fresh(sim_time_s, args.action_timeout)}"
                )
    finally:
        if bridge is not None:
            bridge.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    main()
