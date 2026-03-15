#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import rclpy
import torch
import yaml
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from builtin_interfaces.msg import Time as TimeMsg
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState


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


def _match_pattern(joint_name: str, pattern: str) -> bool:
    import re

    return bool(re.fullmatch(pattern, joint_name))


def _resolve_value(joint_name: str, spec, fallback: float) -> float:
    if isinstance(spec, (float, int)):
        return float(spec)
    if isinstance(spec, dict):
        for pattern, value in spec.items():
            if _match_pattern(joint_name, pattern):
                return float(value)
    return float(fallback)


def load_env_yaml(path: Path) -> dict:
    class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
        def ignore_unknown(self, node):
            return None

    def tuple_constructor(loader, node):
        return tuple(loader.construct_sequence(node))

    SafeLoaderIgnoreUnknown.add_constructor("tag:yaml.org,2002:python/tuple", tuple_constructor)
    SafeLoaderIgnoreUnknown.add_constructor(None, SafeLoaderIgnoreUnknown.ignore_unknown)
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f, Loader=SafeLoaderIgnoreUnknown)


def extract_default_state(env_yaml: dict, dof_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    init_state = env_yaml["scene"]["robot"]["init_state"]
    default_pos = np.zeros(len(dof_names), dtype=np.float32)
    default_vel = np.zeros(len(dof_names), dtype=np.float32)
    for i, joint_name in enumerate(dof_names):
        default_pos[i] = _resolve_value(joint_name, init_state["joint_pos"], 0.0)
        default_vel[i] = _resolve_value(joint_name, init_state["joint_vel"], 0.0)
    return default_pos, default_vel


class G1ActionGraphPolicyNode(Node):
    def __init__(
        self,
        policy_path: Path,
        env_yaml_path: Path,
        joint_state_topic: str,
        odometry_topic: str,
        action_topic: str,
        cmd_vel_topic: str,
        log_interval: int,
        debug_dump_path: Path | None,
        debug_dump_interval: int,
    ) -> None:
        super().__init__("g1_action_graph_policy_controller")
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        env_yaml = load_env_yaml(env_yaml_path)
        self._default_pos, self._default_vel = extract_default_state(env_yaml, EXPECTED_DOF_NAMES)
        self._action_indices = [EXPECTED_DOF_NAMES.index(name) for name in ACTION_JOINT_NAMES]
        self._action_default_pos = self._default_pos[self._action_indices]

        command_cfg = env_yaml["commands"]["base_velocity"]
        self._heading_command = bool(command_cfg["heading_command"])
        self._heading_kp = float(command_cfg["heading_control_stiffness"])
        self._ang_vel_range = tuple(float(v) for v in command_cfg["ranges"]["ang_vel_z"])
        self._action_scale = float(env_yaml["actions"]["joint_pos"]["scale"])
        self._decimation = int(env_yaml["decimation"])

        self.declare_parameter("lin_vel_x", 0.5)
        self.declare_parameter("lin_vel_y", 0.0)
        self.declare_parameter("yaw_rate", 0.0)
        self.declare_parameter("target_heading", 0.0)

        self._log_interval = max(log_interval, 1)
        self._debug_dump_interval = max(debug_dump_interval, 1)
        self._tick_count = 0
        self._policy_tick = 0
        self._last_sim_time_s: float | None = None
        self._previous_action = np.zeros(len(ACTION_JOINT_NAMES), dtype=np.float32)
        self._current_action = np.zeros(len(ACTION_JOINT_NAMES), dtype=np.float32)
        self._latest_cmd_vel = Twist()
        self._sim_dt = float(env_yaml["sim"]["dt"])
        self._debug_dump_file = None
        self._pending_joint_state: dict[int, JointState] = {}
        self._pending_odometry: dict[int, Odometry] = {}
        self._last_processed_stamp_ns: int = -1

        if debug_dump_path is not None:
            debug_dump_path = debug_dump_path.resolve()
            debug_dump_path.parent.mkdir(parents=True, exist_ok=True)
            self._debug_dump_file = debug_dump_path.open("w", encoding="utf-8")
            self.get_logger().info(f"Writing debug dump to: {debug_dump_path}")

        self.get_logger().info(f"Loading TorchScript policy from: {policy_path}")
        self._policy = torch.jit.load(str(policy_path), map_location="cpu")
        self._policy.eval()

        sim_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(Twist, cmd_vel_topic, self._on_cmd_vel, qos_profile=10)
        self._joint_pub = self.create_publisher(JointState, action_topic, qos_profile=sim_qos_profile)
        self.create_subscription(JointState, joint_state_topic, self._on_joint_state, qos_profile=sim_qos_profile)
        self.create_subscription(Odometry, odometry_topic, self._on_odometry, qos_profile=sim_qos_profile)

    def _on_cmd_vel(self, msg: Twist) -> None:
        self._latest_cmd_vel = msg

    def _ordered_joint_state(self, msg: JointState) -> tuple[np.ndarray, np.ndarray]:
        position_map = {name: value for name, value in zip(msg.name, msg.position)}
        velocity_map = {name: value for name, value in zip(msg.name, msg.velocity)}
        current_joint_pos = np.asarray([position_map[name] for name in EXPECTED_DOF_NAMES], dtype=np.float32)
        current_joint_vel = np.asarray([velocity_map[name] for name in EXPECTED_DOF_NAMES], dtype=np.float32)
        return current_joint_pos, current_joint_vel

    def _reset_if_needed(self, sim_time_s: float) -> None:
        if self._last_sim_time_s is None or sim_time_s < self._last_sim_time_s:
            self._last_sim_time_s = sim_time_s
            self._previous_action[:] = 0.0
            self._current_action[:] = 0.0
            self._tick_count = 0
            self._policy_tick = 0
            return
        self._last_sim_time_s = sim_time_s

    def _command_vector(self, heading: float) -> np.ndarray:
        lin_vel_x = float(self.get_parameter("lin_vel_x").value)
        lin_vel_y = float(self.get_parameter("lin_vel_y").value)
        yaw_rate = float(self.get_parameter("yaw_rate").value)
        target_heading = float(self.get_parameter("target_heading").value)

        if not math.isclose(self._latest_cmd_vel.linear.x, 0.0) or not math.isclose(self._latest_cmd_vel.linear.y, 0.0):
            lin_vel_x = float(self._latest_cmd_vel.linear.x)
            lin_vel_y = float(self._latest_cmd_vel.linear.y)
        if self._heading_command:
            yaw_rate = np.clip(
                self._heading_kp * wrap_to_pi(target_heading - heading),
                self._ang_vel_range[0],
                self._ang_vel_range[1],
            )
        elif not math.isclose(self._latest_cmd_vel.angular.z, 0.0):
            yaw_rate = float(np.clip(self._latest_cmd_vel.angular.z, self._ang_vel_range[0], self._ang_vel_range[1]))

        return np.array([lin_vel_x, lin_vel_y, yaw_rate], dtype=np.float32)

    def _time_msg_to_seconds(self, msg: TimeMsg) -> float:
        return float(msg.sec) + float(msg.nanosec) * 1.0e-9

    def _time_msg_to_nanoseconds(self, msg: TimeMsg) -> int:
        return int(msg.sec) * 1_000_000_000 + int(msg.nanosec)

    def _prune_pending(self, current_stamp_ns: int) -> None:
        # Keep only a small rolling window around the latest stamp.
        cutoff = current_stamp_ns - int(20 * self._sim_dt * 1_000_000_000)
        self._pending_joint_state = {k: v for k, v in self._pending_joint_state.items() if k >= cutoff}
        self._pending_odometry = {k: v for k, v in self._pending_odometry.items() if k >= cutoff}

    def _on_joint_state(self, msg: JointState) -> None:
        stamp_ns = self._time_msg_to_nanoseconds(msg.header.stamp)
        self._pending_joint_state[stamp_ns] = msg
        self._prune_pending(stamp_ns)
        self._try_process_stamp(stamp_ns)

    def _on_odometry(self, msg: Odometry) -> None:
        stamp_ns = self._time_msg_to_nanoseconds(msg.header.stamp)
        self._pending_odometry[stamp_ns] = msg
        self._prune_pending(stamp_ns)
        self._try_process_stamp(stamp_ns)

    def _try_process_stamp(self, stamp_ns: int) -> None:
        if stamp_ns <= self._last_processed_stamp_ns:
            return
        joint_state = self._pending_joint_state.get(stamp_ns)
        odometry = self._pending_odometry.get(stamp_ns)
        if joint_state is None or odometry is None:
            return
        self._pending_joint_state.pop(stamp_ns, None)
        self._pending_odometry.pop(stamp_ns, None)
        self._last_processed_stamp_ns = stamp_ns
        self._tick(joint_state, odometry)

    def _compute_observation(
        self, joint_state: JointState, odometry: Odometry, sim_time_s: float
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        current_joint_pos, current_joint_vel = self._ordered_joint_state(joint_state)

        quat_wxyz = np.array(
            [
                odometry.pose.pose.orientation.w,
                odometry.pose.pose.orientation.x,
                odometry.pose.pose.orientation.y,
                odometry.pose.pose.orientation.z,
            ],
            dtype=np.float64,
        )
        rot_wb = quat_wxyz_to_rot(quat_wxyz)
        rot_bw = rot_wb.T
        heading = yaw_from_quat_wxyz(quat_wxyz)

        self._reset_if_needed(sim_time_s)
        lin_vel_b = np.array(
            [
                odometry.twist.twist.linear.x,
                odometry.twist.twist.linear.y,
                odometry.twist.twist.linear.z,
            ],
            dtype=np.float32,
        )
        ang_vel_b = np.array(
            [
                odometry.twist.twist.angular.x,
                odometry.twist.twist.angular.y,
                odometry.twist.twist.angular.z,
            ],
            dtype=np.float32,
        )
        gravity_b = (rot_bw @ np.array([0.0, 0.0, -1.0], dtype=np.float64)).astype(np.float32)
        command = self._command_vector(heading)

        obs = np.zeros(138, dtype=np.float32)
        obs[0:3] = lin_vel_b
        obs[3:6] = ang_vel_b
        obs[6:9] = gravity_b
        obs[9:12] = command
        obs[12:65] = current_joint_pos - self._default_pos
        obs[65:118] = current_joint_vel - self._default_vel
        obs[118:138] = self._previous_action
        return obs, heading, lin_vel_b, ang_vel_b, gravity_b, command

    def _publish_action(self, sim_time_msg) -> None:
        full_joint_targets = self._default_pos.copy()
        full_joint_targets[self._action_indices] = self._action_default_pos + self._action_scale * self._current_action

        joint_msg = JointState()
        joint_msg.header.stamp = sim_time_msg
        joint_msg.name = list(EXPECTED_DOF_NAMES)
        joint_msg.position = full_joint_targets.astype(float).tolist()
        self._joint_pub.publish(joint_msg)
        return full_joint_targets

    def _tick(self, joint_state: JointState, odometry: Odometry) -> None:
        sim_time_s = self._time_msg_to_seconds(joint_state.header.stamp)
        obs, heading, lin_vel_b, ang_vel_b, gravity_b, command = self._compute_observation(joint_state, odometry, sim_time_s)
        odom_time_s = self._time_msg_to_seconds(odometry.header.stamp)
        current_joint_pos, current_joint_vel = self._ordered_joint_state(joint_state)
        obs_previous_action = self._previous_action.copy()

        policy_updated = False
        if self._tick_count % self._decimation == 0:
            with torch.no_grad():
                action = self._policy(torch.from_numpy(obs).view(1, -1)).view(-1).cpu().numpy().astype(np.float32)
            self._current_action = action
            self._previous_action = action.copy()
            self._policy_tick += 1
            policy_updated = True

        self._tick_count += 1
        full_joint_targets = self._publish_action(joint_state.header.stamp)

        if (
            self._debug_dump_file is not None
            and policy_updated
            and (self._policy_tick == 1 or self._policy_tick % self._debug_dump_interval == 0)
        ):
            debug_payload = {
                "source": "action_graph_policy_node",
                "sim_step": int(round(sim_time_s / self._sim_dt)),
                "policy_tick": int(self._policy_tick),
                "sim_time": float(sim_time_s),
                "joint_pos_rel": (current_joint_pos - self._default_pos).astype(float).tolist(),
                "joint_vel_rel": (current_joint_vel - self._default_vel).astype(float).tolist(),
                "lin_vel_b": lin_vel_b.astype(float).tolist(),
                "ang_vel_b": ang_vel_b.astype(float).tolist(),
                "gravity_b": gravity_b.astype(float).tolist(),
                "command": command.astype(float).tolist(),
                "previous_action": obs_previous_action.astype(float).tolist(),
                "raw_action": self._current_action.astype(float).tolist(),
                "joint_target": full_joint_targets.astype(float).tolist(),
            }
            self._debug_dump_file.write(json.dumps(debug_payload, separators=(",", ":")) + "\n")
            self._debug_dump_file.flush()

        if self._tick_count == 1 or self._tick_count % self._log_interval == 0:
            self.get_logger().info(
                "tick=%d policy_tick=%d sim_time=%.3f stamp_delta=%.4f heading=%.3f action_norm=%.3f"
                " lin_vel=[%.3f %.3f %.3f] ang_vel=[%.3f %.3f %.3f] cmd=[%.3f %.3f %.3f] grav_z=%.3f"
                % (
                    self._tick_count,
                    self._policy_tick,
                    sim_time_s,
                    abs(odom_time_s - sim_time_s),
                    heading,
                    float(np.linalg.norm(self._current_action)),
                    lin_vel_b[0],
                    lin_vel_b[1],
                    lin_vel_b[2],
                    ang_vel_b[0],
                    ang_vel_b[1],
                    ang_vel_b[2],
                    command[0],
                    command[1],
                    command[2],
                    gravity_b[2],
                )
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the G1 policy from synchronized joint state and IMU topics.")
    parser.add_argument("--policy-path", required=True, type=Path)
    parser.add_argument("--env-yaml", required=True, type=Path)
    parser.add_argument("--joint-state-topic", default="/g1/joint_states")
    parser.add_argument("--odometry-topic", default="/g1/odometry")
    parser.add_argument("--action-topic", default="/g1/joint_command")
    parser.add_argument("--cmd-vel-topic", default="/g1/cmd_vel")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--debug-dump-path", type=Path, default=None)
    parser.add_argument("--debug-dump-interval", type=int, default=25)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rclpy.init(args=None)
    node = G1ActionGraphPolicyNode(
        policy_path=args.policy_path,
        env_yaml_path=args.env_yaml,
        joint_state_topic=args.joint_state_topic,
        odometry_topic=args.odometry_topic,
        action_topic=args.action_topic,
        cmd_vel_topic=args.cmd_vel_topic,
        log_interval=args.log_interval,
        debug_dump_path=args.debug_dump_path,
        debug_dump_interval=args.debug_dump_interval,
    )
    try:
        rclpy.spin(node)
    finally:
        if node._debug_dump_file is not None:
            node._debug_dump_file.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
