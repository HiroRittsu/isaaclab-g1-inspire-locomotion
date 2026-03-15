#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np
import rclpy
import torch
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray


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


class PolicyNode(Node):
    def __init__(self, policy_path: str, obs_topic: str, action_topic: str, log_interval: int):
        super().__init__("g1_policy_controller")
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        self._log_interval = max(log_interval, 1)
        self._obs_count = 0

        self.get_logger().info(f"Loading TorchScript policy from: {policy_path}")
        self._policy = torch.jit.load(policy_path, map_location="cpu")
        self._policy.eval()

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

        self._action_pub = self.create_publisher(JointState, action_topic, action_qos)
        self._obs_sub = self.create_subscription(Float32MultiArray, obs_topic, self._on_obs, obs_qos)

    def _on_obs(self, msg: Float32MultiArray) -> None:
        obs = np.asarray(msg.data, dtype=np.float32)
        if obs.shape[0] != 138:
            self.get_logger().error(f"Expected 138-dim observation, got {obs.shape[0]}")
            return

        with torch.no_grad():
            action = self._policy(torch.from_numpy(obs).view(1, -1)).view(-1).cpu().numpy().astype(np.float32)

        if action.shape[0] != len(ACTION_JOINT_NAMES):
            self.get_logger().error(f"Expected {len(ACTION_JOINT_NAMES)} actions, got {action.shape[0]}")
            return

        action_msg = JointState()
        action_msg.header.stamp = self.get_clock().now().to_msg()
        action_msg.name = list(ACTION_JOINT_NAMES)
        action_msg.position = action.astype(float).tolist()
        self._action_pub.publish(action_msg)

        self._obs_count += 1
        if self._obs_count == 1 or self._obs_count % self._log_interval == 0:
            self.get_logger().info(
                "obs=%d sim_time=%.3f action_norm=%.3f"
                % (
                    self._obs_count,
                    self.get_clock().now().nanoseconds / 1.0e9,
                    float(np.linalg.norm(action)),
                )
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the exported G1 policy as a ROS 2 node.")
    parser.add_argument("--policy-path", required=True, help="Path to exported TorchScript policy.pt")
    parser.add_argument("--obs-topic", default="/g1/policy_obs")
    parser.add_argument("--action-topic", default="/g1/policy_action")
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rclpy.init(args=None)
    node = PolicyNode(
        policy_path=args.policy_path,
        obs_topic=args.obs_topic,
        action_topic=args.action_topic,
        log_interval=args.log_interval,
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
