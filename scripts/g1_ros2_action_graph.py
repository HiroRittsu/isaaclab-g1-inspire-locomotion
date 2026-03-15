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


def configure_internal_ros2_env(ros_distro: str) -> None:
    ext_root = pathlib.Path("/isaac-sim/exts/isaacsim.ros2.bridge")
    ros_lib = ext_root / ros_distro / "lib"

    if os.environ.get("ISAACSIM_INTERNAL_ROS2_READY") != "1":
        env = os.environ.copy()
        env.setdefault("ROS_DISTRO", ros_distro)
        env.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
        env["LD_LIBRARY_PATH"] = f"{ros_lib}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
        env["ISAACSIM_INTERNAL_ROS2_READY"] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], env)

    os.environ.setdefault("ROS_DISTRO", ros_distro)
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    os.environ["LD_LIBRARY_PATH"] = f"{ros_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}".rstrip(":")


configure_internal_ros2_env("humble")

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


def log(message: str) -> None:
    print(message, flush=True)


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


def wait_for_ros2_context_ready(simulation_app, og, graph_path: str = "/ActionGraph/Context", max_wait_s: float = 30.0) -> None:
    deadline = time.monotonic() + max_wait_s
    attr = og.Controller.attribute(f"{graph_path}.outputs:context")

    while time.monotonic() < deadline:
        try:
            value = og.Controller.get(attr)
        except Exception:
            value = None
        if value not in (None, 0):
            return
        simulation_app.update()

    raise TimeoutError("Timed out waiting for ROS 2 Action Graph context to become ready.")


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


def extract_joint_properties(
    env_yaml: dict, dof_names: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    actuator_cfg = env_yaml["scene"]["robot"]["actuators"]
    init_state = env_yaml["scene"]["robot"]["init_state"]

    stiffness = np.zeros(len(dof_names), dtype=np.float32)
    damping = np.zeros(len(dof_names), dtype=np.float32)
    max_effort = np.full(len(dof_names), np.finfo(np.float32).max, dtype=np.float32)
    max_vel = np.full(len(dof_names), np.finfo(np.float32).max, dtype=np.float32)
    default_pos = np.zeros(len(dof_names), dtype=np.float32)
    default_vel = np.zeros(len(dof_names), dtype=np.float32)

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
        default_pos[i] = _resolve_value(joint_name, init_state["joint_pos"], 0.0)
        default_vel[i] = _resolve_value(joint_name, init_state["joint_vel"], 0.0)

    return default_pos, default_vel, stiffness, damping, max_effort, max_vel


def main() -> None:
    parser = argparse.ArgumentParser(description="Run G1 Isaac Sim headless with ROS 2 Action Graph I/O.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--env-yaml", type=pathlib.Path, required=True)
    parser.add_argument("--usd-path", type=pathlib.Path, required=True)
    parser.add_argument("--num-steps", type=int, default=5000)
    parser.add_argument("--spawn-height", type=float, default=0.8)
    parser.add_argument("--joint-state-topic", default="/g1/joint_states")
    parser.add_argument("--odometry-topic", default="/g1/odometry")
    parser.add_argument("--action-topic", default="/g1/joint_command")
    parser.add_argument("--clock-topic", default="/clock")
    parser.add_argument("--odom-frame-id", default="odom")
    parser.add_argument("--chassis-frame-id", default="pelvis")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--startup-warmup-steps", type=int, default=2000)
    args = parser.parse_args()

    log("Action Graph main start")
    simulation_app = SimulationApp({"headless": args.headless})
    log("Simulation app created")
    try:
        import carb
        import omni.graph.core as og
        import omni.kit.app
        import usdrt.Sdf
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.prims import define_prim
        from isaacsim.core.utils.types import ArticulationAction

        log("Enabling required extensions")
        extension_manager = omni.kit.app.get_app().get_extension_manager()
        extension_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
        for _ in range(30):
            simulation_app.update()

        log("Importing sensor/control modules")
        from omni.physx import get_physx_simulation_interface

        carb.settings.get_settings().set("/persistent/simulation/minFrameRate", 1)

        env_yaml = load_env_yaml(args.env_yaml.resolve())
        sim_dt = float(env_yaml["sim"]["dt"])
        decimation = int(env_yaml["decimation"])

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
        robot.initialize()

        dof_names = list(robot.dof_names)
        if dof_names != EXPECTED_DOF_NAMES:
            raise RuntimeError("Robot DoF order does not match the training env.")

        default_pos, _, stiffness, damping, max_effort, max_vel = extract_joint_properties(env_yaml, dof_names)
        log("Applying articulation gains")
        robot.get_articulation_controller().set_effort_modes("force")
        get_physx_simulation_interface().flush_changes()
        robot.get_articulation_controller().switch_control_mode("position")
        robot._articulation_view.set_gains(stiffness, damping)
        get_physx_simulation_interface().flush_changes()
        robot._articulation_view.set_max_efforts(max_effort)
        robot._articulation_view.set_max_joint_velocities(max_vel)

        articulation_prop = env_yaml["scene"]["robot"]["spawn"]["articulation_props"]
        if articulation_prop.get("solver_position_iteration_count") is not None:
            robot.set_solver_position_iteration_count(articulation_prop["solver_position_iteration_count"])
        if articulation_prop.get("solver_velocity_iteration_count") is not None:
            robot.set_solver_velocity_iteration_count(articulation_prop["solver_velocity_iteration_count"])
        if articulation_prop.get("enabled_self_collisions") is not None:
            robot.set_enabled_self_collisions(articulation_prop["enabled_self_collisions"])
        if articulation_prop.get("sleep_threshold") is not None:
            robot.set_sleep_threshold(articulation_prop["sleep_threshold"])
        if articulation_prop.get("stabilization_threshold") is not None:
            robot.set_stabilization_threshold(articulation_prop["stabilization_threshold"])

        robot.set_joint_positions(default_pos)
        robot.apply_action(ArticulationAction(joint_positions=default_pos.copy()))

        log("Creating ROS 2 Action Graph")
        og.Controller.edit(
            {
                "graph_path": "/ActionGraph",
                "evaluator_name": "execution",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
            },
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPhysicsStep", "isaacsim.core.nodes.OnPhysicsStep"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("ComputeOdometry", "isaacsim.core.nodes.IsaacComputeOdometry"),
                    ("PublishOdometry", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
                    ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                    ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                    ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
                    ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnPhysicsStep.outputs:step", "ComputeOdometry.inputs:execIn"),
                    ("OnPhysicsStep.outputs:step", "PublishJointState.inputs:execIn"),
                    ("OnPhysicsStep.outputs:step", "SubscribeJointState.inputs:execIn"),
                    ("OnPhysicsStep.outputs:step", "ArticulationController.inputs:execIn"),
                    ("OnPhysicsStep.outputs:step", "PublishClock.inputs:execIn"),
                    ("ComputeOdometry.outputs:execOut", "PublishOdometry.inputs:execIn"),
                    ("Context.outputs:context", "PublishOdometry.inputs:context"),
                    ("Context.outputs:context", "PublishJointState.inputs:context"),
                    ("Context.outputs:context", "SubscribeJointState.inputs:context"),
                    ("Context.outputs:context", "PublishClock.inputs:context"),
                    ("ComputeOdometry.outputs:position", "PublishOdometry.inputs:position"),
                    ("ComputeOdometry.outputs:orientation", "PublishOdometry.inputs:orientation"),
                    ("ComputeOdometry.outputs:linearVelocity", "PublishOdometry.inputs:linearVelocity"),
                    ("ComputeOdometry.outputs:angularVelocity", "PublishOdometry.inputs:angularVelocity"),
                    ("ReadSimTime.outputs:simulationTime", "PublishOdometry.inputs:timeStamp"),
                    ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                    ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                    ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                    ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("ComputeOdometry.inputs:chassisPrim", [usdrt.Sdf.Path("/World/G1/pelvis")]),
                    ("PublishOdometry.inputs:topicName", args.odometry_topic),
                    ("PublishOdometry.inputs:odomFrameId", args.odom_frame_id),
                    ("PublishOdometry.inputs:chassisFrameId", args.chassis_frame_id),
                    ("PublishOdometry.inputs:publishRawVelocities", True),
                    ("PublishJointState.inputs:topicName", args.joint_state_topic),
                    ("SubscribeJointState.inputs:topicName", args.action_topic),
                    ("PublishClock.inputs:topicName", args.clock_topic),
                    ("PublishJointState.inputs:targetPrim", [usdrt.Sdf.Path("/World/G1/pelvis")]),
                    ("ArticulationController.inputs:robotPath", "/World/G1/pelvis"),
                ],
            },
        )
        wait_for_ros2_context_ready(simulation_app, og)

        # Run a startup warmup under the default pose so the ROS 2 graph sees
        # real physics ticks before the measured rollout begins.
        warmup_steps = max(args.startup_warmup_steps, 0)
        if warmup_steps > 0:
            log(f"Running ROS 2 physics warmup for {warmup_steps} steps")
            world.play()
            default_action = ArticulationAction(joint_positions=default_pos.copy())
            for _ in range(warmup_steps):
                robot.apply_action(default_action)
                world.step(render=False)
            world.pause()

        log("Resetting world after ROS 2 bridge warmup")
        world.reset()
        robot.initialize()
        robot.get_articulation_controller().set_effort_modes("force")
        get_physx_simulation_interface().flush_changes()
        robot.get_articulation_controller().switch_control_mode("position")
        robot._articulation_view.set_gains(stiffness, damping)
        get_physx_simulation_interface().flush_changes()
        robot._articulation_view.set_max_efforts(max_effort)
        robot._articulation_view.set_max_joint_velocities(max_vel)
        if articulation_prop.get("solver_position_iteration_count") is not None:
            robot.set_solver_position_iteration_count(articulation_prop["solver_position_iteration_count"])
        if articulation_prop.get("solver_velocity_iteration_count") is not None:
            robot.set_solver_velocity_iteration_count(articulation_prop["solver_velocity_iteration_count"])
        if articulation_prop.get("enabled_self_collisions") is not None:
            robot.set_enabled_self_collisions(articulation_prop["enabled_self_collisions"])
        if articulation_prop.get("sleep_threshold") is not None:
            robot.set_sleep_threshold(articulation_prop["sleep_threshold"])
        if articulation_prop.get("stabilization_threshold") is not None:
            robot.set_stabilization_threshold(articulation_prop["stabilization_threshold"])
        robot.set_joint_positions(default_pos)
        robot.apply_action(ArticulationAction(joint_positions=default_pos.copy()))

        world.play()
        log("Simulation running")
        for step in range(args.num_steps):
            world.step(render=False)
            if step == 0 or (step + 1) % max(args.log_interval, 1) == 0:
                pos_w, quat_wxyz = robot.get_world_pose()
                heading = yaw_from_quat_wxyz(quat_wxyz)
                log(
                    f"step={step + 1} "
                    f"pos=[{pos_w[0]:.3f}, {pos_w[1]:.3f}, {pos_w[2]:.3f}] "
                    f"heading={heading:.3f}"
                )
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
