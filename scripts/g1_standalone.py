#!/usr/bin/env python3
from isaacsim import SimulationApp

import argparse
import json
import math
import pathlib
import re
from typing import Sequence

import numpy as np
import yaml


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


def create_simulation_app(headless: bool, enable_cameras: bool):
    try:
        from isaaclab.app import AppLauncher
    except ImportError:
        return SimulationApp({"headless": headless})
    launcher = AppLauncher(headless=headless, enable_cameras=enable_cameras)
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


def rot_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    trace = np.trace(rot)
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    return quat / np.linalg.norm(quat)


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


def extract_default_pos(env_yaml: dict, dof_names: Sequence[str]) -> np.ndarray:
    default_pos, _, _, _, _, _ = extract_joint_properties(env_yaml, dof_names)
    return default_pos


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
    default_pos_cfg = init_state["joint_pos"]
    default_vel_cfg = init_state["joint_vel"]
    for i, joint_name in enumerate(dof_names):
        default_pos[i] = _resolve_value(joint_name, default_pos_cfg, 0.0)
        default_vel[i] = _resolve_value(joint_name, default_vel_cfg, 0.0)

    return default_pos, default_vel, stiffness, damping, max_effort, max_vel


def main():
    parser = argparse.ArgumentParser(description="Deploy the trained G1 Inspire locomotion policy in Isaac Sim.")
    parser.add_argument("--headless", action="store_true", help="Run without GUI.")
    parser.add_argument("--policy-path", type=pathlib.Path, required=True, help="TorchScript policy.pt path.")
    parser.add_argument("--env-yaml", type=pathlib.Path, required=True, help="env.yaml path exported from the run.")
    parser.add_argument("--usd-path", type=pathlib.Path, required=True, help="Robot USD path.")
    parser.add_argument("--num-steps", type=int, default=1000, help="Number of world steps to run before exit.")
    parser.add_argument("--lin-vel-x", type=float, default=0.5, help="Target forward velocity command.")
    parser.add_argument("--lin-vel-y", type=float, default=0.0, help="Target lateral velocity command.")
    parser.add_argument("--target-heading", type=float, default=0.0, help="Heading target in radians.")
    parser.add_argument("--spawn-height", type=float, default=0.8, help="Robot spawn height.")
    parser.add_argument("--record-video", action="store_true", help="Record an MP4 from an offscreen camera.")
    parser.add_argument("--video-path", type=pathlib.Path, default=pathlib.Path("outputs/g1_standalone.mp4"))
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=360)
    parser.add_argument("--camera-position", type=float, nargs=3, default=(2.5, -2.5, 1.8))
    parser.add_argument("--camera-target", type=float, nargs=3, default=(0.4, 0.0, 0.85))
    parser.add_argument(
        "--ui-render-interval",
        type=int,
        default=0,
        help="Render the GUI every N physics steps. Defaults to decimation when GUI is enabled.",
    )
    parser.add_argument(
        "--video-render-interval",
        type=int,
        default=0,
        help="Render/capture video every N physics steps. Defaults to decimation when recording.",
    )
    parser.add_argument(
        "--min-frame-rate",
        type=int,
        default=1,
        help="Value for /persistent/simulation/minFrameRate to reduce GUI-induced sim clamping.",
    )
    parser.add_argument(
        "--debug-dump-path",
        type=pathlib.Path,
        default=None,
        help="Optional JSONL path for dumping observation/action/target snapshots.",
    )
    parser.add_argument(
        "--debug-dump-interval",
        type=int,
        default=25,
        help="Dump every N policy ticks when --debug-dump-path is set.",
    )
    args = parser.parse_args()

    simulation_app = create_simulation_app(args.headless, args.record_video)

    video_writer = None
    rgb_annotator = None
    render_product = None
    debug_dump_file = None
    try:
        log("Importing torch")
        import torch
        log("Importing imageio")
        import imageio.v2 as imageio
        log("Importing Isaac Sim core APIs")
        import carb
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.prims import define_prim
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.core.utils.viewports import set_camera_view
        rep = None
        if args.record_video:
            log("Importing replicator")
            import omni.replicator.core as rep
        log("Importing physx interface")
        from omni.physx import get_physx_simulation_interface

        policy_path = args.policy_path.resolve()
        env_yaml_path = args.env_yaml.resolve()
        usd_path = args.usd_path.resolve()

        log("Starting standalone setup")
        env_yaml = load_env_yaml(env_yaml_path)
        sim_dt = float(env_yaml["sim"]["dt"])
        decimation = int(env_yaml["decimation"])
        render_dt = sim_dt * decimation
        ui_render_interval = args.ui_render_interval if args.ui_render_interval > 0 else decimation
        video_render_interval = args.video_render_interval if args.video_render_interval > 0 else decimation
        command_cfg = env_yaml["commands"]["base_velocity"]
        heading_kp = float(command_cfg["heading_control_stiffness"])
        ang_vel_range = tuple(float(v) for v in command_cfg["ranges"]["ang_vel_z"])
        action_scale = float(env_yaml["actions"]["joint_pos"]["scale"])

        carb.settings.get_settings().set("/persistent/simulation/minFrameRate", int(args.min_frame_rate))

        log("Creating world")
        world = World(stage_units_in_meters=1.0, physics_dt=sim_dt, rendering_dt=render_dt)
        world.scene.add_default_ground_plane(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )

        log("Referencing robot USD")
        prim = define_prim("/World/G1", "Xform")
        prim.GetReferences().AddReference(str(usd_path))
        robot = SingleArticulation(
            prim_path="/World/G1",
            name="g1",
            position=np.array([0.0, 0.0, args.spawn_height], dtype=np.float64),
        )

        if args.record_video:
            log("Preparing video camera")
            video_path = args.video_path.resolve()
            video_path.parent.mkdir(parents=True, exist_ok=True)
            rep.orchestrator.set_capture_on_play(False)
            set_camera_view(
                eye=np.asarray(args.camera_position, dtype=np.float64),
                target=np.asarray(args.camera_target, dtype=np.float64),
                camera_prim_path="/OmniverseKit_Persp",
            )
            render_product = rep.create.render_product("/OmniverseKit_Persp", (args.video_width, args.video_height))
            rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            rgb_annotator.attach([render_product])
            log("Opening video writer")
            video_writer = imageio.get_writer(str(video_path), fps=max(int(round(1.0 / render_dt)), 1))

        if args.debug_dump_path is not None:
            debug_dump_path = args.debug_dump_path.resolve()
            debug_dump_path.parent.mkdir(parents=True, exist_ok=True)
            debug_dump_file = debug_dump_path.open("w", encoding="utf-8")
            log(f"Writing debug dump to: {debug_dump_path}")

        log("Loading TorchScript policy")
        policy = torch.jit.load(str(policy_path), map_location="cpu")
        policy.eval()

        log("Resetting world")
        world.reset()
        log("Initializing robot")
        robot.initialize()

        dof_names = list(robot.dof_names)
        if dof_names != EXPECTED_DOF_NAMES:
            raise RuntimeError(
                "Robot DoF order does not match the training env.\n"
                f"Expected: {EXPECTED_DOF_NAMES}\n"
                f"Actual:   {dof_names}"
            )

        action_joint_indices = [dof_names.index(name) for name in ACTION_JOINT_NAMES]
        default_pos, default_vel, stiffness, damping, max_effort, max_vel = extract_joint_properties(env_yaml, dof_names)

        if len(default_pos) != 53:
            raise RuntimeError(f"Unexpected default_pos length: {len(default_pos)}")
        if len(action_joint_indices) != 20:
            raise RuntimeError(f"Unexpected action joint count: {len(action_joint_indices)}")

        robot.get_articulation_controller().set_effort_modes("force")
        get_physx_simulation_interface().flush_changes()
        robot.get_articulation_controller().switch_control_mode("position")
        robot._articulation_view.set_gains(stiffness, damping)
        robot._articulation_view.set_max_efforts(max_effort)
        get_physx_simulation_interface().flush_changes()
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

        previous_action = np.zeros(len(ACTION_JOINT_NAMES), dtype=np.float32)
        raw_action = np.zeros_like(previous_action)
        policy_tick = 0

        log(f"Loaded policy from: {policy_path}")
        log(f"Loaded env yaml from: {env_yaml_path}")
        log(f"Loaded usd from: {usd_path}")
        log(f"sim_dt={sim_dt}, decimation={decimation}, render_dt={render_dt}")
        log(f"action_scale={action_scale}, heading_kp={heading_kp}, ang_vel_range={ang_vel_range}")
        log(
            f"ui_render_interval={ui_render_interval}, "
            f"video_render_interval={video_render_interval}, "
            f"min_frame_rate={args.min_frame_rate}"
        )
        log(f"num_dof={robot.num_dof}, action_dim={len(ACTION_JOINT_NAMES)}")
        if args.record_video:
            log(f"Recording video to: {video_path}")

        for step in range(args.num_steps):
            pos_w, quat_wxyz = robot.get_world_pose()
            lin_vel_w = robot.get_linear_velocity()
            ang_vel_w = robot.get_angular_velocity()

            rot_wb = quat_wxyz_to_rot(quat_wxyz)
            rot_bw = rot_wb.T
            lin_vel_b = rot_bw @ np.asarray(lin_vel_w, dtype=np.float64)
            ang_vel_b = rot_bw @ np.asarray(ang_vel_w, dtype=np.float64)
            gravity_b = rot_bw @ np.array([0.0, 0.0, -1.0], dtype=np.float64)

            current_heading = yaw_from_quat_wxyz(quat_wxyz)
            yaw_rate_cmd = np.clip(
                heading_kp * wrap_to_pi(args.target_heading - current_heading),
                ang_vel_range[0],
                ang_vel_range[1],
            )
            command = np.array([args.lin_vel_x, args.lin_vel_y, yaw_rate_cmd], dtype=np.float32)

            if step % decimation == 0:
                current_joint_pos = np.asarray(robot.get_joint_positions(), dtype=np.float32)
                current_joint_vel = np.asarray(robot.get_joint_velocities(), dtype=np.float32)
                obs_previous_action = previous_action.copy()

                obs = np.zeros(138, dtype=np.float32)
                obs[0:3] = lin_vel_b.astype(np.float32)
                obs[3:6] = ang_vel_b.astype(np.float32)
                obs[6:9] = gravity_b.astype(np.float32)
                obs[9:12] = command
                obs[12:65] = current_joint_pos - default_pos
                obs[65:118] = current_joint_vel - default_vel
                obs[118:138] = obs_previous_action

                with torch.no_grad():
                    raw_action = policy(torch.from_numpy(obs).view(1, -1)).view(-1).cpu().numpy().astype(np.float32)
                previous_action = raw_action.copy()
                policy_tick += 1

            joint_targets = default_pos.copy()
            for action_idx, joint_idx in enumerate(action_joint_indices):
                joint_targets[joint_idx] = default_pos[joint_idx] + action_scale * raw_action[action_idx]

            if (
                debug_dump_file is not None
                and step % decimation == 0
                and (policy_tick == 1 or policy_tick % max(args.debug_dump_interval, 1) == 0)
            ):
                debug_payload = {
                    "source": "standalone",
                    "sim_step": int(step),
                    "policy_tick": int(policy_tick),
                    "sim_time": float(step * sim_dt),
                    "joint_pos_rel": (current_joint_pos - default_pos).astype(float).tolist(),
                    "joint_vel_rel": (current_joint_vel - default_vel).astype(float).tolist(),
                    "lin_vel_b": lin_vel_b.astype(float).tolist(),
                    "ang_vel_b": ang_vel_b.astype(float).tolist(),
                    "gravity_b": gravity_b.astype(float).tolist(),
                    "command": command.astype(float).tolist(),
                    "previous_action": obs_previous_action.astype(float).tolist(),
                    "raw_action": raw_action.astype(float).tolist(),
                    "joint_target": joint_targets.astype(float).tolist(),
                }
                debug_dump_file.write(json.dumps(debug_payload, separators=(",", ":")) + "\n")
                debug_dump_file.flush()

            robot.apply_action(ArticulationAction(joint_positions=joint_targets))
            should_render_ui = (not args.headless) and (step % ui_render_interval == 0)
            should_render_video = (video_writer is not None) and (step % video_render_interval == 0)
            world.step(render=False)

            if should_render_ui or should_render_video:
                world.render()

            if should_render_video:
                frame = rgb_annotator.get_data()
                frame = np.frombuffer(frame, dtype=np.uint8).reshape(*frame.shape)
                if frame.size == 0:
                    frame = np.zeros((args.video_height, args.video_width, 3), dtype=np.uint8)
                elif frame.shape[-1] == 4:
                    frame = frame[:, :, :3]
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                video_writer.append_data(frame)

            if step == 0 or (step + 1) % 100 == 0:
                log(
                    f"step={step + 1} "
                    f"cmd=[{command[0]:.3f}, {command[1]:.3f}, {command[2]:.3f}] "
                    f"pos=[{pos_w[0]:.3f}, {pos_w[1]:.3f}, {pos_w[2]:.3f}] "
                    f"heading={current_heading:.3f}"
                )
    finally:
        if debug_dump_file is not None:
            debug_dump_file.close()
        if rgb_annotator is not None:
            rgb_annotator.detach()
        if render_product is not None:
            render_product.destroy()
        if video_writer is not None:
            log("Closing video writer")
            video_writer.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
