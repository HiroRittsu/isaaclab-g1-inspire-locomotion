#!/usr/bin/env python3
import argparse
import pathlib
import sys

from isaaclab.app import AppLauncher

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "source" / "isaaclab_g1_inspire_locomotion"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import gymnasium as gym
import torch

import isaaclab_g1_inspire_locomotion  # noqa: F401


TASK_BY_MODE = {
    "default": "Isaac-G1-Inspire-Flat-Default-v0",
    "advanced": "Isaac-G1-Inspire-Flat-Advanced-v0",
    "loose_termination": "Isaac-G1-Inspire-Flat-LooseTermination-v0",
    "unitree_rewards": "Isaac-G1-Inspire-Flat-UnitreeRewards-v0",
}


def main():
    parser = argparse.ArgumentParser(description="Inspect deploy-critical env details in headless Isaac Lab.")
    parser.add_argument("--mode", choices=TASK_BY_MODE.keys(), default="default")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    args_cli.headless = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    env = None
    lines: list[str] = []
    try:
        from isaaclab_g1_inspire_locomotion.tasks.locomotion.velocity.g1_inspire.advanced_env_cfg import (
            G1InspireFlatAdvancedPlayEnvCfg,
        )
        from isaaclab_g1_inspire_locomotion.tasks.locomotion.velocity.g1_inspire.default_env_cfg import (
            G1InspireFlatDefaultPlayEnvCfg,
        )
        from isaaclab_g1_inspire_locomotion.tasks.locomotion.velocity.g1_inspire.loose_termination_env_cfg import (
            G1InspireFlatLooseTerminationPlayEnvCfg,
        )
        from isaaclab_g1_inspire_locomotion.tasks.locomotion.velocity.g1_inspire.unitree_rewards_env_cfg import (
            G1InspireFlatUnitreeRewardsPlayEnvCfg,
        )

        play_cfg_by_mode = {
            "default": G1InspireFlatDefaultPlayEnvCfg,
            "advanced": G1InspireFlatAdvancedPlayEnvCfg,
            "loose_termination": G1InspireFlatLooseTerminationPlayEnvCfg,
            "unitree_rewards": G1InspireFlatUnitreeRewardsPlayEnvCfg,
        }

        env_cfg = play_cfg_by_mode[args_cli.mode]()
        env_cfg.scene.num_envs = args_cli.num_envs
        env = gym.make(TASK_BY_MODE[args_cli.mode], cfg=env_cfg)
        env_cfg = env.unwrapped.cfg

        lines.append(f"TASK={TASK_BY_MODE[args_cli.mode]}")
        lines.append(f"SIM_DT={env_cfg.sim.dt}")
        lines.append(f"DECIMATION={env_cfg.decimation}")
        lines.append(f"STEP_DT={env.unwrapped.step_dt}")

        obs, _ = env.reset()
        _ = obs

        robot = env.unwrapped.scene["robot"]
        action_term = env.unwrapped.action_manager.get_term("joint_pos")
        command_term = env.unwrapped.command_manager.get_term("base_velocity")

        lines.append(f"OBS_DIM_POLICY={env.unwrapped.observation_manager.group_obs_dim['policy']}")
        lines.append(f"ACTION_DIM_TOTAL={env.unwrapped.action_manager.total_action_dim}")
        lines.append(f"ACTION_TERMS={env.unwrapped.action_manager.active_terms}")
        lines.append(f"COMMAND_DIM={tuple(command_term.command.shape[1:])}")
        lines.append(f"COMMAND_HEADING={command_term.cfg.heading_command}")
        lines.append(f"COMMAND_RANGES={command_term.cfg.ranges}")

        lines.append("ROBOT_DOF_NAMES_BEGIN")
        for i, name in enumerate(robot.joint_names):
            lines.append(f"{i:02d} {name}")
        lines.append("ROBOT_DOF_NAMES_END")

        lines.append("ACTION_JOINTS_BEGIN")
        for i, name in enumerate(action_term._joint_names):
            lines.append(f"{i:02d} {name}")
        lines.append("ACTION_JOINTS_END")

        lines.append("DEFAULT_JOINT_POS_BEGIN")
        for i, (name, value) in enumerate(zip(robot.joint_names, robot.data.default_joint_pos[0].tolist(), strict=True)):
            lines.append(f"{i:02d} {name} {value:.6f}")
        lines.append("DEFAULT_JOINT_POS_END")

        policy_obs = env.unwrapped.observation_manager.compute_group("policy")
        joint_pos_rel = robot.data.joint_pos - robot.data.default_joint_pos
        joint_vel_rel = robot.data.joint_vel - robot.data.default_joint_vel
        last_action = env.unwrapped.action_manager.action
        command = env.unwrapped.command_manager.get_command("base_velocity")

        lines.append(f"POLICY_OBS_SHAPE={tuple(policy_obs.shape)}")
        lines.append(f"JOINT_POS_REL_SHAPE={tuple(joint_pos_rel.shape)}")
        lines.append(f"JOINT_VEL_REL_SHAPE={tuple(joint_vel_rel.shape)}")
        lines.append(f"LAST_ACTION_SHAPE={tuple(last_action.shape)}")
        lines.append(f"COMMAND_SHAPE={tuple(command.shape)}")
        lines.append(f"COMMAND_SAMPLE={command[0].tolist()}")
        lines.append(f"ROOT_HEADING_W={robot.data.heading_w[0].item():.6f}")
        if command_term.cfg.heading_command:
            lines.append(f"HEADING_TARGET={command_term.heading_target[0].item():.6f}")
            lines.append(f"IS_HEADING_ENV={bool(command_term.is_heading_env[0].item())}")
            lines.append(f"IS_STANDING_ENV={bool(command_term.is_standing_env[0].item())}")

        with torch.no_grad():
            zero_action = torch.zeros(
                (env.unwrapped.num_envs, env.unwrapped.action_manager.total_action_dim), device=env.unwrapped.device
            )
            env.step(zero_action)
            command_after_step = env.unwrapped.command_manager.get_command("base_velocity")
            lines.append(f"COMMAND_SAMPLE_AFTER_STEP={command_after_step[0].tolist()}")
            lines.append(f"ROOT_HEADING_W_AFTER_STEP={robot.data.heading_w[0].item():.6f}")
            if command_term.cfg.heading_command:
                lines.append(f"HEADING_TARGET_AFTER_STEP={command_term.heading_target[0].item():.6f}")

        output_path = args_cli.output
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("\n".join(lines) + "\n")
        else:
            print("\n".join(lines))

    finally:
        try:
            env.close()
        except Exception:
            pass
        simulation_app.close()


if __name__ == "__main__":
    main()
