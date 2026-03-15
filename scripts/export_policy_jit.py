#!/usr/bin/env python3
import argparse
import os
import pathlib
import sys

from isaaclab.app import AppLauncher

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "source" / "isaaclab_g1_inspire_locomotion"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


TASK_BY_MODE = {
    "default": "Isaac-G1-Inspire-Flat-Default-v0",
    "advanced": "Isaac-G1-Inspire-Flat-Advanced-v0",
    "loose_termination": "Isaac-G1-Inspire-Flat-LooseTermination-v0",
    "unitree_rewards": "Isaac-G1-Inspire-Flat-UnitreeRewards-v0",
}


def main():
    parser = argparse.ArgumentParser(description="Export a trained RSL-RL policy to TorchScript for Isaac Sim.")
    parser.add_argument("--mode", choices=TASK_BY_MODE.keys(), default="default")
    parser.add_argument("--load_run", required=True, help="Run directory name under logs/rsl_rl/<experiment>/")
    parser.add_argument("--checkpoint", default="model_1499.pt", help="Checkpoint file name within the run directory.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of envs to instantiate for export.")
    parser.add_argument("--output", type=pathlib.Path, default=None, help="Output policy.pt path.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    args_cli.headless = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    env = None
    try:
        import gymnasium as gym
        from rsl_rl.runners import OnPolicyRunner

        import isaaclab_g1_inspire_locomotion  # noqa: F401
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit
        from isaaclab_tasks.utils.parse_cfg import get_checkpoint_path, load_cfg_from_registry

        task_name = TASK_BY_MODE[args_cli.mode]
        env_cfg = load_cfg_from_registry(task_name, "play_env_cfg_entry_point")
        agent_cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")

        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.observations.policy.enable_corruption = False

        agent_cfg.load_run = args_cli.load_run
        agent_cfg.load_checkpoint = args_cli.checkpoint
        agent_cfg.device = env_cfg.sim.device

        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        export_dir = args_cli.output.parent if args_cli.output is not None else pathlib.Path(resume_path).parent / "exported"
        export_name = args_cli.output.name if args_cli.output is not None else "policy.pt"

        env = gym.make(task_name, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        ppo_runner.load(resume_path)

        try:
            policy_nn = ppo_runner.alg.policy
        except AttributeError:
            policy_nn = ppo_runner.alg.actor_critic

        export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=str(export_dir), filename=export_name)
        print(f"EXPORTED_POLICY={export_dir / export_name}")
        print(f"CHECKPOINT={resume_path}")
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
