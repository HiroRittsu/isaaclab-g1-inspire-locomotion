import gymnasium as gym


gym.register(
    id="Isaac-G1-Inspire-Flat-Default-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.default_env_cfg:G1InspireFlatDefaultEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.default_env_cfg:G1InspireFlatDefaultPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_default_cfg:G1InspireFlatDefaultPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-Inspire-Flat-Advanced-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.advanced_env_cfg:G1InspireFlatAdvancedEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.advanced_env_cfg:G1InspireFlatAdvancedPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_advanced_cfg:G1InspireFlatAdvancedPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-Inspire-Flat-LooseTermination-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loose_termination_env_cfg:G1InspireFlatLooseTerminationEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.loose_termination_env_cfg:G1InspireFlatLooseTerminationPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_loose_termination_cfg:G1InspireFlatLooseTerminationPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-Inspire-Flat-UnitreeRewards-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.unitree_rewards_env_cfg:G1InspireFlatUnitreeRewardsEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.unitree_rewards_env_cfg:G1InspireFlatUnitreeRewardsPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_unitree_rewards_cfg:G1InspireFlatUnitreeRewardsPPORunnerCfg",
    },
)
