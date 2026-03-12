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
