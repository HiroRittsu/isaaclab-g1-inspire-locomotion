from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_g1_inspire_locomotion.tasks.locomotion.velocity import mdp as custom_mdp
from .default_env_cfg import G1InspireFlatDefaultEnvCfg, G1InspireFlatDefaultPlayEnvCfg


@configclass
class G1InspireFlatAdvancedEnvCfg(G1InspireFlatDefaultEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.direction_relative_hand_width = RewTerm(
            func=custom_mdp.direction_relative_hand_width_penalty,
            weight=-0.05,
            params={
                "command_name": "base_velocity",
                "torso_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
                "left_hand_cfg": SceneEntityCfg("robot", body_names=["left_wrist_yaw_link"]),
                "right_hand_cfg": SceneEntityCfg("robot", body_names=["right_wrist_yaw_link"]),
            },
        )


@configclass
class G1InspireFlatAdvancedPlayEnvCfg(G1InspireFlatDefaultPlayEnvCfg, G1InspireFlatAdvancedEnvCfg):
    def __post_init__(self):
        G1InspireFlatAdvancedEnvCfg.__post_init__(self)
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.episode_length_s = 20.0
