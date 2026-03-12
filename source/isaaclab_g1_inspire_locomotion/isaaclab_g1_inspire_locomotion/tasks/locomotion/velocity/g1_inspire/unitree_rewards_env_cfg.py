import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab_g1_inspire_locomotion.tasks.locomotion.velocity import mdp as custom_mdp

from .default_env_cfg import G1InspireFlatDefaultEnvCfg, G1InspireFlatDefaultPlayEnvCfg


@configclass
class G1InspireFlatUnitreeRewardsEnvCfg(G1InspireFlatDefaultEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Match unitree_rl_lab termination behavior instead of the stricter default trunk-contact terms.
        self.terminations.base_contact = None
        self.terminations.pelvis_height = None
        self.terminations.torso_height = None
        self.terminations.bad_orientation = DoneTerm(
            func=base_mdp.bad_orientation,
            params={"limit_angle": 1.0471975512, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.terminations.base_height = DoneTerm(
            func=base_mdp.root_height_below_minimum,
            params={"minimum_height": 0.2, "asset_cfg": SceneEntityCfg("robot")},
        )

        self.rewards.termination_penalty = None
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.05
        self.rewards.dof_pos_limits.weight = -5.0
        self.rewards.dof_pos_limits.params["asset_cfg"] = SceneEntityCfg("robot")
        self.rewards.dof_torques_l2 = None
        self.rewards.feet_air_time = None
        self.rewards.feet_slide.weight = -0.2
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.joint_deviation_hip.weight = -1.0
        self.rewards.joint_deviation_arms.weight = -0.1
        self.rewards.joint_deviation_torso.weight = -1.0
        self.rewards.joint_deviation_fingers = None

        self.rewards.alive = RewTerm(func=base_mdp.is_alive, weight=0.15)
        self.rewards.joint_vel = RewTerm(func=base_mdp.joint_vel_l2, weight=-0.001)
        self.rewards.energy = RewTerm(func=custom_mdp.energy, weight=-2.0e-5)
        self.rewards.base_height = RewTerm(func=base_mdp.base_height_l2, weight=-10.0, params={"target_height": 0.78})
        self.rewards.gait = RewTerm(
            func=custom_mdp.feet_gait,
            weight=0.5,
            params={
                "period": 0.8,
                "offset": [0.0, 0.5],
                "threshold": 0.55,
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            },
        )
        self.rewards.feet_clearance = RewTerm(
            func=custom_mdp.foot_clearance_reward,
            weight=1.0,
            params={
                "std": 0.05,
                "tanh_mult": 2.0,
                "target_height": 0.1,
                "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            },
        )
        self.rewards.undesired_contacts = RewTerm(
            func=base_mdp.undesired_contacts,
            weight=-1.0,
            params={
                "threshold": 1.0,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
            },
        )


@configclass
class G1InspireFlatUnitreeRewardsPlayEnvCfg(G1InspireFlatDefaultPlayEnvCfg, G1InspireFlatUnitreeRewardsEnvCfg):
    def __post_init__(self):
        G1InspireFlatUnitreeRewardsEnvCfg.__post_init__(self)
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.episode_length_s = 20.0
