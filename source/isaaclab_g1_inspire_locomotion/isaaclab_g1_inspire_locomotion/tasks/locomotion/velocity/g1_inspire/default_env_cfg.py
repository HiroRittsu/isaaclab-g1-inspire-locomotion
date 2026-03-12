import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg, G1FlatEnvCfg_PLAY

from isaaclab_g1_inspire_locomotion.assets.robots.g1_inspire import G1_INSPIRE_DFQ_CFG
from isaaclab_g1_inspire_locomotion.tasks.locomotion.velocity import mdp as custom_mdp


POLICY_JOINTS = [
    ".*_hip_.*",
    ".*_knee_joint",
    ".*_ankle_.*",
    ".*_shoulder_.*",
    ".*_elbow_joint",
    ".*_wrist_.*",
]


@configclass
class G1InspireFlatDefaultEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = G1_INSPIRE_DFQ_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.actions.joint_pos.joint_names = POLICY_JOINTS

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.observations.policy.enable_corruption = False

        self.rewards.joint_deviation_arms.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_.*",
            ],
        )
        self.rewards.joint_deviation_torso.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
        )
        self.rewards.joint_deviation_fingers = None
        self.rewards.action_rate_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.feet_slide.weight = -0.2
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

        self.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names="torso_link",
        )
        self.terminations.bad_orientation = None
        self.terminations.pelvis_height = None
        self.terminations.torso_height = None
        self.terminations.base_height = DoneTerm(
            func=base_mdp.root_height_below_minimum,
            params={"minimum_height": 0.2, "asset_cfg": SceneEntityCfg("robot")},
        )


@configclass
class G1InspireFlatDefaultPlayEnvCfg(G1FlatEnvCfg_PLAY, G1InspireFlatDefaultEnvCfg):
    def __post_init__(self):
        G1InspireFlatDefaultEnvCfg.__post_init__(self)
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.episode_length_s = 20.0
