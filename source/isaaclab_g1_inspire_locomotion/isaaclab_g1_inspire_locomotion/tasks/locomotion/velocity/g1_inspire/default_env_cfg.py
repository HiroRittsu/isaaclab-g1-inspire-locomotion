import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as task_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg, G1FlatEnvCfg_PLAY

from isaaclab_g1_inspire_locomotion.assets.robots.g1_inspire import G1_INSPIRE_DFQ_CFG
from isaaclab_g1_inspire_locomotion.tasks.locomotion.velocity import mdp as custom_mdp


POLICY_JOINTS = [
    ".*_hip_.*",
    ".*_knee_joint",
    ".*_ankle_.*",
    ".*_shoulder_.*",
    ".*_elbow_joint",
]


@configclass
class G1InspireFlatDefaultEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = G1_INSPIRE_DFQ_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.actions.joint_pos.joint_names = POLICY_JOINTS

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.2, 1.2)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = EventTerm(
            func=task_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(8.0, 12.0),
            params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
        )
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
        self.rewards.track_lin_vel_xy_exp.weight = 5.0
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.joint_deviation_hip.weight = -0.3
        self.rewards.joint_deviation_arms.weight = -0.4
        self.rewards.joint_deviation_torso.weight = -1.0
        self.rewards.joint_deviation_fingers = None
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.feet_slide.weight = -0.2
        self.rewards.direction_relative_hand_width = None
        self.rewards.termination_penalty = RewTerm(func=base_mdp.is_terminated, weight=-200.0)
        self.rewards.alive = RewTerm(func=base_mdp.is_alive, weight=0.15)
        self.rewards.joint_vel = RewTerm(func=base_mdp.joint_vel_l2, weight=-0.001)
        self.rewards.energy = RewTerm(func=custom_mdp.energy, weight=-2.0e-5)
        self.rewards.undesired_contacts = None
        self.rewards.base_height = RewTerm(
            func=base_mdp.base_height_l2,
            weight=-8.0,
            params={"target_height": 0.78},
        )
        self.rewards.joint_deviation_support_legs = RewTerm(
            func=base_mdp.joint_deviation_l1,
            weight=-0.4,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[".*_hip_pitch_joint", ".*_knee_joint", ".*_ankle_pitch_joint"],
                )
            },
        )
        self.rewards.gait = RewTerm(
            func=custom_mdp.feet_gait,
            weight=0.2,
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
            weight=0.4,
            params={
                "std": 0.05,
                "tanh_mult": 2.0,
                "target_height": 0.1,
                "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            },
        )

        self.terminations.base_contact = None
        self.terminations.bad_orientation = DoneTerm(
            func=base_mdp.bad_orientation,
            params={"limit_angle": 1.3, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.terminations.pelvis_height = None
        self.terminations.torso_height = None
        self.terminations.base_height = DoneTerm(
            func=base_mdp.root_height_below_minimum,
            params={"minimum_height": 0.15, "asset_cfg": SceneEntityCfg("robot")},
        )


@configclass
class G1InspireFlatDefaultPlayEnvCfg(G1FlatEnvCfg_PLAY, G1InspireFlatDefaultEnvCfg):
    def __post_init__(self):
        G1InspireFlatDefaultEnvCfg.__post_init__(self)
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.episode_length_s = 20.0
