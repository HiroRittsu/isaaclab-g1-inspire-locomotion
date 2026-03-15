from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

UNITREE_ROS_DIR = "/workspace/unitree_ros"
G1_INSPIRE_DFQ_URDF = f"{UNITREE_ROS_DIR}/robots/g1_description/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf"
REPO_ROOT = Path(__file__).resolve().parents[5]
G1_INSPIRE_DFQ_USD = REPO_ROOT / "artifacts" / "usd" / "g1_inspire_dfq" / "g1_29dof_rev_1_0_with_inspire_hand_DFQ.usd"


@configclass
class G1UrdfFileCfg(sim_utils.UrdfFileCfg):
    activate_contact_sensors: bool = True
    fix_base: bool = False
    replace_cylinders_with_capsules = True
    joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    )
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )


@configclass
class G1UsdFileCfg(sim_utils.UsdFileCfg):
    activate_contact_sensors: bool = True
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    )
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )


def _g1_inspire_dfq_spawn_cfg():
    if G1_INSPIRE_DFQ_USD.is_file():
        return G1UsdFileCfg(usd_path=str(G1_INSPIRE_DFQ_USD))
    return G1UrdfFileCfg(asset_path=G1_INSPIRE_DFQ_URDF)


G1_INSPIRE_DFQ_CFG = ArticulationCfg(
    spawn=_g1_inspire_dfq_spawn_cfg(),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            "left_hip_pitch_joint": -0.25,
            "right_hip_pitch_joint": -0.25,
            ".*_knee_joint": 0.65,
            ".*_ankle_pitch_joint": -0.4,
            ".*_shoulder_pitch_joint": 0.3,
            "left_shoulder_roll_joint": 0.25,
            "right_shoulder_roll_joint": -0.25,
            ".*_elbow_joint": 0.97,
            "left_wrist_roll_joint": 0.15,
            "right_wrist_roll_joint": -0.15,
            ".*_wrist_pitch_joint": 0.0,
            ".*_wrist_yaw_joint": 0.0,
            ".*_(thumb|index|middle|ring|pinky)_.*joint": 0.0,
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "hip_yaw_pitch_waist_yaw": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_.*", ".*_hip_yaw_.*", "waist_yaw_joint"],
            effort_limit_sim=88.0,
            velocity_limit_sim=32.0,
            stiffness={".*_hip_.*": 100.0, "waist_yaw_joint": 200.0},
            damping={".*_hip_.*": 2.0, "waist_yaw_joint": 5.0},
            armature=0.01,
        ),
        "hip_roll_knee": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_roll_.*", ".*_knee_.*"],
            effort_limit_sim=139.0,
            velocity_limit_sim=20.0,
            stiffness={".*_hip_roll_.*": 100.0, ".*_knee_.*": 200.0},
            damping={".*_hip_roll_.*": 2.0, ".*_knee_.*": 6.0},
            armature=0.01,
        ),
        "arms_ankles_waist": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_.*",
                ".*_elbow_.*",
                ".*_wrist_roll_.*",
                ".*_ankle_.*",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit_sim=25.0,
            velocity_limit_sim=37.0,
            stiffness={
                ".*_shoulder_.*": 40.0,
                ".*_elbow_.*": 40.0,
                ".*_wrist_roll_.*": 40.0,
                ".*_ankle_.*": 60.0,
                "waist_roll_joint": 200.0,
                "waist_pitch_joint": 200.0,
            },
            damping={
                ".*_shoulder_.*": 1.0,
                ".*_elbow_.*": 1.0,
                ".*_wrist_roll_.*": 1.0,
                ".*_ankle_.*": 4.0,
                "waist_roll_joint": 15.0,
                "waist_pitch_joint": 15.0,
            },
            armature=0.01,
        ),
        "wrists": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_pitch_.*", ".*_wrist_yaw_.*"],
            effort_limit_sim=5.0,
            velocity_limit_sim=22.0,
            stiffness=10.0,
            damping=0.5,
            armature=0.001,
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_(thumb|index|middle|ring|pinky)_.*joint",
            ],
            effort_limit_sim=30.0,
            velocity_limit_sim=10.0,
            stiffness=10.0,
            damping=0.2,
            armature=0.001,
        ),
    },
    soft_joint_pos_limit_factor=0.9,
)
