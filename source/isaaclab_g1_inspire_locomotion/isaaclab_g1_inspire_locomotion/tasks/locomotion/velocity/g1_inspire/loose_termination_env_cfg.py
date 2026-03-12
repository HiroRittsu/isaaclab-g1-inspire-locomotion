import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .default_env_cfg import G1InspireFlatDefaultEnvCfg, G1InspireFlatDefaultPlayEnvCfg


@configclass
class G1InspireFlatLooseTerminationEnvCfg(G1InspireFlatDefaultEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.terminations.bad_orientation = DoneTerm(
            func=base_mdp.bad_orientation,
            params={"limit_angle": 1.0, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.terminations.pelvis_height = DoneTerm(
            func=base_mdp.root_height_below_minimum,
            params={"minimum_height": 0.3, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.terminations.torso_height.params["minimum_height"] = 0.3


@configclass
class G1InspireFlatLooseTerminationPlayEnvCfg(
    G1InspireFlatDefaultPlayEnvCfg, G1InspireFlatLooseTerminationEnvCfg
):
    def __post_init__(self):
        G1InspireFlatLooseTerminationEnvCfg.__post_init__(self)
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.episode_length_s = 20.0
