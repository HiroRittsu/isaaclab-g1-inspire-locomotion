from __future__ import annotations

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def body_height_below_minimum(
    env,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Terminate when the specified rigid body height falls below a threshold."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2] < minimum_height
