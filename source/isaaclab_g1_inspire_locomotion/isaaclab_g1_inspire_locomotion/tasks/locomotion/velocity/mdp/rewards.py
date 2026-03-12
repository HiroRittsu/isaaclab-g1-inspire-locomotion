from __future__ import annotations

import torch

from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat


def direction_relative_hand_width_penalty(
    env,
    command_name: str,
    torso_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["torso_link"]),
    left_hand_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["L_hand_base_link"]),
    right_hand_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["R_hand_base_link"]),
):
    asset = env.scene[torso_cfg.name]
    torso_pos = asset.data.body_pos_w[:, torso_cfg.body_ids[0], :]
    left_rel_w = asset.data.body_pos_w[:, left_hand_cfg.body_ids[0], :] - torso_pos
    right_rel_w = asset.data.body_pos_w[:, right_hand_cfg.body_ids[0], :] - torso_pos

    yaw_only = yaw_quat(asset.data.root_quat_w)
    left_rel = quat_apply_inverse(yaw_only, left_rel_w)
    right_rel = quat_apply_inverse(yaw_only, right_rel_w)

    cmd_xy = env.command_manager.get_command(command_name)[:, :2]
    cmd_norm = torch.linalg.norm(cmd_xy, dim=1, keepdim=True)
    default_dir = torch.zeros_like(cmd_xy)
    default_dir[:, 0] = 1.0
    direction = torch.where(cmd_norm > 1.0e-4, cmd_xy / cmd_norm.clamp_min(1.0e-4), default_dir)
    perp = torch.stack((-direction[:, 1], direction[:, 0]), dim=1)

    left_width = torch.abs(torch.sum(left_rel[:, :2] * perp, dim=1))
    right_width = torch.abs(torch.sum(right_rel[:, :2] * perp, dim=1))
    return left_width + right_width


def feet_air_time_balance_penalty(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    max_err: float = 0.6,
):
    """Penalize persistent left-right imbalance in air/contact durations."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    if air_time.shape[1] != 2:
        raise ValueError("feet_air_time_balance_penalty expects exactly two feet.")

    air_imbalance = torch.abs(air_time[:, 0] - air_time[:, 1])
    contact_imbalance = torch.abs(contact_time[:, 0] - contact_time[:, 1])
    penalty = torch.clamp(air_imbalance, max=max_err) + torch.clamp(contact_imbalance, max=max_err)

    moving = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return penalty * moving
