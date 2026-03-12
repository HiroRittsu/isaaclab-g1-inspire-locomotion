from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat


def energy(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


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


def foot_clearance_reward(env, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float):
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_gait(env, period: float, offset: list[float], sensor_cfg: SceneEntityCfg, threshold: float = 0.5, command_name=None):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward
