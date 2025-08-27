# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable reward functions.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to include
the reward introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import isaaclab.utils.math as math_utils
"""
General.
"""


def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for being alive."""
    return (~env.termination_manager.terminated).float()


def is_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.termination_manager.terminated.float()


class is_terminated_term(ManagerTermBase):
    """Penalize termination for specific terms that don't correspond to episodic timeouts.

    The parameters are as follows:

    * attr:`term_keys`: The termination terms to penalize. This can be a string, a list of strings
      or regular expressions. Default is ".*" which penalizes all terminations.

    The reward is computed as the sum of the termination terms that are not episodic timeouts.
    This means that the reward is 0 if the episode is terminated due to an episodic timeout. Otherwise,
    if two termination terms are active, the reward is 2.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)
        # find and store the termination terms
        term_keys = cfg.params.get("term_keys", ".*")
        self._term_names = env.termination_manager.find_terms(term_keys)

    def __call__(self, env: ManagerBasedRLEnv, term_keys: str | list[str] = ".*") -> torch.Tensor:
        # Return the unweighted reward for the termination terms
        reset_buf = torch.zeros(env.num_envs, device=env.device)
        for term in self._term_names:
            # Sums over terminations term values to account for multiple terminations in the same step
            reset_buf += env.termination_manager.get_term(term)

        return (reset_buf * (~env.termination_manager.time_outs)).float()


"""
Root penalties.
"""


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)


def body_lin_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the linear acceleration of bodies using L2-kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.norm(asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :], dim=-1), dim=1)


"""
Joint penalties.
"""


def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def joint_vel_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation using an L1-kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def joint_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def joint_vel_limits(
    env: ManagerBasedRLEnv, soft_ratio: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint velocities if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint velocity and the soft limits.

    Args:
        soft_ratio: The ratio of the soft limits to be used.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = (
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
        - asset.data.soft_joint_vel_limits[:, asset_cfg.joint_ids] * soft_ratio
    )
    # clip to max error = 1 rad/s per joint to avoid huge penalties
    out_of_limits = out_of_limits.clip_(min=0.0, max=1.0)
    return torch.sum(out_of_limits, dim=1)


"""
Action penalties.
"""


def applied_torque_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize applied torques if they cross the limits.

    This is computed as a sum of the absolute value of the difference between the applied torques and the limits.

    .. caution::
        Currently, this only works for explicit actuators since we manually compute the applied torques.
        For implicit actuators, we currently cannot retrieve the applied torques from the physics engine.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    # TODO: We need to fix this to support implicit joints.
    out_of_limits = torch.abs(
        asset.data.applied_torque[:, asset_cfg.joint_ids] - asset.data.computed_torque[:, asset_cfg.joint_ids]
    )
    return torch.sum(out_of_limits, dim=1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action), dim=1)


"""
Contact sensor.
"""


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)


def desired_contacts(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Penalize if none of the desired contacts are present."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > threshold
    )
    zero_contact = (~contacts).all(dim=1)
    return 1.0 * zero_contact


def contact_forces(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # compute the violation
    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - threshold
    # compute the penalty
    return torch.sum(violation.clip(min=0.0), dim=1)


"""
Velocity-tracking rewards.
"""


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / std**2)

##################################    
# AnimalMath bonus for repeatability of external target
    
def miander_tracking_reward(env) -> torch.Tensor:
    """
    [0 … 1] — награда за совпадение активных DOF с целями.

    • mask  = dof_mask  (1 → активная ось)
    • target= target_joint_pose
    • reward = 1 − MSE/4
    """
    asset = env.scene["robot"]

    # --- команды ----------------------------------------------------
    mask_cmd    = env.command_manager.get_term("dof_mask").command          # (N,J)
    target_cmd  = env.command_manager.get_term("target_joint_pose").command # (N,J)
    
    # print(f"mask_cmd {mask_cmd}")
    # print(f"target_cmd {target_cmd}")
    # print()

    # bool-маска активных DOF
    mask = mask_cmd > 0.5

    # --- нормализованные текущие углы ------------------------------
    joint_pos = asset.data.joint_pos
    joint_min = asset.data.soft_joint_pos_limits[..., 0]
    joint_max = asset.data.soft_joint_pos_limits[..., 1]
    offset    = (joint_min + joint_max) * 0.5
    norm_pos  = 2.0 * (joint_pos - offset) / (joint_max - joint_min + 1e-6)

    # --- MSE только по активным DOF --------------------------------
    se        = (norm_pos - target_cmd).pow(2)
    se_masked = se * mask
    active_n  = mask.sum(dim=1)                          # (N,)

    mse = torch.where(
        active_n > 0,
        se_masked.sum(dim=1) / active_n.clamp(min=1),
        torch.zeros_like(active_n, dtype=se.dtype),
    )

    # --- reward -----------------------------------------------------
    reward = torch.where(
        active_n > 0,
        1.0 - mse / 4.0,
        torch.zeros_like(mse),
    ).clamp_(0.0, 1.0)

    return reward # - 0.5        # если нужен сдвиг

def miander_tracking_reward_exp(env, eps_half: float = 0.25) -> torch.Tensor:
    """
    Гладкий трекинг: r = exp(-beta * MSE_active), где beta = ln(2)/eps_half^2.
    • Масштаб 'eps_half' задаёт «допуск»: при RMS-ошибке = eps_half ревард = 0.5.
    • Всегда даёт ненулевой градиент, даже при большой ошибке.
    """
    import math
    asset = env.scene["robot"]

    # --- команды ---
    mask_cmd   = env.command_manager.get_term("dof_mask").command          # (N,J)
    target_cmd = env.command_manager.get_term("target_joint_pose").command # (N,J)
    mask = mask_cmd > 0.5

    # --- нормализованные текущие углы ---
    q     = asset.data.joint_pos
    qmin  = asset.data.soft_joint_pos_limits[..., 0]
    qmax  = asset.data.soft_joint_pos_limits[..., 1]
    mid   = 0.5 * (qmin + qmax)
    scl   = 2.0 / (qmax - qmin + 1e-6)
    qn    = (q - mid) * scl

    # --- MSE по активным DOF ---
    e2 = (qn - target_cmd).pow(2) * mask
    active_n = mask.sum(dim=1)
    mse = torch.where(
        active_n > 0,
        e2.sum(dim=1) / active_n.clamp(min=1),
        torch.zeros_like(active_n, dtype=qn.dtype),
    )

    # --- экспоненциальный колокол ---
    beta = math.log(2.0) / (eps_half * eps_half + 1e-12)
    r = torch.exp(-beta * mse)

    # если активных DOF нет — 0 (как в вашей версии)
    r = torch.where(active_n > 0, r, torch.zeros_like(r))
    return r

    
#def miander_masked_penalty(env: MathManagerBasedRLEnv) -> torch.Tensor:
#    alpha = 4.0
#    asset = env.scene["robot"]

#    joint_pos = asset.data.joint_pos[:, :]
#    joint_min = asset.data.soft_joint_pos_limits[:, :, 0]
#    joint_max = asset.data.soft_joint_pos_limits[:, :, 1]

#    offset = (joint_min + joint_max) * 0.5
#    norm_joint_pos = 2 * (joint_pos - offset) / (joint_max - joint_min + 1e-6)

    # Создаём булеву маску "неактивных" DOF
#    inverse_mask = env.switcher_mask <= 0.5  # shape: [n_envs, n_joints]

#    abs_error = torch.abs(norm_joint_pos - env.targets)
#    masked_error = abs_error * inverse_mask  # inverse_mask — булева

#    error = masked_error.sum(dim=1)
#    penalty = torch.exp(-alpha * error)

    # penalty даём только если есть хоть один неактивный DOF
#    has_inactive_dof = inverse_mask.any(dim=1)
#    penalty = torch.where(has_inactive_dof, penalty, torch.zeros_like(penalty))

#    return penalty

def miander_untracking_reward(
    env,
    command_name: str = "base_velocity",
    lin_cmd_threshold: float = 0.05,
    leg_bits: tuple[int, ...] = (0,1,3,4,7,8,11,12,15,16,19,20),
) -> torch.Tensor:
    asset = env.scene["robot"]
    mask_cmd = env.command_manager.get_term("dof_mask").command         # (N,J)
    inverse_mask = mask_cmd <= 0.5                                      # неактивные

    q = asset.data.joint_pos
    qmin = asset.data.soft_joint_pos_limits[..., 0]
    qmax = asset.data.soft_joint_pos_limits[..., 1]
    qn = 2.0 * (q - (qmin + qmax) * 0.5) / (qmax - qmin + 1e-6)

    init_norm = env.JOINT_INIT_POS_NORM.to(qn.device).unsqueeze(0)      # (1,J)
    se = (qn - init_norm).pow(2)

    # --- движение вперёд? → выключаем вклад суставов ног из этого терма
    cmd = env.command_manager.get_term(command_name).command
    move_gate = cmd[:, :2].norm(dim=1) > lin_cmd_threshold              # (N,)
    W = inverse_mask.float()                                            # (N,J)
    if move_gate.any():
        if not hasattr(env, "_leg_bits_tensor"):
            env._leg_bits_tensor = torch.as_tensor(leg_bits, device=env.device, dtype=torch.long)
        W[move_gate][:, env._leg_bits_tensor] = 0.0

    se_masked = se * W
    denom = W.sum(dim=1).clamp_min(1.0)
    mse = se_masked.sum(dim=1) / denom
    return (1.0 - mse / 4.0).clamp(0.0, 1.0)




def pelvis_height_target_reward(env: MathManagerBasedRLEnv,
                                target: float = 0.74,
                                alpha: float = 0.2) -> torch.Tensor:
    """
    Экспоненциальная награда: r = exp(-alpha * |z - target|)

    Args:
        env:   среда MathManagerBasedRLEnv.
        target: желаемая высота таза в метрах.
        alpha:  крутизна колокола 

    Returns:
        Tensor[num_envs] — reward в диапазоне (0‥1].
    """
    # Берём Z‑координату pelvis
    asset = env.scene["robot"]
    pelvis_z = asset.data.root_pos_w[:, 2]           # shape [N]
    # print(pelvis_z)

    error = torch.abs(pelvis_z - target)      # |z − 0.7|
    reward = torch.exp(-alpha * error)     # e^(−α·err)

    return reward
    


def feet_separation_and_alignment_penalty(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",

    # (0) прямой штраф по суставам ankle_pitch → к нейтрали
    ankle_pitch_joint_names=("left_ankle_pitch_joint", "right_ankle_pitch_joint"),
    w_ankle_neutral: float = 2.0,

    # (1) «ступня параллельно полу» через нормаль подошвы
    w_tilt: float = 1.0,
    foot_normal_local=(0.0, 0.0, 1.0),

    # (1b) «против носка/пятки»: продольная ось должна быть горизонтальна
    w_pitch_flat: float = 2.0,
    foot_forward_local=(1.0, 0.0, 0.0),

    # (2) выравнивание стоп с направлением таза (только при двухопорной фазе)
    w_align: float = 0.5,

    # (3) продольная длина шага (только при двухопорной)
    w_stride: float = 0.7,
    step_gain: float = 0.40,
    beta_stride: float = 3.0,
    v_near_zero: float = 0.05,
    w_opposite_at_zero: float = 0.5,

    # (4) перехлёст (только при двухопорной)
    w_cross: float = 0.7,
    beta_cross: float = 6.0,

    # боковая ширина (только при двухопорной)
    shoulder_width: float = 0.35,
    beta_sep: float = 2.0,
    w_sep: float = 1.0,

    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """
    Суммарный penalty ≥ 0 за «некорректную» постановку стоп при контактах.
    Усиливает: (а) плоскость касания (без носка/пятки), (б) нейтраль по ankle_pitch,
    (в) корректную геометрию шага и отсутствие перехлёста/узкой стойки.
    """
    device = env.device
    robot: Articulation | RigidObject = env.scene[asset_cfg.name]
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # --- helpers: локальные оси для обеих стоп (1,2,3)
    def _per_foot(vec):
        v = torch.as_tensor(vec, dtype=torch.float32, device=device)
        return v.view(1, 1, 3).expand(1, 2, 3)

    nrm_loc = _per_foot(foot_normal_local)     # нормаль подошвы в ЛСК стопы
    fwd_loc = _per_foot(foot_forward_local)    # продольная ось стопы в ЛСК
    world_up = torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 1, 3)

    # --- контакты (устойчиво к шуму за счёт .amax по истории)
    f_hist = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]    # (N,H,2,3)
    fmag = f_hist.norm(dim=-1).amax(dim=1)                                  # (N,2)
    contacts = fmag > contact_force_threshold                               # (N,2) [L,R]
    both_down = contacts.all(dim=1)                                         # (N,)
    n_down = contacts.float().sum(dim=1).clamp(min=1.0)                     # (N,)

    # --- позы/ориентации
    feet_pos_w  = robot.data.body_pos_w[:, sensor_cfg.body_ids, :]          # (N,2,3)
    feet_quat_w = robot.data.body_quat_w[:, sensor_cfg.body_ids, :]         # (N,2,4)
    root_pos_w  = robot.data.root_pos_w                                     # (N,3)
    root_quat_w = robot.data.root_quat_w                                    # (N,4)

    base_fwd_w = math_utils.quat_apply(
        root_quat_w, torch.tensor([1.0, 0.0, 0.0], device=device).expand_as(root_pos_w)
    )
    base_lat_w = math_utils.quat_apply(
        root_quat_w, torch.tensor([0.0, 1.0, 0.0], device=device).expand_as(root_pos_w)
    )
    f_xy = base_fwd_w[:, :2]
    f_xy = f_xy / f_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    l_xy = base_lat_w[:, :2]
    l_xy = l_xy / l_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    # ---------- (0) ankle_pitch → к нейтрали (только для контактирующих ног)
    if not hasattr(env, "_ankle_pitch_ids"):
        jnames = list(robot.data.joint_names)
        try:
            env._ankle_pitch_ids = torch.tensor(
                [jnames.index(ankle_pitch_joint_names[0]),
                 jnames.index(ankle_pitch_joint_names[1])],
                device=device, dtype=torch.long
            )
        except ValueError as e:
            raise RuntimeError(f"Не найден ankle_pitch сустав: {e}")

        init_norm = env.JOINT_INIT_POS_NORM.to(device)
        env._ankle_pitch_init_norm = init_norm[env._ankle_pitch_ids]  # (2,)

    jidx = env._ankle_pitch_ids
    q    = robot.data.joint_pos
    qmin = robot.data.soft_joint_pos_limits[..., 0]
    qmax = robot.data.soft_joint_pos_limits[..., 1]
    qmid = 0.5 * (qmin + qmax)
    qhal = 0.5 * (qmax - qmin)
    qn   = ((q - qmid) / (qhal + 1e-6)).clamp(-1.0, 1.0)

    qn_ank    = qn[:, jidx]                                # (N,2)
    qn_target = env._ankle_pitch_init_norm.view(1, 2)      # (1,2)
    ankle_err = (qn_ank - qn_target).abs()                 # (N,2)
    ankle_neutral_pen = (ankle_err * contacts.float()).sum(dim=1) / n_down

    # ---------- (1) «ступни параллельно полу» через нормаль подошвы
    nrm_w  = math_utils.quat_apply(feet_quat_w, nrm_loc.expand_as(feet_pos_w))  # (N,2,3)
    cos_up = (nrm_w * world_up).sum(dim=-1).abs().clamp(0.0, 1.0)               # (N,2)
    tilt_each = 1.0 - cos_up
    tilt_pen  = (tilt_each * contacts.float()).sum(dim=1) / n_down

    # ---------- (1b) НОВОЕ: «против носка/пятки» — продольная ось горизонтальна
    foot_fwd_w = math_utils.quat_apply(feet_quat_w, fwd_loc.expand_as(feet_pos_w))  # (N,2,3)
    pitch_vert_each = foot_fwd_w[:, :, 2].abs()                                     # |z-компонента «вперёд»|
    pitch_flat_pen  = (pitch_vert_each * contacts.float()).sum(dim=1) / n_down

    # ---------- (2) выравнивание стоп по направлению таза (только при двухопорной)
    foot_dir_xy = foot_fwd_w[:, :, :2]
    foot_dir_xy = foot_dir_xy / foot_dir_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    cos_to_pelvis = (foot_dir_xy * f_xy.unsqueeze(1)).sum(dim=-1).clamp(-1.0, 1.0)  # (N,2)
    align_term = 0.5 * (1.0 - cos_to_pelvis).mean(dim=1)
    align_pen  = torch.where(both_down, align_term, torch.zeros_like(align_term))

    # ---------- (3) продольная длина шага / «противоположность» при почти нулевой V
    rel_xy = feet_pos_w[:, :, :2] - root_pos_w[:, :2].unsqueeze(1)  # (N,2,2)
    sL = (rel_xy[:, 0, :] * f_xy).sum(dim=-1)
    sR = (rel_xy[:, 1, :] * f_xy).sum(dim=-1)
    d_long = (sL - sR).abs()

    v_cmd = env.command_manager.get_term(command_name).command[:, :2]
    v_mag = v_cmd.norm(dim=-1)
    d_des = step_gain * v_mag
    stride_term = 1.0 - torch.exp(-beta_stride * (d_long - d_des).abs())
    stride_pen  = torch.where(both_down, stride_term, torch.zeros_like(stride_term))

    opposite_bool  = (sL * sR) < 0
    low_speed_gate = (v_near_zero - v_mag).clamp(min=0.0) / max(v_near_zero, 1e-6)
    opposite_pen   = w_opposite_at_zero * (opposite_bool.float() * low_speed_gate)

    # ---------- (4) перехлёст (только при двухопорной)
    yL = (rel_xy[:, 0, :] * l_xy).sum(dim=-1)
    yR = (rel_xy[:, 1, :] * l_xy).sum(dim=-1)
    cross_depth = torch.relu(-yL) + torch.relu(+yR)
    cross_term  = 1.0 - torch.exp(-beta_cross * cross_depth)
    cross_pen   = torch.where(both_down, cross_term, torch.zeros_like(cross_term))

    # ---------- боковая ширина (только при двухопорной)
    dist_xy = (feet_pos_w[:, 0, :2] - feet_pos_w[:, 1, :2]).norm(dim=-1)
    sep_term = 1.0 - torch.exp(-beta_sep * (dist_xy - shoulder_width).abs())
    sep_pen  = torch.where(both_down, sep_term, torch.zeros_like(sep_term))

    # ---------- суммарный penalty
    penalty = (
        w_ankle_neutral * ankle_neutral_pen
      + w_tilt          * tilt_pen
      + w_pitch_flat    * pitch_flat_pen
      + w_align         * align_pen
      + w_stride        * stride_pen
      +                   opposite_pen
      + w_cross         * cross_pen
      + w_sep           * sep_pen
    )
    return penalty.clamp_min(0.0)






# Additional
def masked_progress_reward(env, eps=0.02):
    robot = env.scene["robot"]
    q     = robot.data.joint_pos
    qd    = robot.data.joint_vel

    qmin = robot.data.soft_joint_pos_limits[...,0]
    qmax = robot.data.soft_joint_pos_limits[...,1]
    mid  = 0.5*(qmin+qmax)
    scl  = 2.0/(qmax-qmin+1e-6)

    qn  = (q - mid) * scl                # норм. поза ∈[-1,1]
    qdn = qd * scl                       # норм. скорость
    tgt = env.command_manager.get_term("target_joint_pose").command  # (N,J) в норм. шкале
    msk = env.command_manager.get_term("dof_mask").command > 0.5     # bool

    err = (qn - tgt)                     # куда надо двигаться: «к 0»
    # прогресс = уменьшение |err| ⇒ знак(err) противоположен скорости
    prog = -(err * qdn)                  # положительно, если движемся к цели
    prog = torch.where(msk, prog, torch.zeros_like(prog))
    # усредняем по активным суставам; если активных нет — 0
    cnt  = msk.sum(dim=1).clamp(min=1)
    r    = prog.sum(dim=1) / cnt
    # мягкий клип
    return r.clamp(min=-1.0, max=1.0)
    
def unmasked_stillness_penalty(
    env,
    w_vel: float = 1.0,
    w_tau: float = 0.0,   # включи >0, когда захочешь учитывать усилия
):
    """
    Позитивный cost для НЕмаскированных DOF (mask==0):
      • |joint_vel| всегда
      • |applied_torque| если доступен
    """
    robot = env.scene["robot"]

    # неактивные = (1 - mask)  — маску берём из команды dof_mask
    inactive_mask = (1.0 - env.command_manager.get_term("dof_mask").command).float()  # [N,J]
    denom = inactive_mask.sum(dim=1).clamp_min(1.0)

    # скорость суставов
    qd = robot.data.joint_vel  # [N,J]
    vel_cost = (qd.abs() * inactive_mask).sum(dim=1) / denom

    # усилие (в твоей сборке есть applied_torque; при отсутствии — пробуем computed_torque)
    tau = getattr(robot.data, "applied_torque", None)
    if tau is None:
        tau = getattr(robot.data, "computed_torque", None)

    if tau is not None:
        tau_cost = (tau.abs() * inactive_mask).sum(dim=1) / denom
    else:
        tau_cost = torch.zeros_like(vel_cost)

    return w_vel * vel_cost + w_tau * tau_cost
 
    
    
def masked_success_bonus(env, eps=0.03, bonus=1.0):
    robot = env.scene["robot"]
    q     = robot.data.joint_pos
    qmin  = robot.data.soft_joint_pos_limits[...,0]
    qmax  = robot.data.soft_joint_pos_limits[...,1]
    mid   = 0.5*(qmin+qmax)
    scl   = 2.0/(qmax-qmin+1e-6)

    qn  = (q - mid) * scl
    tgt = env.command_manager.get_term("target_joint_pose").command
    msk = env.command_manager.get_term("dof_mask").command > 0.5

    ok = ((qn - tgt).abs() <= eps) | (~msk)      # немаскированные считаем «ок»
    all_ok = ok.all(dim=1)
    return bonus * all_ok.float()
    
def lateral_slip_penalty(env, command_name="base_velocity"):
    robot = env.scene["robot"]
    cmd   = env.command_manager.get_term(command_name).command  # (N,3): vx, vy, wz in base
    v_b   = robot.data.root_lin_vel_b[:, :2]                    # (N,2)
    # если команда почти нулевая — не штрафуем
    mag = cmd[:,:2].norm(dim=1, keepdim=True) + 1e-6
    dir = cmd[:,:2] / mag
    # поперечная составляющая
    lat = v_b - (v_b*dir).sum(dim=1, keepdim=True)*dir
    return lat.norm(dim=1)    # положительное число   
    
def com_over_support_reward_fast(
    env,
    sensor_cfg: SceneEntityCfg,                 # ContactSensor с телами-опорами (обычно стопы)
    asset_cfg : SceneEntityCfg = SceneEntityCfg("robot"),
    contact_force_threshold: float = 5.0,       # Н: фильтр шума
    sigma: float = 0.06,                        # м: «радиус точности» (строгость)
    weighted: bool = True,                      # True → средняя точка взвешена силой
) -> torch.Tensor:
    """
    R ∈ [0,1] (shape: [N]). Максимум, когда проекция CoM в XY совпадает с
    опорной точкой (средней/взвешенной) по контактирующим телам из sensor_cfg.
    """
    device = env.device
    robot  = env.scene[asset_cfg.name]
    cs     = env.scene.sensors[sensor_cfg.name]

    # ---------- контакты (N,K) ----------
    # net_forces_w_history: [N, hist, B, 3] → по выбранным body_ids и max по истории
    fmag = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).amax(dim=1)  # (N,K)
    contacts = fmag > contact_force_threshold                                                    # (N,K)
    any_contact = contacts.any(dim=1)                                                            # (N,)

    # ---------- опорная точка XY (N,2) ----------
    feet_xy = robot.data.body_pos_w[:, sensor_cfg.body_ids, :2]                                  # (N,K,2)

    if weighted:
        w = (fmag * contacts.float())                                                            # (N,K)
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)
        support_xy = (feet_xy * w.unsqueeze(-1)).sum(dim=1)                                     # (N,2)
    else:
        m = contacts.float()
        support_xy = (feet_xy * m.unsqueeze(-1)).sum(dim=1) / m.sum(dim=1, keepdim=True).clamp_min(1e-6)

    # ---------- центр масс XY (N,2) ----------
    # если доступен com_pos_w — используем его (самый быстрый путь)
    if hasattr(robot.data, "com_pos_w"):
        com_xy = robot.data.com_pos_w[:, :2]
    else:
        # fallback: по массам тел
        pos_w = robot.data.body_pos_w[:, :, :2]                                                  # (N,B,2)
        masses = None
        for attr in ("body_masses", "link_masses", "masses"):
            if hasattr(robot.data, attr):
                masses = getattr(robot.data, attr)                                               # (N,B)
                break
        if masses is None:
            com_xy = pos_w.mean(dim=1)                                                           # грубо, без масс
        else:
            m = masses.unsqueeze(-1)                                                             # (N,B,1)
            com_xy = (pos_w * m).sum(dim=1) / m.sum(dim=1).clamp_min(1e-6)                      # (N,2)

    # ---------- награда ----------
    d2 = (com_xy - support_xy).square().sum(dim=1)                                               # (N,)
    inv_two_sigma2 = 0.5 / (sigma * sigma)
    reward = torch.exp(-d2 * inv_two_sigma2)                                                     # (N,)
    return torch.where(any_contact, reward, torch.zeros_like(reward))
    
def no_command_motion_penalty(
    env,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),

    # когда |cmd_lin| < lin_deadband → штраф почти макс; чем меньше deadband, тем «жёстче»
    lin_deadband: float = 0.05,   # м/с
    ang_deadband: float = 0.05,   # рад/с

    # нормировочные масштабы (примерно под «типичные» макс-скорости)
    lin_scale: float = 1.0,       # м/с  → влияет на величину линейного штрафа
    ang_scale: float = 1.0,       # рад/с → влияет на величину углового штрафа
) -> torch.Tensor:
    """
    Штраф за движение, когда НЕТ команды на движение.
    penalty = gate_lin * (||v_xy||/lin_scale)^2 + gate_ang * (|w_z|/ang_scale)^2,
    где gate_* ≈ 1 при маленькой команде и → 0 при росте команды.
    """
    asset = env.scene[asset_cfg.name]

    # команда [vx, vy, wz] в базе
    cmd = env.command_manager.get_term(command_name).command  # (N,3)
    cmd_lin_mag = cmd[:, :2].norm(dim=1)                      # (N,)
    cmd_ang_mag = cmd[:, 2].abs()                             # (N,)

    # скорости базы
    v_xy = asset.data.root_lin_vel_b[:, :2]                   # (N,2)
    w_z  = asset.data.root_ang_vel_b[:, 2]                    # (N,)

    # плавные «шторки» (1 при нулевой команде → 0 около deadband и дальше)
    # экспонента даёт гладкую и дифференцируемую форму
    gate_lin = torch.exp(- (cmd_lin_mag / max(lin_deadband, 1e-6))**2)  # (N,)
    gate_ang = torch.exp(- (cmd_ang_mag / max(ang_deadband, 1e-6))**2)  # (N,)

    lin_term = (v_xy.norm(dim=1) / max(lin_scale, 1e-6))**2
    ang_term = (w_z.abs() / max(ang_scale, 1e-6))**2

    penalty = gate_lin * lin_term + gate_ang * ang_term
    return penalty

def joint_limit_saturation_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    margin: float = 0.10,     # «запретная зона» у границы в норм. шкале: 0.10 = 10% от края
    beta: float = 8.0,        # крутизна роста штрафа при заходе в margin
    use_mask: bool = False,   # если True — учитываем только активные DOF по dof_mask
    mask_name: str = "dof_mask",
) -> torch.Tensor:
    """
    ПЕНАЛЬТИ за «прилипание к лимитам» ещё ДО их пересечения.

    • Нормируем углы в [-1,1] по soft limits.
    • Считаем slack = 1 - |q_n|  (насколько далеко от края).
    • Если slack < margin → есть нарушение. violation = (margin - slack)/margin ∈ [0,1].
    • per_joint = 1 - exp(-beta * violation)  — гладкий рост к 1 у самой границы.
    • Возврат: средний штраф по (активным|всем) DOF: shape (N,), ≥0.
    """
    asset = env.scene[asset_cfg.name]
    q     = asset.data.joint_pos
    qmin  = asset.data.soft_joint_pos_limits[..., 0]
    qmax  = asset.data.soft_joint_pos_limits[..., 1]

    mid   = 0.5 * (qmin + qmax)
    half  = 0.5 * (qmax - qmin)
    qn    = ((q - mid) / (half + 1e-6)).clamp(-1.0, 1.0)   # норм. позы ∈ [-1,1]

    # расстояние до края (в норм. шкале)
    slack = 1.0 - qn.abs()                   # ∈ [0,1]
    # нарушение «зашли в margin-зону»
    violation = (margin - slack).clamp_min(0.0) / max(margin, 1e-6)   # ∈ [0,1]
    per_joint = 1.0 - torch.exp(-beta * violation)                    # гладко к 1

    if use_mask:
        mask = env.command_manager.get_term(mask_name).command > 0.5   # (N,J) bool
        num = mask.sum(dim=1).clamp_min(1)
        pen = (per_joint * mask.float()).sum(dim=1) / num
    else:
        pen = per_joint.mean(dim=1)

    return pen
    
def single_foot_stationary_penalty(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_force_threshold: float = 5.0,
    lin_deadband: float = 0.03,
    ang_deadband: float = 0.03,
    command_name: str = "base_velocity",
    use_mask: bool = True,
    mask_name: str = "dof_mask",
    leg_bits: list[int] = (0, 1, 3, 4, 7, 8, 11, 12, 15, 16, 19, 20),
    scale_by_asymmetry: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Штраф за одноопорную стойку, когда:
      • |cmd| ≈ 0  ИЛИ  ноги «выключены» по маске.

    Возврат: тензор ≥0 формы [N].
    """
    device = env.device
    robot  = env.scene[asset_cfg.name]
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # --- near-zero команды ---
    cmd   = env.command_manager.get_term(command_name).command  # (N,3): vx,vy,wz
    near0 = (cmd[:, :2].norm(dim=1) < lin_deadband) & (cmd[:, 2].abs() < ang_deadband)

    # --- ноги выключены по маске? ---
    if use_mask and hasattr(env.command_manager, "get_term"):
        dof_mask = env.command_manager.get_term(mask_name).command  # (N,J) in {0,1}
        legs_inactive = (dof_mask[:, leg_bits].sum(dim=1) <= 0.0)
    else:
        legs_inactive = torch.zeros_like(near0, dtype=torch.bool)

    gate = near0 # | legs_inactive  # штрафуем, если покой ИЛИ нет команд на ноги
    if not gate.any():
        return torch.zeros(env.num_envs, device=device)

    # --- контакты стоп из истории (устойчиво к шуму) ---
    fmag = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).amax(dim=1)  # (N,2)
    left_contact  = fmag[:, 0] > contact_force_threshold
    right_contact = fmag[:, 1] > contact_force_threshold
    single_support = left_contact ^ right_contact  # XOR

    pen = (single_support & gate).float()

    if scale_by_asymmetry:
        asym = (fmag[:, 0] - fmag[:, 1]).abs() / (fmag.sum(dim=1) + eps)
        pen = pen * (1.0 + asym)  # 1…~2

    return pen
    
def leg_symmetry_idle_reward_norm(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    dof_mask_name: str = "dof_mask",
    obs_key_norm: str = "dof_pos_norm",   # принимаем для совместимости с конфигом
    lin_deadband: float = 0.03,
    ang_deadband: float = 0.03,
    left_dofs: list[int]  = (0, 3, 7, 11, 15, 19),
    right_dofs: list[int] = (1, 4, 8, 12, 16, 20),
    w_sym: float = 0.6,
    w_init: float = 0.4,
    beta_sym: float = 6.0,
    beta_init: float = 4.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Ревард > 0, когда:
      • |cmd| ≈ 0  И  ноги выключены по маске.
    Поощряет:
      (1) схожесть L/R, (2) близость обеих ног к init-позе (в норме [-1,1]).
    """
    device = env.device
    robot  = env.scene[asset_cfg.name]

    # --- врата: нулевые команды ---
    cmd   = env.command_manager.get_term(command_name).command  # (N,3)
    near0 = (cmd[:, :2].norm(dim=1) < lin_deadband) & (cmd[:, 2].abs() < ang_deadband)

    # --- ноги выключены по маске ---
    dof_mask = env.command_manager.get_term(dof_mask_name).command  # (N,J) in {0,1}
    legs_mask = dof_mask[:, list(left_dofs) + list(right_dofs)]
    legs_inactive = legs_mask.sum(dim=1) <= 0.0

    gate = near0 & legs_inactive
    if not gate.any():
        return torch.zeros(env.num_envs, device=device)

    # --- нормированные позы ∈[-1,1] (как в твоих термах) ---
    q     = robot.data.joint_pos
    qmin  = robot.data.soft_joint_pos_limits[..., 0]
    qmax  = robot.data.soft_joint_pos_limits[..., 1]
    mid   = 0.5 * (qmin + qmax)
    half  = 0.5 * (qmax - qmin)
    qn    = ((q - mid) / (half + eps)).clamp(-1.0, 1.0)

    # init-поза уже в норме
    q0n = env.JOINT_INIT_POS_NORM.to(device=device, dtype=qn.dtype)  # (J,)

    # разбиение на ноги
    L = torch.as_tensor(left_dofs,  device=device, dtype=torch.long)
    R = torch.as_tensor(right_dofs, device=device, dtype=torch.long)
    qnL, qnR = qn[:, L], qn[:, R]
    q0L, q0R = q0n[L], q0n[R]

    # --- ошибки ---
    sym_err  = (qnL - qnR).abs().mean(dim=1)  # (N,)
    init_err = 0.5 * ((qnL - q0L).abs().mean(dim=1) + (qnR - q0R).abs().mean(dim=1))

    # --- колокола ---
    r_sym  = torch.exp(-beta_sym  * sym_err)
    r_init = torch.exp(-beta_init * init_err)
    reward = (w_sym * r_sym + w_init * r_init) * gate.float()
    return reward    

def alternating_step_reward(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",

    # врата
    lin_cmd_threshold: float = 0.05,
    use_mask: bool = True,
    mask_name: str = "dof_mask",
    leg_bits: tuple[int, ...] = (0, 1, 3, 4, 7, 8, 11, 12, 15, 16, 19, 20),

    # контакты/устойчивость
    contact_force_threshold: float = 5.0,
    initial_lead: str = "right",   # первая двухопорная: кто спереди
    step_gain: float = 0.40,
    beta_stride: float = 3.0,
) -> torch.Tensor:
    device = env.device
    robot  = env.scene[asset_cfg.name]
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]
    N = env.num_envs

    # --- инициализация внутреннего состояния ---
    _altstep_state_init_(env, N, device, initial_lead)

    # --- гейт: есть команда двигаться вперёд ---
    cmd = env.command_manager.get_term(command_name).command  # (N,3)
    v_mag = cmd[:, :2].norm(dim=1)
    move_gate = v_mag > lin_cmd_threshold

    # ноги «выключены» по маске?
    if use_mask:
        dof_mask = env.command_manager.get_term(mask_name).command
        legs_inactive = (dof_mask[:, list(leg_bits)].sum(dim=1) <= 0.0)
    else:
        legs_inactive = torch.zeros(N, dtype=torch.bool, device=device)

    gate = move_gate & legs_inactive
    if not gate.any():
        # поддерживаем prev_both актуальным, чтобы не терять синхронизацию
        _altstep_state_update_contacts_(env, cs, sensor_cfg, contact_force_threshold)
        return torch.zeros(N, device=device)

    # --- контакты (устойчиво по истории) ---
    fmag = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).amax(dim=1)  # (N,2)
    Lc = fmag[:, 0] > contact_force_threshold
    Rc = fmag[:, 1] > contact_force_threshold
    both_down = Lc & Rc

    # --- кто впереди (по продольной оси базы) ---
    feet_pos_w  = robot.data.body_pos_w[:, sensor_cfg.body_ids, :2]   # (N,2,2)
    root_pos_w  = robot.data.root_pos_w[:, :2]
    root_quat_w = robot.data.root_quat_w

    base_fwd_w = math_utils.quat_apply(
        root_quat_w, torch.tensor([1.0,0.0,0.0], device=device).expand_as(robot.data.root_pos_w)
    )[:, :2]
    f_xy = base_fwd_w / base_fwd_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    rel_xy = feet_pos_w - root_pos_w.unsqueeze(1)
    sL = (rel_xy[:, 0, :] * f_xy).sum(dim=-1)
    sR = (rel_xy[:, 1, :] * f_xy).sum(dim=-1)
    lead_now = (sR > sL).long()  # 0=L, 1=R

    # --- ресеты эпизодов ---
    resets = env.termination_manager.terminated | env.termination_manager.time_outs
    if resets.any():
        _altstep_state_reset_(env, resets, both_down, lead_now, initial_lead)

    # --- liftoff-трекинг с прошлого DS ---
    env._alt_liftoff_L |= ~Lc
    env._alt_liftoff_R |= ~Rc
    liftoff_lead = torch.where(lead_now.bool(), env._alt_liftoff_R, env._alt_liftoff_L)

    # --- вход в двухопорную фазу (rising edge) ---
    ds_on = both_down & (~env._alt_prev_both)

    # --- ожидаемый лидер и оценка длины шага ---
    exp_lead = env._alt_expected_lead              # 0=L, 1=R
    correct_lead = (lead_now == exp_lead)

    d_long = (sL - sR).abs()
    d_des  = step_gain * v_mag
    stride_score = torch.exp(-beta_stride * (d_long - d_des).abs()).clamp(0.0, 1.0)  # 0..1

    reward = (gate & ds_on & correct_lead & liftoff_lead).float() * (0.5 + 0.5 * stride_score)

    # --- переключить ожидаемого лидера и сбросить liftoff-флаги для тех env, где был DS ---
    if ds_on.any():
        env._alt_expected_lead[ds_on] ^= 1
        env._alt_liftoff_L[ds_on] = False
        env._alt_liftoff_R[ds_on] = False

    env._alt_prev_both = both_down
    return reward


# --- helpers ---
def _altstep_state_init_(env, N, device, initial_lead: str):
    if not hasattr(env, "_alt_prev_both"):
        init_lead_bit = 1 if str(initial_lead).lower().startswith("r") else 0
        env._alt_prev_both     = torch.zeros(N, dtype=torch.bool, device=device)
        env._alt_expected_lead = torch.full((N,), init_lead_bit, dtype=torch.long, device=device)
        env._alt_liftoff_L     = torch.zeros(N, dtype=torch.bool, device=device)
        env._alt_liftoff_R     = torch.zeros(N, dtype=torch.bool, device=device)

def _altstep_state_reset_(env, mask, both_down, lead_now, initial_lead: str):
    init_lead_bit = 1 if str(initial_lead).lower().startswith("r") else 0
    env._alt_prev_both[mask]     = both_down[mask]
    env._alt_expected_lead[mask] = init_lead_bit
    env._alt_liftoff_L[mask]     = False
    env._alt_liftoff_R[mask]     = False

def _altstep_state_update_contacts_(env, cs, sensor_cfg, contact_force_threshold: float):
    fmag = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).amax(dim=1)
    Lc = fmag[:, 0] > contact_force_threshold
    Rc = fmag[:, 1] > contact_force_threshold
    env._alt_prev_both = (Lc & Rc)
    
def alternating_same_lead_penalty(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",

    # врата
    lin_cmd_threshold: float = 0.05,   # «есть команда на движение»
    use_mask: bool = True,
    mask_name: str = "dof_mask",
    leg_bits: tuple[int, ...] = (0, 1, 3, 4, 7, 8, 11, 12, 15, 16, 19, 20),

    # контакты/оценка шага
    contact_force_threshold: float = 5.0,
    step_gain: float = 0.40,   # желаемая длина шага ≈ step_gain * |v_cmd|
    beta_stride: float = 3.0,  # резкость оценки длины шага
) -> torch.Tensor:
    device = env.device
    robot  = env.scene[asset_cfg.name]
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]
    N = env.num_envs

    # --- init внутреннего состояния один раз ---
    _asl_state_init_(env, N, device)

    # --- гейт: есть команда на движение и ноги выключены по маске ---
    cmd   = env.command_manager.get_term(command_name).command  # (N,3)
    v_mag = cmd[:, :2].norm(dim=1)
    move_gate = v_mag > lin_cmd_threshold

    if use_mask:
        dof_mask = env.command_manager.get_term(mask_name).command
        legs_inactive = (dof_mask[:, list(leg_bits)].sum(dim=1) <= 0.0)
    else:
        legs_inactive = torch.zeros(N, dtype=torch.bool, device=device)

    gate = move_gate & legs_inactive
    if not gate.any():
        _asl_update_prev_both(env, cs, sensor_cfg, contact_force_threshold)
        return torch.zeros(N, device=device)

    # --- контакты (устойчиво по истории) ---
    fmag = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).amax(dim=1)  # (N,2)
    Lc = fmag[:, 0] > contact_force_threshold
    Rc = fmag[:, 1] > contact_force_threshold
    both_down = Lc & Rc

    # --- кто впереди вдоль продольной оси базы ---
    feet_pos_w  = robot.data.body_pos_w[:, sensor_cfg.body_ids, :2]  # (N,2,2)
    root_pos_w  = robot.data.root_pos_w[:, :2]
    root_quat_w = robot.data.root_quat_w

    base_fwd_w = math_utils.quat_apply(
        root_quat_w, torch.tensor([1.0,0.0,0.0], device=device).expand_as(robot.data.root_pos_w)
    )[:, :2]
    f_xy = base_fwd_w / base_fwd_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    rel_xy = feet_pos_w - root_pos_w.unsqueeze(1)
    sL = (rel_xy[:, 0, :] * f_xy).sum(dim=-1)
    sR = (rel_xy[:, 1, :] * f_xy).sum(dim=-1)
    lead_now = (sR > sL).long()   # 0=L, 1=R

    # --- ресеты эпизодов ---
    resets = env.termination_manager.terminated | env.termination_manager.time_outs
    if resets.any():
        _asl_state_reset_(env, resets, both_down, lead_now)

    # --- liftoff-трекинг между двухопорными ---
    env._asl_lift_L |= ~Lc
    env._asl_lift_R |= ~Rc
    had_any_liftoff = env._asl_lift_L | env._asl_lift_R    # хоть одна нога отрывалась?

    # --- вход в двухопорную фазу (rising edge) ---
    ds_on = both_down & (~env._asl_prev_both)

    # --- «тот же лидер?» и масштабирование по длине шага/лифтоффу ---
    same_lead = env._asl_have_last & (lead_now == env._asl_last_lead)
    d_long = (sL - sR).abs()
    d_des  = step_gain * v_mag
    stride_miss = 1.0 - torch.exp(-beta_stride * (d_long - d_des).abs())     # 0..1 (чем дальше от желаемого — тем ↑)

    # базовый пенальти: когда гейт активен, вход в DS и лидер не сменился
    pen = (gate & ds_on & same_lead).float()

    # усиление, если вообще не было liftoff'а (скорее всего «проскользили» корпусом)
    no_liftoff = ~had_any_liftoff
    scale = 1.0 + 0.5 * no_liftoff.float()            # 1.0 или 1.5
    pen = pen * scale * (0.5 + 0.5 * stride_miss)     # 0.5..1.5

    # --- обновление состояния после входа в DS ---
    if ds_on.any():
        env._asl_last_lead[ds_on] = lead_now[ds_on]
        env._asl_have_last[ds_on] = True
        env._asl_lift_L[ds_on] = False
        env._asl_lift_R[ds_on] = False

    env._asl_prev_both = both_down
    return pen


# --- helpers: per-env состояние ---
def _asl_state_init_(env, N, device):
    if not hasattr(env, "_asl_prev_both"):
        env._asl_prev_both  = torch.zeros(N, dtype=torch.bool, device=device)
        env._asl_last_lead  = torch.zeros(N, dtype=torch.long, device=device)  # 0=L,1=R
        env._asl_have_last  = torch.zeros(N, dtype=torch.bool, device=device)
        env._asl_lift_L     = torch.zeros(N, dtype=torch.bool, device=device)
        env._asl_lift_R     = torch.zeros(N, dtype=torch.bool, device=device)

def _asl_state_reset_(env, mask, both_down, lead_now):
    env._asl_prev_both[mask] = both_down[mask]
    env._asl_have_last[mask] = False
    env._asl_lift_L[mask]    = False
    env._asl_lift_R[mask]    = False
    # last_lead оставляем как есть; он будет установлен при первом DS после ресета

def _asl_update_prev_both(env, cs, sensor_cfg, thr):
    fmag = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).amax(dim=1)
    Lc = fmag[:, 0] > thr
    Rc = fmag[:, 1] > thr
    env._asl_prev_both = (Lc & Rc)    
    
def heading_alignment_reward(
    env,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lin_cmd_threshold: float = 0.05,   # м/с
    beta: float = 4.0,                 # «резкость»
) -> torch.Tensor:
    """
    Награда ∈[0..1] за согласование продольной оси корпуса с направлением команды скорости.
    Работает только при |v_cmd| > lin_cmd_threshold.
    """
    robot = env.scene[asset_cfg.name]
    cmd   = env.command_manager.get_term(command_name).command  # (N,3)
    v_xy  = cmd[:, :2]
    v_mag = v_xy.norm(dim=1)
    gate  = v_mag > lin_cmd_threshold
    if not gate.any():
        return torch.zeros(env.num_envs, device=env.device)

    # единичный вектор «куда ехать» в мире
    v_dir = v_xy / v_mag.clamp_min(1e-6).unsqueeze(-1)

    # продольная ось корпуса в мире
    fwd_w = math_utils.quat_apply(
        robot.data.root_quat_w,
        torch.tensor([1.0, 0.0, 0.0], device=env.device).expand_as(robot.data.root_pos_w),
    )[:, :2]
    fwd_dir = fwd_w / fwd_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    cosang = (fwd_dir * v_dir).sum(dim=-1).clamp(-1.0, 1.0)
    # 1 при сонаправленности → падает при рассогласовании
    r = torch.exp(-beta * (1.0 - cosang))
    return r * gate.float()
    
def swing_foot_clearance_reward(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    contact_force_threshold: float = 5.0,
    lin_cmd_threshold: float = 0.05,   # активируем только при команде на ходьбу
    h_des: float = 0.06,               # желаемый клиренс, м
    beta: float = 120.0,               # «узость» колокола
) -> torch.Tensor:
    """
    Награда за клиренс маховой стопы при одноопорной фазе.
    r = exp(-beta * (clearance - h_des)^2), где clearance = z_swing - z_stance.
    Только когда |v_cmd| > порога. Награда 0 в двухопоре/беспконтактной фазе.
    """
    device = env.device
    robot  = env.scene[asset_cfg.name]
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # врата по команде
    cmd   = env.command_manager.get_term(command_name).command
    v_mag = cmd[:, :2].norm(dim=1)
    move_gate = v_mag > lin_cmd_threshold
    if not move_gate.any():
        return torch.zeros(env.num_envs, device=device)

    # контакты (устойчиво по истории)
    fmag = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).amax(dim=1)  # (N,2)
    Lc = fmag[:, 0] > contact_force_threshold
    Rc = fmag[:, 1] > contact_force_threshold

    # одноопора?
    single_L = Lc & (~Rc)   # левая — опора, правая — мах
    single_R = Rc & (~Lc)   # правая — опора, левая — мах
    single   = single_L | single_R
    if not single.any():
        return torch.zeros(env.num_envs, device=device)

    feet_z = robot.data.body_pos_w[:, sensor_cfg.body_ids, 2]  # (N,2)
    # clearance = z(swing) - z(stance)
    clr_L = (feet_z[:, 1] - feet_z[:, 0])   # если левая — опора → правая мах
    clr_R = (feet_z[:, 0] - feet_z[:, 1])   # если правая — опора → левая мах
    clearance = torch.zeros(env.num_envs, device=device)
    clearance[single_L] = clr_L[single_L]
    clearance[single_R] = clr_R[single_R]
    clearance = clearance.clamp_min(0.0)    # не поощряем «в землю»

    r = torch.exp(-beta * (clearance - h_des).pow(2))
    return r * (single & move_gate).float()    
    
    
def masked_action_rate_l2(env, mask_name: str = "dof_mask") -> torch.Tensor:
    """
    L2 на Δaction ТОЛЬКО по активным DOF (mask==1). Снижает дёрганье там,
    где мы реально что-то трекаем по UDP.
    """
    act  = env.action_manager.action
    prev = env.action_manager.prev_action
    d    = (act - prev).pow(2)
    m    = (env.command_manager.get_term(mask_name).command > 0.5).float()
    num  = m.sum(dim=1).clamp_min(1.0)
    return (d * m).sum(dim=1) / num
    
def masked_success_stable_bonus(
    env,
    eps: float = 0.03,      # допуск по позе в норме [-1..1]
    vel_eps: float = 0.03,  # допуск по норм. скорости
    bonus: float = 1.0,
    mask_name: str = "dof_mask",
) -> torch.Tensor:
    """
    Бонус, когда активные DOF (mask==1) и близки к целям, и «успокоились» по скорости.
    Хорош для удержания достигнутого таргета (без раскачки).
    """
    robot = env.scene["robot"]
    q     = robot.data.joint_pos
    qd    = robot.data.joint_vel
    qmin, qmax = robot.data.soft_joint_pos_limits[...,0], robot.data.soft_joint_pos_limits[...,1]
    mid, scl = 0.5*(qmin+qmax), 2.0/(qmax-qmin+1e-6)
    qn, qdn  = (q - mid)*scl, qd*scl

    tgt = env.command_manager.get_term("target_joint_pose").command
    msk = env.command_manager.get_term(mask_name).command > 0.5

    ok_pos = (qn - tgt).abs() <= eps
    ok_vel = qdn.abs() <= vel_eps
    ok = torch.where(msk, ok_pos & ok_vel, torch.ones_like(ok_pos, dtype=torch.bool))
    return bonus * ok.all(dim=1).float()     
    
def leg_pelvis_torso_coalignment_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # Имена тел как в твоём USD:
    pelvis_body: str = "pelvis",
    torso_body: str = "torso_link",
    left_thigh_body: str = "left_hip_pitch_link",
    left_shank_body: str = "left_knee_link",
    right_thigh_body: str = "right_hip_pitch_link",
    right_shank_body: str = "right_knee_link",

    # Что считаем «вперёд» в ЛСК звеньев:
    forward_local: tuple[float, float, float] = (1.0, 0.0, 0.0),

    # Веса компонентов внутри терма:
    w_yaw: float = 1.0,    # сонаправленность сегментов с тазом по курсу (XY)
    w_chain: float = 0.7,  # согласованность бедро↔голень (каждая нога)
    w_upright: float = 0.3,# горизонтальность продольной оси таза/торса (меньше «на носок/пятку»)

    # Маска DOF: если соответствующие DOF активны (mask==1), вклад компоненты отключаем
    mask_name: str = "dof_mask",
    left_dofs:  tuple[int, ...] = (0, 3, 7, 11, 15, 19),   # L hip pitch/roll/yaw, knee, ankle pitch/roll
    right_dofs: tuple[int, ...] = (1, 4, 8, 12, 16, 20),   # R hip pitch/roll/yaw, knee, ankle pitch/roll
    torso_bits: tuple[int, ...] = (),                      # при желании добавь waist_yaw и т.п.
) -> torch.Tensor:
    """
    r ∈ [0..1]. Поощряет согласованность направления звеньев ног, таза и торса:
      (A) yaw-сонаправленность звеньев с тазом (через XY),
      (B) согласованность «цепочки» бедро↔голень по yaw,
      (C) горизонтальность продольной оси таза/торса (меньше |z|).
    Если соответствующие DOF находятся под активной маской (mask==1), вклад
    соответствующей компоненты обнуляется, чтобы не конфликтовать с внешними таргетами.
    """
    device = env.device
    robot  = env.scene[asset_cfg.name]

    # --- кэш индексов тел ---
    if not hasattr(env, "_coalignment_body_ids"):
        names = list(robot.data.body_names)
        def _idx(n: str) -> int:
            try:
                return names.index(n)
            except ValueError as e:
                raise RuntimeError(f"[coalignment] не найдено тело '{n}' среди robot.data.body_names") from e
        ids = [
            _idx(pelvis_body),
            _idx(torso_body),
            _idx(left_thigh_body), _idx(left_shank_body),
            _idx(right_thigh_body), _idx(right_shank_body),
        ]
        env._coalignment_body_ids = torch.as_tensor(ids, device=device, dtype=torch.long)
    ids = env._coalignment_body_ids  # порядок: [pelvis, torso, Lth, Lsh, Rth, Rsh]

    # --- «вперёд» в мире для каждого сегмента ---
    quats = robot.data.body_quat_w[:, ids, :]  # (N,6,4)
    f_loc = torch.tensor(forward_local, device=device, dtype=torch.float32)\
                .view(1,1,3).expand(quats.shape[0], quats.shape[1], 3)
    fwd_w = math_utils.quat_apply(quats, f_loc)  # (N,6,3)

    # --- проекция на XY и нормализация ---
    fwd_xy = fwd_w[..., :2]
    fwd_xy = fwd_xy / fwd_xy.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    pelvis_xy = fwd_xy[:, 0, :]  # (N,2)

    # (A) сонаправленность по yaw с тазом: torso, Lth, Lsh, Rth, Rsh
    cos_to_pelvis = (fwd_xy[:, 1:, :] * pelvis_xy.unsqueeze(1)).sum(dim=-1).clamp(-1.0, 1.0)  # (N,5)
    yaw_align = 0.5 * (1.0 + cos_to_pelvis).mean(dim=1)                                       # (N,) → [0..1]

    # (B) согласованность «цепочки» бедро↔голень (каждая нога)
    def _cos(i: int, j: int):
        return (fwd_xy[:, i, :] * fwd_xy[:, j, :]).sum(dim=-1).clamp(-1.0, 1.0)
    chain_align = 0.5 * (1.0 + 0.5 * (_cos(2, 3) + _cos(4, 5)))                               # (N,) → [0..1]

    # (C) «горизонтальность» продольной оси таза и торса (подавляет носок/пятку на уровне корпуса)
    z_pelvis = fwd_w[:, 0, 2].abs()
    z_torso  = fwd_w[:, 1, 2].abs()
    upright  = (1.0 - 0.5 * (z_pelvis + z_torso)).clamp(0.0, 1.0)                              # (N,)

    # --- гейтинг по маске DOF: если DOF активны — компоненту выключаем ---
    m = env.command_manager.get_term(mask_name).command > 0.5  # (N,J) bool

    def any_active(bits: tuple[int, ...]) -> torch.Tensor:
        if len(bits) == 0:
            return torch.zeros(m.shape[0], dtype=torch.bool, device=device)
        return m[:, list(bits)].any(dim=1)

    active_L = any_active(left_dofs)
    active_R = any_active(right_dofs)
    active_T = any_active(torso_bits)

    gateA = ~(active_L | active_R | active_T)  # yaw_align зависит от всех сегментов
    gateB = ~(active_L | active_R)             # цепочки зависят от ног
    gateC = ~active_T                          # upright — про таз/торс

    wA = w_yaw    * gateA.float()
    wB = w_chain  * gateB.float()
    wC = w_upright* gateC.float()

    denom = (wA + wB + wC).clamp_min(1e-6)
    r = (wA * yaw_align + wB * chain_align + wC * upright) / denom
    return r.clamp(0.0, 1.0)
