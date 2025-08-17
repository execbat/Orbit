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

def miander_untracking_reward(env) -> torch.Tensor:
    """
    Награда за «не-трекание» : берём **только НЕ-активные** DOF (mask==0)
    и хотим, чтобы они остались близки к исходной (neutral) позе.

    reward = 1 − MSE/4   ∈ [0,1];  затем смещаем ­0.5, как в исходной версии.
    """
    asset = env.scene["robot"]

    # ---------------------------------------------------------------- mask & target
    mask_cmd = env.command_manager.get_term("dof_mask").command     # (N,J)
    inverse_mask = mask_cmd <= 0.5                                  # inactive DOF

    # ---------------------------------------------------------------- текущие углы, нормированные в [-1,1]
    joint_pos = asset.data.joint_pos
    joint_min = asset.data.soft_joint_pos_limits[..., 0]
    joint_max = asset.data.soft_joint_pos_limits[..., 1]
    norm_joint = 2.0 * (joint_pos - (joint_min + joint_max) * 0.5) \
                 / (joint_max - joint_min + 1e-6)

    # ---------------------------------------------------------------- исходная (нейтральная) поза
    # уже хранится в env, нормирована
    init_norm = env.JOINT_INIT_POS_NORM.to(norm_joint.device).unsqueeze(0)  # (1,J)

    # ---------------------------------------------------------------- MSE по неактивным DOF
    se          = (norm_joint - init_norm).pow(2)
    se_masked   = se * inverse_mask
    inactive_n  = inverse_mask.sum(dim=1)                                 # (N,)

    mse = torch.where(
        inactive_n > 0,
        se_masked.sum(dim=1) / inactive_n.clamp(min=1),
        torch.zeros_like(inactive_n, dtype=se.dtype),
    )

    # ---------------------------------------------------------------- reward ∈ [0,1]
    reward = torch.where(
        inactive_n > 0,
        1.0 - mse / 4.0,
        torch.zeros_like(mse),
    ).clamp_(0.0, 1.0)

    return reward # - 0.5




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
    asset_cfg : SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",

    # (0) прямой штраф по суставам ankle_pitch → к нейтрали
    ankle_pitch_joint_names=("left_ankle_pitch_joint", "right_ankle_pitch_joint"),
    w_ankle_neutral: float = 2.0,          # ↑ вес прямого «выпрямления» стоп по суставам

    # (1) «ступня параллельно полу» через Z-оси
    w_tilt: float = 1.0,                   # можно оставить меньше, основное делает w_ankle_neutral
    foot_normal_local =(0.0, 0.0, 1.0),

    # (2) выравнивание стоп с направлением таза (при 2 контактах)
    w_align: float = 0.5,
    foot_forward_local=(1.0, 0.0, 0.0),

    # (3) продольная длина шага (при 2 контактах)
    w_stride: float = 0.7,
    step_gain: float = 0.40,
    beta_stride: float = 3.0,
    v_near_zero: float = 0.05,
    w_opposite_at_zero: float = 0.5,

    # (4) перехлест
    w_cross: float = 0.7,
    beta_cross: float = 6.0,

    # боковая ширина
    shoulder_width: float = 0.35,
    beta_sep: float = 2.0,
    w_sep: float = 1.0,

    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    device = env.device
    robot  = env.scene[asset_cfg.name]
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # helper: привести локальные оси к форме (1,2,3) [L,R]
    def _per_foot(vec):
        v = torch.tensor(vec, dtype=torch.float32, device=device)
        if v.ndim == 1: v = v.view(1,1,3).expand(1,2,3)
        else:           v = v.view(1,2,3)
        return v

    nrm_loc = _per_foot(foot_normal_local)
    fwd_loc = _per_foot(foot_forward_local)
    world_up = torch.tensor([0.0, 0.0, 1.0], device=device).view(1,1,3)

    # контакты (N,2)
    forces   = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    contacts = forces.norm(dim=-1).max(dim=1)[0] > contact_force_threshold    # (N,2) [L,R]
    both_down = contacts.all(dim=1)                                           # (N,)
    n_down    = contacts.float().sum(dim=1).clamp(min=1.0)                    # (N,)

    # позы/ориентации
    feet_pos_w  = robot.data.body_pos_w[:, sensor_cfg.body_ids, :]   # (N,2,3)
    feet_quat_w = robot.data.body_quat_w[:, sensor_cfg.body_ids, :]  # (N,2,4)
    root_pos_w  = robot.data.root_pos_w                              # (N,3)
    root_quat_w = robot.data.root_quat_w                             # (N,4)

    base_fwd_w = math_utils.quat_apply(root_quat_w, torch.tensor([1.0,0.0,0.0], device=device).expand_as(root_pos_w))
    base_lat_w = math_utils.quat_apply(root_quat_w, torch.tensor([0.0,1.0,0.0], device=device).expand_as(root_pos_w))
    f_xy = base_fwd_w[:, :2]; f_xy = f_xy / f_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    l_xy = base_lat_w[:, :2]; l_xy = l_xy / l_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    # ---------- (0) прямой штраф по ankle_pitch к нейтрали ----------
    # кэшируем индексы один раз
    if not hasattr(env, "_ankle_pitch_ids"):
        jnames = list(robot.data.joint_names)
        try:
            env._ankle_pitch_ids = torch.tensor(
                [jnames.index(ankle_pitch_joint_names[0]), jnames.index(ankle_pitch_joint_names[1])],
                device=device, dtype=torch.long
            )
        except ValueError as e:
            raise RuntimeError(f"Не нашёл ankle_pitch сустав: {e}")

        # возьмём целевые нейтрали из сохранённых нормированных значений энва
        init_norm = env.JOINT_INIT_POS_NORM.to(device)
        env._ankle_pitch_init_norm = init_norm[env._ankle_pitch_ids]  # (2,)

    jidx = env._ankle_pitch_ids                                        # (2,)
    q     = robot.data.joint_pos                                       # (N,J)
    qmin  = robot.data.soft_joint_pos_limits[...,0]                    # (N,J)
    qmax  = robot.data.soft_joint_pos_limits[...,1]                    # (N,J)
    qmid  = 0.5*(qmin+qmax); qhalf = 0.5*(qmax-qmin)
    qn    = ((q - qmid) / (qhalf + 1e-6)).clamp(-1.0, 1.0)             # норм. позы (N,J)

    qn_ankles   = qn[:, jidx]                                          # (N,2) [L,R]
    qn_target   = env._ankle_pitch_init_norm.view(1,2)                 # (1,2)
    ankle_err   = (qn_ankles - qn_target).abs()                        # (N,2)
    # штрафуем по каждой ноге только когда эта нога в контакте
    ankle_neutral_each = ankle_err * contacts.float()                  # (N,2)
    ankle_neutral_pen  = ankle_neutral_each.sum(dim=1) / n_down        # (N,)

    # ---------- (1) «ступни параллельно полу» через Z ----------
    z_world = math_utils.quat_apply(feet_quat_w, nrm_loc.expand_as(feet_pos_w))  # (N,2,3)
    cos_up  = (z_world * world_up).sum(dim=-1).abs().clamp(0.0, 1.0)
    tilt_each = 1.0 - cos_up
    tilt_pen  = (tilt_each * contacts.float()).sum(dim=1) / n_down

    # ---------- (2) выравнивание стоп по направлению таза ----------
    foot_fwd_w  = math_utils.quat_apply(feet_quat_w, fwd_loc.expand_as(feet_pos_w))
    foot_dir_xy = foot_fwd_w[:, :, :2]
    foot_dir_xy = foot_dir_xy / foot_dir_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    cos_to_pelvis = (foot_dir_xy * f_xy.unsqueeze(1)).sum(dim=-1).clamp(-1.0, 1.0)
    align_term = 0.5 * (1.0 - cos_to_pelvis).mean(dim=1)
    align_pen  = torch.where(both_down, align_term, torch.zeros_like(align_term))

    # ---------- (3) продольная длина шага ----------
    rel_xy = (feet_pos_w[:, :, :2] - root_pos_w[:, :2].unsqueeze(1))
    sL = (rel_xy[:, 0, :] * f_xy).sum(dim=-1); sR = (rel_xy[:, 1, :] * f_xy).sum(dim=-1)
    d_long = (sL - sR).abs()
    v_cmd  = env.command_manager.get_term(command_name).command[:, :2]
    v_mag  = v_cmd.norm(dim=-1)
    d_des  = step_gain * v_mag
    stride_term = 1.0 - torch.exp(-beta_stride * (d_long - d_des).abs())
    stride_pen  = torch.where(both_down, stride_term, torch.zeros_like(stride_term))
    opposite_bool  = (sL * sR) < 0
    low_speed_gate = (v_near_zero - v_mag).clamp(min=0.0) / max(v_near_zero, 1e-6)
    opposite_pen   = w_opposite_at_zero * (opposite_bool.float() * low_speed_gate)

    # ---------- (4) перехлёст ----------
    yL = (rel_xy[:, 0, :] * l_xy).sum(dim=-1)
    yR = (rel_xy[:, 1, :] * l_xy).sum(dim=-1)
    cross_depth = torch.relu(-yL) + torch.relu(+yR)
    cross_term  = 1.0 - torch.exp(-beta_cross * cross_depth)
    cross_pen   = torch.where(both_down, cross_term, torch.zeros_like(cross_term))

    # боковая ширина
    dist_xy = (feet_pos_w[:, 0, :2] - feet_pos_w[:, 1, :2]).norm(dim=-1)
    sep_term = 1.0 - torch.exp(-beta_sep * (dist_xy - shoulder_width).abs())
    sep_pen  = torch.where(both_down, sep_term, torch.zeros_like(sep_term))

    penalty = (
        w_ankle_neutral * ankle_neutral_pen
      + w_tilt          * tilt_pen
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
         
