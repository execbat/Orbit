# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import logging
logging.basicConfig(level=logging.INFO)

def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())
    

## AnimalMath curriculum fct
def modify_miander_scale(env, env_ids, data=None):
    # return min(1.0, env.common_step_counter / 100000)
    return 1.0

# change period time from 10 to 1 sec for oscillation
def modify_max_period(env, env_ids, data=None):
    # progress = min(1.0, env.common_step_counter / 100000)
    # period = 10 - 9 * progress
    # return period
    step = env.common_step_counter
    
    if step < 100_000:
        return 10.0  # Этап 1 и 2 — 10 сек
    elif step >= 100_000 and step <= 150_000:
        progress = (step - 100_000) / 50_000  # от 0 до 1
        return 10.0 - 9.0 * progress        # от 10 до 1
    else:
        return 1.0  # После 15 000 шагов — 1 сек

def modify_mask_prob(env, env_ids, data=None):
    # return min(0.8, env.common_step_counter / 100000 * 0.8)
    step = env.common_step_counter

    if step < 50_000:
        return 0.0  # Этап 1
    elif step >= 50_000 and step <= 100_000:
        progress = (step - 50_000) / 50_000  # от 0 до 1
        return 0.8 * progress              # от 0 до 0.8
    else:
        return 0.8  # Этап 3 и далее
        
        
# For adaptive curriculum
def modify_mask_prob_adaptive(env, env_ids, data=None):
    return env.adaptive_state.get_mask_prob()

def modify_max_period_adaptive(env, env_ids, data=None):
    return env.adaptive_state.get_max_period()

def modify_miander_scale_adaptive(env, env_ids, data=None):
    return env.adaptive_state.get_miander_scale()


