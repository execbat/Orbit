# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import NormalVelocityCommandCfg, UniformVelocityCommandCfg


from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
import torch, math
from dataclasses import MISSING, asdict  

# ───────────────────────────── 1. UNIFORM VECTOR ──────────────────────────────
class UniformVectorCommand(CommandTerm):
    r"""N-мерный вектор-команда; новый U(-1,1) на reset, неизменный в шаге."""

    cfg: "UniformVectorCommandCfg"

    # ---------------------------------------------------------------- init
    def __init__(self, cfg: "UniformVectorCommandCfg", env: ManagerBasedEnv):
        # отключаем автоматический ресэмплинг между шагами
        if math.isinf(cfg.resampling_time_range[0]):
            cfg.resampling_time_range = (0.0, 0.0)

        super().__init__(cfg, env)
        
        if cfg.dim is MISSING:
            raise ValueError("UniformVectorCommandCfg.dim должен быть задан")

        self.dim = cfg.dim

        # -------- диапазоны a_i,b_i ----------------------------------
        if isinstance(cfg.ranges, tuple):              # передали кортеж
            pairs = list(cfg.ranges)
        elif cfg.ranges is MISSING or cfg.ranges is None:
            pairs = [(-1.0, 1.0)] * cfg.dim            # дефолт
        else:                                          # dataclass Ranges
            pairs = list(asdict(cfg.ranges).values())

        if cfg.dim is MISSING:
            cfg.dim = len(pairs)

        if len(pairs) != cfg.dim:
            raise ValueError("len(ranges) != dim в UniformVectorCommandCfg")

        self.low  = torch.tensor([p[0] for p in pairs], device=self.device)
        self.high = torch.tensor([p[1] for p in pairs], device=self.device)

        # -------- буфер команды --------------------------------------
        self._command = torch.empty(self.num_envs, cfg.dim, device=self.device)
        self._resample_command(torch.arange(self.num_envs, device=self.device))

    # ---------------------------------------------------------------- property
    @property
    def command(self) -> torch.Tensor:
        return self._command

    # ------------------------------------------------ шаблонные методы
    def _update_metrics(self):
        pass                                            # метрик нет

    def _resample_command(self, env_ids: Sequence[int]):
        """Генерирует новые targets для заданных env_ids (reset)."""
        if len(env_ids) == 0:
            return

        # инициальная поза (shape (dim,)) и текущий масштаб-куррикулум (скаляр)
        init_pose = self._env.JOINT_INIT_POS_NORM.to(self.device)      # (dim,)
        scale     = float(self._env.MIANDER_SCALE)

        low  = init_pose - scale
        high = init_pose + scale

        # случайный U(low, high) для каждого env и каждой DoF
        r = torch.rand((len(env_ids), self.dim), device=self.device)
        self._command[env_ids] = r * (high - low) + low
        # при желании обрезаем до [-1,1]
        x = r * (high - low) + low
        x = torch.clamp(x, -1.0, 1.0)
        self._command[env_ids] = x
        # self._command[env_ids].clamp_(-1.0, 1.0)

    def _update_command(self):
        """Между шагами ничего не меняем – команда постоянна."""
        pass
# ───────────────────────────── 2. BERNOULLI MASK ─────────────────────────────
class BernoulliMaskCommand(CommandTerm):
    r"""Бинарная маска (N_envs, dim):
    • генерируется при reset: u~U(0,1) → (u < p) ? 1.0 : 0.0
    • в течение эпизода неизменна
    • p берём из env.MASK_PROB_LEVEL (если есть), иначе cfg.p_one
    """
    cfg: "BernoulliMaskCommandCfg"

    def __init__(self, cfg: "BernoulliMaskCommandCfg", env: "ManagerBasedEnv"):
        # отключаем автоперегенерацию между шагами: вместо inf ставим ОЧЕНЬ большое число,
        # т.к. torch.uniform_ не любит inf.
        if (math.isinf(cfg.resampling_time_range[0])
                or math.isinf(cfg.resampling_time_range[1])):
            cfg.resampling_time_range = (1e9, 1e9)

        super().__init__(cfg, env)

        self._command = torch.zeros(self.num_envs, cfg.dim, device=self.device)
        # первичная инициализация (как при reset)
        self._resample_command(torch.arange(self.num_envs, device=self.device))

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self):
        pass  # метрик нет

    def _current_threshold(self) -> float:
        # пробуем взять порог из env (для куррикуума), иначе fallback на cfg.p_one
        return float(getattr(self._env, "MASK_PROB_LEVEL", getattr(self.cfg, "p_one", 0.5)))

    def _resample_command(self, env_ids: Sequence[int]):
        """Генерация только при reset(env_ids)."""
        if len(env_ids) == 0:
            return
        p = self._current_threshold()
        u = torch.rand((len(env_ids), self.cfg.dim), device=self.device)
        self._command[env_ids] = (u < p).float()  # строго 0.0 или 1.0

    def _update_command(self):
        """Во время эпизода маска не меняется."""
        pass
