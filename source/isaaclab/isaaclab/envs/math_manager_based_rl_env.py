# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations

import gymnasium as gym
import math
import numpy as np
import torch
from collections.abc import Sequence
from typing import Any, ClassVar

from isaacsim.core.version import get_version

from isaaclab.managers import CommandManager, CurriculumManager, RewardManager, TerminationManager
from isaaclab.ui.widgets import ManagerLiveVisualizer

from .common import VecEnvStepReturn
from .math_manager_based_env import MathManagerBasedEnv
from .math_manager_based_rl_env_cfg import MathManagerBasedRLEnvCfg


class MathManagerBasedRLEnv(MathManagerBasedEnv, gym.Env):
    
    """The superclass for the manager-based workflow reinforcement learning-based environments.

    This class inherits from :class:`ManagerBasedEnv` and implements the core functionality for
    reinforcement learning-based environments. It is designed to be used with any RL
    library. The class is designed to be used with vectorized environments, i.e., the
    environment is expected to be run in parallel with multiple sub-environments. The
    number of sub-environments is specified using the ``num_envs``.

    Each observation from the environment is a batch of observations for each sub-
    environments. The method :meth:`step` is also expected to receive a batch of actions
    for each sub-environment.

    While the environment itself is implemented as a vectorized environment, we do not
    inherit from :class:`gym.vector.VectorEnv`. This is mainly because the class adds
    various methods (for wait and asynchronous updates) which are not required.
    Additionally, each RL library typically has its own definition for a vectorized
    environment. Thus, to reduce complexity, we directly use the :class:`gym.Env` over
    here and leave it up to library-defined wrappers to take care of wrapping this
    environment for their agents.

    Note:
        For vectorized environments, it is recommended to **only** call the :meth:`reset`
        method once before the first call to :meth:`step`, i.e. after the environment is created.
        After that, the :meth:`step` function handles the reset of terminated sub-environments.
        This is because the simulator does not support resetting individual sub-environments
        in a vectorized environment.

    """

    is_vector_env: ClassVar[bool] = True
    """Whether the environment is a vectorized environment."""
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
        "isaac_sim_version": get_version(),
    }
    """Metadata for the environment."""

    cfg: ManagerBasedRLEnvCfg
    """Configuration for the environment."""

    def __init__(self, cfg: MathManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        print('OLOLO')
        """Initialize the environment.

        Args:
            cfg: The configuration for the environment.
            render_mode: The render mode for the environment. Defaults to None, which
                is similar to ``"human"``.
        """
        self._dbg_every = 120  # печатать примерно раз в 120 шагов
        
        # joint init pos normalised
        self.JOINT_INIT_POS_NORM = torch.tensor([-0.05084747076034546, -0.05084747076034546, 0.0, -0.7368420958518982, \
        0.736842155456543, 0.0625, 0.0625, 0.0, 0.0, -1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.2857142686843872, 0.2857142686843872, \
        0.7333333492279053, 0.7333333492279053, 0.0, 0.0, 0.0, 0.0], 
            dtype=torch.float)
        
        #self.adaptive_state = cfg.adaptive_state
        
        # cfg.episode_length_s = 5
        
        self.dev = cfg.sim.device
        # -- counter for curriculum
        self.common_step_counter = 0

        # initialize the episode length buffer BEFORE loading the managers to use it in mdp functions.
        self.episode_length_buf = torch.zeros(cfg.scene.num_envs, device=self.dev, dtype=torch.long)
        
        # ===== MEANDER INITIALISATION =====
        ####################################        
        self.MIANDER_SCALE = 1.0 # should be changed by curriculum
        self.NUM_AXIS = 23
        self.MAX_PERIOD = 5.0
        self.MIN_PERIOD = 0.5
        self.MASK_PROB_LEVEL = 0.05 
        self.EXPONENT_MULTIPLOCATOR = 1.0
        
        self.size = (cfg.scene.num_envs, self.NUM_AXIS)         
        # self.miander_scale = torch.rand(*self.size, device=self.dev) * self.MIANDER_SCALE     
        # self.get_periods()     
        

        # step counter for meanders
        # self._timestamp = torch.zeros(cfg.scene.num_envs, device=self.dev)
        # initialize miander targets
        # self.miander_targets = self.get_miander_targets()
        
        
        # self.periods = None
        # self.make_periods()
        
        # self.switcher_mask = None # self.create_random_mask(env_count=cfg.scene.num_envs, prob_level=self.MASK_PROB_LEVEL)
        # self.make_stair_mask()
        
        # self.targets = None
        # self.targets = (2 * torch.rand(self.size, device=self.dev) - 1.0) * self.MIANDER_SCALE #make_targets() 

        ####################################              
        

        # initialize the base class to setup the scene.
        super().__init__(cfg=cfg)
        # store the render mode
        self.render_mode = render_mode

        # initialize data and constants
        # -- set the framerate of the gym video recorder wrapper so that the playback speed of the produced video matches the simulation
        self.metadata["render_fps"] = 1 / self.step_dt
        
        


        print("[INFO]: Completed setting up the environment...")

    """
    Properties.
    """

    @property
    def max_episode_length_s(self) -> float:
        """Maximum episode length in seconds."""
        return self.cfg.episode_length_s

    @property
    def max_episode_length(self) -> int:
        """Maximum episode length in environment steps."""
        return math.ceil(self.max_episode_length_s / self.step_dt)

    """
    Operations - Setup.
    """

    def load_managers(self):
        # note: this order is important since observation manager needs to know the command and action managers
        # and the reward manager needs to know the termination manager
        # -- command manager
        self.command_manager: CommandManager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)

        # call the parent class to load the managers for observations and actions.
        super().load_managers()

        # prepare the managers
        # -- termination manager
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)
        # -- reward manager
        self.reward_manager = RewardManager(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)
        # -- curriculum manager
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

        # setup the action and observation spaces for Gym
        self._configure_gym_env_spaces()

        # perform events at the start of the simulation
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    def setup_manager_visualizers(self):
        """Creates live visualizers for manager terms."""

        self.manager_visualizers = {
            "action_manager": ManagerLiveVisualizer(manager=self.action_manager),
            "observation_manager": ManagerLiveVisualizer(manager=self.observation_manager),
            "command_manager": ManagerLiveVisualizer(manager=self.command_manager),
            "termination_manager": ManagerLiveVisualizer(manager=self.termination_manager),
            "reward_manager": ManagerLiveVisualizer(manager=self.reward_manager),
            "curriculum_manager": ManagerLiveVisualizer(manager=self.curriculum_manager),
        }

    """
    Operations - MDP
    """

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        # Раз в N шагов печатаем (env_id=0)
        if (self.common_step_counter % getattr(self, "_dbg_every", 120)) == 0:
            self._print_debug_targets_q_init(env_id=0)
        
        
        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        
        # incrementation for meander
        ###################################################
        # -- update timestamp and calculate miander targets  
        # self._timestamp += self.step_dt
        # self.make_targets()
        # self.miander_targets = self.get_miander_targets()
        
        # change target and mask fo those environments
        #env_mask = self._timestamp > self.MAX_PERIOD
        #env_ids = env_mask.nonzero(as_tuple=True)[0]
        #if env_ids.numel() > 0:
        #    self.targets[env_ids] = self.get_targets(env_ids)
        #    self.switcher_mask[env_ids] = self.create_random_mask(env_count=env_ids.numel(), prob_level=self.MASK_PROB_LEVEL)
        #    self._timestamp[env_ids] = 0.0
        
        

        ###################################################
        
        
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        
        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute(update_history=True)
        
        # self.extras["miander_scale_mean"] = self.miander_scale.mean().item()

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def render(self, recompute: bool = False) -> np.ndarray | None:
        """Run rendering without stepping through the physics.

        By convention, if mode is:

        - **human**: Render to the current display and return nothing. Usually for human consumption.
        - **rgb_array**: Return an numpy.ndarray with shape (x, y, 3), representing RGB values for an
          x-by-y pixel image, suitable for turning into a video.

        Args:
            recompute: Whether to force a render even if the simulator has already rendered the scene.
                Defaults to False.

        Returns:
            The rendered image as a numpy array if mode is "rgb_array". Otherwise, returns None.

        Raises:
            RuntimeError: If mode is set to "rgb_data" and simulation render mode does not support it.
                In this case, the simulation render mode must be set to ``RenderMode.PARTIAL_RENDERING``
                or ``RenderMode.FULL_RENDERING``.
            NotImplementedError: If an unsupported rendering mode is specified.
        """
        # run a rendering step of the simulator
        # if we have rtx sensors, we do not need to render again sin
        if not self.sim.has_rtx_sensors() and not recompute:
            self.sim.render()
        # decide the rendering mode
        if self.render_mode == "human" or self.render_mode is None:
            return None
        elif self.render_mode == "rgb_array":
            # check that if any render could have happened
            if self.sim.render_mode.value < self.sim.RenderMode.PARTIAL_RENDERING.value:
                raise RuntimeError(
                    f"Cannot render '{self.render_mode}' when the simulation render mode is"
                    f" '{self.sim.render_mode.name}'. Please set the simulation render mode to:"
                    f"'{self.sim.RenderMode.PARTIAL_RENDERING.name}' or '{self.sim.RenderMode.FULL_RENDERING.name}'."
                    " If running headless, make sure --enable_cameras is set."
                )
            # create the annotator if it does not exist
            if not hasattr(self, "_rgb_annotator"):
                import omni.replicator.core as rep

                # create render product
                self._render_product = rep.create.render_product(
                    self.cfg.viewer.cam_prim_path, self.cfg.viewer.resolution
                )
                # create rgb annotator -- used to read data from the render product
                self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
                self._rgb_annotator.attach([self._render_product])
            # obtain the rgb data
            rgb_data = self._rgb_annotator.get_data()
            # convert to numpy array
            rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
            # return the rgb data
            # note: initially the renerer is warming up and returns empty data
            if rgb_data.size == 0:
                return np.zeros((self.cfg.viewer.resolution[1], self.cfg.viewer.resolution[0], 3), dtype=np.uint8)
            else:
                return rgb_data[:, :, :3]
        else:
            raise NotImplementedError(
                f"Render mode '{self.render_mode}' is not supported. Please use: {self.metadata['render_modes']}."
            )

    def close(self):
        if not self._is_closed:
            # destructor is order-sensitive
            del self.command_manager
            del self.reward_manager
            del self.termination_manager
            del self.curriculum_manager
            # call the parent class to close the environment
            super().close()

    """
    Helper functions.
    """

    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""
        # observation space (unbounded since we don't impose any limits)
        self.single_observation_space = gym.spaces.Dict()
        for group_name, group_term_names in self.observation_manager.active_terms.items():
            # extract quantities about the group
            has_concatenated_obs = self.observation_manager.group_obs_concatenate[group_name]
            group_dim = self.observation_manager.group_obs_dim[group_name]
            # check if group is concatenated or not
            # if not concatenated, then we need to add each term separately as a dictionary
            if has_concatenated_obs:
                self.single_observation_space[group_name] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=group_dim)
            else:
                self.single_observation_space[group_name] = gym.spaces.Dict({
                    term_name: gym.spaces.Box(low=-np.inf, high=np.inf, shape=term_dim)
                    for term_name, term_dim in zip(group_term_names, group_dim)
                })
        # action space (unbounded since we don't impose any limits)
        action_dim = sum(self.action_manager.action_term_dim)
        self.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(action_dim,))

        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        
        # reset timestamp for specific env and create new period for sinusoid
        ##############################
        # self._timestamp[env_ids] = torch.rand(len(env_ids), device=self.dev) * 2 * math.pi  # DO NOT RESET THE TIMESTEP FOR MIANDERS. OTHERWISE AGENT WOULD TEACH START PHASE OF THE MIANDER ONLY
        # creating new periods only for envs which do reset
        # self.miander_periods[env_ids] = (torch.rand((len(env_ids), self.NUM_AXIS), device=self.miander_periods.device) * (self.MAX_PERIOD - self.MIN_PERIOD) + self.MIN_PERIOD)
        
        ###################################
        
       
        
        # update the curriculum for environments that need a reset
        self.curriculum_manager.compute(env_ids=env_ids)
        
        # update curriculum variables
        # if env_ids.numel() > 0:
        #    self.targets[env_ids] = self.get_targets(env_ids)
        #    self.switcher_mask[env_ids] = self.create_random_mask(env_count=env_ids.numel(), prob_level=self.MASK_PROB_LEVEL)
            # self._timestamp[env_ids] = 0.0
            # self.slide_mask(env_ids)
            # self._assign_periods(env_ids)
            # self.get_targets(env_ids)
            # self._assign_targets(env_ids)

        # self.miander_scale[env_ids, : ] = torch.rand(len(env_ids), self.NUM_AXIS, device=self.dev) * self.MIANDER_SCALE
        # self.switcher_mask[env_ids, : ] = self.create_random_mask(prob_level=self.MASK_PROB_LEVEL)[env_ids]
        #self.update_periods_for_envs(env_ids)
      
        # reset the internal buffers of the scene elements
        self.scene.reset(env_ids)
        # apply events such as randomizations for environments that need a reset
        if "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)
           
        # # # # #     
        self._apply_init_pose(torch.as_tensor(env_ids, device=self.device, dtype=torch.long))              
            

        # iterate over all managers and reset them
        # this returns a dictionary of information which is stored in the extras
        # note: This is order-sensitive! Certain things need be reset before others.
        self.extras["log"] = dict()
        # -- observation manager
        info = self.observation_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- action manager
        info = self.action_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- rewards manager
        info = self.reward_manager.reset(env_ids)
        self.extras["log"].update(info)
        
        ###########
        # Curriculum update logic
        mean_tracking_reward = info['Episode_Reward/termination_penalty'].mean().item()
        self._update_adaptive_curriculum(mean_tracking_reward)
        
        # проброс нового лернинг рейта наверх
        #if new_lr is not None:
        #    self.extras.setdefault("curriculum", {})["decreased_lr"] = new_lr
            # self.extras["log"].update({"new_lr" : float(new_lr)})  # для логов
        ###########
        
        # -- curriculum manager
        info = self.curriculum_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- command manager
        info = self.command_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- event manager
        info = self.event_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- termination manager
        info = self.termination_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- recorder manager
        info = self.recorder_manager.reset(env_ids)
        self.extras["log"].update(info)

        # reset the episode length buffer
        self.episode_length_buf[env_ids] = 0
        
        # logging curriculum variables
        self.extras["log"].update({
            "MIANDER_SCALE": float(self.MIANDER_SCALE),
            #"MIN_PERIOD": float(self.MIN_PERIOD),
            "MAX_PERIOD": float(self.MAX_PERIOD),
            "MASK_PROB_LEVEL": float(self.MASK_PROB_LEVEL),
            "EXPONENT_MULTIPLOCATOR": float(self.EXPONENT_MULTIPLOCATOR),
        })
        
    #def get_miander_targets(self) -> torch.Tensor:       
    #    return torch.sin(self._timestamp.view(-1, 1) / self.miander_periods) * self.miander_scale
    
    # def get_targets(self, env_ids: torch.Tensor) -> torch.Tensor:
    #     self.targets[env_ids] = (2 * torch.rand((len(env_ids), self.NUM_AXIS), device=self.dev) - 1.0) * self.MIANDER_SCALE

    #def create_random_mask(self, env_count: int, prob_level=0.5) -> torch.Tensor:
    #    return (torch.rand((env_count, self.NUM_AXIS), device=self.dev) < prob_level).float()

        
    #def get_periods(self):
    #    # Значения от -MAX_PERIOD до +MAX_PERIOD
    #    self.miander_periods = (2 * torch.rand(*self.size, device=self.dev) - 1.0) * self.MAX_PERIOD


        
    #def update_periods_for_envs(self, env_ids):
    #    # Значения от -MAX_PERIOD до +MAX_PERIOD для выбранных энвайронментов и всех 23 осей
    #    self.miander_periods[env_ids, :] = (2 * torch.rand(len(env_ids), self.NUM_AXIS, device=self.dev) - 1.0) * self.MAX_PERIOD

        


    def post_physics_step(self):
        # super().post_physics_step()
        pass

        # Обновляем stage адаптивного куррикулума на основе средней награды
#        if self.curriculum_manager is not None:
#            mean_tracking_reward = self.reward_manager.term_values["miander_tracking_reward"].mean().item()
#            print(f'mean_tracking_reward {mean_tracking_reward}')
#            self.adaptive_state.update(self, mean_tracking_reward)
#
#            self.MIANDER_SCALE   = self.adaptive_state.get_miander_scale()
#            self.MAX_PERIOD      = self.adaptive_state.get_max_period()
#            self.MASK_PROB_LEVEL = self.adaptive_state.get_mask_prob()
            
    def _update_adaptive_curriculum(self, mean_tracking_reward):

        if not mean_tracking_reward:
            return 
        

        #if self.common_step_counter % 500 == 0:
        #    print(f"[CURR] step={self.common_step_counter} mean_track={mean_tracking_reward:.4f} "
        #          f"stage={self.adaptive_state.stage}")

        # print(f'Curriculum aramenters updated, mean_tracking_reward: {mean_tracking_reward}')
        #changed = self.adaptive_state.update(self, mean_tracking_reward)  # должен вернуть bool
        #if not changed:
        #    return None
        
            #new_lr = self.adaptive_state.get_learning_rate()
            #for group in self.optimizer.param_groups:
            #    group['lr'] = new_lr
            #print(f'Learning rate decreased to: {new_lr}')
        #print(f'Curriculum aramenters updated, mean_tracking_reward: {mean_tracking_reward}')
        #self.MIANDER_SCALE   = self.adaptive_state.get_miander_scale()
        # self.MAX_PERIOD      = self.adaptive_state.get_max_period()
        #self.MASK_PROB_LEVEL = self.adaptive_state.get_mask_prob()
        #self.EXPONENT_MULTIPLOCATOR = self.adaptive_state.get_exponent_multiplicator()
            #print(f"[CURR-UP] step={self.common_step_counter} -> stage={self.adaptive_state.stage} | "
            #      f"scale={self.MIANDER_SCALE:.2f} period={self.MAX_PERIOD:.2f} mask={self.MASK_PROB_LEVEL:.2f}")
        # return self.adaptive_state.get_learning_rate()
        
    def make_stair_mask(self) -> torch.Tensor:
        """
        Build a mask of shape (n_envs, n_obs) with
        repeating identity rows and an all-zero row
        after every `n_obs` lines.

        Example for n_envs=10, n_obs=3
        --------------------------------
        1 0 0
        0 1 0
        0 0 1
        0 0 0   ← separator
        1 0 0
        0 1 0
        0 0 1
        0 0 0   ← separator
        1 0 0
        0 1 0
        """
        n_envs, n_obs = self.size
            
        rows = torch.arange(n_envs)               # 0, 1, 2, …, n_envs-1
        pos  = rows % (n_obs + 1)                 # position inside the current block
                                              # 0..n_obs (n_obs  → zero row)
        mask = torch.zeros(n_envs, n_obs, dtype=torch.float32, device = self.dev)
        keep = pos < n_obs                        # skip the last row in every block
        mask[keep, pos[keep]] = 1                 # scatter 1-s on the diagonal
        self.switcher_mask = mask
        # return mask
            
    # def slide_mask(self, env_ids):
    #     self.switcher_mask[env_ids] = torch.roll(self.switcher_mask[env_ids], shifts=1, dims=1)
            
            
    def make_periods(self):
        n_envs, n_obs = self.size
        
        self.periods = torch.linspace(start=self.MIN_PERIOD, end=self.MAX_PERIOD, steps=n_envs, dtype=torch.float32, device = self.dev)
        signs = torch.ones(n_envs, dtype=torch.int8, device = self.dev)
        signs[::2] = -1
        self.periods *=signs
        
    def _assign_periods(self, env_ids: torch.Tensor):
        """Назначает новые синус-периоды только окружениям `env_ids`."""
        n = env_ids.numel()                                    # сколько обновляем

        # 1) длина периода  T ∈ [MIN_PERIOD ; MAX_PERIOD]
        T = torch.linspace(
            self.MIN_PERIOD,
            self.MAX_PERIOD,
            steps=n,
            dtype=torch.float32,
            device=self.dev,
        ) 

        # 2) случайный знак  (+1 или −1) с вероятностью 0.5
        sign = torch.randint(0, 2, (n,), device=self.dev, dtype=torch.float32)
        sign = sign * 2.0 - 1.0                                # 0→−1 , 1→+1

        # 3) сохранить
        self.periods[env_ids] = T * sign                       # in-place    
            
    #def make_targets(self) -> None:
    #    """Пересчитать targets для ВСЕХ окружений с учётом времени."""
    #    n_envs, n_obs = self.size

    #    # осевые индексы 0 … 22         → shape (1, n_obs)  (broadcast по строкам)
    #    idx = torch.arange(n_obs, device=self.dev, dtype=torch.float32).view(1, -1)

    #    # периоды            → shape (n_envs, 1)
    #    periods = self.periods.to(torch.float32).view(-1, 1)
    #    if torch.any(periods == 0):
    #        raise ValueError("Period must be non-zero for all environments.")

        # timestamp          → shape (n_envs, 1)  (текущая фаза)
    #    phase = self._timestamp.to(torch.float32).view(-1, 1)

    #    # sin( 2π·(t + j) / T )
    #    self.targets = self.MIANDER_SCALE * torch.sin(2 * math.pi * (phase + idx) / periods)
    #    # self.targets: (n_envs, n_obs)



    #def _assign_targets(self, env_ids: torch.Tensor):
    #    """Обновить targets только для окружений `env_ids`."""
    #    env_ids = torch.as_tensor(env_ids, device=self.dev, dtype=torch.long)
    #    if env_ids.numel() == 0:
    #        return

    #    n_obs = self.NUM_AXIS
    #    idx = torch.arange(n_obs, device=self.dev, dtype=torch.float32).view(1, -1)  # (1, n_obs)

    #    periods = self.periods[env_ids].to(torch.float32).unsqueeze(1)               # (m,1)
    #    if torch.any(periods == 0):
    #        raise ValueError("Period must be non-zero for all environments.")

    #    phase = self._timestamp[env_ids].to(torch.float32).unsqueeze(1)              # (m,1)

    #    self.targets[env_ids] = self.MIANDER_SCALE * torch.sin(2 * math.pi * (phase + idx) / periods)
    def _print_debug_targets_q_init(self, env_id: int = 0):
        """
        По каждой оси печатает:
          id | joint name | tgt([-1..1]) | q_norm([-1..1]) | init([-1..1]) | mask(0/1)
        """
        robot = self.scene["robot"]
        names = list(robot.data.joint_names)  # порядок соответствует q/targets

        # --- командные термы
        tgt_cmd  = self.command_manager.get_term("target_joint_pose").command[env_id].detach().cpu()
        mask_cmd = self.command_manager.get_term("dof_mask").command[env_id].detach().cpu()

        # --- текущая поза, нормированная в [-1,1]
        q    = robot.data.joint_pos[env_id]
        qmin = robot.data.soft_joint_pos_limits[env_id, :, 0]
        qmax = robot.data.soft_joint_pos_limits[env_id, :, 1]
        qn   = 2.0 * (q - 0.5 * (qmin + qmax)) / (qmax - qmin + 1e-6)
        qn   = qn.detach().cpu()

        # --- init-нормы (у тебя уже в [-1,1] и в нужном порядке)
        q_init = self.JOINT_INIT_POS_NORM.detach().cpu()

        # --- sanity check размеров
        if not (len(names) == tgt_cmd.numel() == qn.numel() == mask_cmd.numel() == q_init.numel()):
            print(f"[DBG e{env_id}] size mismatch: "
                  f"names={len(names)} targets={tgt_cmd.numel()} "
                  f"q_norm={qn.numel()} mask={mask_cmd.numel()} init={q_init.numel()}")

        print(f"\n[DBG e{env_id}] Targets vs q_norm vs init (scale [-1,1])")
        print("-" * 116)
        print(f"{'id':>2}  {'joint name':>32} | {'tgt':>7} | {'q_norm':>7} | {'init':>7} | {'mask':>4}")
        print("-" * 116)
        n = min(len(names), tgt_cmd.numel(), qn.numel(), mask_cmd.numel(), q_init.numel())
        for i in range(n):
            name = names[i] if i < len(names) else f"joint_{i}"
            t = float(tgt_cmd[i])
            qv = float(qn[i])
            qi = float(q_init[i])
            m = 1 if float(mask_cmd[i]) > 0.5 else 0
            print(f"{i:2d}  {name:>32} | {t:+7.3f} | {qv:+7.3f} | {qi:+7.3f} | {m:4d}")
        print("-" * 116)


    def _apply_init_pose(self, env_ids: torch.Tensor):
        robot = self.scene["robot"]
        dev   = self.device

        # init в нормализованных координатах [-1, 1]
        init_n = self.JOINT_INIT_POS_NORM.to(dev).unsqueeze(0).expand(len(env_ids), -1)  # (k,J)

        # soft limits -> радианы
        qmin = robot.data.soft_joint_pos_limits[env_ids, :, 0]
        qmax = robot.data.soft_joint_pos_limits[env_ids, :, 1]
        mid  = 0.5 * (qmin + qmax)
        half = 0.5 * (qmax - qmin)

        # целевая поза (радианы)
        q_des = mid + half * init_n

        # === DEBUG PRINT (только для первого env из env_ids) ===
        try:
            eid_local = 0                     # индекс в переданном списке env_ids
            eid = int(env_ids[eid_local])     # глобальный id среды
            names = getattr(robot.data, "joint_names", [f"joint_{i}" for i in range(q_des.shape[1])])

            q_des_rad = q_des[eid_local].detach().cpu()
            q_des_deg = torch.rad2deg(q_des_rad)

            qmin_deg = torch.rad2deg(qmin[eid_local].detach().cpu())
            qmax_deg = torch.rad2deg(qmax[eid_local].detach().cpu())

            init_n0  = init_n[eid_local].detach().cpu()

            mid0  = mid[eid_local].detach().cpu()
            half0 = half[eid_local].detach().cpu()
            # обратная нормализация: (q_des - mid)/half -> должна совпасть с init_n
            init_n_recon = (q_des_rad - mid0) / (half0 + 1e-12)
            max_err = (init_n_recon - init_n0).abs().max().item()

            torch.set_printoptions(precision=4, linewidth=200)
            #print(f"\n[INIT→RAD/DEG CHECK] env {eid} | max_norm_error={max_err:.3e}")
            #for j in range(q_des_rad.numel()):
            #    name = names[j] if j < len(names) else f"joint_{j}"
            #    print(f"{j:02d} {name:>28s}  n={init_n0[j]:+6.3f}  "
            #          f"rad={q_des_rad[j]:+7.4f}  deg={q_des_deg[j]:+8.2f}  "
            #          f"min/max(deg)=({qmin_deg[j]:+7.2f},{qmax_deg[j]:+7.2f})")
        except Exception as e:
            print(f"[INIT DEBUG PRINT] skipped due to: {e}")

        # применяем позу и сбрасываем скорости
        robot.data.joint_pos[env_ids, :] = q_des
        robot.data.joint_vel[env_ids, :] = 0.0
        robot.data.root_lin_vel_w[env_ids, :] = 0.0
        robot.data.root_ang_vel_w[env_ids, :] = 0.0

