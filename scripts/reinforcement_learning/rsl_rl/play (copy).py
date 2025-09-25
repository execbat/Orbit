# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
# import multiprocessing as mp
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, MathRslRlVecEnvWrapper, MathTeleopRslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

from isaaclab.envs import ManagerBasedRLEnvCfg, MathManagerBasedRLEnvCfg, MathTeleopManagerBasedRLEnvCfg
# PLACEHOLDER: Extension template (do not remove this comment)


def main():
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl

    
    ####
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    elif isinstance(env_cfg, MathManagerBasedRLEnvCfg):
        env = MathRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    elif isinstance(env_cfg, MathTeleopManagerBasedRLEnvCfg):
        env = MathTeleopRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    else:
        print("Incorrect manager type")
        exit(0)

    
    
    ##### 
    
    

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )


#######
    # 1) список командных термов
    if hasattr(env.command_manager, "_terms"):
        print("[CMD TERMS]:", list(env.command_manager._terms.keys()))

    # 2) форма и девайс команды
    c = env.command_manager.get_command("base_velocity")
    print("[CMD SHAPE/DEVICE]:", tuple(c.shape), c.device)

    # 3) проверка, что сенсор контактов тикает (если используешь в наградах)
    if hasattr(env.scene, "sensors") and "contact_forces" in env.scene.sensors:
        cs = env.scene.sensors["contact_forces"]
        print("[CONTACT SENSOR] hist:", hasattr(cs.data, "net_forces_w_history"))
        
    PRINT_EVERY = 50   # печатать раз в N шагов
    LIN_IDLE_TH = 0.05 # «считаем, что стоит»
    CMD_MIN_TH  = 0.10 # «считаем, что команда действительно есть»    
    prev_cmd = None
    resample_events = 0
    
    # sanity: есть ли нужный терм
    if hasattr(env.command_manager, "get_term"):
        try:
            term = env.command_manager.get_term("base_velocity")
            print("[CHECK] base_velocity term type:", type(term).__name__)
        except Exception as e:
            print("[ERROR] нет терма 'base_velocity' в command_manager:", e)

#####

    dt = env.unwrapped.step_dt

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            print(f"actions {actions}")
            # env stepping
            
            
            ################
            # --- ПЕРЕД ШАГОМ (что хочет команда и что видит робот сейчас) ---
            try:
                cmd = env.command_manager.get_command("base_velocity")  # (N,3) vx,vy,wz в базе
            except Exception as e:
                print("[ERROR] get_command('base_velocity') свалился:", e)
                cmd = torch.zeros((env.num_envs, 3), device=env.device)
            v_xy = env.scene["robot"].data.root_lin_vel_b[:, :2]        # (N,2) скорость базы (ЛСК)
            w_z  = env.scene["robot"].data.root_ang_vel_b[:, 2]         # (N,)
            spd  = v_xy.norm(dim=1)                                     # (N,)

            # аггрегаты по батчу
            cmd_mag = cmd[:, :2].norm(dim=1)                            # (N,)
            frac_zero_cmd   = (cmd_mag < 1e-6).float().mean().item()
            frac_move_cmd   = (cmd_mag > CMD_MIN_TH).float().mean().item()
            frac_actually_standing = (spd < LIN_IDLE_TH).float().mean().item()

            # отслеживание ресэмплинга команды (упрощённо)
            if prev_cmd is None:
                prev_cmd = cmd.clone()
            else:
                changed = (cmd - prev_cmd).abs().max(dim=1)[0] > 1e-6
                resample_events += changed.any().item()
                prev_cmd = cmd.clone()
            #################
            obs, rew, _, _ = env.step(actions)
            
            
            ######################################
            # --- ДИАГНОСТИКА РАЗ В N ШАГОВ ---
            if t % PRINT_EVERY == 0:
                i = 0  # показываем первую среду
                print(f"\n[t={t}] CMD vx,vy,wz (env0): {cmd[i].tolist()}")
                print(    f"       Vxy|wz (env0):    {v_xy[i].tolist()} | {w_z[i].item():.3f}")
                print(    f"       |cmd|>th frac:    {frac_move_cmd:.2f}  |  |cmd|==0 frac: {frac_zero_cmd:.2f}")
                print(    f"       standing frac:     {frac_actually_standing:.2f}")
                print(    f"       actions mean/std:  {act_mean:+.3f} / {act_std:.3f}")
                print(    f"       batch rew mean:    {rew.mean().item():+.4f}")
                # если есть reward_buf по термам — можно глянуть
                if hasattr(env, "reward_buf"):
                    print(  f"       reward sum (env0): {env.reward_buf[i].item():+.4f}")

                # простая «ручная» проверка idle-ситуации для env0
                want_move0 = cmd[i, :2].norm() > CMD_MIN_TH
                standing0  = v_xy[i].norm()    < LIN_IDLE_TH
                if want_move0 and standing0:
                    print("   [IDLE] Команда на движение есть, но база почти стоит (env0).")

                # предупреждения
                if frac_zero_cmd > 0.9:
                    print("   [WARN] >90% батча имеют нулевую команду. Проверь генератор команд/конфиг.")
                if frac_move_cmd > 0.9 and frac_actually_standing > 0.9:
                    print("   [WARN] Команда есть почти у всех, но почти все стоят. Слишком слабый трекинг/слишком сильные штрафы/действия ≈ 0?")
                if act_std < 1e-3:
                    print("   [WARN] Действия деградировали к константе (std≈0). Проверь политику/tanh-обрезание/scale.")
            #####################################

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
    # close sim app
    simulation_app.close()
