# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause

"""
Play RSL-RL checkpoint with diagnostics.
Важное: запускаем AppLauncher ПЕРЕД импортами isaaclab/isaacsim.
"""

import argparse
import os
import time

def build_parser():
    import cli_args  # лёгкий локальный модуль
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Play an RL agent (RSL-RL) with diagnostics.")
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
    parser.add_argument("--video_length", type=int, default=200, help="Recorded video length in steps.")
    parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable Fabric.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of envs to simulate.")
    parser.add_argument("--task", type=str, required=True, help="Task, e.g. Isaac-Velocity-UnitreeGo1-PLAY-v0")
    parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use published pretrained checkpoint.")
    parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time if possible.")
    # RSL-RL args
    cli_args.add_rsl_rl_args(parser)
    # App args
    AppLauncher.add_app_launcher_args(parser)
    return parser, AppLauncher

def main():
    # --------- parse args & launch app first ---------
    parser, AppLauncher = build_parser()
    args = parser.parse_args()
    if args.video:
        args.enable_cameras = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app  # после этого можно импортировать isaaclab/*

    # --------- heavy imports (OK after app launch) ---------
    import torch
    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401 (регистрация задач)
    import cli_args

    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab.utils.dict import print_dict
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
    from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

    from isaaclab_rl.rsl_rl import (
        RslRlOnPolicyRunnerCfg,
        RslRlVecEnvWrapper,
        MathRslRlVecEnvWrapper,
        MathTeleopRslRlVecEnvWrapper,
        export_policy_as_jit,
        export_policy_as_onnx,
    )

    from isaaclab.envs import (
        ManagerBasedRLEnvCfg,
        MathManagerBasedRLEnvCfg,
        MathTeleopManagerBasedRLEnvCfg,
    )

    # --------- cfg & checkpoint ---------
    task_name = args.task.split(":")[-1]
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs, use_fabric=not args.disable_fabric)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(task_name, args)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Pre-trained checkpoint is not available for this task.")
            simulation_app.close()
            return
    elif args.checkpoint:
        resume_path = retrieve_file_path(args.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    if not resume_path or not os.path.exists(resume_path):
        print(f"[ERROR] Checkpoint not found: {resume_path}")
        simulation_app.close()
        return

    log_dir = os.path.dirname(resume_path)

    # --------- env ---------
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during evaluation:")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    elif isinstance(env_cfg, MathManagerBasedRLEnvCfg):
        env = MathRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    elif isinstance(env_cfg, MathTeleopManagerBasedRLEnvCfg):
        env = MathTeleopRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    else:
        print("[ERROR] Unsupported manager type for wrapper.")
        simulation_app.close()
        return

    # --------- policy ---------
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        policy_nn = ppo_runner.alg.actor_critic

    export_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_dir, filename="policy.onnx")

    # --------- base env access ---------
    base = env.unwrapped

    # fast preflight
    if hasattr(base, "command_manager") and hasattr(base.command_manager, "_terms"):
        print("[CMD TERMS]:", list(base.command_manager._terms.keys()))
    else:
        print("[WARN] command_manager недоступен на базовой среде")

    try:
        c = base.command_manager.get_command("base_velocity")
        print("[CMD SHAPE/DEVICE]:", tuple(c.shape), c.device)
    except Exception as e:
        print("[ERROR] get_command('base_velocity'):", e)

    if hasattr(base.scene, "sensors") and "contact_forces" in base.scene.sensors:
        cs = base.scene.sensors["contact_forces"]
        print("[CONTACT SENSOR] hist:", hasattr(cs.data, "net_forces_w_history"))

    try:
        term = base.command_manager.get_term("base_velocity")
        print("[CHECK] base_velocity term type:", type(term).__name__)
    except Exception as e:
        print("[ERROR] нет терма 'base_velocity':", e)

    dt = getattr(base, "step_dt", 0.02)

    # --------- reset ---------
    obs, _ = env.reset()
    t = 0

    PRINT_EVERY = 50
    LIN_IDLE_TH = 0.05
    CMD_MIN_TH = 0.10
    prev_cmd = None

    # --------- loop ---------
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            act_mean = actions.mean().item()
            act_std = actions.std().item()

            # diagnostics before step
            try:
                cmd = base.command_manager.get_command("base_velocity")
            except Exception as e:
                print("[ERROR] get_command('base_velocity') свалился:", e)
                cmd = torch.zeros((base.num_envs, 3), device=base.device)

            v_xy = base.scene["robot"].data.root_lin_vel_b[:, :2]
            w_z = base.scene["robot"].data.root_ang_vel_b[:, 2]
            spd = v_xy.norm(dim=1)

            cmd_mag = cmd[:, :2].norm(dim=1)
            frac_zero_cmd = (cmd_mag < 1e-6).float().mean().item()
            frac_move_cmd = (cmd_mag > CMD_MIN_TH).float().mean().item()
            frac_stand = (spd < LIN_IDLE_TH).float().mean().item()

            if prev_cmd is None:
                prev_cmd = cmd.clone()
            else:
                _ = (cmd - prev_cmd).abs().max(dim=1)[0] > 1e-6
                prev_cmd = cmd.clone()

            obs, rew, _, _ = env.step(actions)

            if t % PRINT_EVERY == 0:
                i = 0
                print(f"\n[t={t}] CMD vx,vy,wz (env0): {cmd[i].tolist()}")
                print(f"         Vxy|wz (env0):    {v_xy[i].tolist()} | {w_z[i].item():.3f}")
                print(f"         |cmd|>th frac:    {frac_move_cmd:.2f}  |  |cmd|==0 frac: {frac_zero_cmd:.2f}")
                print(f"         standing frac:     {frac_stand:.2f}")
                print(f"         actions mean/std:  {act_mean:+.3f} / {act_std:.3f}")
                print(f"         batch rew mean:    {rew.mean().item():+.4f}")
                if hasattr(base, "reward_buf"):
                    try:
                        print(f"         reward sum (env0): {base.reward_buf[i].item():+.4f}")
                    except Exception:
                        pass

                want_move0 = cmd[i, :2].norm() > CMD_MIN_TH
                standing0 = v_xy[i].norm() < LIN_IDLE_TH
                if want_move0 and standing0:
                    print("   [IDLE] Команда есть, но база стоит (env0): проверь трекинг/штрафы/scale действий.")
                if frac_zero_cmd > 0.9:
                    print("   [WARN] >90% батча имеют нулевую команду. Проверь генератор команд/конфиг.")
                if frac_move_cmd > 0.9 and frac_stand > 0.9:
                    print("   [WARN] Команды есть почти у всех, но почти все стоят. Усиль track_* и/или ослабь штрафы.")
                if act_std < 1e-3:
                    print("   [WARN] Действия деградировали к константе (std≈0). Проверь tanh/clip/scale/нормализацию.")

        t += 1

        if args.video and t >= args.video_length:
            break

        if args.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()

