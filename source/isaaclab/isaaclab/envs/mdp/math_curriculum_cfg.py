import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
import isaaclab.envs.mdp.param_scheduler as ps


@configclass
class MathCurriculumCfg:
    # === A: сначала баланс и трекинг  ===
    set_upright_0 = CurrTerm(func=mdp.modify_reward_weight,
        params={"term_name":"upright","weight":0.8,"num_steps":0})


    # REWARD WEIGHTS SMOOTH CHANGERS
    target_proximity_exp_product_lin_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.target_proximity_exp_product.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": 0.0, "end": 100.0, "num_steps": 100_000, "start_after": 30_000},
        },
    )
    unmasked_init_proximity_exp_product_lin_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.unmasked_init_proximity_exp_product.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": 0.0, "end": 5.0, "num_steps": 100_000, "start_after": 30_000},
        },
    )    
    
    tr_lin_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.track_lin_vel_xy_exp.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": 0.0, "end": 2.0, "num_steps": 1_000, "start_after": 1_000},
        },
    )
    tr_yaw_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.track_ang_vel_z_exp.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": 0.0, "end": 0.5, "num_steps": 1_000, "start_after": 1_000},
        },
    )
    air_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.feet_air_time.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": 0.0, "end": 0.5, "num_steps": 10_000, "start_after": 20_000},
        },
    )    
#    trot_warmup = CurrTerm(
#        func=mdp.modify_term_cfg,
#        params={
#            "address": "rewards.trot_rew.weight", 
#            "modify_fn": ps.lerp_scalar,          
#            "modify_params": {"start": 0.0, "end": 1.0, "num_steps": 100_000},
#        },
#    )    
    com_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.com_over_support.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": 0.0, "end": 3.0, "num_steps": 1_000, "start_after": 10_000},
        },
    )  


    # PENALTY WEIGHTS SMOOTH CHANGERS
    imp_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.feet_impact_vel.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": 0.0, "end": -0.1, "num_steps": 1_000, "start_after": 50_000},
        },
    )       
    slip_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.feet_slide.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": 0.0, "end": -0.1, "num_steps": 1_000, "start_after": 20_000},
        },
    )    
    act_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.action_rate_l2.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": 0.0, "end": -0.015, "num_steps": 1_000, "start_after": 50_000},
        },
    )     
    tq_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.dof_torques_l2.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": 0.0, "end": -1e-4, "num_steps": 1_000, "start_after": 50_000},
        },
    )     
    lin_vel_z_l2_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.lin_vel_z_l2.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": 0.0, "end": -2.0, "num_steps": 1_000, "start_after": 50_000},
        },
    )  
    ang_vel_xy_l2_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.ang_vel_xy_l2.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": 0.0, "end": -0.05, "num_steps": 1_000, "start_after": 50_000},
        },
    ) 
    joint_vel_l2_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.joint_vel_l2.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": 0.0, "end": -1.0e-4 , "num_steps": 1_000, "start_after": 50_000},
        },
    )
    dof_acc_l2_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.dof_acc_l2.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": 0.0, "end": -1e-07 , "num_steps": 1_000, "start_after": 50_000},
        },
    )
    dof_pos_limits_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.dof_pos_limits.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": 0.0, "end": -0.05 , "num_steps": 1_000, "start_after": 50_000},
        },
    )
    undesired_contacts_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.undesired_contacts.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": 0.0, "end": -5.0 , "num_steps": 1_000, "start_after": 10_000},
        },
    )
    swing_clearance_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.swing_clearance.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": 0.0, "end": 1.0 , "num_steps": 1_000, "start_after": 20_000},
        },
    )    
    feet_sep_align_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.feet_sep_align.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": 0.0, "end": -0.5 , "num_steps": 1_000, "start_after": 20_000},
        },
    )    
    no_cmd_motion_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.no_cmd_motion.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": 0.0, "end": -0.3 , "num_steps": 1_000, "start_after": 20_000},
        },
    )  
    lateral_slip_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.lateral_slip.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": 0.0, "end": -0.1 , "num_steps": 1_000, "start_after": 20_000},
        },
    )      
    limit_saturation_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.limit_saturation.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": 0.0, "end": -1.0 , "num_steps": 1_000, "start_after": 50_000},
        },
    )         
    single_foot_stationary_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.single_foot_stationary.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": 0.0, "end": -0.05 , "num_steps": 1_000, "start_after": 20_000},
        },
    ) 
    alternating_steps_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.alternating_steps_reward.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": 0.0, "end": 0.1, "num_steps": 1_000, "start_after": 20_000},
        },
    ) 
    same_lead_penalty_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.same_lead_penalty.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": 0.0, "end": -0.5, "num_steps": 10_000, "start_after": 20_000},
        },
    )     
    coalignment_chain_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.coalignment_chain.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": 0.0, "end": 0.5, "num_steps": 1_000, "start_after": 20_000},
        },
    )       
    body_lin_acc_l2_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.body_lin_acc_l2.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": 0.0, "end": -2.5e-7, "num_steps": 1_000, "start_after": 50_000},
        },
    )   
    prolonged_single_support_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.prolonged_single_support.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": 0.0, "end": -5.0e-1, "num_steps": 1_000, "start_after": 30_000},
        },
    )        
       
    
    # COMMANDS SMOOTH CHANGERS 
    cmd_lin_x_range = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.lin_vel_x",
            "modify_fn": ps.lerp_tuple,
            "modify_params": {"start": (0.0, 0.1), "end": (0.0, 1.0), "num_steps": 1_000, "start_after": 1_000},
            
        },
    )
    
    cmd_yaw_range_sched = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.ang_vel_z",
            "modify_fn": ps.lerp_tuple,
            "modify_params": {
                "start": (-0.1, 0.1),
                "end":   (-1.0, 1.0),
                "num_steps": 1_000,
                "start_after": 1_000
            },
        },
    )



    cmd_mask_prob_sched_1 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.dof_mask.p_one",
            "modify_fn": ps.lerp_scalar,
            "modify_params": {
                "start": 0.1,
                "end":   0.1,
                "num_steps": 30_000,
                "start_after": 1_000
            },
        },
    )
    cmd_mask_prob_sched_2 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.dof_mask.p_one",
            "modify_fn": ps.lerp_scalar,
            "modify_params": {
                "start": 0.1,
                "end":   0.5,
                "num_steps": 100_000,
                "start_after": 30_000
            },
        },
    )











    '''
    # контакты выключены на старте (оставляем как есть: 0)

    # === B: включаем шаговый шейпинг ===
    air_on_20k = CurrTerm(func=mdp.modify_reward_weight,
        params={"term_name":"feet_air_time","weight":0.05,"num_steps":20_000})
    air_more_30k = CurrTerm(func=mdp.modify_reward_weight,
        params={"term_name":"feet_air_time","weight":0.10,"num_steps":30_000})
    clr_on_40k = CurrTerm(func=mdp.modify_reward_weight,
        params={"term_name":"swing_clearance","weight":0.05,"num_steps":40_000})

    # === C: аккуратность и энергия (200k → 400k) ===
    ar_50k = CurrTerm(func=mdp.modify_reward_weight,
        params={"term_name":"action_rate_l2","weight":-0.02,"num_steps":50_000})
    tq_50k = CurrTerm(func=mdp.modify_reward_weight,
        params={"term_name":"dof_torques_l2","weight":-3e-5,"num_steps":50_000})
    jv_50k = CurrTerm(func=mdp.modify_reward_weight,
        params={"term_name":"joint_vel_l2","weight":-1e-4,"num_steps":50_000})
    ja_50k = CurrTerm(func=mdp.modify_reward_weight,
        params={"term_name":"dof_acc_l2","weight":-1e-6,"num_steps":50_000})

    slip_55k = CurrTerm(func=mdp.modify_reward_weight,
        params={"term_name":"feet_slide","weight":-0.05,"num_steps":55_000})
    imp_55k = CurrTerm(func=mdp.modify_reward_weight,
        params={"term_name":"feet_impact_vel","weight":-0.03,"num_steps":55_000})

    # мягче трекинг yaw поначалу → усилим позже
    yaw_55k = CurrTerm(func=mdp.modify_reward_weight,
        params={"term_name":"track_ang_vel_z_exp","weight":2.0,"num_steps":55_000})

#    # === D: робастность и расширение команд (400k → …) ===
#    # Расширим диапазоны команд (пример — по X); раскомментируйте, если используете утилиту modify_env_param:
#    vx1 = CurrTerm(func=mdp.modify_env_param,
#        params={"address":"commands.base_velocity.ranges.lin_vel_x","value":(0.0,1.6),"num_steps":400_000})
#    vy1 = CurrTerm(func=mdp.modify_env_param,
#        params={"address":"commands.base_velocity.ranges.lin_vel_y","value":(-0.4,0.4),"num_steps":400_000})
#    wz1 = CurrTerm(func=mdp.modify_env_param,
#        params={"address":"commands.base_velocity.ranges.ang_vel_z","value":(-1.2,1.2),"num_steps":400_000})

    # === TELEOP STAGE 1===   
    mask_prob_1 = CurrTerm(func=mdp.modify_env_param,
       params={"address":"commands.dof_mask.p_one","value":0.05,"num_steps":60_000}) 
    teleop_to_target_1 = CurrTerm(func=mdp.modify_reward_weight,
        params={"term_name":"target_proximity_exp_product","weight":3.0,"num_steps":60_000})
    teleop_to_init_1 = CurrTerm(func=mdp.modify_reward_weight,
        params={"term_name":"unmasked_init_proximity_exp_product","weight":0.5,"num_steps":60_000})    
        
    # === TELEOP STAGE 2===   
    mask_prob_2 = CurrTerm(func=mdp.modify_env_param,
       params={"address":"commands.dof_mask.p_one","value":0.1,"num_steps":65_000})  
    teleop_to_target_2 = CurrTerm(func=mdp.modify_reward_weight,
        params={"term_name":"target_proximity_exp_product","weight":6.0,"num_steps":65_000})
    teleop_to_init_2 = CurrTerm(func=mdp.modify_reward_weight,
        params={"term_name":"unmasked_init_proximity_exp_product","weight":1.0,"num_steps":65_000})         

    # === TELEOP STAGE 3===    
    mask_prob_3 = CurrTerm(func=mdp.modify_env_param,
       params={"address":"commands.dof_mask.p_one","value":0.2,"num_steps":70_000}) 
    teleop_to_target_3 = CurrTerm(func=mdp.modify_reward_weight,
        params={"term_name":"target_proximity_exp_product","weight":10.0,"num_steps":70_000})
    teleop_to_init_3 = CurrTerm(func=mdp.modify_reward_weight,
        params={"term_name":"unmasked_init_proximity_exp_product","weight":1.0,"num_steps":70_000})     
        
    # === TELEOP STAGE 4===  
    mask_prob_4 = CurrTerm(func=mdp.modify_env_param,
       params={"address":"commands.dof_mask.p_one","value":0.3,"num_steps":75_000})   
    teleop_to_target_4 = CurrTerm(func=mdp.modify_reward_weight,
        params={"term_name":"target_proximity_exp_product","weight":20.0,"num_steps":75_000})
    teleop_to_init_4 = CurrTerm(func=mdp.modify_reward_weight,
        params={"term_name":"unmasked_init_proximity_exp_product","weight":1.0,"num_steps":75_000})               

    imp_420k = CurrTerm(func=mdp.modify_reward_weight,
        params={"term_name":"feet_impact_vel","weight":-0.06,"num_steps":420_000})
    slip_420k = CurrTerm(func=mdp.modify_reward_weight,
        params={"term_name":"feet_slide","weight":-0.08,"num_steps":420_000})
    '''    
