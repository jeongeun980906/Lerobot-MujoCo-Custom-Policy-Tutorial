import sys
import random
import numpy as np
import xml.etree.ElementTree as ET
from src.mujoco_helper import MuJoCoParserClass
from src.object_helpers import ObjectSpawner
from src.mujoco_helper.utils import prettify, sample_xyzs, rotation_matrix
from src.mujoco_helper.ik import solve_ik
from src.mujoco_helper.transforms import rpy2r, r2rpy
from src.env.success_checker import condition_checker
import os
import mujoco
import matplotlib.pyplot as plt
import copy
from src.env.build_mjcf import build_mjcf_based_on_config

class RILAB_OMY_ENV:
    def __init__(self, 
                 cfg,
                action_type='joint', 
                obs_type='joint_pos',
                vis_mode = 'teleop',
                name = 'tabletop_env',
                seed = None):
        self.cfg = cfg
        xml_path = cfg['xml_file']
        obj_configs = cfg['init_pose']
        self.vis_mode = vis_mode
        self.init_states = {}
        for obj_name, obj_config in obj_configs['receptacles'].items():
            if 'init_state' in obj_config:
                self.init_states[obj_name] = obj_config['init_state']
        self.recp_names = [obj for obj in obj_configs['receptacles'].keys()] 
        self.object_names = [obj for obj in obj_configs['objects'].keys()]
        obj_names_all = self.recp_names + self.object_names
        xml_path = build_mjcf_based_on_config(base_xml_path=xml_path, recp_names=[obj for obj in obj_configs['receptacles'].keys()], obj_names= [obj for obj in obj_configs['objects'].keys()])
        self.env = MuJoCoParserClass(name=name,rel_xml_path=xml_path)
        self.save_original_color()
        self.action_type = action_type
        self.obs_type = obs_type
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        
        self.tcp_link_name = 'tcp_link'
        self.obj_spawner = ObjectSpawner(self.env, cfg['init_pose'])
        self.success_checker = condition_checker( self.env, cfg['conditions'], obj_names_all)
        self.init_viewer()
        self.reset(seed)
        
    
    def init_viewer(self):
        self.env.reset()
        if self.vis_mode == 'teleop':
            self.env.init_viewer(
                distance          = 2.5,
                elevation         = -50, 
                transparent       = False,
                black_sky         = True,
                azimuth            = 60, 
                lookat            = [0.6,0.0,0.5], # None,
            )
        else:
            self.env.init_viewer(
                distance          = 2.0,
                elevation         = -40, 
                transparent       = False,
                black_sky         = True,
                azimuth            = 180, 
                lookat            = [0.4,0.0,0.6], # None,
            )
    def reset(self, seed = None, leader_pose = True):
        self.env.reset_wall_time()
        self.restore_original_color()
        if seed != None: np.random.seed(seed=0) 
        # print(self.cfg['init_pose']['robot'])
        # self.env.reset()
        mujoco.mj_resetData(self.env.model,self.env.data) # reset data
        idxs_step = self.env.get_idxs_step(joint_names=self.joint_names)
        # if 'joint' in self.action_type:
        if leader_pose:
            q_zero = np.array([-0.02914743, -1.5657328,   2.6794806 , -1.1105849 ,  1.5718971  ,-0.01073957])
        else:
            q_init = np.deg2rad([0,0,0,0,0,0])
            q_zero,ik_err_stack,ik_info = solve_ik(
                env = self.env,
                joint_names_for_ik = self.joint_names,
                body_name_trgt     = 'tcp_link',
                q_init       = q_init, # ik from zero pose
                p_trgt       = np.array([0.3,-0.1,1.0]),
                R_trgt       = rpy2r(np.deg2rad([90,-0.,90 ])),
            )
        self.q_zero = q_zero
        # print(q_zero)
        self.env.forward(q=q_zero,joint_names=self.joint_names,increase_tick=False)
        self.env.forward(q = np.zeros(4), joint_names=['rh_r1', 'rh_r2', 'rh_l1', 'rh_l2' ], increase_tick=False)
        # Set other joints that is not in joint_names to 0
        other_joint_names = set(self.env.joint_names) - set(self.joint_names) - set([None])
        q_temp = np.zeros(len(other_joint_names))
        self.env.forward(q=q_temp,joint_names=list(other_joint_names),increase_tick=False)
        self.env.set_p_body(body_name='base', p=self.cfg['init_pose']['robot']['position'])
        self.env.set_R_body(body_name='base', R=rpy2r(np.deg2rad(self.cfg['init_pose']['robot']['rotation'])))

        # Set object positions
        self.q = np.concatenate([q_zero, np.array([0.0]*2)])
        
        self.obj_spawner.spawn_recp()
        self.success_checker.reset(self.init_states)
        self.obj_spawner.spawn_objects()

        self.env.forward(increase_tick=False)
        self.p0, self.R0 = self.env.get_pR_body(body_name=self.tcp_link_name)
        self.obj_init_poses = self.get_object_pose()
        # self.obj_init_pose = np.concatenate([mug_init_pose, plate_init_pose],dtype=np.float32)

        print("DONE INITIALIZATION")

        self.gripper_state = False
        self.past_eef = self.get_ee_pose()

    def step(self, action, gripper_mode = 'binary'):
        if self.action_type == 'delta_eef_pose':
            q = self.env.get_qpos_joints(joint_names=self.joint_names)
            # self.p0, self.R0 = self.env.get_pR_body(body_name=self.tcp_link_name)
            self.p0 += action[:3]
            self.R0 = self.R0.dot(rpy2r(action[3:6]))
            q ,ik_err_stack,ik_info = solve_ik(
                env                = self.env,
                joint_names_for_ik = self.joint_names,
                body_name_trgt     = self.tcp_link_name,#'ur_tcp_link',
                q_init             = q,
                p_trgt             = self.p0,
                R_trgt             = self.R0,
                max_ik_tick        = 100,
                ik_stepsize        = 2.0,
                ik_eps             = 1e-3,
                ik_th              = np.radians(1),
                render             = False,
                verbose_warning    = False,
            )
            # print(ik_err_stack)
        elif self.action_type == 'eef_pose':
            self.p0 = action[:3]
            rpy = action[3:6]
            self.R0 = rpy2r(rpy)
            q = self.env.get_qpos_joints(joint_names=self.joint_names)
            q ,ik_err_stack,ik_info = solve_ik(
                env                = self.env,
                joint_names_for_ik = self.joint_names,
                body_name_trgt     = self.tcp_link_name,#'ur_tcp_link',
                q_init             = q,
                p_trgt             = self.p0,
                R_trgt             = self.R0,
                max_ik_tick        = 100,
                ik_stepsize        = 2.0,
                ik_eps             = 1e-3,
                ik_th              = np.radians(4),
                render             = False,
                verbose_warning    = False,
            )
        elif self.action_type == 'joint':
            q = action[:-1]
        elif self.action_type == 'delta_joint':
            q_current = self.env.get_qpos_joints(joint_names=self.joint_names)
            q = q_current + action[:-1]
        else:
            raise ValueError('action_type not recognized')
        if (self.action_type == 'eef_pose' or self.action_type == 'delta_eef_pose') and gripper_mode == 'binary':
            if action[-1] > 0.5:
                gripper_cmd = np.array([1.0]*2) #
            else:
                gripper_cmd = np.array([0.2]*2)
            q = np.concatenate([q, gripper_cmd])
        else:
            q = np.concatenate([q, np.array([-action[-1]]*2)])
        self.q = q
        observation = self.get_observation()
        return observation

    def step_env(self):
        self.env.step(self.q)

    def forward_env(self, action):
        joint_names = self.env.rev_joint_names[:10]
        self.env.forward(action, joint_names=joint_names)

    def grab_image(self, return_side = False):
        self.rgb_agent = self.env.get_fixed_cam_rgb(
            cam_name='agentview')
        self.rgb_ego = self.env.get_fixed_cam_rgb(
            cam_name='egocentric')
        # self.rgb_top = self.env.get_fixed_cam_rgbd_pcd(
        #     cam_name='topview')
        self.rgb_side = self.env.get_fixed_cam_rgb(
            cam_name='sideview')
        if return_side:
            return self.rgb_agent, self.rgb_ego, self.rgb_side
        return self.rgb_agent, self.rgb_ego
        

    def render(self, task='', guideline='', fail_signal=False):
        self.env.plot_time()
        base_pos, base_R = self.env.get_pR_body(body_name='base')
        p_current, R_current = self.env.get_pR_body(body_name=self.tcp_link_name) 
        R_current = R_current @ np.array([[1,0,0],[0,0,1],[0,1,0 ]])
        self.env.plot_sphere(p=p_current, r=0.02, rgba=[0.95,0.05,0.05,0.5])
        self.env.plot_capsule(p=p_current, R=R_current, r=0.01, h=0.2, rgba=[0.05,0.95,0.05,0.5])
        R_vertical = R_current @ np.array([[1,0,0],[0,0,1],[0,1,0 ]])
        self.env.viewer.plot_rgb_overlay(rgb=self.rgb_agent, loc='top right')
        self.env.viewer.plot_rgb_overlay(rgb=self.rgb_ego, loc='bottom right')
        self.env.viewer.plot_rgb_overlay(rgb=self.rgb_side, loc='top left')
        self.env.plot_T(np.array([-0.3,-0.15,0.1]) + base_pos,label=task, plot_axis=False, plot_sphere=False)
        self.env.plot_T(np.array([-0.2,-0.15,0.12]) + base_pos,label=f"Guide: {guideline}", plot_axis=False, plot_sphere=False)
        
        self.env.render()


    def get_observation(self):
        gripper_state = self.env.get_qpos_joints(joint_names=['rh_l1', 'rh_l2'])
        if gripper_state[0] < 0.5:
            gripper_state_val = 0.0
        else:
            gripper_state_val = 1.0
        if self.obs_type == 'joint_pos':
            qpos = self.env.get_qpos_joints(joint_names=self.joint_names)
            return np.concatenate([qpos, [gripper_state_val]], dtype=np.float32)
        elif self.obs_type == 'eef_pose':
            eef_pose = self.get_ee_pose()
            eef_pose = np.concatenate([eef_pose, [gripper_state_val]], dtype=np.float32)
            return eef_pose
        elif self.obs_type == 'full_state':
            qpos = self.env.get_qpos_joints(joint_names=self.joint_names)
            eef_pose = self.get_ee_pose()
            gripper_state = self.env.get_qpos_joints(joint_names=['rh_l1', 'rh_l2'])
            if gripper_state[0] < 0.5:
                gripper_state_val = 0.0
            else:
                gripper_state_val = 1.0
            eef_pose = np.concatenate([eef_pose, [gripper_state_val]], dtype=np.float32)
            return np.concatenate([qpos, eef_pose], dtype=np.float32)
        else:
            raise NotImplementedError(f"Observation type {self.obs_type} not implemented.")
    def infer_other_actions(self):
        p, R = self.env.get_pR_body(body_name=self.tcp_link_name) #'ur_tcp_link')
        dp = p - self.p0
        dR = R.dot(self.R0.T)
        drpy = r2rpy(dR)
        delta_eef = np.concatenate([dp, drpy],dtype=np.float32)
        qpos = self.env.get_qpos_joints(joint_names=self.joint_names)
        self.p0, self.R0 = p, R
        gripper_state = self.env.get_qpos_joints(joint_names=['rh_l1', 'rh_l2'])
        if gripper_state[0] < 0.5:
            gripper_state_val = 0.0
        else:
            gripper_state_val = 1.0
        return {
            'delta_eef': delta_eef.astype(np.float32),
            'eef_pose': np.concatenate([p, r2rpy(R)],dtype=np.float32),
            'joint_pos': qpos.astype(np.float32),
            'gripper_state': np.array([gripper_state_val], dtype=np.float32)
        }
    def get_full_joint_state(self):
        # Get the full joint state including gripper
        joint_names = self.env.rev_joint_names[:10]
        return self.env.get_qpos_joints(joint_names=joint_names).astype(np.float32)
    
    def get_ee_pose(self):
        p, R = self.env.get_pR_body(body_name=self.tcp_link_name) #'ur_tcp_link')
        rpy = r2rpy(R)
        return np.concatenate([p, rpy],dtype=np.float32)
    
    def get_object_pose(self, pad = None):
        obj_names = self.object_names
        obj_names = ['body_obj_'+name for name in obj_names]
        poses = []
        for obj_name in obj_names:
            p, R = self.env.get_pR_body(body_name=obj_name)
            poses.append(np.concatenate([p, r2rpy(R)],dtype=np.float32))
        # receptacles
        recp_names = self.recp_names
        recp_names = ['body_obj_'+name for name in recp_names]
        for recp_name in recp_names:
            p, R = self.env.get_pR_body(body_name=recp_name)
            poses.append(np.concatenate([p, r2rpy(R)],dtype=np.float32))
            obj_names.append(recp_name)
        q_states = self.success_checker.get_q_pose_of_recp()   
        q_names = list(q_states.keys())
        q_poses = list(q_states.values()) 
        if pad is not None:
            while len(poses) < pad:
                poses.append(np.zeros(6, dtype=np.float32))
                obj_names.append('pad')
            while len(q_poses) < pad:
                q_poses.append(0.0)
                q_names.append('pad')
        return {'poses': np.array(poses, dtype=np.float32), 'names': obj_names}, \
            {'poses': np.array(q_poses, dtype=np.float32), 'names': q_names}
    
    def set_object_pose(self, poeses, names, q_poses, q_names):
        for obj_name, pose in zip(names, poeses):
            if 'pad' in obj_name:
                continue
            elif 'cabinet' in obj_name:
                self.set_receptacle_pose(pose, obj_name)
            else:
                p, rpy = pose[:3], pose[3:]
                R = rpy2r(rpy)
                self.env.set_p_base_body(body_name=obj_name, p=p)
                self.env.set_R_base_body(body_name=obj_name, R=R)
        for joint_name, q_state in zip(q_names, q_poses):
            if 'pad' in joint_name:
                continue
            q_pos = self.env.get_qpos_joint(joint_name)
            joint_idx = self.env.get_idxs_fwd(joint_names=[joint_name])[0]
            self.env.data.qpos[joint_idx] = q_state

    def set_receptacle_pose(self, pose, name):
        p, rpy = pose[:3], pose[3:]
        R = rpy2r(rpy)
        self.env.set_p_body(body_name=name, p=p)
        self.env.set_R_body(body_name=name, R=R)

    def check_success(self, verbose=False):
        # print(self.env.model.geom_rgba.shape)
        return self.success_checker.check_success(verbose=verbose)
    
    def save_original_color(self):
        self.original_colors = {}
        for obj_name in self.object_names:
            geom_name = obj_name + '_geom'
            geom_idx = self.env.geom_names.index(geom_name)
            self.original_colors[obj_name] = copy.deepcopy(self.env.model.geom_rgba[geom_idx])
    

    def restore_original_color(self):
        for obj_name in self.object_names:
            geom_name = obj_name + '_geom'
            geom_idx = self.env.geom_names.index(geom_name)
            self.env.model.geom_rgba[geom_idx] = copy.deepcopy(self.original_colors[obj_name])
    

    def agument_object_random_color(self):
        # color set (set3)
        color_sets = plt.get_cmap('tab20').colors #('Set3').colors
        for obj_name in self.object_names:
            geom_name = obj_name + '_geom'
            geom_idx = self.env.geom_names.index(geom_name)
            new_color = color_sets[random.randint(0, len(color_sets)-1)]
            self.env.model.geom_rgba[geom_idx][:3] = new_color