import numpy as np
from src.env.env import RILAB_OMY_ENV
import glfw
from src.mujoco_helper.utils import prettify, sample_xyzs, rotation_matrix

from src.mujoco_helper.transforms import rpy2r, r2rpy

class controller:
    def __init__(self, pose_gain, rot_gain):
        self.pose_gain = pose_gain
        self.rot_gain = rot_gain
        self.gripper_state = False
        self.past_button = False

    def reset(self, env: RILAB_OMY_ENV):
        self.past_button = False
        self.gripper_state = False
        self.env = env

    def get_action(self):
        chars = self.env.env.get_key_pressed_list()
        if self.env.env.is_key_pressed_once(glfw.KEY_SPACE):
            self.gripper_state = not self.gripper_state
        dpos = np.zeros(3)
        drot = np.eye(3)
        if len(chars) > 0: 
            if glfw.KEY_W in chars:
                dpos[0] -= self.pose_gain
            if glfw.KEY_S in chars:
                dpos[0] += self.pose_gain
            if glfw.KEY_A in chars:
                dpos[1] -= self.pose_gain
            if glfw.KEY_D in chars:
                dpos[1] += self.pose_gain
            if glfw.KEY_R in chars:
                dpos[2] += self.pose_gain
            if glfw.KEY_F in chars:
                dpos[2] -= self.pose_gain
            if glfw.KEY_Q in chars:
                drot = rotation_matrix(angle=self.rot_gain, direction=[0.0, 0.0, 1.0])[:3, :3]
            if glfw.KEY_E in chars:
                drot = rotation_matrix(angle=-self.rot_gain, direction=[0.0, 0.0, 1.0])[:3, :3]
            if glfw.KEY_DOWN in chars:
                drot = rotation_matrix(angle=self.rot_gain, direction=[1.0, 0.0, 0.0])[:3, :3]
            if glfw.KEY_UP in chars:
                drot = rotation_matrix(angle=-self.rot_gain, direction=[1.0, 0.0, 0.0])[:3, :3]
            if glfw.KEY_LEFT in chars:
                drot = rotation_matrix(angle=self.rot_gain, direction=[0.0, -1.0, 0.0])[:3, :3]
            if glfw.KEY_RIGHT in chars: 
                drot = rotation_matrix(angle=-self.rot_gain, direction=[0.0, -1.0, 0.0])[:3, :3]
        drot = r2rpy(drot)
        action = np.concatenate([dpos, drot, np.array([self.gripper_state],dtype=np.float32)],dtype=np.float32)
        return action
    
    def close(self):
        return